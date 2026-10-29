"""Stage 1: fine-tuned DistilBERT text classifier. Spec: specs/03-text-model.md."""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from phishguard.config import TextTrainConfig, TrainResult
from phishguard.evaluate import compute_metrics
from phishguard.utils import select_device, set_seed, stratified_subset

PHISHING_INDEX = 1  # the classification head's second output is P(phishing)


class PhishTextClassifier:
    """TEXT-02: wraps AutoTokenizer + AutoModelForSequenceClassification (num_labels=2, index 1 = phishing).

    Attributes: model, tokenizer, device (torch.device), max_length, batch_size.
    """

    def __init__(
        self,
        model_name_or_path: str | Path,
        max_length: int = 256,
        device: str | torch.device | None = None,
        batch_size: int = 32,
    ) -> None:
        self.device = select_device(device.type if isinstance(device, torch.device) else device)
        self.max_length = max_length
        self.batch_size = batch_size
        # A Hugging Face name (downloads once, then cached) or a folder written by save().
        # For the pretrained "distilbert-base-uncased", the 2-class head on top starts random.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels=2)
        self.model.to(self.device)
        self.model.eval()  # inference mode: dropout off, so predictions are deterministic

    def tokenize(self, texts: Sequence[str]) -> dict[str, torch.Tensor]:
        """Token ids + attention mask for one batch, cut to max_length and padded to the longest text."""
        batch = self.tokenizer(
            [str(t) for t in texts],
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )
        return {name: tensor.to(self.device) for name, tensor in batch.items()}

    @torch.no_grad()
    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """P(phishing) per text, shape (len(texts),). Eval mode, no grad, batched, truncated."""
        texts = list(texts)
        if not texts:
            return np.empty((0,), dtype=float)
        self.model.eval()
        probs = []
        for start in range(0, len(texts), self.batch_size):
            logits = self.model(**self.tokenize(texts[start : start + self.batch_size])).logits
            probs.append(torch.softmax(logits, dim=-1)[:, PHISHING_INDEX].float().cpu())
        return torch.cat(probs).numpy().astype(float)

    def save(self, path: str | Path) -> None:
        """save_pretrained for both the model and the tokenizer."""
        Path(path).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    @classmethod
    def load(cls, path: str | Path, **kwargs) -> "PhishTextClassifier":
        return cls(path, **kwargs)


LOG_EVERY_STEPS = 200  # progress line frequency during an epoch


def linear_warmup_then_decay(warmup_steps: int, total_steps: int):
    """Learning-rate multiplier per step: rises 0 -> 1 over the warmup, then falls linearly to 0."""

    def factor(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    return factor


def train_text_model(
    train_texts: Sequence[str],
    train_labels: Sequence[int],
    val_texts: Sequence[str],
    val_labels: Sequence[int],
    config: TextTrainConfig,
) -> TrainResult:
    """TEXT-03/04/05: fine-tuning loop (AdamW + linear warmup/decay, per-epoch val F1,
    save best to config.output_dir, early stopping on patience)."""
    set_seed(config.seed)
    train_texts, train_labels = stratified_subset(
        list(train_texts), [int(y) for y in train_labels], config.max_train_samples, config.seed
    )
    val_texts, val_labels = list(val_texts), [int(y) for y in val_labels]

    clf = PhishTextClassifier(
        config.model_name, max_length=config.max_length, device=config.device, batch_size=config.eval_batch_size
    )
    model = clf.model
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    steps_per_epoch = math.ceil(len(train_texts) / config.batch_size)
    total_steps = steps_per_epoch * config.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, linear_warmup_then_decay(int(config.warmup_ratio * total_steps), total_steps)
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    shuffler = np.random.default_rng(config.seed)
    print(
        f"Training on {len(train_texts):,} emails, {steps_per_epoch:,} steps/epoch, device {clf.device}",
        flush=True,
    )

    history: list[dict] = []
    best_f1, best_epoch, epochs_without_improvement = -1.0, 0, 0
    for epoch in range(1, config.epochs + 1):
        model.train()  # dropout on while learning
        order = shuffler.permutation(len(train_texts))  # new order every epoch
        epoch_start = time.perf_counter()
        loss_sum = torch.zeros((), device=clf.device)  # summed on the device: no GPU sync every step
        for step, start in enumerate(range(0, len(order), config.batch_size), start=1):
            idx = order[start : start + config.batch_size]
            batch = clf.tokenize([train_texts[i] for i in idx])
            targets = torch.tensor([train_labels[i] for i in idx], device=clf.device)

            loss = loss_fn(model(**batch).logits, targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # guard against loss spikes
            optimizer.step()
            scheduler.step()
            loss_sum += loss.detach() * len(idx)

            if step % LOG_EVERY_STEPS == 0:
                elapsed = time.perf_counter() - epoch_start
                eta = elapsed / step * (steps_per_epoch - step)
                print(
                    f"  epoch {epoch} step {step:,}/{steps_per_epoch:,}  "
                    f"avg loss {loss_sum.item() / (start + len(idx)):.4f}  "
                    f"{elapsed / 60:.1f} min elapsed, ~{eta / 60:.1f} min left",
                    flush=True,
                )

        train_loss = loss_sum.item() / len(train_texts)
        val_f1 = compute_metrics(val_labels, clf.predict_proba(val_texts))["f1"]
        minutes = (time.perf_counter() - epoch_start) / 60
        history.append({"epoch": epoch, "train_loss": train_loss, "val_f1": val_f1})
        print(f"epoch {epoch}: train_loss {train_loss:.4f}  val_f1 {val_f1:.4f}  ({minutes:.1f} min)", flush=True)

        if val_f1 > best_f1:  # keep only the best checkpoint
            best_f1, best_epoch, epochs_without_improvement = val_f1, epoch, 0
            clf.save(config.output_dir)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                print(f"Stopping early: no val F1 improvement for {config.patience} epoch(s)", flush=True)
                break

    return TrainResult(history=history, best_epoch=best_epoch, best_val_f1=best_f1, output_dir=config.output_dir)
