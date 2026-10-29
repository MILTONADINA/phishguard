"""Stage 1: fine-tuned DistilBERT text classifier. Spec: specs/03-text-model.md."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from phishguard.config import TextTrainConfig, TrainResult
from phishguard.utils import select_device

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


def train_text_model(
    train_texts: Sequence[str],
    train_labels: Sequence[int],
    val_texts: Sequence[str],
    val_labels: Sequence[int],
    config: TextTrainConfig,
) -> TrainResult:
    """TEXT-03/04/05: fine-tuning loop (AdamW + linear warmup/decay, per-epoch val F1,
    save best to config.output_dir, early stopping on patience)."""
    raise NotImplementedError("TEXT-03: see specs/03-text-model.md")
