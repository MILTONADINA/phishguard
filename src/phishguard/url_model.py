"""Stage 2: character-level CNN URL classifier. Spec: specs/04-url-model.md."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn

from phishguard.config import TrainResult, UrlTrainConfig
from phishguard.evaluate import compute_metrics
from phishguard.utils import select_device, set_seed, stratified_subset

LOG_EVERY_STEPS = 200  # progress line frequency during an epoch
PREDICT_BATCH_SIZE = 1024  # URLs are short, so prediction can use big batches


class CharVocab:
    """URL-01: PAD=0, UNK=1, printable ASCII chr(32)..chr(126) -> ord(c) - 30 (ids 2..96). vocab_size = 97."""

    PAD = 0
    UNK = 1
    FIRST, LAST = 32, 126  # printable ASCII: space ... tilde

    def __init__(self, max_len: int = 200) -> None:
        self.max_len = max_len

    @property
    def vocab_size(self) -> int:
        return self.LAST - 30 + 1  # ids 0 (PAD) and 1 (UNK), then 2..96 for the 95 printable characters

    def _char_id(self, c: str) -> int:
        code = ord(c)
        return code - 30 if self.FIRST <= code <= self.LAST else self.UNK

    def encode(self, url: str) -> list[int]:
        """Strip, map chars (others -> UNK), keep the first max_len, right-pad with PAD to max_len."""
        ids = [self._char_id(c) for c in url.strip()[: self.max_len]]
        return ids + [self.PAD] * (self.max_len - len(ids))

    def encode_batch(self, urls: Sequence[str]) -> torch.Tensor:
        """LongTensor of shape (len(urls), max_len)."""
        return torch.tensor([self.encode(u) for u in urls], dtype=torch.long).reshape(-1, self.max_len)


class CharCNN(nn.Module):
    """URL-02: Embedding(padding_idx=0) -> Conv1d per kernel -> ReLU -> global max-pool -> concat
    -> dropout -> Linear(.., 1). forward returns logits of shape (B,)."""

    def __init__(
        self,
        vocab_size: int = 97,
        embed_dim: int = 32,
        num_filters: int = 64,
        kernel_sizes: tuple[int, ...] = (3, 4, 5),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        # Kept so save() can write them and load() can rebuild the same network.
        self.config = {
            "vocab_size": vocab_size,
            "embed_dim": embed_dim,
            "num_filters": num_filters,
            "kernel_sizes": list(kernel_sizes),
            "dropout": dropout,
        }
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=CharVocab.PAD)
        # One convolution per window size: 3, 4 and 5 characters ("pa1", ".xyz", "login", ...).
        self.convs = nn.ModuleList(nn.Conv1d(embed_dim, num_filters, k) for k in kernel_sizes)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(num_filters * len(kernel_sizes), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x).transpose(1, 2)  # (B, L, E) -> (B, E, L): Conv1d slides over the last axis
        # Each filter's strongest match anywhere in the URL (global max-pool over positions).
        pooled = [torch.relu(conv(emb)).max(dim=2).values for conv in self.convs]
        features = self.dropout(torch.cat(pooled, dim=1))
        return self.out(features).squeeze(-1)  # logits, shape (B,)


class URLClassifier:
    """URL-03/05: predict_proba over URL strings, plus save/load."""

    def __init__(self, model: CharCNN, vocab: CharVocab, device: str | torch.device | None = None) -> None:
        self.device = select_device(device.type if isinstance(device, torch.device) else device)
        self.vocab = vocab
        self.model = model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_proba(self, urls: Sequence[str]) -> np.ndarray:
        """sigmoid(logits), shape (len(urls),). Empty -> shape (0,). Eval mode, no grad."""
        urls = list(urls)
        if not urls:
            return np.empty((0,), dtype=float)
        self.model.eval()
        probs = []
        for start in range(0, len(urls), PREDICT_BATCH_SIZE):
            batch = self.vocab.encode_batch(urls[start : start + PREDICT_BATCH_SIZE]).to(self.device)
            probs.append(torch.sigmoid(self.model(batch)).float().cpu())
        return torch.cat(probs).numpy().astype(float)

    def save(self, path: str | Path) -> None:
        """Write path/url_model.pt (state_dict) and path/url_config.json (max_len + CharCNN args)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        weights = {name: tensor.cpu() for name, tensor in self.model.state_dict().items()}  # loadable anywhere
        torch.save(weights, path / "url_model.pt")
        config = {"max_len": self.vocab.max_len, **self.model.config}
        (path / "url_config.json").write_text(json.dumps(config, indent=2))

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device | None = None) -> "URLClassifier":
        path = Path(path)
        config = json.loads((path / "url_config.json").read_text())
        vocab = CharVocab(max_len=config.pop("max_len"))
        model = CharCNN(**{**config, "kernel_sizes": tuple(config["kernel_sizes"])})
        # weights_only=True: load plain tensors only, never run code stored in the file.
        model.load_state_dict(torch.load(path / "url_model.pt", map_location="cpu", weights_only=True))
        return cls(model, vocab, device=device)


def train_url_model(
    train_urls: Sequence[str],
    train_labels: Sequence[int],
    val_urls: Sequence[str],
    val_labels: Sequence[int],
    config: UrlTrainConfig,
) -> TrainResult:
    """URL-04: same loop contract as TEXT-03 (BCEWithLogitsLoss, save best, early stopping)."""
    set_seed(config.seed)
    train_urls, train_labels = stratified_subset(
        list(train_urls), [int(y) for y in train_labels], config.max_train_samples, config.seed
    )
    val_urls, val_labels = list(val_urls), [int(y) for y in val_labels]

    model = CharCNN(
        vocab_size=CharVocab().vocab_size,
        embed_dim=config.embed_dim,
        num_filters=config.num_filters,
        kernel_sizes=tuple(config.kernel_sizes),
        dropout=config.dropout,
    )
    clf = URLClassifier(model, CharVocab(max_len=config.max_len), device=config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()  # sigmoid + binary cross-entropy in one numerically stable step
    shuffler = np.random.default_rng(config.seed)
    steps_per_epoch = math.ceil(len(train_urls) / config.batch_size)
    print(f"Training on {len(train_urls):,} URLs, {steps_per_epoch:,} steps/epoch, device {clf.device}", flush=True)

    history: list[dict] = []
    best_f1, best_epoch, epochs_without_improvement = -1.0, 0, 0
    for epoch in range(1, config.epochs + 1):
        model.train()  # dropout on while learning
        order = shuffler.permutation(len(train_urls))  # new order every epoch
        epoch_start = time.perf_counter()
        loss_sum = torch.zeros((), device=clf.device)  # summed on the device: no GPU sync every step
        for step, start in enumerate(range(0, len(order), config.batch_size), start=1):
            idx = order[start : start + config.batch_size]
            batch = clf.vocab.encode_batch([train_urls[i] for i in idx]).to(clf.device)
            targets = torch.tensor([train_labels[i] for i in idx], dtype=torch.float32, device=clf.device)

            loss = loss_fn(model(batch), targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.detach() * len(idx)

            if step % LOG_EVERY_STEPS == 0:
                elapsed = time.perf_counter() - epoch_start
                print(
                    f"  epoch {epoch} step {step:,}/{steps_per_epoch:,}  "
                    f"avg loss {loss_sum.item() / (start + len(idx)):.4f}  {elapsed:.0f} s elapsed",
                    flush=True,
                )

        train_loss = loss_sum.item() / len(train_urls)
        val_f1 = compute_metrics(val_labels, clf.predict_proba(val_urls))["f1"]
        seconds = time.perf_counter() - epoch_start
        history.append({"epoch": epoch, "train_loss": train_loss, "val_f1": val_f1})
        print(f"epoch {epoch}: train_loss {train_loss:.4f}  val_f1 {val_f1:.4f}  ({seconds:.0f} s)", flush=True)

        if val_f1 > best_f1:  # keep only the best checkpoint
            best_f1, best_epoch, epochs_without_improvement = val_f1, epoch, 0
            clf.save(config.output_dir)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                print(f"Stopping early: no val F1 improvement for {config.patience} epoch(s)", flush=True)
                break

    return TrainResult(history=history, best_epoch=best_epoch, best_val_f1=best_f1, output_dir=config.output_dir)
