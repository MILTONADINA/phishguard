"""Stage 2: character-level CNN URL classifier. Spec: specs/04-url-model.md."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn

from phishguard.config import TrainResult, UrlTrainConfig


class CharVocab:
    """URL-01: PAD=0, UNK=1, printable ASCII chr(32)..chr(126) -> ord(c) - 30 (ids 2..96). vocab_size = 97."""

    PAD = 0
    UNK = 1

    def __init__(self, max_len: int = 200) -> None:
        raise NotImplementedError("URL-01: see specs/04-url-model.md")

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError("URL-01: see specs/04-url-model.md")

    def encode(self, url: str) -> list[int]:
        """Strip, map chars (others -> UNK), keep the first max_len, right-pad with PAD to max_len."""
        raise NotImplementedError("URL-01: see specs/04-url-model.md")

    def encode_batch(self, urls: Sequence[str]) -> torch.Tensor:
        """LongTensor of shape (len(urls), max_len)."""
        raise NotImplementedError("URL-01: see specs/04-url-model.md")


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
        raise NotImplementedError("URL-02: see specs/04-url-model.md")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("URL-02: see specs/04-url-model.md")


class URLClassifier:
    """URL-03/05: predict_proba over URL strings, plus save/load."""

    def __init__(self, model: CharCNN, vocab: CharVocab, device: str | torch.device | None = None) -> None:
        raise NotImplementedError("URL-03: see specs/04-url-model.md")

    def predict_proba(self, urls: Sequence[str]) -> np.ndarray:
        """sigmoid(logits), shape (len(urls),). Empty -> shape (0,). Eval mode, no grad."""
        raise NotImplementedError("URL-03: see specs/04-url-model.md")

    def save(self, path: str | Path) -> None:
        """Write path/url_model.pt (state_dict) and path/url_config.json (max_len + CharCNN args)."""
        raise NotImplementedError("URL-05: see specs/04-url-model.md")

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device | None = None) -> "URLClassifier":
        raise NotImplementedError("URL-05: see specs/04-url-model.md")


def train_url_model(
    train_urls: Sequence[str],
    train_labels: Sequence[int],
    val_urls: Sequence[str],
    val_labels: Sequence[int],
    config: UrlTrainConfig,
) -> TrainResult:
    """URL-04: same loop contract as TEXT-03 (BCEWithLogitsLoss, save best, early stopping)."""
    raise NotImplementedError("URL-04: see specs/04-url-model.md")
