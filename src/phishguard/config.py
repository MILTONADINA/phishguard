"""Shared constants and data types (contracts). Fully defined — no implementation needed.

See specs/00-overview.md (conventions), specs/03-text-model.md, specs/04-url-model.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field

SEED = 42
DEFAULT_THRESHOLD = 0.5
MAX_EMAIL_CHARS = 50_000
MAX_URLS = 20

PHISHING = 1
LEGITIMATE = 0


@dataclass
class TextTrainConfig:
    """TEXT-03 training configuration."""

    model_name: str = "distilbert-base-uncased"
    max_length: int = 256
    batch_size: int = 16
    eval_batch_size: int = 64
    epochs: int = 2
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    patience: int = 1
    seed: int = SEED
    output_dir: str = "models/text"
    device: str | None = None
    max_train_samples: int | None = None


@dataclass
class UrlTrainConfig:
    """URL-04 training configuration."""

    max_len: int = 200
    embed_dim: int = 32
    num_filters: int = 64
    kernel_sizes: tuple[int, ...] = (3, 4, 5)
    dropout: float = 0.3
    batch_size: int = 256
    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 2
    seed: int = SEED
    output_dir: str = "models/url"
    device: str | None = None
    max_train_samples: int | None = None


@dataclass
class TrainResult:
    """Returned by train_text_model and train_url_model (TEXT-03, TEXT-05, URL-04).

    history: one dict per completed epoch: {"epoch": int (1-based), "train_loss": float, "val_f1": float}
    best_epoch: 1-based epoch with the highest val_f1
    best_val_f1: max val_f1 in history
    output_dir: where the best checkpoint was saved
    """

    history: list[dict] = field(default_factory=list)
    best_epoch: int = 0
    best_val_f1: float = 0.0
    output_dir: str = ""
