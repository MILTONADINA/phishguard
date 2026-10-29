"""Stage 1: fine-tuned DistilBERT text classifier. Spec: specs/03-text-model.md."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from phishguard.config import TextTrainConfig, TrainResult


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
        raise NotImplementedError("TEXT-02: see specs/03-text-model.md")

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """P(phishing) per text, shape (len(texts),). Eval mode, no grad, batched, truncated."""
        raise NotImplementedError("TEXT-02: see specs/03-text-model.md")

    def save(self, path: str | Path) -> None:
        """save_pretrained for both the model and the tokenizer."""
        raise NotImplementedError("TEXT-02: see specs/03-text-model.md")

    @classmethod
    def load(cls, path: str | Path, **kwargs) -> "PhishTextClassifier":
        raise NotImplementedError("TEXT-02: see specs/03-text-model.md")


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
