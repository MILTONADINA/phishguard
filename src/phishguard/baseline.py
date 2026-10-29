"""TF-IDF + logistic regression baseline. Spec: specs/02-baseline.md."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.pipeline import Pipeline


def build_baseline(
    max_features: int = 50_000,
    ngram_range: tuple[int, int] = (1, 2),
    C: float = 1.0,
    seed: int = 42,
) -> Pipeline:
    """BASE-01: unfitted Pipeline([("tfidf", TfidfVectorizer(...)), ("clf", LogisticRegression(...))])."""
    raise NotImplementedError("BASE-01: see specs/02-baseline.md")


def train_baseline(texts: Sequence[str], labels: Sequence[int], **kwargs) -> Pipeline:
    """BASE-01: build_baseline(**kwargs), then fit it."""
    raise NotImplementedError("BASE-01: see specs/02-baseline.md")


def baseline_predict_proba(model: Pipeline, texts: Sequence[str]) -> np.ndarray:
    """BASE-02: P(phishing) per text, shape (len(texts),). Empty input -> shape (0,)."""
    raise NotImplementedError("BASE-02: see specs/02-baseline.md")


def save_baseline(model: Pipeline, path: str | Path) -> None:
    """BASE-03: joblib dump, creating parent folders."""
    raise NotImplementedError("BASE-03: see specs/02-baseline.md")


def load_baseline(path: str | Path) -> Pipeline:
    """BASE-03: joblib load."""
    raise NotImplementedError("BASE-03: see specs/02-baseline.md")
