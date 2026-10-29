"""TF-IDF + logistic regression baseline. Spec: specs/02-baseline.md."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_baseline(
    max_features: int = 50_000,
    ngram_range: tuple[int, int] = (1, 2),
    C: float = 1.0,
    seed: int = 42,
) -> Pipeline:
    """BASE-01: unfitted Pipeline([("tfidf", TfidfVectorizer(...)), ("clf", LogisticRegression(...))])."""
    return Pipeline(
        [
            # Words and word pairs, weighted by how distinctive they are. sublinear_tf dampens
            # long emails that repeat the same word many times.
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True, sublinear_tf=True, max_features=max_features, ngram_range=ngram_range
                ),
            ),
            # class_weight="balanced" counts both classes equally even if one has more emails.
            ("clf", LogisticRegression(C=C, class_weight="balanced", max_iter=1000, random_state=seed)),
        ]
    )


def train_baseline(texts: Sequence[str], labels: Sequence[int], **kwargs) -> Pipeline:
    """BASE-01: build_baseline(**kwargs), then fit it."""
    model = build_baseline(**kwargs)
    model.fit(list(texts), list(labels))
    return model


def baseline_predict_proba(model: Pipeline, texts: Sequence[str]) -> np.ndarray:
    """BASE-02: P(phishing) per text, shape (len(texts),). Empty input -> shape (0,)."""
    texts = list(texts)
    if not texts:
        return np.empty((0,), dtype=float)
    phishing_column = list(model.classes_).index(1)
    return model.predict_proba(texts)[:, phishing_column].astype(float)


def save_baseline(model: Pipeline, path: str | Path) -> None:
    """BASE-03: joblib dump, creating parent folders."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_baseline(path: str | Path) -> Pipeline:
    """BASE-03: joblib load."""
    # joblib uses pickle, which can run code from a malicious file. That's acceptable here: the spec
    # requires joblib and we only ever load the model file this project saved itself (models/).
    return joblib.load(path)
