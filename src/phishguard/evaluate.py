"""Metrics and the full evaluation. Spec: specs/06-evaluation.md."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence


def compute_metrics(y_true: Sequence[int], y_prob: Sequence[float], threshold: float = 0.5) -> dict:
    """EVAL-01/02: n, threshold, tp, fp, tn, fn, accuracy, precision, recall, f1, fpr, fnr, roc_auc.

    Zero-division -> 0.0. roc_auc None for a single class. Length mismatch -> ValueError.
    """
    raise NotImplementedError("EVAL-01: see specs/06-evaluation.md")


def compare_models(results: dict[str, dict]) -> str:
    """EVAL-03: Markdown table | Model | Accuracy | Precision | Recall | F1 | FPR | ROC-AUC |, 3 decimals, n/a for None."""
    raise NotImplementedError("EVAL-03: see specs/06-evaluation.md")


def evaluate_all(data_dir: str | Path, models_dir: str | Path, reports_dir: str | Path = "reports") -> dict:
    """EVAL-04: evaluate every model on the real splits, measure latency, write metrics.json + metrics.md."""
    raise NotImplementedError("EVAL-04: see specs/06-evaluation.md")
