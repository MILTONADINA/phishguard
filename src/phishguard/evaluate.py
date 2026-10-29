"""Metrics and the full evaluation. Spec: specs/06-evaluation.md."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

# (column header, metrics key) for the comparison table, in display order.
TABLE_COLUMNS = [
    ("Accuracy", "accuracy"),
    ("Precision", "precision"),
    ("Recall", "recall"),
    ("F1", "f1"),
    ("FPR", "fpr"),
    ("ROC-AUC", "roc_auc"),
]


def _ratio(numerator: float, denominator: float) -> float:
    """numerator / denominator as a Python float, or 0.0 when the denominator is 0."""
    return float(numerator / denominator) if denominator else 0.0


def compute_metrics(y_true: Sequence[int], y_prob: Sequence[float], threshold: float = 0.5) -> dict:
    """EVAL-01/02: n, threshold, tp, fp, tn, fn, accuracy, precision, recall, f1, fpr, fnr, roc_auc.

    Zero-division -> 0.0. roc_auc None for a single class. Length mismatch -> ValueError.
    """
    if len(y_true) != len(y_prob):
        raise ValueError(f"y_true has {len(y_true)} values but y_prob has {len(y_prob)}")
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    # Confusion matrix, with phishing (1) as the positive class.
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    both_classes = len(np.unique(y_true)) == 2  # ROC-AUC is undefined with only one class
    return {
        "n": len(y_true),
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": _ratio(tp + tn, len(y_true)),
        "precision": precision,
        "recall": recall,
        "f1": _ratio(2 * precision * recall, precision + recall),
        "fpr": _ratio(fp, fp + tn),  # normal email wrongly flagged
        "fnr": _ratio(fn, fn + tp),  # phishing missed
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if both_classes else None,
    }


def _format(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def compare_models(results: dict[str, dict]) -> str:
    """EVAL-03: Markdown table | Model | Accuracy | Precision | Recall | F1 | FPR | ROC-AUC |, 3 decimals, n/a for None."""
    lines = [
        "| Model | " + " | ".join(header for header, _ in TABLE_COLUMNS) + " |",
        "|---|" + "---:|" * len(TABLE_COLUMNS),
    ]
    for name, metrics in results.items():
        values = " | ".join(_format(metrics.get(key)) for _, key in TABLE_COLUMNS)
        lines.append(f"| {name} | {values} |")
    return "\n".join(lines) + "\n"


def evaluate_all(data_dir: str | Path, models_dir: str | Path, reports_dir: str | Path = "reports") -> dict:
    """EVAL-04: evaluate every model on the real splits, measure latency, write metrics.json + metrics.md."""
    raise NotImplementedError("EVAL-04: see specs/06-evaluation.md")
