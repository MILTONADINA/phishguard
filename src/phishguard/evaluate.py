"""Metrics and the full evaluation. Spec: specs/06-evaluation.md."""
from __future__ import annotations

import json
import time
from dataclasses import asdict
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


LATENCY_EMAILS = 20  # emails timed through PhishDetector.analyze (after one warm-up call)


def _detector_metrics(detector, texts: Sequence[str], labels: Sequence[int]) -> dict:
    """The full pipeline, one email at a time, exactly as the web app calls it."""
    scores = [detector.analyze(t).score for t in texts]
    return compute_metrics(labels, scores, threshold=detector.fusion.threshold)


def evaluate_all(data_dir: str | Path, models_dir: str | Path, reports_dir: str | Path = "reports") -> dict:
    """EVAL-04: evaluate every model on the real splits, measure latency, write metrics.json + metrics.md."""
    # Imported here: the detector and the models import this module (compute_metrics) themselves.
    from phishguard.baseline import baseline_predict_proba, load_baseline
    from phishguard.data import read_split
    from phishguard.detector import PhishDetector

    data_dir, models_dir, reports_dir = Path(data_dir), Path(models_dir), Path(reports_dir)
    test = read_split(data_dir / "test.csv")
    heldout_path = data_dir / "heldout.csv"
    heldout = read_split(heldout_path) if heldout_path.exists() else None

    baseline = load_baseline(models_dir / "baseline.joblib")
    detector = PhishDetector.from_dir(models_dir)  # text/ (+ url/ and fusion.json when present)
    texts, labels = test["text"].tolist(), test["label"].tolist()

    results: dict[str, dict] = {}
    print("Scoring the test split...", flush=True)
    results["baseline"] = compute_metrics(labels, baseline_predict_proba(baseline, texts))
    results["text_model"] = compute_metrics(labels, detector.text_model.predict_proba(texts))
    if detector.url_model is not None:
        url_test = read_split(data_dir / "url_test.csv")
        url_probs = detector.url_model.predict_proba(url_test["url"].tolist())
        results["url_model"] = compute_metrics(url_test["label"].tolist(), url_probs)
    results["detector"] = _detector_metrics(detector, texts, labels)

    if heldout is not None:
        # A different source that no model was trained or tuned on. The baseline and text-model rows are
        # extra (not required by EVAL-04): they show which stage generalises.
        print("Scoring the held-out set...", flush=True)
        h_texts, h_labels = heldout["text"].tolist(), heldout["label"].tolist()
        results["baseline_heldout"] = compute_metrics(h_labels, baseline_predict_proba(baseline, h_texts))
        results["text_model_heldout"] = compute_metrics(h_labels, detector.text_model.predict_proba(h_texts))
        results["detector_heldout"] = _detector_metrics(detector, h_texts, h_labels)

    detector.analyze(texts[0])  # warm-up: the first call pays one-off setup costs
    times = []
    for text in texts[:LATENCY_EMAILS]:
        start = time.perf_counter()
        detector.analyze(text)
        times.append((time.perf_counter() - start) * 1000)
    latency = {"median": float(np.median(times)), "p95": float(np.percentile(times, 95))}

    metrics = {**results, "latency_ms": latency, "fusion": asdict(detector.fusion)}
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (reports_dir / "metrics.md").write_text(_metrics_markdown(results, latency, detector, len(texts), heldout))
    return metrics


def _metrics_markdown(results: dict, latency: dict, detector, n_test: int, heldout) -> str:
    fusion = detector.fusion
    stage2 = "text + URL" if detector.url_model is not None else "text only (no URL model)"
    lines = [
        "# PhishGuard evaluation",
        "",
        f"Written by `python -m phishguard evaluate`. Positive class = phishing. Test split: {n_test:,} emails"
        + (f"; held-out set (Zenodo 13474746, a different source): {len(heldout):,} emails." if heldout is not None else "."),
        f"Detector: {stage2}, fusion weight {fusion.weight} on the text score, threshold {fusion.threshold} "
        "(fitted on the validation split). Every other row uses threshold 0.5.",
        "",
        compare_models(results),
        f"Latency of `PhishDetector.analyze` over {LATENCY_EMAILS} test emails after a warm-up: "
        f"median {latency['median']:.1f} ms, 95th percentile {latency['p95']:.1f} ms.",
        "",
    ]
    return "\n".join(lines)
