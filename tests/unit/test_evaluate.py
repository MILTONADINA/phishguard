"""specs/06-evaluation.md (EVAL-04 is checked by the acceptance tests)."""
from __future__ import annotations

import pytest

from phishguard.evaluate import compare_models, compute_metrics

KEYS = {"n", "threshold", "tp", "fp", "tn", "fn", "accuracy", "precision", "recall", "f1", "fpr", "fnr", "roc_auc"}


@pytest.mark.req("EVAL-01")
def test_compute_metrics_hand_computed():
    y_true = [1, 1, 1, 0, 0, 0, 0, 1]
    y_prob = [0.9, 0.8, 0.3, 0.2, 0.6, 0.1, 0.4, 0.7]
    m = compute_metrics(y_true, y_prob, threshold=0.5)
    assert set(m) == KEYS
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (3, 1, 3, 1)
    assert m["n"] == 8 and m["threshold"] == 0.5
    for k in ("accuracy", "precision", "recall", "f1"):
        assert m[k] == pytest.approx(0.75)
    assert m["fpr"] == pytest.approx(0.25) and m["fnr"] == pytest.approx(0.25)
    assert m["roc_auc"] == pytest.approx(14 / 16)
    assert all(isinstance(m[k], (int, float)) for k in KEYS)


@pytest.mark.req("EVAL-01")
def test_compute_metrics_threshold_matters():
    m = compute_metrics([1, 0], [0.4, 0.3], threshold=0.35)
    assert (m["tp"], m["fp"]) == (1, 0)


@pytest.mark.req("EVAL-01")
def test_compute_metrics_length_mismatch():
    with pytest.raises(ValueError):
        compute_metrics([1, 0], [0.5])


@pytest.mark.req("EVAL-02")
def test_compute_metrics_single_class_and_zero_division():
    m = compute_metrics([0, 0, 0], [0.1, 0.2, 0.3])
    assert m["roc_auc"] is None
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["f1"] == 0.0
    assert m["fpr"] == 0.0 and m["fnr"] == 0.0 and m["accuracy"] == 1.0
    m2 = compute_metrics([1, 1], [0.9, 0.2])
    assert m2["roc_auc"] is None and m2["fpr"] == 0.0 and m2["recall"] == pytest.approx(0.5)


@pytest.mark.req("EVAL-03")
def test_compare_models_table():
    a = compute_metrics([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.3])
    b = compute_metrics([1, 1], [0.9, 0.8])  # roc_auc None
    table = compare_models({"baseline": a, "text_model": b})
    lines = table.strip().splitlines()
    assert lines[0].replace(" ", "") == "|Model|Accuracy|Precision|Recall|F1|FPR|ROC-AUC|"
    assert set(lines[1].replace("|", "").replace(" ", "")) <= {"-", ":"}
    assert lines[2].split("|")[1].strip() == "baseline"
    assert lines[3].split("|")[1].strip() == "text_model"
    assert "1.000" in lines[2]
    assert "n/a" in lines[3]
