"""specs/09-acceptance.md — real data + trained models. Opt-in:

    PHISHGUARD_ACCEPTANCE=1 pytest -m acceptance

Missing data or models -> skip with a message (never a false failure).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.acceptance

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("PHISHGUARD_DATA_DIR", ROOT / "data" / "processed"))
MODELS_DIR = Path(os.environ.get("PHISHGUARD_MODELS_DIR", ROOT / "models"))
REPORTS_DIR = ROOT / "reports"


def _require(*paths: Path):
    if os.environ.get("PHISHGUARD_ACCEPTANCE") != "1":
        pytest.skip("set PHISHGUARD_ACCEPTANCE=1 to run acceptance tests")
    for p in paths:
        if not p.exists():
            pytest.skip(f"missing {p} — run prepare-data / training first")


@pytest.fixture(scope="module")
def metrics():
    _require(DATA_DIR / "test.csv", MODELS_DIR / "baseline.joblib", MODELS_DIR / "text")
    from phishguard.evaluate import evaluate_all

    return evaluate_all(DATA_DIR, MODELS_DIR, REPORTS_DIR)


@pytest.mark.req("AC-01")
@pytest.mark.req("EVAL-04")
def test_ac01_text_model_f1(metrics):
    assert (REPORTS_DIR / "metrics.json").exists() and (REPORTS_DIR / "metrics.md").exists()
    assert metrics["text_model"]["f1"] >= 0.95, metrics["text_model"]


@pytest.mark.req("AC-02")
def test_ac02_beats_baseline(metrics):
    assert metrics["text_model"]["f1"] >= metrics["baseline"]["f1"], (metrics["text_model"], metrics["baseline"])


@pytest.mark.req("AC-03")
def test_ac03_url_model_auc(metrics):
    if "url_model" not in metrics:
        pytest.skip("no URL model (Should tier not built yet)")
    assert metrics["url_model"]["roc_auc"] >= 0.95, metrics["url_model"]


@pytest.mark.req("AC-04")
def test_ac04_heldout_generalisation(metrics):
    if "detector_heldout" not in metrics:
        pytest.skip("no heldout.csv")
    assert metrics["detector_heldout"]["f1"] >= 0.75, metrics["detector_heldout"]


@pytest.mark.req("AC-05")
def test_ac05_latency(metrics):
    assert metrics["latency_ms"]["median"] < 1000, metrics["latency_ms"]


@pytest.mark.req("AC-06")
def test_ac06_demo_samples():
    _require(MODELS_DIR / "text")
    from fastapi.testclient import TestClient

    from phishguard.app import create_app

    client = TestClient(create_app(models_dir=MODELS_DIR))
    samples = ROOT / "demo_samples"
    phish = client.post("/api/analyze", json={"email_text": (samples / "obvious_phish.txt").read_text()}).json()
    legit = client.post("/api/analyze", json={"email_text": (samples / "legit.txt").read_text()}).json()
    assert phish["verdict"] == "phishing", phish
    assert legit["verdict"] == "legitimate", legit
