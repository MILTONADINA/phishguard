"""specs/07-api.md — the app is tested with an injected fake detector."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from phishguard.app import create_app
from phishguard.config import MAX_EMAIL_CHARS
from phishguard.detector import PhishDetector
from phishguard.fusion import FusionParams

PHISH = "URGENT verify your password at http://paypa1-login.com/verify"


@pytest.fixture
def client(fake_text_model, fake_url_model):
    det = PhishDetector(fake_text_model, fake_url_model, FusionParams(weight=0.5, threshold=0.5))
    return TestClient(create_app(detector=det))


@pytest.fixture
def client_no_models(tmp_path):
    return TestClient(create_app(models_dir=tmp_path / "does-not-exist"))


@pytest.mark.req("API-01")
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "models_loaded": True}


@pytest.mark.req("API-02")
def test_analyze_phishing(client):
    r = client.post("/api/analyze", json={"email_text": PHISH})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"score", "verdict", "threshold", "text_score", "url_score", "urls", "reasons"}
    assert body["verdict"] == "phishing"
    assert 0.0 <= body["score"] <= 1.0
    assert body["urls"] == [{"url": "http://paypa1-login.com/verify", "score": pytest.approx(0.97), "flagged": True}]
    assert body["reasons"]


@pytest.mark.req("API-02")
def test_analyze_legit_no_links(client):
    body = client.post("/api/analyze", json={"email_text": "Lunch on Friday?"}).json()
    assert body["verdict"] == "legitimate"
    assert body["url_score"] is None and body["urls"] == []


@pytest.mark.req("API-03")
@pytest.mark.parametrize("payload", [{}, {"email_text": ""}, {"email_text": "   \n "}, {"email_text": 123}, {"text": "hi"}])
def test_analyze_validation(client, payload):
    assert client.post("/api/analyze", json=payload).status_code == 422


@pytest.mark.req("API-04")
def test_analyze_too_long(client):
    r = client.post("/api/analyze", json={"email_text": "a" * (MAX_EMAIL_CHARS + 1)})
    assert r.status_code == 413
    assert "detail" in r.json()
    ok = client.post("/api/analyze", json={"email_text": "a" * MAX_EMAIL_CHARS})
    assert ok.status_code == 200


@pytest.mark.req("API-05")
def test_models_not_loaded(client_no_models):
    h = client_no_models.get("/health")
    assert h.status_code == 200 and h.json() == {"status": "ok", "models_loaded": False}
    r = client_no_models.post("/api/analyze", json={"email_text": PHISH})
    assert r.status_code == 503
    assert "detail" in r.json()


@pytest.mark.req("API-06")
def test_index_page(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    html = r.text
    assert '<textarea' in html and 'id="email_text"' in html
    assert "/api/analyze" in html
    assert "<button" in html.lower()
    assert "cdn" not in html.lower()  # works offline
