"""specs/08-detector-and-cli.md (DET-01 … DET-05, UTIL-01)."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from phishguard.config import MAX_URLS
from phishguard.detector import AnalysisResult, PhishDetector, UrlResult
from phishguard.fusion import FusionParams
from phishguard.utils import set_seed

PHISH = 'URGENT: your account is suspended. <a href="http://paypa1-login.com/verify">Verify</a> or visit https://github.com'
LEGIT = "Hi team, the meeting notes are attached. See you Thursday."


@pytest.mark.req("DET-01")
def test_analyze_full_pipeline(fake_text_model, fake_url_model):
    det = PhishDetector(fake_text_model, fake_url_model, FusionParams(weight=0.5, threshold=0.5))
    r = det.analyze(PHISH)
    assert isinstance(r, AnalysisResult)
    assert r.text_score == pytest.approx(0.95)
    assert [u.url for u in r.urls] == ["http://paypa1-login.com/verify", "https://github.com"]
    assert all(isinstance(u, UrlResult) for u in r.urls)
    assert [u.flagged for u in r.urls] == [True, False]
    assert r.url_score == pytest.approx(0.97)
    assert r.score == pytest.approx(0.5 * 0.95 + 0.5 * 0.97)
    assert r.verdict == "phishing" and r.threshold == 0.5
    assert any("paypa1-login.com" in s for s in r.reasons)
    assert len(fake_url_model.calls) == 1  # one batched call
    # text model sees CLEANED text (no HTML tags)
    assert "<a" not in fake_text_model.calls[0][0]


@pytest.mark.req("DET-01")
def test_analyze_legit_and_to_dict(fake_text_model, fake_url_model):
    det = PhishDetector(fake_text_model, fake_url_model, FusionParams(weight=0.5, threshold=0.5))
    r = det.analyze(LEGIT)
    assert r.verdict == "legitimate"
    assert r.url_score is None and r.urls == []
    d = r.to_dict()
    json.dumps(d)  # serialisable
    assert set(d) == {"score", "verdict", "threshold", "text_score", "url_score", "urls", "reasons"}
    assert d["url_score"] is None and isinstance(d["score"], float)
    d2 = det.analyze(PHISH).to_dict()
    assert d2["urls"][0] == {"url": "http://paypa1-login.com/verify", "score": pytest.approx(0.97), "flagged": True}


@pytest.mark.req("DET-02")
def test_text_only_mode(fake_text_model):
    det = PhishDetector(fake_text_model, None, FusionParams(weight=0.3, threshold=0.5))
    r = det.analyze(PHISH)
    assert r.url_score is None
    assert r.score == pytest.approx(r.text_score)
    assert len(r.urls) == 2 and all(u.score == 0.0 and not u.flagged for u in r.urls)
    assert r.reasons


@pytest.mark.req("DET-03")
@pytest.mark.parametrize("bad", ["", "   \n\t "])
def test_empty_input_raises(fake_text_model, bad):
    with pytest.raises(ValueError):
        PhishDetector(fake_text_model).analyze(bad)


@pytest.mark.req("DET-04")
def test_url_cap(fake_text_model, fake_url_model):
    many = " ".join(f"http://site{i}.com/p" for i in range(MAX_URLS + 15))
    r = PhishDetector(fake_text_model, fake_url_model).analyze("check these " + many)
    assert len(r.urls) == MAX_URLS
    assert len(fake_url_model.calls[0]) == MAX_URLS
    assert r.urls[0].url == "http://site0.com/p"


@pytest.mark.slow
@pytest.mark.req("DET-05")
def test_from_dir(tmp_path, tiny_text_model_dir):
    from phishguard.text_model import PhishTextClassifier
    from phishguard.url_model import CharCNN, CharVocab, URLClassifier

    models = tmp_path / "models"
    PhishTextClassifier(tiny_text_model_dir, max_length=32, device="cpu").save(models / "text")

    det = PhishDetector.from_dir(models)  # text only, default fusion
    r = det.analyze(PHISH)
    assert r.url_score is None and r.score == pytest.approx(r.text_score)
    assert det.analyze(LEGIT).threshold == 0.5

    torch.manual_seed(0)
    URLClassifier(CharCNN(embed_dim=8, num_filters=4), CharVocab(max_len=40), device="cpu").save(models / "url")
    r2 = PhishDetector.from_dir(models).analyze(PHISH)
    assert r2.url_score is not None
    assert r2.score == pytest.approx(0.5 * r2.text_score + 0.5 * r2.url_score)  # default weight 0.5 with URL model

    FusionParams(weight=0.8, threshold=0.4).save(models / "fusion.json")
    r3 = PhishDetector.from_dir(models).analyze(PHISH)
    assert r3.threshold == 0.4
    assert r3.score == pytest.approx(0.8 * r3.text_score + 0.2 * r3.url_score)


@pytest.mark.req("DET-05")
def test_from_dir_missing_text_model(tmp_path):
    with pytest.raises(FileNotFoundError):
        PhishDetector.from_dir(tmp_path / "nothing")


@pytest.mark.req("UTIL-01")
def test_set_seed_reproducible():
    set_seed(123)
    a_t, a_n = torch.rand(3), np.random.rand(3)
    set_seed(123)
    b_t, b_n = torch.rand(3), np.random.rand(3)
    assert torch.equal(a_t, b_t)
    np.testing.assert_array_equal(a_n, b_n)
