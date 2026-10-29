"""specs/05-fusion.md"""
from __future__ import annotations

import json

import pytest

from phishguard.fusion import (
    FusionParams,
    aggregate_url_scores,
    build_reasons,
    fit_fusion,
    fuse,
    verdict,
)


@pytest.mark.req("FUSE-01")
def test_aggregate_url_scores():
    assert aggregate_url_scores([0.1, 0.9, 0.3]) == pytest.approx(0.9)
    assert aggregate_url_scores([]) is None


@pytest.mark.req("FUSE-02")
def test_fuse_rules():
    assert fuse(0.8, None, 0.3) == pytest.approx(0.8)
    assert fuse(0.8, 0.2, 0.25) == pytest.approx(0.25 * 0.8 + 0.75 * 0.2)
    assert fuse(1.0, 1.0, 0.5) <= 1.0
    assert isinstance(fuse(0.5, 0.5, 0.5), float)
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError):
            fuse(0.5, 0.5, bad)


@pytest.mark.req("FUSE-03")
def test_fit_fusion_prefers_informative_url_scores():
    labels = [1, 1, 1, 1, 0, 0, 0, 0]
    text = [0.6, 0.4, 0.7, 0.45, 0.55, 0.3, 0.2, 0.5]
    url = [0.95, 0.9, None, 0.92, 0.05, 0.1, None, 0.02]
    p = fit_fusion(text, url, labels)
    assert isinstance(p, FusionParams)
    assert p.weight < 1.0
    preds = [int(fuse(t, u, p.weight) >= p.threshold) for t, u in zip(text, url)]
    assert preds == labels  # perfect separation is reachable and must be found


@pytest.mark.req("FUSE-03")
def test_fit_fusion_tie_break_and_determinism():
    # Text alone separates perfectly and there are no URLs -> every weight ties -> largest weight (1.0)
    labels = [1, 1, 0, 0]
    text = [0.9, 0.8, 0.2, 0.1]
    url = [None, None, None, None]
    p = fit_fusion(text, url, labels)
    assert p.weight == pytest.approx(1.0)
    assert p.threshold == pytest.approx(0.5)  # closest to 0.5 among the perfect thresholds
    assert fit_fusion(text, url, labels) == p


@pytest.mark.req("FUSE-03")
def test_fit_fusion_custom_grid():
    p = fit_fusion([0.9, 0.1], [None, None], [1, 0], weights=[0.2], thresholds=[0.3])
    assert p == FusionParams(weight=0.2, threshold=0.3)


@pytest.mark.req("FUSE-04")
def test_fusion_params_roundtrip(tmp_path):
    p = FusionParams(weight=0.7, threshold=0.35)
    path = tmp_path / "deep" / "fusion.json"
    p.save(path)
    assert json.loads(path.read_text()) == {"weight": 0.7, "threshold": 0.35}
    assert FusionParams.load(path) == p
    assert FusionParams() == FusionParams(weight=0.5, threshold=0.5)


@pytest.mark.req("FUSE-05")
def test_verdict():
    assert verdict(0.5, 0.5) == "phishing"
    assert verdict(0.49, 0.5) == "legitimate"
    assert verdict(0.2, 0.1) == "phishing"


@pytest.mark.req("FUSE-06")
def test_reasons_phishing_with_flagged_link():
    r = build_reasons(0.93, [("http://paypa1.com/x", 0.97), ("https://github.com", 0.02)], 0.5)
    joined = " ".join(r)
    assert "looks like phishing" in joined and "0.93" in joined
    assert "http://paypa1.com/x" in joined and "0.97" in joined
    assert "https://github.com" not in joined  # not flagged -> not listed


@pytest.mark.req("FUSE-06")
def test_reasons_normal_with_clean_links():
    r = build_reasons(0.12, [("https://github.com", 0.02), ("https://python.org", 0.1)], 0.5)
    joined = " ".join(r)
    assert "looks normal" in joined and "0.12" in joined
    assert "No suspicious links" in joined and "2" in joined


@pytest.mark.req("FUSE-06")
def test_reasons_no_links():
    r = build_reasons(0.3, [], 0.5)
    assert r and any("No links" in s for s in r)
