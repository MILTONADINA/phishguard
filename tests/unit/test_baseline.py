"""specs/02-baseline.md"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from phishguard.baseline import (
    baseline_predict_proba,
    build_baseline,
    load_baseline,
    save_baseline,
    train_baseline,
)


@pytest.mark.req("BASE-01")
def test_build_baseline_structure():
    pipe = build_baseline(max_features=1000, ngram_range=(1, 1), C=2.0, seed=3)
    assert isinstance(pipe, Pipeline)
    assert [name for name, _ in pipe.steps] == ["tfidf", "clf"]
    tfidf, clf = pipe.named_steps["tfidf"], pipe.named_steps["clf"]
    assert isinstance(tfidf, TfidfVectorizer) and isinstance(clf, LogisticRegression)
    assert tfidf.lowercase and tfidf.sublinear_tf
    assert tfidf.max_features == 1000 and tfidf.ngram_range == (1, 1)
    assert clf.C == 2.0 and clf.class_weight == "balanced" and clf.random_state == 3
    assert not hasattr(clf, "coef_")  # unfitted


@pytest.mark.req("BASE-01")
def test_train_baseline_fits(corpus):
    texts, labels = corpus
    model = train_baseline(texts, labels, max_features=500)
    assert hasattr(model.named_steps["clf"], "coef_")


@pytest.mark.req("BASE-02")
def test_predict_proba_shape_range_and_ranking(corpus):
    texts, labels = corpus
    model = train_baseline(texts, labels)
    probs = baseline_predict_proba(model, texts)
    assert probs.shape == (len(texts),)
    assert probs.dtype.kind == "f"
    assert np.all((probs >= 0) & (probs <= 1))
    labels = np.array(labels)
    assert probs[labels == 1].min() > probs[labels == 0].max()


@pytest.mark.req("BASE-02")
def test_predict_proba_empty(corpus):
    model = train_baseline(*corpus)
    assert baseline_predict_proba(model, []).shape == (0,)


@pytest.mark.req("BASE-03")
def test_save_load_roundtrip(corpus, tmp_path):
    texts, labels = corpus
    model = train_baseline(texts, labels)
    path = tmp_path / "nested" / "baseline.joblib"
    save_baseline(model, path)
    assert path.exists()
    loaded = load_baseline(path)
    np.testing.assert_array_equal(baseline_predict_proba(model, texts), baseline_predict_proba(loaded, texts))
