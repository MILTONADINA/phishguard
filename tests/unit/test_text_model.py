"""specs/03-text-model.md — uses a tiny randomly initialised DistilBERT (offline)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from phishguard.config import TextTrainConfig, TrainResult
from phishguard.text_model import PhishTextClassifier, train_text_model
from phishguard.utils import select_device

# ---------------------------------------------------------------- TEXT-01


@pytest.mark.req("TEXT-01")
def test_select_device_cpu():
    assert select_device("cpu") == torch.device("cpu")


@pytest.mark.req("TEXT-01")
def test_select_device_auto_and_fallback():
    auto = select_device()
    assert isinstance(auto, torch.device) and auto.type in {"cuda", "mps", "cpu"}
    if not torch.cuda.is_available():
        fallback = select_device("cuda")  # must not raise
        assert fallback.type in {"mps", "cpu"}


# ---------------------------------------------------------------- TEXT-02


@pytest.mark.req("TEXT-02")
def test_classifier_loads_and_predicts(tiny_text_model_dir, corpus):
    texts, _ = corpus
    clf = PhishTextClassifier(tiny_text_model_dir, max_length=32, device="cpu", batch_size=7)
    assert clf.device == torch.device("cpu")
    assert clf.max_length == 32
    assert clf.model is not None and clf.tokenizer is not None
    probs = clf.predict_proba(texts)
    assert isinstance(probs, np.ndarray)
    assert probs.shape == (len(texts),)
    assert probs.dtype.kind == "f"
    assert np.all((probs >= 0) & (probs <= 1))


@pytest.mark.req("TEXT-02")
def test_classifier_empty_and_long_input(tiny_text_model_dir):
    clf = PhishTextClassifier(tiny_text_model_dir, max_length=16, device="cpu")
    assert clf.predict_proba([]).shape == (0,)
    long_text = "verify " * 5000  # far beyond max_length: must truncate, not crash
    assert clf.predict_proba([long_text]).shape == (1,)


@pytest.mark.req("TEXT-02")
def test_classifier_deterministic_and_batch_independent(tiny_text_model_dir, corpus):
    texts, _ = corpus
    a = PhishTextClassifier(tiny_text_model_dir, max_length=32, device="cpu", batch_size=4).predict_proba(texts)
    b = PhishTextClassifier(tiny_text_model_dir, max_length=32, device="cpu", batch_size=64).predict_proba(texts)
    np.testing.assert_allclose(a, b, atol=1e-5)


@pytest.mark.req("TEXT-02")
def test_classifier_save_load_roundtrip(tiny_text_model_dir, corpus, tmp_path):
    texts, _ = corpus
    clf = PhishTextClassifier(tiny_text_model_dir, max_length=32, device="cpu")
    out = tmp_path / "saved"
    clf.save(out)
    loaded = PhishTextClassifier.load(out, max_length=32, device="cpu")
    np.testing.assert_allclose(clf.predict_proba(texts), loaded.predict_proba(texts), atol=1e-6)


# ---------------------------------------------------------------- TEXT-03/04/05


def _cfg(model_dir, out_dir, **kw):
    base = dict(
        model_name=str(model_dir),
        max_length=32,
        batch_size=8,
        eval_batch_size=32,
        epochs=3,
        lr=1e-3,
        weight_decay=0.0,
        warmup_ratio=0.0,
        patience=10,
        seed=0,
        output_dir=str(out_dir),
        device="cpu",
    )
    base.update(kw)
    return TextTrainConfig(**base)


@pytest.mark.slow
@pytest.mark.req("TEXT-03")
@pytest.mark.req("TEXT-05")
def test_train_text_model_contract(tiny_text_model_dir, tmp_path):
    from tests.conftest import make_corpus

    tr_x, tr_y = make_corpus(16, seed=1)
    va_x, va_y = make_corpus(6, seed=2)
    out = tmp_path / "text"
    result = train_text_model(tr_x, tr_y, va_x, va_y, _cfg(tiny_text_model_dir, out, epochs=3))

    assert isinstance(result, TrainResult)
    assert 1 <= len(result.history) <= 3
    for i, h in enumerate(result.history, start=1):
        assert h["epoch"] == i
        assert np.isfinite(h["train_loss"])
        assert 0.0 <= h["val_f1"] <= 1.0
    best = max(h["val_f1"] for h in result.history)
    assert result.best_val_f1 == pytest.approx(best)
    assert result.history[result.best_epoch - 1]["val_f1"] == pytest.approx(best)
    assert result.output_dir == str(out)
    reloaded = PhishTextClassifier.load(out, max_length=32, device="cpu")
    assert reloaded.predict_proba(va_x).shape == (len(va_x),)


@pytest.mark.slow
@pytest.mark.req("TEXT-03")
def test_train_text_model_early_stopping(tiny_text_model_dir, tmp_path):
    from tests.conftest import make_corpus

    tr_x, tr_y = make_corpus(8, seed=1)
    va_x, va_y = make_corpus(4, seed=2)
    # lr=0 -> val_f1 never improves after epoch 1 -> stops after 1 + patience epochs
    result = train_text_model(
        tr_x, tr_y, va_x, va_y, _cfg(tiny_text_model_dir, tmp_path / "t", epochs=10, lr=0.0, patience=1)
    )
    assert len(result.history) == 2


@pytest.mark.slow
@pytest.mark.req("TEXT-03")
def test_train_text_model_max_train_samples(tiny_text_model_dir, tmp_path):
    from tests.conftest import make_corpus

    tr_x, tr_y = make_corpus(20, seed=1)
    va_x, va_y = make_corpus(4, seed=2)
    result = train_text_model(
        tr_x, tr_y, va_x, va_y, _cfg(tiny_text_model_dir, tmp_path / "t", epochs=1, max_train_samples=10)
    )
    assert len(result.history) == 1


@pytest.mark.slow
@pytest.mark.req("TEXT-04")
def test_train_text_model_learns(tiny_text_model_dir, tmp_path):
    from tests.conftest import make_corpus

    tr_x, tr_y = make_corpus(16, seed=1)
    va_x, va_y = make_corpus(6, seed=2)
    result = train_text_model(
        tr_x, tr_y, va_x, va_y, _cfg(tiny_text_model_dir, tmp_path / "t", epochs=6, lr=2e-3)
    )
    losses = [h["train_loss"] for h in result.history]
    assert losses[-1] < losses[0]
