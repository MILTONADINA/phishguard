"""specs/04-url-model.md"""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from phishguard.config import TrainResult, UrlTrainConfig
from phishguard.url_model import CharCNN, CharVocab, URLClassifier, train_url_model

# ---------------------------------------------------------------- URL-01


@pytest.mark.req("URL-01")
def test_vocab_ids_and_size():
    v = CharVocab(max_len=10)
    assert v.PAD == 0 and v.UNK == 1
    assert v.vocab_size == 97
    assert v.encode("a b")[1] == 2  # chr(32) (inner space) -> 2
    assert v.encode("~")[0] == 96  # chr(126) -> 96
    assert v.encode("a")[0] == ord("a") - 30


@pytest.mark.req("URL-01")
def test_vocab_pad_truncate_unknown_strip():
    v = CharVocab(max_len=5)
    assert v.encode("ab") == [ord("a") - 30, ord("b") - 30, 0, 0, 0]
    assert v.encode("abcdefgh") == [ord(c) - 30 for c in "abcde"]  # keeps the FIRST max_len chars
    assert v.encode("é")[0] == 1  # non-ASCII -> UNK
    assert v.encode("  ab  ") == v.encode("ab")  # stripped
    assert v.encode("A") != v.encode("a")  # case preserved


@pytest.mark.req("URL-01")
def test_vocab_encode_batch():
    v = CharVocab(max_len=12)
    t = v.encode_batch(["http://a.com", "x"])
    assert isinstance(t, torch.Tensor) and t.dtype == torch.long
    assert tuple(t.shape) == (2, 12)
    assert t[1].tolist() == v.encode("x")


# ---------------------------------------------------------------- URL-02


@pytest.mark.req("URL-02")
def test_charcnn_forward_shape_and_grad():
    torch.manual_seed(0)
    m = CharCNN(vocab_size=97, embed_dim=8, num_filters=4, kernel_sizes=(2, 3), dropout=0.0)
    x = CharVocab(max_len=20).encode_batch(["http://a.com", "https://bank.example/login"])
    logits = m(x)
    assert tuple(logits.shape) == (2,)
    assert logits.dtype == torch.float32
    logits.sum().backward()
    emb = [p for p in m.modules() if isinstance(p, torch.nn.Embedding)][0]
    assert emb.padding_idx == 0
    assert emb.weight.grad is not None and emb.weight.grad.abs().sum() > 0


# ---------------------------------------------------------------- URL-03


@pytest.mark.req("URL-03")
def test_urlclassifier_predict_proba():
    torch.manual_seed(0)
    clf = URLClassifier(CharCNN(embed_dim=8, num_filters=4), CharVocab(max_len=30), device="cpu")
    urls = ["http://a.com", "http://paypa1.com/login", "www.x.org"]
    p1 = clf.predict_proba(urls)
    assert isinstance(p1, np.ndarray) and p1.shape == (3,)
    assert np.all((p1 >= 0) & (p1 <= 1))
    np.testing.assert_allclose(p1, clf.predict_proba(urls))  # deterministic (eval mode)
    assert clf.predict_proba([]).shape == (0,)


# ---------------------------------------------------------------- URL-04


def _url_data(n=60, seed=0):
    rng = np.random.default_rng(seed)
    good = ["https://www.wikipedia.org", "https://github.com", "https://www.python.org", "https://news.ycombinator.com"]
    bad = ["http://paypa1-login.xyz/verify?id={}", "http://192.168.0.{}/bank.php", "http://secure-update.evil.io/{}"]
    urls, labels = [], []
    for i in range(n):
        urls.append(good[i % len(good)] + f"/p{i}")
        labels.append(0)
        urls.append(bad[i % len(bad)].format(int(rng.integers(1, 999))))
        labels.append(1)
    return urls, labels


def _ucfg(out, **kw):
    base = dict(
        max_len=48, embed_dim=8, num_filters=8, kernel_sizes=(3, 4), dropout=0.0,
        batch_size=16, epochs=5, lr=5e-3, patience=10, seed=0, output_dir=str(out), device="cpu",
    )
    base.update(kw)
    return UrlTrainConfig(**base)


@pytest.mark.slow
@pytest.mark.req("URL-04")
def test_train_url_model_contract_and_learning(tmp_path):
    tr_u, tr_y = _url_data(60, seed=1)
    va_u, va_y = _url_data(10, seed=2)
    out = tmp_path / "url"
    result = train_url_model(tr_u, tr_y, va_u, va_y, _ucfg(out, epochs=5))
    assert isinstance(result, TrainResult)
    assert 1 <= len(result.history) <= 5
    assert [h["epoch"] for h in result.history] == list(range(1, len(result.history) + 1))
    assert result.history[-1]["train_loss"] < result.history[0]["train_loss"]
    assert result.best_val_f1 == pytest.approx(max(h["val_f1"] for h in result.history))
    assert result.history[result.best_epoch - 1]["val_f1"] == pytest.approx(result.best_val_f1)
    loaded = URLClassifier.load(out, device="cpu")
    assert loaded.predict_proba(va_u).shape == (len(va_u),)


@pytest.mark.slow
@pytest.mark.req("URL-04")
def test_train_url_model_early_stopping(tmp_path):
    tr_u, tr_y = _url_data(20, seed=1)
    va_u, va_y = _url_data(5, seed=2)
    result = train_url_model(tr_u, tr_y, va_u, va_y, _ucfg(tmp_path / "u", epochs=10, lr=0.0, patience=2))
    assert len(result.history) == 3


# ---------------------------------------------------------------- URL-05


@pytest.mark.req("URL-05")
def test_urlclassifier_save_load_roundtrip(tmp_path):
    torch.manual_seed(1)
    clf = URLClassifier(
        CharCNN(embed_dim=12, num_filters=6, kernel_sizes=(2, 5), dropout=0.1), CharVocab(max_len=40), device="cpu"
    )
    urls = ["http://a.com/x", "http://paypa1.com/verify"]
    out = tmp_path / "u"
    clf.save(out)
    assert (out / "url_model.pt").exists()
    cfg = json.loads((out / "url_config.json").read_text())
    assert cfg["max_len"] == 40
    assert {"vocab_size", "embed_dim", "num_filters", "kernel_sizes", "dropout"} <= set(cfg)
    loaded = URLClassifier.load(out, device="cpu")
    np.testing.assert_allclose(clf.predict_proba(urls), loaded.predict_proba(urls), atol=1e-6)
