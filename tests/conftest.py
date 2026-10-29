"""Shared fixtures. Everything here is offline and synthetic: no network, no real datasets."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

PHISH_TEMPLATES = [
    "urgent your account is suspended verify your password now at {url}",
    "you won a free prize claim your reward immediately click {url}",
    "security alert unusual login confirm your bank details today {url}",
    "your invoice payment failed update billing information urgently {url}",
    "final notice your mailbox is full login to restore access {url}",
]
LEGIT_TEMPLATES = [
    "hi team the meeting notes from monday are attached see you thursday",
    "thanks for lunch yesterday let me know when you are free next week",
    "the quarterly report draft is ready for review please add comments",
    "reminder the project demo moved to friday afternoon in room two",
    "here is the agenda for the planning session and the shared document",
]
PHISH_URLS = [
    "http://paypa1-secure-login.com/verify?id=88213",
    "http://192.168.10.5/bank/login.php",
    "https://account-update.example-security.xyz/confirm",
]
LEGIT_URLS = ["https://www.wikipedia.org", "https://github.com", "https://www.python.org"]

# Words for the tiny tokenizer vocabulary (everything the synthetic corpus uses).
_VOCAB_WORDS = sorted(
    {w for t in PHISH_TEMPLATES + LEGIT_TEMPLATES for w in t.replace("{url}", "").split()}
)


def make_corpus(n_per_class: int = 30, seed: int = 0, with_urls: bool = True) -> tuple[list[str], list[int]]:
    """Clearly separable synthetic emails. Returns (texts, labels), 1 = phishing."""
    rng = random.Random(seed)
    texts, labels = [], []
    for i in range(n_per_class):
        url = rng.choice(PHISH_URLS) if with_urls else ""
        texts.append(rng.choice(PHISH_TEMPLATES).format(url=url) + f" ref {i}")
        labels.append(1)
        texts.append(rng.choice(LEGIT_TEMPLATES) + f" ref {i}")
        labels.append(0)
    return texts, labels


@pytest.fixture
def corpus():
    return make_corpus()


@pytest.fixture(scope="session")
def tiny_text_model_dir(tmp_path_factory) -> Path:
    """A tiny randomly initialised DistilBERT + WordPiece tokenizer saved to disk (offline)."""
    import torch
    from transformers import (
        DistilBertConfig,
        DistilBertForSequenceClassification,
        DistilBertTokenizerFast,
    )

    d = tmp_path_factory.mktemp("tiny_distilbert")
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + _VOCAB_WORDS + [str(i) for i in range(100)]
    vocab_file = d / "vocab.txt"
    vocab_file.write_text("\n".join(vocab) + "\n", encoding="utf-8")
    tokenizer = DistilBertTokenizerFast(vocab_file=str(vocab_file), do_lower_case=True)
    config = DistilBertConfig(
        vocab_size=len(vocab),
        dim=32,
        n_layers=1,
        n_heads=2,
        hidden_dim=64,
        max_position_embeddings=128,
        num_labels=2,
        dropout=0.0,
        attention_dropout=0.0,
        seq_classif_dropout=0.0,
    )
    torch.manual_seed(0)
    model = DistilBertForSequenceClassification(config)
    model.save_pretrained(d)
    tokenizer.save_pretrained(d)
    return d


class FakeTextModel:
    """predict_proba via keyword rule. Records calls."""

    def __init__(self, phish_words=("urgent", "verify", "prize", "suspended", "password")):
        self.phish_words = phish_words
        self.calls: list[list[str]] = []

    def predict_proba(self, texts):
        texts = list(texts)
        self.calls.append(texts)
        return np.array(
            [0.95 if any(w in t.lower() for w in self.phish_words) else 0.05 for t in texts], dtype=float
        )


class FakeUrlModel:
    """Flags URLs containing suspicious tokens. Records calls."""

    def __init__(self, bad_tokens=("paypa1", "192.168", "xyz", "evil")):
        self.bad_tokens = bad_tokens
        self.calls: list[list[str]] = []

    def predict_proba(self, urls):
        urls = list(urls)
        self.calls.append(urls)
        return np.array([0.97 if any(b in u for b in self.bad_tokens) else 0.03 for u in urls], dtype=float)


@pytest.fixture
def fake_text_model():
    return FakeTextModel()


@pytest.fixture
def fake_url_model():
    return FakeUrlModel()
