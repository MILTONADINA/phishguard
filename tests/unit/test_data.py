"""specs/01-data.md"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from phishguard.data import (
    clean_text,
    deduplicate,
    extract_urls,
    load_email_corpus,
    load_url_dataset,
    load_validation_emails,
    prepare_data,
    split_dataset,
)

# ---------------------------------------------------------------- DATA-01


@pytest.mark.req("DATA-01")
@pytest.mark.parametrize(
    "raw, expected",
    [
        (None, ""),
        (float("nan"), ""),
        ("", ""),
        ("  hello \n\n world\t ", "hello world"),
        ("<p>Hello</p><b>there</b>", "Hello there"),
        ("Tom &amp; Jerry &lt;3", "Tom & Jerry <3"),
        ("Visit https://Example.com/A?b=1 NOW", "Visit https://Example.com/A?b=1 NOW"),
    ],
)
def test_clean_text_cases(raw, expected):
    assert clean_text(raw) == expected


@pytest.mark.req("DATA-01")
def test_clean_text_idempotent():
    s = "<div>Dear  user,&nbsp;<br>click <a href='http://x.com'>here</a></div>\n"
    once = clean_text(s)
    assert clean_text(once) == once
    assert "  " not in once and "<" not in once


# ---------------------------------------------------------------- DATA-02


@pytest.mark.req("DATA-02")
def test_extract_urls_basic_and_order():
    text = "Go to https://a.com/x then http://b.org, also www.c.net/page."
    assert extract_urls(text) == ["https://a.com/x", "http://b.org", "www.c.net/page"]


@pytest.mark.req("DATA-02")
def test_extract_urls_dedup_and_trailing_punctuation():
    text = "(see https://a.com/x) and https://a.com/x! and 'https://b.com/y?q=1';"
    assert extract_urls(text) == ["https://a.com/x", "https://b.com/y?q=1"]


@pytest.mark.req("DATA-02")
def test_extract_urls_inside_html_attribute():
    html = '<a href="https://evil.example/login">Click</a>'
    assert extract_urls(html) == ["https://evil.example/login"]


@pytest.mark.req("DATA-02")
def test_extract_urls_case_insensitive_scheme():
    assert extract_urls("HTTPS://Shout.com/Path") == ["HTTPS://Shout.com/Path"]


@pytest.mark.req("DATA-02")
@pytest.mark.parametrize("text", [None, "", "no links here at all"])
def test_extract_urls_none(text):
    assert extract_urls(text) == []


# ---------------------------------------------------------------- DATA-03


def _write(path, df):
    df.to_csv(path, index=False)
    return path


@pytest.mark.req("DATA-03")
def test_load_combined_schema(tmp_path):
    p = _write(
        tmp_path / "phishing_email.csv",
        pd.DataFrame({"text_combined": ["  Verify <b>now</b> ", "Lunch?", ""], "label": [1, 0, 1]}),
    )
    df = load_email_corpus(p)
    assert list(df.columns) == ["text", "label", "source"]
    assert df["text"].tolist() == ["Verify now", "Lunch?"]  # empty row dropped, text cleaned
    assert df["label"].tolist() == [1, 0]
    assert set(df["source"]) == {"phishing_email"}
    assert df.index.tolist() == [0, 1]


@pytest.mark.req("DATA-03")
def test_load_subject_body_schema(tmp_path):
    p = _write(
        tmp_path / "Enron.csv",
        pd.DataFrame(
            {
                "sender": ["a@x.com", "b@y.com"],
                "subject": ["Meeting", np.nan],
                "body": ["See you at 3", "Click http://bad.io now"],
                "label": [0, 1],
                "urls": [0, 1],
            }
        ),
    )
    df = load_email_corpus(p)
    assert df["text"].tolist() == ["Meeting See you at 3", "Click http://bad.io now"]
    assert df["label"].tolist() == [0, 1]
    assert df["source"].tolist() == ["Enron", "Enron"]


@pytest.mark.req("DATA-03")
def test_load_directory_concatenates_sources(tmp_path):
    _write(tmp_path / "Ling.csv", pd.DataFrame({"subject": ["s1"], "body": ["b1"], "label": [0]}))
    _write(tmp_path / "Nazario.csv", pd.DataFrame({"subject": ["s2"], "body": ["b2"], "label": [1]}))
    df = load_email_corpus(tmp_path)
    assert sorted(df["source"]) == ["Ling", "Nazario"]
    assert len(df) == 2


@pytest.mark.req("DATA-03")
def test_load_rejects_unknown_schema(tmp_path):
    p = _write(tmp_path / "weird.csv", pd.DataFrame({"foo": ["x"], "bar": [1]}))
    with pytest.raises(ValueError):
        load_email_corpus(p)


@pytest.mark.req("DATA-03")
def test_load_rejects_bad_labels(tmp_path):
    p = _write(tmp_path / "x.csv", pd.DataFrame({"text_combined": ["a", "b"], "label": [0, 2]}))
    with pytest.raises(ValueError):
        load_email_corpus(p)


@pytest.mark.req("DATA-03")
def test_load_tolerates_undecodable_bytes(tmp_path):
    p = tmp_path / "bytes.csv"
    p.write_bytes(b"text_combined,label\n\"hello \xff\xfe world\",0\n\"verify now\",1\n")
    df = load_email_corpus(p)
    assert len(df) == 2
    assert df["label"].tolist() == [0, 1]


@pytest.mark.req("DATA-03")
def test_leakage_guard_combined_plus_per_source(tmp_path):
    _write(tmp_path / "phishing_email.csv", pd.DataFrame({"text_combined": ["a"], "label": [1]}))
    _write(tmp_path / "Enron.csv", pd.DataFrame({"subject": ["s"], "body": ["b"], "label": [0]}))
    with pytest.raises(ValueError):
        load_email_corpus(tmp_path)


# ---------------------------------------------------------------- DATA-04


@pytest.mark.req("DATA-04")
def test_load_validation_emails(tmp_path):
    p = _write(
        tmp_path / "Phishing_validation_emails.csv",
        pd.DataFrame(
            {
                "Email Text": ["Dear Jordan, renewed.", "You won! Click here.", "  "],
                "Email Type": ["Safe Email", " Phishing Email ", "Safe Email"],
            }
        ),
    )
    df = load_validation_emails(p)
    assert list(df.columns) == ["text", "label"]
    assert df["label"].tolist() == [0, 1]
    assert df.index.tolist() == [0, 1]


@pytest.mark.req("DATA-04")
def test_load_validation_emails_rejects_unknown_type(tmp_path):
    p = _write(tmp_path / "v.csv", pd.DataFrame({"Email Text": ["x"], "Email Type": ["Spam"]}))
    with pytest.raises(ValueError):
        load_validation_emails(p)


# ---------------------------------------------------------------- DATA-05


@pytest.mark.req("DATA-05")
def test_load_url_dataset_inverts_phiusiil_labels(tmp_path):
    p = _write(
        tmp_path / "PhiUSIIL_Phishing_URL_Dataset.csv",
        pd.DataFrame(
            {
                "FILENAME": ["a", "b", "c", "d"],
                "URL": [" https://www.google.com ", "http://paypa1.com/login", "https://www.google.com", ""],
                "URLLength": [22, 23, 22, 0],
                "label": [1, 0, 1, 1],  # PhiUSIIL: 1 = legitimate
            }
        ),
    )
    df = load_url_dataset(p)
    assert list(df.columns) == ["url", "label"]
    assert df["url"].tolist() == ["https://www.google.com", "http://paypa1.com/login"]  # stripped, dedup, empty dropped
    assert df["label"].tolist() == [0, 1]  # now 1 = phishing


@pytest.mark.req("DATA-05")
def test_load_url_dataset_no_inversion(tmp_path):
    p = _write(tmp_path / "u.csv", pd.DataFrame({"url": ["http://a.com", "http://b.com"], "Label": [1, 0]}))
    df = load_url_dataset(p, legitimate_is_one=False)
    assert df["label"].tolist() == [1, 0]


# ---------------------------------------------------------------- DATA-06


@pytest.mark.req("DATA-06")
def test_deduplicate_rules():
    df = pd.DataFrame(
        {
            "text": ["Hello  World", "hello world", "Pay now", "pay   NOW", "unique"],
            "label": [0, 0, 1, 0, 1],
        }
    )
    original = df.copy()
    out = deduplicate(df)
    pd.testing.assert_frame_equal(df, original)  # input untouched
    assert out["text"].tolist() == ["Hello  World", "unique"]  # first kept, conflicts dropped entirely
    assert out["label"].tolist() == [0, 1]
    assert out.index.tolist() == [0, 1]


# ---------------------------------------------------------------- DATA-07


def _big_df(n=1000, phish_rate=0.3, dup=0):
    rng = np.random.default_rng(0)
    labels = (rng.random(n) < phish_rate).astype(int)
    texts = [f"email number {i}" for i in range(n)]
    df = pd.DataFrame({"text": texts, "label": labels})
    if dup:
        df = pd.concat([df, df.iloc[:dup]], ignore_index=True)
    return df


@pytest.mark.req("DATA-07")
def test_split_sizes_and_disjoint():
    df = _big_df(dup=50)
    train, val, test = split_dataset(df, val_size=0.1, test_size=0.1, seed=42)
    n_unique = 1000
    assert len(train) + len(val) + len(test) == n_unique
    assert abs(len(val) - round(n_unique * 0.1)) <= 1
    assert abs(len(test) - round(n_unique * 0.1)) <= 1
    s_tr, s_va, s_te = set(train.text), set(val.text), set(test.text)
    assert not (s_tr & s_va) and not (s_tr & s_te) and not (s_va & s_te)
    for part in (train, val, test):
        assert part.index.tolist() == list(range(len(part)))
        assert list(part.columns) == ["text", "label"]


@pytest.mark.req("DATA-07")
def test_split_stratified():
    df = _big_df(n=2000, phish_rate=0.3)
    overall = df.label.mean()
    for part in split_dataset(df, seed=1):
        assert abs(part.label.mean() - overall) <= 0.05


@pytest.mark.req("DATA-07")
def test_split_deterministic():
    df = _big_df()
    a = split_dataset(df, seed=7)
    b = split_dataset(df, seed=7)
    c = split_dataset(df, seed=8)
    for x, y in zip(a, b):
        pd.testing.assert_frame_equal(x, y)
    assert a[2].text.tolist() != c[2].text.tolist()


# ---------------------------------------------------------------- DATA-08


@pytest.mark.req("DATA-08")
def test_prepare_data_writes_all_splits(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    n = 200
    rng = np.random.default_rng(3)
    _write(
        raw / "phishing_email.csv",
        pd.DataFrame({"text_combined": [f"mail {i}" for i in range(n)], "label": rng.integers(0, 2, n)}),
    )
    _write(
        raw / "PhiUSIIL_Phishing_URL_Dataset.csv",
        pd.DataFrame({"URL": [f"http://site{i}.com/p" for i in range(n)], "label": rng.integers(0, 2, n)}),
    )
    _write(
        raw / "Phishing_validation_emails.csv",
        pd.DataFrame(
            {
                "Email Text": ["mail 3", "brand new phish text", "a safe note"],  # "mail 3" leaks from corpus
                "Email Type": ["Safe Email", "Phishing Email", "Safe Email"],
            }
        ),
    )
    out = tmp_path / "processed"
    counts = prepare_data(raw, out, seed=42)

    expected = {"train.csv", "val.csv", "test.csv", "url_train.csv", "url_val.csv", "url_test.csv", "heldout.csv"}
    assert expected <= set(counts)
    for name in expected:
        assert (out / name).exists()
        assert counts[name] == len(pd.read_csv(out / name))
    assert list(pd.read_csv(out / "train.csv").columns) == ["text", "label"]
    assert list(pd.read_csv(out / "url_test.csv").columns) == ["url", "label"]
    assert counts["train.csv"] + counts["val.csv"] + counts["test.csv"] == n
    held = pd.read_csv(out / "heldout.csv")
    assert "mail 3" not in held.text.tolist()  # leakage removed
    assert len(held) == 2


@pytest.mark.req("DATA-08")
def test_prepare_data_heldout_optional(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write(raw / "phishing_email.csv", pd.DataFrame({"text_combined": [f"m{i}" for i in range(50)], "label": [0, 1] * 25}))
    _write(raw / "PhiUSIIL_Phishing_URL_Dataset.csv", pd.DataFrame({"URL": [f"http://s{i}.io" for i in range(50)], "label": [0, 1] * 25}))
    counts = prepare_data(raw, tmp_path / "out")
    assert "heldout.csv" not in counts
    assert not (tmp_path / "out" / "heldout.csv").exists()
