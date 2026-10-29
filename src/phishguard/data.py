"""Data loading, cleaning, URL extraction and splitting. Spec: specs/01-data.md."""
from __future__ import annotations

import html
import re
from pathlib import Path

import pandas as pd

# ---- DATA-01 patterns -------------------------------------------------------
# <style>/<script> blocks hold CSS/JavaScript, not words a reader sees. Dropping their contents
# keeps that code from filling DistilBERT's limited input window (max_length tokens).
_SCRIPT_STYLE_RE = re.compile(r"<(script|style)\b[^>]*>.*?</\1\s*>", re.IGNORECASE | re.DOTALL)
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
# A real tag starts with a tag name: <p>, </b>, <br/>, <a href="...">, <o:p>, plus <!DOCTYPE ...>.
# Requiring the name keeps plain-text forms like <http://x.com>, <john@x.com> and "<3" intact.
_TAG_RE = re.compile(r"</?[A-Za-z][\w:-]*(?:\s[^>]*)?/?>|<![^>]*>")
_WHITESPACE_RE = re.compile(r"\s+")  # also matches the non-breaking space that &nbsp; becomes

# ---- DATA-02 patterns -------------------------------------------------------
# A URL starts with http://, https:// or www. and runs until whitespace, a quote or an angle bracket
# (so it stops at the end of an HTML attribute such as href="...").
_URL_RE = re.compile(r"\b(?:https?://|www\.)[^\s<>\"']+", re.IGNORECASE)
_URL_TRAILING = ".,;:!?)]}'\">"
_URL_HAS_BODY_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)


def _is_missing(value) -> bool:
    """True for None, NaN and pandas NA."""
    return value is None or (not isinstance(value, str) and pd.isna(value))


def _clean_once(text: str) -> str:
    text = _SCRIPT_STYLE_RE.sub(" ", text)
    text = _COMMENT_RE.sub(" ", text)
    text = _TAG_RE.sub(" ", text)  # tags first, then entities, so "&lt;3" becomes "<3", not a tag
    text = html.unescape(text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def clean_text(text) -> str:
    """DATA-01: strip HTML tags (to a space), unescape entities, collapse whitespace, strip.

    None/NaN -> "". Case preserved. URLs left intact. Idempotent.
    """
    if _is_missing(text):
        return ""
    text = str(text)
    # Repeat until nothing changes, so cleaning twice gives the same result as cleaning once,
    # even for escaped markup such as "&lt;b&gt;" (which only becomes a tag after unescaping).
    # Every pass that changes the text makes it shorter or only normalises whitespace, so this ends.
    while True:
        cleaned = _clean_once(text)
        if cleaned == text:
            return cleaned
        text = cleaned


def extract_urls(text) -> list[str]:
    """DATA-02: URLs starting with http://, https:// or www. (case-insensitive scheme).

    Finds URLs inside HTML attributes too. Strips trailing punctuation . , ; : ! ? ) ] } ' " >
    Deduplicated, first-seen order. None/empty/no URLs -> [].
    """
    if _is_missing(text):
        return []
    urls: list[str] = []
    for match in _URL_RE.finditer(str(text)):
        url = match.group(0).rstrip(_URL_TRAILING)
        if _URL_HAS_BODY_RE.fullmatch(url) and url not in urls:  # skip a bare "https://"
            urls.append(url)
    return urls


def _read_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    """Read a raw CSV. Undecodable bytes become the replacement character instead of crashing."""
    return pd.read_csv(path, encoding="utf-8", encoding_errors="replace", **kwargs)


def _check_labels(labels: pd.Series, where: str | Path) -> pd.Series:
    """Return the labels as ints, or raise ValueError if any value isn't 0 or 1."""
    numeric = pd.to_numeric(labels, errors="coerce")
    valid = numeric.isin([0, 1])
    if not valid.all():
        bad = sorted(set(labels[~valid].astype(str)))[:5]
        raise ValueError(f"{where}: labels must be 0 or 1, found {bad}")
    return numeric.astype(int)


def _email_text(df: pd.DataFrame, path: Path) -> pd.Series:
    """Raw email text for one file: text_combined, or subject + blank line + body."""
    if "text_combined" in df.columns:
        return df["text_combined"]
    if "body" in df.columns:
        body = df["body"].fillna("").astype(str)
        if "subject" not in df.columns:
            return body
        subject = df["subject"].fillna("").astype(str)
        has_subject = subject.str.strip() != ""
        return body.where(~has_subject, subject + "\n\n" + body)
    raise ValueError(
        f"{path}: unknown email schema (need a 'text_combined' or 'body' column); columns are {list(df.columns)}"
    )


def _load_email_file(path: Path) -> pd.DataFrame:
    df = _read_csv(path)
    if "label" not in df.columns:
        raise ValueError(f"{path}: no 'label' column; columns are {list(df.columns)}")
    text = _email_text(df, path).map(clean_text)
    labels = _check_labels(df["label"], path)
    out = pd.DataFrame({"text": text, "label": labels, "source": path.stem})
    return out[out["text"] != ""]


def _check_no_combined_mix(files: list[Path]) -> None:
    """Leakage guard: the combined Kaggle file repeats the per-source files, so never load both."""
    combined = [f.name for f in files if "text_combined" in _read_csv(f, nrows=0).columns]
    if combined and len(combined) < len(files):
        raise ValueError(
            f"Found the combined file(s) {combined} next to per-source files. The combined file repeats "
            "the same emails, so they would leak between train and test. Keep only one kind."
        )


def load_email_corpus(path: str | Path) -> pd.DataFrame:
    """DATA-03: load the Kaggle email corpus (one CSV or a directory of CSVs).

    Returns columns ["text", "label", "source"]. Detects the text_combined or subject/body schema
    per file. Labels must be 0/1. Refuses a directory that mixes the combined file with
    per-source files (leakage guard).
    """
    path = Path(path)
    if path.is_dir():
        files = sorted(path.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"No CSV files in {path}")
        _check_no_combined_mix(files)
    else:
        files = [path]
    df = pd.concat([_load_email_file(f) for f in files], ignore_index=True)
    return df[["text", "label", "source"]]


def load_validation_emails(path: str | Path) -> pd.DataFrame:
    """DATA-04: load the Zenodo held-out set. "Phishing Email" -> 1, "Safe Email" -> 0.

    Returns columns ["text", "label"].
    """
    df = _read_csv(path)
    types = df["Email Type"].astype(str).str.strip()
    label_of = {"Phishing Email": 1, "Safe Email": 0}
    unknown = sorted(set(types) - set(label_of), key=str)
    if unknown:
        raise ValueError(f"{path}: unknown 'Email Type' values {unknown[:5]}")
    out = pd.DataFrame({"text": df["Email Text"].map(clean_text), "label": types.map(label_of).astype(int)})
    return out[out["text"] != ""].reset_index(drop=True)


def load_url_dataset(path: str | Path, legitimate_is_one: bool = True) -> pd.DataFrame:
    """DATA-05: load PhiUSIIL. Returns ["url", "label"] with 1 = phishing (labels inverted by default)."""
    df = _read_csv(path)
    by_name = {c.strip().lstrip("﻿").lower(): c for c in df.columns}  # case-insensitive, BOM-safe
    if "url" not in by_name or "label" not in by_name:
        raise ValueError(f"{path}: need 'URL' and 'label' columns; columns are {list(df.columns)}")
    out = pd.DataFrame({"url": df[by_name["url"]], "label": df[by_name["label"]]})
    out = out.dropna(subset=["url"])  # drop missing URLs before converting to text, so none become "nan"
    out["url"] = out["url"].astype(str).str.strip()
    out = out[out["url"] != ""]
    labels = _check_labels(out["label"], path)
    out["label"] = 1 - labels if legitimate_is_one else labels  # PhiUSIIL uses 1 = legitimate
    return deduplicate(out, text_col="url")


def _normalize(texts: pd.Series) -> pd.Series:
    """Duplicate-detection key: lowercase with whitespace runs collapsed to one space."""
    return texts.astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()


def deduplicate(df: pd.DataFrame, text_col: str = "text", label_col: str = "label") -> pd.DataFrame:
    """DATA-06: drop duplicate texts (case/whitespace-insensitive).

    Same label -> keep the first. Conflicting labels -> drop every copy. Doesn't mutate the input.
    """
    key = _normalize(df[text_col]).to_numpy()
    labels_per_text = df.groupby(key)[label_col].transform("nunique").to_numpy()
    first_copy = ~pd.Series(key).duplicated().to_numpy()
    keep = (labels_per_text == 1) & first_copy  # conflicting copies have 2 labels: all dropped
    return df[keep].reset_index(drop=True)


def split_dataset(
    df: pd.DataFrame,
    val_size: float = 0.1,
    test_size: float = 0.1,
    seed: int = 42,
    text_col: str = "text",
    label_col: str = "label",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """DATA-07: deduplicate, then stratified train/val/test split. Deterministic per seed. Disjoint texts."""
    raise NotImplementedError("DATA-07: see specs/01-data.md")


def prepare_data(raw_dir: str | Path, out_dir: str | Path, seed: int = 42) -> dict[str, int]:
    """DATA-08: build every processed CSV split from the raw files. Returns {file name: row count}."""
    raise NotImplementedError("DATA-08: see specs/01-data.md")
