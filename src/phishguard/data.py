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


def load_email_corpus(path: str | Path) -> pd.DataFrame:
    """DATA-03: load the Kaggle email corpus (one CSV or a directory of CSVs).

    Returns columns ["text", "label", "source"]. Detects the text_combined or subject/body schema
    per file. Labels must be 0/1. Refuses a directory that mixes the combined file with
    per-source files (leakage guard).
    """
    raise NotImplementedError("DATA-03: see specs/01-data.md")


def load_validation_emails(path: str | Path) -> pd.DataFrame:
    """DATA-04: load the Zenodo held-out set. "Phishing Email" -> 1, "Safe Email" -> 0.

    Returns columns ["text", "label"].
    """
    raise NotImplementedError("DATA-04: see specs/01-data.md")


def load_url_dataset(path: str | Path, legitimate_is_one: bool = True) -> pd.DataFrame:
    """DATA-05: load PhiUSIIL. Returns ["url", "label"] with 1 = phishing (labels inverted by default)."""
    raise NotImplementedError("DATA-05: see specs/01-data.md")


def deduplicate(df: pd.DataFrame, text_col: str = "text", label_col: str = "label") -> pd.DataFrame:
    """DATA-06: drop duplicate texts (case/whitespace-insensitive).

    Same label -> keep the first. Conflicting labels -> drop every copy. Doesn't mutate the input.
    """
    raise NotImplementedError("DATA-06: see specs/01-data.md")


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
