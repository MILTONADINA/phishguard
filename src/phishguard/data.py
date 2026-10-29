"""Data loading, cleaning, URL extraction and splitting. Spec: specs/01-data.md."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def clean_text(text) -> str:
    """DATA-01: strip HTML tags (to a space), unescape entities, collapse whitespace, strip.

    None/NaN -> "". Case preserved. URLs left intact. Idempotent.
    """
    raise NotImplementedError("DATA-01: see specs/01-data.md")


def extract_urls(text) -> list[str]:
    """DATA-02: URLs starting with http://, https:// or www. (case-insensitive scheme).

    Finds URLs inside HTML attributes too. Strips trailing punctuation . , ; : ! ? ) ] } ' " >
    Deduplicated, first-seen order. None/empty/no URLs -> [].
    """
    raise NotImplementedError("DATA-02: see specs/01-data.md")


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
