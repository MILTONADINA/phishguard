"""Data loading, cleaning, URL extraction and splitting. Spec: specs/01-data.md."""
from __future__ import annotations

import html
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# ---- DATA-01 patterns -------------------------------------------------------
# <style>/<script> blocks hold CSS/JavaScript, not words a reader sees. Dropping their contents
# keeps that code from filling DistilBERT's limited input window (max_length tokens).
_SCRIPT_STYLE_RE = re.compile(r"<(script|style)\b[^>]*>.*?</\1\s*>", re.IGNORECASE | re.DOTALL)
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
# A real tag starts with a tag name: <p>, </b>, <br/>, <a href="...">, <o:p>, plus <!DOCTYPE ...>.
# Requiring the name keeps plain-text forms like <http://x.com>, <john@x.com> and "<3" intact.
_TAG_RE = re.compile(r"</?[A-Za-z][\w:-]*(?:\s[^>]*)?/?>|<![^>]*>")
_WHITESPACE_RE = re.compile(r"\s+")  # also matches the non-breaking space that &nbsp; becomes
_SCRIPT_STYLE_START_RE = re.compile(r"<(script|style)\b", re.IGNORECASE)
# clean_text stops after this many changing passes. Real emails need at most 2. Only crafted
# input (e.g. "&amp;amp;amp;...lt;" or "<b<b<b...>>>") needs more, and each extra pass reads the whole text.
_MAX_CLEAN_PASSES = 10
# The same pattern html.unescape uses to find character references (Lib/html/__init__.py).
_CHARREF_RE = re.compile(r"&(#[0-9]+;?|#[xX][0-9a-fA-F]+;?|[^\t\n\f <&#;]{1,32};?)")
# Where markup could start: "<!" or "<" + optional "/" + tag name, then what follows the name.
_MARKUP_START_RE = re.compile(r"<(?:!|/?[A-Za-z][\w:-]*(?:(\s|/?>)|(?=(<)))?)")

# ---- DATA-02 patterns -------------------------------------------------------
# A URL starts with http://, https:// or www. and runs until whitespace, a quote or an angle bracket
# (so it stops at the end of an HTML attribute such as href="...").
_URL_RE = re.compile(r"\b(?:https?://|www\.)[^\s<>\"']+", re.IGNORECASE)
_URL_TRAILING = ".,;:!?)]}'\">"
_URL_HAS_BODY_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)


def _is_missing(value) -> bool:
    """True for None, NaN and pandas NA."""
    return value is None or (not isinstance(value, str) and pd.isna(value))


def _sub_up_to_last(pattern: re.Pattern, text: str, terminator: str) -> str:
    """pattern.sub(" ", text) for a pattern whose matches all end with terminator.

    No match can run past the last terminator, so the text after it is kept as is. This stops
    unclosed openers there ("<a ", "<!--") from making the regex rescan to the end of the text.
    """
    end = text.rfind(terminator)
    if end < 0:
        return text
    end += len(terminator)
    return pattern.sub(" ", text[:end]) + text[end:]


def _drop_script_style(text: str) -> str:
    """_SCRIPT_STYLE_RE.sub(" ", text), without rescanning the text for openers that can't close.

    When a block fails to match, no closing tag for that exact opener spelling (e.g. "script",
    "SCRIPT") follows. Later openers with the same spelling can't match either, so they are skipped.
    """
    parts: list[str] = []
    copied = search_from = 0
    unclosed: set[str] = set()
    while (start := _SCRIPT_STYLE_START_RE.search(text, search_from)) is not None:
        block = None if start.group(1) in unclosed else _SCRIPT_STYLE_RE.match(text, start.start())
        if block is None:
            unclosed.add(start.group(1))
            search_from = start.start() + 1
            continue
        parts += [text[copied : block.start()], " "]
        copied = search_from = block.end()
    parts.append(text[copied:])
    return "".join(parts)


def _clean_once(text: str) -> str:
    text = _drop_script_style(text)
    text = _sub_up_to_last(_COMMENT_RE, text, "-->")
    text = _sub_up_to_last(_TAG_RE, text, ">")  # tags first, then entities, so "&lt;3" becomes "<3", not a tag
    text = html.unescape(text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _blank_markup_starts(text: str) -> str:
    """Replace with a space each "<" that starts, or could start, a tag, comment or script block.

    Markup needs a ">" after its "<". A "<" directly after a tag name counts too, because blanking
    it puts a space after that name ("<a<b>" would become the tag "<a b>"). Plain-text forms such
    as "<3", "<http://x.com>" and "<john@x.com>" are left alone.
    """
    blanked: set[int] = set()
    for m in reversed(list(_MARKUP_START_RE.finditer(text, 0, text.rfind(">") + 1))):
        if m.group(0) == "<!" or m.group(1) or (m.group(2) and m.end() in blanked):
            blanked.add(m.start())
    if not blanked:
        return text
    chars = list(text)
    for i in blanked:
        chars[i] = " "
    return "".join(chars)


def _neutralize(text: str) -> str:
    """Make text stable under _clean_once by blanking single characters, in linear time.

    Used only when clean_text hits its pass limit. Each "&" that still starts an entity becomes a
    space, then so does each "<" that still starts markup. Everything else is kept, including
    plain "&", "<3" and URLs. Text that is already stable comes back unchanged.
    """
    text = _CHARREF_RE.sub(lambda m: m[0] if html.unescape(m[0]) == m[0] else " " + m[0][1:], text)
    text = _WHITESPACE_RE.sub(" ", _blank_markup_starts(text)).strip()
    if _clean_once(text) != text:  # safety net (e.g. "<ſcript>" blocks): with no "<" left, no markup can remain
        text = _WHITESPACE_RE.sub(" ", text.replace("<", " ")).strip()
    return text


def clean_text(text) -> str:
    """DATA-01: strip HTML tags (to a space), unescape entities, collapse whitespace, strip.

    None/NaN -> "". Case preserved. URLs left intact. Idempotent.
    """
    if _is_missing(text):
        return ""
    text = str(text)
    # Repeat until nothing changes, so cleaning twice gives the same result as cleaning once,
    # even for escaped markup such as "&lt;b&gt;" (which only becomes a tag after unescaping).
    # Allow up to _MAX_CLEAN_PASSES changing passes plus one pass that confirms nothing changes.
    for _ in range(_MAX_CLEAN_PASSES + 1):
        cleaned = _clean_once(text)
        if cleaned == text:
            return cleaned
        text = cleaned
    # Still changing (crafted, deeply nested input): stop here and blank what is left instead.
    return _neutralize(text)


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
    df = deduplicate(df, text_col=text_col, label_col=label_col)  # no text can land in two splits
    n = len(df)
    # Whole-row counts (not fractions) so val and test are exactly round(n * size) rows each.
    rest, test = _split_off(df, round(n * test_size), seed, label_col)
    train, val = _split_off(rest, round(n * val_size), seed, label_col)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def _split_off(df: pd.DataFrame, n_rows: int, seed: int, label_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Randomly take n_rows out of df, keeping the label mix the same. Returns (rest, taken)."""
    if n_rows == 0:
        return df, df.iloc[0:0]
    label_counts = df[label_col].value_counts()
    stratify = df[label_col] if label_counts.min() >= 2 else None  # stratifying needs 2+ rows per class
    rest, taken = train_test_split(df, test_size=n_rows, random_state=seed, stratify=stratify)
    return rest, taken


def read_split(path: str | Path) -> pd.DataFrame:
    """Read a processed CSV (train.csv, url_test.csv, ...) exactly as prepare_data wrote it.

    keep_default_na=False stops pandas from turning an email whose whole text is "NA", "null" or
    "None" into a missing value, which would crash the text models.
    """
    return pd.read_csv(path, keep_default_na=False)


def prepare_data(raw_dir: str | Path, out_dir: str | Path, seed: int = 42) -> dict[str, int]:
    """DATA-08: build every processed CSV split from the raw files. Returns {file name: row count}."""
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}

    def write(df: pd.DataFrame, name: str) -> None:
        df.to_csv(out_dir / name, index=False)
        counts[name] = len(df)

    # Emails: the combined Kaggle file if present, otherwise the per-source files.
    combined = raw_dir / "phishing_email.csv"
    emails = load_email_corpus(combined if combined.exists() else raw_dir / "emails")
    email_splits = split_dataset(emails[["text", "label"]], seed=seed)
    for name, part in zip(("train.csv", "val.csv", "test.csv"), email_splits):
        write(part, name)

    urls = load_url_dataset(raw_dir / "PhiUSIIL_Phishing_URL_Dataset.csv")
    url_splits = split_dataset(urls, seed=seed, text_col="url")
    for name, part in zip(("url_train.csv", "url_val.csv", "url_test.csv"), url_splits):
        write(part, name)

    # Held-out set (optional): drop any email that also appears in the training corpus.
    heldout_path = raw_dir / "Phishing_validation_emails.csv"
    if heldout_path.exists():
        heldout = load_validation_emails(heldout_path)
        corpus_texts = set(_normalize(emails["text"]))
        leaked = _normalize(heldout["text"]).isin(corpus_texts).to_numpy()
        write(heldout[~leaked].reset_index(drop=True), "heldout.csv")

    return counts
