# 01 — Data

Module: `src/phishguard/data.py`

## Datasets

| Name | Use | File (put in `data/raw/`) | Columns | Label |
|---|---|---|---|---|
| Email corpus: Champa et al., Zenodo 8339691 (CC BY 4.0), the same six sources as the Kaggle *Phishing Email Dataset* | train / val / test for text model and baseline | **`data/raw/emails/`**: `CEAS_08.csv`, `Enron.csv`, `Ling.csv`, `Nazario.csv`, `Nigerian_Fraud.csv`, `SpamAssasin.csv` (**downloaded**). Alternative: Kaggle combined `data/raw/phishing_email.csv` | `subject`, `body`, `label` (+ `sender`, `receiver`, `date`, `urls` in some). Kaggle combined: `text_combined`, `label` | `1` phishing/spam, `0` legitimate |
| PhiUSIIL Phishing URL Dataset (UCI #967) | train / val / test for URL model | `PhiUSIIL_Phishing_URL_Dataset.csv` (**downloaded**) | `URL`, `label` (+ 54 feature columns we ignore) | **`1` = legitimate, `0` = phishing → must be inverted** |
| Phishing validation emails (Zenodo 13474746, CC BY 4.0) | **held-out test only, never trained on** | `Phishing_validation_emails.csv` (**downloaded**) | `Email Text`, `Email Type` | text: `"Phishing Email"` / `"Safe Email"` |

**Verified on download (2026-10-28):**

| File | Rows | label=1 | label=0 |
|---|---|---|---|
| CEAS_08 | 39,154 | 21,842 | 17,312 |
| Enron | 29,767 | 13,976 | 15,791 |
| Ling | 2,859 | 458 | 2,401 |
| Nazario | 1,565 | 1,565 | 0 |
| Nigerian_Fraud | 3,332 | 3,332 | 0 |
| SpamAssasin | 5,809 | 1,718 | 4,091 |
| **Total emails** | **82,486** | **42,891** | **39,595** |
| PhiUSIIL URLs | 235,795 | 134,850 (legitimate!) | 100,945 (phishing) |
| Zenodo held-out | 2,000 emails | text labels | |

Row counts are raw, before cleaning and deduplication. The PhiUSIIL header's first column carries a UTF-8 BOM (`\ufeffFILENAME`). That's harmless because only `URL` and `label` are used.

## Requirements

### DATA-01 `clean_text(text) -> str`
- `None` or NaN returns `""`.
- HTML tags (`<...>`) are replaced by a single space. HTML entities are unescaped (`&amp;` → `&`, `&lt;` → `<`).
- All whitespace runs (spaces, tabs, newlines) collapse to one space. The result is stripped.
- Case is preserved. URLs are left intact.
- Idempotent: `clean_text(clean_text(x)) == clean_text(x)`.

### DATA-02 `extract_urls(text) -> list[str]`
- Finds URLs starting with `http://`, `https://` or `www.` (case-insensitive scheme).
- Also finds them inside HTML attributes, e.g. `href="https://a.com/x"` gives `https://a.com/x`.
- Strips trailing punctuation that is not part of the URL: `. , ; : ! ? ) ] } ' " >`.
- Removes duplicates, keeping first-seen order.
- `None`, empty text or no URLs returns `[]`.

### DATA-03 `load_email_corpus(path) -> DataFrame`
Returns columns exactly `["text", "label", "source"]`.
- `path` is a CSV file or a directory of CSVs.
- **Schema detection, per file:**
  - has `text_combined` → `text = text_combined`
  - else has `body` → `text = subject + "\n\n" + body` when the subject is non-empty, else `body`
  - otherwise → `ValueError` naming the file and its columns.
- `text` is passed through `clean_text`. Rows whose cleaned text is empty are dropped.
- `label` must be in `{0, 1}` after `int` conversion. Any other value → `ValueError`.
- `source` is the file name without extension (e.g. `"Enron"`).
- Undecodable bytes must not crash the loader (replace them).
- **Leakage guard:** if a directory contains a combined-schema file *and* per-source files, raise `ValueError` (the combined file duplicates the others, so emails would leak between splits).
- Index is reset (`0..n-1`).

### DATA-04 `load_validation_emails(path) -> DataFrame`
Columns exactly `["text", "label"]`. Maps `"Phishing Email"` → 1 and `"Safe Email"` → 0 (surrounding whitespace ignored). Any other value → `ValueError`. Text cleaned with `clean_text`, empty rows dropped, index reset.

### DATA-05 `load_url_dataset(path, legitimate_is_one=True) -> DataFrame`
Columns exactly `["url", "label"]`. Finds the URL and label columns case-insensitively (`URL`/`url`, `label`/`Label`). Missing URL cells (read by pandas as NaN/NA, not as the string `"nan"`, especially in pandas ≥ 3) must be dropped, not turned into the text `"nan"`. With `legitimate_is_one=True` (PhiUSIIL), the output label is `1 - original` so that `1` = phishing. URLs are stripped of surrounding whitespace, empty URLs dropped, then `deduplicate(df, text_col="url")` is applied. Index reset.

### DATA-06 `deduplicate(df, text_col="text", label_col="label") -> DataFrame`
- Two rows are duplicates when their `text_col` values match after lowercasing and collapsing whitespace.
- Duplicates with the **same** label: keep the first.
- Duplicates with **conflicting** labels: drop every copy (we can't know the truth).
- Does not modify the input. Returns a new frame with the index reset and the original column values unchanged.

### DATA-07 `split_dataset(df, val_size=0.1, test_size=0.1, seed=42, text_col="text", label_col="label") -> (train, val, test)`
- Calls `deduplicate` first, so the three splits share no text.
- Stratified on `label_col`. Each split's phishing rate is within 0.05 of the overall rate (for ≥ 100 rows).
- `len(val)` and `len(test)` are each within 1 row of `round(n * size)`.
- Same seed → identical splits. Different seed → different splits.
- Every split has its index reset. Columns are unchanged.

### DATA-08 `prepare_data(raw_dir, out_dir, seed=42) -> dict[str, int]`
Reads from `raw_dir`:
- the email corpus (`phishing_email.csv`, or per-source CSVs in `raw_dir/emails/`),
- `PhiUSIIL_Phishing_URL_Dataset.csv`,
- `Phishing_validation_emails.csv` (optional; skip it if missing).

Writes to `out_dir`:
- `train.csv`, `val.csv`, `test.csv` with columns `text,label` (from `split_dataset`),
- `url_train.csv`, `url_val.csv`, `url_test.csv` with columns `url,label`,
- `heldout.csv` with columns `text,label`, written only if the Zenodo file exists.

**Leakage check:** heldout rows whose normalized text appears in the email corpus are removed before writing.

Returns a dict of file name → row count. Creates `out_dir` if missing.

## Known data caveats (report material)
- In Enron, Ling, CEAS and SpamAssassin, **label 1 means spam *or* phishing**. The positive class is "malicious or unwanted email", not strictly phishing.
- Nazario and Nigerian_Fraud contain **only** label 1. A model can partly learn *which source* an email came from instead of *whether* it is phishing. The held-out Zenodo set (AC-04) is the check on this. Discuss the result.
- The emails date from 1998 to 2008-ish, while today's phishing differs (e.g. QR codes, MFA prompts). Another reason the held-out score matters.

## Why these rules (report material)

- **Deduplication before splitting** prevents the same email from appearing in both train and test, which would inflate test scores.
- **Stratified splits** keep the phishing rate the same across splits, so the metrics are comparable.
- **The held-out set comes from a different source** and is never trained on, so it measures generalization, not memorization.
- **PhiUSIIL's inverted labels** are a real trap: forgetting to flip them gives a model that is confidently backwards.
