# 00 — Overview

## Goal

PhishGuard is a two-stage phishing email detector served through a small web app. A user pastes an email and gets a phishing score, the links it found (each scored and flagged), and short plain-English reasons.

Approved proposal: *"A two-stage phishing email detector served through a small web app. A user pastes an email and gets a phishing score, any flagged links, and a short reason."*

## How this repo works (spec-driven)

1. Each spec file defines numbered requirements (`DATA-03`, `API-02`, …).
2. Each requirement is checked by at least one test, marked `@pytest.mark.req("ID")`. `tests/unit/test_traceability.py` enforces this.
3. The code in `src/phishguard/` starts as **stubs**: real signatures and docstrings, with bodies that raise `NotImplementedError`. You implement them until the tests pass.
4. Work through `TASKS.md` in order. Commit after each task goes green.

A test is the contract. If a test and a spec disagree, the spec wins: fix the test in a separate commit and say why in the commit message.

## Architecture

```
                     ┌──────────────────────────────────────────────┐
 raw email text ───► │ data.extract_urls()   data.clean_text()      │
                     └───────┬──────────────────────┬───────────────┘
                             │ urls (≤ MAX_URLS)    │ cleaned text
                             ▼                      ▼
                 ┌──────────────────┐    ┌──────────────────────┐
                 │ Stage 2: URL     │    │ Stage 1: text        │
                 │ char-CNN         │    │ fine-tuned DistilBERT│
                 │ url_model.py     │    │ text_model.py        │
                 └────────┬─────────┘    └──────────┬───────────┘
                 per-URL P(phish)          P(phish) for the text
                          │  max over URLs          │
                          └──────────┬──────────────┘
                                     ▼
                      ┌─────────────────────────────┐
                      │ fusion.fuse(weight)          │
                      │ verdict(threshold), reasons  │
                      └──────────────┬──────────────┘
                                     ▼
                       detector.PhishDetector.analyze()
                                     ▼
                          app.py (FastAPI + web page)
```

The baseline (`baseline.py`, TF-IDF + logistic regression) is not part of the live detector. It only exists so the evaluation can show what fine-tuning gained.

## Scope tiers

| Tier | Contents | Specs |
|---|---|---|
| **Must** | data, baseline, text model, detector in text-only mode, API + page, evaluation | 01, 02, 03, 05 (FUSE-02, FUSE-05, FUSE-06), 06, 07, 08 |
| **Should** | URL model, fusion weight fitting, held-out evaluation | 04, 05 (rest), 09 (AC-04) |
| **Stretch** | robustness test on disguised text, word highlighting | not specified, add specs before building |

The detector **must** work with no URL model (`url_model=None`, DET-02). That is the safety net: if Stage 2 slips, the Must tier is still a complete project.

## Conventions (apply everywhere)

| Rule | Value |
|---|---|
| Label encoding | `1` = phishing, `0` = legitimate, in every dataframe, model output and metric |
| Probabilities | `P(phishing)` as a float in `[0, 1]` |
| Random seed | `42` (`config.SEED`) unless a test passes another |
| Text column / label column | `text`, `label` (URL data: `url`, `label`) |
| Max email length accepted by the API | `config.MAX_EMAIL_CHARS = 50_000` characters |
| Max URLs scored per email | `config.MAX_URLS = 20` |
| Default decision threshold | `config.DEFAULT_THRESHOLD = 0.5` |
| Python | 3.11 – 3.13 |
| Framework | PyTorch (+ Hugging Face Transformers for DistilBERT) |

## Spec index

| File | Area | IDs |
|---|---|---|
| 01-data.md | loading, cleaning, URL extraction, splits | DATA-01 … DATA-08 |
| 02-baseline.md | TF-IDF + logistic regression | BASE-01 … BASE-03 |
| 03-text-model.md | DistilBERT classifier + training loop | TEXT-01 … TEXT-05 |
| 04-url-model.md | character-level CNN for URLs | URL-01 … URL-05 |
| 05-fusion.md | combining scores, threshold, reasons | FUSE-01 … FUSE-06 |
| 06-evaluation.md | metrics, comparison table, full evaluation | EVAL-01 … EVAL-04 |
| 07-api.md | FastAPI app and web page | API-01 … API-06 |
| 08-detector-and-cli.md | end-to-end detector, CLI, utilities | DET-01 … DET-05, CLI-01, UTIL-01 |
| 09-acceptance.md | real-data performance targets, demo | AC-01 … AC-06 |

## Test markers

| Marker | Meaning | Runs by default |
|---|---|---|
| `req("ID")` | links the test to a requirement | — |
| `slow` | trains a tiny model (a few seconds, still offline) | yes |
| `acceptance` | needs real data and trained models | no: `pytest -m acceptance` with `PHISHGUARD_ACCEPTANCE=1` |

Unit tests never touch the network and never need the real datasets.
