# TASKS — build order

Work top to bottom. For each task: read the spec → run its tests (red) → implement → tests green → commit.
Commit after **every** task (the rubric grades incremental history: 10 pts). Use the given messages (prefixed `T<n>:`).
In Claude Code: `/next-task` does exactly one task and commits; `/status` shows progress.

## Progress (tick as each task is committed)
- [x] T0 Setup: venv, install, git init, first commit
- [x] T1 Seeding + device selection
- [x] T2 Text cleaning + URL extraction
- [ ] T3 Dataset loaders
- [ ] T4 Dedup, splits, prepare-data (+ CLI parser) — run it on the real data
- [ ] T5 Evaluation metrics
- [ ] T6 Baseline — train it
- [ ] T7 DistilBERT classifier wrapper
- [ ] T8 Fine-tuning loop — quick run, then full run
- [ ] T9 Fusion, verdict, reasons
- [ ] T10 Detector
- [ ] T11 Web app + API — **Must tier done**
- [ ] T12 URL model — train it
- [ ] T13 Fusion fitting + full evaluation + acceptance — **Should tier done**
- [ ] T14 Report, demo, recording

Starting state: `pytest` → 3 passed (traceability), everything else fails with `NotImplementedError`.

| # | Task | Spec | Run | Done when | Commit message |
|---|---|---|---|---|---|
| T0 | Set up: `python3.12 -m venv .venv`, `.venv/bin/pip install -e ".[dev]"`, `git init`, first commit. Milton: PyCharm project + GitHub repo (add `dgriff24` if private) | README, CLAUDE.md | `.venv/bin/python -m pytest -q` | 3 passed, rest red | `T0: Add specs, test suite and stubs` |
| T1 | `set_seed`, `select_device` | 08 UTIL-01, 03 TEXT-01 | `.venv/bin/python -m pytest -k "set_seed or select_device"` | green | `T1: Implement seeding and device selection` |
| T2 | `clean_text`, `extract_urls` | 01 DATA-01/02 | `.venv/bin/python -m pytest tests/unit/test_data.py -k "clean or extract"` | green | `T2: Implement text cleaning and URL extraction` |
| T3 | Loaders (`load_email_corpus`, `load_validation_emails`, `load_url_dataset`) | 01 DATA-03/04/05 | `.venv/bin/python -m pytest tests/unit/test_data.py -k load` | green | `T3: Implement dataset loaders` |
| T4 | `deduplicate`, `split_dataset`, `prepare_data`; CLI parser + `prepare-data` handler. Data is already in `data/raw/` (verified, spec 01). Run `.venv/bin/python -m phishguard prepare-data` and note the row counts after dedup in NOTES.md | 01 DATA-06/07/08, 08 CLI-01 | `.venv/bin/python -m pytest tests/unit/test_data.py tests/unit/test_cli.py` | green + `data/processed/*.csv` exist. **Sanity check (reference run on this data):** emails 65,510 / 8,189 / 8,189 (train/val/test, from 82,486 raw), URLs 188,296 / 23,537 / 23,537, heldout 2,000 (1,000/1,000). Small differences are OK if `clean_text` differs slightly; big ones mean a bug | `T4: Add dedup, stratified splits and prepare-data` |
| T5 | `compute_metrics`, `compare_models` | 06 EVAL-01/02/03 | `.venv/bin/python -m pytest tests/unit/test_evaluate.py` | green | `T5: Implement evaluation metrics` |
| T6 | Baseline + `train-baseline` handler. Train it and record the val metrics in NOTES.md | 02 BASE-01/02/03 | `.venv/bin/python -m pytest tests/unit/test_baseline.py` | green + `models/baseline.joblib` | `T6: Add TF-IDF + logistic regression baseline` |
| T7 | `PhishTextClassifier` | 03 TEXT-02 | `.venv/bin/python -m pytest tests/unit/test_text_model.py -k classifier` | green | `T7: Add DistilBERT classifier wrapper` |
| T8 | `train_text_model` + `train-text` handler. First real run: `--max-train-samples 20000`, then the full run (overnight is fine) | 03 TEXT-03/04/05 | `.venv/bin/python -m pytest tests/unit/test_text_model.py` | green + `models/text/` | `T8: Add fine-tuning loop with early stopping` |
| T9 | Fusion functions + `FusionParams` | 05 FUSE-01…06 | `.venv/bin/python -m pytest tests/unit/test_fusion.py` | green | `T9: Implement score fusion, verdict and reasons` |
| T10 | `PhishDetector` (+ `AnalysisResult.to_dict`) | 08 DET-01…05 | `.venv/bin/python -m pytest tests/unit/test_detector.py` | green | `T10: Add end-to-end detector (text-only mode)` |
| T11 | FastAPI app + page + `serve` handler. Open http://127.0.0.1:8000 and paste the demo samples | 07 API-01…06 | `.venv/bin/python -m pytest tests/unit/test_app.py` | green + page works | `T11: Add web app and API` — **Must tier done** |
| T12 | URL model + `train-url` handler. Train it. Check the URL-length shortcut (spec 04 risk) | 04 URL-01…05 | `.venv/bin/python -m pytest tests/unit/test_url_model.py` | green + `models/url/` | `T12: Add character-level CNN URL model` |
| T13 | `fit-fusion` handler, `evaluate_all` + `evaluate` handler. Run `fit-fusion`, `evaluate`, then the acceptance suite | 06 EVAL-04, 09 AC-01…06 | `PHISHGUARD_ACCEPTANCE=1 .venv/bin/python -m pytest -m acceptance` | `reports/metrics.md` written; ACs pass or misses explained | `T13: Fit fusion weights and add full evaluation` — **Should tier done** |
| T14 | REPORT.md, rehearse the 5-minute demo, screen recording | 09 checklist | `.venv/bin/python -m pytest` (all green) | submitted | `T14: Add final report` |

## Calendar (due before class Wed Nov 11)

| Date | Tasks | Also due in AI |
|---|---|---|
| Wed Oct 28 (tonight) | T0 | HW Embeddings + HW Logistic Regression (11:59 PM) |
| Thu Oct 29 | T1–T3 | |
| Fri Oct 30 | T4–T5 | |
| Sat Oct 31 | T6–T7 | |
| Sun Nov 1 – Mon Nov 2 | T8 (training runs overnight) | |
| Tue Nov 3 | T9–T10 | |
| Wed Nov 4 | T11 → **Must tier complete** | HW Classification + HW Neural Networks |
| Thu Nov 5 – Fri Nov 6 | T12 | |
| Sat Nov 7 – Sun Nov 8 | T13 → **Should tier complete** | |
| Mon Nov 9 – Tue Nov 10 | T14 | |
| Wed Nov 11 | submit before class, present | HW Ethics |

**If behind on Nov 5:** skip T12 and do T13 in text-only mode (the detector supports it, DET-02). The Must tier is a complete project.

## Rules
- Don't edit a test to make it pass. If a test is wrong, fix the **spec** first, then the test, in its own commit with the reason.
- Never train or tune on `test.csv` or `heldout.csv`. Fit fusion on `val.csv` only.
- Write problems and decisions in `NOTES.md` as they happen. That becomes the report.
