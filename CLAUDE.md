# PhishGuard — instructions for Claude Code

Two-stage phishing email detector for the AI course Machine Learning Project (due **before class Wed Nov 11**).
Stage 1: fine-tuned DistilBERT on email text. Stage 2: character-level CNN on URLs. Weighted fusion. FastAPI + one-page UI.

## This repo is spec-driven
- `specs/` is the source of truth: 50 numbered requirements (`DATA-03`, `API-02`, …). Read the relevant spec before writing code.
- `tests/` encodes every requirement (`@pytest.mark.req("ID")`). Tests were written first and validated against a reference implementation, so they are passable as written.
- `src/phishguard/` contains **stubs**: the signatures, docstrings and dataclasses are the contract. Replace the `raise NotImplementedError` bodies. Don't rename or change public signatures.
- `TASKS.md` is the build order (T0–T14), with a progress checklist at the top.

## Workflow for every task
1. Find the first unchecked task in `TASKS.md`. Read its spec section(s).
2. Run that task's tests and confirm they fail.
3. Implement in `src/phishguard/` only.
4. Run the task's tests until they pass, then run the full suite: `.venv/bin/python -m pytest -q` (no regressions).
5. Add a line to `NOTES.md`: what you did, any problem and its fix, decisions and their reasons, numbers (row counts, metrics, training time).
6. Tick the task in the `TASKS.md` checklist.
7. Commit with the message from `TASKS.md` (prefix `T<n>:`). **One commit per task.** The commit history is graded.
8. Stop and report: tests passing, what changed, anything Milton should know. Continue to the next task only when asked.

## Hard rules
- **Never edit `tests/` or `specs/` to make something pass.** If a test or spec looks wrong, stop, explain why, and propose the change. Milton decides. (The project settings ask before any edit there.)
- **Never train, tune or pick thresholds on `test.csv`, `url_test.csv` or `heldout.csv`.** Fit on train; select on val.
- Never commit `data/`, `models/`, `logs/` or `.venv/` (they're in `.gitignore`).
- Don't modify `data/raw/`: it holds the downloaded datasets.
- Keep code readable: Milton must be able to explain every part in a 5-minute demo and in the report.

## Environment
- Python venv at `.venv`, created in T0 with `python3.12 -m venv .venv`. This Mac's default Python is 3.14, so install 3.12 first with `brew install python@3.12` (or `uv python install 3.12` if Homebrew isn't available). Only fall back to 3.14 if 3.12 can't be installed, and check that `torch` and `transformers` install cleanly before continuing.
- Always call tools through the venv: `.venv/bin/python -m pytest`, `.venv/bin/python -m phishguard <cmd>`, `.venv/bin/pip install …`. Shell activation doesn't persist between commands.
- Device: Apple silicon, so use `mps` for training (`select_device()` picks it automatically).
- DistilBERT (`distilbert-base-uncased`, ~260 MB) downloads from Hugging Face on the first training run.

## Data (already downloaded into `data/raw/`, gitignored)
| Path | What |
|---|---|
| `data/raw/emails/*.csv` | CEAS_08, Enron, Ling, Nazario, Nigerian_Fraud, SpamAssasin. Per-source email corpora from Zenodo 8339691 (Champa et al., CC BY 4.0): the same sources as the Kaggle "Phishing Email Dataset". Columns include `subject`, `body`, `label` (1 = phishing). |
| `data/raw/PhiUSIIL_Phishing_URL_Dataset.csv` | 235,795 URLs. Columns `URL`, `label`, **label 1 = legitimate** (134,850) and 0 = phishing (100,945), so the loader inverts it. The first column header has a UTF-8 BOM (irrelevant: we only use `URL` and `label`). |
| `data/raw/Phishing_validation_emails.csv` | Zenodo 13474746, 2,000 emails, `Email Text` / `Email Type`. **Held-out test only.** |

`prepare_data` uses `data/raw/phishing_email.csv` if present, otherwise `data/raw/emails/`. Don't put a combined file into `data/raw/emails/` (DATA-03 leakage guard).

## Long-running commands
- Training can take hours. Run it in the background with a log, for example:
  `nohup .venv/bin/python -m phishguard train-text --max-train-samples 20000 > logs/train_text_quick.log 2>&1 &`
  then check with `tail -n 20 logs/train_text_quick.log`. Create `logs/` first.
- Always do a quick run (`--max-train-samples 20000`) before the full run. Report time per epoch. If an epoch takes more than about 2 h on MPS, propose `--max-length 128` or Colab.
- Print per-epoch history from the training handlers (it goes into the report).

## Acceptance (after T13)
`PHISHGUARD_ACCEPTANCE=1 .venv/bin/python -m pytest -m acceptance`. If a target is missed, don't loosen it: record the real number and the likely cause in `NOTES.md` for the report.

## Report and honesty
`REPORT.md` has the five sections the rubric grades. Fill it from `NOTES.md` and `reports/metrics.md`. The specs, tests and much of the code were produced with AI assistance (Claude), so keep the "Tools used" section accurate.
