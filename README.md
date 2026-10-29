# PhishGuard

A two-stage phishing email detector served through a small web app. Paste an email and get a phishing score, the links it contains (each scored and flagged), and short reasons.

- **Stage 1, text:** DistilBERT fine-tuned on labelled phishing and legitimate emails (PyTorch + Hugging Face Transformers)
- **Stage 2, links:** a character-level CNN that scores each URL (PyTorch)
- **Fusion:** a weighted combination of both scores, with the weight and threshold fitted on the validation set
- **App:** FastAPI backend with a single-page UI

Built spec-first: requirements live in [`specs/`](specs/), every requirement is checked by a test in [`tests/`](tests/), and the build order is in [`TASKS.md`](TASKS.md).

## Setup (macOS, Apple silicon)

```bash
python3.12 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -c "import torch; print(torch.backends.mps.is_available())"   # should print True
.venv/bin/python -m pytest                                                      # unit tests (offline, ~10 s)
```

Working in Claude Code? Read `CLAUDE.md`. Use `/next-task` to build one task at a time and `/status` to see progress.

## Data

Already downloaded into `data/raw/` (gitignored). To re-download on another machine:

```bash
mkdir -p data/raw/emails && cd data/raw
curl -L -o Phishing_validation_emails.csv "https://zenodo.org/records/13474746/files/Phishing_validation_emails.csv?download=1"
curl -L -o phiusiil.zip "https://archive.ics.uci.edu/static/public/967/phiusiil+phishing+url+dataset.zip" && unzip phiusiil.zip
for f in CEAS_08 Enron Ling Nazario Nigerian_Fraud SpamAssasin; do
  curl -L -o "emails/$f.csv" "https://zenodo.org/records/8339691/files/$f.csv?download=1"
done
```

| Data | Source | License |
|---|---|---|
| `emails/*.csv` (82,486 emails) | Champa et al., *Phishing Email Curated Datasets*, Zenodo 8339691 (same sources as the Kaggle "Phishing Email Dataset") | CC BY 4.0 |
| `PhiUSIIL_Phishing_URL_Dataset.csv` (235,795 URLs) | UCI ML Repository #967 | CC BY 4.0 |
| `Phishing_validation_emails.csv` (2,000 emails, held-out only) | Zenodo 13474746 | CC BY 4.0 |

Note: **PhiUSIIL uses 1 = legitimate**; the loader inverts it. The verified counts and caveats are in `specs/01-data.md`.

## Pipeline

```bash
python -m phishguard prepare-data                         # data/raw -> data/processed
python -m phishguard train-baseline
python -m phishguard train-text --max-train-samples 20000 # quick first run; drop the flag for the full run
python -m phishguard train-url
python -m phishguard fit-fusion                           # on the validation split only
python -m phishguard evaluate                             # writes reports/metrics.md
python -m phishguard serve                                # http://127.0.0.1:8000
```

## Tests

```bash
pytest                                          # all unit tests (default: excludes acceptance)
pytest -m "not slow"                            # skip the tiny-training tests
pytest tests/unit/test_fusion.py                # one area
PHISHGUARD_ACCEPTANCE=1 pytest -m acceptance    # real-data targets (after training)
```

`tests/unit/test_traceability.py` fails if any requirement in `specs/` has no test.

## Layout

```
specs/          requirements (source of truth)
src/phishguard/ data, baseline, text_model, url_model, fusion, detector, evaluate, app, cli
tests/unit/     offline tests, one file per spec
tests/acceptance/ real-data performance targets (opt-in)
demo_samples/   emails for the live demo
TASKS.md        build order and calendar
NOTES.md        running log of problems and decisions (feeds the report)
REPORT.md       final report
```
