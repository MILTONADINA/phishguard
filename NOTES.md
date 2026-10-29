# NOTES — running log

Log anything you'd want in the report while it's fresh: problems, fixes, decisions and why, numbers.

| Date | Task | What happened / decided | Why / numbers |
|---|---|---|---|
| 2026-10-28 | T0 | Repo set up from specs + tests | spec-driven: tests written before code, validated against a reference implementation (all 102 unit tests pass on it; 5 injected bugs all caught) |
| 2026-10-28 | data | Downloaded all data into data/raw (Zenodo 8339691 per-source emails instead of Kaggle: same sources, no login) | 82,486 emails (42,891 / 39,595); PhiUSIIL 235,795 URLs, label 1 = legitimate, so invert; held-out 2,000 |
| 2026-10-28 | data | Caveats found | label 1 includes spam in Enron/Ling/CEAS/SpamAssassin; Nazario and Nigerian_Fraud are 100% label 1 (source shortcut risk); ~71% of emails contain a URL |
| 2026-10-28 | T0 | Mac only had Python 3.14, so installed 3.12 with Homebrew (3.12.16) and built `.venv` on it; `pip install -e ".[dev]"` | Matches the 3.11–3.13 range in the specs. Got torch 2.15.0 (MPS available), transformers 5.21.0, scikit-learn 1.9.1, pandas 3.1.0 — transformers and pandas are a major version newer than the docs assumed; watch for API changes. Starting state: 3 passed, 88 failed + 11 errors, all `NotImplementedError` |
| 2026-10-28 | T1 | `set_seed` seeds random, numpy and torch; `select_device` tries cuda → mps → cpu and warns instead of raising when the requested device is missing | On this Mac auto picks `mps`; asking for `cuda` warns and falls back to `mps`. `torch.manual_seed` already seeds every device; CUDA is seeded explicitly because the spec names it |
| | | | |
