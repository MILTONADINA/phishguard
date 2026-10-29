# 09 — Acceptance criteria (real data, trained models)

Run after training:
```
PHISHGUARD_ACCEPTANCE=1 pytest -m acceptance
```
Paths come from env `PHISHGUARD_DATA_DIR` (default `data/processed`) and `PHISHGUARD_MODELS_DIR` (default `models`). The tests run `evaluate_all` once and check the numbers below. If something is missing, the test is **skipped** with a message, not failed.

These are the proposal's promises, written as numbers. If one fails, don't loosen it quietly: write down the real number and the reason in the report. A missed target with an honest analysis is still good report material.

| ID | Criterion | Target | Tier |
|---|---|---|---|
| **AC-01** | Text model F1 on the in-distribution test split | ≥ 0.95 | Must |
| **AC-02** | Text model F1 ≥ baseline F1 on the same test split (shows fine-tuning gained something) | ≥ baseline | Must |
| **AC-03** | URL model ROC-AUC on the PhiUSIIL test split | ≥ 0.95 | Should |
| **AC-04** | Full detector F1 on the held-out Zenodo set (different source, never trained on) | ≥ 0.75 | Should |
| **AC-05** | Median `analyze` latency on the demo laptop (20 emails, after warm-up) | < 1000 ms | Must |
| **AC-06** | Demo samples: the app classifies `demo_samples/obvious_phish.txt` as phishing and `demo_samples/legit.txt` as legitimate via `POST /api/analyze` | exact | Must |

`demo_samples/subtle_phish.txt` is for the live demo. It is not asserted, because it's meant to be hard; talk about whatever the model does with it.

## Submission checklist (from the assignment)
- [ ] Repo URL submitted; if private, `dgriff24` added as a collaborator
- [ ] Commit history shows daily progress (task-by-task commits)
- [ ] `REPORT.md` complete (all five sections)
- [ ] Screen recording of the demo
- [ ] Demo ≤ 5 minutes, rehearsed
