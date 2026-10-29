# 06 — Evaluation

Module: `src/phishguard/evaluate.py`

## Requirements

### EVAL-01 `compute_metrics(y_true, y_prob, threshold=0.5) -> dict`
Predictions are `y_prob >= threshold`. Returns these keys with Python numbers:

| Key | Definition |
|---|---|
| `n` | number of samples |
| `threshold` | the threshold used |
| `tp`, `fp`, `tn`, `fn` | confusion-matrix counts (positive = phishing) |
| `accuracy` | (tp + tn) / n |
| `precision` | tp / (tp + fp), **0.0** when tp + fp = 0 |
| `recall` | tp / (tp + fn), **0.0** when tp + fn = 0 |
| `f1` | 2PR / (P + R), **0.0** when P + R = 0 |
| `fpr` | fp / (fp + tn): normal email wrongly flagged, **0.0** when fp + tn = 0 |
| `fnr` | fn / (fn + tp): phishing missed, **0.0** when fn + tp = 0 |
| `roc_auc` | area under the ROC curve, **`None`** when only one class is present |

`y_true` and `y_prob` must be the same length, else `ValueError`.

### EVAL-02 Edge cases
A single class in `y_true` does not raise: `roc_auc` is `None` and the rest is computed with the zero rules above.

### EVAL-03 `compare_models(results: dict[str, dict]) -> str`
Returns a Markdown table:
- header `| Model | Accuracy | Precision | Recall | F1 | FPR | ROC-AUC |`
- an alignment row
- one row per model, in insertion order, with values to 3 decimals; `roc_auc=None` shows as `n/a`.

### EVAL-04 `evaluate_all(data_dir, models_dir, reports_dir="reports") -> dict`
Real-data evaluation, run by `python -m phishguard evaluate` and the acceptance tests.
- **Loads:**
  - `data_dir/test.csv`, `data_dir/url_test.csv`, optional `data_dir/heldout.csv`
  - `models_dir/baseline.joblib`, `models_dir/text/`, optional `models_dir/url/`, optional `models_dir/fusion.json`
- **Computes `compute_metrics` for:**
  - `"baseline"` on test
  - `"text_model"` on test
  - `"url_model"` on url_test (if a URL model exists)
  - `"detector"` on test (full `PhishDetector`)
  - `"detector_heldout"` on heldout (if the file exists)
- **Latency:** times `PhishDetector.analyze` on 20 test emails (after one warm-up call) and adds `"latency_ms": {"median": …, "p95": …}`.
- **Writes** `reports_dir/metrics.json` (the full dict) and `reports_dir/metrics.md` (the `compare_models` table plus the latency).
- **Returns** the dict.
