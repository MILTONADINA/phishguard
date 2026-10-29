# PhishGuard evaluation

Written by `python -m phishguard evaluate`. Positive class = phishing. Test split: 8,189 emails; held-out set (Zenodo 13474746, a different source): 2,000 emails.
Detector: text + URL, fusion weight 1.0 on the text score, threshold 0.1 (fitted on the validation split). Every other row uses threshold 0.5.

| Model | Accuracy | Precision | Recall | F1 | FPR | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.989 | 0.989 | 0.989 | 0.989 | 0.012 | 0.999 |
| text_model | 0.994 | 0.995 | 0.994 | 0.994 | 0.005 | 1.000 |
| url_model | 0.998 | 1.000 | 0.995 | 0.998 | 0.000 | 0.999 |
| detector | 0.995 | 0.995 | 0.995 | 0.995 | 0.006 | 1.000 |
| baseline_heldout | 0.855 | 0.775 | 1.000 | 0.873 | 0.290 | 0.912 |
| text_model_heldout | 0.948 | 0.988 | 0.906 | 0.945 | 0.011 | 0.986 |
| detector_heldout | 0.944 | 0.981 | 0.906 | 0.942 | 0.018 | 0.986 |

Latency of `PhishDetector.analyze` over 20 test emails after a warm-up: median 4.6 ms, 95th percentile 6.2 ms.
