# 05 — Fusion, verdict and reasons

Module: `src/phishguard/fusion.py`

## Requirements

### FUSE-01 `aggregate_url_scores(scores) -> float | None`
The maximum of `scores` (one bad link is enough), or `None` when `scores` is empty.

### FUSE-02 `fuse(text_score, url_score, weight) -> float`
- `url_score is None` → returns `text_score` (whatever the weight).
- Otherwise → `weight * text_score + (1 - weight) * url_score`.
- `weight` outside `[0, 1]` → `ValueError`. The result is clipped to `[0, 1]` and returned as a Python `float`.

### FUSE-03 `fit_fusion(text_scores, url_scores, labels, weights=None, thresholds=None) -> FusionParams`
- `url_scores[i]` may be `None` (an email with no links).
- Defaults: `weights = 0.0, 0.1, …, 1.0` (11 values); `thresholds = 0.10, 0.15, …, 0.90` (17 values).
- Grid search: for each `(weight, threshold)`, fuse every sample, predict `score >= threshold`, and compute F1 for class 1.
- Pick the best F1. On a tie, prefer the **larger weight** (trust the text model), then the threshold **closest to 0.5**.
- Fit on the **validation** split only, never on test.

### FUSE-04 `FusionParams(weight: float = 0.5, threshold: float = 0.5)`
A dataclass with `.save(path)` (writes JSON `{"weight": …, "threshold": …}`, creating parent folders) and `FusionParams.load(path)`. A round trip gives equal values.

### FUSE-05 `verdict(score, threshold=0.5) -> str`
`"phishing"` if `score >= threshold`, else `"legitimate"`.

### FUSE-06 `build_reasons(text_score, url_results, threshold) -> list[str]`
`url_results` is a list of `(url, score)` pairs. Returns 1–3+ short sentences, always non-empty:
1. Text sentence:
   - `text_score >= threshold` → `"The wording looks like phishing (text model: 0.93)."`
   - otherwise → `"The wording looks normal (text model: 0.12)."`
2. For each URL with `score >= threshold`: `"Suspicious link: <url> (0.97)."`
3. If there are URLs but none flagged: `"No suspicious links found (N checked)."`
4. If there are no URLs: `"No links found in the email."`

Scores are formatted with 2 decimals. Tests check the key phrases (`looks like phishing` / `looks normal`, the URL text, `No suspicious links`, `No links`), not every character.
