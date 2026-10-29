# 08 — Detector, CLI and utilities

Modules: `src/phishguard/detector.py`, `src/phishguard/cli.py` (+ `__main__.py`), `src/phishguard/utils.py`

## Detector

```python
@dataclass
class UrlResult:
    url: str
    score: float
    flagged: bool

@dataclass
class AnalysisResult:
    score: float
    verdict: str              # "phishing" | "legitimate"
    threshold: float
    text_score: float
    url_score: float | None   # aggregated (max), None if no URLs or no URL model
    urls: list[UrlResult]
    reasons: list[str]
    def to_dict(self) -> dict  # JSON-serialisable, Python floats, urls as dicts

class PhishDetector:
    def __init__(self, text_model, url_model=None, fusion=FusionParams()): ...
    def analyze(self, email_text: str) -> AnalysisResult
    @classmethod
    def from_dir(cls, models_dir) -> "PhishDetector"
```

`text_model` and `url_model` are anything with `predict_proba(list[str]) -> np.ndarray` (duck typing, so tests use fakes).

### DET-01 `analyze` pipeline
1. `urls = extract_urls(email_text)` on the **raw** text, keeping the first `MAX_URLS`.
2. `text_score = text_model.predict_proba([clean_text(email_text)])[0]`
3. If there are URLs and a URL model: `url_scores = url_model.predict_proba(urls)`, called **once** with the whole list.
4. `url_score = aggregate_url_scores(url_scores)`, `score = fuse(text_score, url_score, fusion.weight)`
5. `verdict(score, fusion.threshold)`. Each `UrlResult.flagged = url score >= fusion.threshold`.
6. `reasons = build_reasons(text_score, [(u, s) …], fusion.threshold)`

### DET-02 Text-only mode
With `url_model=None`: URLs are still extracted and listed (each `score` is `0.0`, `flagged` is `False`), `url_score` is `None`, `score == text_score`, and the reasons still say how many links were found or that there were none. This is the **Must-tier** fallback.

### DET-03 Bad input
`analyze` with an empty or whitespace-only string → `ValueError`.

### DET-04 URL cap
An email with more than `MAX_URLS` distinct URLs scores only the first `MAX_URLS`. `len(result.urls) <= MAX_URLS`.

### DET-05 `from_dir(models_dir)`
Expected layout:
```
models_dir/
  text/          # PhishTextClassifier.save output (required)
  url/           # URLClassifier.save output (optional)
  fusion.json    # FusionParams.save output (optional)
```
- `text/` missing → `FileNotFoundError`.
- `url/` missing → text-only mode.
- `fusion.json` missing → `FusionParams(weight=1.0, threshold=0.5)` without a URL model, `FusionParams(weight=0.5, threshold=0.5)` with one.

## CLI

### CLI-01 `build_parser() -> argparse.ArgumentParser`
`python -m phishguard <command>`:

| Command | Arguments (defaults) | Does |
|---|---|---|
| `prepare-data` | `--raw-dir data/raw` `--out-dir data/processed` `--seed 42` | `data.prepare_data` |
| `train-baseline` | `--data-dir data/processed` `--out models/baseline.joblib` | fit on train, print val metrics |
| `train-text` | `--data-dir` `--out-dir models/text` `--epochs 2` `--batch-size 16` `--lr 2e-5` `--max-length 256` `--max-train-samples` (none) `--device` (auto) | `train_text_model` |
| `train-url` | `--data-dir` `--out-dir models/url` `--epochs 5` `--batch-size 256` `--lr 1e-3` `--max-train-samples` `--device` | `train_url_model` |
| `fit-fusion` | `--data-dir` `--models-dir models` | score the val split, `fit_fusion`, save `models/fusion.json` |
| `evaluate` | `--data-dir` `--models-dir` `--reports-dir reports` | `evaluate_all` |
| `serve` | `--host 127.0.0.1` `--port 8000` `--models-dir models` | run uvicorn |

The test only checks parsing (every command exists, defaults and types are right). The handlers are verified by actually running them.

## Utilities

### UTIL-01 `set_seed(seed)`
Seeds `random`, `numpy` and `torch` (and CUDA if available). Calling `set_seed(n)` twice gives the same `torch.rand` and `np.random.rand` sequences.
