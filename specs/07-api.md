# 07 — Web app and API

Module: `src/phishguard/app.py`. Served with `uvicorn phishguard.app:app` or `python -m phishguard serve`.

## Factory

`create_app(detector=None, models_dir=None) -> FastAPI`
- If `detector` is given, use it (tests inject a fake).
- Otherwise load with `PhishDetector.from_dir(models_dir or env PHISHGUARD_MODELS_DIR or "models")`. If loading fails, the app still starts, with `models_loaded = False` (log the error).

`app = create_app()` exists at module level for uvicorn.

## Requirements

### API-01 `GET /health`
`200` with `{"status": "ok", "models_loaded": <bool>}`.

### API-02 `POST /api/analyze`
- **Request:** JSON `{"email_text": "<string>"}`.
- **Response:** `200` with
  ```json
  {
    "score": 0.97,
    "verdict": "phishing",
    "threshold": 0.5,
    "text_score": 0.95,
    "url_score": 0.99,
    "urls": [{"url": "http://paypa1-login.com/verify", "score": 0.99, "flagged": true}],
    "reasons": ["The wording looks like phishing (text model: 0.95).",
                "Suspicious link: http://paypa1-login.com/verify (0.99)."]
  }
  ```
  `url_score` is `null` when the email has no links. This body is `AnalysisResult.to_dict()`.

### API-03 Validation
A missing `email_text`, non-string, empty, or whitespace-only value → `422`.

### API-04 Size limit
`email_text` longer than `MAX_EMAIL_CHARS` (50 000) → `413` with a JSON `detail` message.

### API-05 Models not loaded
When no detector could be loaded, `POST /api/analyze` → `503` with a JSON `detail` saying the models aren't loaded. `/health` still returns `200` with `models_loaded: false`.

### API-06 `GET /`
`200`, `text/html`. The page has:
- a `<textarea id="email_text">`,
- an **Analyze** button,
- a result area,
- inline JavaScript that `fetch`es `/api/analyze`.

The result area shows the verdict, the score as a percentage, each link with a ✓/⚠ marker, and the reasons. No external CDN is needed, so the demo works offline.

## Demo notes (not tested)
- Colour the verdict: red for phishing, green for legitimate.
- Keep it to one page. The 5-minute demo is: paste → Analyze → explain the reasons.
