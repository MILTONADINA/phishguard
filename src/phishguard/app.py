"""FastAPI app + single-page UI. Spec: specs/07-api.md (API-01 … API-06)."""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, StrictStr, field_validator

from phishguard.config import MAX_EMAIL_CHARS
from phishguard.detector import PhishDetector

logger = logging.getLogger(__name__)
INDEX_HTML = Path(__file__).with_name("index.html")


class AnalyzeRequest(BaseModel):
    """API-03: the request body. Anything else (missing, not a string, blank) gets a 422 from FastAPI."""

    email_text: StrictStr  # strict: 123 is rejected instead of turned into "123"

    @field_validator("email_text")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("email_text must not be empty or only whitespace")
        return value


def create_app(detector=None, models_dir: str | Path | None = None) -> FastAPI:
    """Build the app.

    detector given -> use it. Otherwise PhishDetector.from_dir(models_dir or $PHISHGUARD_MODELS_DIR
    or "models"); on failure start anyway with models_loaded=False.

    Routes: GET /health, POST /api/analyze, GET / (HTML page).
    """
    if detector is None:
        models_dir = models_dir or os.environ.get("PHISHGUARD_MODELS_DIR", "models")
        try:
            detector = PhishDetector.from_dir(models_dir)
        except Exception as err:  # API-05: start anyway; /health reports it and /api/analyze answers 503
            logger.error("Models not loaded from %s: %s", models_dir, err)

    app = FastAPI(title="PhishGuard")
    page = INDEX_HTML.read_text(encoding="utf-8")
    model_lock = threading.Lock()  # FastAPI runs requests in threads; let one at a time use the model

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "models_loaded": detector is not None}

    @app.post("/api/analyze")
    def analyze(body: AnalyzeRequest) -> dict:
        if detector is None:
            raise HTTPException(503, "The models aren't loaded. Train them, then restart the server.")
        if len(body.email_text) > MAX_EMAIL_CHARS:
            raise HTTPException(413, f"The email is too long: the limit is {MAX_EMAIL_CHARS:,} characters.")
        with model_lock:
            return detector.analyze(body.email_text).to_dict()

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return page

    return app


# Module-level app for `uvicorn phishguard.app:app`. `python -m phishguard serve` sets
# PHISHGUARD_MODELS_DIR before importing this module, so the models are loaded once.
app = create_app()
