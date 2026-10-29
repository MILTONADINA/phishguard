"""FastAPI app + single-page UI. Spec: specs/07-api.md (API-01 … API-06)."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI


def create_app(detector=None, models_dir: str | Path | None = None) -> FastAPI:
    """Build the app.

    detector given -> use it. Otherwise PhishDetector.from_dir(models_dir or $PHISHGUARD_MODELS_DIR
    or "models"); on failure start anyway with models_loaded=False.

    Routes: GET /health, POST /api/analyze, GET / (HTML page).
    """
    raise NotImplementedError("API-01..06: see specs/07-api.md")


def _make_module_app() -> FastAPI | None:
    # Module-level app for `uvicorn phishguard.app:app`.
    # Stays None until create_app is implemented, so importing this module still works.
    try:
        return create_app()
    except NotImplementedError:
        return None


app = _make_module_app()
