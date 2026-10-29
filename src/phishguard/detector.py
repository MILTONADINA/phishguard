"""End-to-end detector. Spec: specs/08-detector-and-cli.md (DET-01 … DET-05)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from phishguard.fusion import FusionParams


@dataclass
class UrlResult:
    url: str
    score: float
    flagged: bool


@dataclass
class AnalysisResult:
    score: float
    verdict: str
    threshold: float
    text_score: float
    url_score: float | None
    urls: list[UrlResult] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """JSON-serialisable dict (Python floats; urls as list of dicts). Shape: specs/07-api.md API-02."""
        raise NotImplementedError("DET-01: see specs/08-detector-and-cli.md")


class PhishDetector:
    """DET-01..05. text_model / url_model: anything with predict_proba(list[str]) -> np.ndarray."""

    def __init__(self, text_model, url_model=None, fusion: FusionParams | None = None) -> None:
        raise NotImplementedError("DET-01: see specs/08-detector-and-cli.md")

    def analyze(self, email_text: str) -> AnalysisResult:
        """DET-01..04: extract URLs (raw text, cap MAX_URLS) -> text score -> URL scores (one call)
        -> fuse -> verdict -> reasons. Empty/whitespace input -> ValueError."""
        raise NotImplementedError("DET-01: see specs/08-detector-and-cli.md")

    @classmethod
    def from_dir(cls, models_dir: str | Path) -> "PhishDetector":
        """DET-05: load text/ (required), url/ (optional), fusion.json (optional, defaults per spec)."""
        raise NotImplementedError("DET-05: see specs/08-detector-and-cli.md")
