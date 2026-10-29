"""End-to-end detector. Spec: specs/08-detector-and-cli.md (DET-01 … DET-05)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from phishguard.config import MAX_URLS
from phishguard.data import clean_text, extract_urls
from phishguard.fusion import FusionParams, aggregate_url_scores, build_reasons, fuse, verdict


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
        return {
            "score": float(self.score),
            "verdict": self.verdict,
            "threshold": float(self.threshold),
            "text_score": float(self.text_score),
            "url_score": None if self.url_score is None else float(self.url_score),
            "urls": [{"url": u.url, "score": float(u.score), "flagged": bool(u.flagged)} for u in self.urls],
            "reasons": list(self.reasons),
        }


class PhishDetector:
    """DET-01..05. text_model / url_model: anything with predict_proba(list[str]) -> np.ndarray."""

    def __init__(self, text_model, url_model=None, fusion: FusionParams | None = None) -> None:
        self.text_model = text_model
        self.url_model = url_model  # None = text-only mode (DET-02)
        self.fusion = fusion if fusion is not None else FusionParams()

    def analyze(self, email_text: str) -> AnalysisResult:
        """DET-01..04: extract URLs (raw text, cap MAX_URLS) -> text score -> URL scores (one call)
        -> fuse -> verdict -> reasons. Empty/whitespace input -> ValueError."""
        if not isinstance(email_text, str) or not email_text.strip():
            raise ValueError("email_text must be a non-empty string")
        threshold = self.fusion.threshold

        # Links come from the raw text: cleaning removes the HTML tags that hold href="..." links.
        urls = extract_urls(email_text)[:MAX_URLS]
        text_score = float(self.text_model.predict_proba([clean_text(email_text)])[0])

        if self.url_model is not None and urls:
            url_scores = [float(s) for s in self.url_model.predict_proba(urls)]  # one batched call
            url_score = aggregate_url_scores(url_scores)
            url_results = [UrlResult(u, s, s >= threshold) for u, s in zip(urls, url_scores)]
        else:
            url_score = None  # no links, or no URL model to score them
            url_results = [UrlResult(u, 0.0, False) for u in urls]

        score = fuse(text_score, url_score, self.fusion.weight)

        if self.url_model is None and urls:
            # build_reasons would call these links "checked"; without a URL model they were only found.
            text_reason = build_reasons(text_score, [], threshold)[0]
            reasons = [text_reason, f"Found {len(urls)} link(s), not checked: no URL model is loaded."]
        else:
            reasons = build_reasons(text_score, [(r.url, r.score) for r in url_results], threshold)

        return AnalysisResult(
            score=score,
            verdict=verdict(score, threshold),
            threshold=threshold,
            text_score=text_score,
            url_score=url_score,
            urls=url_results,
            reasons=reasons,
        )

    @classmethod
    def from_dir(cls, models_dir: str | Path) -> "PhishDetector":
        """DET-05: load text/ (required), url/ (optional), fusion.json (optional, defaults per spec)."""
        # Imported here, not at the top: only loading saved models needs PyTorch, and it keeps
        # the detector usable with any model object (the tests pass in fakes).
        from phishguard.text_model import PhishTextClassifier
        from phishguard.url_model import URLClassifier

        models_dir = Path(models_dir)
        text_dir, url_dir, fusion_path = models_dir / "text", models_dir / "url", models_dir / "fusion.json"
        if not text_dir.is_dir():
            raise FileNotFoundError(f"No text model in {text_dir}: run `python -m phishguard train-text` first")

        text_model = PhishTextClassifier.load(text_dir)
        url_model = URLClassifier.load(url_dir) if url_dir.is_dir() else None
        if fusion_path.exists():
            fusion = FusionParams.load(fusion_path)
        else:
            # Not fitted yet: text-only trusts the text model fully; with URLs, weigh both equally.
            fusion = FusionParams(weight=1.0 if url_model is None else 0.5, threshold=0.5)
        return cls(text_model, url_model, fusion)
