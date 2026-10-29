"""Combining stage scores, verdict and reasons. Spec: specs/05-fusion.md."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass
class FusionParams:
    """FUSE-04: weight for the text score (1 - weight for the URL score) and decision threshold."""

    weight: float = 0.5
    threshold: float = 0.5

    def save(self, path: str | Path) -> None:
        """Write JSON {"weight": ..., "threshold": ...}, creating parent folders."""
        raise NotImplementedError("FUSE-04: see specs/05-fusion.md")

    @classmethod
    def load(cls, path: str | Path) -> "FusionParams":
        raise NotImplementedError("FUSE-04: see specs/05-fusion.md")


def aggregate_url_scores(scores: Sequence[float]) -> float | None:
    """FUSE-01: max of scores, or None when empty."""
    raise NotImplementedError("FUSE-01: see specs/05-fusion.md")


def fuse(text_score: float, url_score: float | None, weight: float) -> float:
    """FUSE-02: text_score if url_score is None, else weight*text + (1-weight)*url.

    weight outside [0, 1] -> ValueError. Result clipped to [0, 1], returned as a Python float.
    """
    raise NotImplementedError("FUSE-02: see specs/05-fusion.md")


def fit_fusion(
    text_scores: Sequence[float],
    url_scores: Sequence[float | None],
    labels: Sequence[int],
    weights: Sequence[float] | None = None,
    thresholds: Sequence[float] | None = None,
) -> FusionParams:
    """FUSE-03: grid search maximising class-1 F1. Tie-break: larger weight, then threshold closest to 0.5."""
    raise NotImplementedError("FUSE-03: see specs/05-fusion.md")


def verdict(score: float, threshold: float = 0.5) -> str:
    """FUSE-05: "phishing" if score >= threshold, else "legitimate"."""
    raise NotImplementedError("FUSE-05: see specs/05-fusion.md")


def build_reasons(text_score: float, url_results: Sequence[tuple[str, float]], threshold: float) -> list[str]:
    """FUSE-06: short human-readable reasons (always non-empty). See the spec for the sentence templates."""
    raise NotImplementedError("FUSE-06: see specs/05-fusion.md")
