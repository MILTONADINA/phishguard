"""Combining stage scores, verdict and reasons. Spec: specs/05-fusion.md."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.metrics import f1_score

# Search grids for fit_fusion. Rounded so the chosen values are exactly 0.3, not 0.30000000000000004.
DEFAULT_WEIGHTS = [round(0.1 * i, 2) for i in range(11)]  # 0.0, 0.1, ..., 1.0
DEFAULT_THRESHOLDS = [round(0.10 + 0.05 * i, 2) for i in range(17)]  # 0.10, 0.15, ..., 0.90


@dataclass
class FusionParams:
    """FUSE-04: weight for the text score (1 - weight for the URL score) and decision threshold."""

    weight: float = 0.5
    threshold: float = 0.5

    def save(self, path: str | Path) -> None:
        """Write JSON {"weight": ..., "threshold": ...}, creating parent folders."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "FusionParams":
        values = json.loads(Path(path).read_text())
        return cls(weight=float(values["weight"]), threshold=float(values["threshold"]))


def aggregate_url_scores(scores: Sequence[float]) -> float | None:
    """FUSE-01: max of scores, or None when empty."""
    scores = list(scores)
    if not scores:
        return None
    return float(max(scores))  # one bad link is enough to make the email suspicious


def fuse(text_score: float, url_score: float | None, weight: float) -> float:
    """FUSE-02: text_score if url_score is None, else weight*text + (1-weight)*url.

    weight outside [0, 1] -> ValueError. Result clipped to [0, 1], returned as a Python float.
    """
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"weight must be between 0 and 1, got {weight}")
    if url_score is None:  # no links: the text model decides alone
        score = text_score
    else:
        score = weight * text_score + (1 - weight) * url_score
    return float(min(1.0, max(0.0, score)))


def fit_fusion(
    text_scores: Sequence[float],
    url_scores: Sequence[float | None],
    labels: Sequence[int],
    weights: Sequence[float] | None = None,
    thresholds: Sequence[float] | None = None,
) -> FusionParams:
    """FUSE-03: grid search maximising class-1 F1. Tie-break: larger weight, then threshold closest to 0.5."""
    if not len(text_scores) == len(url_scores) == len(labels):
        raise ValueError("text_scores, url_scores and labels must have the same length")
    weights = DEFAULT_WEIGHTS if weights is None else weights
    thresholds = DEFAULT_THRESHOLDS if thresholds is None else thresholds
    if not weights or not thresholds:
        raise ValueError("weights and thresholds must not be empty")
    labels = np.asarray(labels, dtype=int)

    best_key, best = None, None
    for weight in weights:
        # The fused scores depend only on the weight, so compute them once per weight.
        fused = np.array([fuse(t, u, weight) for t, u in zip(text_scores, url_scores)])
        for threshold in thresholds:
            f1 = f1_score(labels, (fused >= threshold).astype(int), pos_label=1, zero_division=0)
            # Compare (F1, weight, closeness to 0.5): higher wins at each step, so ties go to the
            # larger weight (trust the text model), then to the threshold nearest 0.5.
            key = (f1, weight, -abs(threshold - 0.5))
            if best_key is None or key > best_key:
                best_key, best = key, FusionParams(weight=float(weight), threshold=float(threshold))
    return best


def verdict(score: float, threshold: float = 0.5) -> str:
    """FUSE-05: "phishing" if score >= threshold, else "legitimate"."""
    return "phishing" if score >= threshold else "legitimate"


def build_reasons(text_score: float, url_results: Sequence[tuple[str, float]], threshold: float) -> list[str]:
    """FUSE-06: short human-readable reasons (always non-empty). See the spec for the sentence templates."""
    url_results = list(url_results)
    if text_score >= threshold:
        reasons = [f"The wording looks like phishing (text model: {text_score:.2f})."]
    else:
        reasons = [f"The wording looks normal (text model: {text_score:.2f})."]

    flagged = [(url, score) for url, score in url_results if score >= threshold]
    reasons += [f"Suspicious link: {url} ({score:.2f})." for url, score in flagged]
    if not url_results:
        reasons.append("No links found in the email.")
    elif not flagged:
        reasons.append(f"No suspicious links found ({len(url_results)} checked).")
    return reasons
