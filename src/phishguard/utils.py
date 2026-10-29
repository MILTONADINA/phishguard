"""Utilities. Spec: specs/08-detector-and-cli.md (UTIL-01), specs/03-text-model.md (TEXT-01)."""
from __future__ import annotations

import random
import warnings

import numpy as np
import torch
from sklearn.model_selection import train_test_split

# Order tried when no device is requested: NVIDIA GPU, then Apple GPU, then CPU.
_AUTO_ORDER = ("cuda", "mps", "cpu")


def set_seed(seed: int) -> None:
    """UTIL-01: seed `random`, `numpy` and `torch` (and CUDA if available)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # seeds the CPU generator and every GPU device (CUDA and MPS)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _is_available(name: str) -> bool:
    """True if this machine can run on the named device."""
    if name == "cpu":
        return True
    if name == "cuda":
        return torch.cuda.is_available()
    if name == "mps":
        return torch.backends.mps.is_available()
    return False


def select_device(prefer: str | None = None) -> torch.device:
    """TEXT-01: return the requested device if available, otherwise fall back.

    prefer in {"cpu", "cuda", "mps", None}. None -> cuda, then mps, then cpu.
    Never raises because a device is unavailable (warn and fall back instead).
    """
    if prefer is not None:
        if _is_available(prefer):
            return torch.device(prefer)
        warnings.warn(f"Device {prefer!r} is not available on this machine; choosing one automatically.")
    for name in _AUTO_ORDER:
        if _is_available(name):
            return torch.device(name)
    return torch.device("cpu")


def stratified_subset(texts: list[str], labels: list[int], n: int | None, seed: int) -> tuple[list[str], list[int]]:
    """A random subset of n examples with the same phishing rate as the full set (for quick runs)."""
    if n is None or n >= len(texts):
        return texts, labels
    stratify = labels if min(labels.count(0), labels.count(1)) >= 2 else None
    keep, _ = train_test_split(np.arange(len(texts)), train_size=n, random_state=seed, stratify=stratify)
    return [texts[i] for i in keep], [labels[i] for i in keep]
