"""Utilities. Spec: specs/08-detector-and-cli.md (UTIL-01), specs/03-text-model.md (TEXT-01)."""
from __future__ import annotations

import torch


def set_seed(seed: int) -> None:
    """UTIL-01: seed `random`, `numpy` and `torch` (and CUDA if available)."""
    raise NotImplementedError("UTIL-01: see specs/08-detector-and-cli.md")


def select_device(prefer: str | None = None) -> torch.device:
    """TEXT-01: return the requested device if available, otherwise fall back.

    prefer in {"cpu", "cuda", "mps", None}. None -> cuda, then mps, then cpu.
    Never raises because a device is unavailable (warn and fall back instead).
    """
    raise NotImplementedError("TEXT-01: see specs/03-text-model.md")
