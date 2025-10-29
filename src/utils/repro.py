"""Shared reproducibility helpers."""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore


def seed_everything(seed: int, deterministic_torch: bool = True) -> None:
    """Seed random, numpy, and torch (if available)."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))

    random.seed(seed)
    np.random.seed(seed)

    if torch is None:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        try:
            torch.use_deterministic_algorithms(True)  # type: ignore[attr-defined]
        except (AttributeError, RuntimeError):  # pragma: no cover
            pass
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]


def child_seed(base_seed: Optional[int], offset: int) -> Optional[int]:
    """Return a deterministic seed for child workers."""
    if base_seed is None:
        return None
    return int(base_seed) + int(offset)

