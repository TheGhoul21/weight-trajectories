"""Thin wrapper exposing shared reproducibility helpers to scripts."""

from __future__ import annotations

from src.utils.repro import child_seed, seed_everything  # re-export

__all__ = ["seed_everything", "child_seed"]

