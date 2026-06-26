"""Configuration for the PDIM (Predictive Deduplicating Inference Manager)."""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field


def _default_base_dir() -> str:
    env = os.getenv("PDIM_RESULTS_DIR")
    if env:
        return env
    return os.path.join(tempfile.gettempdir(), "maxcore_pdim_results")


@dataclass
class PDIMConfig:
    namespace: str = "maxcore"
    batch_size: int = 16
    ttl_seconds: int = 3600
    base_dir: str = field(default_factory=_default_base_dir)
    idle_sleep: float = 0.2
    inflight_wait_seconds: float = 180.0
