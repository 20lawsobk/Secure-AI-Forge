"""Hardware/thread configuration — size BLAS thread pools to the *actual* host.

Why this exists: the dev box has 2 vCPUs; the production Reserved VM has 16. A
hardcoded thread count (e.g. ``16``) is wrong in one of those places — starving
prod or oversubscribing dev. So every value here derives from ``os.cpu_count()``
at runtime.

Honest notes about what this does and does not buy you:
  * For a **single-process** server, OpenBLAS already defaults to all cores, so
    setting the env explicitly is mostly reproducibility — it equals the default.
  * It becomes load-bearing only when you run **multiple worker processes**: N
    processes each spawning all-core BLAS pools oversubscribes the CPU
    (``N x cpus`` threads on ``cpus`` cores), which is net *slower*. Then cap each
    worker to ``cpus // N``. The current ``PDIMWorker`` is thread-based (one
    process, one shared BLAS pool), so there is nothing to cap yet.
  * BLAS reads these env vars when numpy is first imported, so call
    ``configure_blas_threads`` *before* importing numpy in each process.

This module imports only the stdlib (no numpy) so it is safe to call first.
"""
from __future__ import annotations

import os

_BLAS_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def cpu_count() -> int:
    """Logical CPUs available to this process (>= 1)."""
    return os.cpu_count() or 1


def plan_blas_threads(cpus: int, num_workers: int = 1, reserve: int = 0) -> int:
    """Threads each worker should use so the total stays within ``cpus``.

    ``reserve`` keeps cores free for non-BLAS work (async I/O loops, the web
    server). Always returns >= 1.
    """
    if cpus < 1:
        raise ValueError(f"cpus must be >= 1, got {cpus}")
    if num_workers < 1:
        raise ValueError(f"num_workers must be >= 1, got {num_workers}")
    usable = max(1, cpus - max(0, reserve))
    return max(1, usable // num_workers)


def configure_blas_threads(
    num_workers: int = 1, reserve: int = 0, override: bool = False
) -> dict[str, str]:
    """Set BLAS/OpenMP thread env vars to ``plan_blas_threads`` for this process.

    Respects any pre-existing values (operator override wins) unless ``override``
    is True. Returns the mapping now in effect. Must run before numpy import to
    affect the current process.
    """
    val = str(plan_blas_threads(cpu_count(), num_workers, reserve))
    effective: dict[str, str] = {}
    for var in _BLAS_ENV_VARS:
        if override or var not in os.environ:
            os.environ[var] = val
        effective[var] = os.environ[var]
    return effective
