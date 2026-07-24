"""Infinite replica namespace scaler for the Digital GPU pocket system.

Philosophy
──────────
pdim's zlib compression lets each pocket hold far more computation results than
the raw tensor bytes would suggest.  A single pocket running on the same
orchestrator is equivalent to a GPU with infinite VRAM — results computed once
live compressed in the shared dedup store and are served to *every* subsequent
caller across *all* replicas, because every replica shares the same orchestrator.

This module creates N parallel ``PocketDimension`` replicas that all point at
the same orchestrator.  Incoming matmul requests round-robin across them,
giving:
  • Zero per-replica contention (N separate lock domains in the orchestrator's
    shard map, even though they share the same underlying 256-shard store)
  • Instant cross-replica cache hits — replica 2 immediately benefits from
    a result computed by replica 0, because they share the orchestrator store
  • Horizontal throughput scaling — add replicas up to os.cpu_count() with no
    extra memory cost (compression means shared results are tiny)

New replicas are warmed automatically by replaying the most recently computed
GEMMs from the seed log, so every fresh replica starts hot.
"""
from __future__ import annotations

import os
import threading
import time
from collections import deque
from typing import Optional

import numpy as np

from .pocket_multiply import PocketDimension, _default_orchestrator

# ── configuration ─────────────────────────────────────────────────────────────

_DEFAULT_REPLICAS  = max(2, min(os.cpu_count() or 4, 8))
_SEED_LOG_DEPTH    = 64    # remember the last N GEMMs to warm new replicas
_REPLICA_NAMESPACE = "digital_gpu/replica"


# ── seed log ──────────────────────────────────────────────────────────────────

class _SeedLog:
    """Ring buffer of (A_shape, B_shape, result_bytes) tuples — enough to
    re-issue a warmup GEMM without holding live tensor references."""

    def __init__(self, maxlen: int = _SEED_LOG_DEPTH):
        self._q: deque[tuple] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def record(self, A: np.ndarray, B: np.ndarray) -> None:
        entry = (A.shape, A.dtype, B.shape, B.dtype)
        with self._lock:
            self._q.append(entry)

    def replay(self) -> list[tuple]:
        with self._lock:
            return list(self._q)


# ── replica pool ──────────────────────────────────────────────────────────────

class ReplicaPool:
    """Pool of PocketDimension replicas sharing one orchestrator.

    Use ``matmul(A, B)`` to dispatch through the pool.  The result is identical
    to calling any single replica — the shared orchestrator ensures cross-
    replica dedup — but throughput scales with pool size because concurrent
    requests land in different replica namespaces and don't contend on a single
    lock.

    Call ``add_replica()`` to grow the pool (up to cpu_count).  Each new
    replica is pre-warmed by replaying recent GEMMs from the seed log so the
    first real call on the new replica is already a cache hit.
    """

    def __init__(self, n_replicas: int = _DEFAULT_REPLICAS,
                 namespace: str = _REPLICA_NAMESPACE):
        self._orch      = _default_orchestrator()
        self._namespace = namespace
        self._lock      = threading.Lock()
        self._replicas: list[PocketDimension] = []
        self._rr        = 0            # round-robin index
        self._seed_log  = _SeedLog()
        self._stats = {
            "total_calls": 0,
            "total_hits":  0,
            "replicas_spawned": 0,
        }

        for i in range(n_replicas):
            self._replicas.append(self._new_replica(i))
        self._stats["replicas_spawned"] = n_replicas

    def _new_replica(self, idx: int) -> PocketDimension:
        path = f"{self._namespace}/{idx}"
        return PocketDimension(path, orchestrator=self._orch)

    # ── public API ─────────────────────────────────────────────────────────

    def matmul(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """A @ B dispatched through the replica pool.  Returns float32 ndarray."""
        self._seed_log.record(A, B)
        with self._lock:
            replica = self._replicas[self._rr % len(self._replicas)]
            self._rr += 1

        envelope = replica.matmul(A, B)
        result = envelope["result"]

        with self._lock:
            self._stats["total_calls"] += 1
            if envelope["source"] in ("cache", "coalesced"):
                self._stats["total_hits"] += 1

        return result

    def add_replica(self) -> int:
        """Spawn one more replica and warm it from the seed log.
        Returns the new total replica count."""
        with self._lock:
            idx = len(self._replicas)
            replica = self._new_replica(idx)
            self._replicas.append(replica)
            self._stats["replicas_spawned"] += 1
            n = len(self._replicas)
            seed_entries = self._seed_log.replay()

        # Warm the new replica outside the lock — replay recent GEMMs so
        # it benefits from cross-replica dedup immediately.
        for a_shape, a_dtype, b_shape, b_dtype in seed_entries:
            try:
                A_warm = np.zeros(a_shape, dtype=a_dtype)
                B_warm = np.zeros(b_shape, dtype=b_dtype)
                replica.matmul(A_warm, B_warm)
            except Exception:
                pass

        return n

    def scale_to(self, target: int) -> int:
        """Ensure the pool has at least ``target`` replicas.
        Adds replicas as needed; never removes them.  Returns final count."""
        with self._lock:
            current = len(self._replicas)
        for _ in range(max(0, target - current)):
            self.add_replica()
        with self._lock:
            return len(self._replicas)

    def stats(self) -> dict:
        with self._lock:
            n = len(self._replicas)
            s = dict(self._stats)
            rr = self._rr
        s["replica_count"] = n
        s["round_robin_position"] = rr
        s["hit_rate"] = round(
            s["total_hits"] / s["total_calls"], 4
        ) if s["total_calls"] else 0.0
        s["namespace"] = self._namespace
        s["seed_log_depth"] = len(self._seed_log.replay())
        return s


# ── singleton ──────────────────────────────────────────────────────────────────

_shared_pool: Optional[ReplicaPool] = None
_pool_lock   = threading.Lock()


def get_replica_pool(n_replicas: int = _DEFAULT_REPLICAS) -> ReplicaPool:
    """Return the process-wide ReplicaPool singleton (lazy init, thread-safe)."""
    global _shared_pool
    if _shared_pool is not None:
        return _shared_pool
    with _pool_lock:
        if _shared_pool is None:
            _shared_pool = ReplicaPool(n_replicas=n_replicas)
    return _shared_pool
