"""Pocket Accelerator — in-process GEMM dedup cache for the Digital GPU.

Every GEMM (gemm / gemm_batched / mixed_gemm / …) is offered to this cache
before it reaches the TensorCoreUnit.  Identical operand arrays — identified by
a sha256 content hash — return the stored result in microseconds instead of
recomputing.

Architecture
────────────
The accelerator is a 256-shard LRU keyed by operand content-hash.  Each shard
owns its own ``threading.Lock`` and its own ``OrderedDict``.  Because the cache
key is already a sha256 hex digest, the last two hex characters provide a
perfectly uniform 0-255 shard index: concurrent GEMMs for *different* keys
land in *different* shards and never contend.

At 90 000 concurrent unique requests, each making N GEMM calls, the lock wait
is proportional to (90 000 × N) / 256 — roughly 350 waiters per shard instead
of 90 000 × N waiters on a single lock.

Results are stored in FP16 to halve cache footprint (doubles achievable hit
rate before eviction kicks in).  Stored FP16 is promoted back to FP32 on the
way out so callers see the same dtype as a fresh compute() result.

Adaptive gate
─────────────
Pockets with consistently low hit rates are muted to avoid wasting hash time on
matrices whose inputs change every call.  The gate re-probes every
``reprobe_every`` calls so it can recover if the workload becomes repetitive.
Small GEMMs (< ``min_flops``) are always bypassed — the hash cost exceeds the
compute cost.

Sharding contract
─────────────────
``_ShardedLRU`` shards both ``_store`` (the LRU dict) and per-shard byte
accounting.  ``_ShardedBuckets`` shards the per-pocket adaptive-gate state.
Stats (hits/misses/bypass counts) are separated into a dedicated lightweight
``_stats_lock`` that is acquired far less often than the data-path locks.
"""
from __future__ import annotations

import hashlib
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from ..observability import METRICS

# ── feature flag ──────────────────────────────────────────────────────────────

_ENV_ENABLED  = "POCKET_ACCEL_ENABLED"
_ENV_BUDGET_MB = "POCKET_ACCEL_BUDGET_MB"


def _enabled() -> bool:
    return os.environ.get(_ENV_ENABLED, "1") not in ("0", "false", "False")


# ── shard count ───────────────────────────────────────────────────────────────

_NUM_SHARDS = 256   # must be a power-of-two for `& 0xFF` trick; 256 = 2^8


# ── adaptive gate bucket ──────────────────────────────────────────────────────

@dataclass
class _PocketBucket:
    attempts: int = 0
    hits:     int = 0
    muted:    bool = False
    skipped:  int = field(default=0, repr=False)


# ── 256-shard LRU store ───────────────────────────────────────────────────────

class _ShardedLRU:
    """256-bucket sharded LRU.

    GEMMs for different cache keys land in different shards and never block
    each other.  The shard index is the integer value of the last two hex
    characters of the key — already uniformly distributed because keys are
    sha256 hex digests.
    """

    def __init__(self, budget_bytes: int) -> None:
        # Each shard gets an equal share of the byte budget.
        self._shard_budget = max(budget_bytes // _NUM_SHARDS, 512 * 1024)
        self._stores: list[OrderedDict] = [OrderedDict() for _ in range(_NUM_SHARDS)]
        self._locks   = [threading.Lock() for _ in range(_NUM_SHARDS)]
        self._bytes   = [0] * _NUM_SHARDS
        # Eviction counter needs its own lock (written rarely; not on hot path).
        self._evictions   = 0
        self._evict_lock  = threading.Lock()

    @staticmethod
    def _idx(key: str) -> int:
        """Map a hex-digest key to a shard index in O(1) — no extra hash."""
        return int(key[-2:], 16)   # last 2 hex chars → 0x00-0xFF

    def get(self, key: str) -> "Optional[tuple[np.ndarray, float]]":
        idx = self._idx(key)
        with self._locks[idx]:
            entry = self._stores[idx].get(key)
            if entry is not None:
                self._stores[idx].move_to_end(key)   # LRU promote
            return entry

    def put(self, key: str, value: np.ndarray, compute_seconds: float) -> None:
        # FP16-compress floating results to double effective cache capacity.
        if np.issubdtype(value.dtype, np.floating):
            arr = np.ascontiguousarray(value).astype(np.float16, copy=False).copy()
        else:
            arr = np.ascontiguousarray(value).copy()
        size = arr.nbytes
        if size > self._shard_budget:
            return   # single result too large for any shard
        idx    = self._idx(key)
        evicted = 0
        with self._locks[idx]:
            if key in self._stores[idx]:
                return   # already present — skip double-write
            while self._bytes[idx] + size > self._shard_budget and self._stores[idx]:
                _, (old, _) = self._stores[idx].popitem(last=False)
                self._bytes[idx] -= old.nbytes
                evicted += 1
            self._stores[idx][key] = (arr, compute_seconds)
            self._bytes[idx] += size
        if evicted:
            with self._evict_lock:
                self._evictions += evicted

    # ── aggregate stats (for observability only — not on hot path) ────────────

    @property
    def total_bytes(self) -> int:
        return sum(self._bytes)

    @property
    def total_entries(self) -> int:
        return sum(len(s) for s in self._stores)

    @property
    def total_evictions(self) -> int:
        with self._evict_lock:
            return self._evictions

    def clear(self) -> None:
        for i in range(_NUM_SHARDS):
            with self._locks[i]:
                self._stores[i].clear()
                self._bytes[i] = 0
        with self._evict_lock:
            self._evictions = 0


# ── 256-shard adaptive gate buckets ──────────────────────────────────────────

class _ShardedBuckets:
    """256-bucket sharded pocket-gate registry.

    Different pocket names land in different shards; concurrent gate/settle
    calls for different pockets never block each other.
    """

    def __init__(self) -> None:
        self._shards: list[dict[str, _PocketBucket]] = [{} for _ in range(_NUM_SHARDS)]
        self._locks  = [threading.Lock() for _ in range(_NUM_SHARDS)]

    @staticmethod
    def _idx(pocket: str) -> int:
        return hash(pocket) & 0xFF   # uniform 0-255 from string hash

    def gate(self, pocket: str, flops: float, min_flops: float,
             reprobe_every: int) -> bool:
        """Return True if this GEMM should be offered to the cache."""
        if flops < min_flops:
            return False
        idx = self._idx(pocket)
        with self._locks[idx]:
            b = self._shards[idx].get(pocket)
            if b is None:
                b = self._shards[idx][pocket] = _PocketBucket()
            if b.muted:
                b.skipped += 1
                if b.skipped % reprobe_every:   # not a re-probe turn
                    return False
                b.muted = False                 # re-probe this call
            return True

    def settle(self, pocket: str, hit: bool, warmup: int,
               hit_rate_floor: float) -> None:
        idx = self._idx(pocket)
        with self._locks[idx]:
            b = self._shards[idx].get(pocket)
            if b is None:
                return
            b.attempts += 1
            if hit:
                b.hits   += 1
                b.muted   = False
                b.skipped = 0
            elif (b.attempts >= warmup
                    and b.hits / b.attempts < hit_rate_floor):
                b.muted = True

    def snapshot(self) -> tuple[int, int]:
        """(total_pockets, muted_pockets) — called only for /gpu/status."""
        total = muted = 0
        for i in range(_NUM_SHARDS):
            with self._locks[i]:
                for b in self._shards[i].values():
                    total += 1
                    if b.muted:
                        muted += 1
        return total, muted

    def clear(self) -> None:
        for i in range(_NUM_SHARDS):
            with self._locks[i]:
                self._shards[i].clear()


# ── Pocket Accelerator ────────────────────────────────────────────────────────

class PocketAccelerator:
    """Content-hash GEMM dedup cache with 256-shard concurrency.

    ``accelerate(kind, operands, flops, compute)`` serves the result from the
    sharded LRU when the operand content-hash matches a stored entry; otherwise
    it calls ``compute()``, stores the result, and returns it.

    All data-path operations (get/put/gate/settle) acquire only the shard lock
    for their specific key — they never block unrelated keys.  Stats counters
    are updated under a separate lightweight ``_stats_lock`` that is never held
    during computation.
    """

    def __init__(
        self,
        budget_bytes:    Optional[int] = None,
        min_flops:       float = 2e6,
        warmup:          int   = 8,
        hit_rate_floor:  float = 0.05,
        reprobe_every:   int   = 16,
    ) -> None:
        if budget_bytes is None:
            budget_bytes = int(float(os.environ.get(_ENV_BUDGET_MB, "256")) * 1e6)
        self.budget_bytes   = max(budget_bytes, 1_000_000)
        self.min_flops      = min_flops
        self.warmup         = warmup
        self.hit_rate_floor = hit_rate_floor
        self.reprobe_every  = max(reprobe_every, 2)

        # Sharded data structures — no global lock on the hot path.
        self._lru     = _ShardedLRU(self.budget_bytes)
        self._buckets = _ShardedBuckets()

        # Stats: a dedicated lightweight lock, never held during compute.
        self._stats_lock            = threading.Lock()
        self._hits                  = 0
        self._misses                = 0
        self._bypass_small          = 0
        self._bypass_muted          = 0
        self._compute_seconds_saved = 0.0
        self._hit_serving_seconds   = 0.0

    # ── content-hash ──────────────────────────────────────────────────────────

    @staticmethod
    def _digest(*arrays: Optional[np.ndarray]) -> str:
        h = hashlib.sha256()
        for a in arrays:
            if a is None:
                h.update(b"<none>")
                continue
            c = np.ascontiguousarray(a)
            h.update(str(c.dtype).encode())
            h.update(str(c.shape).encode())
            h.update(c.tobytes())
        return h.hexdigest()

    # ── adaptive gate (uses sharded buckets) ──────────────────────────────────

    def _gate(self, pocket: str, flops: float) -> bool:
        if flops < self.min_flops:
            with self._stats_lock:
                self._bypass_small += 1
            return False
        allowed = self._buckets.gate(
            pocket, flops, self.min_flops, self.reprobe_every
        )
        if not allowed:
            with self._stats_lock:
                self._bypass_muted += 1
        return allowed

    def _settle(self, pocket: str, hit: bool) -> None:
        self._buckets.settle(pocket, hit, self.warmup, self.hit_rate_floor)

    # ── wired entry point ─────────────────────────────────────────────────────

    def accelerate(
        self,
        kind:      str,
        operands:  tuple,
        flops:     float,
        compute:   Callable[[], np.ndarray],
        extra_key: str = "",
    ) -> tuple[np.ndarray, str]:
        """Serve ``compute()`` through the 256-shard pocket cache.

        Returns ``(result, source)`` where source is one of:
        - ``"pocket"``  — served from cache (no compute)
        - ``"compute"`` — computed and cached
        - ``"bypass"``  — gate rejected; compute called directly, not cached
        """
        if not _enabled():
            return compute(), "bypass"

        first     = operands[0]
        shape_sig = "x".join(str(d) for d in first.shape) + "@" + \
                    "x".join(str(d) for d in operands[1].shape)
        pocket = f"gpu/{kind}/{shape_sig}"

        if not self._gate(pocket, flops):
            return compute(), "bypass"

        t0  = time.perf_counter()
        key = f"{pocket}:{self._digest(*operands)}{extra_key}"

        # ── cache hit path (acquires only shard lock for this key) ────────────
        entry = self._lru.get(key)
        if entry is not None:
            stored, saved = entry
            dt = time.perf_counter() - t0
            with self._stats_lock:
                self._hits                  += 1
                self._compute_seconds_saved += saved
                self._hit_serving_seconds   += dt
            self._settle(pocket, hit=True)
            METRICS.incr("pocket_accel.hit")
            # Dequantize: FP16-stored results promoted to FP32 on the way out
            # so callers see the same dtype as a fresh compute() result.
            if stored.dtype == np.float16:
                return stored.astype(np.float32), "pocket"   # always a new copy
            return stored.copy(), "pocket"

        # ── cache miss path ───────────────────────────────────────────────────
        c0             = time.perf_counter()
        result         = compute()
        compute_seconds = time.perf_counter() - c0

        self._lru.put(key, result, compute_seconds)
        with self._stats_lock:
            self._misses += 1
        self._settle(pocket, hit=False)
        METRICS.incr("pocket_accel.miss")
        return result, "compute"

    # ── observability ─────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        with self._stats_lock:
            hits    = self._hits
            misses  = self._misses
            bypS    = self._bypass_small
            bypM    = self._bypass_muted
            saved   = self._compute_seconds_saved
            serving = self._hit_serving_seconds

        lookups   = hits + misses
        total_pk, muted_pk = self._buckets.snapshot()

        speedup: Any
        if hits and serving > 0:
            speedup = round(saved / serving, 2)
        elif hits:
            speedup = float("inf")
        else:
            speedup = None

        return {
            "enabled":                    _enabled(),
            "shards":                     _NUM_SHARDS,
            "hits":                       hits,
            "misses":                     misses,
            "hit_rate":                   round(hits / lookups, 4) if lookups else 0.0,
            "bypass_small":               bypS,
            "bypass_adaptive_muted":      bypM,
            "pockets":                    total_pk,
            "pockets_muted":              muted_pk,
            "entries":                    self._lru.total_entries,
            "bytes_held":                 self._lru.total_bytes,
            "budget_bytes":               self.budget_bytes,
            "evictions":                  self._lru.total_evictions,
            "compute_seconds_saved":      round(saved, 6),
            "hit_serving_seconds":        round(serving, 6),
            "effective_speedup_on_hits":  speedup,
        }

    def clear(self) -> None:
        self._lru.clear()
        self._buckets.clear()
        with self._stats_lock:
            self._hits                  = 0
            self._misses                = 0
            self._bypass_small          = 0
            self._bypass_muted          = 0
            self._compute_seconds_saved = 0.0
            self._hit_serving_seconds   = 0.0


# ── shared singleton (one pocket tree per process, both GPUs feed it) ─────────

_shared:      Optional[PocketAccelerator] = None
_shared_lock = threading.Lock()


def get_pocket_accelerator() -> PocketAccelerator:
    global _shared
    if _shared is None:
        with _shared_lock:
            if _shared is None:
                _shared = PocketAccelerator()
    return _shared
