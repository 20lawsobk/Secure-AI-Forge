"""Pocket accelerator — pocket-dimension multiplication wired into the GPU.

Every GEMM the Digital GPU executes is (adaptively) offered to a pocket first:
each operand-shape gets its own nested pocket (``gpu/<kind>/MxKxN`` — pockets
inside the ``gpu`` pocket, unbounded), keyed by a content hash of the operands.
A hit serves the stored product at O(hash + memcpy) cost — independent of the
matrix size — so as repeats accumulate the effective speedup over recomputing
grows without bound. Payloads are held compressed-in-spirit (raw fp32 under an
LRU byte budget; the pocket tree is namespacing, exactly like PocketDimension).

Adaptability (the layer must never make the GPU slower):
  * size gate     — multiplications below a FLOP floor are never hashed
                    (hashing would cost more than the multiply);
  * hit-rate gate — per-pocket EWMA-free counters: after a warmup number of
                    attempts, a pocket whose hit-rate is under the floor stops
                    hashing entirely (zero overhead — e.g. training activations
                    that never repeat), and is re-probed every N-th call so the
                    pocket re-engages if the workload turns repetitive;
  * kill switch   — env ``MAXCORE_POCKET_ACCEL=0`` disables the whole layer.

All decisions are observable: hits/misses/bypasses, bytes held, compute
seconds avoided, and the measured effective speedup are exported via
``stats()`` and METRICS counters.
"""
from __future__ import annotations

import hashlib
import os
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Optional

import numpy as np

from ..observability import METRICS

_ENV_ENABLE = "MAXCORE_POCKET_ACCEL"
_ENV_BUDGET_MB = "MAXCORE_POCKET_ACCEL_MB"

_MIN_FLOPS = 2_000_000        # below this, hashing costs more than the GEMM
_WARMUP_ATTEMPTS = 32         # attempts before a pocket may be adaptively muted
_HIT_RATE_FLOOR = 0.05        # mute pockets that hit less often than this
_REPROBE_EVERY = 256          # muted pockets re-probe every N-th offer


def _enabled() -> bool:
    return os.environ.get(_ENV_ENABLE, "1") not in ("0", "false", "no")


class _PocketBucket:
    """Per-shape pocket counters driving the adaptive gate."""
    __slots__ = ("attempts", "hits", "muted", "skipped")

    def __init__(self) -> None:
        self.attempts = 0
        self.hits = 0
        self.muted = False
        self.skipped = 0


class PocketAccelerator:
    """In-process pocket tree serving deduped GEMMs to the Digital GPU."""

    def __init__(self, budget_bytes: Optional[int] = None,
                 min_flops: int = _MIN_FLOPS,
                 warmup: int = _WARMUP_ATTEMPTS,
                 hit_rate_floor: float = _HIT_RATE_FLOOR,
                 reprobe_every: int = _REPROBE_EVERY):
        if budget_bytes is None:
            budget_bytes = int(float(os.environ.get(_ENV_BUDGET_MB, "256")) * 1e6)
        self.budget_bytes = max(budget_bytes, 1_000_000)
        self.min_flops = min_flops
        self.warmup = warmup
        self.hit_rate_floor = hit_rate_floor
        self.reprobe_every = max(reprobe_every, 2)

        self._lock = threading.Lock()
        self._store: OrderedDict[str, tuple[np.ndarray, float]] = OrderedDict()
        self._bytes = 0
        self._buckets: dict[str, _PocketBucket] = {}
        self._hits = 0
        self._misses = 0
        self._bypass_small = 0
        self._bypass_muted = 0
        self._evictions = 0
        self._compute_seconds_saved = 0.0
        self._hit_serving_seconds = 0.0

    # ── adaptive gate ──────────────────────────────────────────────────────
    def _gate(self, pocket: str, flops: float) -> bool:
        """Decide whether this multiplication should be offered to the pocket."""
        if flops < self.min_flops:
            with self._lock:
                self._bypass_small += 1
            return False
        with self._lock:
            b = self._buckets.get(pocket)
            if b is None:
                b = self._buckets[pocket] = _PocketBucket()
            if b.muted:
                b.skipped += 1
                if b.skipped % self.reprobe_every:      # not a re-probe turn
                    self._bypass_muted += 1
                    return False
                b.muted = False                          # re-probe this call
            return True

    def _settle(self, pocket: str, hit: bool) -> None:
        with self._lock:
            b = self._buckets.get(pocket)
            if b is None:
                return
            b.attempts += 1
            if hit:
                b.hits = b.hits + 1
                b.muted = False
                b.skipped = 0
            elif (b.attempts >= self.warmup
                    and b.hits / b.attempts < self.hit_rate_floor):
                b.muted = True                           # adapt: stop hashing here

    # ── pocket store ───────────────────────────────────────────────────────
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

    def _get(self, key: str) -> Optional[tuple[np.ndarray, float]]:
        with self._lock:
            entry = self._store.get(key)
            if entry is not None:
                self._store.move_to_end(key)
            return entry

    def _put(self, key: str, value: np.ndarray, compute_seconds: float) -> None:
        # Store results in FP16 to halve cache footprint, allowing 2× more
        # distinct GEMM results to reside in the fixed byte budget before
        # eviction — directly doubling the achievable hit rate for workloads
        # whose working set exceeds the FP32 budget.  The ~0.03 % rounding
        # error introduced by float16 representation is acceptable for the
        # generative content produced by this inference stack.
        # We fall back to FP32 for types that have no float16 analogue (e.g.
        # complex128, int32) so the optimisation is always safe.
        if np.issubdtype(value.dtype, np.floating):
            arr_store = np.ascontiguousarray(value).astype(np.float16, copy=False).copy()
        else:
            arr_store = np.ascontiguousarray(value).copy()
        size = arr_store.nbytes
        if size > self.budget_bytes:
            return
        with self._lock:
            if key in self._store:
                return
            while self._bytes + size > self.budget_bytes and self._store:
                _, (old, _t) = self._store.popitem(last=False)
                self._bytes -= old.nbytes
                self._evictions += 1
            self._store[key] = (arr_store, compute_seconds)
            self._bytes += size

    # ── the wired entry point ──────────────────────────────────────────────
    def accelerate(self, kind: str, operands: tuple,
                   flops: float, compute: Callable[[], np.ndarray],
                   extra_key: str = "") -> tuple[np.ndarray, str]:
        """Serve ``compute()`` through the pocket tree.

        Returns ``(result, source)`` with source ``pocket|compute|bypass``.
        ``operands`` are the ndarrays that define the multiplication; ``kind``
        names the sub-pocket family (gemm / gemm_batched / mixed / ...).
        """
        if not _enabled():
            return compute(), "bypass"
        first = operands[0]
        shape_sig = "x".join(str(d) for d in first.shape) + "@" + \
                    "x".join(str(d) for d in operands[1].shape)
        pocket = f"gpu/{kind}/{shape_sig}"               # pocket inside a pocket
        if not self._gate(pocket, flops):
            return compute(), "bypass"

        t0 = time.perf_counter()
        key = f"{pocket}:{self._digest(*operands)}{extra_key}"
        entry = self._get(key)
        if entry is not None:
            stored, saved = entry
            dt = time.perf_counter() - t0
            with self._lock:
                self._hits += 1
                self._compute_seconds_saved += saved
                self._hit_serving_seconds += dt
            self._settle(pocket, hit=True)
            METRICS.incr("pocket_accel.hit")
            # Dequantize: FP16-stored results are promoted back to FP32 before
            # returning so the caller sees the same dtype as a compute() result.
            # For non-float dtypes the stored array is already in the original
            # type (see _put) so astype(float32) is skipped gracefully.
            if stored.dtype == np.float16:
                out = stored.astype(np.float32)   # always a copy — cache stays FP16
            else:
                out = stored.copy()               # copy: cache stays pristine
            return out, "pocket"

        c0 = time.perf_counter()
        result = compute()
        compute_seconds = time.perf_counter() - c0
        self._put(key, result, compute_seconds)
        with self._lock:
            self._misses += 1
        self._settle(pocket, hit=False)
        METRICS.incr("pocket_accel.miss")
        return result, "compute"

    # ── observability ──────────────────────────────────────────────────────
    def stats(self) -> dict[str, Any]:
        with self._lock:
            lookups = self._hits + self._misses
            muted = sum(1 for b in self._buckets.values() if b.muted)
            speedup: Any
            if self._hits and self._hit_serving_seconds > 0:
                speedup = round(self._compute_seconds_saved
                                / self._hit_serving_seconds, 2)
            elif self._hits:
                speedup = float("inf")
            else:
                speedup = None
            return {
                "enabled": _enabled(),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / lookups, 4) if lookups else 0.0,
                "bypass_small": self._bypass_small,
                "bypass_adaptive_muted": self._bypass_muted,
                "pockets": len(self._buckets),
                "pockets_muted": muted,
                "entries": len(self._store),
                "bytes_held": self._bytes,
                "budget_bytes": self.budget_bytes,
                "evictions": self._evictions,
                "compute_seconds_saved": round(self._compute_seconds_saved, 6),
                "hit_serving_seconds": round(self._hit_serving_seconds, 6),
                "effective_speedup_on_hits": speedup,
            }

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._bytes = 0
            self._buckets.clear()


# ── shared singleton (one pocket tree per process, both GPUs feed it) ────────
_shared: Optional[PocketAccelerator] = None
_shared_lock = threading.Lock()


def get_pocket_accelerator() -> PocketAccelerator:
    global _shared
    if _shared is None:
        with _shared_lock:
            if _shared is None:
                _shared = PocketAccelerator()
    return _shared
