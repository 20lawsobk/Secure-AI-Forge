"""PDIM orchestrator — deduplicate, coalesce, batch, cache.

Two complementary paths, both built on what already exists in the repo:

1. ``compute()`` — synchronous. Hashes the request, returns a cached result on a
   hit (via the existing fleet-wide ``dedup_cache``), and *single-flights*
   concurrent identical requests so only one actually runs while the rest wait
   and share its result. Needs no durable storage.

2. ``submit()`` / ``poll()`` / ``process_queue_once()`` — durable async. Enqueue
   jobs to ``PDIMStorage``, drain them in micro-batches in a worker, store
   results to the payload/index tiers, and poll by hash. The preview→full stage
   is a *pluggable policy* (caller supplies ``preview_fn`` + ``quality_fn``);
   there is no hard-coded fake heuristic — default behaviour is full compute.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from typing import Any, Callable, Optional

from ..observability import METRICS
from .config import PDIMConfig

try:
    from ai_model import dedup_cache as _dedup
except Exception:  # pragma: no cover - exercised only if repo layout changes
    _dedup = None


_DEDUP_SHARDS = 256   # shard count; last-2-hex of sha256 key → 0-255


class _FallbackDedup:
    """In-process mirror of the dedup_cache API, used only if the real module
    cannot be imported.  Mirrors its contract: only dict values are stored.

    256-shard implementation: keys are sha256 hex digests so the last two hex
    characters give a perfectly uniform 0-255 shard index.  Concurrent get/put
    for *different* keys acquire *different* shard locks and never block each
    other — critical for 90 000-concurrent-request workloads where every unique
    request performs one dedup lookup per generation call.
    """

    def __init__(self) -> None:
        self._shards: list[dict[str, tuple[dict, float]]] = [
            {} for _ in range(_DEDUP_SHARDS)
        ]
        self._locks = [threading.Lock() for _ in range(_DEDUP_SHARDS)]
        # Stats use a separate lightweight lock — never held during data access.
        self._stats      = {"hits": 0, "misses": 0, "stores": 0, "errors": 0}
        self._stats_lock = threading.Lock()

    @staticmethod
    def _idx(key: str) -> int:
        """Shard index from the last two hex chars of a sha256 digest key."""
        return int(key[-2:], 16)   # 0x00-0xFF, O(1), no extra hash

    def key_for(self, namespace: str, req: Any) -> Optional[str]:
        try:
            blob = json.dumps(req, sort_keys=True, default=str)
            return f"dedupe:{namespace}:" + hashlib.sha256(blob.encode()).hexdigest()
        except Exception:
            return None

    def get(self, key: Optional[str]) -> Optional[dict]:
        if not key:
            return None
        idx = self._idx(key)
        now = time.time()
        with self._locks[idx]:
            item = self._shards[idx].get(key)
            hit  = item is not None and item[1] > now
            val  = item[0] if hit else None
        with self._stats_lock:
            if hit:
                self._stats["hits"] += 1
            else:
                self._stats["misses"] += 1
        return val

    def put(self, key: Optional[str], value: Any, ttl: Optional[int] = None) -> None:
        if not key or not isinstance(value, dict):
            return
        idx = self._idx(key)
        with self._locks[idx]:
            self._shards[idx][key] = (value, time.time() + (ttl or 3600))
        with self._stats_lock:
            self._stats["stores"] += 1

    def stats(self) -> dict:
        with self._stats_lock:
            h, m = self._stats["hits"], self._stats["misses"]
            total = h + m
            return {**self._stats, "hit_rate": round(h / total, 3) if total else 0.0}


class _Slot:
    """Single-flight coordination slot.

    Carries the threading.Event followers wait on, plus the result the leader
    writes *before* signalling.  Followers read ``result`` directly off the
    slot they already hold — no storage round-trip required.  The slot is
    discarded once its key leaves ``_inflight``; it is not a cache."""

    __slots__ = ("event", "result")

    def __init__(self) -> None:
        self.event: threading.Event = threading.Event()
        self.result: Optional[dict] = None


_INFLIGHT_SHARDS = 256   # same hex-tail trick as pocket_accelerator / _FallbackDedup


class PDIMOrchestrator:
    """PDIM orchestrator with 256-shard single-flight inflight tracking.

    The ``_inflight`` dict previously used a single ``threading.Lock`` shared
    by every concurrent request.  At 90 M concurrent unique requests the single
    lock serialises all slot-check/insert/remove operations even for completely
    unrelated keys.

    The 256-shard design assigns each key to exactly one shard via the last two
    hex characters of the sha256 key digest (0x00–0xFF → shard 0–255).
    Concurrent requests for different keys acquire different shard locks and
    never block each other — reducing worst-case contention from O(N) to
    O(N / 256) ≈ O(350) per shard at 90 M concurrent.
    """

    def __init__(self, storage=None, config: PDIMConfig | None = None, dedup=None):
        self.config = config or PDIMConfig()
        self.storage = storage
        self.dedup = dedup if dedup is not None else (_dedup or _FallbackDedup())
        # 256-shard inflight dicts replace the single _lock + _inflight pair.
        self._inflight_shards: list[dict[str, _Slot]] = [
            {} for _ in range(_INFLIGHT_SHARDS)
        ]
        self._shard_locks = [threading.Lock() for _ in range(_INFLIGHT_SHARDS)]

    # ── shard routing ─────────────────────────────────────────────────────────

    @staticmethod
    def _shard_idx(key: str) -> int:
        """Uniform 0-255 shard index from the last two hex chars of a sha256 key."""
        return int(key[-2:], 16)

    # ── synchronous: dedup + single-flight ────────────────────────────────────
    def compute(self, request: Any, compute_fn: Callable[[Any], dict],
                namespace: str | None = None) -> dict:
        ns = namespace or self.config.namespace
        key = self.dedup.key_for(ns, request)

        # Fleet-wide dedup via pdim storage.
        cached = self.dedup.get(key)
        if cached is not None:
            METRICS.incr("pdim.cache_hit")
            return {"result": cached, "source": "cache"}

        # Un-hashable request: nothing to dedup or coalesce on — compute directly.
        if not key:
            METRICS.incr("pdim.compute")
            with METRICS.timer("pdim.compute_ms"):
                result = compute_fn(request)
            return {"result": result, "source": "compute"}

        idx = self._shard_idx(key)

        # Single-flight loop. Exactly one caller per key computes at a time; the
        # rest wait and share its result.  If the leader times out (still
        # running), dies, or produces a non-dict result, we re-contend so
        # exactly ONE waiter recomputes — never a parallel storm.
        while True:
            leader = False
            slot: _Slot
            with self._shard_locks[idx]:
                existing = self._inflight_shards[idx].get(key)
                if existing is None:
                    slot = _Slot()
                    self._inflight_shards[idx][key] = slot
                    leader = True
                else:
                    slot = existing

            if leader:
                # Double-checked: a prior leader may have finished between our
                # cache miss above and winning leadership here.
                cached = self.dedup.get(key)
                if cached is not None:
                    self._release(key, slot, idx)
                    METRICS.incr("pdim.cache_hit")
                    return {"result": cached, "source": "cache"}
                try:
                    METRICS.incr("pdim.compute")
                    with METRICS.timer("pdim.compute_ms"):
                        result = compute_fn(request)
                    # Write result onto slot BEFORE signalling followers so
                    # they receive it directly — no storage round-trip needed.
                    if isinstance(result, dict):
                        slot.result = result
                    # Persist to pdim + release the slot in a background thread.
                    # The slot stays in _inflight_shards during the entire write
                    # (~85ms WAN), so any new request for this key that arrives
                    # during that window becomes a follower and reads slot.result
                    # directly — no recompute, no post-release/pre-pdim race.
                    threading.Thread(
                        target=self._put_and_release,
                        args=(key, slot, result, idx),
                        daemon=True,
                    ).start()
                    return {"result": result, "source": "compute"}
                except Exception:
                    # compute_fn raised or thread start failed; release immediately.
                    self._release(key, slot, idx)
                    raise

            # Follower: wait for the leader then read its result off the slot.
            # slot.result is set before ev.set(), so no storage race is possible.
            signaled = slot.event.wait(timeout=self.config.inflight_wait_seconds)
            if slot.result is not None:
                METRICS.incr("pdim.coalesced")
                return {"result": slot.result, "source": "coalesced"}
            # Leader timed out or produced a non-dict result — fall back to
            # pdim then re-contend if still a miss.
            cached = self.dedup.get(key)
            if cached is not None:
                METRICS.incr("pdim.coalesced")
                return {"result": cached, "source": "coalesced"}
            METRICS.incr("pdim.inflight_timeout" if not signaled else "pdim.recontend")

    def _release(self, key: str, slot: _Slot, idx: int) -> None:
        with self._shard_locks[idx]:
            self._inflight_shards[idx].pop(key, None)
        slot.event.set()

    def _put_and_release(self, key: str, slot: _Slot, result: Any, idx: int) -> None:
        """Persist result to the dedup cache then release the single-flight slot.

        Runs in a daemon thread so the leader's ``compute()`` call returns to its
        caller immediately after the result is computed.  The slot stays in
        ``_inflight_shards`` for the entire duration of the pdim write (~85 ms
        WAN), so any concurrent request for the same key that arrives during that
        window still finds the slot, becomes a follower, and reads ``slot.result``
        directly — no recompute, no race window between _release and pdim commit.
        """
        try:
            self.dedup.put(key, result)
        finally:
            self._release(key, slot, idx)

    # ── durable async: submit / poll / drain ──────────────────────────────────
    def submit(self, request: Any = None, *, queue: str = "default", model_id: str = "model",
               prompt: str = "", params: dict | None = None, context_sig: str = "") -> dict:
        if self.storage is None:
            raise RuntimeError("submit() requires a PDIMStorage; construct with storage=...")
        h = self.storage.make_hash(model_id, prompt, params or {}, context_sig)
        cached = self.storage.get_result(h)
        if cached is not None:
            METRICS.incr("pdim.cache_hit")
            return {"hash": h, "result": cached, "source": "cache"}
        job = {
            "hash": h, "model_id": model_id, "prompt": prompt,
            "params": params or {}, "context_sig": context_sig, "request": request,
        }
        self.storage.enqueue_job(queue, job)
        METRICS.incr("pdim.enqueued")
        return {"hash": h, "status": "queued", "source": "queued"}

    def poll(self, h: str) -> dict:
        if self.storage is None:
            raise RuntimeError("poll() requires a PDIMStorage")
        r = self.storage.get_result(h)
        return {"status": "done", "result": r} if r is not None else {"status": "pending"}

    def process_queue_once(self, compute_fn: Callable[[dict], dict], *, queue: str = "default",
                           preview_fn: Callable[[dict], dict] | None = None,
                           quality_fn: Callable[[dict, dict], bool] | None = None) -> int:
        if self.storage is None:
            raise RuntimeError("process_queue_once() requires a PDIMStorage")
        batch = self.storage.dequeue_batch(queue, self.config.batch_size)
        if not batch:
            return 0
        for job in batch:
            h = job.get("hash")
            if h and self.storage.get_result(h) is not None:
                METRICS.incr("pdim.queue_dedup")
                continue
            result: Optional[dict] = None
            if preview_fn is not None and quality_fn is not None:
                preview = preview_fn(job)
                if quality_fn(preview, job):
                    result = preview
                    METRICS.incr("pdim.preview_accepted")
            if result is None:
                with METRICS.timer("pdim.full_compute_ms"):
                    result = compute_fn(job)
                METRICS.incr("pdim.full_compute")
            if h and isinstance(result, dict):
                self.storage.store_result(h, result)
        return len(batch)

    def stats(self) -> dict:
        out: dict = {"dedup": self.dedup.stats()}
        # Sum inflight counts across all 256 shards without holding any lock
        # longer than necessary — each shard lock is acquired and released
        # independently so in-flight compute is not stalled by the stats read.
        total_inflight = 0
        for i in range(_INFLIGHT_SHARDS):
            with self._shard_locks[i]:
                total_inflight += len(self._inflight_shards[i])
        out["inflight"] = total_inflight
        out["inflight_shards"] = _INFLIGHT_SHARDS
        if self.storage is not None:
            out["storage"] = self.storage.status()
        out["counters"] = METRICS.snapshot().get("counters", {})
        return out
