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


class _FallbackDedup:
    """In-process mirror of the dedup_cache API, used only if the real module
    cannot be imported. Mirrors its contract: only dict values are stored."""

    def __init__(self) -> None:
        self._d: dict[str, tuple[dict, float]] = {}
        self._stats = {"hits": 0, "misses": 0, "stores": 0, "errors": 0}
        self._lock = threading.Lock()

    def key_for(self, namespace: str, req: Any) -> Optional[str]:
        try:
            blob = json.dumps(req, sort_keys=True, default=str)
            return f"dedupe:{namespace}:" + hashlib.sha256(blob.encode()).hexdigest()
        except Exception:
            return None

    def get(self, key: Optional[str]) -> Optional[dict]:
        if not key:
            return None
        with self._lock:
            item = self._d.get(key)
            if item and item[1] > time.time():
                self._stats["hits"] += 1
                return item[0]
            self._stats["misses"] += 1
            return None

    def put(self, key: Optional[str], value: Any, ttl: Optional[int] = None) -> None:
        if not key or not isinstance(value, dict):
            return
        with self._lock:
            self._d[key] = (value, time.time() + (ttl or 3600))
            self._stats["stores"] += 1

    def stats(self) -> dict:
        with self._lock:
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


class PDIMOrchestrator:
    def __init__(self, storage=None, config: PDIMConfig | None = None, dedup=None):
        self.config = config or PDIMConfig()
        self.storage = storage
        self.dedup = dedup if dedup is not None else (_dedup or _FallbackDedup())
        self._inflight: dict[str, _Slot] = {}
        self._lock = threading.Lock()

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

        # Single-flight loop. Exactly one caller per key computes at a time; the
        # rest wait and share its result.  If the leader times out (still
        # running), dies, or produces a non-dict result, we re-contend so
        # exactly ONE waiter recomputes — never a parallel storm.
        while True:
            leader = False
            slot: _Slot
            with self._lock:
                existing = self._inflight.get(key)
                if existing is None:
                    slot = _Slot()
                    self._inflight[key] = slot
                    leader = True
                else:
                    slot = existing

            if leader:
                # Double-checked: a prior leader may have finished between our
                # cache miss above and winning leadership here.
                cached = self.dedup.get(key)
                if cached is not None:
                    self._release(key, slot)
                    METRICS.incr("pdim.cache_hit")
                    return {"result": cached, "source": "cache"}
                try:
                    METRICS.incr("pdim.compute")
                    with METRICS.timer("pdim.compute_ms"):
                        result = compute_fn(request)
                    # Write result onto slot BEFORE _release() signals followers
                    # so they receive it directly — no storage round-trip needed.
                    # dedup.put() persists to pdim for cross-node fleet dedup.
                    if isinstance(result, dict):
                        slot.result = result
                    self.dedup.put(key, result)
                    return {"result": result, "source": "compute"}
                finally:
                    # Runs after return — slot.result is already set above.
                    self._release(key, slot)

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

    def _release(self, key: str, slot: _Slot) -> None:
        with self._lock:
            self._inflight.pop(key, None)
        slot.event.set()

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
        with self._lock:
            out["inflight"] = len(self._inflight)
        if self.storage is not None:
            out["storage"] = self.storage.status()
        out["counters"] = METRICS.snapshot().get("counters", {})
        return out
