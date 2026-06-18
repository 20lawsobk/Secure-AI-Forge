"""
MaxCore Retrieval — Generated-Asset Ingestor (self-healing, Watchdog-pattern).

Folds the system's OWN successful image generations back into the retrieval
index, so the real-asset pool — and each brand's learned identity centroid —
grows from actual production output. This closes the retrieval loop: today's
render becomes tomorrow's conditioning evidence.

Mirrors CoverageWatchdog by design:
  • a daemon thread drains a locked, bounded queue every POLL_INTERVAL seconds;
  • every enqueue is deduped by resolved path (bounded dedupe memory);
  • ingestion failures atomically re-enqueue with a bounded retry budget, so a
    transient read error never silently loses an asset;
  • the poll loop is wrapped so one bad item can never kill the daemon;
  • references are injected after construction (no circular imports);
  • a rolling alert log + stats are exposed via get_status()/get_log().

Hard invariants (per project constraints):
  • TOTAL — enqueue() and the worker never raise into the caller or the
    render/generation path. Worst case is a logged drop, never an exception.
  • ADDITIVE — generated assets are added as NON-anchor and the backlog is
    bounded, so they can never crowd out or evict the curated anchor core.

Scope (Phase 4): DURABLE IMAGE outputs only. Video (mp4) is out of scope here.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Optional, Set

from ai_model.retrieval.asset_index import AssetIndex
from ai_model.retrieval.asset_pipeline import get_asset_index, ingest_generated_asset

logger = logging.getLogger("generated_ingestor")

POLL_INTERVAL = 5          # seconds between queue drains (light, in-process work)
MAX_QUEUE = 256            # bounded backlog — never grow without limit
MAX_SEEN = 4096            # bounded dedupe memory (oldest keys evicted)
MAX_RETRIES = 2            # per-item re-enqueue budget before giving up
BATCH = 16                 # items claimed per drain
MAX_LOG_ENTRIES = 200      # rolling alert log size


class IngestEvent:
    __slots__ = ("ts", "level", "kind", "message", "detail")

    def __init__(self, level: str, kind: str, message: str, detail: str = ""):
        self.ts = time.time()
        self.level = level          # "warning" | "info" | "ok"
        self.kind = kind
        self.message = message
        self.detail = detail

    def to_dict(self) -> dict:
        return {
            "ts": self.ts,
            "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.ts)),
            "level": self.level,
            "kind": self.kind,
            "message": self.message,
            "detail": self.detail,
        }


class GeneratedIngestor:
    """Self-healing daemon that folds produced images back into the index."""

    def __init__(self) -> None:
        self._lock = threading.Lock()           # guards stats + log
        self._queue_lock = threading.Lock()     # serializes queue + dedupe mutations
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._log: Deque[IngestEvent] = deque(maxlen=MAX_LOG_ENTRIES)

        self._queue: Deque[Dict[str, Any]] = deque()
        self._seen: Set[str] = set()
        self._seen_order: Deque[str] = deque()

        # Injected after construction (avoids circular imports). When None, the
        # process-wide asset index is resolved lazily at ingest time.
        self.index: Optional[AssetIndex] = None

        self.stats: Dict[str, Any] = {
            "started_at": None,
            "status": "stopped",
            "enqueued_total": 0,
            "ingested_total": 0,
            "deduped_total": 0,
            "dropped_total": 0,
            "retried_total": 0,
            "queue_len": 0,
            "last_ingest_at": None,
            "last_ingest_name": None,
        }

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    def start(self) -> dict:
        if self._thread and self._thread.is_alive():
            return {"already_running": True}
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="GeneratedIngestor")
        self._thread.start()
        with self._lock:
            self.stats["started_at"] = time.time()
            self.stats["status"] = "running"
        logger.info(f"[GeneratedIngestor] Started — draining every {POLL_INTERVAL}s")
        return {"started": True, "poll_interval_s": POLL_INTERVAL}

    def stop(self) -> None:
        self._stop_event.set()
        with self._lock:
            self.stats["status"] = "stopped"
        logger.info("[GeneratedIngestor] Stopped")

    def get_status(self) -> dict:
        with self._lock:
            s = dict(self.stats)
        with self._queue_lock:
            s["queue_len"] = len(self._queue)
        s["recent_events"] = [e.to_dict() for e in list(self._log)[-20:]]
        return s

    def get_log(self, limit: int = 50) -> list:
        entries = list(self._log)
        return [e.to_dict() for e in entries[-limit:]]

    # ------------------------------------------------------------------ #
    # Enqueue (called from the request/render path — must be TOTAL)        #
    # ------------------------------------------------------------------ #

    def enqueue(
        self,
        image_path: Any,
        *,
        brand: Optional[str] = None,
        endpoint: Optional[str] = None,
        platform: Optional[str] = None,
        job_id: Optional[str] = None,
    ) -> bool:
        """
        Queue a produced image for background ingestion. Non-blocking and TOTAL:
        returns True if newly queued, False if deduped/full/invalid — never raises.
        """
        try:
            if not image_path:
                return False
            key = self._dedupe_key(image_path)
            item = {
                "path": str(image_path),
                "brand": str(brand) if brand else None,
                "endpoint": str(endpoint) if endpoint else None,
                "platform": str(platform) if platform else None,
                "job_id": str(job_id) if job_id else None,
                "attempts": 0,
            }
            overflow = None
            with self._queue_lock:
                if key in self._seen:
                    deduped = True
                else:
                    deduped = False
                    if len(self._queue) >= MAX_QUEUE:
                        # Stay bounded: drop the oldest pending rather than block
                        # the caller or evict the curated anchor core downstream.
                        dropped = self._queue.popleft()
                        self._seen.discard(self._dedupe_key(dropped.get("path")))
                        overflow = dropped.get("path")
                    self._queue.append(item)
                    self._mark_seen(key)
                qlen = len(self._queue)
            with self._lock:
                self.stats["queue_len"] = qlen
                if deduped:
                    self.stats["deduped_total"] += 1
                else:
                    self.stats["enqueued_total"] += 1
                    if overflow is not None:
                        self.stats["dropped_total"] += 1
            if overflow is not None:
                self._alert(
                    "warning", "queue_overflow",
                    f"Queue full ({MAX_QUEUE}); dropped oldest pending "
                    f"{Path(str(overflow)).name} to stay bounded", "")
            return not deduped
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    # Main poll loop                                                       #
    # ------------------------------------------------------------------ #

    def _loop(self) -> None:
        logger.info("[GeneratedIngestor] Loop started")
        while not self._stop_event.is_set():
            try:
                self._drain_once()
            except Exception as e:
                logger.error(f"[GeneratedIngestor] Unexpected drain error: {e}")
            for _ in range(POLL_INTERVAL):
                if self._stop_event.is_set():
                    break
                time.sleep(1)
        with self._lock:
            self.stats["status"] = "stopped"
        logger.info("[GeneratedIngestor] Loop ended")

    def _drain_once(self) -> int:
        """Claim and process one bounded batch. Returns the number ingested."""
        with self._queue_lock:
            if not self._queue:
                return 0
            batch = [self._queue.popleft()
                     for _ in range(min(BATCH, len(self._queue)))]

        ingested = 0
        for item in batch:
            if self._process_item(item):
                ingested += 1

        with self._queue_lock:
            qlen = len(self._queue)
        with self._lock:
            self.stats["queue_len"] = qlen
        return ingested

    def _process_item(self, item: Dict[str, Any]) -> bool:
        path = item.get("path")
        try:
            idx = self.index if self.index is not None else get_asset_index()
            meta: Dict[str, Any] = {"source": "generated"}
            for k in ("endpoint", "platform", "job_id"):
                v = item.get(k)
                if v:
                    meta[k] = v
            ok = ingest_generated_asset(
                idx, path, brand=item.get("brand"), metadata=meta)
            if ok:
                name = Path(str(path)).name
                with self._lock:
                    self.stats["ingested_total"] += 1
                    self.stats["last_ingest_at"] = time.time()
                    self.stats["last_ingest_name"] = name
                self._alert("info", "ingested",
                            f"Folded generated asset into index: {name}",
                            item.get("brand") or "")
                return True
            # Unreadable/degenerate image — retry a bounded number of times.
            return self._maybe_retry(item, "ingest returned False")
        except Exception as e:
            return self._maybe_retry(item, f"exception: {e}")

    def _maybe_retry(self, item: Dict[str, Any], reason: str) -> bool:
        item["attempts"] = int(item.get("attempts", 0)) + 1
        if item["attempts"] <= MAX_RETRIES:
            with self._queue_lock:
                # Re-enqueue for a later drain; dedupe key stays set so the same
                # path cannot pile up multiple times while in flight. Honour the
                # bound — if the queue is full, give up rather than overshoot it.
                room = len(self._queue) < MAX_QUEUE
                if room:
                    self._queue.append(item)
            if room:
                with self._lock:
                    self.stats["retried_total"] += 1
                return False
            reason = f"{reason}; queue full, no room to retry"
        # Give up: drop and free the dedupe key so a future regen may retry.
        with self._queue_lock:
            self._seen.discard(self._dedupe_key(item.get("path")))
        with self._lock:
            self.stats["dropped_total"] += 1
        self._alert(
            "warning", "ingest_failed",
            f"Dropped generated asset after {item['attempts']} attempts "
            f"({Path(str(item.get('path'))).name}): {reason}", "")
        return False

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _dedupe_key(path: Any) -> str:
        try:
            return os.path.abspath(str(path))
        except Exception:
            return str(path)

    def _mark_seen(self, key: str) -> None:
        """Record a dedupe key; evict oldest to stay within MAX_SEEN. Caller holds queue lock."""
        if key in self._seen:
            return
        self._seen.add(key)
        self._seen_order.append(key)
        while len(self._seen_order) > MAX_SEEN:
            old = self._seen_order.popleft()
            self._seen.discard(old)

    def _alert(self, level: str, kind: str, message: str, detail: str = "") -> None:
        event = IngestEvent(level, kind, message, detail)
        self._log.append(event)
        if level == "warning":
            logger.warning(f"[GeneratedIngestor][WARNING][{kind}] {message}")
        else:
            logger.info(f"[GeneratedIngestor][{level.upper()}][{kind}] {message}")


# ------------------------------------------------------------------ #
# Singleton                                                           #
# ------------------------------------------------------------------ #

_generated_ingestor: Optional[GeneratedIngestor] = None


def get_generated_ingestor() -> GeneratedIngestor:
    global _generated_ingestor
    if _generated_ingestor is None:
        _generated_ingestor = GeneratedIngestor()
    return _generated_ingestor
