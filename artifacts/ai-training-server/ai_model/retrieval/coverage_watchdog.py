"""
MaxCore Retrieval Coverage Watchdog — self-healing for the asset index.

Mirrors workers/watchdog.py: a daemon thread that polls every POLL_INTERVAL
seconds, and for each check runs detect → alert → auto-heal, recording the fix
applied. It keeps the "no broken fallback" invariant true at runtime.

Checks every cycle:
  ① Anchor core present   — reload domain anchors if the index has none, so the
                            cascade's final rung (and thus a non-empty result)
                            is always available.
  ② Coverage gate         — if probe coverage is degraded/critical, log the weak
                            queries as durable ingestion targets and (if wired)
                            trigger an ingestion drain.
  ③ Ingestion backlog     — if the gap queue grows past a threshold, drive a
                            drain so coverage catches up instead of falling behind.
  ④ Index integrity       — verify a non-empty index never answers a usable query
                            with nothing; rebuild from persisted state if it does.

Like the server watchdog: references are injected after construction (no circular
imports), every check guards on missing refs, the poll loop is wrapped so one
failing check can never kill the daemon, and state is persisted to storage KV.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any, Callable, Dict, List, Optional

from ai_model.retrieval.asset_index import AssetIndex, MIN_ANCHORS

logger = logging.getLogger("coverage_watchdog")

POLL_INTERVAL = 60          # seconds between coverage checks
GAP_QUEUE_WARN = 200        # pending ingestion targets before we force a drain
DRAIN_COOLDOWN = 120        # min seconds between ingestion drains
MAX_LOG_ENTRIES = 200       # rolling alert log size
PROBE_SAMPLE = 32           # max probe vectors evaluated per coverage check


class CoverageAlert:
    __slots__ = ("ts", "level", "check", "message", "fix_applied")

    def __init__(self, level: str, check: str, message: str, fix_applied: str = ""):
        self.ts = time.time()
        self.level = level          # "critical" | "warning" | "info" | "ok"
        self.check = check
        self.message = message
        self.fix_applied = fix_applied

    def to_dict(self) -> dict:
        return {
            "ts": self.ts,
            "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.ts)),
            "level": self.level,
            "check": self.check,
            "message": self.message,
            "fix_applied": self.fix_applied,
        }


class CoverageWatchdog:
    """Self-healing daemon that keeps the retrieval index covered and non-empty."""

    STATE_KEY = "mb:coverage:state"
    GAP_QUEUE_KEY = "mb:coverage:gap_queue"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._gap_lock = threading.Lock()  # serializes all gap-queue mutations
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._log: deque = deque(maxlen=MAX_LOG_ENTRIES)

        # Injected after construction (avoids circular imports).
        self.index: Optional[AssetIndex] = None
        self.storage: Optional[Any] = None
        # anchor_loader_fn() → int: (re)load domain anchors into the index, return count.
        self.anchor_loader_fn: Optional[Callable[[], int]] = None
        # probe_source_fn() → list of recent query vectors to measure coverage with.
        self.probe_source_fn: Optional[Callable[[], List[Any]]] = None
        # ingestion_fn(list_of_gaps) → int: ingest assets for the given gaps, return count.
        self.ingestion_fn: Optional[Callable[[List[Any]], int]] = None

        self._last_drain_at: Optional[float] = None
        self._drain_in_flight = False

        self.stats: Dict[str, Any] = {
            "started_at": None,
            "checks_run": 0,
            "fixes_applied": 0,
            "alerts_critical": 0,
            "alerts_warning": 0,
            "last_check_at": None,
            "last_fix_at": None,
            "last_fix_desc": None,
            "last_gate": None,
            "gap_queue_len": 0,
            "status": "stopped",
        }

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    def start(self) -> dict:
        if self._thread and self._thread.is_alive():
            return {"already_running": True}
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="CoverageWatchdog")
        self._thread.start()
        with self._lock:
            self.stats["started_at"] = time.time()
            self.stats["status"] = "running"
        logger.info(f"[CoverageWatchdog] Started — polling every {POLL_INTERVAL}s")
        return {"started": True, "poll_interval_s": POLL_INTERVAL}

    def stop(self) -> None:
        self._stop_event.set()
        with self._lock:
            self.stats["status"] = "stopped"
        logger.info("[CoverageWatchdog] Stopped")

    def get_status(self) -> dict:
        with self._lock:
            s = dict(self.stats)
        s["recent_alerts"] = [a.to_dict() for a in list(self._log)[-20:]]
        return s

    def get_log(self, limit: int = 50) -> list:
        entries = list(self._log)
        return [a.to_dict() for a in entries[-limit:]]

    def reset_alerts(self) -> None:
        self._log.clear()
        with self._lock:
            self.stats["alerts_critical"] = 0
            self.stats["alerts_warning"] = 0

    # ------------------------------------------------------------------ #
    # Main poll loop                                                       #
    # ------------------------------------------------------------------ #

    def _loop(self) -> None:
        logger.info("[CoverageWatchdog] Loop started")
        while not self._stop_event.is_set():
            try:
                self._run_all_checks()
            except Exception as e:
                logger.error(f"[CoverageWatchdog] Unexpected poll error: {e}")
            for _ in range(POLL_INTERVAL):
                if self._stop_event.is_set():
                    break
                time.sleep(1)
        with self._lock:
            self.stats["status"] = "stopped"
        logger.info("[CoverageWatchdog] Loop ended")

    def _run_all_checks(self) -> None:
        now = time.time()
        with self._lock:
            self.stats["checks_run"] += 1
            self.stats["last_check_at"] = now
            self.stats["status"] = "checking"

        self._check_anchor_core(now)
        self._check_coverage(now)
        self._check_gap_backlog(now)
        self._check_index_integrity(now)

        with self._lock:
            self.stats["status"] = "running"
        self._persist_to_storage()

    # ------------------------------------------------------------------ #
    # Individual checks                                                    #
    # ------------------------------------------------------------------ #

    def _check_anchor_core(self, now: float) -> None:
        """Guarantee the always-real anchor rung exists; reload it if missing."""
        if self.index is None:
            return
        if self.index.anchor_count >= MIN_ANCHORS:
            return
        if self.anchor_loader_fn is None:
            self._alert(
                "critical", "anchor_core_empty",
                "Anchor core is empty and no anchor_loader_fn is wired — retrieval "
                "cannot guarantee a non-empty result until real anchors are loaded.",
                "",
            )
            return
        try:
            loaded = int(self.anchor_loader_fn())
        except Exception as e:
            self._alert("critical", "anchor_loader_error",
                        f"anchor_loader_fn raised: {e}", "")
            return
        if loaded >= MIN_ANCHORS or self.index.anchor_count >= MIN_ANCHORS:
            self._alert(
                "info", "anchor_core_reloaded",
                f"Anchor core reloaded ({self.index.anchor_count} anchors present).",
                f"Loaded {loaded} domain anchors into the index",
            )
        else:
            self._alert(
                "critical", "anchor_core_unfilled",
                "anchor_loader_fn ran but the index still has no anchors.",
                "",
            )

    def _check_coverage(self, now: float) -> None:
        """Measure coverage; enqueue weak queries as ingestion targets."""
        if self.index is None:
            return
        probes: List[Any] = []
        if self.probe_source_fn is not None:
            try:
                probes = list(self.probe_source_fn() or [])[:PROBE_SAMPLE]
            except Exception as e:
                logger.error(f"[CoverageWatchdog] probe_source_fn error: {e}")
                probes = []

        report = self.index.coverage_report(probes or None)
        gate = report.get("gate", "healthy")
        with self._lock:
            self.stats["last_gate"] = gate

        if gate == "healthy":
            return

        # Identify the probes that retrieval could not cover well and queue them.
        gaps = self._weak_probes(probes)
        enqueued = self._enqueue_gaps(gaps)

        level = "critical" if gate == "critical" else "warning"
        fix = ""
        if enqueued:
            fix = f"Queued {enqueued} weak queries as ingestion targets"
        self._alert(
            level, "coverage_gap",
            f"Coverage gate={gate} "
            f"(fraction_within_radius={report.get('fraction_within_radius', 0.0):.2f}, "
            f"assets={report.get('n_assets', 0)}, anchors={report.get('n_anchors', 0)}).",
            fix,
        )

        if enqueued:
            self._maybe_drain(now)

    def _check_gap_backlog(self, now: float) -> None:
        """If the ingestion queue grows too large, force a drain."""
        qlen = self._gap_queue_len()
        with self._lock:
            self.stats["gap_queue_len"] = qlen
        if qlen >= GAP_QUEUE_WARN:
            self._alert(
                "warning", "gap_backlog",
                f"Ingestion gap queue has {qlen} pending targets (≥ {GAP_QUEUE_WARN}).",
                "Forcing ingestion drain" if self.ingestion_fn else "",
            )
            self._maybe_drain(now, force=True)

    def _check_index_integrity(self, now: float) -> None:
        """A non-empty index must never answer a usable probe with nothing."""
        if self.index is None or self.index.size == 0:
            return
        probe = None
        if self.probe_source_fn is not None:
            try:
                ps = list(self.probe_source_fn() or [])
                probe = ps[0] if ps else None
            except Exception:
                probe = None
        if probe is None:
            return
        result = self.index.query(probe)
        if result is not None:
            return
        self._alert(
            "critical", "index_integrity",
            "Non-empty index returned no asset for a usable query — attempting "
            "rebuild from persisted state.",
            self._rebuild_from_storage(),
        )

    # ------------------------------------------------------------------ #
    # Heal helpers                                                         #
    # ------------------------------------------------------------------ #

    def _weak_probes(self, probes: List[Any]) -> List[Any]:
        if self.index is None or not probes:
            return []
        weak: List[Any] = []
        for p in probes:
            res = self.index.query(p)
            if res is None or res.rung in ("brand_prior", "anchor") or res.distance > 0.35:
                weak.append(p)
        return weak

    def _enqueue_gaps(self, gaps: List[Any]) -> int:
        if not gaps or self.storage is None:
            return 0
        try:
            import numpy as np
            payload = []
            for g in gaps:
                arr = np.asarray(g, dtype=np.float64).reshape(-1)
                payload.append({"ts": time.time(), "vector": arr.tolist()})
            if not payload:
                return 0
            with self._gap_lock:
                self.storage.lpush(self.GAP_QUEUE_KEY, *payload)
            return len(payload)
        except Exception as e:
            logger.error(f"[CoverageWatchdog] enqueue_gaps error: {e}")
            return 0

    def _gap_queue_len(self) -> int:
        if self.storage is None:
            return 0
        try:
            return int(self.storage.llen(self.GAP_QUEUE_KEY))
        except Exception:
            return 0

    def _maybe_drain(self, now: float, force: bool = False) -> None:
        if self.ingestion_fn is None or self.storage is None:
            return

        # Atomically claim the current batch under _gap_lock so a concurrent
        # enqueue cannot be lost in the gap between reading and clearing the queue.
        with self._gap_lock:
            if self._drain_in_flight:
                return
            if (not force and self._last_drain_at is not None
                    and (now - self._last_drain_at) < DRAIN_COOLDOWN):
                return
            try:
                raw = self.storage.lrange(self.GAP_QUEUE_KEY, 0, -1) or []
            except Exception:
                raw = []
            if not raw:
                return
            try:
                self.storage.delete(self.GAP_QUEUE_KEY)
            except Exception:
                return
            self._drain_in_flight = True
            self._last_drain_at = now

        def _run() -> None:
            try:
                count = int(self.ingestion_fn(raw))  # type: ignore[misc]
                self._alert(
                    "info", "ingestion_drain",
                    f"Drained ingestion queue: {count} assets ingested from {len(raw)} targets.",
                    f"Ingested {count} assets",
                )
            except Exception as ex:
                # Ingestion failed — re-queue the claimed batch so nothing is lost.
                try:
                    with self._gap_lock:
                        self.storage.lpush(self.GAP_QUEUE_KEY, *raw)
                except Exception:
                    pass
                self._alert("warning", "ingestion_drain_error",
                            f"Ingestion drain raised: {ex} — re-queued {len(raw)} targets.", "")
            finally:
                with self._gap_lock:
                    self._drain_in_flight = False

        threading.Thread(target=_run, daemon=True, name="CoverageDrain").start()

    def _rebuild_from_storage(self) -> str:
        if self.index is None or self.storage is None:
            return ""
        try:
            state = self.storage.get(self.STATE_KEY + ":index")
            if state and self.index.load_state(state):
                return f"Rebuilt index from persisted state ({self.index.size} assets)"
        except Exception as e:
            logger.error(f"[CoverageWatchdog] rebuild error: {e}")
        return ""

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _alert(self, level: str, check: str, message: str, fix_applied: str = "") -> None:
        alert = CoverageAlert(level, check, message, fix_applied)
        self._log.append(alert)

        if level == "critical":
            logger.error(f"[CoverageWatchdog][CRITICAL][{check}] {message}"
                         + (f" → FIX: {fix_applied}" if fix_applied else ""))
            with self._lock:
                self.stats["alerts_critical"] += 1
        elif level == "warning":
            logger.warning(f"[CoverageWatchdog][WARNING][{check}] {message}")
            with self._lock:
                self.stats["alerts_warning"] += 1
        else:
            logger.info(f"[CoverageWatchdog][{level.upper()}][{check}] {message}")

        if fix_applied:
            with self._lock:
                self.stats["fixes_applied"] += 1
                self.stats["last_fix_at"] = alert.ts
                self.stats["last_fix_desc"] = f"[{check}] {fix_applied}"

    def _persist_to_storage(self) -> None:
        if self.storage is None or not getattr(self.storage, "is_available", False):
            return
        try:
            snapshot = {
                "stats": self.stats,
                "recent_alerts": [a.to_dict() for a in list(self._log)[-20:]],
            }
            self.storage.set(self.STATE_KEY, snapshot)
            if self.index is not None:
                self.storage.set(self.STATE_KEY + ":index", self.index.to_state())
        except Exception:
            pass


# ------------------------------------------------------------------ #
# Singleton                                                           #
# ------------------------------------------------------------------ #

_coverage_watchdog: Optional[CoverageWatchdog] = None


def get_coverage_watchdog() -> CoverageWatchdog:
    global _coverage_watchdog
    if _coverage_watchdog is None:
        _coverage_watchdog = CoverageWatchdog()
    return _coverage_watchdog
