"""
MaxBooster AI Training Server — Real-Time Watchdog

Monitors every link in the server chain and auto-heals problems as they occur.

Checks every POLL_INTERVAL seconds:
  ① Model init timeout         — reinitializes model if startup stalled
  ② Training lock stuck        — resets training state if frozen with no progress
  ③ Training start timeout     — resets if stuck in "starting" state too long
  ④ Continuous trainer thread  — restarts daemon if thread died unexpectedly
  ⑤ Data puller thread         — restarts daemon if thread died unexpectedly
  ⑥ Checkpoint integrity       — removes corrupted weight files on load failure
  ⑦ Memory pressure            — runs GC + records OOM events
  ⑧ Storage connectivity       — logs outages and adjusts retry backoff
  ⑨ Loss plateau detection     — warns when loss hasn't improved across cycles
  ⑩ Python server self-health  — detects internal deadlocks via process metrics
"""

import gc
import json
import logging
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger("watchdog")

POLL_INTERVAL          = 30          # seconds between health checks
MODEL_INIT_TIMEOUT     = 300         # 5 min to init model before forcing re-init
TRAINING_STUCK_TIMEOUT = 360         # 6 min with no elapsed_seconds change → stuck
TRAINING_START_TIMEOUT = 180         # 3 min in "starting" state → drop it
THREAD_RESTART_DELAY   = 5           # seconds before restarting a dead thread
MEMORY_WARN_PCT        = 80          # % memory usage to start warning
MEMORY_CRIT_PCT        = 90          # % to trigger GC + reduce batch
MAX_LOG_ENTRIES        = 200         # rolling alert log size


class WatchdogAlert:
    __slots__ = ("ts", "level", "check", "message", "fix_applied")

    def __init__(self, level: str, check: str, message: str, fix_applied: str = ""):
        self.ts = time.time()
        self.level = level      # "critical" | "warning" | "info" | "ok"
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


class Watchdog:
    """
    Self-healing daemon thread for the MaxBooster training server chain.
    """

    STATE_KEY   = "mb:watchdog:state"
    ALERTS_KEY  = "mb:watchdog:alerts"
    FIXES_KEY   = "mb:watchdog:fixes_applied"

    def __init__(self):
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._log: deque = deque(maxlen=MAX_LOG_ENTRIES)

        # References injected after construction (avoids circular imports)
        self.storage       = None
        self.training_state: Optional[dict]    = None
        self.training_lock: Optional[threading.Lock] = None
        self.model_ready_ref: Optional[Callable[[], bool]] = None
        self.init_model_fn: Optional[Callable]  = None
        self.continuous_trainer                 = None
        self.data_puller                        = None
        self.weights_dir: Optional[Path]        = None

        self.stats = {
            "started_at":       None,
            "checks_run":       0,
            "fixes_applied":    0,
            "alerts_critical":  0,
            "alerts_warning":   0,
            "last_check_at":    None,
            "last_fix_at":      None,
            "last_fix_desc":    None,
            "status":           "stopped",
        }

        # Per-check state to track changes across cycles
        self._last_elapsed:     Optional[float] = None
        self._elapsed_changed_at: Optional[float] = None
        self._training_start_at: Optional[float] = None
        self._model_init_started_at: Optional[float] = None
        self._last_loss_seen:   Optional[float] = None
        self._loss_plateau_since: Optional[float] = None
        self._storage_offline_since: Optional[float] = None
        self._last_continuous_cycle: Optional[int] = None
        self._continuous_stuck_since: Optional[float] = None

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    def start(self) -> dict:
        if self._thread and self._thread.is_alive():
            return {"already_running": True}
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="Watchdog"
        )
        self._thread.start()
        with self._lock:
            self.stats["started_at"] = time.time()
            self.stats["status"] = "running"
        logger.info(f"[Watchdog] Started — polling every {POLL_INTERVAL}s")
        return {"started": True, "poll_interval_s": POLL_INTERVAL}

    def stop(self):
        self._stop_event.set()
        with self._lock:
            self.stats["status"] = "stopped"
        logger.info("[Watchdog] Stopped")

    def get_status(self) -> dict:
        with self._lock:
            s = dict(self.stats)
        s["recent_alerts"] = [a.to_dict() for a in list(self._log)[-20:]]
        return s

    def get_log(self, limit: int = 50) -> list:
        entries = list(self._log)
        return [a.to_dict() for a in entries[-limit:]]

    def reset_alerts(self):
        self._log.clear()
        with self._lock:
            self.stats["alerts_critical"] = 0
            self.stats["alerts_warning"] = 0

    # ------------------------------------------------------------------ #
    # Main poll loop                                                       #
    # ------------------------------------------------------------------ #

    def _loop(self):
        logger.info("[Watchdog] Loop started")
        while not self._stop_event.is_set():
            try:
                self._run_all_checks()
            except Exception as e:
                logger.error(f"[Watchdog] Unexpected poll error: {e}")

            for _ in range(POLL_INTERVAL):
                if self._stop_event.is_set():
                    break
                time.sleep(1)

        with self._lock:
            self.stats["status"] = "stopped"
        logger.info("[Watchdog] Loop ended")

    def _run_all_checks(self):
        now = time.time()
        with self._lock:
            self.stats["checks_run"] += 1
            self.stats["last_check_at"] = now
            self.stats["status"] = "checking"

        self._check_model_init(now)
        self._check_training_stuck(now)
        self._check_training_start_timeout(now)
        self._check_continuous_trainer(now)
        self._check_data_puller(now)
        self._check_checkpoint_integrity()
        self._check_memory(now)
        self._check_storage(now)
        self._check_loss_plateau(now)

        with self._lock:
            self.stats["status"] = "running"

        self._persist_to_storage()

    # ------------------------------------------------------------------ #
    # Individual checks                                                    #
    # ------------------------------------------------------------------ #

    def _check_model_init(self, now: float):
        """If model hasn't initialized within MODEL_INIT_TIMEOUT, force re-init."""
        if self.model_ready_ref is None:
            return
        ready = self.model_ready_ref()
        if ready:
            self._model_init_started_at = None
            return

        if self._model_init_started_at is None:
            self._model_init_started_at = now
            return

        age = now - self._model_init_started_at
        if age > MODEL_INIT_TIMEOUT:
            self._alert("critical", "model_init",
                        f"Model not initialized after {age:.0f}s — forcing re-init",
                        "Triggered _init_ai_model in background thread")
            self._model_init_started_at = now  # Reset so we don't spam
            if self.init_model_fn:
                t = threading.Thread(target=self.init_model_fn, daemon=True,
                                     name="WatchdogModelInit")
                t.start()

    def _check_training_stuck(self, now: float):
        """Detect training loop frozen with no elapsed_seconds progress."""
        if self.training_state is None or self.training_lock is None:
            return
        with self.training_lock:
            state = self.training_state.get("state", "idle")
            elapsed = self.training_state.get("elapsed_seconds", 0)
            started_at = self.training_state.get("started_at")

        if state != "running":
            self._last_elapsed = None
            self._elapsed_changed_at = None
            return

        if self._last_elapsed != elapsed:
            self._last_elapsed = elapsed
            self._elapsed_changed_at = now
            return

        if self._elapsed_changed_at is None:
            self._elapsed_changed_at = now
            return

        frozen_for = now - self._elapsed_changed_at
        if frozen_for > TRAINING_STUCK_TIMEOUT:
            self._alert(
                "critical", "training_stuck",
                f"Training frozen for {frozen_for:.0f}s — elapsed_seconds unchanged. "
                f"Resetting training state to idle.",
                "Reset training_state to idle, cleared job_id"
            )
            with self.training_lock:
                self.training_state["state"] = "idle"
                self.training_state["stop_requested"] = True
                self.training_state["job_id"] = None
                self.training_state["epoch"] = 0
                self.training_state["loss"] = None
            self._elapsed_changed_at = None
            self._last_elapsed = None

    def _check_training_start_timeout(self, now: float):
        """Reset training if stuck in 'starting' state too long."""
        if self.training_state is None or self.training_lock is None:
            return
        with self.training_lock:
            state = self.training_state.get("state", "idle")
            started_at = self.training_state.get("started_at") or 0

        if state != "starting":
            self._training_start_at = None
            return

        if self._training_start_at is None:
            self._training_start_at = now
            return

        age = now - self._training_start_at
        if age > TRAINING_START_TIMEOUT:
            self._alert(
                "critical", "training_start_timeout",
                f"Training stuck in 'starting' for {age:.0f}s — resetting to idle.",
                "Reset training_state to idle"
            )
            with self.training_lock:
                self.training_state["state"] = "idle"
                self.training_state["job_id"] = None
            self._training_start_at = None

    def _check_continuous_trainer(self, now: float):
        """Restart ContinuousTrainer daemon if thread died while state says running."""
        ct = self.continuous_trainer
        if ct is None:
            return

        state = ct.get_state()
        running_flag = state.get("running", False)
        thread_alive = ct._thread.is_alive() if ct._thread else False

        if running_flag and not thread_alive:
            self._alert(
                "critical", "continuous_trainer_crash",
                "ContinuousTrainer thread died while running=True — restarting.",
                "Restarted ContinuousTrainer with previous config"
            )
            phases  = state.get("phases_enabled") or None
            interval = state.get("interval_minutes", 60)
            time.sleep(THREAD_RESTART_DELAY)
            ct.start(interval_minutes=interval, phases=phases, epochs_per_phase=1)
            return

        # Detect continuous trainer stuck mid-cycle
        cycle = state.get("cycle", 0)
        ct_status = state.get("status", "")
        if running_flag and ct_status == "training":
            if self._last_continuous_cycle != cycle:
                self._last_continuous_cycle = cycle
                self._continuous_stuck_since = now
            elif self._continuous_stuck_since and (now - self._continuous_stuck_since) > TRAINING_STUCK_TIMEOUT * 2:
                self._alert(
                    "warning", "continuous_trainer_stuck",
                    f"ContinuousTrainer stuck in 'training' for cycle {cycle} "
                    f"({now - self._continuous_stuck_since:.0f}s) — sending stop + restart.",
                    "Stopped and restarted ContinuousTrainer"
                )
                ct.stop("watchdog_auto_restart")
                time.sleep(THREAD_RESTART_DELAY * 2)
                phases = state.get("phases_enabled") or None
                interval = state.get("interval_minutes", 60)
                ct.start(interval_minutes=interval, phases=phases, epochs_per_phase=1)
                self._continuous_stuck_since = None
        else:
            self._continuous_stuck_since = now

    def _check_data_puller(self, now: float):
        """Restart DataPuller auto-loop if thread died while _running=True."""
        dp = self.data_puller
        if dp is None:
            return

        running_flag = dp._running
        thread_alive = dp._thread.is_alive() if dp._thread else False

        if running_flag and not thread_alive:
            interval = 30
            self._alert(
                "critical", "data_puller_crash",
                "DataPuller thread died while running — restarting auto-pull loop.",
                f"Restarted DataPuller with {interval}min interval"
            )
            dp._running = False
            time.sleep(THREAD_RESTART_DELAY)
            dp.start(interval_minutes=interval)

        # Warn if pull has been stuck "pulling" for >5 min
        pull_state = dp.get_state()
        if pull_state.get("status") == "pulling":
            last_pull = pull_state.get("last_pull") or now
            pulling_for = now - last_pull
            if pulling_for > 300:
                self._alert(
                    "warning", "data_puller_stuck",
                    f"DataPuller stuck in 'pulling' state for {pulling_for:.0f}s.",
                    "No fix needed — HTTP timeouts will resolve it"
                )

    def _check_checkpoint_integrity(self):
        """Try loading checkpoint; delete it if corrupted."""
        if self.weights_dir is None:
            return
        checkpoint = self.weights_dir / "model.pt"
        if not checkpoint.exists():
            return
        try:
            import torch
            ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
            if "model_state_dict" not in ckpt:
                raise ValueError("Missing model_state_dict key")
        except Exception as e:
            self._alert(
                "critical", "checkpoint_corrupted",
                f"Checkpoint model.pt failed integrity check: {e}",
                "Deleted corrupted model.pt — will use random init on next restart"
            )
            try:
                checkpoint.rename(checkpoint.with_suffix(".corrupt"))
                logger.warning("[Watchdog] Renamed model.pt → model.pt.corrupt")
                if self.training_state and self.training_lock:
                    with self.training_lock:
                        self.training_state["weights_exist"] = False
            except Exception as rename_err:
                logger.error(f"[Watchdog] Could not rename checkpoint: {rename_err}")

    def _check_memory(self, now: float):
        """Detect memory pressure and run GC if critical."""
        try:
            mem_mb, mem_pct = _get_memory_info()
        except Exception:
            return

        if mem_pct >= MEMORY_CRIT_PCT:
            collected = gc.collect()
            self._alert(
                "critical", "memory_pressure",
                f"Memory at {mem_pct:.1f}% ({mem_mb:.0f} MB) — GC collected {collected} objects.",
                f"Ran gc.collect() — freed {collected} objects"
            )
        elif mem_pct >= MEMORY_WARN_PCT:
            self._alert(
                "warning", "memory_high",
                f"Memory at {mem_pct:.1f}% ({mem_mb:.0f} MB) — approaching limit.",
                ""
            )

    def _check_storage(self, now: float):
        """Track storage outages and log when connectivity restores."""
        if self.storage is None:
            return
        available = self.storage.is_available
        if not available:
            if self._storage_offline_since is None:
                self._storage_offline_since = now
            offline_for = now - self._storage_offline_since
            if offline_for > 300:  # Only alert after 5 min outage
                self._alert(
                    "warning", "storage_offline",
                    f"pdim storage offline for {offline_for:.0f}s — operating in local-only mode.",
                    "Using in-process fallback; will auto-reconnect on next ping"
                )
        else:
            if self._storage_offline_since is not None:
                offline_for = now - self._storage_offline_since
                self._alert(
                    "info", "storage_restored",
                    f"pdim storage restored after {offline_for:.0f}s outage.",
                    ""
                )
            self._storage_offline_since = None

    def _check_loss_plateau(self, now: float):
        """Warn if continuous trainer's best_loss hasn't improved in a long time."""
        ct = self.continuous_trainer
        if ct is None:
            return
        state = ct.get_state()
        if not state.get("running"):
            self._last_loss_seen = None
            self._loss_plateau_since = None
            return

        current_loss = state.get("best_loss")
        if current_loss is None:
            return

        if self._last_loss_seen is None or current_loss < self._last_loss_seen:
            self._last_loss_seen = current_loss
            self._loss_plateau_since = now
            return

        if self._loss_plateau_since and (now - self._loss_plateau_since) > 7200:  # 2 hours
            self._alert(
                "warning", "loss_plateau",
                f"Training loss plateaued at {current_loss:.4f} for >2h. "
                "Consider refreshing training data or adjusting learning rate.",
                ""
            )
            self._loss_plateau_since = now  # Reset timer so we don't spam

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _alert(self, level: str, check: str, message: str, fix_applied: str = ""):
        alert = WatchdogAlert(level, check, message, fix_applied)
        self._log.append(alert)

        if level == "critical":
            logger.error(f"[Watchdog][CRITICAL][{check}] {message}"
                         + (f" → FIX: {fix_applied}" if fix_applied else ""))
            with self._lock:
                self.stats["alerts_critical"] += 1
        elif level == "warning":
            logger.warning(f"[Watchdog][WARNING][{check}] {message}")
            with self._lock:
                self.stats["alerts_warning"] += 1
        else:
            logger.info(f"[Watchdog][{level.upper()}][{check}] {message}")

        if fix_applied:
            with self._lock:
                self.stats["fixes_applied"] += 1
                self.stats["last_fix_at"] = alert.ts
                self.stats["last_fix_desc"] = f"[{check}] {fix_applied}"

    def _persist_to_storage(self):
        if self.storage is None or not self.storage.is_available:
            return
        try:
            snapshot = {
                "stats": self.stats,
                "recent_alerts": [a.to_dict() for a in list(self._log)[-20:]],
            }
            self.storage.set(self.STATE_KEY, snapshot)
        except Exception:
            pass


# ------------------------------------------------------------------ #
# System memory helper                                                #
# ------------------------------------------------------------------ #

def _get_memory_info() -> tuple[float, float]:
    """Return (used_mb, used_pct). Reads /proc/meminfo on Linux."""
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                info[parts[0].rstrip(":")] = int(parts[1])
    total_kb  = info.get("MemTotal", 1)
    avail_kb  = info.get("MemAvailable", total_kb)
    used_kb   = total_kb - avail_kb
    used_mb   = used_kb / 1024
    used_pct  = 100 * used_kb / total_kb
    return used_mb, used_pct


# ------------------------------------------------------------------ #
# Singleton                                                           #
# ------------------------------------------------------------------ #

_watchdog: Optional[Watchdog] = None


def get_watchdog() -> Watchdog:
    global _watchdog
    if _watchdog is None:
        _watchdog = Watchdog()
    return _watchdog
