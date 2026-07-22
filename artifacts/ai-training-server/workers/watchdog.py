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
import logging
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger("watchdog")

POLL_INTERVAL          = 30          # seconds between health checks
MODEL_INIT_TIMEOUT     = 300         # 5 min to init model before forcing re-init
TRAINING_STUCK_TIMEOUT = 360         # 6 min with no elapsed_seconds change → stuck
TRAINING_START_TIMEOUT = 180         # 3 min in "starting" state → drop it
THREAD_RESTART_DELAY   = 5           # seconds before restarting a dead thread
MEMORY_WARN_PCT        = 75          # % memory usage to start warning
MEMORY_CRIT_PCT        = 85          # % to trigger GC aggressively
MEMORY_DANGER_PCT      = 92          # % to take extreme measures (clear all caches)
KEEPALIVE_INTERVAL         = 300         # 5 min between rendering keep-alive probes
MAX_LOG_ENTRIES            = 200         # rolling alert log size
QUALITY_HARVEST_STALE_SEC  = 86400      # 24 h before triggering auto-harvest


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

        # Extended stay-alive references — injected from server.py
        self.coverage_watchdog                  = None   # CoverageWatchdog instance
        self.generated_ingestor                 = None   # GeneratedIngestor instance
        self.flywheel_ingestor_fn: Optional[Callable] = None  # () → FlywheelIngestor|None
        self.storage_client_ref                 = None   # StorageClient (not the storage abstraction)

        # Content generation service references
        self.gen_coalescer                      = None   # GenerateCoalescer instance
        self.render_manager                     = None   # RenderManager instance
        self.reinstall_coalescer_fn: Optional[Callable] = None  # () reinstalls coalescer in-place

        # Rendering system health — injected from server.py
        # rendering_health_fn() → {"ready": bool, "objects": {name: bool, ...}}
        self.rendering_health_fn: Optional[Callable[[], dict]] = None
        # keepalive_fn() → bool — runs a minimal end-to-end inference probe
        self.keepalive_fn: Optional[Callable[[], bool]] = None
        self._last_keepalive_at: Optional[float] = None
        self._reinit_in_flight = False

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
        # Extended stay-alive per-check state
        self._audio_empty_since: Optional[float] = None
        self._quality_harvest_alerted_at: Optional[float] = None

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

        self._check_rendering_system(now)
        self._check_model_init(now)
        self._check_training_stuck(now)
        self._check_training_start_timeout(now)
        self._check_continuous_trainer(now)
        self._check_data_puller(now)
        self._check_coverage_watchdog(now)
        self._check_generated_ingestor(now)
        self._check_flywheel_ingestor(now)
        self._check_audio_dataset(now)
        self._check_gen_coalescer(now)
        self._check_render_manager(now)
        self._check_quality_harvester_freshness(now)
        self._check_storage_health_thread(now)
        self._check_checkpoint_integrity()
        self._check_storage(now)
        self._check_loss_plateau(now)

        with self._lock:
            self.stats["status"] = "running"

        self._persist_to_storage()

    # ------------------------------------------------------------------ #
    # Individual checks                                                    #
    # ------------------------------------------------------------------ #

    def _check_rendering_system(self, now: float):
        """
        Verify every rendering object is alive and the pipeline can produce output.

        Two sub-checks:
          A) Object presence — all required AI objects must be non-None.
             If any are missing, immediately trigger _init_ai_model.
          B) Keep-alive probe — every KEEPALIVE_INTERVAL seconds, run a minimal
             inference to prove the pipeline is end-to-end healthy.
             If the probe fails, trigger _init_ai_model.
        """
        if self.rendering_health_fn is None:
            return

        # If the model hasn't finished first-time init yet, let _check_model_init
        # handle it — don't race against the startup thread.
        if self.model_ready_ref is not None and not self.model_ready_ref():
            return

        try:
            health = self.rendering_health_fn()
        except Exception as e:
            logger.error(f"[Watchdog] rendering_health_fn error: {e}")
            return

        objects: dict = health.get("objects", {})
        missing = [name for name, ok in objects.items() if not ok]

        if missing:
            if not self._reinit_in_flight:
                self._alert(
                    "critical", "rendering_objects_missing",
                    f"Rendering system missing objects: {missing}. Forcing re-init.",
                    "Triggered _init_ai_model in background thread"
                )
                self._trigger_reinit("WatchdogRenderingReinit")
            return

        # All objects present — run keep-alive probe on schedule
        if self.keepalive_fn is None:
            return

        if (self._last_keepalive_at is not None and
                now - self._last_keepalive_at < KEEPALIVE_INTERVAL):
            return

        self._last_keepalive_at = now

        def _probe():
            try:
                ok = self.keepalive_fn()
                if not ok:
                    self._alert(
                        "critical", "rendering_keepalive_fail",
                        "Keep-alive inference probe returned False — rendering degraded.",
                        "Triggered _init_ai_model to restore rendering system"
                    )
                    self._trigger_reinit("WatchdogKeepaliveReinit")
                else:
                    logger.info("[Watchdog] Rendering keep-alive probe OK.")
            except Exception as ex:
                self._alert(
                    "warning", "rendering_keepalive_error",
                    f"Keep-alive probe raised: {ex}",
                    ""
                )

        threading.Thread(target=_probe, daemon=True, name="WatchdogKeepalive").start()

    def _trigger_reinit(self, thread_name: str = "WatchdogReinit"):
        """Spawn _init_ai_model in a background thread (guarded against double-start)."""
        if self._reinit_in_flight or self.init_model_fn is None:
            return
        self._reinit_in_flight = True

        def _run():
            try:
                self.init_model_fn()
            finally:
                self._reinit_in_flight = False

        threading.Thread(target=_run, daemon=True, name=thread_name).start()

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
            self._trigger_reinit("WatchdogModelInit")

    def _check_training_stuck(self, now: float):
        """Detect training loop frozen with no elapsed_seconds progress."""
        if self.training_state is None or self.training_lock is None:
            return
        with self.training_lock:
            state = self.training_state.get("state", "idle")
            elapsed = self.training_state.get("elapsed_seconds", 0)
            _started_at = self.training_state.get("started_at")

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
            _started_at = self.training_state.get("started_at") or 0

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

    def _check_coverage_watchdog(self, now: float):
        """Restart the CoverageWatchdog daemon if its thread died silently."""
        cw = self.coverage_watchdog
        if cw is None:
            return
        thread_alive = cw._thread.is_alive() if getattr(cw, "_thread", None) else False
        if thread_alive:
            return
        self._alert(
            "critical", "coverage_watchdog_crash",
            "CoverageWatchdog thread died — restarting asset-coverage daemon.",
            "Called coverage_watchdog.start()"
        )
        try:
            cw.start()
        except Exception as exc:
            logger.error("[Watchdog] Could not restart CoverageWatchdog: %s", exc)

    def _check_generated_ingestor(self, now: float):
        """Restart the GeneratedIngestor daemon if its thread died silently.

        GeneratedIngestor folds rendered images back into the retrieval index.
        If it dies quietly, produced assets stop feeding the flywheel.
        """
        gi = self.generated_ingestor
        if gi is None:
            return
        thread_alive = gi._thread.is_alive() if getattr(gi, "_thread", None) else False
        if thread_alive:
            return
        self._alert(
            "critical", "generated_ingestor_crash",
            "GeneratedIngestor thread died — produced assets no longer folding back into index. Restarting.",
            "Called generated_ingestor.start()"
        )
        try:
            gi.start()
        except Exception as exc:
            logger.error("[Watchdog] Could not restart GeneratedIngestor: %s", exc)

    def _check_flywheel_ingestor(self, now: float):
        """Recreate the FlywheelIngestor executor if it was shut down."""
        if self.flywheel_ingestor_fn is None:
            return
        try:
            fw = self.flywheel_ingestor_fn()
        except Exception:
            return
        if fw is None:
            return
        # ThreadPoolExecutor exposes _shutdown=True once shutdown() has been called.
        if getattr(fw._executor, "_shutdown", False):
            self._alert(
                "critical", "flywheel_executor_dead",
                "FlywheelIngestor executor shut down — admin content ingestion stalled. Restarting.",
                "Replaced executor with a new ThreadPoolExecutor via FlywheelIngestor.restart()"
            )
            try:
                fw.restart()
            except Exception as exc:
                logger.error("[Watchdog] Could not restart FlywheelIngestor: %s", exc)

    def _check_audio_dataset(self, now: float):
        """Re-seed the audio dataset if all chunks have been lost from storage.

        Waits 60 s before acting to avoid re-seeding spuriously on the first
        poll after a clean restart (chunks may just be slow to appear).
        """
        if self.storage is None or not self.storage.is_available:
            return
        try:
            meta = self.storage.get("mb:dataset:audio:meta")
            chunks = int((meta or {}).get("num_chunks", 0))
        except Exception:
            return

        if chunks > 0:
            self._audio_empty_since = None
            return

        # Dataset is empty — start the grace-period timer
        if self._audio_empty_since is None:
            self._audio_empty_since = now
            return
        if now - self._audio_empty_since < 60:
            return

        # Reset so we don't re-trigger until the next empty-cycle
        self._audio_empty_since = now
        self._alert(
            "warning", "audio_dataset_empty",
            "Audio dataset has 0 chunks — triggering background re-seed to restore playback.",
            "Spawned seed_audio_dataset.seed(count=12, replace=False) in background thread"
        )
        storage = self.storage

        def _reseed():
            try:
                from workers.seed_audio_dataset import (  # noqa: PLC0415
                    seed as _seed_fn,
                    AlreadySeedingError,
                )
                _seed_fn(storage, count=12, replace=False)
                logger.info("[Watchdog] Audio dataset re-seed complete.")
            except AlreadySeedingError:
                logger.info("[Watchdog] Audio re-seed skipped — already seeding.")
            except Exception as exc:
                logger.warning("[Watchdog] Audio re-seed failed: %s", exc)

        threading.Thread(
            target=_reseed, daemon=True, name="WatchdogAudioReseed"
        ).start()

    def _check_gen_coalescer(self, now: float):
        """Reinstall the dynamic-batching coalescer if its worker threads died.

        GenerateCoalescer uses two one-shot daemon threads (_collector, _executor).
        Once dead they cannot be individually restarted — the whole coalescer must
        be reinstalled via the injected reinstall_coalescer_fn callable.
        While reinstalling, the model's .generate fallback path stays active so
        requests are never dropped, just un-batched.
        """
        gc = self.gen_coalescer
        if gc is None:
            return
        collector_alive = gc._collector.is_alive() if getattr(gc, "_collector", None) else True
        executor_alive  = gc._executor.is_alive()  if getattr(gc, "_executor",  None) else True
        if collector_alive and executor_alive:
            return

        dead = []
        if not collector_alive:
            dead.append("gen-collector")
        if not executor_alive:
            dead.append("gen-executor")

        self._alert(
            "critical", "gen_coalescer_threads_dead",
            f"GenerateCoalescer threads dead ({dead}) — batching degraded, reinstalling.",
            "Spawned reinstall_coalescer_fn() in background thread"
        )
        if self.reinstall_coalescer_fn is not None:
            def _do():
                try:
                    new_gc = self.reinstall_coalescer_fn()
                    if new_gc is not None:
                        self.gen_coalescer = new_gc
                        logger.info("[Watchdog] GenerateCoalescer reinstalled successfully.")
                except Exception as exc:
                    logger.error("[Watchdog] Coalescer reinstall failed: %s", exc)
            threading.Thread(target=_do, daemon=True, name="WatchdogCoalescerReinit").start()

    def _check_render_manager(self, now: float):
        """Keep RenderManager's GC thread and executor alive.

        If the GC thread dies, completed render jobs accumulate in memory indefinitely.
        If the executor shuts down, new render jobs fail silently on submit.
        """
        rm = self.render_manager
        if rm is None:
            return

        gc_thread = getattr(rm, "_gc_thread", None)
        if gc_thread is not None and not gc_thread.is_alive():
            self._alert(
                "critical", "render_manager_gc_dead",
                "RenderManager GC thread died — completed render jobs will accumulate. Restarting.",
                "Called RenderManager.restart_gc_thread()"
            )
            try:
                rm.restart_gc_thread()
            except Exception as exc:
                logger.error("[Watchdog] RenderManager GC restart failed: %s", exc)

        executor = getattr(rm, "_executor", None)
        if executor is not None and getattr(executor, "_shutdown", False):
            self._alert(
                "critical", "render_manager_executor_dead",
                "RenderManager ThreadPoolExecutor shut down — render jobs will fail. Restarting.",
                "Called RenderManager.restart_executor()"
            )
            try:
                rm.restart_executor()
            except Exception as exc:
                logger.error("[Watchdog] RenderManager executor restart failed: %s", exc)

    def _check_quality_harvester_freshness(self, now: float):
        """Trigger a background quality harvest when awareness data goes stale.

        quality_harvester.harvest() writes mb:awareness:quality:doc.  If that
        data is absent or older than QUALITY_HARVEST_STALE_SEC, generation
        quality degrades because the awareness blending layer has no live
        chart or engagement signal to draw from.
        """
        if self.storage is None or not self.storage.is_available:
            return

        try:
            doc = self.storage.get("mb:awareness:quality:doc")
            if isinstance(doc, dict):
                harvested_at_str = doc.get("harvested_at", "")
                if harvested_at_str:
                    ts = time.strptime(harvested_at_str, "%Y-%m-%dT%H:%M:%SZ")
                    age_sec = now - time.mktime(ts)
                    if age_sec < QUALITY_HARVEST_STALE_SEC:
                        self._quality_harvest_alerted_at = None
                        return
        except Exception:
            pass  # missing or unparseable — treat as stale

        # Rate-limit alerts: at most one per QUALITY_HARVEST_STALE_SEC window
        if (self._quality_harvest_alerted_at is not None and
                now - self._quality_harvest_alerted_at < QUALITY_HARVEST_STALE_SEC):
            return
        self._quality_harvest_alerted_at = now

        self._alert(
            "warning", "quality_harvest_stale",
            f"Quality awareness data is stale or absent (>{QUALITY_HARVEST_STALE_SEC // 3600}h) — "
            "triggering background harvest so generation has live chart signal.",
            "Spawned quality_harvester.harvest() in background thread"
        )

        def _do_harvest():
            try:
                from workers.quality_harvester import harvest as _harvest  # noqa: PLC0415
                _harvest(replace=False)   # never overwrite a concurrent successful harvest
                logger.info("[Watchdog] Quality harvest complete.")
            except Exception as exc:
                logger.warning("[Watchdog] Quality harvest failed (non-fatal): %s", exc)

        threading.Thread(
            target=_do_harvest, daemon=True, name="WatchdogQualityHarvest"
        ).start()

    def _check_storage_health_thread(self, now: float):
        """Restart the StorageClient's periodic health-check thread if it died.

        When this thread is dead the client's ``is_available`` property can
        permanently return stale False even after pdim comes back online.
        """
        sc = self.storage_client_ref
        if sc is None:
            return
        check_thread = getattr(sc, "_check_thread", None)
        if check_thread is None or check_thread.is_alive():
            return
        self._alert(
            "critical", "storage_health_thread_dead",
            "StorageClient._check_thread died — availability flag may be stale forever. Restarting.",
            "Spawned a fresh _periodic_health_check daemon thread on the StorageClient"
        )
        try:
            new_thread = threading.Thread(
                target=sc._periodic_health_check,
                daemon=True,
                name="StorageHealthCheck",
            )
            sc._check_thread = new_thread
            new_thread.start()
        except Exception as exc:
            logger.error("[Watchdog] Could not restart storage health thread: %s", exc)

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
        """Detect memory pressure and free resources before the OS OOM-kills the process."""
        try:
            mem_mb, mem_pct = _get_memory_info()
        except Exception:
            return

        if mem_pct >= MEMORY_DANGER_PCT:
            # Extreme pressure — free everything we safely can
            collected = gc.collect(2)
            _free_torch_cache()
            self._alert(
                "critical", "memory_danger",
                f"Memory at {mem_pct:.1f}% ({mem_mb:.0f} MB) — DANGER. "
                f"Ran full GC ({collected} objects freed) + cleared all caches.",
                "gc.collect(2) + torch/malloc cache cleared"
            )
        elif mem_pct >= MEMORY_CRIT_PCT:
            collected = gc.collect()
            _free_torch_cache()
            self._alert(
                "critical", "memory_pressure",
                f"Memory at {mem_pct:.1f}% ({mem_mb:.0f} MB) — running GC + cache clear.",
                f"gc.collect() freed {collected} objects; torch cache cleared"
            )
        elif mem_pct >= MEMORY_WARN_PCT:
            # Early warning — light GC to stay ahead of pressure
            gc.collect(0)
            self._alert(
                "warning", "memory_high",
                f"Memory at {mem_pct:.1f}% ({mem_mb:.0f} MB) — ran light GC.",
                "gc.collect(0)"
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


def _free_torch_cache():
    """Free PyTorch allocator caches without touching model weights."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        # Free the memory allocator's cached pages back to the OS
        if hasattr(torch, "cuda") and hasattr(torch.cuda, "memory"):
            pass  # Digital GPU path: rely on Python GC + ctypes malloc_trim
        try:
            import ctypes
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except Exception:
            pass
    except Exception:
        pass


# ------------------------------------------------------------------ #
# Singleton                                                           #
# ------------------------------------------------------------------ #

_watchdog: Optional[Watchdog] = None


def get_watchdog() -> Watchdog:
    global _watchdog
    if _watchdog is None:
        _watchdog = Watchdog()
    return _watchdog
