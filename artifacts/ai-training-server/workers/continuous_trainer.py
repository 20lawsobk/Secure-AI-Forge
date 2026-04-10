"""
MaxBooster AI Training Server — 24/7 Continuous Trainer

Runs an infinite pull → train → checkpoint → sleep loop.
Each cycle:
  1. DataPuller fetches fresh data from pdim + public sources
  2. Merges with existing local training data
  3. Runs one curriculum pass (all 5 phases or a configured subset)
  4. Saves checkpoint locally + pushes weights snapshot to pdim
  5. Sleeps for the configured interval, then repeats

State is persisted to pdim at mb:training:continuous:state so it survives restarts.
"""

import logging
import threading
import time
from typing import Optional, Callable

logger = logging.getLogger("continuous_trainer")

DEFAULT_TRAIN_INTERVAL_MINUTES = 60
DEFAULT_PULL_EVERY_N_CYCLES = 2
DEFAULT_EPOCHS_PER_PHASE = 1


class ContinuousTrainer:
    """24/7 training daemon. Integrates DataPuller + model training."""

    STATE_KEY = "mb:training:continuous:state"
    HISTORY_KEY = "mb:training:continuous:history"

    def __init__(self, storage, data_puller, run_training_fn: Callable, curriculum_phases: list):
        """
        Args:
            storage:           StorageClient instance
            data_puller:       DataPuller instance
            run_training_fn:   Callable(texts, epochs, phase_label) → dict with loss info
            curriculum_phases: List of phase dicts from CURRICULUM_PHASES
        """
        self.storage = storage
        self.data_puller = data_puller
        self.run_training = run_training_fn
        self.curriculum_phases = curriculum_phases

        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self.state = {
            "running": False,
            "status": "stopped",
            "cycle": 0,
            "total_samples_trained": 0,
            "total_training_time_s": 0,
            "current_phase": None,
            "last_cycle_at": None,
            "next_cycle_at": None,
            "last_loss": None,
            "best_loss": None,
            "sessions_completed": 0,
            "interval_minutes": DEFAULT_TRAIN_INTERVAL_MINUTES,
            "phases_enabled": [p["id"] for p in curriculum_phases],
            "started_at": None,
            "stop_reason": None,
        }

        self._restore_state()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def start(self, interval_minutes: int = DEFAULT_TRAIN_INTERVAL_MINUTES,
              phases: Optional[list] = None,
              epochs_per_phase: int = DEFAULT_EPOCHS_PER_PHASE,
              pull_every_n: int = DEFAULT_PULL_EVERY_N_CYCLES) -> dict:
        with self._lock:
            if self.state["running"]:
                return {
                    "success": False,
                    "message": "Continuous trainer already running",
                    "state": dict(self.state),
                }
            self._stop_event.clear()
            self.state["running"] = True
            self.state["status"] = "starting"
            self.state["interval_minutes"] = interval_minutes
            self.state["phases_enabled"] = phases or [p["id"] for p in self.curriculum_phases]
            self.state["started_at"] = time.time()
            self.state["stop_reason"] = None
            self._persist_state()

        self._thread = threading.Thread(
            target=self._loop,
            args=(interval_minutes, epochs_per_phase, pull_every_n),
            daemon=True,
            name="ContinuousTrainer",
        )
        self._thread.start()

        logger.info(
            f"[ContinuousTrainer] Started — interval={interval_minutes}m "
            f"phases={self.state['phases_enabled']}"
        )
        return {
            "success": True,
            "message": f"Continuous 24/7 training started (every {interval_minutes} min)",
            "state": dict(self.state),
        }

    def stop(self, reason: str = "user_requested") -> dict:
        with self._lock:
            if not self.state["running"]:
                return {"success": False, "message": "Continuous trainer is not running"}
            self._stop_event.set()
            self.state["running"] = False
            self.state["status"] = "stopping"
            self.state["stop_reason"] = reason
            self._persist_state()

        logger.info(f"[ContinuousTrainer] Stop requested: {reason}")
        return {"success": True, "message": "Continuous training stop signal sent"}

    def get_state(self) -> dict:
        with self._lock:
            return dict(self.state)

    # ------------------------------------------------------------------ #
    # Main loop                                                            #
    # ------------------------------------------------------------------ #

    def _loop(self, interval_minutes: int, epochs_per_phase: int, pull_every_n: int):
        logger.info("[ContinuousTrainer] Loop started")
        cycle = 0

        while not self._stop_event.is_set():
            cycle += 1
            with self._lock:
                self.state["cycle"] = cycle
                self.state["status"] = "running"
                self.state["last_cycle_at"] = time.time()

            try:
                logger.info(f"[ContinuousTrainer] === Cycle {cycle} ===")

                # Step 1: Pull fresh data every N cycles
                if cycle == 1 or (cycle % pull_every_n == 0):
                    logger.info("[ContinuousTrainer] Pulling fresh training data…")
                    with self._lock:
                        self.state["status"] = "pulling_data"
                    pull_result = self.data_puller.pull_now()
                    logger.info(f"[ContinuousTrainer] Pull done: {pull_result}")

                # Step 2: Collect training data
                texts = self.data_puller.get_local_samples(max_samples=1000)
                if not texts:
                    logger.warning("[ContinuousTrainer] No training data available, skipping cycle")
                    self._wait_interval(interval_minutes)
                    continue

                logger.info(f"[ContinuousTrainer] Training on {len(texts)} samples")

                # Step 3: Train through enabled phases
                enabled = self.state.get("phases_enabled", [])
                phases_to_run = [p for p in self.curriculum_phases if p["id"] in enabled]
                if not phases_to_run:
                    phases_to_run = self.curriculum_phases

                cycle_start = time.time()
                last_loss = None

                for phase in phases_to_run:
                    if self._stop_event.is_set():
                        break

                    with self._lock:
                        self.state["current_phase"] = phase.get("name", phase["id"])
                        self.state["status"] = "training"

                    logger.info(f"[ContinuousTrainer] Training phase: {phase['name']}")
                    try:
                        result = self.run_training(
                            texts=texts,
                            epochs=epochs_per_phase,
                            phase_label=phase["id"],
                            loss_target=phase.get("loss_target"),
                        )
                        if result and "loss" in result:
                            last_loss = result["loss"]
                            with self._lock:
                                self.state["last_loss"] = last_loss
                                if self.state["best_loss"] is None or last_loss < self.state["best_loss"]:
                                    self.state["best_loss"] = last_loss
                    except Exception as e:
                        logger.error(f"[ContinuousTrainer] Phase {phase['id']} error: {e}")

                cycle_time = time.time() - cycle_start
                samples_this_cycle = len(texts) * len(phases_to_run) * epochs_per_phase

                with self._lock:
                    self.state["sessions_completed"] += 1
                    self.state["total_samples_trained"] += samples_this_cycle
                    self.state["total_training_time_s"] += cycle_time
                    self.state["current_phase"] = None
                    self.state["status"] = "sleeping"
                    self._persist_state()

                # Record in history
                self._record_history(cycle, last_loss, samples_this_cycle, cycle_time)

                logger.info(
                    f"[ContinuousTrainer] Cycle {cycle} complete — "
                    f"loss={last_loss} samples={samples_this_cycle} time={cycle_time:.1f}s"
                )

            except Exception as e:
                logger.error(f"[ContinuousTrainer] Cycle {cycle} unexpected error: {e}")
                with self._lock:
                    self.state["status"] = "error"

            # Step 4: Sleep until next cycle
            if not self._stop_event.is_set():
                self._wait_interval(interval_minutes)

        # Cleanup
        with self._lock:
            self.state["running"] = False
            self.state["status"] = "stopped"
            self.state["current_phase"] = None
            self._persist_state()

        logger.info("[ContinuousTrainer] Loop ended")

    def _wait_interval(self, interval_minutes: int):
        """Sleep in 5-second ticks so stop signal is respected promptly."""
        wake_at = time.time() + interval_minutes * 60
        with self._lock:
            self.state["next_cycle_at"] = wake_at
            self.state["status"] = "sleeping"
        logger.info(f"[ContinuousTrainer] Sleeping {interval_minutes}m until next cycle…")
        while time.time() < wake_at:
            if self._stop_event.is_set():
                break
            time.sleep(5)

    # ------------------------------------------------------------------ #
    # State persistence                                                    #
    # ------------------------------------------------------------------ #

    def _persist_state(self):
        """Write current state to pdim (called under _lock)."""
        try:
            if self.storage.is_available:
                self.storage.set(self.STATE_KEY, self.state)
        except Exception as e:
            logger.debug(f"[ContinuousTrainer] persist failed: {e}")

    def _restore_state(self):
        """Try to recover state from pdim on startup."""
        try:
            if self.storage.is_available:
                saved = self.storage.get(self.STATE_KEY)
                if saved and isinstance(saved, dict):
                    # Don't restore running=True — that was a previous process
                    saved["running"] = False
                    saved["status"] = "stopped (restored)"
                    self.state.update({
                        k: v for k, v in saved.items()
                        if k in ("cycle", "total_samples_trained",
                                 "total_training_time_s", "best_loss",
                                 "sessions_completed", "interval_minutes",
                                 "phases_enabled")
                    })
                    logger.info(
                        f"[ContinuousTrainer] Restored state — "
                        f"cycle={self.state['cycle']} sessions={self.state['sessions_completed']}"
                    )
        except Exception as e:
            logger.debug(f"[ContinuousTrainer] restore failed: {e}")

    def _record_history(self, cycle: int, loss, samples: int, duration_s: float):
        record = {
            "cycle": cycle,
            "ts": time.time(),
            "loss": loss,
            "samples": samples,
            "duration_s": round(duration_s, 1),
        }
        try:
            if self.storage.is_available:
                self.storage.lpush(self.HISTORY_KEY, record)
                self.storage.ltrim(self.HISTORY_KEY, 0, 99)  # Keep last 100
        except Exception:
            pass
