"""PDIM worker — drains the durable job queue in a loop.

A worker repeatedly calls ``process_queue_once`` and sleeps when the queue is
idle. It is provided as a class you start explicitly; nothing here auto-starts a
thread, so importing this module never affects the running server.
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Optional

from .orchestrator import PDIMOrchestrator


class PDIMWorker:
    def __init__(self, orchestrator: PDIMOrchestrator, compute_fn: Callable[[dict], dict],
                 queue: str = "default",
                 preview_fn: Callable[[dict], dict] | None = None,
                 quality_fn: Callable[[dict, dict], bool] | None = None,
                 idle_sleep: float | None = None):
        self.orchestrator = orchestrator
        self.compute_fn = compute_fn
        self.queue = queue
        self.preview_fn = preview_fn
        self.quality_fn = quality_fn
        self.idle_sleep = idle_sleep if idle_sleep is not None else orchestrator.config.idle_sleep
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def run_once(self) -> int:
        return self.orchestrator.process_queue_once(
            self.compute_fn, queue=self.queue,
            preview_fn=self.preview_fn, quality_fn=self.quality_fn,
        )

    def run_forever(self, stop_event: threading.Event | None = None) -> None:
        stop = stop_event or self._stop
        while not stop.is_set():
            processed = self.run_once()
            if processed == 0:
                time.sleep(self.idle_sleep)

    def start(self) -> threading.Thread:
        if self._thread and self._thread.is_alive():
            return self._thread
        self._stop.clear()
        self._thread = threading.Thread(target=self.run_forever, daemon=True)
        self._thread.start()
        return self._thread

    def stop(self) -> None:
        self._stop.set()
