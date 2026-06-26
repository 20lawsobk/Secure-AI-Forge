"""Lightweight, thread-safe metrics for the DigitalGPU / MaxCore stack.

Records counters (events), gauges (point-in-time values) and timers
(count/total/avg/max latency). Intentionally dependency-free so it can be used
from any subsystem (compiler, runtime, PDIM) without importing heavy modules.
The single process-wide ``METRICS`` instance is the canonical registry; callers
read a JSON-able snapshot via ``METRICS.snapshot()``.
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, float] = {}
        self._gauges: dict[str, float] = {}
        self._timers: dict[str, tuple[int, float, float]] = {}

    def incr(self, name: str, n: float = 1) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + n

    def gauge(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = value

    def observe(self, name: str, ms: float) -> None:
        with self._lock:
            count, total, mx = self._timers.get(name, (0, 0.0, 0.0))
            self._timers[name] = (count + 1, total + ms, max(mx, ms))

    @contextmanager
    def timer(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.observe(name, (time.perf_counter() - t0) * 1000.0)

    def snapshot(self) -> dict:
        with self._lock:
            timers = {
                k: {
                    "count": c,
                    "total_ms": round(t, 4),
                    "avg_ms": round(t / c, 4) if c else 0.0,
                    "max_ms": round(m, 4),
                }
                for k, (c, t, m) in self._timers.items()
            }
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "timers": timers,
            }

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._timers.clear()


# Process-wide registry.
METRICS = Metrics()
