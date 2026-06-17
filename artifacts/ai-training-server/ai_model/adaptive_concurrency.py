"""Adaptive concurrency gates that auto-size to the deployment's resources.

The deployed AI server runs a single Python worker that holds a large in-memory
transformer model. When the consuming platform (MaxBooster) bursts many requests
at once, the old code accepted them all unconditionally: every video request span
its own render threads and every inference request contended for CPU, so the
server thrashed and requests timed out en masse.

These gates bound how much heavy work runs concurrently and *auto-size that bound
from the container's actual CPU and available memory* — re-checked periodically so
the limit shrinks under pressure and grows again when resources free up. Excess
work waits (queues) on ``acquire`` instead of all starting at once, so a burst
degrades gracefully (slower) rather than failing (crashes/timeouts).

Pure stdlib — no psutil dependency. Memory is read cgroup-v2-aware so the limit
reflects the container's real allowance, not the host's.
"""
from __future__ import annotations
import os
import math
import time
import threading


def _cgroup_v2_available_bytes() -> int | None:
    """Available bytes within the container's cgroup v2 memory allowance."""
    try:
        with open("/sys/fs/cgroup/memory.max") as f:
            raw = f.read().strip()
        if raw == "max":
            return None
        limit = int(raw)
        with open("/sys/fs/cgroup/memory.current") as f:
            current = int(f.read().strip())
        return max(0, limit - current)
    except (OSError, ValueError):
        return None


def _proc_meminfo_available_bytes() -> int | None:
    """Host-reported MemAvailable in bytes."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return None


def available_memory_bytes() -> int:
    """Best estimate of currently-available memory, cgroup-aware.

    Uses the *minimum* of the cgroup allowance and host MemAvailable so the
    tighter of the two constraints wins. Falls back to a conservative 2 GiB.
    """
    candidates = [
        b for b in (_cgroup_v2_available_bytes(), _proc_meminfo_available_bytes())
        if b is not None
    ]
    if not candidates:
        return 2 * 1024 ** 3
    return min(candidates)


def usable_cpu_count() -> int:
    """CPUs actually usable by this process (respects scheduler affinity)."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


class GateBusy(Exception):
    """Raised when a slot could not be acquired within the timeout."""


class AdaptiveGate:
    """A concurrency limiter whose capacity auto-sizes to available resources."""

    def __init__(
        self,
        name: str,
        mem_per_slot_gb: float,
        cpu_per_slot: float = 1.0,
        min_capacity: int = 1,
        max_capacity: int = 8,
        refresh_interval: float = 3.0,
    ) -> None:
        self.name = name
        self.mem_per_slot_gb = float(mem_per_slot_gb)
        self.cpu_per_slot = float(cpu_per_slot)
        self.min_capacity = int(min_capacity)
        self.max_capacity = int(max_capacity)
        self.refresh_interval = float(refresh_interval)

        self._cond = threading.Condition()
        self._active = 0
        self._peak_active = 0
        self._capacity = self.min_capacity
        self._last_refresh = 0.0
        with self._cond:
            self._refresh(force=True)

    def _compute_capacity(self) -> int:
        cpu = usable_cpu_count()
        mem_gb = available_memory_bytes() / (1024 ** 3)
        by_cpu = cpu / self.cpu_per_slot
        by_mem = mem_gb / self.mem_per_slot_gb
        cap = int(math.floor(min(by_cpu, by_mem)))
        return max(self.min_capacity, min(self.max_capacity, cap))

    def _refresh(self, force: bool = False) -> None:
        """Recompute capacity if stale. Must hold ``self._cond``."""
        now = time.monotonic()
        if not force and (now - self._last_refresh) < self.refresh_interval:
            return
        self._last_refresh = now
        previous = self._capacity
        self._capacity = self._compute_capacity()
        if self._capacity > previous:
            self._cond.notify_all()

    @property
    def capacity(self) -> int:
        with self._cond:
            self._refresh()
            return self._capacity

    def acquire(self, timeout: float | None = None) -> bool:
        """Reserve a slot, waiting if the gate is full. Returns False on timeout."""
        deadline = None if timeout is None else (time.monotonic() + timeout)
        with self._cond:
            while True:
                self._refresh()
                if self._active < self._capacity:
                    self._active += 1
                    self._peak_active = max(self._peak_active, self._active)
                    return True
                if deadline is None:
                    wait_for = self.refresh_interval
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                    # Bounded by refresh_interval so capacity growth from freed
                    # memory (even without a release) is re-checked promptly.
                    wait_for = min(remaining, self.refresh_interval)
                self._cond.wait(timeout=wait_for)

    def release(self) -> None:
        with self._cond:
            if self._active > 0:
                self._active -= 1
            self._cond.notify()

    def slot(self, timeout: float | None = None) -> "_GateSlot":
        return _GateSlot(self, timeout)

    def stats(self) -> dict:
        with self._cond:
            self._refresh()
            return {
                "name": self.name,
                "active": self._active,
                "capacity": self._capacity,
                "peak_active": self._peak_active,
                "min_capacity": self.min_capacity,
                "max_capacity": self.max_capacity,
                "usable_cpu": usable_cpu_count(),
                "available_mem_gb": round(available_memory_bytes() / (1024 ** 3), 2),
            }


class _GateSlot:
    def __init__(self, gate: AdaptiveGate, timeout: float | None) -> None:
        self._gate = gate
        self._timeout = timeout

    def __enter__(self) -> AdaptiveGate:
        if not self._gate.acquire(timeout=self._timeout):
            raise GateBusy(
                f"{self._gate.name} gate at capacity (no slot within {self._timeout}s)"
            )
        return self._gate

    def __exit__(self, *exc_info: object) -> None:
        self._gate.release()


# ── Global gates ────────────────────────────────────────────────────────────
# Model inference (captions/hooks/CTAs/scripts): CPU-bound, shares the loaded
# model so marginal memory per call is modest.
INFERENCE_GATE = AdaptiveGate(
    name="inference",
    mem_per_slot_gb=0.5,
    cpu_per_slot=1.5,
    min_capacity=1,
    max_capacity=6,
)

# Video scene rendering: each scene spawns an ffmpeg encode that is itself
# multi-threaded and CPU-hungry, so budget ~2 cores per concurrent render to
# avoid oversubscribing the CPU (which slows every encode and trips timeouts).
# Dev (4 vCPU) -> 2 concurrent; production Reserved VM (8 vCPU) -> 4 concurrent.
RENDER_GATE = AdaptiveGate(
    name="render",
    mem_per_slot_gb=0.5,
    cpu_per_slot=2.0,
    min_capacity=1,
    max_capacity=6,
)
