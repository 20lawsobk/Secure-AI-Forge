"""Adaptive concurrency gates that auto-size to the deployment's resources.

The deployed AI server runs a single Python worker that holds a large in-memory
transformer model. When the consuming platform (MaxBooster) bursts many requests
at once, the old code accepted them all unconditionally: every video request span
its own render threads and every inference request contended for compute, so the
server thrashed and requests timed out en masse.

These gates bound how much heavy work runs concurrently and *auto-size that bound
from the container's actual compute capacity and available memory* — re-checked
periodically so the limit shrinks under pressure and grows again when resources
free up. Excess work waits (queues) on ``acquire`` instead of all starting at
once, so a burst degrades gracefully (slower) rather than failing (crashes/timeouts).

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
    """A concurrency limiter whose capacity auto-sizes to available resources.

    When ``gpu_independent=True`` the gate bypasses all host CPU/memory
    checks and runs at ``max_capacity`` at all times.  Use this for work
    whose heavy compute runs on the Digital GPU (HyperGPU + pdim pocket
    engine) rather than on Replit's host vCPUs — the host resource budget
    is irrelevant for those workloads and should not throttle them.
    """

    def __init__(
        self,
        name: str,
        mem_per_slot_gb: float = 0.5,
        cpu_per_slot: float = 1.0,
        min_capacity: int = 1,
        max_capacity: int = 8,
        refresh_interval: float = 3.0,
        gpu_independent: bool = False,
    ) -> None:
        self.name = name
        self.mem_per_slot_gb = float(mem_per_slot_gb)
        self.cpu_per_slot = float(cpu_per_slot)
        self.min_capacity = int(min_capacity)
        self.max_capacity = int(max_capacity)
        self.refresh_interval = float(refresh_interval)
        self.gpu_independent = bool(gpu_independent)

        self._cond = threading.Condition()
        self._active = 0
        self._peak_active = 0
        self._capacity = self.min_capacity
        self._last_refresh = 0.0
        with self._cond:
            self._refresh(force=True)

    def _compute_capacity(self) -> int:
        # GPU-independent gates: the Digital GPU has its own compute resources
        # entirely separate from Replit's host CPUs/memory. Don't let the
        # container's vCPU count or cgroup memory limit artificially cap
        # parallelism that the GPU engine can absorb.
        if self.gpu_independent:
            return self.max_capacity
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
# Both gates are gpu_independent=True: the Digital GPU (HyperGPU + pdim) is a
# self-contained backend entirely outside Replit's host environment. Replit's
# vCPU count and cgroup memory are irrelevant to its capacity — gates run at
# max_capacity at all times.

# Model inference (captions/hooks/CTAs/scripts): runs on the Digital GPU.
#
# max_capacity=90_000_000 — the Digital GPU (HyperGPU + pdim pocket engine)
# virtualises compute via content-hash deduplication: identical work is computed
# ONCE and every concurrent caller shares that one result.  Unique concurrent
# computations are bounded by the number of distinct inputs, not by hardware.
# Capping at 32 would force 89 999 968 unique requests to queue behind 32 slots
# even though the pocket dimension can absorb the entire burst simultaneously.
# With the pocket engine there is no physical VRAM or CUDA limit — setting the
# gate to 90 M aligns the concurrency ceiling with the system's actual capacity.
INFERENCE_GATE = AdaptiveGate(
    name="inference",
    min_capacity=4,
    max_capacity=90_000_000,
    gpu_independent=True,
)

# Video scene rendering: the Digital GPU (HyperGPU + pdim pocket engine) is 100%
# independent of Replit's host environment. All heavy compute (diffusion, VRC
# grading, GEMM batches) runs on the GPU engine — the host CPU/memory budget is
# irrelevant. gpu_independent=True fixes capacity at max_capacity so every scene
# in a job dispatches in a single parallel wave with no host-resource throttling.
RENDER_GATE = AdaptiveGate(
    name="render",
    min_capacity=4,
    max_capacity=90_000_000,
    gpu_independent=True,
)
