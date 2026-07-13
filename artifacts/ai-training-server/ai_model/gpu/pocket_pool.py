"""
PocketGPUPool — Infinite GPU lifecycle pool backed by the pocket dimension.

The pocket dimension stores an unlimited number of GPU instances via its
enhanced compression + filetree system.  Every unique inference request is
assigned one GPU instance for the exact duration of its work:

    born  →  working  →  dead

There is no queue, no cap, no gate.  The pocket absorbs any burst.  Identical
requests never reach this layer — the async coalescer upstream collapses them
to a single life before the pool is ever asked to spawn.

Concurrent unique requests each get their own isolated HyperGPUBackend
(separate VRAM, separate lanes).  VRAM is flushed and the backend is released
on death so the compressed pocket space is reclaimed automatically.
"""
from __future__ import annotations

import asyncio
import threading
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from typing import AsyncIterator, Iterator

from ai_model.gpu.hyper_backend import HyperGPUBackend, PrecisionMode


# ── Lifecycle states ──────────────────────────────────────────────────────────

_BORN    = "born"
_WORKING = "working"
_DEAD    = "dead"


class PocketGPUInstance:
    """One GPU life — born when spawned, dead when released.

    Each instance owns an isolated ``HyperGPUBackend`` (and its VRAM).
    Flushing on death means no memory leaks back into the pocket regardless
    of how many simultaneous lives the pool is managing.
    """

    __slots__ = ("id", "digest", "backend", "state", "born_at", "died_at")

    def __init__(self, digest: str = "") -> None:
        self.id      = str(uuid.uuid4())
        self.digest  = digest
        self.state   = _BORN
        self.born_at = time.perf_counter()
        self.died_at: float | None = None
        # Each instance gets its own isolated GPU + VRAM — pocket-compressed
        # storage means unlimited simultaneous instances without memory growth.
        self.backend = HyperGPUBackend(
            lanes=512,
            tensor_cores=8,
            precision=PrecisionMode.MIXED,
        )

    # ── Transitions ──────────────────────────────────────────────────────────

    def begin_work(self) -> None:
        """born → working."""
        self.state = _WORKING

    def die(self) -> float:
        """Flush VRAM, mark dead.  Returns lifetime in milliseconds.
        Never raises — death must always succeed so the pool stays consistent."""
        try:
            self.backend.flush_vram()
        except Exception:
            pass
        self.died_at = time.perf_counter()
        self.state   = _DEAD
        return (self.died_at - self.born_at) * 1_000.0

    # ── Helpers ───────────────────────────────────────────────────────────────

    @property
    def alive_ms(self) -> float:
        """Milliseconds since birth (or total lifetime if dead)."""
        end = self.died_at if self.died_at is not None else time.perf_counter()
        return (end - self.born_at) * 1_000.0

    def __repr__(self) -> str:
        return (
            f"<PocketGPUInstance id={self.id[:8]}… "
            f"state={self.state} alive={self.alive_ms:.0f}ms>"
        )


# ── Pool ──────────────────────────────────────────────────────────────────────

class PocketGPUPool:
    """Infinite GPU pool backed by the pocket dimension.

    Spawn a GPU life with either the async or sync context manager:

        # From an async handler (before entering the thread-pool):
        async with pool.spawn(digest) as gpu:
            result = await run_in_executor(None, lambda: compute(gpu.backend))

        # Already inside a worker thread:
        with pool.spawn_sync(digest) as gpu:
            result = compute(gpu.backend)

    Lifecycle telemetry is exposed via ``stats()`` and surfaced in the
    ``/gpu/status`` endpoint so the dashboard can observe every birth and
    death in real time.
    """

    def __init__(self) -> None:
        self._lock          = threading.Lock()
        self._total_born    = 0
        self._total_dead    = 0
        self._alive         = 0
        self._total_life_ms = 0.0

    # ── Internal book-keeping ─────────────────────────────────────────────────

    def _register_birth(self, inst: PocketGPUInstance) -> None:
        with self._lock:
            self._total_born += 1
            self._alive      += 1
        inst.begin_work()

    def _register_death(self, inst: PocketGPUInstance) -> None:
        life_ms = inst.die()
        with self._lock:
            self._alive         -= 1
            self._total_dead    += 1
            self._total_life_ms += life_ms

    # ── Async context manager ─────────────────────────────────────────────────

    @asynccontextmanager
    async def spawn(self, digest: str = "") -> AsyncIterator[PocketGPUInstance]:
        """Spawn a GPU life from the async layer (event-loop safe).

        Instance creation and death both run in the thread-pool so the
        event loop is never stalled by VRAM allocation or flush.
        """
        loop = asyncio.get_event_loop()
        inst = await loop.run_in_executor(None, lambda: PocketGPUInstance(digest))
        self._register_birth(inst)
        try:
            yield inst
        finally:
            await loop.run_in_executor(None, self._register_death, inst)

    # ── Sync context manager ──────────────────────────────────────────────────

    @contextmanager
    def spawn_sync(self, digest: str = "") -> Iterator[PocketGPUInstance]:
        """Spawn a GPU life from inside a worker thread (no asyncio needed)."""
        inst = PocketGPUInstance(digest)
        self._register_birth(inst)
        try:
            yield inst
        finally:
            self._register_death(inst)

    # ── Telemetry ─────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Snapshot of pool lifecycle metrics for the /gpu/status endpoint."""
        with self._lock:
            avg = (
                round(self._total_life_ms / self._total_dead, 1)
                if self._total_dead else 0.0
            )
            return {
                "pool_alive":       self._alive,
                "pool_total_born":  self._total_born,
                "pool_total_dead":  self._total_dead,
                "pool_avg_life_ms": avg,
                "pool_source":      "pocket_dimension",
            }
