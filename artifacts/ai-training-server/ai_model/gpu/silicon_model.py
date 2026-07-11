"""Digital silicon *performance model* — a virtual-die planner, not an executor.

IMPORTANT — read before trusting any number this module produces.

This is an **architectural performance model** (the same class of tool as gem5,
Accel-Sim, or a roofline calculator). It *estimates* how many cycles / how much
time an operation would take on a hypothetical chip described by constants you
choose (tile count, per-tile FLOPs/cycle, memory bandwidth, clock). It does
**not** perform the arithmetic and it does **not** make anything run faster —
the real math still executes in the numpy/torch engine at the speed of whatever
physical hardware this process runs on.

Consequently, every value it returns is a *prediction*, prefixed ``estimated_``
and marked ``is_measurement = False``. These numbers MUST NEVER be presented as
measured throughput, and MUST NEVER be mixed with real wall-clock timings. A
simulated cycle count divided by a clock constant you picked is not a benchmark.

What it is genuinely good for:
  * capacity / what-if planning ("how would this workload behave on a die with N
    tiles and X TB/s?") before committing to real silicon;
  * exploring tile counts, specialization, and memory hierarchy trade-offs;
  * stress-testing scheduler / routing behavior under modeled contention.

The model attaches to DigitalGPU / HyperGPU as an optional telemetry layer: each
executed op is *also* recorded here to accumulate an estimated cycle budget,
alongside — never instead of — the real computation.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Marks every result of this module: a modeled estimate, never a measurement.
MODE = "performance_model"


@dataclass
class SRAM:
    """Modeled on-tile scratchpad (capacity bookkeeping only)."""
    size_bytes: int
    used_bytes: int = 0


@dataclass
class RegisterFile:
    registers: Dict[str, int] = field(default_factory=dict)


@dataclass
class KVCacheSlice:
    entries: Dict[str, int] = field(default_factory=dict)


class GlobalMemory:
    """Modeled HBM-like pool: a bandwidth constant + a transfer estimator.

    ``bandwidth_bytes_per_sec`` is a *modeling assumption*, not a measurement of
    this host's real memory bandwidth.
    """

    def __init__(self, bandwidth_bytes_per_sec: float):
        self.bandwidth = float(bandwidth_bytes_per_sec)
        self.estimated_bytes_moved: float = 0.0

    def request_transfer(self, size_bytes: float) -> float:
        """Record a modeled transfer; return its estimated seconds."""
        size_bytes = max(0.0, float(size_bytes))
        self.estimated_bytes_moved += size_bytes
        return size_bytes / self.bandwidth if self.bandwidth > 0 else 0.0

    def report(self) -> dict:
        return {
            "bandwidth_bytes_per_sec": self.bandwidth,
            "estimated_bytes_moved": self.estimated_bytes_moved,
            "estimated_transfer_seconds": (
                self.estimated_bytes_moved / self.bandwidth if self.bandwidth > 0 else 0.0
            ),
            "is_measurement": False,
        }


@dataclass
class MaxCoreOp:
    """A unit of work to be *modeled* (never executed here)."""
    kind: str
    precision: str = "fp16"
    estimated_flops: float = 0.0
    kv_size: float = 0.0
    bytes_moved: float = 0.0
    assigned_tile: Optional[int] = None
    cycles_remaining: Optional[int] = None


class ComputeTile:
    """Modeled compute block (an SM/tile). Accumulates an estimated cycle budget.

    Note: ``step`` / ``queue`` provide a cycle-accurate-style drain simulation for
    what-if runs; the telemetry path uses ``add_estimated_cycles`` directly.
    """

    #: modeled peak throughput of this tile — a chosen constant, not measured.
    flops_per_cycle: float = 1.0e12

    def __init__(self, tile_id: int, sram_size_bytes: int):
        self.id = tile_id
        self.sram = SRAM(sram_size_bytes)
        self.register_file = RegisterFile()
        self.kv_cache = KVCacheSlice()
        self.queue: List[MaxCoreOp] = []
        self.estimated_busy_cycles: int = 0
        self.ops_modeled: int = 0

    def estimate_cycles(self, op: MaxCoreOp) -> int:
        return max(1, int(op.estimated_flops / self.flops_per_cycle))

    # --- telemetry path (accumulate) ---
    def add_estimated_cycles(self, cycles: int) -> None:
        self.estimated_busy_cycles += max(0, int(cycles))
        self.ops_modeled += 1

    # --- what-if drain simulation ---
    def submit_op(self, op: MaxCoreOp) -> None:
        if op.cycles_remaining is None:
            op.cycles_remaining = self.estimate_cycles(op)
        self.queue.append(op)

    def step(self) -> None:
        if not self.queue:
            return
        current = self.queue[0]
        assert current.cycles_remaining is not None
        current.cycles_remaining -= 1
        self.estimated_busy_cycles += 1
        if current.cycles_remaining <= 0:
            self.finish_op(current)
            self.queue.pop(0)

    def finish_op(self, op: MaxCoreOp) -> None:
        # Modeling only — no outputs are written, no math is performed.
        self.ops_modeled += 1

    @property
    def pending(self) -> int:
        return len(self.queue)

    def report(self) -> dict:
        return {
            "tile_id": self.id,
            "type": type(self).__name__,
            "flops_per_cycle": self.flops_per_cycle,
            "estimated_busy_cycles": self.estimated_busy_cycles,
            "ops_modeled": self.ops_modeled,
            "pending": self.pending,
        }


class GemmTile(ComputeTile):
    flops_per_cycle: float = 1.0e12


class AttentionTile(ComputeTile):
    flops_per_cycle: float = 1.0e12
    kv_bandwidth_per_cycle: float = 1.0e9

    def estimate_cycles(self, op: MaxCoreOp) -> int:
        base = op.estimated_flops / self.flops_per_cycle
        kv_penalty = op.kv_size / self.kv_bandwidth_per_cycle if self.kv_bandwidth_per_cycle > 0 else 0.0
        return max(1, int(base + kv_penalty))


class SiliconScheduler:
    """Hardware-style dispatcher: maps modeled ops onto the least-loaded
    compatible tile and estimates their cycle cost. Pure modeling."""

    def __init__(self, tiles: List[ComputeTile], global_mem: GlobalMemory):
        self.tiles = tiles
        self.global_mem = global_mem

    def _supports(self, tile: ComputeTile, op: MaxCoreOp) -> bool:
        if op.kind == "attention":
            return isinstance(tile, AttentionTile) or not any(
                isinstance(t, AttentionTile) for t in self.tiles
            )
        if op.kind in ("gemm", "conv", "mlp"):
            return isinstance(tile, GemmTile) or not any(
                isinstance(t, GemmTile) for t in self.tiles
            )
        return True

    def pick_tile(self, op: MaxCoreOp) -> ComputeTile:
        candidates = [t for t in self.tiles if self._supports(t, op)] or self.tiles
        return min(candidates, key=lambda t: t.estimated_busy_cycles)

    def estimate_cycles(self, op: MaxCoreOp, tile: ComputeTile) -> int:
        return tile.estimate_cycles(op)

    def model(self, op: MaxCoreOp) -> int:
        """Telemetry path: assign, estimate, accumulate. Returns estimated cycles."""
        tile = self.pick_tile(op)
        cycles = self.estimate_cycles(op, tile)
        op.assigned_tile = tile.id
        op.cycles_remaining = cycles
        tile.add_estimated_cycles(cycles)
        if op.bytes_moved or op.kv_size:
            self.global_mem.request_transfer(op.bytes_moved + op.kv_size)
        return cycles

    def schedule(self, ops: List[MaxCoreOp]) -> None:
        """What-if path: queue ops onto tiles for a drain simulation."""
        for op in ops:
            tile = self.pick_tile(op)
            op.assigned_tile = tile.id
            op.cycles_remaining = self.estimate_cycles(op, tile)
            tile.submit_op(op)
            if op.bytes_moved or op.kv_size:
                self.global_mem.request_transfer(op.bytes_moved + op.kv_size)


class MaxCoreSilicon:
    """A virtual die: tiles + global memory + a scheduler.

    Attach it to a DigitalGPU/HyperGPU to accumulate an estimated cycle/time
    budget as real ops execute, or drive it directly with ``simulate(ops)`` for a
    standalone what-if run. All outputs are estimates (``is_measurement=False``).
    """

    def __init__(
        self,
        num_tiles: int = 64,
        sram_per_tile: int = 512 * 1024,
        global_mem_bandwidth: float = 8e12,
        clock_hz: float = 1.5e9,
        num_attention_tiles: int = 0,
    ):
        if num_tiles < 1:
            raise ValueError("num_tiles must be >= 1")
        tiles: List[ComputeTile] = []
        for i in range(num_tiles):
            if i < num_attention_tiles:
                tiles.append(AttentionTile(i, sram_per_tile))
            else:
                tiles.append(GemmTile(i, sram_per_tile))
        self.tiles = tiles
        self.global_mem = GlobalMemory(global_mem_bandwidth)
        self.clock_hz = float(clock_hz)
        self.scheduler = SiliconScheduler(self.tiles, self.global_mem)
        self._lock = threading.Lock()
        self.total_ops_modeled = 0
        self.total_estimated_flops = 0.0

    # --- telemetry path used by the live engines ---
    def model_op(
        self, kind: str, estimated_flops: float,
        kv_size: float = 0.0, bytes_moved: float = 0.0, precision: str = "fp16",
    ) -> int:
        """Record one executed op as a modeled estimate. Returns estimated cycles.

        This performs NO computation — the real op already ran in the engine.
        """
        op = MaxCoreOp(
            kind=kind, precision=precision,
            estimated_flops=max(0.0, float(estimated_flops)),
            kv_size=max(0.0, float(kv_size)),
            bytes_moved=max(0.0, float(bytes_moved)),
        )
        with self._lock:
            cycles = self.scheduler.model(op)
            self.total_ops_modeled += 1
            self.total_estimated_flops += op.estimated_flops
        return cycles

    # --- what-if drain simulation ---
    def submit_ops(self, ops: List[MaxCoreOp]) -> None:
        self.scheduler.schedule(ops)

    def step(self) -> None:
        for tile in self.tiles:
            tile.step()

    def simulate(self, ops: List[MaxCoreOp], max_steps: int = 10_000_000) -> dict:
        """Schedule ``ops`` and advance modeled cycles until all tiles drain.

        Returns an estimate report. The wall-clock cost of running THIS function
        is real CPU time and has nothing to do with ``estimated_seconds``.
        """
        self.submit_ops(ops)
        steps = 0
        while any(t.pending for t in self.tiles) and steps < max_steps:
            self.step()
            steps += 1
        return self.report(critical_path_cycles=steps)

    @property
    def critical_path_cycles(self) -> int:
        """Max busy cycles across tiles = modeled wall-time in cycles (parallel)."""
        return max((t.estimated_busy_cycles for t in self.tiles), default=0)

    def report(self, critical_path_cycles: Optional[int] = None) -> dict:
        with self._lock:
            return self._report_locked(critical_path_cycles)

    def _report_locked(self, critical_path_cycles: Optional[int] = None) -> dict:
        cp = self.critical_path_cycles if critical_path_cycles is None else critical_path_cycles
        total_cycles = sum(t.estimated_busy_cycles for t in self.tiles)
        return {
            "mode": MODE,
            "is_measurement": False,
            "disclaimer": (
                "Estimated from modeling constants, NOT measured. Real math runs "
                "at this host's actual speed; these numbers are a hardware "
                "what-if, never a benchmark."
            ),
            "num_tiles": len(self.tiles),
            "clock_hz": self.clock_hz,
            "total_ops_modeled": self.total_ops_modeled,
            "total_estimated_flops": self.total_estimated_flops,
            "estimated_total_tile_cycles": total_cycles,
            "estimated_critical_path_cycles": cp,
            "estimated_seconds": cp / self.clock_hz if self.clock_hz > 0 else 0.0,
            "global_memory": self.global_mem.report(),
            "tiles": [t.report() for t in self.tiles],
        }


def make_default_silicon() -> MaxCoreSilicon:
    """A plausible modeled die for quick attachment. Constants are assumptions."""
    return MaxCoreSilicon(
        num_tiles=64,
        sram_per_tile=512 * 1024,
        global_mem_bandwidth=8e12,
        clock_hz=1.5e9,
        num_attention_tiles=16,
    )
