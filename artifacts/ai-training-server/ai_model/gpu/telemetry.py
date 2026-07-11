"""Execution telemetry for the digital GPU.

Honesty split — the whole point of this module:
  * ``wall_ms`` is a REAL measurement (``is_wall_measured=True``): the actual time
    the op took on this CPU.
  * ``flops`` and ``bytes_moved`` are ANALYTIC (``is_flops_derived=True``): counts
    derived from shapes, not hardware counters. They are labelled as derived so a
    reader never mistakes a modelled FLOP count for a measured one.

This is deliberately distinct from :mod:`ai_model.gpu.silicon_model`, which
estimates a *hypothetical* cycle/time budget for imagined silicon. Telemetry here
is about what actually happened on this box.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class OpRecord:
    opcode: str
    numeric_profile: str
    wall_ms: float                 # measured
    flops: float                   # analytic (derived from shapes)
    bytes_moved: int               # analytic
    is_wall_measured: bool = True
    is_flops_derived: bool = True  # NOT measured — computed from shapes

    @property
    def gflops_per_s(self) -> float:
        """Throughput = derived FLOPs / measured time. Mixed provenance, so treat
        as indicative, not a benchmarked hardware number."""
        if self.wall_ms <= 0:
            return 0.0
        return (self.flops / 1e9) / (self.wall_ms / 1000.0)


class Telemetry:
    def __init__(self):
        self.ops: List[OpRecord] = []

    def record(self, opcode: str, numeric_profile: str, wall_ms: float,
               flops: float = 0.0, bytes_moved: int = 0) -> OpRecord:
        rec = OpRecord(
            opcode=opcode, numeric_profile=numeric_profile,
            wall_ms=float(wall_ms), flops=float(flops),
            bytes_moved=int(bytes_moved),
        )
        self.ops.append(rec)
        return rec

    def total_wall_ms(self) -> float:
        return sum(r.wall_ms for r in self.ops)

    def summary(self) -> dict:
        return {
            "ops": len(self.ops),
            "total_wall_ms": round(self.total_wall_ms(), 4),
            "total_flops_derived": sum(r.flops for r in self.ops),
            "total_bytes_moved": sum(r.bytes_moved for r in self.ops),
            "note": ("wall_ms is measured; flops/bytes are analytic (derived "
                     "from shapes), not hardware counters."),
            "by_opcode": self._by_opcode(),
        }

    def _by_opcode(self) -> dict:
        out: dict = {}
        for r in self.ops:
            b = out.setdefault(r.opcode, {"count": 0, "wall_ms": 0.0, "flops": 0.0})
            b["count"] += 1
            b["wall_ms"] = round(b["wall_ms"] + r.wall_ms, 4)
            b["flops"] += r.flops
        return out

    def reset(self) -> None:
        self.ops.clear()
