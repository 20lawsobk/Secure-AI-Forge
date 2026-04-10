from __future__ import annotations
import threading
import numpy as np
from enum import Enum, auto
from typing import Dict, Optional
from dataclasses import dataclass, field
from ai_model.gpu.digital_gpu import (
    VRAM, SIMDCore, Scheduler,
    Program, GPUError
)


class StreamState(Enum):
    IDLE = auto()
    RUNNING = auto()
    PAUSED = auto()
    FINISHED = auto()
    ERROR = auto()


@dataclass
class StreamProfile:
    stream_id: int
    instructions_executed: int = 0
    total_time_ms: float = 0.0
    op_breakdown: Dict[str, float] = field(default_factory=dict)
    vram_peak_bytes: int = 0


class VRAMPartition:
    def __init__(self, partition_id: int, quota_bytes: int = 0):
        self.partition_id = partition_id
        self.quota_bytes = quota_bytes
        self._vram = VRAM()
        self._peak_bytes = 0
        self._lock = threading.Lock()

    def alloc(self, array: np.ndarray) -> int:
        with self._lock:
            current = self._current_bytes()
            incoming = array.nbytes
            if self.quota_bytes > 0 and (current + incoming) > self.quota_bytes:
                raise GPUError(
                    f"VRAMPartition {self.partition_id} OOM: "
                    f"{current + incoming} > quota {self.quota_bytes}"
                )
            hid = self._vram.alloc(array)
            new_total = self._current_bytes()
            if new_total > self._peak_bytes:
                self._peak_bytes = new_total
            return hid

    def get(self, hid: int) -> np.ndarray:
        with self._lock:
            return self._vram.get(hid)

    def meta(self, hid: int) -> dict:
        with self._lock:
            return self._vram.meta(hid)

    def free(self, hid: int):
        with self._lock:
            self._vram.free(hid)

    def flush(self):
        with self._lock:
            self._vram._store.clear()
            self._vram._meta.clear()

    def _current_bytes(self) -> int:
        return sum(a.nbytes for a in self._vram._store.values())

    @property
    def handle_count(self) -> int:
        return len(self._vram._store)

    @property
    def used_bytes(self) -> int:
        return self._current_bytes()

    @property
    def peak_bytes(self) -> int:
        return self._peak_bytes

    @property
    def internal_vram(self) -> VRAM:
        return self._vram


class LaneAllocator:
    def __init__(self, total_lanes: int = 32):
        self.total_lanes = total_lanes
        self._allocations: Dict[int, int] = {}
        self._lock = threading.Lock()

    def allocate(self, stream_id: int, requested: int = 0) -> int:
        with self._lock:
            used = sum(self._allocations.values())
            available = self.total_lanes - used
            if requested <= 0:
                granted = max(1, available // max(1, len(self._allocations) + 1))
            else:
                granted = min(requested, available)
            if granted < 1:
                granted = 1
            self._allocations[stream_id] = granted
            return granted

    def release(self, stream_id: int):
        with self._lock:
            self._allocations.pop(stream_id, None)

    def rebalance(self):
        with self._lock:
            n = len(self._allocations)
            if n == 0:
                return
            per_stream = max(1, self.total_lanes // n)
            remainder = self.total_lanes - (per_stream * n)
            for i, sid in enumerate(self._allocations):
                extra = 1 if i < remainder else 0
                self._allocations[sid] = per_stream + extra

    def get_lanes(self, stream_id: int) -> int:
        with self._lock:
            return self._allocations.get(stream_id, 0)

    def utilization(self) -> dict:
        with self._lock:
            used = sum(self._allocations.values())
            return {
                "total_lanes": self.total_lanes,
                "used_lanes": used,
                "free_lanes": self.total_lanes - used,
                "allocations": dict(self._allocations),
            }


class GPUStream:
    def __init__(self, stream_id: int, partition: VRAMPartition, core: SIMDCore):
        self.stream_id = stream_id
        self.partition = partition
        self.state = StreamState.IDLE
        self.profile = StreamProfile(stream_id=stream_id)
        self._scheduler = Scheduler(partition.internal_vram, core)
        self._lock = threading.Lock()
        self._error: Optional[str] = None

    def submit(self, program: Program):
        with self._lock:
            self.state = StreamState.RUNNING
        try:
            self._scheduler.run(program)
            for entry in self._scheduler.last_profile:
                self.profile.instructions_executed += 1
                self.profile.total_time_ms += entry["duration_ms"]
                op = entry["opcode"]
                self.profile.op_breakdown[op] = (
                    self.profile.op_breakdown.get(op, 0.0) + entry["duration_ms"]
                )
            peak = self.partition.peak_bytes
            if peak > self.profile.vram_peak_bytes:
                self.profile.vram_peak_bytes = peak
            with self._lock:
                self.state = StreamState.IDLE
        except Exception as e:
            with self._lock:
                self.state = StreamState.ERROR
                self._error = str(e)
            raise

    def flush_vram(self):
        self.partition.flush()

    def status(self) -> dict:
        return {
            "stream_id": self.stream_id,
            "state": self.state.name,
            "vram_handles": self.partition.handle_count,
            "vram_bytes": self.partition.used_bytes,
            "vram_peak_bytes": self.profile.vram_peak_bytes,
            "instructions_executed": self.profile.instructions_executed,
            "total_time_ms": round(self.profile.total_time_ms, 2),
            "error": self._error,
        }

    @property
    def last_profile(self):
        return self._scheduler.last_profile


class MultiStreamGPU:
    def __init__(self, total_lanes: int = 32, default_vram_quota: int = 0):
        self.total_lanes = total_lanes
        self.default_vram_quota = default_vram_quota
        self.lane_allocator = LaneAllocator(total_lanes)
        self._streams: Dict[int, GPUStream] = {}
        self._next_stream_id = 0
        self._lock = threading.Lock()
        self._shared_core = SIMDCore(lanes=total_lanes)

    def create_stream(
        self,
        lanes: int = 0,
        vram_quota: int = 0,
        stream_name: str = "",
    ) -> GPUStream:
        with self._lock:
            sid = self._next_stream_id
            self._next_stream_id += 1

        quota = vram_quota if vram_quota > 0 else self.default_vram_quota
        partition = VRAMPartition(partition_id=sid, quota_bytes=quota)

        granted_lanes = self.lane_allocator.allocate(sid, requested=lanes)
        core = SIMDCore(lanes=granted_lanes)

        stream = GPUStream(stream_id=sid, partition=partition, core=core)

        with self._lock:
            self._streams[sid] = stream

        self.rebalance_lanes()
        return stream

    def destroy_stream(self, stream_id: int):
        with self._lock:
            stream = self._streams.pop(stream_id, None)
        if stream:
            stream.flush_vram()
            self.lane_allocator.release(stream_id)
            self.rebalance_lanes()

    def get_stream(self, stream_id: int) -> GPUStream:
        with self._lock:
            if stream_id not in self._streams:
                raise GPUError(f"Stream {stream_id} not found")
            return self._streams[stream_id]

    def rebalance_lanes(self):
        self.lane_allocator.rebalance()
        with self._lock:
            for sid, stream in self._streams.items():
                new_lanes = self.lane_allocator.get_lanes(sid)
                stream._scheduler.core = SIMDCore(lanes=new_lanes)

    def run_concurrent(self, stream_programs: Dict[int, Program]):
        threads = []
        errors = {}

        def _run(sid, prog):
            try:
                stream = self.get_stream(sid)
                stream.submit(prog)
            except Exception as e:
                errors[sid] = str(e)

        for sid, prog in stream_programs.items():
            t = threading.Thread(target=_run, args=(sid, prog), daemon=True)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        if errors:
            raise GPUError(f"Concurrent execution errors: {errors}")

    def flush_all(self):
        with self._lock:
            for stream in self._streams.values():
                stream.flush_vram()

    def status(self) -> dict:
        with self._lock:
            stream_statuses = {
                sid: s.status() for sid, s in self._streams.items()
            }
        lane_util = self.lane_allocator.utilization()
        total_vram = sum(
            s["vram_bytes"] for s in stream_statuses.values()
        )
        return {
            "total_lanes": self.total_lanes,
            "lane_utilization": lane_util,
            "active_streams": len(stream_statuses),
            "streams": stream_statuses,
            "total_vram_bytes": total_vram,
            "total_vram_mb": round(total_vram / (1024 * 1024), 2),
        }

    def profile_all(self) -> Dict[int, StreamProfile]:
        with self._lock:
            return {sid: s.profile for sid, s in self._streams.items()}
