from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from ai_model.gpu.digital_gpu import DigitalGPU, Program, Instruction, OpCode
from ai_model.gpu.multi_stream import (
    MultiStreamGPU, GPUStream, VRAMPartition, StreamState
)
from ai_model.gpu.torch_backend import (
    _DigitalGEMM, _DigitalAttention, _DigitalSoftmax,
    _DigitalGEMMBiasReLU,
    DigitalGPULinear, DigitalGPUAttention, DigitalGPUSoftmax,
)


class StreamContext:
    def __init__(self, stream: GPUStream, multi_gpu: MultiStreamGPU):
        self.stream = stream
        self.multi_gpu = multi_gpu
        self._gpu_facade = _StreamGPUFacade(stream)

    @property
    def stream_id(self) -> int:
        return self.stream.stream_id

    @property
    def gpu(self):
        return self._gpu_facade

    def flush_vram(self):
        self.stream.flush_vram()

    def status(self) -> dict:
        return self.stream.status()

    def profile(self):
        return self.stream.last_profile


class _StreamGPUFacade:
    def __init__(self, stream: GPUStream):
        self._stream = stream
        self._vram = stream.partition.internal_vram
        self._core = stream._scheduler.core

    @property
    def vram(self):
        return self._vram

    @property
    def core(self):
        return self._core

    def gemm(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        hA = self._stream.partition.alloc(A)
        hB = self._stream.partition.alloc(B)
        hOut = self._stream.partition.alloc(
            np.zeros((A.shape[0], B.shape[1]), dtype=A.dtype)
        )
        prog = Program()
        prog.add(Instruction(OpCode.GEMM, {"a": hA, "b": hB, "out": hOut}))
        self._stream.submit(prog)
        return self._stream.partition.get(hOut)

    def add(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        hA = self._stream.partition.alloc(A)
        hB = self._stream.partition.alloc(B)
        hOut = self._stream.partition.alloc(np.zeros_like(A))
        prog = Program()
        prog.add(Instruction(OpCode.ADD, {"a": hA, "b": hB, "out": hOut}))
        self._stream.submit(prog)
        return self._stream.partition.get(hOut)

    def softmax(self, X: np.ndarray, axis: int = -1) -> np.ndarray:
        hX = self._stream.partition.alloc(X)
        hOut = self._stream.partition.alloc(np.zeros_like(X))
        prog = Program()
        prog.add(Instruction(OpCode.SOFTMAX, {"x": hX, "out": hOut, "axis": axis}))
        self._stream.submit(prog)
        return self._stream.partition.get(hOut)

    def attention(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                  causal: bool = False) -> np.ndarray:
        hQ = self._stream.partition.alloc(Q)
        hK = self._stream.partition.alloc(K)
        hV = self._stream.partition.alloc(V)
        hOut = self._stream.partition.alloc(np.zeros_like(Q))
        prog = Program()
        prog.add(Instruction(OpCode.ATTENTION, {
            "q": hQ, "k": hK, "v": hV, "out": hOut, "causal": causal
        }))
        self._stream.submit(prog)
        return self._stream.partition.get(hOut)

    def gemm_bias_relu(self, A: np.ndarray, B: np.ndarray,
                       bias: np.ndarray) -> np.ndarray:
        hA = self._stream.partition.alloc(A)
        hB = self._stream.partition.alloc(B)
        hBias = self._stream.partition.alloc(bias)
        hOut = self._stream.partition.alloc(
            np.zeros((A.shape[0], B.shape[1]), dtype=A.dtype)
        )
        prog = Program()
        prog.add(Instruction(OpCode.GEMM_BIAS_RELU, {
            "a": hA, "b": hB, "bias": hBias, "out": hOut
        }))
        self._stream.submit(prog)
        return self._stream.partition.get(hOut)

    def last_profile(self):
        return self._stream.last_profile


class StreamBackend:
    def __init__(self, context: StreamContext):
        self.context = context
        self.gpu = context.gpu

    def linear(self, in_features, out_features, bias=True, fused_relu=False):
        return DigitalGPULinear(
            in_features, out_features, self.gpu,
            bias=bias, fused_relu=fused_relu
        )

    def attention(self, dim, n_heads):
        return DigitalGPUAttention(dim, n_heads, self.gpu)

    def softmax(self, dim=-1):
        return DigitalGPUSoftmax(self.gpu, dim=dim)

    def gemm(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return _DigitalGEMM.apply(A, B, self.gpu)

    def flush_vram(self):
        self.context.flush_vram()

    def profile(self):
        return self.context.profile()

    def status(self):
        st = self.context.status()
        return {
            "stream_id": st["stream_id"],
            "state": st["state"],
            "lanes": self.gpu._core.lanes,
            "tile_size": f"{self.gpu._core.tile_m}x{self.gpu._core.tile_n}x{self.gpu._core.tile_k}",
            "vram_handles": st["vram_handles"],
            "vram_bytes": st["vram_bytes"],
            "vram_mb": round(st["vram_bytes"] / (1024 * 1024), 2),
            "vram_peak_mb": round(st["vram_peak_bytes"] / (1024 * 1024), 2),
        }


class MultiStreamBackend:
    def __init__(self, total_lanes: int = 32, default_vram_quota: int = 0):
        self.multi_gpu = MultiStreamGPU(
            total_lanes=total_lanes,
            default_vram_quota=default_vram_quota,
        )
        self._stream_backends: dict[int, StreamBackend] = {}

    def create_stream(
        self,
        name: str = "",
        lanes: int = 0,
        vram_quota: int = 0,
    ) -> StreamBackend:
        stream = self.multi_gpu.create_stream(
            lanes=lanes, vram_quota=vram_quota, stream_name=name
        )
        ctx = StreamContext(stream, self.multi_gpu)
        backend = StreamBackend(ctx)
        self._stream_backends[stream.stream_id] = backend
        return backend

    def destroy_stream(self, stream_id: int):
        self._stream_backends.pop(stream_id, None)
        self.multi_gpu.destroy_stream(stream_id)

    def get_backend(self, stream_id: int) -> StreamBackend:
        if stream_id not in self._stream_backends:
            raise ValueError(f"Stream backend {stream_id} not found")
        return self._stream_backends[stream_id]

    def rebalance(self):
        self.multi_gpu.rebalance_lanes()

    def flush_all(self):
        self.multi_gpu.flush_all()

    def status(self) -> dict:
        gpu_status = self.multi_gpu.status()
        per_stream = {}
        for sid, sb in self._stream_backends.items():
            per_stream[sid] = sb.status()
        gpu_status["stream_backends"] = per_stream
        return gpu_status

    def profile_all(self) -> dict:
        profiles = self.multi_gpu.profile_all()
        return {
            sid: {
                "instructions_executed": p.instructions_executed,
                "total_time_ms": round(p.total_time_ms, 2),
                "op_breakdown": p.op_breakdown,
                "vram_peak_bytes": p.vram_peak_bytes,
                "vram_peak_mb": round(p.vram_peak_bytes / (1024 * 1024), 2),
            }
            for sid, p in profiles.items()
        }
