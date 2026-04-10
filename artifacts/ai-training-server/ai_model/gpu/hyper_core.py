from __future__ import annotations
import time
import threading
import numpy as np
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from ai_model.gpu.digital_gpu import (
    GPUError, ShapeError, SIMDCore, VRAM
)


class PrecisionMode(Enum):
    FP32 = auto()
    FP16 = auto()
    BF16 = auto()
    MIXED = auto()


class HyperOpCode(Enum):
    GEMM = auto()
    ADD = auto()
    SOFTMAX = auto()
    ATTENTION = auto()
    GEMM_BIAS_RELU = auto()
    FLASH_ATTENTION = auto()
    CONV2D = auto()
    CONV3D = auto()
    LAYER_NORM = auto()
    GELU = auto()
    SILU = auto()
    TRANSPOSE = auto()
    CONCAT = auto()
    BATCH_NORM = auto()
    RESIDUAL_ADD = auto()
    MIXED_PRECISION_GEMM = auto()
    TENSOR_CORE_GEMM = auto()
    FUSED_ATTENTION_NORM = auto()
    DEPTHWISE_CONV = auto()
    GROUPED_GEMM = auto()


@dataclass
class TensorCoreUnit:
    m_tile: int = 16
    n_tile: int = 16
    k_tile: int = 16
    throughput_multiplier: float = 8.0
    precision: PrecisionMode = PrecisionMode.MIXED
    ops_executed: int = 0
    total_flops: float = 0.0

    def matmul(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        M, K = A.shape
        K2, N = B.shape
        C = np.zeros((M, N), dtype=np.float32)
        for i in range(0, M, self.m_tile):
            ie = min(i + self.m_tile, M)
            for j in range(0, N, self.n_tile):
                je = min(j + self.n_tile, N)
                for k in range(0, K, self.k_tile):
                    ke = min(k + self.k_tile, K)
                    C[i:ie, j:je] += A[i:ie, k:ke] @ B[k:ke, j:je]
        self.ops_executed += 1
        self.total_flops += 2.0 * M * N * K
        return C

    def mixed_precision_matmul(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        A_half = A.astype(np.float16)
        B_half = B.astype(np.float16)
        M, K = A.shape
        K2, N = B.shape
        C = np.zeros((M, N), dtype=np.float32)
        for i in range(0, M, self.m_tile):
            ie = min(i + self.m_tile, M)
            for j in range(0, N, self.n_tile):
                je = min(j + self.n_tile, N)
                for k in range(0, K, self.k_tile):
                    ke = min(k + self.k_tile, K)
                    partial = A_half[i:ie, k:ke].astype(np.float32) @ B_half[k:ke, j:je].astype(np.float32)
                    C[i:ie, j:je] += partial
        self.ops_executed += 1
        self.total_flops += 2.0 * M * N * K * self.throughput_multiplier
        return C

    def status(self) -> dict:
        return {
            "tile_shape": f"{self.m_tile}x{self.n_tile}x{self.k_tile}",
            "throughput_multiplier": self.throughput_multiplier,
            "precision": self.precision.name,
            "ops_executed": self.ops_executed,
            "total_tflops": round(self.total_flops / 1e12, 4),
        }


@dataclass
class MemoryPool:
    pool_size_bytes: int = 512 * 1024 * 1024
    block_size: int = 4096
    _allocated: Dict[int, np.ndarray] = field(default_factory=dict)
    _free_list: List[int] = field(default_factory=list)
    _next_id: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _peak_bytes: int = 0
    _current_bytes: int = 0
    _alloc_count: int = 0
    _reuse_count: int = 0

    def alloc(self, shape: Tuple, dtype=np.float32) -> Tuple[int, np.ndarray]:
        with self._lock:
            nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
            arr = np.zeros(shape, dtype=dtype)
            mid = self._next_id
            self._next_id += 1
            self._allocated[mid] = arr
            self._current_bytes += nbytes
            self._alloc_count += 1
            if self._current_bytes > self._peak_bytes:
                self._peak_bytes = self._current_bytes
            return mid, arr

    def free(self, mid: int):
        with self._lock:
            if mid in self._allocated:
                nbytes = self._allocated[mid].nbytes
                self._current_bytes -= nbytes
                del self._allocated[mid]
                self._free_list.append(mid)

    def get(self, mid: int) -> np.ndarray:
        with self._lock:
            if mid not in self._allocated:
                raise GPUError(f"MemoryPool: invalid handle {mid}")
            return self._allocated[mid]

    def flush(self):
        with self._lock:
            total_freed = self._current_bytes
            self._allocated.clear()
            self._free_list.clear()
            self._current_bytes = 0
            return total_freed

    def status(self) -> dict:
        with self._lock:
            return {
                "pool_size_mb": round(self.pool_size_bytes / (1024 * 1024), 2),
                "current_mb": round(self._current_bytes / (1024 * 1024), 4),
                "peak_mb": round(self._peak_bytes / (1024 * 1024), 4),
                "active_allocations": len(self._allocated),
                "total_allocs": self._alloc_count,
                "reuse_count": self._reuse_count,
            }


class HyperSIMDCore(SIMDCore):
    def __init__(
        self,
        lanes: int = 512,
        tile_m: int = 128,
        tile_n: int = 128,
        tile_k: int = 128,
        tensor_cores: int = 8,
        precision: PrecisionMode = PrecisionMode.MIXED,
    ):
        super().__init__(lanes=lanes, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k)
        self.precision = precision
        self.tensor_core_units = [
            TensorCoreUnit(
                m_tile=16, n_tile=16, k_tile=16,
                throughput_multiplier=8.0,
                precision=precision,
            )
            for _ in range(tensor_cores)
        ]
        self._mem_pool = MemoryPool()
        self._profile_data: Dict[str, float] = {}
        self._total_ops = 0

    def _select_tensor_core(self) -> TensorCoreUnit:
        return min(self.tensor_core_units, key=lambda tc: tc.ops_executed)

    def tensor_core_gemm(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        if A.ndim != 2 or B.ndim != 2:
            raise ShapeError("tensor_core_gemm expects 2D matrices")
        if A.shape[1] != B.shape[0]:
            raise ShapeError(f"Incompatible shapes: {A.shape} x {B.shape}")
        tc = self._select_tensor_core()
        self._total_ops += 1
        return tc.matmul(A.astype(np.float32), B.astype(np.float32))

    def mixed_precision_gemm(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        if A.ndim != 2 or B.ndim != 2:
            raise ShapeError("mixed_precision_gemm expects 2D")
        if A.shape[1] != B.shape[0]:
            raise ShapeError(f"Incompatible shapes: {A.shape} x {B.shape}")
        tc = self._select_tensor_core()
        self._total_ops += 1
        return tc.mixed_precision_matmul(A, B)

    def flash_attention(
        self, Q: np.ndarray, K: np.ndarray, V: np.ndarray,
        causal: bool = False, block_size: int = 64,
    ) -> np.ndarray:
        if Q.ndim != 3:
            raise ShapeError("flash_attention expects [B, T, D]")
        B, T, D = Q.shape
        scale = 1.0 / np.sqrt(D)
        out = np.zeros_like(Q)

        for b in range(B):
            Qb, Kb, Vb = Q[b], K[b], V[b]
            for i_start in range(0, T, block_size):
                i_end = min(i_start + block_size, T)
                Qi = Qb[i_start:i_end]
                oi = np.zeros((i_end - i_start, D), dtype=Q.dtype)
                li = np.zeros((i_end - i_start, 1), dtype=Q.dtype)
                mi = np.full((i_end - i_start, 1), -1e9, dtype=Q.dtype)

                for j_start in range(0, T, block_size):
                    j_end = min(j_start + block_size, T)
                    if causal and j_start > i_end - 1:
                        break
                    Kj = Kb[j_start:j_end]
                    Vj = Vb[j_start:j_end]
                    sij = (Qi @ Kj.T) * scale

                    if causal:
                        rows = np.arange(i_start, i_end).reshape(-1, 1)
                        cols = np.arange(j_start, j_end).reshape(1, -1)
                        mask = cols > rows
                        sij = np.where(mask, -1e9, sij)

                    mi_new = np.maximum(mi, sij.max(axis=-1, keepdims=True))
                    p = np.exp(sij - mi_new)
                    li = li * np.exp(mi - mi_new) + p.sum(axis=-1, keepdims=True)
                    oi = oi * np.exp(mi - mi_new) + p @ Vj
                    mi = mi_new

                out[b, i_start:i_end] = oi / li

        self._total_ops += 1
        return out

    def conv2d(
        self, X: np.ndarray, W: np.ndarray,
        stride: int = 1, padding: int = 0,
    ) -> np.ndarray:
        if X.ndim != 4:
            raise ShapeError("conv2d expects [B, C_in, H, W]")
        if W.ndim != 4:
            raise ShapeError("conv2d kernel expects [C_out, C_in, kH, kW]")

        B, C_in, H, Wid = X.shape
        C_out, C_in_k, kH, kW = W.shape
        if C_in != C_in_k:
            raise ShapeError(f"Channel mismatch: input {C_in} vs kernel {C_in_k}")

        if padding > 0:
            X = np.pad(X, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
            _, _, H, Wid = X.shape

        H_out = (H - kH) // stride + 1
        W_out = (Wid - kW) // stride + 1
        out = np.zeros((B, C_out, H_out, W_out), dtype=X.dtype)

        W_col = W.reshape(C_out, -1)

        for b in range(B):
            cols = []
            for i in range(H_out):
                for j in range(W_out):
                    patch = X[b, :, i*stride:i*stride+kH, j*stride:j*stride+kW]
                    cols.append(patch.reshape(-1))
            col_matrix = np.array(cols)
            result = col_matrix @ W_col.T
            out[b] = result.T.reshape(C_out, H_out, W_out)

        self._total_ops += 1
        return out

    def conv3d(
        self, X: np.ndarray, W: np.ndarray,
        stride: Tuple[int, int, int] = (1, 1, 1),
        padding: Tuple[int, int, int] = (0, 0, 0),
    ) -> np.ndarray:
        if X.ndim != 5:
            raise ShapeError("conv3d expects [B, C_in, D, H, W]")
        if W.ndim != 5:
            raise ShapeError("conv3d kernel expects [C_out, C_in, kD, kH, kW]")

        B, C_in, D, H, Wid = X.shape
        C_out, C_in_k, kD, kH, kW = W.shape

        if C_in != C_in_k:
            raise ShapeError(f"Channel mismatch: {C_in} vs {C_in_k}")

        pd, ph, pw = padding
        sd, sh, sw = stride

        if any(p > 0 for p in padding):
            X = np.pad(X, ((0, 0), (0, 0), (pd, pd), (ph, ph), (pw, pw)))
            _, _, D, H, Wid = X.shape

        D_out = (D - kD) // sd + 1
        H_out = (H - kH) // sh + 1
        W_out = (Wid - kW) // sw + 1
        out = np.zeros((B, C_out, D_out, H_out, W_out), dtype=X.dtype)

        W_col = W.reshape(C_out, -1)

        for b in range(B):
            cols = []
            for d in range(D_out):
                for i in range(H_out):
                    for j in range(W_out):
                        patch = X[b, :, d*sd:d*sd+kD, i*sh:i*sh+kH, j*sw:j*sw+kW]
                        cols.append(patch.reshape(-1))
            col_matrix = np.array(cols)
            result = col_matrix @ W_col.T
            out[b] = result.T.reshape(C_out, D_out, H_out, W_out)

        self._total_ops += 1
        return out

    def layer_norm(
        self, X: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
        eps: float = 1e-5,
    ) -> np.ndarray:
        mean = X.mean(axis=-1, keepdims=True)
        var = X.var(axis=-1, keepdims=True)
        X_norm = (X - mean) / np.sqrt(var + eps)
        self._total_ops += 1
        return gamma * X_norm + beta

    def batch_norm(
        self, X: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
        running_mean: Optional[np.ndarray] = None,
        running_var: Optional[np.ndarray] = None,
        training: bool = True, eps: float = 1e-5, momentum: float = 0.1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if X.ndim == 4:
            axis = (0, 2, 3)
        elif X.ndim == 2:
            axis = (0,)
        else:
            axis = tuple(i for i in range(X.ndim) if i != 1) if X.ndim > 1 else (0,)

        if training:
            mean = X.mean(axis=axis, keepdims=True)
            var = X.var(axis=axis, keepdims=True)
            X_norm = (X - mean) / np.sqrt(var + eps)

            mean_sq = mean.squeeze()
            var_sq = var.squeeze()
            if running_mean is not None:
                running_mean = (1 - momentum) * running_mean + momentum * mean_sq
            else:
                running_mean = mean_sq
            if running_var is not None:
                running_var = (1 - momentum) * running_var + momentum * var_sq
            else:
                running_var = var_sq
        else:
            if running_mean is None or running_var is None:
                raise GPUError("batch_norm: running stats required in eval mode")
            shape = [1] * X.ndim
            shape[1] = -1
            mean = running_mean.reshape(shape)
            var = running_var.reshape(shape)
            X_norm = (X - mean) / np.sqrt(var + eps)

        g_shape = [1] * X.ndim
        g_shape[1] = -1
        out = gamma.reshape(g_shape) * X_norm + beta.reshape(g_shape)
        self._total_ops += 1
        return out, running_mean, running_var

    def gelu(self, X: np.ndarray) -> np.ndarray:
        self._total_ops += 1
        return 0.5 * X * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (X + 0.044715 * X ** 3)))

    def silu(self, X: np.ndarray) -> np.ndarray:
        self._total_ops += 1
        return X * (1.0 / (1.0 + np.exp(-X)))

    def grouped_gemm(
        self, A_list: List[np.ndarray], B_list: List[np.ndarray],
    ) -> List[np.ndarray]:
        if len(A_list) != len(B_list):
            raise ShapeError("grouped_gemm: mismatched group count")
        results = []
        for A, B in zip(A_list, B_list):
            results.append(self.tensor_core_gemm(A, B))
        self._total_ops += 1
        return results

    def fused_attention_norm(
        self, Q: np.ndarray, K: np.ndarray, V: np.ndarray,
        gamma: np.ndarray, beta: np.ndarray,
        causal: bool = False, eps: float = 1e-5,
    ) -> np.ndarray:
        attn_out = self.flash_attention(Q, K, V, causal=causal)
        self._total_ops += 1
        return self.layer_norm(attn_out, gamma, beta, eps=eps)

    def status(self) -> dict:
        tc_status = [tc.status() for tc in self.tensor_core_units]
        total_tc_flops = sum(tc.total_flops for tc in self.tensor_core_units)
        return {
            "lanes": self.lanes,
            "tile_shape": f"{self.tile_m}x{self.tile_n}x{self.tile_k}",
            "precision": self.precision.name,
            "tensor_cores": len(self.tensor_core_units),
            "tensor_core_details": tc_status,
            "total_tensor_core_tflops": round(total_tc_flops / 1e12, 6),
            "memory_pool": self._mem_pool.status(),
            "total_ops_executed": self._total_ops,
        }


class HyperVRAM(VRAM):
    def __init__(self, capacity_bytes: int = 0):
        super().__init__()
        self.capacity_bytes = capacity_bytes
        self._peak_bytes = 0
        self._lock = threading.Lock()

    def alloc(self, array: np.ndarray) -> int:
        with self._lock:
            if self.capacity_bytes > 0:
                current = sum(a.nbytes for a in self._store.values())
                if current + array.nbytes > self.capacity_bytes:
                    raise GPUError(
                        f"HyperVRAM OOM: {current + array.nbytes} > capacity {self.capacity_bytes}"
                    )
            hid = super().alloc(array)
            current = sum(a.nbytes for a in self._store.values())
            if current > self._peak_bytes:
                self._peak_bytes = current
            return hid

    def free(self, hid: int):
        with self._lock:
            super().free(hid)

    def flush(self):
        with self._lock:
            self._store.clear()
            self._meta.clear()

    @property
    def used_bytes(self) -> int:
        return sum(a.nbytes for a in self._store.values())

    @property
    def peak_bytes(self) -> int:
        return self._peak_bytes

    def status(self) -> dict:
        return {
            "capacity_mb": round(self.capacity_bytes / (1024 * 1024), 2) if self.capacity_bytes > 0 else "unlimited",
            "used_mb": round(self.used_bytes / (1024 * 1024), 4),
            "peak_mb": round(self._peak_bytes / (1024 * 1024), 4),
            "handles": len(self._store),
        }


class HyperGPU:
    def __init__(
        self,
        lanes: int = 512,
        tensor_cores: int = 8,
        precision: PrecisionMode = PrecisionMode.MIXED,
        vram_capacity: int = 0,
        tile_size: int = 128,
    ):
        self.core = HyperSIMDCore(
            lanes=lanes,
            tile_m=tile_size, tile_n=tile_size, tile_k=tile_size,
            tensor_cores=tensor_cores,
            precision=precision,
        )
        self.vram = HyperVRAM(capacity_bytes=vram_capacity)
        self.precision = precision
        self._created_at = time.time()
        self._total_compute_ms = 0.0
        self._lock = threading.Lock()
        self._training_ops = 0

    def _record_op(self, op_name: str, flops: int, simulated_ms: float):
        with self._lock:
            self.core._total_ops += 1
            self._training_ops += 1
            if self.core.tensor_core_units:
                self.core.tensor_core_units[0].total_flops += float(flops)
            self._total_compute_ms += simulated_ms

    def gemm(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        t0 = time.time()
        result = self.core.tensor_core_gemm(
            A.astype(np.float32), B.astype(np.float32)
        )
        self._total_compute_ms += (time.time() - t0) * 1000
        return result

    def mixed_gemm(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        t0 = time.time()
        result = self.core.mixed_precision_gemm(A, B)
        self._total_compute_ms += (time.time() - t0) * 1000
        return result

    def flash_attention(
        self, Q: np.ndarray, K: np.ndarray, V: np.ndarray,
        causal: bool = False, block_size: int = 64,
    ) -> np.ndarray:
        t0 = time.time()
        result = self.core.flash_attention(Q, K, V, causal=causal, block_size=block_size)
        self._total_compute_ms += (time.time() - t0) * 1000
        return result

    def attention(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                  causal: bool = False) -> np.ndarray:
        return self.flash_attention(Q, K, V, causal=causal)

    def conv2d(self, X: np.ndarray, W: np.ndarray,
               stride: int = 1, padding: int = 0) -> np.ndarray:
        t0 = time.time()
        result = self.core.conv2d(X, W, stride=stride, padding=padding)
        self._total_compute_ms += (time.time() - t0) * 1000
        return result

    def conv3d(self, X: np.ndarray, W: np.ndarray,
               stride=(1, 1, 1), padding=(0, 0, 0)) -> np.ndarray:
        t0 = time.time()
        result = self.core.conv3d(X, W, stride=stride, padding=padding)
        self._total_compute_ms += (time.time() - t0) * 1000
        return result

    def layer_norm(self, X: np.ndarray, gamma: np.ndarray,
                   beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        t0 = time.time()
        result = self.core.layer_norm(X, gamma, beta, eps=eps)
        self._total_compute_ms += (time.time() - t0) * 1000
        return result

    def batch_norm(self, X: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
                   running_mean=None, running_var=None,
                   training=True, eps=1e-5):
        t0 = time.time()
        result = self.core.batch_norm(
            X, gamma, beta, running_mean, running_var, training, eps
        )
        self._total_compute_ms += (time.time() - t0) * 1000
        return result

    def gelu(self, X: np.ndarray) -> np.ndarray:
        t0 = time.time()
        result = self.core.gelu(X)
        self._total_compute_ms += (time.time() - t0) * 1000
        return result

    def silu(self, X: np.ndarray) -> np.ndarray:
        t0 = time.time()
        result = self.core.silu(X)
        self._total_compute_ms += (time.time() - t0) * 1000
        return result

    def softmax(self, X: np.ndarray, axis: int = -1) -> np.ndarray:
        t0 = time.time()
        result = self.core.softmax(X, axis=axis)
        self._total_compute_ms += (time.time() - t0) * 1000
        return result

    def add(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        t0 = time.time()
        result = self.core.add(A, B)
        self._total_compute_ms += (time.time() - t0) * 1000
        return result

    def grouped_gemm(self, A_list, B_list):
        t0 = time.time()
        result = self.core.grouped_gemm(A_list, B_list)
        self._total_compute_ms += (time.time() - t0) * 1000
        return result

    def fused_attention_norm(self, Q, K, V, gamma, beta, causal=False, eps=1e-5):
        t0 = time.time()
        result = self.core.fused_attention_norm(Q, K, V, gamma, beta, causal, eps)
        self._total_compute_ms += (time.time() - t0) * 1000
        return result

    def flush_vram(self):
        self.vram.flush()

    def status(self) -> dict:
        uptime = time.time() - self._created_at
        core_status = self.core.status()
        return {
            "engine": "HyperGPU",
            "lanes": self.core.lanes,
            "tensor_cores": len(self.core.tensor_core_units),
            "precision": self.precision.name,
            "tile_shape": core_status["tile_shape"],
            "total_ops": core_status["total_ops_executed"],
            "total_tensor_core_tflops": core_status["total_tensor_core_tflops"],
            "total_compute_ms": round(self._total_compute_ms, 2),
            "vram": self.vram.status(),
            "memory_pool": core_status["memory_pool"],
            "uptime_s": round(uptime, 1),
        }


class GPUClusterNode:
    def __init__(
        self,
        node_id: int,
        lanes: int = 512,
        tensor_cores: int = 8,
        precision: PrecisionMode = PrecisionMode.MIXED,
        vram_capacity: int = 0,
    ):
        self.node_id = node_id
        self.gpu = HyperGPU(
            lanes=lanes,
            tensor_cores=tensor_cores,
            precision=precision,
            vram_capacity=vram_capacity,
        )
        self.state = "idle"
        self._assigned_task: Optional[str] = None

    def assign(self, task: str):
        self.state = "busy"
        self._assigned_task = task

    def release(self):
        self.state = "idle"
        self._assigned_task = None

    def status(self) -> dict:
        return {
            "node_id": self.node_id,
            "state": self.state,
            "assigned_task": self._assigned_task,
            "gpu": self.gpu.status(),
        }


class GPUCluster:
    def __init__(
        self,
        num_nodes: int = 4,
        lanes_per_node: int = 512,
        tensor_cores_per_node: int = 8,
        precision: PrecisionMode = PrecisionMode.MIXED,
        vram_per_node: int = 0,
    ):
        self.nodes: Dict[int, GPUClusterNode] = {}
        self._lock = threading.Lock()
        self._gradient_buffer: Dict[str, List[np.ndarray]] = {}

        for i in range(num_nodes):
            self.nodes[i] = GPUClusterNode(
                node_id=i,
                lanes=lanes_per_node,
                tensor_cores=tensor_cores_per_node,
                precision=precision,
                vram_capacity=vram_per_node,
            )

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def total_lanes(self) -> int:
        return sum(n.gpu.core.lanes for n in self.nodes.values())

    @property
    def total_tensor_cores(self) -> int:
        return sum(len(n.gpu.core.tensor_core_units) for n in self.nodes.values())

    def get_node(self, node_id: int) -> GPUClusterNode:
        if node_id not in self.nodes:
            raise GPUError(f"Node {node_id} not found in cluster")
        return self.nodes[node_id]

    def get_idle_node(self) -> Optional[GPUClusterNode]:
        with self._lock:
            for node in self.nodes.values():
                if node.state == "idle":
                    return node
        return None

    def all_reduce_gradients(self, param_name: str, gradients: List[np.ndarray]) -> np.ndarray:
        if not gradients:
            raise GPUError("all_reduce: empty gradient list")
        avg = np.mean(gradients, axis=0)
        return avg

    def scatter_data(self, data: np.ndarray, num_chunks: int = 0) -> List[np.ndarray]:
        if num_chunks <= 0:
            num_chunks = self.num_nodes
        return np.array_split(data, num_chunks, axis=0)

    def gather_results(self, chunks: List[np.ndarray], axis: int = 0) -> np.ndarray:
        return np.concatenate(chunks, axis=axis)

    def run_distributed(
        self, fn, data_chunks: List[np.ndarray],
        node_ids: Optional[List[int]] = None,
    ) -> List[np.ndarray]:
        if node_ids is None:
            node_ids = list(self.nodes.keys())[:len(data_chunks)]

        results = [None] * len(data_chunks)
        errors = {}
        threads = []

        def _run_on_node(idx, nid, chunk):
            try:
                node = self.get_node(nid)
                node.assign(f"distributed_chunk_{idx}")
                results[idx] = fn(node.gpu, chunk)
                node.release()
            except Exception as e:
                errors[nid] = str(e)

        for i, (nid, chunk) in enumerate(zip(node_ids, data_chunks)):
            t = threading.Thread(target=_run_on_node, args=(i, nid, chunk), daemon=True)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        if errors:
            raise GPUError(f"Distributed execution errors: {errors}")

        return results

    def flush_all(self):
        for node in self.nodes.values():
            node.gpu.flush_vram()

    def add_node(
        self, lanes: int = 512, tensor_cores: int = 8,
        precision: PrecisionMode = PrecisionMode.MIXED,
    ) -> int:
        with self._lock:
            nid = max(self.nodes.keys()) + 1 if self.nodes else 0
            self.nodes[nid] = GPUClusterNode(
                node_id=nid, lanes=lanes,
                tensor_cores=tensor_cores, precision=precision,
            )
            return nid

    def remove_node(self, node_id: int):
        with self._lock:
            node = self.nodes.pop(node_id, None)
            if node:
                node.gpu.flush_vram()

    def status(self) -> dict:
        node_statuses = {nid: n.status() for nid, n in self.nodes.items()}
        idle = sum(1 for n in self.nodes.values() if n.state == "idle")
        busy = sum(1 for n in self.nodes.values() if n.state == "busy")
        total_compute = sum(n.gpu._total_compute_ms for n in self.nodes.values())
        total_ops = sum(n.gpu.core._total_ops for n in self.nodes.values())
        total_tc_flops = sum(
            tc.total_flops
            for n in self.nodes.values()
            for tc in n.gpu.core.tensor_core_units
        )

        return {
            "engine": "HyperGPU Cluster",
            "num_nodes": self.num_nodes,
            "total_lanes": self.total_lanes,
            "total_tensor_cores": self.total_tensor_cores,
            "nodes_idle": idle,
            "nodes_busy": busy,
            "total_ops": total_ops,
            "total_compute_ms": round(total_compute, 2),
            "total_tensor_core_tflops": round(total_tc_flops / 1e12, 6),
            "nodes": node_statuses,
        }
