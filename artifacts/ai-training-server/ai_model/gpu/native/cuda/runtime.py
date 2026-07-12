"""CUDARuntime — CUDA execution model on the digital GPU.

Maps CUDA's thread/block/warp primitives to digital GPU operations:

  CUDA concept          Digital GPU mapping
  ─────────────────     ───────────────────────────────────────────
  warp (32 lanes)       SIMDCore with lanes=32
  __shared__ float[N]   numpy array, scope-local to one block call
  __syncthreads()       sequential fence (no-op in single-thread exec)
  __shfl_down_sync()    vectorised numpy shift+mask reduction
  atomicAdd()           numpy += under a threading.Lock
  threadIdx / blockIdx  Python integers computed by the launch loop
  gridDim / blockDim    launch parameters passed by CUDAModule.__call__

This is not a CUDA interpreter — it is the execution substrate that
SM102 kernel implementations are written against.  Kernel bodies are
Python functions that receive a `CUDARuntime` context and call its
methods directly.
"""
from __future__ import annotations

import threading
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Dim3 — mirrors CUDA's dim3 struct
# ---------------------------------------------------------------------------
class Dim3:
    """Thread/block/grid dimension triple."""
    __slots__ = ("x", "y", "z")

    def __init__(self, x: int = 1, y: int = 1, z: int = 1):
        self.x, self.y, self.z = x, y, z

    @classmethod
    def from_arg(cls, arg) -> "Dim3":
        if isinstance(arg, cls):
            return arg
        if isinstance(arg, int):
            return cls(arg)
        if isinstance(arg, (tuple, list)):
            padded = list(arg) + [1, 1]
            return cls(*padded[:3])
        raise TypeError(f"Cannot convert {type(arg)} to Dim3")

    def __repr__(self) -> str:
        return f"Dim3({self.x}, {self.y}, {self.z})"

    @property
    def volume(self) -> int:
        return self.x * self.y * self.z


# ---------------------------------------------------------------------------
# SharedMemory — per-block scratchpad (float32 by default)
# ---------------------------------------------------------------------------
class SharedMemory:
    """Fixed-size float32 scratchpad — models GPU shared memory per block."""

    def __init__(self, n_floats: int):
        self._buf = np.zeros(n_floats, dtype=np.float32)

    # Slice views so kernels can do:  m_i = shm.view(0, T)
    def view(self, start: int, stop: int) -> np.ndarray:
        return self._buf[start:stop]

    def __getitem__(self, idx):
        return self._buf[idx]

    def __setitem__(self, idx, val):
        self._buf[idx] = val

    def reset(self):
        self._buf[:] = 0.0

    @property
    def nbytes(self) -> int:
        return self._buf.nbytes


# ---------------------------------------------------------------------------
# WarpState — warp-level primitives
# ---------------------------------------------------------------------------
class WarpState:
    """Models one warp (32 lanes) of execution on a SIMDCore."""

    WARP_SIZE: int = 32

    def __init__(self, lane_values: Optional[np.ndarray] = None):
        self._vals = np.zeros(self.WARP_SIZE, dtype=np.float32)
        if lane_values is not None:
            n = min(len(lane_values), self.WARP_SIZE)
            self._vals[:n] = lane_values[:n]

    def load(self, values: np.ndarray):
        n = min(len(values), self.WARP_SIZE)
        self._vals[:n] = values[:n]

    # __shfl_down_sync — butterfly reduction step
    def shfl_down_sync(self, mask: int, val: np.ndarray, offset: int) -> np.ndarray:
        """Vectorised __shfl_down_sync across all 32 lanes simultaneously."""
        result = val.copy()
        shifted = np.zeros_like(val)
        shifted[:self.WARP_SIZE - offset] = val[offset:]
        # Lanes whose source is out-of-range keep their own value
        active = np.arange(self.WARP_SIZE) + offset < self.WARP_SIZE
        result[active] = val[active] + shifted[active]
        result[~active] = val[~active]
        return result

    def reduce_sum(self, vals: np.ndarray) -> np.ndarray:
        """Full warp horizontal sum via butterfly — maps to __shfl_down_sync loop."""
        v = vals.copy()
        offset = self.WARP_SIZE // 2
        while offset > 0:
            v = self.shfl_down_sync(0xFFFFFFFF, v, offset)
            offset >>= 1
        return v


# ---------------------------------------------------------------------------
# Atomics
# ---------------------------------------------------------------------------
class _AtomicFloat:
    """Thread-safe float accumulator — models atomicAdd on a scalar."""

    def __init__(self, initial: float = 0.0):
        self._val = np.float32(initial)
        self._lock = threading.Lock()

    def add(self, delta: float):
        with self._lock:
            self._val += np.float32(delta)

    @property
    def value(self) -> float:
        return float(self._val)

    def reset(self):
        with self._lock:
            self._val = np.float32(0.0)


def atomic_add(arr: np.ndarray, idx, delta: float):
    """In-place atomic add to a numpy array element."""
    arr[idx] += delta   # single-threaded digital GPU: lock is implicit


# ---------------------------------------------------------------------------
# CUDARuntime — per-block execution context
# ---------------------------------------------------------------------------
class CUDARuntime:
    """Execution context for one block of a CUDA kernel running on digital GPU.

    Kernel implementations receive a ``CUDARuntime`` and call its methods
    instead of CUDA intrinsics:

        def my_kernel(rt: CUDARuntime, A, B, out, N):
            shm   = rt.shared(N)
            lane  = rt.thread_idx.x % rt.WARP_SIZE
            block = rt.block_idx.x
            ...
            rt.syncthreads()
            ...
            rt.atomic_add(out, 0, shm[0])
    """

    WARP_SIZE: int = 32

    def __init__(self,
                 thread_idx: Dim3,
                 block_idx: Dim3,
                 block_dim: Dim3,
                 grid_dim: Dim3,
                 shared_n_floats: int = 0):
        self.thread_idx  = thread_idx
        self.block_idx   = block_idx
        self.block_dim   = block_dim
        self.grid_dim    = grid_dim
        self._shm        = SharedMemory(shared_n_floats) if shared_n_floats else None
        self._warp       = WarpState()
        self._atomics: Dict[int, _AtomicFloat] = {}

    # ── shared memory ───────────────────────────────────────────────────────
    def shared(self, n_floats: int) -> SharedMemory:
        if self._shm is None or len(self._shm._buf) < n_floats:
            self._shm = SharedMemory(n_floats)
        return self._shm

    # ── sync / barriers ─────────────────────────────────────────────────────
    def syncthreads(self):
        """__syncthreads() — sequential execution: acts as a compiler fence."""
        pass  # no-op: digital GPU runs one thread at a time per block call

    # ── warp intrinsics ─────────────────────────────────────────────────────
    def shfl_down_sync(self, mask: int, val: np.ndarray, offset: int) -> np.ndarray:
        return self._warp.shfl_down_sync(mask, val, offset)

    def warp_reduce_sum(self, vals: np.ndarray) -> np.ndarray:
        return self._warp.reduce_sum(vals)

    # ── atomics ─────────────────────────────────────────────────────────────
    def atomic_add(self, arr: np.ndarray, idx, delta: float):
        atomic_add(arr, idx, delta)

    # ── math intrinsics (mirrors CUDA device functions) ──────────────────────
    @staticmethod
    def rsqrtf(x: float) -> float:
        return 1.0 / np.sqrt(float(x))

    @staticmethod
    def expf(x: float) -> float:
        return float(np.exp(np.float32(x)))

    @staticmethod
    def fmaxf(a: float, b: float) -> float:
        return float(max(a, b))

    # ── fp conversion (mirrors CUDA intrinsics) ──────────────────────────────
    @staticmethod
    def float2half(x: np.ndarray) -> np.ndarray:
        return x.astype(np.float16)

    @staticmethod
    def half2float(x: np.ndarray) -> np.ndarray:
        return x.astype(np.float32)


# ---------------------------------------------------------------------------
# Launch helper — iterates the grid and calls one Python block function
# ---------------------------------------------------------------------------
def launch_kernel(kernel_fn: Callable,
                  grid: Any,
                  block: Any,
                  shared_n_floats: int,
                  *args) -> None:
    """Drive a Python kernel function over a CUDA-style grid.

    ``kernel_fn(rt: CUDARuntime, *args)`` is called once per block.
    For SM102 kernels the body is already vectorised over threads, so a
    single Python call per block is exact.
    """
    grid_d  = Dim3.from_arg(grid)
    block_d = Dim3.from_arg(block)

    for bz in range(grid_d.z):
        for by in range(grid_d.y):
            for bx in range(grid_d.x):
                rt = CUDARuntime(
                    thread_idx=Dim3(0, 0, 0),   # vectorised: all threads at once
                    block_idx=Dim3(bx, by, bz),
                    block_dim=block_d,
                    grid_dim=grid_d,
                    shared_n_floats=shared_n_floats,
                )
                kernel_fn(rt, *args)
