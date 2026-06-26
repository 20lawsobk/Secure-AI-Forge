"""Tensor wrapper for the DigitalGPU / MaxCore public layer.

A thin, backend-agnostic handle over a contiguous numpy buffer. The CPU backend
materializes these as numpy arrays; future backends (GPU/cluster/ASIC) may keep
the payload on a device and only realize it lazily. Keeping a single ``Tensor``
type at the API boundary means model code never touches a concrete backend
buffer directly.
"""
from __future__ import annotations

from typing import Any

import numpy as np

_DTYPE_MAP = {
    "float32": np.float32,
    "float64": np.float64,
    "float16": np.float16,
    "int32": np.int32,
    "int64": np.int64,
    "bool": np.bool_,
}


class Tensor:
    """Backend-agnostic tensor handle wrapping a numpy buffer."""

    __slots__ = ("data", "device")

    def __init__(self, data: Any, dtype: str | None = "float32", device: str = "cpu"):
        if isinstance(data, Tensor):
            data = data.data
        arr = np.asarray(data)
        if dtype is not None:
            np_dtype = _DTYPE_MAP.get(dtype, dtype)
            if arr.dtype != np.dtype(np_dtype):
                arr = arr.astype(np_dtype)
        self.data = arr
        self.device = device

    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def dtype(self) -> str:
        return str(self.data.dtype)

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return int(self.data.size)

    def numpy(self) -> np.ndarray:
        return self.data

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, dtype={self.dtype}, device={self.device})"


def as_tensor(x: Any, dtype: str | None = None, device: str = "cpu") -> Tensor:
    """Coerce ``x`` to a ``Tensor`` without copying when it already is one."""
    if isinstance(x, Tensor):
        return x
    return Tensor(x, dtype=dtype or "float32", device=device)


def to_numpy(x: Any) -> np.ndarray:
    """Return the underlying numpy buffer for a ``Tensor`` or array-like."""
    if isinstance(x, Tensor):
        return x.data
    return np.asarray(x)
