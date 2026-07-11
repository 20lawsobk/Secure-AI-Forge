"""Library layer (BLAS / DNN) on top of the digital GPU.

These are reference implementations whose heavy matmuls route through the real
tiled ``DigitalGPU.gemm``. They are correct and readable, not tuned kernels —
the "build it better" here is using im2col + gemm for conv (one matmul per batch)
instead of six nested Python loops.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ai_model.gpu.digital_gpu import DigitalGPU, ShapeMismatchError


class DigitalBLAS:
    def __init__(self, gpu: Optional[DigitalGPU] = None):
        self.gpu = gpu or DigitalGPU()

    def gemm(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return self.gpu.gemm(np.asarray(A), np.asarray(B))

    def gemm_bias_relu(self, A: np.ndarray, B: np.ndarray,
                       bias: np.ndarray) -> np.ndarray:
        return self.gpu.gemm_bias_relu(np.asarray(A), np.asarray(B),
                                       np.asarray(bias))


class DigitalDNN:
    def __init__(self, gpu: Optional[DigitalGPU] = None):
        self.gpu = gpu or DigitalGPU()

    def conv2d(self, x: np.ndarray, w: np.ndarray, bias: Optional[np.ndarray] = None,
               stride: int = 1, padding: int = 0) -> np.ndarray:
        """2D convolution via im2col + gemm. x[N,C,H,W], w[F,C,KH,KW]."""
        x = np.asarray(x, dtype=np.float64)
        w = np.asarray(w, dtype=np.float64)
        if x.ndim != 4 or w.ndim != 4:
            raise ShapeMismatchError("conv2d expects x[N,C,H,W] and w[F,C,KH,KW]")
        N, C, H, W = x.shape
        F, C2, KH, KW = w.shape
        if C != C2:
            raise ShapeMismatchError(
                f"conv2d channel mismatch: input C={C} vs weight C={C2}")

        OH = (H + 2 * padding - KH) // stride + 1
        OW = (W + 2 * padding - KW) // stride + 1
        if OH <= 0 or OW <= 0:
            raise ShapeMismatchError(
                f"conv2d output has non-positive size: OH={OH}, OW={OW}")

        xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
        col = np.zeros((N, C, KH, KW, OH, OW), dtype=np.float64)
        for i in range(KH):
            i_max = i + stride * OH
            for j in range(KW):
                j_max = j + stride * OW
                col[:, :, i, j, :, :] = xp[:, :, i:i_max:stride, j:j_max:stride]
        col = col.reshape(N, C * KH * KW, OH * OW)
        wcol = w.reshape(F, C * KH * KW)

        out = np.empty((N, F, OH * OW), dtype=np.float64)
        for n in range(N):
            out[n] = self.gpu.gemm(wcol, col[n])      # heavy math on the digital GPU
        out = out.reshape(N, F, OH, OW)
        if bias is not None:
            out = out + np.asarray(bias, dtype=np.float64).reshape(1, F, 1, 1)
        return out

    def relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(np.asarray(x), 0.0)
