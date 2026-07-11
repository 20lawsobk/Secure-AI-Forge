"""Fused native SIMD kernels with numpy fallback.

Each kernel is a single fused loop compiled by :class:`NativeCompiler` so it makes
one vectorized pass over memory instead of the several passes (and temporaries)
idiomatic numpy produces. Every kernel has a numpy fallback with identical math,
so results are correct whether or not a compiler is available — the native path
is a speed optimization, never a correctness dependency.

Honesty: this is a real *compiled CPU SIMD* path (SPMD-on-CPU). It delivers a
measured, bounded speedup over numpy via fusion + vectorization. It is NOT GPU
hardware and does not claim GPU throughput — ``describe()["is_hardware_execution"]``
is always False.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ai_model.gpu.native.compiler import NativeCompiler

_FP = None  # set lazily to avoid importing ctypes at module import for pure-numpy


_C_SOURCE = r"""
#include <stddef.h>
#include <math.h>

/* y = scale * relu(a*x + b)^2 */
void affine_relu_sq(const float* x, float* y, float a, float b,
                    float scale, size_t n) {
    #pragma GCC ivdep
    for (size_t i = 0; i < n; ++i) {
        float t = a * x[i] + b;
        t = t > 0.0f ? t : 0.0f;
        y[i] = scale * t * t;
    }
}

/* hardswish: x * clamp(x+3, 0, 6) / 6 */
void hardswish(const float* x, float* y, size_t n) {
    const float inv6 = 1.0f / 6.0f;
    #pragma GCC ivdep
    for (size_t i = 0; i < n; ++i) {
        float t = x[i] + 3.0f;
        t = t < 0.0f ? 0.0f : (t > 6.0f ? 6.0f : t);
        y[i] = x[i] * t * inv6;
    }
}

/* out = a*x + b*y */
void axpby(const float* x, const float* y, float* out,
           float a, float b, size_t n) {
    #pragma GCC ivdep
    for (size_t i = 0; i < n; ++i) {
        out[i] = a * x[i] + b * y[i];
    }
}

/* rowwise RMSNorm: y[r,c] = x[r,c] / sqrt(mean_c(x[r,:]^2) + eps) * gamma[c] */
void rmsnorm_rows(const float* x, const float* gamma, float* y,
                  float eps, size_t rows, size_t cols) {
    for (size_t r = 0; r < rows; ++r) {
        const float* xr = x + r * cols;
        float* yr = y + r * cols;
        float ss = 0.0f;
        for (size_t c = 0; c < cols; ++c) ss += xr[c] * xr[c];
        float inv = 1.0f / sqrtf(ss / (float) cols + eps);
        #pragma GCC ivdep
        for (size_t c = 0; c < cols; ++c) yr[c] = xr[c] * inv * gamma[c];
    }
}
"""


class NativeKernels:
    """Compiled fused kernels; transparently falls back to numpy."""

    def __init__(self, compiler: Optional[NativeCompiler] = None):
        global _FP
        self.compiler = compiler or NativeCompiler()
        self._lib = self.compiler.compile(_C_SOURCE)
        self.available = self._lib is not None
        self.stats: Dict[str, int] = {"native": 0, "fallback": 0}
        if self.available:
            import ctypes
            _FP = ctypes.POINTER(ctypes.c_float)
            self._bind(ctypes)

    def _bind(self, ctypes) -> None:
        fp, sz, f = _FP, ctypes.c_size_t, ctypes.c_float
        self._lib.affine_relu_sq.restype = None
        self._lib.affine_relu_sq.argtypes = [fp, fp, f, f, f, sz]
        self._lib.hardswish.restype = None
        self._lib.hardswish.argtypes = [fp, fp, sz]
        self._lib.axpby.restype = None
        self._lib.axpby.argtypes = [fp, fp, fp, f, f, sz]
        self._lib.rmsnorm_rows.restype = None
        self._lib.rmsnorm_rows.argtypes = [fp, fp, fp, f, sz, sz]

    # ── helpers ────────────────────────────────────────────────────────────
    @staticmethod
    def _f32(x) -> np.ndarray:
        return np.ascontiguousarray(x, dtype=np.float32)

    def _ptr(self, a: np.ndarray):
        return a.ctypes.data_as(_FP)

    def _use_native(self, *arrs: np.ndarray) -> bool:
        return self.available and all(a.dtype == np.float32 for a in arrs)

    # ── kernels ────────────────────────────────────────────────────────────
    def affine_relu_sq(self, x, a: float, b: float, scale: float) -> np.ndarray:
        x = self._f32(x)
        if not self._use_native(x):
            self.stats["fallback"] += 1
            return (np.float32(scale) *
                    np.maximum(np.float32(a) * x + np.float32(b), 0.0) ** 2
                    ).astype(np.float32)
        y = np.empty_like(x)
        self._lib.affine_relu_sq(self._ptr(x), self._ptr(y),
                                 float(a), float(b), float(scale), x.size)
        self.stats["native"] += 1
        return y

    def hardswish(self, x) -> np.ndarray:
        x = self._f32(x)
        if not self._use_native(x):
            self.stats["fallback"] += 1
            return (x * np.clip(x + 3.0, 0.0, 6.0) / 6.0).astype(np.float32)
        y = np.empty_like(x)
        self._lib.hardswish(self._ptr(x), self._ptr(y), x.size)
        self.stats["native"] += 1
        return y

    def axpby(self, x, y, a: float, b: float) -> np.ndarray:
        x, y = self._f32(x), self._f32(y)
        if x.shape != y.shape:
            raise ValueError(f"axpby shape mismatch: {x.shape} vs {y.shape}")
        if not self._use_native(x, y):
            self.stats["fallback"] += 1
            return (np.float32(a) * x + np.float32(b) * y).astype(np.float32)
        out = np.empty_like(x)
        self._lib.axpby(self._ptr(x), self._ptr(y), self._ptr(out),
                        float(a), float(b), x.size)
        self.stats["native"] += 1
        return out

    def rmsnorm_rows(self, x, gamma, eps: float = 1e-6) -> np.ndarray:
        x = self._f32(x)
        if x.ndim != 2:
            raise ValueError(f"rmsnorm_rows expects 2D [rows, cols], got {x.shape}")
        gamma = self._f32(gamma)
        if gamma.shape != (x.shape[1],):
            raise ValueError(
                f"gamma shape {gamma.shape} != (cols={x.shape[1]},)")
        if not self._use_native(x, gamma):
            self.stats["fallback"] += 1
            inv = 1.0 / np.sqrt(np.mean(x ** 2, axis=1, keepdims=True) + eps)
            return (x * inv * gamma).astype(np.float32)
        rows, cols = x.shape
        y = np.empty_like(x)
        self._lib.rmsnorm_rows(self._ptr(x), self._ptr(gamma), self._ptr(y),
                               float(eps), rows, cols)
        self.stats["native"] += 1
        return y

    # ── provenance ─────────────────────────────────────────────────────────
    def describe(self) -> Dict[str, object]:
        return {
            "backend": "native-cpu-simd" if self.available else "numpy-fallback",
            "compiler": self.compiler.cc,
            "cc_version": self.compiler.cc_version,
            "flags": self.compiler._flags() if self.available else [],
            "is_hardware_execution": False,   # honesty: compiled CPU SIMD, not GPU
            "note": ("real compiled CPU SIMD (SPMD-on-CPU) fused kernels; measured "
                     "speedup over idiomatic numpy via fusion + vectorization, "
                     "bounded and still on CPU (not GPU throughput)."),
            "last_error": self.compiler.last_error,
        }


_DEFAULT: Optional[NativeKernels] = None


def get_native_kernels() -> NativeKernels:
    """Process-wide singleton so the shared lib is compiled/loaded once."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = NativeKernels()
    return _DEFAULT
