"""Path-A native SIMD: compile fused kernels to CPU vector code (SPMD-on-CPU).

Real, measured speedups over idiomatic numpy via fusion + auto-vectorization —
still CPU, never GPU hardware. See kernels.NativeKernels.describe().
"""
from ai_model.gpu.native.compiler import NativeCompiler
from ai_model.gpu.native.kernels import NativeKernels, get_native_kernels

__all__ = ["NativeCompiler", "NativeKernels", "get_native_kernels"]
