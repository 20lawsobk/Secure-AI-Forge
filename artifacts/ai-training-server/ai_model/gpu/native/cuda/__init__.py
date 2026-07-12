"""MaxCore SM102 CUDA layer.

Custom nvcc + runtime that compiles .cu kernel sources and executes them on
the digital GPU (VRAM + SIMDCore / HyperSIMDCore) rather than on physical
NVIDIA hardware.

Public surface:
    CUDANvcc        — compile .cu source → CUDAModule
    CUDAModule      — callable kernel container
    CUDARuntime     — CUDA execution primitives on digital GPU
"""

from .nvcc import CUDANvcc, CUDAModule
from .runtime import CUDARuntime

__all__ = ["CUDANvcc", "CUDAModule", "CUDARuntime"]
