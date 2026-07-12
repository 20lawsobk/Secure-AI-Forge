"""CUDANvcc — custom nvcc for MaxCore SM102.

Compiles .cu kernel sources by:
  1. Parsing ``__global__`` kernel function names from the source text.
  2. Looking each name up in the SM102 kernel registry.
  3. Returning a ``CUDAModule`` whose kernels run on the digital GPU.

Caches compiled modules by SHA-1 of (source + registry version) so repeated
calls to ``compile()`` with the same source are free after the first.

Never-raise: unknown kernel names are skipped with a warning; the caller
always receives a valid (possibly empty) ``CUDAModule``.

Design mirrors ``native/compiler.py``:
  - ``NativeCompiler`` : ``gcc`` → ``.so`` → ``ctypes.CDLL``
  - ``CUDANvcc``       : ``.cu`` → registry lookup → ``CUDAModule``
"""
from __future__ import annotations

import functools
import hashlib
import logging
import re
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np

from ai_model.gpu.native.cuda.runtime import CUDARuntime, Dim3, launch_kernel
from ai_model.gpu.native.cuda import sm102_kernels as _sm102

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry version — bump when sm102_kernels.py changes so caches invalidate
# ---------------------------------------------------------------------------
_REGISTRY_VERSION = "sm102-v1"

# ---------------------------------------------------------------------------
# SM102 kernel registry
# name (as it appears after __global__ void in the .cu) → Python callable
# ---------------------------------------------------------------------------
_REGISTRY: Dict[str, Callable] = {
    # Flash attention — both standalone and unified variants
    "flash_attn_sm102_kernel":          _sm102.flash_attn_sm102,
    "flash_attn_sm102_kernel_unified":  _sm102.flash_attn_sm102,

    # im2col
    "im2col_sm102_kernel":              _sm102.im2col_sm102,
    "im2col_sm102_kernel_unified":      _sm102.im2col_sm102,

    # Tensor-core WMMA convolution
    "conv_wmma_sm102_kernel":           _sm102.conv_wmma_sm102,
    "conv_wmma_sm102_kernel_unified":   _sm102.conv_wmma_sm102,

    # Dot-product reductions
    "reduction_current_sm102_kernel":           _sm102.reduction_current_sm102,
    "reduction_current_sm102_kernel_unified":   _sm102.reduction_current_sm102,
    "reduction_redesigned_sm102_kernel":        _sm102.reduction_redesigned_sm102,
    "reduction_redesigned_sm102_kernel_unified":_sm102.reduction_redesigned_sm102,
}


# ---------------------------------------------------------------------------
# CUDAModule — the compiled "binary" returned by CUDANvcc.compile()
# ---------------------------------------------------------------------------
class CUDAModule:
    """Container of compiled SM102 kernels backed by digital GPU callables.

    Usage::

        module = nvcc.compile_file("flashattn_sm102.cu")
        kernel = module.get_kernel("flash_attn_sm102_kernel")
        kernel(grid=(B*H,), block=(32,), Q=Q, K=K, V=V, O=O, causal=True)
    """

    def __init__(self,
                 kernels: Dict[str, Callable],
                 source_hash: str,
                 unknown: Sequence[str] = ()):
        self._kernels     = dict(kernels)
        self.source_hash  = source_hash
        self.unknown      = list(unknown)   # names that had no registry entry

    # ------------------------------------------------------------------
    def get_kernel(self, name: str) -> Optional[Callable]:
        """Return the bound kernel callable or ``None`` if not found."""
        return self._kernels.get(name)

    def __call__(self,
                 kernel_name: str,
                 grid,
                 block,
                 *args,
                 shared_mem: int = 0,
                 gpu=None,
                 **kwargs) -> Any:
        """Launch ``kernel_name`` over ``grid`` blocks of ``block`` threads.

        Args:
            kernel_name:  Name of the ``__global__`` function.
            grid:         Grid dimensions (int or (x,y,z)).
            block:        Block dimensions (int or (x,y,z)).
            *args:        Positional arguments forwarded to the kernel.
            shared_mem:   Shared-memory size hint in floats (not bytes).
            gpu:          HyperGPU instance to inject into the kernel.
            **kwargs:     Keyword arguments forwarded to the kernel.
        """
        fn = self.get_kernel(kernel_name)
        if fn is None:
            raise ValueError(
                f"CUDAModule: kernel '{kernel_name}' not found. "
                f"Available: {sorted(self._kernels)}"
            )

        grid_d  = Dim3.from_arg(grid)
        block_d = Dim3.from_arg(block)

        # Build a runtime for block (0,0,0) — sm102 kernels are vectorised
        # over the full block so we call the function once per grid block.
        # kwargs are captured in the closure; launch_kernel only receives
        # positional args so its signature stays clean.
        _kw = kwargs  # closure capture

        def _block_fn(rt: CUDARuntime, *a):
            return fn(rt, *a, _gpu=gpu, **_kw)

        launch_kernel(_block_fn, grid_d, block_d, shared_mem, *args)

    def __repr__(self) -> str:
        return (f"CUDAModule(hash={self.source_hash[:8]}, "
                f"kernels={sorted(self._kernels)}, "
                f"unknown={self.unknown})")


# ---------------------------------------------------------------------------
# Source parser — extracts __global__ kernel names from .cu text
# ---------------------------------------------------------------------------
_KERNEL_RE = re.compile(
    r"__global__\s+(?:void|[\w:<>,\s]+?)\s+([\w]+)\s*\(",
    re.MULTILINE,
)

# Also handle template instantiations like:
#   template __global__ void flash_attn_sm102_kernel<FP8_E4M3>(...)
_TEMPLATE_KERNEL_RE = re.compile(
    r"template\s+__global__\s+(?:void|[\w:<>,\s]+?)\s+([\w]+)\s*<",
    re.MULTILINE,
)


def _parse_kernel_names(source: str) -> list[str]:
    names = _KERNEL_RE.findall(source) + _TEMPLATE_KERNEL_RE.findall(source)
    # De-duplicate preserving order
    seen: set = set()
    out = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


# ---------------------------------------------------------------------------
# CUDANvcc
# ---------------------------------------------------------------------------
class CUDANvcc:
    """Custom nvcc for MaxCore SM102.

    Compiles .cu source (string or file path) into a ``CUDAModule`` backed
    by digital GPU kernel implementations.  Results are cached by source
    SHA-1 so compiling the same file twice costs nothing after the first call.
    """

    def __init__(self, gpu=None):
        """
        Args:
            gpu: HyperGPU (or compatible) instance injected into kernels.
                 Pass ``None`` to use the numpy fallback paths.
        """
        self.gpu  = gpu
        self._cache: Dict[str, CUDAModule] = {}
        self.last_unknown: list[str] = []

    # ------------------------------------------------------------------
    def _cache_key(self, source: str) -> str:
        sig = source + "|" + _REGISTRY_VERSION
        return hashlib.sha1(sig.encode()).hexdigest()[:16]

    # ------------------------------------------------------------------
    def compile(self, source: str) -> CUDAModule:
        """Compile a .cu source string → CUDAModule.

        Never raises.  Unknown kernels are recorded in ``module.unknown``
        and ``self.last_unknown``.
        """
        key = self._cache_key(source)
        if key in self._cache:
            return self._cache[key]

        names   = _parse_kernel_names(source)
        kernels: Dict[str, Callable] = {}
        unknown = []

        for name in names:
            fn = _REGISTRY.get(name)
            if fn is None:
                unknown.append(name)
                log.warning("CUDANvcc: no SM102 implementation for kernel '%s'", name)
            else:
                kernels[name] = fn

        self.last_unknown = unknown
        module = CUDAModule(kernels, key, unknown)
        self._cache[key] = module
        return module

    def compile_file(self, path) -> CUDAModule:
        """Compile a .cu file by path.  Reads the file then calls ``compile()``."""
        source = Path(path).read_text()
        return self.compile(source)

    # Convenience: compile the unified SM102 source that ships with the repo
    def compile_sm102(self) -> CUDAModule:
        """Compile the bundled ``maxcore_sm102_unified.cu`` source."""
        here   = Path(__file__).parent
        # The unified file #includes the others, so read all three and concat
        sources = []
        for fname in ("flashattn_sm102.cu", "conv_sm102.cu", "reduction_sm102.cu"):
            p = here / fname
            if p.exists():
                sources.append(p.read_text())
        if not sources:
            return CUDAModule({}, "empty", list(_REGISTRY))
        return self.compile("\n".join(sources))

    # ------------------------------------------------------------------
    def register_kernel(self, name: str, fn: Callable):
        """Register an additional kernel implementation at runtime."""
        _REGISTRY[name] = fn
        # Invalidate any cached modules that parsed this name as unknown
        stale = [k for k, m in self._cache.items() if name in m.unknown]
        for k in stale:
            del self._cache[k]

    def list_kernels(self) -> list[str]:
        return sorted(_REGISTRY)

    def __repr__(self) -> str:
        return (f"CUDANvcc(gpu={type(self.gpu).__name__ if self.gpu else None}, "
                f"cached={len(self._cache)}, registry={len(_REGISTRY)})")
