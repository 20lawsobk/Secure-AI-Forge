"""Real device (GPU) backend — dispatches the MaxCore kernel contract to torch.

Unlike the ``future_backends`` stubs (which raise ``NotImplementedError`` by
design), this is a *working* backend: every kernel in the ``Backend`` contract
executes on a real torch device. It is registered as the ``"gpu"`` backend.

Honesty rules this backend obeys:
  * ``is_available()`` returns ``torch.cuda.is_available()``. On a Digital GPU
    host (like this one) it is therefore ``False`` — there is no pretending.
  * If a kernel is called while the requested device is absent, it raises a
    clear, hardware-honest error naming the missing device. It never silently
    falls back and reports itself as something it is not.
  * Numerics match ``DigitalGPUBackend`` within floating-point tolerance, so a
    graph produces the same result on either backend (the runtime can pick
    whichever hardware is present).

Validating without a CUDA GPU: pass ``device="cpu"`` to run the exact same code
path on torch-CPU. That is how the implementation is proven correct against
``DigitalGPUBackend`` here; the day this is deployed on a CUDA host,
``device="cuda"`` runs the identical kernels on the GPU.

Torch is imported lazily (only when this backend is constructed or probed) so
the default Digital GPU backend path never pays torch's import cost.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..tensor import Tensor, to_numpy
from .base import Backend

_TORCH = None


def _torch():
    """Import torch on first use and cache the module."""
    global _TORCH
    if _TORCH is None:
        import torch  # local import: keep torch off the default Digital GPU path
        _TORCH = torch
    return _TORCH


def cuda_is_available() -> bool:
    """True only if a real CUDA device is present. Never raises."""
    try:
        return bool(_torch().cuda.is_available())
    except Exception:
        return False


class GPUBackend(Backend):
    """Executes the MaxCore kernel contract on a real torch device.

    ``device`` defaults to ``"cuda"`` (the whole point of a GPU backend). Pass
    ``"cpu"`` to validate the identical code path without a GPU.
    """

    name = "gpu"

    def __init__(self, device: str = "cuda"):
        self._requested = device

    # ── device / availability ────────────────────────────────────────────────
    def is_available(self) -> bool:
        if self._requested == "cuda":
            return cuda_is_available()
        # An explicit cpu device is always runnable (used for validation).
        if self._requested == "cpu":
            return True
        # Any other device string is not something we can truthfully claim.
        return False

    def _device(self):
        torch = _torch()
        if self._requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "gpu backend: no CUDA device available on this host. This backend "
                "dispatches the MaxCore kernel contract to a real GPU via torch; "
                "deploy on a CUDA-capable host to run it. On this Digital GPU node the "
                "runnable backend is 'digital_gpu' (get_backend('digital_gpu')). To "
                "validate this backend's code path without a GPU, construct "
                "GPUBackend(device='cpu')."
            )
        return torch.device(self._requested)

    def info(self) -> dict:
        avail = self.is_available()
        out = {"name": self.name, "requested_device": self._requested, "available": avail}
        if avail:
            torch = _torch()
            if self._requested == "cuda" and torch.cuda.is_available():
                out["device_name"] = torch.cuda.get_device_name(0)
                out["device_count"] = torch.cuda.device_count()
        return out

    # ── helpers ───────────────────────────────────────────────────────────────
    def _t(self, x: Any):
        """Array-like -> float32 torch tensor on the resolved device."""
        torch = _torch()
        arr = to_numpy(x).astype(np.float32, copy=False)
        return torch.from_numpy(np.ascontiguousarray(arr)).to(self._device())

    @staticmethod
    def _out(t) -> Tensor:
        """Realize a device tensor back into a backend-agnostic Tensor."""
        return Tensor(t.detach().to("cpu").numpy(), dtype=None)

    def _activate(self, t, activation):
        if activation in (None, "none", "linear"):
            return t
        torch = _torch()
        if activation == "relu":
            return torch.clamp(t, min=0.0)
        if activation == "gelu":
            # tanh approximation — matches DigitalGPUBackend's gelu formula.
            return torch.nn.functional.gelu(t, approximate="tanh")
        if activation == "silu":
            return t * torch.sigmoid(torch.clamp(t, -60.0, 60.0))
        if activation == "tanh":
            return torch.tanh(t)
        if activation == "sigmoid":
            return torch.sigmoid(torch.clamp(t, -60.0, 60.0))
        raise ValueError(f"unknown activation '{activation}'")

    # ── contract kernels ──────────────────────────────────────────────────────
    def create_tensor(self, data, dtype: str = "float32"):
        return Tensor(data, dtype=dtype, device=self._requested)

    def gemm(self, a, b, bias=None, activation=None):
        torch = _torch()
        A = self._t(a)
        B = self._t(b)
        out = torch.matmul(A, B)
        if bias is not None:
            out = out + self._t(bias)
        out = self._activate(out, activation)
        return self._out(out)

    def add(self, a, b):
        return self._out(self._t(a) + self._t(b))

    def relu(self, x):
        torch = _torch()
        return self._out(torch.clamp(self._t(x), min=0.0))

    def softmax(self, x, axis: int = -1):
        torch = _torch()
        out = torch.softmax(self._t(x), dim=axis)
        return self._out(out)

    def attention(self, q, k, v, mask=None, causal: bool = False):
        torch = _torch()
        Q = self._t(q)
        K = self._t(k)
        V = self._t(v)
        d = Q.shape[-1]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / float(np.sqrt(np.float32(d)))
        if causal:
            t_q, t_k = scores.shape[-2], scores.shape[-1]
            cm = torch.triu(
                torch.full((t_q, t_k), -1e9, dtype=scores.dtype, device=scores.device),
                diagonal=1,
            )
            scores = scores + cm
        if mask is not None:
            scores = scores + self._t(mask)
        probs = torch.softmax(scores, dim=-1)
        return self._out(torch.matmul(probs, V))

    def conv2d(self, x, w, bias=None, stride: int = 1, padding: int = 0):
        torch = _torch()
        X = self._t(x)
        W = self._t(w)
        if X.ndim != 4 or W.ndim != 4:
            raise ValueError("conv2d expects X[N,C,H,W] and W[O,C,kh,kw]")
        if W.shape[1] != X.shape[1]:
            raise ValueError(
                f"conv2d channel mismatch: input C={X.shape[1]} vs weight C={W.shape[1]}"
            )
        b = None if bias is None else self._t(bias)
        out = torch.nn.functional.conv2d(X, W, bias=b, stride=stride, padding=padding)
        return self._out(out)

    def mlp(self, x, w1, b1, w2, b2, activation: str = "relu"):
        h = self.gemm(x, w1, bias=b1, activation=activation)
        return self.gemm(h, w2, bias=b2)

    def reduce(self, x, op: str, axis, keepdims: bool = False):
        torch = _torch()
        X = self._t(x)
        dim = axis
        if op == "sum":
            out = X.sum(dim=dim, keepdim=keepdims)
        elif op == "mean":
            out = X.mean(dim=dim, keepdim=keepdims)
        elif op == "max":
            out = X.amax(dim=dim, keepdim=keepdims)
        elif op == "min":
            out = X.amin(dim=dim, keepdim=keepdims)
        elif op == "prod":
            # torch.prod takes a single int dim; fold multi-axis prod (which
            # numpy/DigitalGPUBackend support) into sequential reductions. Reduce the
            # highest axis first so the remaining axis indices stay valid.
            if isinstance(dim, (tuple, list)):
                out = X
                for d in sorted((a % X.ndim for a in dim), reverse=True):
                    out = out.prod(dim=d, keepdim=keepdims)
            else:
                out = X.prod(dim=dim, keepdim=keepdims)
        else:
            raise ValueError(f"unsupported reduce op '{op}'")
        return self._out(out)
