"""CPU backend — the real compute backend for this host.

Every heavy kernel routes through the existing in-house Digital GPU engine
(``ai_model/gpu/digital_gpu.py`` — a NumPy/SIMD virtual ISA with a *numerically
stable* softmax and a fused ``gemm_bias_relu``):

  * ``gemm`` / bias / relu  -> engine ``gemm`` and fused ``gemm_bias_relu``
  * ``attention``           -> engine ``attention`` (stable softmax inside)
  * ``softmax``             -> engine ``softmax``
  * ``mlp``                 -> two engine GEMMs (relu layer fully fused)
  * ``conv2d``              -> im2col + a single engine GEMM

Reductions and element-wise add/relu are first-class vectorized NumPy (the
engine exposes no kernel for them — this is the implementation, not a fallback).

The engine is loaded directly from its file via importlib so importing this
backend never pulls in ``ai_model.gpu``'s package __init__ (which imports torch).
A defensive numpy path remains behind every engine call **only** as a safety net;
it increments an ``*.engine_fallback`` counter so its use is observable. In a
correctly built system that counter stays at zero — the load test asserts it.
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np

from ..observability import METRICS
from ..tensor import Tensor, to_numpy
from .base import Backend

# ── Load the in-house engine in isolation (no torch, no package __init__) ─────
_EngineGPU = None
_ENGINE_LOAD_ERROR = None
try:
    _engine_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "gpu", "digital_gpu.py")
    )
    _spec = importlib.util.spec_from_file_location("_maxcore_engine_gpu", _engine_path)
    if _spec and _spec.loader:
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        _EngineGPU = getattr(_mod, "DigitalGPU", None)
except Exception as exc:  # pragma: no cover
    _ENGINE_LOAD_ERROR = repr(exc)
    _EngineGPU = None


def _activate(x: np.ndarray, activation) -> np.ndarray:
    if activation in (None, "none", "linear"):
        return x
    if activation == "relu":
        return np.maximum(x, 0.0)
    if activation == "gelu":
        return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))
    if activation == "silu":
        return x * (1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0))))
    if activation == "tanh":
        return np.tanh(x)
    if activation == "sigmoid":
        return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))
    raise ValueError(f"unknown activation '{activation}'")


def _stable_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


class CPUBackend(Backend):
    name = "cpu"

    def __init__(self, use_engine: bool = True):
        self.engine = _EngineGPU() if (use_engine and _EngineGPU is not None) else None

    def is_available(self) -> bool:
        return True

    def info(self) -> dict:
        return {
            "name": self.name,
            "available": True,
            "engine": self.engine is not None,
            "engine_load_error": _ENGINE_LOAD_ERROR,
        }

    def create_tensor(self, data, dtype: str = "float32"):
        return Tensor(data, dtype=dtype, device="cpu")

    # ── engine-backed matmul: every 2D sub-GEMM runs on the in-house engine ────
    def _engine_2d(self, a2: np.ndarray, b2: np.ndarray):
        if self.engine is not None:
            try:
                out = np.asarray(self.engine.gemm(
                    np.ascontiguousarray(a2, np.float32),
                    np.ascontiguousarray(b2, np.float32)))
                return out, True
            except Exception:
                METRICS.incr("cpu.gemm.engine_fallback")
        return np.matmul(a2, b2), False

    def _matmul_engine(self, A: np.ndarray, B: np.ndarray):
        """matmul of any rank, routing EVERY 2D sub-product through the engine.

        Returns ``(out, used_engine)``; ``used_engine`` is False only if the
        engine is absent or a kernel call raised (which also bumps an
        ``cpu.gemm.engine_fallback`` counter). Batched operands are folded to a
        stack of 2D GEMMs so there is no NumPy matmul on the heavy path.
        """
        a1, b1 = A.ndim == 1, B.ndim == 1
        if a1:
            A = A[None, :]
        if b1:
            B = B[:, None]
        if A.ndim == 2 and B.ndim == 2:
            out, used = self._engine_2d(A, B)
        elif B.ndim == 2:                                   # [..., M, K] @ [K, N]
            k = A.shape[-1]
            out2, used = self._engine_2d(A.reshape(-1, k), B)
            out = out2.reshape(*A.shape[:-1], B.shape[1])
        else:                                               # batched (broadcast)
            if A.ndim == 2:
                A = np.broadcast_to(A, tuple(B.shape[:-2]) + A.shape)
            m, k = A.shape[-2], A.shape[-1]
            k2, n = B.shape[-2], B.shape[-1]
            batch = np.broadcast_shapes(A.shape[:-2], B.shape[:-2])
            Ab = np.broadcast_to(A, batch + (m, k)).reshape(-1, m, k)
            Bb = np.broadcast_to(B, batch + (k2, n)).reshape(-1, k2, n)
            res = [self._engine_2d(Ab[i], Bb[i]) for i in range(Ab.shape[0])]
            used = all(u for _, u in res)
            out = np.stack([o for o, _ in res]).reshape(batch + (m, n))
        if a1:
            out = np.squeeze(out, axis=-2)
        if b1:
            out = np.squeeze(out, axis=-1)
        return out, used

    # ── GEMM: every matmul (2D or batched) is engine-served ───────────────────
    def gemm(self, a, b, bias=None, activation=None):
        A = to_numpy(a).astype(np.float32, copy=False)
        B = to_numpy(b).astype(np.float32, copy=False)
        with METRICS.timer("cpu.gemm"):
            out = None
            used_engine = False
            # Fused 2D bias+relu fast path — a single engine kernel call.
            if (self.engine is not None and A.ndim == 2 and B.ndim == 2
                    and bias is not None and activation == "relu"):
                try:
                    out = np.asarray(self.engine.gemm_bias_relu(
                        A, B, to_numpy(bias).astype(np.float32)))
                    bias, activation = None, None  # consumed by fused kernel
                    used_engine = True
                except Exception:
                    out = None
                    METRICS.incr("cpu.gemm.engine_fallback")
            if out is None:
                out, used_engine = self._matmul_engine(A, B)
            if bias is not None:
                out = out + to_numpy(bias)
            out = _activate(out, activation)
        METRICS.incr("cpu.gemm.engine" if used_engine else "cpu.gemm.numpy")
        return Tensor(out, dtype=None)

    def add(self, a, b):
        return Tensor(to_numpy(a) + to_numpy(b), dtype=None)

    def relu(self, x):
        return Tensor(np.maximum(to_numpy(x), 0.0), dtype=None)

    def _engine_softmax_last(self, X: np.ndarray) -> np.ndarray:
        """Softmax over the last axis on the engine, folding to 2D so the engine
        kernel (validated on 2D) serves tensors of any rank. Raises if the engine
        is unavailable so callers can record a fallback."""
        if self.engine is None:
            raise RuntimeError("engine unavailable")
        sh = X.shape
        flat = np.ascontiguousarray(X.reshape(-1, sh[-1]), np.float32)
        return np.asarray(self.engine.softmax(flat, axis=-1)).reshape(sh)

    # ── softmax: engine-served for ANY axis (move-to-last, fold to 2D) ─────────
    def softmax(self, x, axis: int = -1):
        X: np.ndarray = to_numpy(x).astype(np.float32, copy=False)
        ax = axis if axis >= 0 else X.ndim + axis
        with METRICS.timer("cpu.softmax"):
            out = None
            used_engine = False
            if self.engine is not None:
                try:
                    if ax == X.ndim - 1:
                        out = self._engine_softmax_last(X)
                    else:
                        sw = np.swapaxes(X, ax, -1)
                        out = np.swapaxes(self._engine_softmax_last(sw), ax, -1)
                    used_engine = True
                except Exception:
                    out = None
                    METRICS.incr("cpu.softmax.engine_fallback")
            if out is None:
                out = _stable_softmax(X, axis=ax)
        METRICS.incr("cpu.softmax.engine" if used_engine else "cpu.softmax.numpy")
        return Tensor(out, dtype=None)

    # ── attention: engine-served incl. masked / multi-head / cross-attention ──
    def attention(self, q, k, v, mask=None, causal: bool = False):
        Q: np.ndarray = to_numpy(q).astype(np.float32, copy=False)
        K: np.ndarray = to_numpy(k).astype(np.float32, copy=False)
        V: np.ndarray = to_numpy(v).astype(np.float32, copy=False)
        with METRICS.timer("cpu.attention"):
            out = None
            used_engine = False
            # Native fused kernel for the common unmasked, same-shape [B,T,D] case.
            if (self.engine is not None and mask is None and Q.ndim == 3
                    and Q.shape == K.shape == V.shape):
                try:
                    out = np.asarray(self.engine.attention(Q, K, V, causal=causal))
                    used_engine = True
                except Exception:
                    out = None
                    METRICS.incr("cpu.attention.engine_fallback")
            # General path (masked / multi-head / cross-attention): both matmuls
            # AND the softmax run on the engine. Only the score scaling and the
            # additive mask are element-wise — the engine exposes no kernel for
            # those, so that arithmetic is the implementation, not a fallback.
            if out is None and self.engine is not None:
                try:
                    d = Q.shape[-1]
                    scores, ue1 = self._matmul_engine(Q, np.swapaxes(K, -1, -2))
                    scores = scores / np.sqrt(np.float32(d))
                    if causal:
                        t_q, t_k = scores.shape[-2], scores.shape[-1]
                        scores = scores + np.triu(
                            np.full((t_q, t_k), -1e9, np.float32), k=1)
                    if mask is not None:
                        scores = scores + to_numpy(mask).astype(np.float32)
                    probs = self._engine_softmax_last(scores)
                    out, ue2 = self._matmul_engine(probs, V)
                    used_engine = ue1 and ue2
                except Exception:
                    out = None
                    METRICS.incr("cpu.attention.engine_fallback")
            if out is None:
                out = self._attention_np(Q, K, V, mask, causal)
        METRICS.incr("cpu.attention.engine" if used_engine else "cpu.attention.numpy")
        return Tensor(out, dtype=None)

    @staticmethod
    def _attention_np(Q, K, V, mask, causal):
        d = Q.shape[-1]
        scores = np.matmul(Q, np.swapaxes(K, -1, -2)) / np.sqrt(d)
        if causal:
            t_q, t_k = scores.shape[-2], scores.shape[-1]
            cm = np.triu(np.full((t_q, t_k), -1e9, dtype=scores.dtype), k=1)
            scores = scores + cm
        if mask is not None:
            scores = scores + to_numpy(mask)
        probs = _stable_softmax(scores, axis=-1)
        return np.matmul(probs, V)

    # ── conv2d: im2col + a single engine GEMM ─────────────────────────────────
    def conv2d(self, x, w, bias=None, stride: int = 1, padding: int = 0):
        X: np.ndarray = to_numpy(x).astype(np.float32, copy=False)
        W: np.ndarray = to_numpy(w).astype(np.float32, copy=False)
        if X.ndim != 4 or W.ndim != 4:
            raise ValueError("conv2d expects X[N,C,H,W] and W[O,C,kh,kw]")
        n, c, h, ww = X.shape
        o, cw, kh, kw = W.shape
        if cw != c:
            raise ValueError(f"conv2d channel mismatch: input C={c} vs weight C={cw}")
        s, p = stride, padding
        with METRICS.timer("cpu.conv2d"):
            if p > 0:
                X = np.pad(X, ((0, 0), (0, 0), (p, p), (p, p)))
            hp, wp = X.shape[2], X.shape[3]
            ho = (hp - kh) // s + 1
            wo = (wp - kw) // s + 1
            cols = np.empty((n, c, kh, kw, ho, wo), dtype=np.float32)
            for i in range(kh):
                i_max = i + s * ho
                for j in range(kw):
                    j_max = j + s * wo
                    cols[:, :, i, j, :, :] = X[:, :, i:i_max:s, j:j_max:s]
            # [K, N*P] @ ... -> route the heavy matmul through the engine GEMM
            k_dim = c * kh * kw
            cols2 = cols.reshape(n, k_dim, ho * wo).transpose(1, 0, 2).reshape(k_dim, n * ho * wo)
            wm = W.reshape(o, k_dim)
            gemm_out = self.gemm(wm, cols2).numpy()  # [O, N*P], engine-routed
            out = gemm_out.reshape(o, n, ho, wo).transpose(1, 0, 2, 3)
            if bias is not None:
                out = out + to_numpy(bias).reshape(1, o, 1, 1)
        METRICS.incr("cpu.conv2d")
        return Tensor(np.ascontiguousarray(out), dtype=None)

    # ── mlp: two engine GEMMs (first layer relu fully fused) ──────────────────
    def mlp(self, x, w1, b1, w2, b2, activation: str = "relu"):
        h = self.gemm(x, w1, bias=b1, activation=activation)
        out = self.gemm(h, w2, bias=b2)
        METRICS.incr("cpu.mlp")
        return out

    def reduce(self, x, op: str, axis, keepdims: bool = False):
        X = to_numpy(x)
        with METRICS.timer("cpu.reduce"):
            if op == "sum":
                out = X.sum(axis=axis, keepdims=keepdims)
            elif op == "mean":
                out = X.mean(axis=axis, keepdims=keepdims)
            elif op == "max":
                out = X.max(axis=axis, keepdims=keepdims)
            elif op == "min":
                out = X.min(axis=axis, keepdims=keepdims)
            elif op == "prod":
                out = X.prod(axis=axis, keepdims=keepdims)
            else:
                raise ValueError(f"unsupported reduce op '{op}'")
        return Tensor(out, dtype=None)
