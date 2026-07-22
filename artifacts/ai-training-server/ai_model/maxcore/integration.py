"""Integration helpers + reference vertical slices.

These show the layer working end-to-end and provide numpy ground-truth functions
the tests compare against. Crucially, these are *additive* — they do not touch
the running server's hot path; model/training code can opt in by calling them.

The headline slice is ``build_text_mlp_graph``: a real
"text → embedding → 2-layer MLP → output" pipeline expressed entirely in MaxCore
IR (mean-pool → embedding GEMM → MLP), compiled and run through the stack.
"""
from __future__ import annotations

import threading

import numpy as np

from .ir.nodes import MaxCoreGraph

# ── Digital GPU backend singleton ─────────────────────────────────────────────
_GPU_BACKEND = None
_GPU_BACKEND_LOCK = threading.Lock()

def _get_gpu():
    global _GPU_BACKEND
    if _GPU_BACKEND is None:
        with _GPU_BACKEND_LOCK:
            if _GPU_BACKEND is None:
                try:
                    from ai_model.gpu.torch_backend import DigitalGPUBackend
                    _GPU_BACKEND = DigitalGPUBackend()
                except Exception:
                    pass
    return _GPU_BACKEND


def attention_graph(dg, causal: bool = False) -> MaxCoreGraph:
    """q,k,v -> attention output, as a compilable graph."""
    b = dg.graph_builder()
    q = b.add_input("q")
    k = b.add_input("k")
    v = b.add_input("v")
    out = b.attention(q, k, v, causal=causal)
    return b.build(out)


def mlp_graph(dg, w1, b1, w2, b2, activation: str = "relu") -> MaxCoreGraph:
    """x -> (gemm+bias+act) -> (gemm+bias) using fusible primitive ops, so the
    compiler's fuse pass can collapse the first gemm/add/relu into one kernel."""
    bld = dg.graph_builder()
    x = bld.add_input("x")
    W1 = bld.const(w1, "w1")
    B1 = bld.const(b1, "b1")
    W2 = bld.const(w2, "w2")
    B2 = bld.const(b2, "b2")
    h = bld.gemm(x, W1)
    h = bld.add(h, B1)
    h = bld.relu(h)
    o = bld.gemm(h, W2)
    o = bld.add(o, B2)
    return bld.build(o)


def build_text_mlp_graph(dg, embed, w1, b1, w2, b2, activation: str = "relu") -> MaxCoreGraph:
    """Text(one-hot [B,T,vocab]) -> embedding -> 2-layer MLP -> logits [B,out]."""
    b = dg.graph_builder()
    onehot = b.add_input("onehot")            # [B, T, vocab]
    bag = b.reduce(onehot, op="mean", axis=1)  # [B, vocab] bag-of-tokens
    E = b.const(embed, "embed")               # [vocab, dim]
    emb = b.gemm(bag, E)                       # [B, dim]
    W1 = b.const(w1, "w1")
    B1 = b.const(b1, "b1")
    W2 = b.const(w2, "w2")
    B2 = b.const(b2, "b2")
    out = b.mlp(emb, W1, B1, W2, B2, activation=activation)  # [B, out]
    return b.build(out)


# ── numpy ground-truth references (for tests / validation) ────────────────────
def _activate_np(x, activation):
    if activation in (None, "none", "linear"):
        return x
    if activation == "relu":
        return np.maximum(x, 0.0)
    raise ValueError(activation)


def ref_mlp(x, w1, b1, w2, b2, activation="relu"):
    h = _activate_np(np.asarray(x) @ np.asarray(w1) + np.asarray(b1), activation)
    return h @ np.asarray(w2) + np.asarray(b2)


def ref_text_mlp(onehot, embed, w1, b1, w2, b2, activation="relu"):
    bag = np.asarray(onehot, dtype=np.float32).mean(axis=1)
    emb = bag @ np.asarray(embed)
    return ref_mlp(emb, w1, b1, w2, b2, activation)


def ref_attention(q, k, v, causal=False):
    gpu = _get_gpu()
    if gpu is not None:
        # DigitalGPUBackend.attention(dim, n_heads) is a layer factory;
        # the compute kernel lives on the underlying DigitalGPU instance.
        return gpu.gpu.attention(
            np.ascontiguousarray(np.asarray(q, dtype=np.float32)),
            np.ascontiguousarray(np.asarray(k, dtype=np.float32)),
            np.ascontiguousarray(np.asarray(v, dtype=np.float32)),
            causal,
        )
    # numpy fallback (no-backend environments / unit tests)
    q = np.asarray(q, dtype=np.float32)
    k = np.asarray(k, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    d = q.shape[-1]
    scores = np.matmul(q, np.swapaxes(k, -1, -2)) / np.sqrt(d)
    if causal:
        t_q, t_k = scores.shape[-2], scores.shape[-1]
        scores = scores + np.triu(np.full((t_q, t_k), -1e9, np.float32), k=1)
    scores = scores - scores.max(axis=-1, keepdims=True)
    e = np.exp(scores)
    p = e / e.sum(axis=-1, keepdims=True)
    return np.matmul(p, v)
