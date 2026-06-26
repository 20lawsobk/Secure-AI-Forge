"""Tests for the MaxCore / DigitalGPU stack.

Runnable two ways:
  * pytest:  uv run pytest ai_model/maxcore/tests/test_maxcore.py
  * direct:  uv run python ai_model/maxcore/tests/test_maxcore.py

Every kernel and the end-to-end graph path are validated against independent
numpy ground truth. Numerical stability and PDIM dedup/single-flight behaviours
are asserted explicitly.
"""
from __future__ import annotations

import os
import sys
import threading
import time

import numpy as np

# Make the training-server root importable when run directly.
_SERVER_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if _SERVER_DIR not in sys.path:
    sys.path.insert(0, _SERVER_DIR)

from ai_model.maxcore import (  # noqa: E402
    Compiler,
    DigitalGPU,
    GraphBuilder,
    OpType,
    PDIMConfig,
    PDIMOrchestrator,
    PDIMStorage,
    available,
    available_runtime,
    get_backend,
)
from ai_model.maxcore.integration import (  # noqa: E402
    build_text_mlp_graph,
    mlp_graph,
    ref_attention,
    ref_mlp,
    ref_text_mlp,
)
from ai_model.maxcore.observability import METRICS  # noqa: E402

TOL = 1e-3


# ── IR ────────────────────────────────────────────────────────────────────────
def test_ir_build_validate_hash():
    b = GraphBuilder()
    x = b.add_input("x")
    w = b.const(np.eye(3, dtype=np.float32), "w")
    y = b.gemm(x, w)
    g = b.build(y)
    assert g.validate() is True
    assert [n.output for n in g.topo()] == [y]
    # structural hash is deterministic and structure-sensitive
    assert g.structural_hash() == g.structural_hash()
    b2 = GraphBuilder()
    x2 = b2.add_input("x")
    w2 = b2.const(np.eye(3, dtype=np.float32), "w")
    g2 = b2.build(b2.gemm(x2, w2))
    assert g.structural_hash() == g2.structural_hash()


def test_ir_rejects_undefined_input():
    b = GraphBuilder()
    b.add_input("x")
    try:
        b.gemm("x", "missing")
        raised = False
    except ValueError:
        raised = True
    assert raised


# ── compiler ────────────────────────────────────────────────────────────────
def test_compiler_fuses_gemm_add_relu():
    dg = DigitalGPU()
    rng = np.random.default_rng(0)
    w1 = rng.standard_normal((4, 5)).astype(np.float32)
    b1 = rng.standard_normal((5,)).astype(np.float32)
    w2 = rng.standard_normal((5, 2)).astype(np.float32)
    b2 = rng.standard_normal((2,)).astype(np.float32)
    g = mlp_graph(dg, w1, b1, w2, b2)
    raw_nodes = len(g.nodes)
    compiled = dg.compile(g)
    fused_nodes = len(compiled.order)
    # gemm+add+relu (3 nodes) collapse to 1; second gemm+add (2) collapse to 1
    assert fused_nodes < raw_nodes
    assert any("fuse" in s for s in compiled.pass_log)
    gemm_nodes = [n for n in compiled.order if n.op_type == OpType.GEMM]
    assert any(n.attrs.get("activation") == "relu" for n in gemm_nodes)

    # fused result must equal the unfused numpy reference
    x = rng.standard_normal((3, 4)).astype(np.float32)
    out = dg.run_graph(compiled, {"x": x})
    y = list(out.values())[0].numpy()
    assert np.allclose(y, ref_mlp(x, w1, b1, w2, b2), atol=TOL)


def test_compiler_cache_hit():
    dg = DigitalGPU()
    g = mlp_graph(dg, np.eye(3, dtype=np.float32), np.zeros(3, np.float32),
                  np.eye(3, dtype=np.float32), np.zeros(3, np.float32))
    c1 = dg.compile(g)
    c2 = dg.compile(g)
    assert c1 is c2  # served from compile cache


# ── backend kernels vs numpy ground truth ─────────────────────────────────────
def test_gemm_matches_numpy():
    dg = DigitalGPU()
    rng = np.random.default_rng(1)
    a = rng.standard_normal((6, 7)).astype(np.float32)
    b = rng.standard_normal((7, 5)).astype(np.float32)
    bias = rng.standard_normal((5,)).astype(np.float32)
    assert np.allclose(dg.gemm(a, b).numpy(), a @ b, atol=TOL)
    assert np.allclose(dg.gemm(a, b, bias=bias).numpy(), a @ b + bias, atol=TOL)
    assert np.allclose(dg.gemm(a, b, bias=bias, activation="relu").numpy(),
                       np.maximum(a @ b + bias, 0), atol=TOL)


def test_attention_matches_reference():
    dg = DigitalGPU()
    rng = np.random.default_rng(2)
    q = rng.standard_normal((2, 4, 8)).astype(np.float32)
    k = rng.standard_normal((2, 4, 8)).astype(np.float32)
    v = rng.standard_normal((2, 4, 8)).astype(np.float32)
    assert np.allclose(dg.attention(q, k, v).numpy(), ref_attention(q, k, v), atol=TOL)
    assert np.allclose(dg.attention(q, k, v, causal=True).numpy(),
                       ref_attention(q, k, v, causal=True), atol=TOL)


def test_conv2d_matches_naive():
    dg = DigitalGPU()
    rng = np.random.default_rng(3)
    x = rng.standard_normal((2, 3, 7, 7)).astype(np.float32)
    w = rng.standard_normal((4, 3, 3, 3)).astype(np.float32)
    bias = rng.standard_normal((4,)).astype(np.float32)
    out = dg.conv2d(x, w, bias=bias, stride=2, padding=1).numpy()
    ref = _naive_conv2d(x, w, bias, stride=2, padding=1)
    assert out.shape == ref.shape
    assert np.allclose(out, ref, atol=1e-2)


def _naive_conv2d(x, w, bias, stride, padding):
    x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    n, c, h, ww = x.shape
    o, _, kh, kw = w.shape
    ho = (h - kh) // stride + 1
    wo = (ww - kw) // stride + 1
    out = np.zeros((n, o, ho, wo), np.float32)
    for ni in range(n):
        for oi in range(o):
            for yi in range(ho):
                for xi in range(wo):
                    patch = x[ni, :, yi * stride:yi * stride + kh, xi * stride:xi * stride + kw]
                    out[ni, oi, yi, xi] = np.sum(patch * w[oi]) + bias[oi]
    return out


def test_reduce_ops():
    dg = DigitalGPU()
    x = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert np.allclose(dg.reduce(x, "sum", 1).numpy(), x.sum(1))
    assert np.allclose(dg.reduce(x, "mean", 0).numpy(), x.mean(0))
    assert np.allclose(dg.reduce(x, "max", 1).numpy(), x.max(1))


def test_softmax_is_numerically_stable():
    dg = DigitalGPU()
    big = np.array([[1000.0, 1001.0, 1002.0]], dtype=np.float32)
    out = dg.softmax(big).numpy()
    assert np.all(np.isfinite(out))            # naive exp(1002) would overflow
    assert np.allclose(out.sum(axis=-1), 1.0, atol=TOL)
    ref = np.exp(big - big.max()) / np.exp(big - big.max()).sum()
    assert np.allclose(out, ref, atol=TOL)


# ── end-to-end graph ──────────────────────────────────────────────────────────
def test_text_to_mlp_end_to_end():
    dg = DigitalGPU()
    rng = np.random.default_rng(4)
    vocab, dim, hidden, out_dim, T, B = 10, 6, 8, 3, 5, 2
    embed = rng.standard_normal((vocab, dim)).astype(np.float32)
    w1 = rng.standard_normal((dim, hidden)).astype(np.float32)
    b1 = rng.standard_normal((hidden,)).astype(np.float32)
    w2 = rng.standard_normal((hidden, out_dim)).astype(np.float32)
    b2 = rng.standard_normal((out_dim,)).astype(np.float32)
    onehot = np.zeros((B, T, vocab), np.float32)
    for bi in range(B):
        for ti in range(T):
            onehot[bi, ti, rng.integers(0, vocab)] = 1.0
    g = build_text_mlp_graph(dg, embed, w1, b1, w2, b2)
    out = dg.run_graph(g, {"onehot": onehot})
    y = list(out.values())[0].numpy()
    ref = ref_text_mlp(onehot, embed, w1, b1, w2, b2)
    assert y.shape == (B, out_dim)
    assert np.allclose(y, ref, atol=TOL)


def test_deterministic_run_repeatable():
    dg = DigitalGPU(deterministic=True)
    g = mlp_graph(dg, np.eye(4, dtype=np.float32), np.zeros(4, np.float32),
                  np.eye(4, dtype=np.float32), np.ones(4, np.float32))
    x = np.random.default_rng(5).standard_normal((2, 4)).astype(np.float32)
    a = list(dg.run_graph(g, {"x": x}, {"seed": 7}).values())[0].numpy()
    b = list(dg.run_graph(g, {"x": x}, {"seed": 7}).values())[0].numpy()
    assert np.array_equal(a, b)


# ── backends registry / honesty ──────────────────────────────────────────────
def test_registry_and_future_backends_are_honest():
    assert "cpu" in available()
    runnable = available_runtime()
    assert runnable["cpu"] is True
    assert runnable["gpu"] is False
    gpu = get_backend("gpu")
    raised = False
    try:
        gpu.gemm(None, None)
    except NotImplementedError as e:
        raised = "Triton" in str(e) or "not implemented" in str(e)
    assert raised


# ── PDIM ──────────────────────────────────────────────────────────────────────
def test_pdim_dedup_cache_hit():
    orch = PDIMOrchestrator()
    calls = {"n": 0}

    def compute(req):
        calls["n"] += 1
        return {"value": req["topic"].upper()}

    req = {"topic": "drop a single", "id": "abc"}        # 'id' is volatile metadata
    r1 = orch.compute(req, compute, namespace="test_dedup")
    r2 = orch.compute({"topic": "drop a single", "id": "xyz"}, compute, namespace="test_dedup")
    assert r1["source"] == "compute"
    assert r2["source"] == "cache"           # different id, same semantic request
    assert r2["result"]["value"] == "DROP A SINGLE"
    assert calls["n"] == 1


def test_pdim_single_flight_collapses_concurrent():
    orch = PDIMOrchestrator()
    calls = {"n": 0}
    started = threading.Event()

    def compute(req):
        calls["n"] += 1
        started.set()
        time.sleep(0.4)                       # hold so others pile up behind it
        return {"v": 1}

    results = []

    def worker():
        results.append(orch.compute({"k": "same"}, compute, namespace="test_sf"))

    threads = [threading.Thread(target=worker) for _ in range(6)]
    threads[0].start()
    started.wait(1.0)
    for t in threads[1:]:
        t.start()
    for t in threads:
        t.join(5.0)
    assert calls["n"] == 1                    # exactly one real computation
    assert len(results) == 6
    assert sum(1 for r in results if r["source"] == "coalesced") >= 1


def test_pdim_durable_queue_roundtrip(tmp_path_factory=None):
    import tempfile
    base = tempfile.mkdtemp(prefix="pdim_test_")
    cfg = PDIMConfig(namespace="test_q", batch_size=8, base_dir=base)
    storage = PDIMStorage(config=cfg)            # in-process store fallback
    orch = PDIMOrchestrator(storage=storage, config=cfg)

    sub = orch.submit(model_id="m", prompt="hello", params={"mode": "x"}, context_sig="s1")
    assert sub["source"] == "queued"
    h = sub["hash"]
    assert orch.poll(h)["status"] == "pending"

    def compute(job):
        return {"echo": job["prompt"], "mode": job["params"]["mode"]}

    n = orch.process_queue_once(compute, queue="default")
    assert n == 1
    polled = orch.poll(h)
    assert polled["status"] == "done"
    assert polled["result"]["echo"] == "hello"

    # second identical submit short-circuits to cached result
    sub2 = orch.submit(model_id="m", prompt="hello", params={"mode": "x"}, context_sig="s1")
    assert sub2["source"] == "cache"


def test_pdim_preview_policy_pluggable():
    import tempfile
    base = tempfile.mkdtemp(prefix="pdim_prev_")
    cfg = PDIMConfig(namespace="test_prev", base_dir=base)
    storage = PDIMStorage(config=cfg)
    orch = PDIMOrchestrator(storage=storage, config=cfg)
    orch.submit(model_id="m", prompt="p", params={}, context_sig="c")

    def preview(job):
        return {"q": 0.9, "kind": "preview"}

    def quality(prev, job):
        return prev["q"] >= 0.8               # accept good previews

    def full(job):
        raise AssertionError("full compute should not run when preview accepted")

    n = orch.process_queue_once(full, preview_fn=preview, quality_fn=quality)
    assert n == 1


# ── engine coverage: heavy paths must be engine-served (no numpy / fallback) ──
def _counter(name):
    return METRICS.snapshot()["counters"].get(name, 0)


def test_batched_gemm_is_engine_served():
    dg = DigitalGPU()
    rng = np.random.default_rng(11)
    A = rng.standard_normal((3, 5, 7)).astype(np.float32)    # [B, M, K]
    W = rng.standard_normal((7, 4)).astype(np.float32)       # [K, N]    (shared)
    Bb = rng.standard_normal((3, 7, 4)).astype(np.float32)   # [B, K, N] (batched)
    n0, f0 = _counter("cpu.gemm.numpy"), _counter("cpu.gemm.engine_fallback")
    o1 = dg.gemm(A, W).numpy()
    o2 = dg.gemm(A, Bb).numpy()
    assert np.allclose(o1, A @ W, atol=TOL)
    assert np.allclose(o2, A @ Bb, atol=TOL)
    assert _counter("cpu.gemm.numpy") == n0              # never bypassed to numpy
    assert _counter("cpu.gemm.engine_fallback") == f0   # engine served it all


def test_softmax_any_axis_is_engine_served():
    dg = DigitalGPU()
    rng = np.random.default_rng(12)
    X = rng.standard_normal((2, 6, 4)).astype(np.float32)
    n0, f0 = _counter("cpu.softmax.numpy"), _counter("cpu.softmax.engine_fallback")
    out = dg.softmax(X, axis=1).numpy()                  # non-last axis
    ref = np.exp(X - X.max(1, keepdims=True))
    ref = ref / ref.sum(1, keepdims=True)
    assert np.allclose(out, ref, atol=TOL)
    assert np.allclose(out.sum(axis=1), 1.0, atol=TOL)
    assert _counter("cpu.softmax.numpy") == n0
    assert _counter("cpu.softmax.engine_fallback") == f0


def test_masked_multihead_attention_is_engine_served():
    dg = DigitalGPU()
    rng = np.random.default_rng(13)
    B, H, Tq, Tk, D = 2, 2, 5, 5, 8
    q = rng.standard_normal((B, H, Tq, D)).astype(np.float32)
    k = rng.standard_normal((B, H, Tk, D)).astype(np.float32)
    v = rng.standard_normal((B, H, Tk, D)).astype(np.float32)
    mask = np.where(rng.random((Tq, Tk)) < 0.3, -1e9, 0.0).astype(np.float32)
    n0, f0 = _counter("cpu.attention.numpy"), _counter("cpu.attention.engine_fallback")
    out = dg.attention(q, k, v, mask=mask).numpy()       # masked, 4D multi-head
    scores = (q @ np.swapaxes(k, -1, -2)) / np.sqrt(D) + mask
    p = np.exp(scores - scores.max(-1, keepdims=True))
    p = p / p.sum(-1, keepdims=True)
    ref = p @ v
    assert out.shape == (B, H, Tq, D)
    assert np.allclose(out, ref, atol=1e-2)
    assert _counter("cpu.attention.numpy") == n0         # masked path stayed on engine
    assert _counter("cpu.attention.engine_fallback") == f0


# ── manual runner ─────────────────────────────────────────────────────────────
def _run_all():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            passed += 1
            print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            import traceback
            print(f"FAIL {t.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed, {len(tests)} total")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
