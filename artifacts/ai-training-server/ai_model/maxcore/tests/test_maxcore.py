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
    assert "digital_gpu" in available()
    runnable = available_runtime()
    assert runnable["digital_gpu"] is True
    # No CUDA device on this host, so the (real) gpu backend is not runnable and
    # must say so rather than silently pretend to be a GPU.
    assert runnable["gpu"] is False
    # cluster/asic remain honest plug-points: kernels raise NotImplementedError.
    cluster = get_backend("cluster")
    raised = False
    try:
        cluster.gemm(None, None)
    except NotImplementedError as e:
        raised = "not implemented" in str(e)
    assert raised


def test_gpu_backend_is_real_but_honest_without_hardware():
    """The gpu backend is a genuine torch implementation, not a stub. Without a
    CUDA device it reports unavailable and raises a clear, hardware-honest error
    (naming the missing device) instead of faking GPU compute on the Digital GPU engine."""
    import numpy as _np

    from ai_model.maxcore.backend.device_backend import GPUBackend, cuda_is_available

    gpu = get_backend("gpu")
    if cuda_is_available():
        # On a real GPU host the kernel runs and matches the Digital GPU backend.
        dgpu = get_backend("digital_gpu")
        a = _np.random.default_rng(0).standard_normal((4, 5)).astype(_np.float32)
        b = _np.random.default_rng(1).standard_normal((5, 3)).astype(_np.float32)
        assert _np.allclose(gpu.gemm(a, b).numpy(), dgpu.gemm(a, b).numpy(), atol=1e-3)
    else:
        assert gpu.is_available() is False
        raised = False
        try:
            gpu.gemm(_np.zeros((2, 2), _np.float32), _np.zeros((2, 2), _np.float32))
        except RuntimeError as e:
            raised = "CUDA" in str(e) or "no CUDA device" in str(e)
        assert raised

    # The same code path is validated on torch-CPU (the correctness proof for
    # the kernels that will run on the GPU): it matches DigitalGPUBackend numerics.
    dgpu = get_backend("digital_gpu")
    val = GPUBackend(device="cpu")
    a = _np.random.default_rng(2).standard_normal((6, 7)).astype(_np.float32)
    b = _np.random.default_rng(3).standard_normal((7, 4)).astype(_np.float32)
    assert _np.allclose(val.gemm(a, b).numpy(), dgpu.gemm(a, b).numpy(), atol=1e-3)


def test_gpu_backend_reduce_parity_with_digital_gpu():
    """reduce() on the gpu backend must match DigitalGPUBackend for all ops and
    for multi-axis reductions (numpy supports tuple axes; torch's dim does not,
    so the backend folds them — this guards that parity)."""
    import numpy as _np

    from ai_model.maxcore.backend.device_backend import GPUBackend

    dgpu = get_backend("digital_gpu")
    val = GPUBackend(device="cpu")
    x = _np.random.default_rng(9).standard_normal((3, 4, 5)).astype(_np.float32)
    for op in ("sum", "mean", "max", "min", "prod"):
        for axis in (1, (1, 2), (0, 2)):
            for keepdims in (False, True):
                r_gpu = val.reduce(x, op, axis=axis, keepdims=keepdims).numpy()
                r_dgpu = dgpu.reduce(x, op, axis=axis, keepdims=keepdims).numpy()
                assert r_gpu.shape == r_dgpu.shape, (op, axis, keepdims)
                assert _np.allclose(r_gpu, r_dgpu, atol=1e-3, rtol=1e-3), (op, axis, keepdims)


# ── PDIM ──────────────────────────────────────────────────────────────────────
# Namespaces are unique per run: the fleet dedup store persists across test
# runs when pdim storage is online, so fixed namespaces would replay results.
import uuid as _uuid  # noqa: E402

_RUN = _uuid.uuid4().hex[:8]


def test_pdim_dedup_cache_hit():
    orch = PDIMOrchestrator()
    calls = {"n": 0}
    ns = f"test_dedup_{_RUN}"

    def compute(req):
        calls["n"] += 1
        return {"value": req["topic"].upper()}

    req = {"topic": "drop a single", "id": "abc"}        # 'id' is volatile metadata
    r1 = orch.compute(req, compute, namespace=ns)
    assert r1["source"] == "compute"
    # The leader persists via a background thread — poll until it lands.
    deadline = time.time() + 2.0
    while time.time() < deadline:
        r2 = orch.compute({"topic": "drop a single", "id": "xyz"}, compute, namespace=ns)
        if r2["source"] == "cache":
            break
        time.sleep(0.02)
    assert r2["source"] == "cache"           # different id, same semantic request
    assert r2["result"]["value"] == "DROP A SINGLE"
    n_settled = calls["n"]
    r3 = orch.compute({"topic": "drop a single", "id": "zzz"}, compute, namespace=ns)
    assert r3["source"] == "cache" and calls["n"] == n_settled  # deduped for good


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
        results.append(orch.compute({"k": "same"}, compute,
                                    namespace=f"test_sf_{_RUN}"))

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
    cfg = PDIMConfig(namespace=f"test_q_{_RUN}", batch_size=8, base_dir=base)
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
    cfg = PDIMConfig(namespace=f"test_prev_{_RUN}", base_dir=base)
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
    n0, f0 = _counter("dgpu.gemm.numpy"), _counter("dgpu.gemm.engine_fallback")
    o1 = dg.gemm(A, W).numpy()
    o2 = dg.gemm(A, Bb).numpy()
    assert np.allclose(o1, A @ W, atol=TOL)
    assert np.allclose(o2, A @ Bb, atol=TOL)
    assert _counter("dgpu.gemm.numpy") == n0              # never bypassed to numpy
    assert _counter("dgpu.gemm.engine_fallback") == f0   # engine served it all


def test_softmax_any_axis_is_engine_served():
    dg = DigitalGPU()
    rng = np.random.default_rng(12)
    X = rng.standard_normal((2, 6, 4)).astype(np.float32)
    n0, f0 = _counter("dgpu.softmax.numpy"), _counter("dgpu.softmax.engine_fallback")
    out = dg.softmax(X, axis=1).numpy()                  # non-last axis
    ref = np.exp(X - X.max(1, keepdims=True))
    ref = ref / ref.sum(1, keepdims=True)
    assert np.allclose(out, ref, atol=TOL)
    assert np.allclose(out.sum(axis=1), 1.0, atol=TOL)
    assert _counter("dgpu.softmax.numpy") == n0
    assert _counter("dgpu.softmax.engine_fallback") == f0


def test_masked_multihead_attention_is_engine_served():
    dg = DigitalGPU()
    rng = np.random.default_rng(13)
    B, H, Tq, Tk, D = 2, 2, 5, 5, 8
    q = rng.standard_normal((B, H, Tq, D)).astype(np.float32)
    k = rng.standard_normal((B, H, Tk, D)).astype(np.float32)
    v = rng.standard_normal((B, H, Tk, D)).astype(np.float32)
    mask = np.where(rng.random((Tq, Tk)) < 0.3, -1e9, 0.0).astype(np.float32)
    n0, f0 = _counter("dgpu.attention.numpy"), _counter("dgpu.attention.engine_fallback")
    out = dg.attention(q, k, v, mask=mask).numpy()       # masked, 4D multi-head
    scores = (q @ np.swapaxes(k, -1, -2)) / np.sqrt(D) + mask
    p = np.exp(scores - scores.max(-1, keepdims=True))
    p = p / p.sum(-1, keepdims=True)
    ref = p @ v
    assert out.shape == (B, H, Tq, D)
    assert np.allclose(out, ref, atol=1e-2)
    assert _counter("dgpu.attention.numpy") == n0         # masked path stayed on engine
    assert _counter("dgpu.attention.engine_fallback") == f0


# ── pocket-dimension multiplication ──────────────────────────────────────────
def _fresh_pocket(path):
    """Pocket backed by a fresh in-process dedup (isolated from fleet cache)."""
    from ai_model.maxcore.pdim.orchestrator import _FallbackDedup
    from ai_model.maxcore.pdim.pocket_multiply import PocketDimension
    orch = PDIMOrchestrator(dedup=_FallbackDedup())
    return PocketDimension(path, orchestrator=orch), orch


def test_pocket_matmul_correct_and_deduped_inside_one_pocket():
    pocket, _ = _fresh_pocket("test/parent")
    rng = np.random.default_rng(21)
    A = rng.standard_normal((6, 9)).astype(np.float32)
    B = rng.standard_normal((9, 4)).astype(np.float32)

    r1 = pocket.matmul(A, B)
    assert r1["source"] == "compute"
    assert np.allclose(r1["result"], A @ B, atol=TOL)

    # Identical multiplication inside the SAME pocket: served, not recomputed.
    deadline = time.time() + 2.0
    while time.time() < deadline:            # leader persists via a bg thread
        r2 = pocket.matmul(A, B)
        if r2["source"] in ("cache", "coalesced"):
            break
        time.sleep(0.02)
    assert r2["source"] in ("cache", "coalesced")
    assert np.allclose(r2["result"], A @ B, atol=TOL)
    assert r2["compression"]["codec"] == "zlib+b64"
    assert r2["compression"]["stored_bytes"] > 0


def test_pocket_nesting_pockets_inside_one_pocket_are_isolated():
    outer, orch = _fresh_pocket("test/outer")
    inner_a = outer.pocket("a")
    inner_deep = outer.pocket("a").pocket("b").pocket("c")   # unbounded nesting
    assert inner_a.path == "test/outer/a"
    assert inner_deep.path == "test/outer/a/b/c"
    assert inner_deep.namespace == "pocket:test/outer/a/b/c"

    rng = np.random.default_rng(22)
    A = rng.standard_normal((3, 5)).astype(np.float32)
    B = rng.standard_normal((5, 2)).astype(np.float32)

    # Same operands in three different pockets: each pocket computes its own.
    assert outer.matmul(A, B)["source"] == "compute"
    assert inner_a.matmul(A, B)["source"] == "compute"
    assert inner_deep.matmul(A, B)["source"] == "compute"


def test_pocket_matmul_batched_and_shape_errors():
    pocket, _ = _fresh_pocket("test/batched")
    rng = np.random.default_rng(23)
    A = rng.standard_normal((2, 4, 6)).astype(np.float32)    # [B, M, K]
    W = rng.standard_normal((6, 3)).astype(np.float32)
    out = pocket.matmul(A, W)
    assert out["result"].shape == (2, 4, 3)
    assert np.allclose(out["result"], A @ W, atol=TOL)

    raised = False
    try:
        pocket.matmul(np.zeros((3, 4), np.float32), np.zeros((5, 2), np.float32))
    except ValueError:
        raised = True
    assert raised


# ── pocket accelerator wired into the Digital GPU ────────────────────────────
def test_pocket_accelerator_hit_serves_repeat_at_hash_cost():
    from ai_model.maxcore.pdim.pocket_accelerator import PocketAccelerator
    accel = PocketAccelerator(budget_bytes=64_000_000)
    rng = np.random.default_rng(31)
    A = rng.standard_normal((128, 128)).astype(np.float32)   # 4.2 MFLOP > gate
    B = rng.standard_normal((128, 128)).astype(np.float32)
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return A @ B

    r1, s1 = accel.accelerate("gemm", (A, B), 2.0 * A.size * B.shape[-1], compute)
    r2, s2 = accel.accelerate("gemm", (A, B), 2.0 * A.size * B.shape[-1], compute)
    assert (s1, s2) == ("compute", "pocket")
    assert calls["n"] == 1                       # repeat never recomputed
    assert np.allclose(r2, A @ B, atol=TOL)
    r2[0, 0] = 999.0                             # mutating a hit must not
    r3, s3 = accel.accelerate("gemm", (A, B), 2.0 * A.size * B.shape[-1], compute)
    assert s3 == "pocket" and np.allclose(r3, A @ B, atol=TOL)  # poison the pocket
    st = accel.stats()
    assert st["hits"] == 2 and st["misses"] == 1
    assert st["compute_seconds_saved"] > 0


def test_pocket_accelerator_miss_result_mutation_cannot_poison_cache():
    from ai_model.maxcore.pdim.pocket_accelerator import PocketAccelerator
    accel = PocketAccelerator()
    rng = np.random.default_rng(34)
    A = rng.standard_normal((128, 128)).astype(np.float32)
    B = rng.standard_normal((128, 128)).astype(np.float32)
    ref = A @ B
    flops = 2.0 * A.size * B.shape[-1]

    r1, s1 = accel.accelerate("gemm", (A, B), flops, lambda: A @ B)
    assert s1 == "compute"
    r1[:] = -1.0                                 # caller trashes its MISS result
    r2, s2 = accel.accelerate("gemm", (A, B), flops,
                              lambda: (_ for _ in ()).throw(AssertionError))
    assert s2 == "pocket"
    assert np.allclose(r2, ref, atol=TOL)        # pocket stayed pristine


def test_pocket_accelerator_adaptive_gating():
    from ai_model.maxcore.pdim.pocket_accelerator import PocketAccelerator
    accel = PocketAccelerator(warmup=6, reprobe_every=4)
    rng = np.random.default_rng(32)
    B = rng.standard_normal((128, 128)).astype(np.float32)

    # Tiny multiply: below the FLOP floor — never hashed.
    a_small = rng.standard_normal((4, 4)).astype(np.float32)
    b_small = rng.standard_normal((4, 4)).astype(np.float32)
    _, src = accel.accelerate("gemm", (a_small, b_small),
                              2.0 * a_small.size * 4, lambda: a_small @ b_small)
    assert src == "bypass"
    assert accel.stats()["bypass_small"] == 1

    # Never-repeating operands (training-style): pocket mutes itself after
    # warmup, then hashing stops entirely — the adaptive "never slower" gate.
    def offer_unique():
        A = rng.standard_normal((128, 128)).astype(np.float32)
        return accel.accelerate("gemm", (A, B),
                                2.0 * A.size * B.shape[-1], lambda: A @ B)[1]

    for _ in range(6):                            # warmup: all misses
        assert offer_unique() == "compute"
    assert accel.stats()["pockets_muted"] == 1
    sources = [offer_unique() for _ in range(3)]
    assert "bypass" in sources                    # hashing switched off
    assert accel.stats()["bypass_adaptive_muted"] >= 1
    # Re-probe turn arrives every 4th skip — the pocket can re-engage.
    sources += [offer_unique() for _ in range(6)]
    assert "compute" in sources                   # a re-probe actually ran


def test_digital_gpu_gemm_served_from_pocket_on_repeat():
    dg = DigitalGPU()
    rng = np.random.default_rng(33)
    A = rng.standard_normal((128, 96)).astype(np.float32)
    B = rng.standard_normal((96, 128)).astype(np.float32)
    h0 = _counter("pocket_accel.hit")
    o1 = dg.gemm(A, B).numpy()
    o2 = dg.gemm(A, B).numpy()                   # identical repeat
    assert np.allclose(o1, A @ B, atol=TOL)
    assert np.allclose(o2, o1, atol=0.0)         # bit-identical: pocket-served
    assert _counter("pocket_accel.hit") >= h0 + 1


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
