"""Tests for the digital-GPU execution stack: opcode contract, precision models,
execution graph + scheduler, telemetry, and the BLAS/DNN library.

Honesty is part of the contract, so these assert it directly: the fp8/sm102 op is
labelled a CPU numerics model (not hardware), fp8 rounding actually changes values,
and telemetry keeps measured wall-clock separate from derived FLOPs.
"""
import numpy as np

from ai_model.gpu.opcode_spec import get_spec, OPCODES
from ai_model.gpu.digital_gpu import (
    DigitalGPU, InvalidOpcodeError, OOMError, ShapeMismatchError,
)
from ai_model.gpu import precision
from ai_model.gpu.execution_graph import Node, ExecutionGraph, DigitalScheduler
from ai_model.gpu.telemetry import Telemetry
from ai_model.gpu.digital_library import DigitalBLAS, DigitalDNN


# ── opcode contract ───────────────────────────────────────────────────────────
def test_get_spec_bare_and_versioned():
    assert get_spec("gemm").key == "gemm:v1"
    assert get_spec("gemm:v1").name == "gemm"


def test_get_spec_unknown_raises_invalid_opcode():
    try:
        get_spec("no_such_op")
        assert False, "unknown opcode must raise"
    except InvalidOpcodeError:
        pass


def test_fp8_opcode_is_labelled_model_not_hardware():
    s = get_spec("flash_attention_fp8_sm102")
    assert s.is_hardware_execution is False        # the crux: not silicon
    assert s.target_arch == "sm_102"               # intent captured as metadata
    assert s.numeric_profile == "fp8_mixed"
    assert "model" in s.describe().lower()


# ── precision numerics models ─────────────────────────────────────────────────
def test_fp8_e4m3_saturates_and_rounds_zero():
    assert precision.to_fp8(np.array([1e9]), "e4m3")[0] == 448.0
    assert precision.to_fp8(np.array([0.0]), "e4m3")[0] == 0.0


def test_fp8_values_lie_on_grid_and_change_inputs():
    x = np.linspace(-10, 10, 101)
    q = precision.to_fp8(x, "e4m3")
    # idempotent: rounding an already-fp8 value is a no-op (true grid)
    assert np.allclose(q, precision.to_fp8(q, "e4m3"))
    # fp8 is lossy: at least some values must move
    assert np.any(q != x)


def test_fp16_bf16_roundtrip():
    x = np.array([1.0, 0.5, 3.14159, -2.5], dtype=np.float32)
    assert np.allclose(precision.to_fp16(x), x, atol=1e-2)
    assert np.allclose(precision.to_bf16(x), x, atol=3e-2)
    # bf16 keeps exponent range but ~7 mantissa bits -> coarser than fp16
    fine = np.float32(1.0 + 2 ** -10)
    assert precision.to_bf16(np.array([fine]))[0] == 1.0


def test_fp8_non_finite_handling():
    # e4m3 has no inf: overflow and inf-input both saturate to +-448; nan stays nan
    e4 = precision.to_fp8(np.array([np.inf, -np.inf, np.nan, 1e30]), "e4m3")
    assert e4[0] == 448.0 and e4[1] == -448.0
    assert np.isnan(e4[2])
    assert e4[3] == 448.0
    # e5m2 is IEEE-like: inf preserved, finite overflow rounds up to inf, nan->nan
    e5 = precision.to_fp8(np.array([np.inf, -np.inf, np.nan, 1e30]), "e5m2")
    assert np.isposinf(e5[0]) and np.isneginf(e5[1])
    assert np.isnan(e5[2])
    assert np.isposinf(e5[3])


def test_quantization_error_report():
    r = precision.quantization_error(np.linspace(-5, 5, 200), "fp8_mixed")
    assert r["max_abs_err"] > 0 and r["profile"] == "fp8_mixed"


def test_flash_attention_fp8_model_shape_and_lossy():
    rng = np.random.default_rng(0)
    Q = rng.standard_normal((2, 3, 5, 8))
    K = rng.standard_normal((2, 3, 5, 8))
    V = rng.standard_normal((2, 3, 5, 8))
    O = precision.flash_attention_fp8_model(Q, K, V)
    assert O.shape == Q.shape
    # a full-precision reference should differ from the fp8 model (it's lossy)
    D = Q.shape[-1]
    s = np.matmul(Q, np.swapaxes(K, -1, -2)) / np.sqrt(D)
    s = s - s.max(-1, keepdims=True)
    w = np.exp(s); w = w / w.sum(-1, keepdims=True)
    ref = np.matmul(w, V)
    assert np.abs(O - ref).max() > 0


# ── execution graph + scheduler ───────────────────────────────────────────────
def test_graph_topo_order_respects_dependencies():
    g = ExecutionGraph()
    # add before gemm in insertion order, but gemm feeds add -> must reorder
    g.add_node(Node("n2", "add", {"A": "t_c", "B": "t_bias"}, {"C": "t_out"}))
    g.add_node(Node("n1", "gemm", {"A": "t_a", "B": "t_b"}, {"C": "t_c"}))
    order = [n.id for n in g.topological_order()]
    assert order.index("n1") < order.index("n2")


def test_graph_cycle_detected():
    g = ExecutionGraph()
    g.add_node(Node("a", "add", {"A": "x", "B": "y"}, {"C": "y"}))  # y depends on y
    g.add_node(Node("b", "add", {"A": "y", "B": "z"}, {"C": "x"}))  # x<-y, y<-x cycle
    try:
        g.topological_order()
        assert False, "cycle must raise"
    except Exception:
        pass


def test_scheduler_runs_gemm_then_add_matches_numpy():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((8, 6))
    B = rng.standard_normal((6, 4))
    bias = rng.standard_normal((8, 4))
    g = ExecutionGraph()
    g.add_node(Node("mm", "gemm", {"A": "A", "B": "B"}, {"C": "C"}))
    g.add_node(Node("bi", "add", {"A": "C", "B": "bias"}, {"C": "out"}))
    tel = Telemetry()
    sched = DigitalScheduler(telemetry=tel)
    out = sched.run(g, {"A": A, "B": B, "bias": bias})
    assert np.allclose(out["out"], A @ B + bias, atol=1e-6)
    assert len(tel.ops) == 2


def test_scheduler_runs_fp8_flash_attention_opcode():
    rng = np.random.default_rng(2)
    Q = rng.standard_normal((1, 2, 4, 8))
    K = rng.standard_normal((1, 2, 4, 8))
    V = rng.standard_normal((1, 2, 4, 8))
    g = ExecutionGraph()
    g.add_node(Node("fa", "flash_attention_fp8_sm102",
                    {"Q": "Q", "K": "K", "V": "V"}, {"O": "O"}))
    out = DigitalScheduler().run(g, {"Q": Q, "K": K, "V": V})
    assert out["O"].shape == Q.shape


def test_scheduler_oom_budget():
    A = np.ones((64, 64)); B = np.ones((64, 64))
    g = ExecutionGraph()
    g.add_node(Node("mm", "gemm", {"A": "A", "B": "B"}, {"C": "C"}))
    try:
        DigitalScheduler(max_bytes=1024).run(g, {"A": A, "B": B})
        assert False, "should exceed 1KB budget"
    except OOMError:
        pass


def test_graph_rejects_duplicate_node_id():
    g = ExecutionGraph()
    g.add_node(Node("dup", "add", {"A": "a", "B": "b"}, {"C": "c"}))
    g.add_node(Node("dup", "add", {"A": "c", "B": "d"}, {"C": "e"}))
    try:
        g.topological_order()
        assert False, "duplicate node id must raise"
    except Exception:
        pass


def test_graph_rejects_duplicate_producer():
    g = ExecutionGraph()
    g.add_node(Node("n1", "add", {"A": "a", "B": "b"}, {"C": "shared"}))
    g.add_node(Node("n2", "add", {"A": "c", "B": "d"}, {"C": "shared"}))
    try:
        g.topological_order()
        assert False, "two producers of one tensor must raise"
    except Exception:
        pass


def test_scheduler_contract_missing_input_raises():
    g = ExecutionGraph()
    # gemm needs A and B; only A supplied
    g.add_node(Node("mm", "gemm", {"A": "A"}, {"C": "C"}))
    try:
        DigitalScheduler().run(g, {"A": np.ones((2, 2)), "B": np.ones((2, 2))})
        assert False, "missing required input must raise"
    except ShapeMismatchError:
        pass


def test_scheduler_contract_wrong_output_raises():
    g = ExecutionGraph()
    g.add_node(Node("mm", "gemm", {"A": "A", "B": "B"}, {"WRONG": "C"}))
    try:
        DigitalScheduler().run(g, {"A": np.ones((2, 2)), "B": np.ones((2, 2))})
        assert False, "wrong output name must raise"
    except ShapeMismatchError:
        pass


def test_scheduler_oom_live_set_not_monotonic():
    # Overwriting an existing tensor id must REPLACE its bytes, not add — so a
    # graph whose live set fits the budget must not false-positive.
    t = np.ones((8, 8), dtype=np.float64)          # 512 bytes each
    g = ExecutionGraph()
    # writes t1 (overwrites the initial t1); t1 is not read by this node
    g.add_node(Node("bi", "add", {"A": "t2", "B": "t3"}, {"C": "t1"}))
    # live set stays 3*512 = 1536; a naive add-only counter would hit 2048
    out = DigitalScheduler(max_bytes=1600).run(g, {"t1": t, "t2": t, "t3": t})
    assert np.allclose(out["t1"], t + t)


def test_scheduler_unknown_opcode_raises():
    g = ExecutionGraph()
    g.add_node(Node("x", "bogus_op", {"A": "A", "B": "B"}, {"C": "C"}))
    try:
        DigitalScheduler().run(g, {"A": np.ones((2, 2)), "B": np.ones((2, 2))})
        assert False
    except InvalidOpcodeError:
        pass


# ── telemetry honesty ─────────────────────────────────────────────────────────
def test_telemetry_separates_measured_from_derived():
    t = Telemetry()
    r = t.record("gemm:v1", "fp32_strict", wall_ms=1.5, flops=2048, bytes_moved=512)
    assert r.is_wall_measured is True
    assert r.is_flops_derived is True         # flops are analytic, not measured
    s = t.summary()
    assert s["ops"] == 1 and "analytic" in s["note"]


# ── library layer ─────────────────────────────────────────────────────────────
def test_blas_gemm_matches_numpy():
    rng = np.random.default_rng(3)
    A = rng.standard_normal((10, 7)); B = rng.standard_normal((7, 5))
    assert np.allclose(DigitalBLAS().gemm(A, B), A @ B, atol=1e-6)


def test_dnn_conv2d_matches_reference():
    rng = np.random.default_rng(4)
    x = rng.standard_normal((2, 3, 7, 7))
    w = rng.standard_normal((4, 3, 3, 3))
    bias = rng.standard_normal((4,))
    y = DigitalDNN().conv2d(x, w, bias=bias, stride=1, padding=1)
    # naive reference conv
    N, C, H, W = x.shape
    F, _, KH, KW = w.shape
    OH, OW = H, W  # stride 1, pad 1, k 3 -> same size
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    ref = np.zeros((N, F, OH, OW))
    for n in range(N):
        for f in range(F):
            for oh in range(OH):
                for ow in range(OW):
                    patch = xp[n, :, oh:oh + KH, ow:ow + KW]
                    ref[n, f, oh, ow] = np.sum(patch * w[f]) + bias[f]
    assert np.allclose(y, ref, atol=1e-6)


def test_dnn_conv2d_channel_mismatch_raises():
    try:
        DigitalDNN().conv2d(np.ones((1, 3, 5, 5)), np.ones((2, 4, 3, 3)))
        assert False
    except ShapeMismatchError:
        pass
