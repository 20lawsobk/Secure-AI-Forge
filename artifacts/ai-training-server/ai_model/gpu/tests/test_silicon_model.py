"""Tests for the digital-silicon performance model and its wiring.

The model is a *what-if estimator*. These tests pin two things:
  1. it produces coherent, honestly-labeled estimates; and
  2. attaching it to DigitalGPU / HyperGPU changes NOTHING about the real
     numerical results (it is telemetry, never an execution path).
"""
from __future__ import annotations

import numpy as np

from ai_model.gpu import DigitalGPU, MaxCoreSilicon, make_default_silicon
from ai_model.gpu.hyper_core import HyperGPU
from ai_model.gpu.silicon_model import MODE, MaxCoreOp


def test_report_is_labeled_estimate_not_measurement():
    sil = make_default_silicon()
    rep = sil.report()
    assert rep["mode"] == MODE == "performance_model"
    assert rep["is_measurement"] is False
    assert "disclaimer" in rep and "NOT measured" in rep["disclaimer"]
    assert rep["global_memory"]["is_measurement"] is False


def test_model_op_accumulates_estimates():
    sil = MaxCoreSilicon(num_tiles=4, num_attention_tiles=2)
    c1 = sil.model_op("gemm", estimated_flops=8e12)  # ~8 cycles at 1e12/cyc
    c2 = sil.model_op("attention", estimated_flops=2e12, kv_size=4e9)
    assert c1 >= 1 and c2 >= 1
    rep = sil.report()
    assert rep["total_ops_modeled"] == 2
    assert rep["total_estimated_flops"] == 8e12 + 2e12
    assert rep["estimated_critical_path_cycles"] >= 1
    assert rep["estimated_seconds"] >= 0.0


def test_gemm_routes_to_gemm_tile_attention_to_attention_tile():
    sil = MaxCoreSilicon(num_tiles=4, num_attention_tiles=2)
    g = MaxCoreOp(kind="gemm", estimated_flops=1e12)
    a = MaxCoreOp(kind="attention", estimated_flops=1e12)
    gt = sil.scheduler.pick_tile(g)
    at = sil.scheduler.pick_tile(a)
    assert type(gt).__name__ == "GemmTile"
    assert type(at).__name__ == "AttentionTile"


def test_simulate_drains_and_reports_cycles():
    sil = MaxCoreSilicon(num_tiles=2, num_attention_tiles=0)
    ops = [MaxCoreOp(kind="gemm", estimated_flops=3e12) for _ in range(4)]
    rep = sil.simulate(ops)
    # 4 ops of ~3 cycles across 2 tiles -> ~6 cycles critical path.
    assert rep["estimated_critical_path_cycles"] >= 3
    assert all(t["pending"] == 0 for t in rep["tiles"])
    assert rep["is_measurement"] is False


def test_attaching_silicon_does_not_change_digitalgpu_results():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((8, 6)).astype(np.float32)
    B = rng.standard_normal((6, 5)).astype(np.float32)
    Q = rng.standard_normal((2, 4, 8)).astype(np.float32)
    K = rng.standard_normal((2, 4, 8)).astype(np.float32)
    V = rng.standard_normal((2, 4, 8)).astype(np.float32)

    plain = DigitalGPU()
    modeled = DigitalGPU(silicon=make_default_silicon())

    assert np.array_equal(plain.gemm(A, B), modeled.gemm(A, B))
    assert np.array_equal(plain.attention(Q, K, V), modeled.attention(Q, K, V))
    assert np.array_equal(plain.softmax(A), modeled.softmax(A))

    # ...and the model actually recorded the executed ops as estimates.
    rep = modeled.silicon_report()
    assert rep is not None and rep["total_ops_modeled"] == 3
    assert plain.silicon_report() is None


def test_attaching_silicon_does_not_change_hypergpu_results():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((10, 7)).astype(np.float32)
    B = rng.standard_normal((7, 4)).astype(np.float32)

    plain = HyperGPU()
    modeled = HyperGPU(silicon=make_default_silicon())
    assert np.allclose(plain.gemm(A, B), modeled.gemm(A, B), atol=1e-4)

    status = modeled.status()
    assert status["silicon"] is not None
    assert status["silicon"]["total_ops_modeled"] >= 1
    assert plain.status()["silicon"] is None


def test_mixed_gemm_with_silicon_does_not_crash():
    # Regression: HyperGPU.mixed_gemm passes precision= into _model; the hook
    # must accept it rather than raising TypeError with silicon attached.
    rng = np.random.default_rng(2)
    A = rng.standard_normal((6, 5)).astype(np.float32)
    B = rng.standard_normal((5, 3)).astype(np.float32)
    plain = HyperGPU()
    modeled = HyperGPU(silicon=make_default_silicon())
    assert np.allclose(plain.mixed_gemm(A, B), modeled.mixed_gemm(A, B), atol=1e-2)
    assert modeled.silicon_report()["total_ops_modeled"] >= 1


def test_concurrent_model_and_report_are_stable():
    import threading
    sil = MaxCoreSilicon(num_tiles=8, num_attention_tiles=2)
    errors = []

    def worker():
        try:
            for _ in range(200):
                sil.model_op("gemm", estimated_flops=1e12)
                sil.report()
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert sil.report()["total_ops_modeled"] == 4 * 200
