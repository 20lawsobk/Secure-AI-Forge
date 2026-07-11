"""Comprehensive tests for the digital GPU system.

Covers the surface not already exercised by the focused unit tests:
  - VRAM (alloc/get/meta/free, error paths)
  - SIMDCore (all ops, error paths, causal attention, dtype guards)
  - Scheduler + Program (instruction execution, profiling)
  - DigitalGPU high-level API (functional + handle-based, silicon wiring)
  - HyperVRAM (capacity OOM, flush, peak tracking)
  - MemoryPool (alloc/free/get/flush, peak, status)
  - TensorCoreUnit (matmul, mixed-precision matmul, status)
  - HyperSIMDCore (every op: batched GEMM, conv3d, layer_norm, batch_norm,
    gelu, silu, grouped_gemm, fused_attention_norm, status)
  - HyperGPU (every public method, status, vram flush)
  - GPUCluster / GPUClusterNode (lifecycle, distributed ops, add/remove)
"""
from __future__ import annotations

import threading

import numpy as np
import pytest

from ai_model.gpu.digital_gpu import (
    DigitalGPU, VRAM, SIMDCore, Scheduler, Program, Instruction, OpCode,
    GPUError, ShapeError, TypeErrorGPU,
)
from ai_model.gpu.hyper_core import (
    HyperGPU, HyperVRAM, HyperSIMDCore, MemoryPool, TensorCoreUnit,
    PrecisionMode, GPUCluster, GPUClusterNode,
)

rng = np.random.default_rng(42)


# ─── helpers ──────────────────────────────────────────────────────────────────

def f32(*shape):
    return rng.standard_normal(shape).astype(np.float32)


def _naive_softmax(X, axis=-1):
    m = X.max(axis=axis, keepdims=True)
    e = np.exp(X - m)
    return e / e.sum(axis=axis, keepdims=True)


def _naive_attn(Q, K, V, causal=False):
    """Reference scaled dot-product attention (any leading batch dims)."""
    D = Q.shape[-1]
    Tq = Q.shape[-2]
    Tk = K.shape[-2]
    scale = 1.0 / np.sqrt(D)
    scores = np.matmul(Q, np.swapaxes(K, -1, -2)) * scale
    if causal:
        qi = np.arange(Tq)[:, None]
        kj = np.arange(Tk)[None, :]
        scores = np.where(kj > qi, -1e9, scores)
    return np.matmul(_naive_softmax(scores, axis=-1), V)


# ═══════════════════════════════════════════════════════════════════════════════
# VRAM
# ═══════════════════════════════════════════════════════════════════════════════

class TestVRAM:
    def test_alloc_returns_integer_handle(self):
        v = VRAM()
        h = v.alloc(np.zeros((3, 4), dtype=np.float32))
        assert isinstance(h, int)

    def test_alloc_non_ndarray_raises_type_error(self):
        v = VRAM()
        with pytest.raises(TypeErrorGPU):
            v.alloc([[1, 2], [3, 4]])  # plain list is not ndarray

    def test_get_returns_same_array(self):
        v = VRAM()
        arr = f32(5, 5)
        h = v.alloc(arr)
        np.testing.assert_array_equal(v.get(h), arr)

    def test_get_invalid_handle_raises(self):
        v = VRAM()
        with pytest.raises(GPUError):
            v.get(999)

    def test_meta_returns_shape_dtype_size(self):
        v = VRAM()
        arr = np.zeros((3, 7), dtype=np.float32)
        h = v.alloc(arr)
        m = v.meta(h)
        assert m["shape"] == (3, 7)
        assert m["dtype"] == np.float32
        assert m["size"] == 21

    def test_meta_invalid_handle_raises(self):
        v = VRAM()
        with pytest.raises(GPUError):
            v.meta(999)

    def test_free_removes_handle(self):
        v = VRAM()
        h = v.alloc(f32(2, 2))
        v.free(h)
        with pytest.raises(GPUError):
            v.get(h)

    def test_handles_are_monotonically_increasing(self):
        v = VRAM()
        h0 = v.alloc(f32(2, 2))
        h1 = v.alloc(f32(2, 2))
        h2 = v.alloc(f32(2, 2))
        assert h0 < h1 < h2

    def test_free_nonexistent_handle_is_silent(self):
        v = VRAM()
        v.free(9999)  # must not raise


# ═══════════════════════════════════════════════════════════════════════════════
# SIMDCore
# ═══════════════════════════════════════════════════════════════════════════════

class TestSIMDCore:
    def setup_method(self):
        self.core = SIMDCore(tile_m=32, tile_n=32, tile_k=32)

    # ── gemm ──────────────────────────────────────────────────────────────────
    def test_gemm_tiled_matches_numpy(self):
        A = f32(20, 15)
        B = f32(15, 10)
        np.testing.assert_allclose(self.core.gemm_tiled(A, B), A @ B, atol=1e-5)

    def test_gemm_non_square_matches_numpy(self):
        A = f32(37, 13)
        B = f32(13, 41)
        np.testing.assert_allclose(self.core.gemm_tiled(A, B), A @ B, atol=1e-5)

    def test_gemm_1d_input_raises(self):
        with pytest.raises(ShapeError):
            self.core.gemm_tiled(f32(4), f32(4))

    def test_gemm_inner_dim_mismatch_raises(self):
        with pytest.raises(ShapeError):
            self.core.gemm_tiled(f32(4, 5), f32(6, 3))

    # ── add ───────────────────────────────────────────────────────────────────
    def test_add_correct(self):
        A = f32(8, 8)
        B = f32(8, 8)
        np.testing.assert_allclose(self.core.add(A, B), A + B)

    def test_add_shape_mismatch_raises(self):
        with pytest.raises(ShapeError):
            self.core.add(f32(4, 4), f32(4, 5))

    def test_add_dtype_mismatch_raises(self):
        A = np.ones((4, 4), dtype=np.float32)
        B = np.ones((4, 4), dtype=np.float64)
        with pytest.raises(TypeErrorGPU):
            self.core.add(A, B)

    # ── softmax ───────────────────────────────────────────────────────────────
    def test_softmax_rows_sum_to_one(self):
        X = f32(6, 10)
        Y = self.core.softmax(X, axis=-1)
        np.testing.assert_allclose(Y.sum(axis=-1), np.ones(6), atol=1e-6)

    def test_softmax_numerically_stable_large_values(self):
        X = np.array([[1e9, 1e9 + 1, 1e9 + 2]], dtype=np.float32)
        Y = self.core.softmax(X, axis=-1)
        assert np.all(np.isfinite(Y))
        np.testing.assert_allclose(Y.sum(axis=-1), [1.0], atol=1e-5)

    def test_softmax_axis_0(self):
        X = f32(5, 5)
        Y = self.core.softmax(X, axis=0)
        np.testing.assert_allclose(Y.sum(axis=0), np.ones(5), atol=1e-6)

    # ── attention ─────────────────────────────────────────────────────────────
    def test_attention_self_matches_reference(self):
        Q = f32(2, 6, 8)
        got = self.core.attention(Q, Q, Q)
        ref = _naive_attn(Q, Q, Q)
        np.testing.assert_allclose(got, ref, atol=1e-5)

    def test_attention_causal_masks_future(self):
        # With causal masking, output must match the reference causal attention.
        # Also verify causal differs from non-causal (future tokens suppressed).
        T, D = 6, 8
        Q = f32(1, T, D)
        K = f32(1, T, D)
        V = f32(1, T, D)
        out_causal = self.core.attention(Q, K, V, causal=True)
        out_noncausal = self.core.attention(Q, K, V, causal=False)
        ref_causal = _naive_attn(Q, K, V, causal=True)
        np.testing.assert_allclose(out_causal, ref_causal, atol=1e-5)
        assert not np.allclose(out_causal, out_noncausal), (
            "causal and non-causal outputs must differ"
        )

    def test_attention_batched_higher_rank(self):
        # [B, H, T, D] — two leading batch dims
        Q = f32(2, 3, 5, 8)
        out = self.core.attention(Q, Q, Q)
        assert out.shape == Q.shape

    def test_attention_shape_mismatch_raises(self):
        with pytest.raises(ShapeError):
            self.core.attention(f32(2, 4, 8), f32(2, 4, 8), f32(2, 5, 8))

    def test_attention_1d_raises(self):
        with pytest.raises(ShapeError):
            self.core.attention(f32(4), f32(4), f32(4))

    # ── gemm_bias_relu ────────────────────────────────────────────────────────
    def test_gemm_bias_relu_nonnegative(self):
        A = f32(8, 6)
        B = f32(6, 4)
        bias = f32(4)  # 1D bias
        out = self.core.gemm_bias_relu(A, B, bias)
        assert np.all(out >= 0.0)

    def test_gemm_bias_relu_matches_manual(self):
        A = f32(5, 4)
        B = f32(4, 3)
        bias = np.array([0.5, -100.0, 2.0], dtype=np.float32)
        out = self.core.gemm_bias_relu(A, B, bias)
        ref = np.maximum(A @ B + bias, 0.0)
        np.testing.assert_allclose(out, ref, atol=1e-5)


# ═══════════════════════════════════════════════════════════════════════════════
# Scheduler + Program (from digital_gpu.py)
# ═══════════════════════════════════════════════════════════════════════════════

class TestSchedulerProgram:
    def _make(self):
        vram = VRAM()
        core = SIMDCore()
        sched = Scheduler(vram, core)
        return vram, core, sched

    def test_gemm_program_correct(self):
        vram, _, sched = self._make()
        A = f32(4, 6)
        B = f32(6, 3)
        hA = vram.alloc(A)
        hB = vram.alloc(B)
        hC = vram.alloc(np.zeros((4, 3), np.float32))
        prog = Program()
        prog.add(Instruction(OpCode.GEMM, {"a": hA, "b": hB, "out": hC}))
        sched.run(prog)
        np.testing.assert_allclose(vram.get(hC), A @ B, atol=1e-5)

    def test_add_program_correct(self):
        vram, _, sched = self._make()
        A = f32(5, 5)
        B = f32(5, 5)
        hA = vram.alloc(A)
        hB = vram.alloc(B)
        hC = vram.alloc(np.zeros_like(A))
        prog = Program()
        prog.add(Instruction(OpCode.ADD, {"a": hA, "b": hB, "out": hC}))
        sched.run(prog)
        np.testing.assert_allclose(vram.get(hC), A + B)

    def test_softmax_program_correct(self):
        vram, _, sched = self._make()
        X = f32(3, 8)
        hX = vram.alloc(X)
        hY = vram.alloc(np.zeros_like(X))
        prog = Program()
        prog.add(Instruction(OpCode.SOFTMAX, {"x": hX, "out": hY, "axis": -1}))
        sched.run(prog)
        Y = vram.get(hY)
        np.testing.assert_allclose(Y.sum(axis=-1), np.ones(3), atol=1e-6)

    def test_multi_instruction_program_and_profile(self):
        vram, _, sched = self._make()
        A = f32(4, 6)
        B = f32(6, 3)
        X = f32(4, 3)
        hA = vram.alloc(A)
        hB = vram.alloc(B)
        hC = vram.alloc(np.zeros((4, 3), np.float32))
        hX = vram.alloc(X)
        hOut = vram.alloc(np.zeros((4, 3), np.float32))

        prog = Program()
        prog.add(Instruction(OpCode.GEMM, {"a": hA, "b": hB, "out": hC}))
        prog.add(Instruction(OpCode.ADD, {"a": hC, "b": hX, "out": hOut}))
        sched.run(prog)

        # Correct result
        np.testing.assert_allclose(vram.get(hOut), A @ B + X, atol=1e-5)
        # Profile recorded two entries
        assert len(sched.last_profile) == 2
        for entry in sched.last_profile:
            assert "opcode" in entry
            assert "duration_ms" in entry
            assert entry["duration_ms"] >= 0.0

    def test_profile_opcode_names(self):
        vram, _, sched = self._make()
        A = f32(4, 6)
        B = f32(6, 3)
        hA = vram.alloc(A)
        hB = vram.alloc(B)
        hC = vram.alloc(np.zeros((4, 3), np.float32))
        prog = Program()
        prog.add(Instruction(OpCode.GEMM, {"a": hA, "b": hB, "out": hC}))
        sched.run(prog)
        assert sched.last_profile[0]["opcode"] == "GEMM"

    def test_bad_handle_raises_gpu_error(self):
        vram, _, sched = self._make()
        hA = vram.alloc(f32(4, 6))
        hC = vram.alloc(np.zeros((4, 3), np.float32))
        prog = Program()
        # hB = 999 does not exist
        prog.add(Instruction(OpCode.GEMM, {"a": hA, "b": 999, "out": hC}))
        with pytest.raises(GPUError):
            sched.run(prog)


# ═══════════════════════════════════════════════════════════════════════════════
# DigitalGPU high-level functional API
# ═══════════════════════════════════════════════════════════════════════════════

class TestDigitalGPUFunctional:
    def setup_method(self):
        self.gpu = DigitalGPU()

    def test_gemm_matches_numpy(self):
        A = f32(10, 8)
        B = f32(8, 6)
        np.testing.assert_allclose(self.gpu.gemm(A, B), A @ B, atol=1e-5)

    def test_add_matches_numpy(self):
        A = f32(6, 6)
        B = f32(6, 6)
        np.testing.assert_allclose(self.gpu.add(A, B), A + B)

    def test_softmax_sums_to_one(self):
        X = f32(4, 10)
        Y = self.gpu.softmax(X)
        np.testing.assert_allclose(Y.sum(axis=-1), np.ones(4), atol=1e-6)

    def test_attention_shape_preserved(self):
        Q = f32(2, 5, 8)
        out = self.gpu.attention(Q, Q, Q)
        assert out.shape == Q.shape

    def test_attention_causal_matches_reference(self):
        Q = f32(1, 6, 8)
        got = self.gpu.attention(Q, Q, Q, causal=True)
        ref = _naive_attn(Q, Q, Q, causal=True)
        np.testing.assert_allclose(got, ref, atol=1e-5)

    def test_gemm_bias_relu_nonneg(self):
        A = f32(5, 4)
        B = f32(4, 3)
        bias = np.array([-1e3, 0.0, 1e3], dtype=np.float32)
        out = self.gpu.gemm_bias_relu(A, B, bias)
        assert np.all(out >= 0.0)

    def test_last_profile_populated_after_gemm(self):
        A = f32(4, 4)
        B = f32(4, 4)
        self.gpu.gemm(A, B)
        profile = self.gpu.last_profile()
        assert len(profile) >= 1
        assert all("opcode" in p and "duration_ms" in p for p in profile)

    def test_silicon_report_none_without_model(self):
        assert self.gpu.silicon_report() is None


class TestDigitalGPUHandleAPI:
    """Handle-based (h_*) variants of DigitalGPU."""

    def setup_method(self):
        self.gpu = DigitalGPU()

    def _alloc(self, arr):
        return self.gpu.vram.alloc(arr)

    def test_h_gemm_correct(self):
        A = f32(5, 4)
        B = f32(4, 3)
        hA = self._alloc(A)
        hB = self._alloc(B)
        hC = self.gpu.h_gemm(hA, hB)
        np.testing.assert_allclose(self.gpu.vram.get(hC), A @ B, atol=1e-5)

    def test_h_add_correct(self):
        A = f32(4, 4)
        B = f32(4, 4)
        hA = self._alloc(A)
        hB = self._alloc(B)
        hC = self.gpu.h_add(hA, hB)
        np.testing.assert_allclose(self.gpu.vram.get(hC), A + B)

    def test_h_softmax_correct(self):
        X = f32(3, 8)
        hX = self._alloc(X)
        hY = self.gpu.h_softmax(hX)
        Y = self.gpu.vram.get(hY)
        np.testing.assert_allclose(Y.sum(axis=-1), np.ones(3), atol=1e-6)

    def test_h_attention_shape(self):
        Q = f32(2, 5, 8)
        hQ = self._alloc(Q)
        hK = self._alloc(Q.copy())
        hV = self._alloc(Q.copy())
        hO = self.gpu.h_attention(hQ, hK, hV)
        assert self.gpu.vram.get(hO).shape == Q.shape

    def test_h_attention_causal(self):
        Q = f32(1, 4, 8)
        hQ = self._alloc(Q)
        hK = self._alloc(Q.copy())
        hV = self._alloc(Q.copy())
        hO_c = self.gpu.h_attention(hQ, hK, hV, causal=True)
        hO_f = self.gpu.h_attention(hQ, hK, hV, causal=False)
        # Causal and non-causal outputs differ
        assert not np.allclose(self.gpu.vram.get(hO_c), self.gpu.vram.get(hO_f))

    def test_h_gemm_bias_relu_nonneg(self):
        A = f32(5, 4)
        B = f32(4, 3)
        bias = np.full(3, -1e6, dtype=np.float32)
        hA = self._alloc(A)
        hB = self._alloc(B)
        hBias = self._alloc(bias)
        hO = self.gpu.h_gemm_bias_relu(hA, hB, hBias)
        assert np.all(self.gpu.vram.get(hO) >= 0.0)

    def test_h_gemm_bad_handle_raises(self):
        hA = self._alloc(f32(4, 6))
        with pytest.raises(GPUError):
            self.gpu.h_gemm(hA, 9999)


# ═══════════════════════════════════════════════════════════════════════════════
# HyperVRAM
# ═══════════════════════════════════════════════════════════════════════════════

class TestHyperVRAM:
    def test_oom_raises_when_capacity_exceeded(self):
        v = HyperVRAM(capacity_bytes=1024)
        with pytest.raises(GPUError, match="OOM"):
            v.alloc(np.zeros(300, dtype=np.float32))  # 1200 bytes > 1024

    def test_unlimited_capacity_accepts_large_array(self):
        v = HyperVRAM(capacity_bytes=0)  # 0 = unlimited
        h = v.alloc(f32(100, 100))
        assert h is not None

    def test_used_bytes_correct_after_alloc(self):
        v = HyperVRAM()
        arr = np.zeros((10, 10), dtype=np.float32)  # 400 bytes
        v.alloc(arr)
        assert v.used_bytes == 400

    def test_peak_bytes_tracks_high_water_mark(self):
        v = HyperVRAM()
        h1 = v.alloc(np.zeros((10, 10), dtype=np.float32))  # 400 B
        peak_after_one = v.peak_bytes
        h2 = v.alloc(np.zeros((20, 10), dtype=np.float32))  # 800 B more
        peak_after_two = v.peak_bytes
        v.free(h2)
        # Peak should not drop on free
        assert v.peak_bytes == peak_after_two
        assert peak_after_two > peak_after_one

    def test_flush_clears_all_handles(self):
        v = HyperVRAM()
        h = v.alloc(f32(5, 5))
        v.flush()
        assert v.used_bytes == 0
        with pytest.raises(GPUError):
            v.get(h)

    def test_status_returns_expected_keys(self):
        v = HyperVRAM(capacity_bytes=4096)
        v.alloc(f32(4, 4))
        s = v.status()
        assert "capacity_mb" in s
        assert "used_mb" in s
        assert "peak_mb" in s
        assert "handles" in s
        assert s["handles"] == 1


# ═══════════════════════════════════════════════════════════════════════════════
# MemoryPool
# ═══════════════════════════════════════════════════════════════════════════════

class TestMemoryPool:
    def test_alloc_returns_id_and_array(self):
        pool = MemoryPool()
        mid, arr = pool.alloc((4, 4), dtype=np.float32)
        assert isinstance(mid, int)
        assert arr.shape == (4, 4)

    def test_get_returns_same_object(self):
        pool = MemoryPool()
        mid, arr = pool.alloc((3, 3))
        assert pool.get(mid) is arr

    def test_get_invalid_handle_raises(self):
        pool = MemoryPool()
        with pytest.raises(GPUError):
            pool.get(9999)

    def test_free_reduces_current_bytes(self):
        pool = MemoryPool()
        mid, arr = pool.alloc((10,), dtype=np.float32)  # 40 bytes
        before = pool._current_bytes
        pool.free(mid)
        assert pool._current_bytes == before - arr.nbytes

    def test_peak_bytes_does_not_drop_on_free(self):
        pool = MemoryPool()
        mid, _ = pool.alloc((20,), dtype=np.float32)
        peak = pool._peak_bytes
        pool.free(mid)
        assert pool._peak_bytes == peak

    def test_flush_returns_freed_bytes_and_clears(self):
        pool = MemoryPool()
        pool.alloc((10,), dtype=np.float32)
        pool.alloc((10,), dtype=np.float32)
        freed = pool.flush()
        assert freed > 0
        assert pool._current_bytes == 0
        assert len(pool._allocated) == 0

    def test_status_keys(self):
        pool = MemoryPool()
        pool.alloc((4, 4))
        s = pool.status()
        for key in ("pool_size_mb", "current_mb", "peak_mb",
                    "active_allocations", "total_allocs", "reuse_count"):
            assert key in s

    def test_concurrent_alloc_free_no_crash(self):
        pool = MemoryPool()
        errors = []

        def worker():
            try:
                for _ in range(50):
                    mid, _ = pool.alloc((4, 4))
                    pool.free(mid)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors


# ═══════════════════════════════════════════════════════════════════════════════
# TensorCoreUnit
# ═══════════════════════════════════════════════════════════════════════════════

class TestTensorCoreUnit:
    def test_matmul_correct(self):
        tc = TensorCoreUnit()
        A = f32(8, 6)
        B = f32(6, 4)
        C = tc.matmul(A, B)
        np.testing.assert_allclose(C, A @ B, atol=1e-5)

    def test_matmul_increments_ops_and_flops(self):
        tc = TensorCoreUnit()
        A = f32(4, 4)
        B = f32(4, 4)
        tc.matmul(A, B)
        assert tc.ops_executed == 1
        assert tc.total_flops > 0.0

    def test_mixed_precision_matmul_close_to_fp32(self):
        tc = TensorCoreUnit()
        A = f32(8, 6)
        B = f32(6, 4)
        C = tc.mixed_precision_matmul(A, B)
        # Mixed-precision (FP32 SGEMM or emulated FP16) must be within fp16 tolerance
        np.testing.assert_allclose(C, A @ B, atol=5e-2)

    def test_mixed_precision_increments_flops_by_multiplier(self):
        tc = TensorCoreUnit(throughput_multiplier=8.0)
        A = f32(4, 4)
        B = f32(4, 4)
        before = tc.total_flops
        tc.mixed_precision_matmul(A, B)
        added = tc.total_flops - before
        # flops = 2*M*N*K * multiplier = 2*4*4*4*8 = 1024
        assert added == pytest.approx(2.0 * 4 * 4 * 4 * 8.0)

    def test_status_keys_and_types(self):
        tc = TensorCoreUnit()
        s = tc.status()
        assert isinstance(s["tile_shape"], str)
        assert isinstance(s["throughput_multiplier"], float)
        assert isinstance(s["ops_executed"], int)
        assert isinstance(s["total_tflops"], float)


# ═══════════════════════════════════════════════════════════════════════════════
# HyperSIMDCore
# ═══════════════════════════════════════════════════════════════════════════════

class TestHyperSIMDCore:
    def setup_method(self):
        self.core = HyperSIMDCore(tensor_cores=2)

    # ── tensor_core_gemm ──────────────────────────────────────────────────────
    def test_tensor_core_gemm_correct(self):
        A = f32(8, 6)
        B = f32(6, 4)
        np.testing.assert_allclose(self.core.tensor_core_gemm(A, B), A @ B, atol=1e-5)

    def test_tensor_core_gemm_1d_raises(self):
        with pytest.raises(ShapeError):
            self.core.tensor_core_gemm(f32(4), f32(4))

    def test_tensor_core_gemm_inner_dim_mismatch_raises(self):
        with pytest.raises(ShapeError):
            self.core.tensor_core_gemm(f32(4, 5), f32(6, 3))

    def test_tensor_core_gemm_increments_total_ops(self):
        before = self.core._total_ops
        self.core.tensor_core_gemm(f32(4, 4), f32(4, 4))
        assert self.core._total_ops == before + 1

    # ── batched_gemm ──────────────────────────────────────────────────────────
    def test_batched_gemm_3d_correct(self):
        A = rng.standard_normal((3, 5, 4)).astype(np.float32)
        B = rng.standard_normal((3, 4, 6)).astype(np.float32)
        got = self.core.batched_gemm(A, B)
        ref = np.matmul(A, B)
        np.testing.assert_allclose(got, ref, atol=1e-5)

    def test_batched_gemm_2d_correct(self):
        A = f32(5, 4)
        B = f32(4, 3)
        np.testing.assert_allclose(self.core.batched_gemm(A, B), A @ B, atol=1e-5)

    def test_batched_gemm_inner_dim_mismatch_raises(self):
        with pytest.raises(ShapeError):
            self.core.batched_gemm(f32(2, 4, 5), f32(2, 6, 3))

    # ── mixed_precision_gemm ──────────────────────────────────────────────────
    def test_mixed_precision_gemm_close_to_fp32(self):
        A = f32(8, 6)
        B = f32(6, 4)
        got = self.core.mixed_precision_gemm(A, B)
        np.testing.assert_allclose(got, A @ B, atol=5e-2)

    def test_mixed_precision_gemm_wrong_rank_raises(self):
        with pytest.raises(ShapeError):
            self.core.mixed_precision_gemm(f32(2, 4, 6), f32(2, 6, 3))

    # ── layer_norm ────────────────────────────────────────────────────────────
    def test_layer_norm_output_shape(self):
        X = f32(4, 8)
        gamma = np.ones(8, dtype=np.float32)
        beta = np.zeros(8, dtype=np.float32)
        out = self.core.layer_norm(X, gamma, beta)
        assert out.shape == X.shape

    def test_layer_norm_zero_mean_unit_var(self):
        X = rng.standard_normal((10, 16)).astype(np.float32) * 5.0 + 3.0
        gamma = np.ones(16, dtype=np.float32)
        beta = np.zeros(16, dtype=np.float32)
        out = self.core.layer_norm(X, gamma, beta)
        np.testing.assert_allclose(out.mean(axis=-1), np.zeros(10), atol=1e-5)
        np.testing.assert_allclose(out.var(axis=-1), np.ones(10), atol=1e-4)

    # ── batch_norm ────────────────────────────────────────────────────────────
    def test_batch_norm_training_output_shape(self):
        X = f32(8, 4, 5, 5)
        gamma = np.ones(4, dtype=np.float32)
        beta = np.zeros(4, dtype=np.float32)
        out, rm, rv = self.core.batch_norm(X, gamma, beta, training=True)
        assert out.shape == X.shape
        assert rm.shape == (4,)
        assert rv.shape == (4,)

    def test_batch_norm_eval_without_running_stats_raises(self):
        X = f32(4, 3)
        gamma = np.ones(3, dtype=np.float32)
        beta = np.zeros(3, dtype=np.float32)
        with pytest.raises(GPUError):
            self.core.batch_norm(X, gamma, beta, training=False)

    def test_batch_norm_eval_with_running_stats(self):
        X = f32(4, 3)
        gamma = np.ones(3, dtype=np.float32)
        beta = np.zeros(3, dtype=np.float32)
        rm = np.zeros(3, dtype=np.float32)
        rv = np.ones(3, dtype=np.float32)
        out, _, _ = self.core.batch_norm(X, gamma, beta,
                                          running_mean=rm, running_var=rv,
                                          training=False)
        assert out.shape == X.shape

    # ── activations ───────────────────────────────────────────────────────────
    def test_gelu_shape_preserved(self):
        X = f32(5, 8)
        assert self.core.gelu(X).shape == X.shape

    def test_gelu_matches_tanh_approximation(self):
        X = f32(100)
        got = self.core.gelu(X)
        ref = 0.5 * X * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (X + 0.044715 * X ** 3)))
        np.testing.assert_allclose(got, ref.astype(np.float32), atol=1e-5)

    def test_silu_shape_preserved(self):
        X = f32(5, 8)
        assert self.core.silu(X).shape == X.shape

    def test_silu_matches_sigmoid_gate(self):
        X = f32(100)
        got = self.core.silu(X)
        ref = X * (1.0 / (1.0 + np.exp(-X)))
        np.testing.assert_allclose(got, ref.astype(np.float32), atol=1e-5)

    # ── conv3d ────────────────────────────────────────────────────────────────
    def test_conv3d_output_shape_no_padding(self):
        X = rng.standard_normal((2, 3, 8, 8, 8)).astype(np.float32)
        W = rng.standard_normal((4, 3, 3, 3, 3)).astype(np.float32)
        out = self.core.conv3d(X, W)
        # D_out = H_out = W_out = (8-3)//1 + 1 = 6
        assert out.shape == (2, 4, 6, 6, 6)

    def test_conv3d_channel_mismatch_raises(self):
        X = rng.standard_normal((1, 3, 4, 4, 4)).astype(np.float32)
        W = rng.standard_normal((4, 5, 3, 3, 3)).astype(np.float32)
        with pytest.raises(ShapeError):
            self.core.conv3d(X, W)

    def test_conv3d_wrong_rank_raises(self):
        with pytest.raises(ShapeError):
            self.core.conv3d(f32(2, 3, 8, 8), f32(4, 3, 3, 3, 3))

    # ── grouped_gemm ──────────────────────────────────────────────────────────
    def test_grouped_gemm_all_correct(self):
        A_list = [f32(4, 3) for _ in range(3)]
        B_list = [f32(3, 5) for _ in range(3)]
        results = self.core.grouped_gemm(A_list, B_list)
        assert len(results) == 3
        for A, B, C in zip(A_list, B_list, results):
            np.testing.assert_allclose(C, A @ B, atol=1e-5)

    def test_grouped_gemm_mismatched_count_raises(self):
        with pytest.raises(ShapeError):
            self.core.grouped_gemm([f32(4, 3)], [f32(3, 5), f32(3, 5)])

    # ── fused_attention_norm ──────────────────────────────────────────────────
    def test_fused_attention_norm_shape(self):
        Q = f32(2, 6, 8)
        gamma = np.ones(8, dtype=np.float32)
        beta = np.zeros(8, dtype=np.float32)
        out = self.core.fused_attention_norm(Q, Q, Q, gamma, beta)
        assert out.shape == Q.shape

    def test_fused_attention_norm_equals_attn_then_layernorm(self):
        Q = f32(1, 4, 8)
        gamma = np.ones(8, dtype=np.float32) * 2.0
        beta = np.ones(8, dtype=np.float32) * 0.5
        fused = self.core.fused_attention_norm(Q, Q, Q, gamma, beta)
        attn = self.core.flash_attention(Q, Q, Q)
        manual = self.core.layer_norm(attn, gamma, beta)
        np.testing.assert_allclose(fused, manual, atol=1e-5)

    # ── status ────────────────────────────────────────────────────────────────
    def test_status_has_expected_keys(self):
        self.core.tensor_core_gemm(f32(4, 4), f32(4, 4))
        s = self.core.status()
        for key in ("lanes", "tile_shape", "precision", "tensor_cores",
                    "tensor_core_details", "memory_pool", "total_ops_executed"):
            assert key in s
        assert s["total_ops_executed"] >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# HyperGPU
# ═══════════════════════════════════════════════════════════════════════════════

class TestHyperGPU:
    def setup_method(self):
        self.gpu = HyperGPU(tensor_cores=2)

    def test_attention_matches_flash(self):
        Q = f32(2, 5, 8)
        out1 = self.gpu.attention(Q, Q, Q)
        out2 = self.gpu.flash_attention(Q, Q, Q)
        np.testing.assert_allclose(out1, out2, atol=1e-6)

    def test_flash_attention_causal_differs_from_noncausal(self):
        Q = f32(1, 6, 8)
        causal = self.gpu.flash_attention(Q, Q, Q, causal=True)
        noncausal = self.gpu.flash_attention(Q, Q, Q, causal=False)
        assert not np.allclose(causal, noncausal)

    def test_conv2d_output_shape(self):
        X = rng.standard_normal((2, 3, 8, 8)).astype(np.float32)
        W = rng.standard_normal((4, 3, 3, 3)).astype(np.float32)
        out = self.gpu.conv2d(X, W, stride=1, padding=1)
        assert out.shape == (2, 4, 8, 8)

    def test_conv3d_output_shape(self):
        X = rng.standard_normal((1, 2, 6, 6, 6)).astype(np.float32)
        W = rng.standard_normal((3, 2, 3, 3, 3)).astype(np.float32)
        out = self.gpu.conv3d(X, W)
        assert out.shape == (1, 3, 4, 4, 4)

    def test_layer_norm_output_shape(self):
        X = f32(4, 8)
        gamma = np.ones(8, dtype=np.float32)
        beta = np.zeros(8, dtype=np.float32)
        assert self.gpu.layer_norm(X, gamma, beta).shape == X.shape

    def test_batch_norm_training_output_shape(self):
        X = f32(8, 4)
        gamma = np.ones(4, dtype=np.float32)
        beta = np.zeros(4, dtype=np.float32)
        out, rm, rv = self.gpu.batch_norm(X, gamma, beta)
        assert out.shape == X.shape

    def test_gelu_shape_preserved(self):
        X = f32(4, 8)
        assert self.gpu.gelu(X).shape == X.shape

    def test_silu_shape_preserved(self):
        X = f32(4, 8)
        assert self.gpu.silu(X).shape == X.shape

    def test_softmax_sums_to_one(self):
        X = f32(4, 10)
        Y = self.gpu.softmax(X)
        np.testing.assert_allclose(Y.sum(axis=-1), np.ones(4), atol=1e-6)

    def test_add_correct(self):
        A = f32(5, 5)
        B = f32(5, 5)
        np.testing.assert_allclose(self.gpu.add(A, B), A + B)

    def test_grouped_gemm_all_correct(self):
        A_list = [f32(4, 3) for _ in range(2)]
        B_list = [f32(3, 5) for _ in range(2)]
        results = self.gpu.grouped_gemm(A_list, B_list)
        assert len(results) == 2

    def test_fused_attention_norm_shape(self):
        Q = f32(1, 4, 8)
        gamma = np.ones(8, dtype=np.float32)
        beta = np.zeros(8, dtype=np.float32)
        out = self.gpu.fused_attention_norm(Q, Q, Q, gamma, beta)
        assert out.shape == Q.shape

    def test_flush_vram_no_crash(self):
        self.gpu.vram.alloc(f32(5, 5))
        self.gpu.flush_vram()
        assert self.gpu.vram.used_bytes == 0

    def test_silicon_report_none_without_model(self):
        assert self.gpu.silicon_report() is None

    def test_status_has_expected_keys(self):
        s = self.gpu.status()
        for key in ("engine", "lanes", "tensor_cores", "precision",
                    "total_ops", "total_compute_ms", "vram", "uptime_s"):
            assert key in s
        assert s["engine"] == "HyperGPU"

    def test_total_compute_ms_grows_after_ops(self):
        before = self.gpu._total_compute_ms
        self.gpu.layer_norm(f32(32, 64),
                            np.ones(64, np.float32), np.zeros(64, np.float32))
        assert self.gpu._total_compute_ms >= before


class TestHyperGPUGemmPaths:
    """Test gemm paths separately — they route through pocket accelerator."""

    def test_gemm_correct(self):
        gpu = HyperGPU()
        A = f32(8, 6)
        B = f32(6, 4)
        np.testing.assert_allclose(gpu.gemm(A, B), A @ B, atol=1e-4)

    def test_mixed_gemm_close_to_fp32(self):
        gpu = HyperGPU()
        A = f32(8, 6)
        B = f32(6, 4)
        np.testing.assert_allclose(gpu.mixed_gemm(A, B), A @ B, atol=5e-2)

    def test_gemm_batched_3d_correct(self):
        gpu = HyperGPU()
        A = rng.standard_normal((3, 5, 4)).astype(np.float32)
        B = rng.standard_normal((3, 4, 6)).astype(np.float32)
        got = gpu.gemm_batched(A, B)
        np.testing.assert_allclose(got, np.matmul(A, B), atol=1e-4)


# ═══════════════════════════════════════════════════════════════════════════════
# GPUCluster / GPUClusterNode
# ═══════════════════════════════════════════════════════════════════════════════

class TestGPUClusterNode:
    def test_initial_state_is_idle(self):
        node = GPUClusterNode(node_id=0)
        assert node.state == "idle"
        assert node._assigned_task is None

    def test_assign_sets_busy(self):
        node = GPUClusterNode(node_id=0)
        node.assign("task_A")
        assert node.state == "busy"
        assert node._assigned_task == "task_A"

    def test_release_restores_idle(self):
        node = GPUClusterNode(node_id=0)
        node.assign("task_A")
        node.release()
        assert node.state == "idle"
        assert node._assigned_task is None

    def test_status_has_expected_keys(self):
        node = GPUClusterNode(node_id=3)
        node.assign("my_task")
        s = node.status()
        assert s["node_id"] == 3
        assert s["state"] == "busy"
        assert s["assigned_task"] == "my_task"
        assert "gpu" in s


class TestGPUCluster:
    def setup_method(self):
        self.cluster = GPUCluster(num_nodes=3)

    def test_num_nodes_correct(self):
        assert self.cluster.num_nodes == 3

    def test_get_idle_node_returns_node(self):
        node = self.cluster.get_idle_node()
        assert node is not None
        assert node.state == "idle"

    def test_get_idle_node_none_when_all_busy(self):
        for node in self.cluster.nodes.values():
            node.assign("busy")
        assert self.cluster.get_idle_node() is None

    def test_get_node_invalid_id_raises(self):
        with pytest.raises(GPUError):
            self.cluster.get_node(999)

    def test_all_reduce_gradients_averages(self):
        grads = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        avg = self.cluster.all_reduce_gradients("w", grads)
        np.testing.assert_allclose(avg, [2.0, 3.0])

    def test_all_reduce_empty_raises(self):
        with pytest.raises(GPUError):
            self.cluster.all_reduce_gradients("w", [])

    def test_scatter_data_splits_evenly(self):
        data = np.arange(12).reshape(6, 2).astype(np.float32)
        chunks = self.cluster.scatter_data(data, num_chunks=3)
        assert len(chunks) == 3
        assert all(c.shape == (2, 2) for c in chunks)

    def test_gather_results_concatenates(self):
        chunks = [np.ones((2, 3)) * i for i in range(3)]
        out = self.cluster.gather_results(chunks)
        assert out.shape == (6, 3)

    def test_scatter_then_gather_roundtrip(self):
        data = rng.standard_normal((9, 4)).astype(np.float32)
        chunks = self.cluster.scatter_data(data, num_chunks=3)
        reassembled = self.cluster.gather_results(chunks)
        np.testing.assert_array_equal(reassembled, data)

    def test_add_node_increments_count(self):
        before = self.cluster.num_nodes
        self.cluster.add_node()
        assert self.cluster.num_nodes == before + 1

    def test_remove_node_decrements_count(self):
        nid = self.cluster.add_node()
        before = self.cluster.num_nodes
        self.cluster.remove_node(nid)
        assert self.cluster.num_nodes == before - 1

    def test_total_lanes_computed(self):
        assert self.cluster.total_lanes == 3 * 512  # default 512 per node

    def test_total_tensor_cores_computed(self):
        assert self.cluster.total_tensor_cores == 3 * 8  # default 8 per node

    def test_flush_all_no_crash(self):
        for node in self.cluster.nodes.values():
            node.gpu.vram.alloc(f32(5, 5))
        self.cluster.flush_all()
        for node in self.cluster.nodes.values():
            assert node.gpu.vram.used_bytes == 0

    def test_status_has_expected_keys(self):
        s = self.cluster.status()
        for key in ("engine", "num_nodes", "total_lanes", "total_tensor_cores",
                    "nodes_idle", "nodes_busy", "total_ops", "nodes"):
            assert key in s
        assert s["engine"] == "HyperGPU Cluster"
        assert s["num_nodes"] == 3

    def test_run_distributed_correct(self):
        """run_distributed fans a lambda out across nodes and collects results."""
        data = f32(6, 4)
        chunks = self.cluster.scatter_data(data, num_chunks=3)

        def identity_fn(gpu, chunk):
            return gpu.add(chunk, np.zeros_like(chunk))

        results = self.cluster.run_distributed(identity_fn, chunks)
        reassembled = self.cluster.gather_results(results)
        np.testing.assert_allclose(reassembled, data, atol=1e-6)

    def test_run_distributed_thread_safety(self):
        """Concurrent distributed calls on disjoint node sets must not crash."""
        errors = []

        def one_run():
            try:
                data = f32(3, 4)
                chunks = self.cluster.scatter_data(data, num_chunks=3)
                self.cluster.run_distributed(
                    lambda gpu, c: gpu.add(c, np.zeros_like(c)), chunks)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=one_run) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
