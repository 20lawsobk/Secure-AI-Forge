"""Tests for the DigitalGPUBackend bridge methods: alloc/free/run_kernel.

These verify the byte-sized allocator and the named-kernel dispatch are *real*
numpy execution (matching numpy reference results), and that a hardware-implying
kernel name is honestly refused rather than silently run as plain numpy.
"""
import numpy as np

from ai_model.gpu.torch_backend import DigitalGPUBackend


def _mk():
    return DigitalGPUBackend()


def test_alloc_returns_usable_handle_and_free():
    b = _mk()
    h = b.alloc(1024)
    assert isinstance(h, int)
    assert b.gpu.vram.get(h).nbytes == 1024  # real buffer of the requested bytes
    b.free(h)
    # freed handle is gone
    try:
        b.gpu.vram.get(h)
        assert False, "handle should be freed"
    except Exception:
        pass


def test_alloc_rejects_negative():
    b = _mk()
    try:
        b.alloc(-1)
        assert False, "negative size must raise"
    except ValueError:
        pass


def test_alloc_rejects_fractional_size():
    """A fractional byte size must raise, not silently truncate."""
    b = _mk()
    try:
        b.alloc(3.5)
        assert False, "fractional size must raise"
    except ValueError:
        pass
    # a whole-valued float is fine
    h = b.alloc(8.0)
    assert b.gpu.vram.get(h).nbytes == 8


def test_run_kernel_gemm_matches_numpy():
    b = _mk()
    A = np.random.default_rng(0).standard_normal((16, 24)).astype(np.float64)
    B = np.random.default_rng(1).standard_normal((24, 8)).astype(np.float64)
    out = b.run_kernel("gemm", A, B)
    assert np.allclose(out, A @ B, atol=1e-6)


def test_run_kernel_attention_matches_reference():
    b = _mk()
    rng = np.random.default_rng(2)
    Q = rng.standard_normal((2, 5, 4))
    K = rng.standard_normal((2, 5, 4))
    V = rng.standard_normal((2, 5, 4))
    out = b.run_kernel("attention", Q, K, V, causal=False)
    # reference softmax attention
    scores = np.einsum("bid,bjd->bij", Q, K) / np.sqrt(Q.shape[-1])
    w = np.exp(scores - scores.max(-1, keepdims=True))
    w = w / w.sum(-1, keepdims=True)
    ref = np.einsum("bij,bjd->bid", w, V)
    assert np.allclose(out, ref, atol=1e-6)


def test_run_kernel_case_insensitive():
    b = _mk()
    A = np.ones((4, 4)); B = np.ones((4, 4))
    assert np.allclose(b.run_kernel("GEMM", A, B), A @ B)


def test_run_kernel_refuses_fp8_sm102_kernel():
    """A hardware-implying name must raise, not silently run numpy."""
    b = _mk()
    raised = False
    try:
        b.run_kernel("flash_attention_fp8_sm102", np.ones((1, 2, 2)),
                     np.ones((1, 2, 2)), np.ones((1, 2, 2)))
    except ValueError as e:
        raised = "no kernel named" in str(e)
    assert raised, "fp8/sm102 kernel name must be honestly refused"


def test_run_kernel_unknown_name_lists_supported():
    b = _mk()
    try:
        b.run_kernel("does_not_exist")
        assert False, "unknown kernel must raise"
    except ValueError as e:
        assert "gemm" in str(e) and "attention" in str(e)
