"""Tests for the native SIMD kernel library.

These assert two things: (1) every kernel is numerically correct vs a numpy
reference on both the native path and the forced-numpy fallback path, and (2) the
honesty contract holds (the path never claims to be GPU hardware).
"""
import os

import numpy as np

from ai_model.gpu.native.compiler import NativeCompiler
from ai_model.gpu.native.kernels import NativeKernels, get_native_kernels


def _ref_affine_relu_sq(x, a, b, s):
    return (np.float32(s) * np.maximum(np.float32(a) * x + np.float32(b), 0.0) ** 2
            ).astype(np.float32)


def _ref_hardswish(x):
    return (x * np.clip(x + 3.0, 0.0, 6.0) / 6.0).astype(np.float32)


def _ref_axpby(x, y, a, b):
    return (np.float32(a) * x + np.float32(b) * y).astype(np.float32)


def _ref_rmsnorm(x, g, eps):
    inv = 1.0 / np.sqrt(np.mean(x ** 2, axis=1, keepdims=True) + eps)
    return (x * inv * g).astype(np.float32)


def _fallback_kernels():
    """A NativeKernels forced onto the numpy fallback path (no compiler)."""
    comp = NativeCompiler()
    comp.available = False
    return NativeKernels(compiler=comp)


rng = np.random.default_rng(0)
X = rng.standard_normal(5000).astype(np.float32)
Y = rng.standard_normal(5000).astype(np.float32)
X2 = rng.standard_normal((64, 128)).astype(np.float32)
G = rng.standard_normal(128).astype(np.float32)


def test_fallback_is_available_without_compiler():
    k = _fallback_kernels()
    assert k.available is False
    # still produces correct results (correctness never depends on the compiler)
    assert np.allclose(k.hardswish(X), _ref_hardswish(X), atol=1e-5)
    assert k.stats["fallback"] >= 1 and k.stats["native"] == 0


def test_affine_relu_sq_correct():
    for k in (get_native_kernels(), _fallback_kernels()):
        got = k.affine_relu_sq(X, 1.3, -0.2, 0.5)
        assert np.allclose(got, _ref_affine_relu_sq(X, 1.3, -0.2, 0.5), atol=1e-4)


def test_hardswish_correct():
    for k in (get_native_kernels(), _fallback_kernels()):
        assert np.allclose(k.hardswish(X), _ref_hardswish(X), atol=1e-5)


def test_axpby_correct_and_shape_guard():
    for k in (get_native_kernels(), _fallback_kernels()):
        assert np.allclose(k.axpby(X, Y, 2.0, -3.0),
                           _ref_axpby(X, Y, 2.0, -3.0), atol=1e-4)
    try:
        get_native_kernels().axpby(X, Y[:10], 1.0, 1.0)
        assert False, "shape mismatch must raise"
    except ValueError:
        pass


def test_rmsnorm_rows_correct_and_guards():
    for k in (get_native_kernels(), _fallback_kernels()):
        got = k.rmsnorm_rows(X2, G, eps=1e-6)
        assert np.allclose(got, _ref_rmsnorm(X2, G, 1e-6), atol=1e-4)
    try:
        get_native_kernels().rmsnorm_rows(X, G)          # 1D input
        assert False
    except ValueError:
        pass
    try:
        get_native_kernels().rmsnorm_rows(X2, G[:10])    # bad gamma
        assert False
    except ValueError:
        pass


def test_non_float32_input_routes_correctly():
    k = get_native_kernels()
    x64 = X.astype(np.float64)
    # float64 in -> still correct (fallback), output is float32
    out = k.hardswish(x64)
    assert out.dtype == np.float32
    assert np.allclose(out, _ref_hardswish(X), atol=1e-5)


def test_describe_never_claims_hardware():
    d = get_native_kernels().describe()
    assert d["is_hardware_execution"] is False
    assert d["backend"] in ("native-cpu-simd", "numpy-fallback")
    assert "CPU" in d["note"] or "cpu" in d["note"]


def test_compiler_cache_reuse():
    c = NativeCompiler()
    if not c.available:
        return                       # no compiler on this host; nothing to test
    src = "void _t(float* x, unsigned long n){ for(unsigned long i=0;i<n;++i) x[i]+=1.0f; }"
    lib1 = c.compile(src, link=())
    lib2 = c.compile(src, link=())   # second call hits the cached .so
    assert lib1 is not None and lib2 is not None


def test_never_raise_on_broken_cache_dir():
    # Point the cache dir at a regular FILE so every fs op inside compile() fails.
    import tempfile as _tf
    fd, bogus = _tf.mkstemp()
    os.close(fd)
    try:
        c = NativeCompiler()
        c.cache_dir = bogus          # not a directory -> mkstemp(dir=...) will raise
        assert c.compile("void _z(void){}", link=()) is None   # never raises
        assert c.last_error is not None
        # NativeKernels must still initialize and produce correct numpy results
        k = NativeKernels(compiler=c)
        assert k.available is False
        assert np.allclose(k.hardswish(X), _ref_hardswish(X), atol=1e-5)
    finally:
        os.remove(bogus)


def test_concurrent_same_key_compile_no_raise():
    c = NativeCompiler()
    if not c.available:
        return
    from concurrent.futures import ThreadPoolExecutor
    src = "void _cc(float* x, unsigned long n){ for(unsigned long i=0;i<n;++i) x[i]*=2.0f; }"
    with ThreadPoolExecutor(max_workers=8) as ex:
        libs = list(ex.map(lambda _: c.compile(src, link=()), range(16)))
    assert all(lib is not None for lib in libs)   # no collisions / exceptions
