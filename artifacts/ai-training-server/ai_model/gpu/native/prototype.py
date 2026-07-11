"""PROTOTYPE — Path A: compile a fused SIMD kernel to CPU and beat idiomatic numpy.

This is the honest "software GPU done right" direction: instead of *interpreting*
an op as a chain of numpy calls (each a full read+write pass over memory, with
temporaries), we *compile* one fused loop that gcc auto-vectorizes to AVX and
that touches memory once. The win is memory traffic + fusion, not magic — and it
is a real, measured CPU speedup, still on CPU (not GPU hardware).

Kernel: y = scale * relu(a*x + b)^2   (an affine -> relu -> square -> scale chain)

Run:  python -m ai_model.gpu.native.prototype
"""
import ctypes
import hashlib
import os
import subprocess
import tempfile
import time

import numpy as np

C_SOURCE = r"""
#include <stddef.h>
// Fused affine -> relu -> square -> scale, one pass over memory.
void affine_relu_sq(const float* x, float* y, float a, float b,
                    float scale, size_t n) {
    #pragma GCC ivdep
    for (size_t i = 0; i < n; ++i) {
        float t = a * x[i] + b;
        t = t > 0.0f ? t : 0.0f;
        y[i] = scale * t * t;
    }
}
"""


def _compile(src: str) -> ctypes.CDLL:
    key = hashlib.sha1(src.encode()).hexdigest()[:12]
    d = os.path.join(tempfile.gettempdir(), "digital_gpu_native")
    os.makedirs(d, exist_ok=True)
    cpath = os.path.join(d, f"k_{key}.c")
    sopath = os.path.join(d, f"k_{key}.so")
    with open(cpath, "w") as f:
        f.write(src)
    # -march=native is stripped on NixOS; use an explicit ISA flag instead.
    flags = ["gcc", "-O3", "-mavx2", "-mfma", "-funroll-loops", "-ffast-math",
             "-shared", "-fPIC", "-o", sopath, cpath, "-lm"]
    subprocess.run(flags, check=True, capture_output=True, text=True)
    return ctypes.CDLL(sopath)


def main() -> None:
    lib = _compile(C_SOURCE)
    fn = lib.affine_relu_sq
    fn.restype = None
    fn.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                   ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_size_t]
    fp = ctypes.POINTER(ctypes.c_float)

    n = 8_000_000
    rng = np.random.default_rng(0)
    x = rng.standard_normal(n).astype(np.float32)
    a, b, scale = np.float32(1.3), np.float32(-0.2), np.float32(0.5)

    def numpy_op(x):
        return scale * np.maximum(a * x + b, 0.0) ** 2   # idiomatic: temporaries

    def native_op(x):
        y = np.empty_like(x)
        fn(x.ctypes.data_as(fp), y.ctypes.data_as(fp), a, b, scale, x.size)
        return y

    ref = numpy_op(x)
    got = native_op(x)
    ok = np.allclose(ref, got, rtol=1e-5, atol=1e-5)
    max_err = float(np.max(np.abs(ref - got)))

    def bench(f, reps=50):
        f(x)  # warm
        t0 = time.perf_counter()
        for _ in range(reps):
            f(x)
        return (time.perf_counter() - t0) / reps * 1000.0

    t_np = bench(numpy_op)
    t_c = bench(native_op)
    print("Path-A prototype: fused affine->relu->square->scale")
    print(f"  n={n:,}  numerically equal: {ok}  (max_abs_err={max_err:.2e})")
    print(f"  numpy (idiomatic): {t_np:.3f} ms/call")
    print(f"  native SIMD (gcc): {t_c:.3f} ms/call")
    print(f"  speedup: {t_np / t_c:.2f}x")


if __name__ == "__main__":
    main()
