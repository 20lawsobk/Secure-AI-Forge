"""Tests for the precision/quantization numerics reference.

Runnable two ways:
  * pytest:  uv run pytest ai_model/maxcore/tests/test_precision.py
  * direct:  uv run python ai_model/maxcore/tests/test_precision.py

These validate the *numerics* (round-trip fidelity, matmul error bounds, and that
per-channel scaling beats per-tensor at INT4) against independent numpy ground
truth. They make NO speed claims — integer matmul is a reference, not a CPU fast
path (see ai_model/maxcore/precision.py).
"""
from __future__ import annotations

import os
import sys

import numpy as np

_SERVER_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if _SERVER_DIR not in sys.path:
    sys.path.insert(0, _SERVER_DIR)

from ai_model.maxcore.precision import (  # noqa: E402
    calibrate_scale,
    dequantize,
    quantize,
    quantized_matmul,
    quantized_matmul_per_channel,
    relative_error,
)


def test_quant_roundtrip_within_one_step():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((64, 64)).astype(np.float32)
    scale = calibrate_scale(x, num_bits=8)
    q, s = quantize(x, num_bits=8, scale=scale)
    xr = dequantize(q, s)
    # Values inside the calibrated range reconstruct within half a step (pure
    # rounding error). A few outliers beyond the 0.999 percentile are clipped by
    # design, so they are excluded from the tight bound.
    clip = s * 127  # max_int for symmetric int8
    inside = np.abs(x) <= clip
    assert np.max(np.abs(xr[inside] - x[inside])) <= s * 0.5 + 1e-5
    assert q.dtype == np.int8


def test_int8_per_tensor_matmul_accurate():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((128, 96)).astype(np.float32)
    b = rng.standard_normal((96, 80)).astype(np.float32)
    ref = a @ b
    out, meta = quantized_matmul(a, b, num_bits_a=8, num_bits_b=8)
    assert set(meta) == {"scale_a", "scale_b", "out_scale"}
    assert relative_error(out, ref) < 0.05  # INT8 is comfortably usable


def test_int4_per_channel_beats_per_tensor():
    # Build a weight matrix whose columns have wildly different magnitudes — the
    # case where a single per-tensor scale wastes resolution and per-channel wins.
    rng = np.random.default_rng(2)
    a = rng.standard_normal((64, 48)).astype(np.float32)
    cols = []
    for j in range(40):
        mag = np.float32(10.0 ** ((j % 4) - 1))  # 0.1 .. 100 across columns
        cols.append(rng.standard_normal(48).astype(np.float32) * mag)
    b = np.stack(cols, axis=1).astype(np.float32)
    ref = a @ b
    out_pt, _ = quantized_matmul(a, b, num_bits_a=8, num_bits_b=4)
    out_pc, _ = quantized_matmul_per_channel(a, b, num_bits_a=8, num_bits_b=4)
    err_pt = relative_error(out_pt, ref)
    err_pc = relative_error(out_pc, ref)
    assert err_pc < err_pt  # per-channel strictly improves INT4 weights
    assert err_pc < 0.10


def test_outliers_saturate_at_clip_boundary():
    # Calibration intentionally clips beyond the percentile, so extreme values
    # saturate at +/- max_int*scale (they do NOT round-trip). This documents that
    # expected, honest behavior rather than pretending quantization is lossless.
    x = np.array([0.0, 0.1, 0.2, 0.3, 1000.0], dtype=np.float32)
    scale = calibrate_scale(x, num_bits=8, percentile=0.5)  # tight clip
    q, s = quantize(x, num_bits=8, scale=scale)
    assert q.max() == 127  # the 1000.0 outlier saturates
    xr = dequantize(q, s)
    assert xr[-1] == 127 * s  # reconstructed to the clip ceiling, not 1000.0
    assert xr[-1] < 1000.0


def test_invalid_percentile_raises():
    x = np.ones((4, 4), dtype=np.float32)
    for bad in (0.0, -0.1, 1.5):
        try:
            calibrate_scale(x, percentile=bad)
            raise AssertionError(f"expected ValueError for percentile={bad}")
        except ValueError:
            pass


def test_deterministic():
    rng = np.random.default_rng(3)
    a = rng.standard_normal((32, 32)).astype(np.float32)
    b = rng.standard_normal((32, 32)).astype(np.float32)
    o1, _ = quantized_matmul(a, b)
    o2, _ = quantized_matmul(a, b)
    assert np.array_equal(o1, o2)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
