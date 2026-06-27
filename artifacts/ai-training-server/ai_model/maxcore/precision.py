"""Precision / quantization — the GPU tensor-core *numerics reference*.

IMPORTANT — this module is a reference spec, **not** a CPU speed optimization.
Measured on this host, integer matmul through NumPy is ~50x *slower* than float32
BLAS: CPUs have no int8/int4 fast path — that speed lives in real GPU tensor
cores. The value here is exact, validated low-bit numerics so a future real-GPU
backend can reproduce behavior bit-for-bit, plus accuracy/memory modelling for
low bit-widths.

Schemes implemented (symmetric, zero-point = 0):
  * **per-tensor**  — one scale for the whole tensor. Simple; fine at INT8,
    lossy at INT4.
  * **per-channel** — one scale per output column. Markedly better at low
    bit-widths because columns with different dynamic range get their own scale.

All functions are pure NumPy and side-effect free. ``MODE`` documents intent so
callers cannot mistake this for a production CPU path.
"""
from __future__ import annotations

import numpy as np

# Marks every result of this module: a numerics reference, never a CPU fast path.
MODE = "reference_numerics"


def _max_int(num_bits: int) -> int:
    """Largest representable magnitude for a symmetric signed ``num_bits`` int."""
    if num_bits < 2:
        raise ValueError(f"num_bits must be >= 2, got {num_bits}")
    return (2 ** (num_bits - 1)) - 1


def calibrate_scale(x, num_bits: int = 8, percentile: float = 0.999) -> float:
    """Per-tensor symmetric scale mapping the abs ``percentile`` to ``max_int``.

    Using a high percentile (not the raw max) clips a few outliers so the bulk of
    the distribution keeps resolution — the standard calibration trick.
    """
    if not 0.0 < percentile <= 1.0:
        raise ValueError(f"percentile must be in (0, 1], got {percentile}")
    arr = np.asarray(x, dtype=np.float32)
    mx = float(np.quantile(np.abs(arr), percentile))
    return mx / _max_int(num_bits) if mx > 0 else 1.0


def quantize(x, num_bits: int = 8, scale: float | None = None) -> tuple[np.ndarray, float]:
    """Symmetric per-tensor quantize to int (stored in int8). Returns ``(q, scale)``.

    Low bit-widths (e.g. 4) still use int8 storage but a smaller value range — a
    faithful *numeric* model of INT4 without bit-packing (which buys nothing on a
    CPU and would only obscure the reference).
    """
    arr = np.asarray(x, dtype=np.float32)
    if scale is None:
        scale = calibrate_scale(arr, num_bits)
    mi = _max_int(num_bits)
    q = np.clip(np.round(arr / np.float32(scale)), -mi, mi).astype(np.int8)
    return q, scale


def dequantize(q, scale: float) -> np.ndarray:
    return np.asarray(q, dtype=np.float32) * np.float32(scale)


def quantize_per_channel(
    x, axis: int = 0, num_bits: int = 8, percentile: float = 0.999
) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric per-channel quantize: an independent scale per slice along ``axis``.

    For a weight matrix ``[K, N]`` use ``axis=0`` to get one scale per output
    column (shape ``[1, N]``). Returns ``(q, scale)`` where ``scale`` broadcasts
    against ``x``.
    """
    if not 0.0 < percentile <= 1.0:
        raise ValueError(f"percentile must be in (0, 1], got {percentile}")
    arr = np.asarray(x, dtype=np.float32)
    mi = _max_int(num_bits)
    absmax = np.quantile(np.abs(arr), percentile, axis=axis, keepdims=True)
    scale = np.where(absmax > 0, absmax / mi, np.float32(1.0)).astype(np.float32)
    q = np.clip(np.round(arr / scale), -mi, mi).astype(np.int8)
    return q, scale


def dequantize_per_channel(q, scale) -> np.ndarray:
    return np.asarray(q, dtype=np.float32) * np.asarray(scale, dtype=np.float32)


def quantized_matmul(
    a, b, num_bits_a: int = 8, num_bits_b: int = 8, percentile: float = 0.999
) -> tuple[np.ndarray, dict[str, float]]:
    """Per-tensor quantized ``a @ b``: quantize both, accumulate in int32, rescale.

    Numerically equivalent to a real integer-tensor-core matmul + dequant; see the
    module note on why this is a reference, not a CPU speedup.
    """
    A = np.asarray(a, dtype=np.float32)
    B = np.asarray(b, dtype=np.float32)
    sa = calibrate_scale(A, num_bits_a, percentile)
    sb = calibrate_scale(B, num_bits_b, percentile)
    qa, _ = quantize(A, num_bits_a, sa)
    qb, _ = quantize(B, num_bits_b, sb)
    acc = qa.astype(np.int32) @ qb.astype(np.int32)
    out = acc.astype(np.float32) * np.float32(sa * sb)
    return out, {"scale_a": sa, "scale_b": sb, "out_scale": sa * sb}


def quantized_matmul_per_channel(
    a, b, num_bits_a: int = 8, num_bits_b: int = 8, percentile: float = 0.999
) -> tuple[np.ndarray, np.ndarray]:
    """Quantized ``a @ b`` with per-tensor activations and per-output-column
    weights — the scheme that makes INT4 weights usable. Returns ``(out, scale_b)``.
    """
    A = np.asarray(a, dtype=np.float32)
    B = np.asarray(b, dtype=np.float32)
    sa = calibrate_scale(A, num_bits_a, percentile)
    qa, _ = quantize(A, num_bits_a, sa)
    qb, sb = quantize_per_channel(B, axis=0, num_bits=num_bits_b, percentile=percentile)
    acc = qa.astype(np.int32) @ qb.astype(np.int32)
    out = acc.astype(np.float32) * (np.float32(sa) * sb.reshape(1, -1))
    return out, sb


def relative_error(approx, exact) -> float:
    """Relative Frobenius-norm error ``||approx - exact|| / ||exact||``."""
    ap = np.asarray(approx, dtype=np.float32)
    ex = np.asarray(exact, dtype=np.float32)
    return float(np.linalg.norm(ap - ex) / (np.linalg.norm(ex) + 1e-9))
