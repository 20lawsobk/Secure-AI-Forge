"""CPU numerics models for low-precision formats (fp16 / bf16 / fp8).

WHAT THIS IS — and is NOT
-------------------------
* This is a *numerics model*. It rounds float values onto the grid representable
  by a lower-precision format, so you can measure the error that format would
  introduce. All arithmetic still happens on this CPU in fp32/fp64.
* It is NOT hardware fp8/fp16 execution and it is NOT a speedup — the rounding
  is extra work, so these paths are *slower* than plain fp32 here. On real
  silicon these formats are fast because dedicated units execute them; this
  module only reproduces their *rounding behaviour*, honestly.
* Use it for quantization-error study, calibration, and mixed-precision numerics
  validation — never to claim tensor-core / sm_10x throughput.

fp8 formats follow the OCP 8-bit spec:
  * e4m3: 1-4-3, bias 7, max normal 448.0, saturating (no inf in this model).
  * e5m2: 1-5-2, bias 15, max normal 57344.0.
"""
from __future__ import annotations

from typing import Literal

import numpy as np

FP8Format = Literal["e4m3", "e5m2"]

# mant_bits, exp_bits, bias, max_normal, has_inf
#   e4m3: OCP saturating variant — no inf; finite overflow saturates to 448.
#   e5m2: IEEE-like — has inf/nan; finite overflow rounds up to inf.
_FP8 = {
    "e4m3": dict(mant_bits=3, exp_bits=4, bias=7, max_normal=448.0, has_inf=False),
    "e5m2": dict(mant_bits=2, exp_bits=5, bias=15, max_normal=57344.0, has_inf=True),
}


def _round_to_grid(x: np.ndarray, mant_bits: int, min_exp: int,
                   max_normal: float, has_inf: bool) -> np.ndarray:
    """Round to a float grid with ``mant_bits`` mantissa bits and normals down to
    ``2**min_exp`` (values below that use subnormal spacing). Round-half-to-even.

    Non-finite handling (so the model is faithful at the edges):
      * ``nan`` -> ``nan``.
      * finite overflow beyond ``max_normal``: saturates to ``max_normal`` when
        ``has_inf`` is False (e4m3), else rounds up to ``inf`` (e5m2).
      * ``±inf`` input: preserved as ``±inf`` when ``has_inf`` (e5m2), else
        saturated to ``±max_normal`` (e4m3, which has no infinity).
    """
    x = np.asarray(x, dtype=np.float64)
    sign = np.signbit(x)
    ax = np.abs(x)
    finite = np.isfinite(x)
    nz = finite & (ax > 0)

    # Compute log2 only on safe (finite, non-zero) magnitudes to avoid warnings.
    safe_ax = np.where(nz, ax, 1.0)
    e = np.floor(np.log2(safe_ax))
    e_clamped = np.maximum(e, float(min_exp))     # fixed subnormal spacing below min
    step = np.power(2.0, e_clamped - mant_bits)

    work_ax = np.where(finite, ax, 0.0)
    q = np.round(work_ax / step) * step           # numpy round is round-half-to-even
    q = np.where(nz, q, 0.0)

    over = q > max_normal
    q = np.where(over, (np.inf if has_inf else max_normal), q)

    signed = np.where(sign, -q, q)
    # Restore non-finite inputs explicitly.
    signed = np.where(np.isnan(x), np.nan, signed)
    inf_val = np.inf if has_inf else max_normal
    signed = np.where(np.isposinf(x), inf_val, signed)
    signed = np.where(np.isneginf(x), -inf_val, signed)
    return signed


def to_fp8(x, fmt: FP8Format = "e4m3") -> np.ndarray:
    """Round ``x`` to the fp8 grid and return represented values as float64.

    (On real hardware each element would occupy one byte; here we return the
    decoded value because the point is the *numerics*, not the byte layout.)
    """
    if fmt not in _FP8:
        raise ValueError(f"unknown fp8 format {fmt!r}; valid: {tuple(_FP8)}")
    p = _FP8[fmt]
    min_exp = 1 - p["bias"]
    return _round_to_grid(x, p["mant_bits"], min_exp, p["max_normal"], p["has_inf"])


def to_fp16(x) -> np.ndarray:
    """Round to IEEE fp16 and back to fp32 (models fp16 storage/compute grid)."""
    return np.asarray(x, dtype=np.float32).astype(np.float16).astype(np.float32)


def to_bf16(x) -> np.ndarray:
    """Round fp32 -> bf16 (8 exp / 7 mantissa bits) round-to-nearest-even -> fp32."""
    a = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
    u = a.view(np.uint32).astype(np.uint64)
    lsb = (u >> 16) & 1
    u = (u + 0x7FFF + lsb) & 0xFFFF0000
    return u.astype(np.uint32).view(np.float32)


def cast_numeric(x, profile: str) -> np.ndarray:
    """Apply a numeric profile (see opcode_spec.VALID_PROFILES) as a rounding model."""
    if profile == "fp32_strict":
        return np.asarray(x, dtype=np.float32)
    if profile == "fp16_mixed":
        return to_fp16(x)
    if profile == "bf16_mixed":
        return to_bf16(x)
    if profile == "fp8_mixed":
        return to_fp8(x, "e4m3")
    raise ValueError(f"unknown numeric profile {profile!r}")


def flash_attention_fp8_model(Q, K, V, causal: bool = False,
                              fmt: FP8Format = "e4m3") -> np.ndarray:
    """MODEL of fp8 FlashAttention: fp8 inputs, fp16 accumulate, fp32 softmax,
    fp16 output. Batch dims are any leading axes; attention is over the last two
    (T, D). This reproduces the *numerics* of an sm_102-style fp8 kernel on CPU —
    it is not that kernel and is not faster than the fp32 attention here.
    """
    Qf = to_fp8(Q, fmt)
    Kf = to_fp8(K, fmt)
    Vf = to_fp8(V, fmt)
    D = Qf.shape[-1]
    scale = 1.0 / np.sqrt(float(D))

    scores = np.matmul(Qf, np.swapaxes(Kf, -1, -2)) * scale
    scores = to_fp16(scores)               # fp16 accumulate (modelled)
    if causal:
        Tq, Tk = scores.shape[-2], scores.shape[-1]
        mask = np.triu(np.full((Tq, Tk), -1e9, dtype=np.float64), k=1)
        scores = scores + mask

    scores = scores - scores.max(axis=-1, keepdims=True)   # fp32 softmax
    w = np.exp(scores)
    w = w / w.sum(axis=-1, keepdims=True)
    O = np.matmul(to_fp16(w), Vf)
    return to_fp16(O)


def quantization_error(x, profile: str = "fp8_mixed") -> dict:
    """Report max/mean absolute error introduced by a numeric profile on ``x``.

    A convenience for the honest use case this module exists for: *measuring*
    what a low-precision format costs numerically.
    """
    ref = np.asarray(x, dtype=np.float64)
    q = np.asarray(cast_numeric(ref, profile), dtype=np.float64)
    err = np.abs(q - ref)
    denom = np.abs(ref).mean() or 1.0
    return {
        "profile": profile,
        "max_abs_err": float(err.max()) if err.size else 0.0,
        "mean_abs_err": float(err.mean()) if err.size else 0.0,
        "rel_mean_abs_err": float(err.mean() / denom) if err.size else 0.0,
    }
