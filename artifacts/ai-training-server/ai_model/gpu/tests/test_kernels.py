"""Regression tests for the redesigned HyperSIMDCore kernels.

Covers the edge cases surfaced in review: cross-attention (Tk != Tq), sequence
length not a multiple of block_size, causal masking, block_size clamping, and
the conv dtype contract.
"""
from __future__ import annotations

import numpy as np

from ai_model.gpu.hyper_core import HyperSIMDCore


def _naive_attn(Q, K, V, causal=False):
    B, Tq, D = Q.shape
    Tk = K.shape[1]
    s = 1.0 / np.sqrt(D)
    sc = np.matmul(Q, K.transpose(0, 2, 1)) * s
    if causal:
        qi = np.arange(Tq)[:, None]; kj = np.arange(Tk)[None, :]
        sc = np.where(kj > qi, -1e9, sc)
    sc -= sc.max(-1, keepdims=True)
    np.exp(sc, out=sc)
    sc /= sc.sum(-1, keepdims=True)
    return np.matmul(sc, V)


def test_flash_matches_naive_self_attention():
    core = HyperSIMDCore()
    rng = np.random.default_rng(0)
    for causal in (False, True):
        for T in (7, 64, 130):  # incl. T not a multiple of block_size
            Q = rng.standard_normal((3, T, 16)).astype(np.float32)
            K = rng.standard_normal((3, T, 16)).astype(np.float32)
            V = rng.standard_normal((3, T, 16)).astype(np.float32)
            got = core.flash_attention(Q, K, V, causal=causal, block_size=32)
            assert np.abs(got - _naive_attn(Q, K, V, causal)).max() < 1e-4


def test_flash_cross_attention_tk_ne_tq():
    # Regression: block loop must iterate over Tk, not Tq.
    core = HyperSIMDCore()
    rng = np.random.default_rng(1)
    Q = rng.standard_normal((2, 5, 16)).astype(np.float32)   # Tq=5
    K = rng.standard_normal((2, 40, 16)).astype(np.float32)  # Tk=40 > Tq
    V = rng.standard_normal((2, 40, 16)).astype(np.float32)
    got = core.flash_attention(Q, K, V, block_size=16)
    assert got.shape == (2, 5, 16)
    assert np.abs(got - _naive_attn(Q, K, V)).max() < 1e-4

    Q2 = rng.standard_normal((2, 30, 16)).astype(np.float32)  # Tq=30 > Tk
    K2 = rng.standard_normal((2, 8, 16)).astype(np.float32)
    V2 = rng.standard_normal((2, 8, 16)).astype(np.float32)
    got2 = core.flash_attention(Q2, K2, V2, block_size=16)
    assert np.abs(got2 - _naive_attn(Q2, K2, V2)).max() < 1e-4


def test_flash_block_size_clamped_not_crashing():
    core = HyperSIMDCore()
    rng = np.random.default_rng(2)
    Q = rng.standard_normal((1, 12, 8)).astype(np.float32)
    K = rng.standard_normal((1, 12, 8)).astype(np.float32)
    V = rng.standard_normal((1, 12, 8)).astype(np.float32)
    # legacy no-op value (0) and an over-large value must both work.
    a = core.flash_attention(Q, K, V, block_size=0)
    b = core.flash_attention(Q, K, V, block_size=9999)
    assert np.abs(a - b).max() < 1e-5


def test_conv2d_preserves_input_dtype():
    core = HyperSIMDCore()
    rng = np.random.default_rng(3)
    for dt in (np.float32, np.float64):
        X = rng.standard_normal((2, 4, 10, 10)).astype(dt)
        W = rng.standard_normal((5, 4, 3, 3)).astype(dt)
        out = core.conv2d(X, W, stride=1, padding=1)
        assert out.dtype == dt
        assert out.shape == (2, 5, 10, 10)


def test_conv2d_matches_naive_strided_padded():
    core = HyperSIMDCore()
    rng = np.random.default_rng(4)

    def naive(X, W, stride, pad):
        N, C, H, Wd = X.shape; Co, _, kH, kW = W.shape
        Xp = np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
        Ho = (Xp.shape[2] - kH) // stride + 1
        Wo = (Xp.shape[3] - kW) // stride + 1
        o = np.zeros((N, Co, Ho, Wo), np.float32)
        for n in range(N):
            for co in range(Co):
                for i in range(Ho):
                    for j in range(Wo):
                        o[n, co, i, j] = np.sum(
                            Xp[n, :, i*stride:i*stride+kH, j*stride:j*stride+kW] * W[co])
        return o

    for stride in (1, 2):
        for pad in (0, 1):
            X = rng.standard_normal((2, 4, 11, 11)).astype(np.float32)
            W = rng.standard_normal((5, 4, 3, 3)).astype(np.float32)
            got = core.conv2d(X, W, stride=stride, padding=pad)
            assert np.abs(got - naive(X, W, stride, pad)).max() < 1e-3
