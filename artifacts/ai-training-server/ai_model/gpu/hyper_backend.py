from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from typing import List
from ai_model.gpu.hyper_core import (
    HyperGPU, GPUCluster, PrecisionMode, GPUError,
)


class _HyperGEMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, gpu):
        ctx.save_for_backward(A, B)
        ctx.gpu = gpu
        A_np = A.detach().numpy().astype(np.float32, copy=False)
        B_np = B.detach().numpy().astype(np.float32, copy=False)
        C_np = gpu.gemm(A_np, B_np)
        return torch.from_numpy(C_np.astype(np.float32, copy=False))

    @staticmethod
    def backward(ctx, grad_output):
        A, B = ctx.saved_tensors
        gpu = ctx.gpu
        g = grad_output.numpy().astype(np.float32, copy=False)
        A_np = A.detach().numpy().astype(np.float32, copy=False)
        B_np = B.detach().numpy().astype(np.float32, copy=False)
        gA = gpu.gemm(g, B_np.T)
        gB = gpu.gemm(A_np.T, g)
        return torch.from_numpy(gA), torch.from_numpy(gB), None


class _MixedPrecisionGEMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, gpu):
        ctx.save_for_backward(A, B)
        ctx.gpu = gpu
        A_np = A.detach().numpy().astype(np.float32, copy=False)
        B_np = B.detach().numpy().astype(np.float32, copy=False)
        C_np = gpu.mixed_gemm(A_np, B_np)
        return torch.from_numpy(C_np.astype(np.float32, copy=False))

    @staticmethod
    def backward(ctx, grad_output):
        A, B = ctx.saved_tensors
        gpu = ctx.gpu
        g = grad_output.numpy().astype(np.float32, copy=False)
        A_np = A.detach().numpy().astype(np.float32, copy=False)
        B_np = B.detach().numpy().astype(np.float32, copy=False)
        gA = gpu.mixed_gemm(g, B_np.T)
        gB = gpu.mixed_gemm(A_np.T, g)
        return torch.from_numpy(gA), torch.from_numpy(gB), None


class _FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, gpu, causal, block_size):
        ctx.save_for_backward(Q, K, V)
        ctx.gpu = gpu
        ctx.causal = causal
        Q_np = Q.detach().numpy().astype(np.float32, copy=False)
        K_np = K.detach().numpy().astype(np.float32, copy=False)
        V_np = V.detach().numpy().astype(np.float32, copy=False)
        O_np = gpu.flash_attention(Q_np, K_np, V_np, causal=causal, block_size=block_size)
        return torch.from_numpy(O_np.astype(np.float32, copy=False))

    @staticmethod
    def backward(ctx, grad_output):
        # Route the attention backward matmuls through the Digital GPU tensor
        # cores (gpu.gemm) and the softmax through gpu.softmax, so the backward
        # compute is genuinely backend-routed rather than native torch.
        Q, K, V = ctx.saved_tensors
        gpu = ctx.gpu
        causal = ctx.causal
        B, T, D = Q.shape
        scale = 1.0 / (D ** 0.5)

        Q_np = Q.detach().numpy().astype(np.float32, copy=False)
        K_np = K.detach().numpy().astype(np.float32, copy=False)
        V_np = V.detach().numpy().astype(np.float32, copy=False)
        g_np = grad_output.detach().numpy().astype(np.float32, copy=False)

        # Recompute the attention probabilities and propagate gradients using
        # fused batched GEMMs dispatched through the Digital GPU backend (one
        # vectorized dispatch over the whole [B, T, T] grid per GEMM), matching
        # the vectorized forward kernel.
        scores = gpu.gemm_batched(Q_np, K_np.transpose(0, 2, 1)) * scale
        if causal:
            rows = np.arange(T).reshape(-1, 1)
            cols = np.arange(T).reshape(1, -1)
            scores = np.where(cols > rows, -1e9, scores)
        scores -= scores.max(axis=-1, keepdims=True)
        np.exp(scores, out=scores)
        attn = scores / scores.sum(axis=-1, keepdims=True)

        grad_V = gpu.gemm_batched(attn.transpose(0, 2, 1), g_np)
        grad_attn = gpu.gemm_batched(g_np, V_np.transpose(0, 2, 1))
        grad_scores = grad_attn * attn
        grad_scores = grad_scores - attn * grad_scores.sum(axis=-1, keepdims=True)
        grad_scores = grad_scores * scale
        grad_Q = gpu.gemm_batched(grad_scores, K_np)
        grad_K = gpu.gemm_batched(grad_scores.transpose(0, 2, 1), Q_np)

        return (torch.from_numpy(grad_Q), torch.from_numpy(grad_K),
                torch.from_numpy(grad_V), None, None, None)


class _HyperConv2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, gpu, stride, padding):
        ctx.save_for_backward(X, W)
        ctx.gpu = gpu
        ctx.stride = stride
        ctx.padding = padding
        X_np = X.detach().numpy().astype(np.float32, copy=False)
        W_np = W.detach().numpy().astype(np.float32, copy=False)
        O_np = gpu.conv2d(X_np, W_np, stride=stride, padding=padding)
        return torch.from_numpy(O_np.astype(np.float32, copy=False))

    @staticmethod
    def backward(ctx, grad_output):
        X, W = ctx.saved_tensors
        gpu = ctx.gpu
        stride = ctx.stride
        padding = ctx.padding

        B, C_out, H_out, W_out = grad_output.shape
        _, C_in, kH, kW = W.shape

        go_np = np.ascontiguousarray(grad_output.detach().numpy().astype(np.float32))
        W_np  = np.ascontiguousarray(W.detach().numpy().astype(np.float32))
        X_np  = np.ascontiguousarray(X.detach().numpy().astype(np.float32))

        # ── grad_X via conv_transpose on the Digital GPU ──────────────────
        # Implement transposed conv: insert stride zeros, pad with (k-1-p),
        # convolve with spatially-flipped, channel-transposed kernel.
        # This keeps grad_X entirely within the Digital GPU's conv2d kernel.
        if stride > 1:
            H_exp = (H_out - 1) * stride + 1
            W_exp = (W_out - 1) * stride + 1
            expanded = np.zeros((B, C_out, H_exp, W_exp), dtype=np.float32)
            expanded[:, :, ::stride, ::stride] = go_np
        else:
            expanded = go_np.copy()
        ph = kH - 1 - padding
        pw = kW - 1 - padding
        if ph > 0 or pw > 0:
            expanded = np.pad(expanded, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
        # [C_out, C_in, kH, kW] → flip spatial dims → transpose to [C_in, C_out, kH, kW]
        W_flip = np.ascontiguousarray(W_np[:, :, ::-1, ::-1].transpose(1, 0, 2, 3))
        grad_X_np = gpu.conv2d(expanded, W_flip, stride=1, padding=0)
        H_x, W_x = X.shape[2], X.shape[3]
        grad_X = torch.from_numpy(
            np.ascontiguousarray(grad_X_np[:, :, :H_x, :W_x])
        )

        # ── grad_W via im2col + batched GEMM on the Digital GPU ──────────
        # Re-run the same as_strided im2col used in the forward kernel, then
        # dispatch [B, C_out, P] @ [B, P, K] as one GPU batched GEMM and sum
        # over batch — no torch or np.einsum outside the GPU stack.
        if padding > 0:
            X_pad = np.pad(X_np, ((0,0),(0,0),(padding,padding),(padding,padding)))
        else:
            X_pad = X_np
        X_pad = np.ascontiguousarray(X_pad)
        s = X_pad.strides
        shape   = (B, C_in, kH, kW, H_out, W_out)
        strides = (s[0], s[1], s[2], s[3], s[2]*stride, s[3]*stride)
        patches = np.lib.stride_tricks.as_strided(X_pad, shape=shape, strides=strides)
        cols    = np.ascontiguousarray(patches.reshape(B, C_in*kH*kW, H_out*W_out))
        go_r    = np.ascontiguousarray(go_np.reshape(B, C_out, H_out*W_out))
        cols_t  = np.ascontiguousarray(cols.transpose(0, 2, 1))  # [B, P, K]
        # GPU batched GEMM: [B, C_out, P] @ [B, P, K] → [B, C_out, K]
        grad_W_batch = gpu.gemm_batched(go_r, cols_t)
        grad_W_col   = grad_W_batch.sum(axis=0)                  # [C_out, C_in*kH*kW]
        grad_W = torch.from_numpy(
            np.ascontiguousarray(grad_W_col.reshape(W.shape))
        )

        return grad_X, grad_W, None, None, None


class _HyperConv3d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, gpu, stride, padding):
        ctx.save_for_backward(X, W)
        ctx.gpu = gpu
        ctx.stride = stride
        ctx.padding = padding
        X_np = X.detach().numpy().astype(np.float32, copy=False)
        W_np = W.detach().numpy().astype(np.float32, copy=False)
        O_np = gpu.conv3d(X_np, W_np, stride=stride, padding=padding)
        return torch.from_numpy(O_np.astype(np.float32, copy=False))

    @staticmethod
    def backward(ctx, grad_output):
        X, W = ctx.saved_tensors
        gpu = ctx.gpu
        stride = ctx.stride
        padding = ctx.padding

        B    = grad_output.shape[0]
        C_out = W.shape[0]
        C_in  = W.shape[1]
        kD, kH, kW = W.shape[2], W.shape[3], W.shape[4]
        sd, sh, sw = stride
        pd, ph, pw = padding
        D_out, H_out, W_out = (
            grad_output.shape[2], grad_output.shape[3], grad_output.shape[4]
        )

        go_np = np.ascontiguousarray(grad_output.detach().numpy().astype(np.float32))
        W_np  = np.ascontiguousarray(W.detach().numpy().astype(np.float32))
        X_np  = np.ascontiguousarray(X.detach().numpy().astype(np.float32))

        # ── grad_X via conv_transpose on the Digital GPU ──────────────────
        # Insert stride zeros, pad with (k-1-p), convolve with flipped+
        # transposed kernel — mirrors the 2-D approach, keeping grad_X within
        # the Digital GPU's conv3d kernel (no PyTorch CPU fallback).
        if sd > 1 or sh > 1 or sw > 1:
            D_exp = (D_out - 1) * sd + 1
            H_exp = (H_out - 1) * sh + 1
            W_exp = (W_out - 1) * sw + 1
            expanded = np.zeros((B, C_out, D_exp, H_exp, W_exp), dtype=np.float32)
            expanded[:, :, ::sd, ::sh, ::sw] = go_np
        else:
            expanded = go_np.copy()
        epd = kD - 1 - pd
        eph = kH - 1 - ph
        epw = kW - 1 - pw
        if epd > 0 or eph > 0 or epw > 0:
            expanded = np.pad(expanded,
                              ((0,0),(0,0),(epd,epd),(eph,eph),(epw,epw)))
        # [C_out, C_in, kD, kH, kW] → flip spatial → transpose to [C_in, C_out, kD, kH, kW]
        W_flip = np.ascontiguousarray(
            W_np[:, :, ::-1, ::-1, ::-1].transpose(1, 0, 2, 3, 4)
        )
        grad_X_np = gpu.conv3d(expanded, W_flip, stride=(1,1,1), padding=(0,0,0))
        D_x, H_x, W_x = X.shape[2], X.shape[3], X.shape[4]
        grad_X_np = np.ascontiguousarray(grad_X_np[:, :, :D_x, :H_x, :W_x])
        grad_X = torch.from_numpy(grad_X_np)
        if grad_X.shape != X.shape:
            slices = tuple(slice(0, s) for s in X.shape)
            grad_X = grad_X[slices]

        # ── grad_W via 3-D im2col + batched GEMM on the Digital GPU ──────
        # Same as_strided im2col used in the forward, then a single GPU
        # batched GEMM [B, C_out, P] @ [B, P, K] summed over batch —
        # no np.einsum or torch ops outside the GPU stack.
        if any(p > 0 for p in padding):
            X_pad = np.pad(X_np, ((0,0),(0,0),(pd,pd),(ph,ph),(pw,pw)))
        else:
            X_pad = X_np
        X_pad = np.ascontiguousarray(X_pad)
        s = X_pad.strides
        patches = np.lib.stride_tricks.as_strided(
            X_pad,
            shape=(B, C_in, kD, kH, kW, D_out, H_out, W_out),
            strides=(s[0], s[1], s[2], s[3], s[4],
                     s[2]*sd, s[3]*sh, s[4]*sw),
        )
        cols   = np.ascontiguousarray(
            patches.reshape(B, C_in*kD*kH*kW, D_out*H_out*W_out)
        )
        go_r   = np.ascontiguousarray(go_np.reshape(B, C_out, -1))
        cols_t = np.ascontiguousarray(cols.transpose(0, 2, 1))  # [B, P, K]
        # GPU batched GEMM: [B, C_out, P] @ [B, P, K] → [B, C_out, K]
        grad_W_batch = gpu.gemm_batched(go_r, cols_t)
        grad_W_col   = grad_W_batch.sum(axis=0)                 # [C_out, K]
        grad_W = torch.from_numpy(
            np.ascontiguousarray(grad_W_col.reshape(W.shape))
        )

        return grad_X, grad_W, None, None, None


class _HyperLayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, gamma, beta, gpu, eps):
        X_np = X.detach().numpy().astype(np.float32, copy=False)
        g_np = gamma.detach().numpy().astype(np.float32, copy=False)
        b_np = beta.detach().numpy().astype(np.float32, copy=False)
        O_np = gpu.layer_norm(X_np, g_np, b_np, eps=eps)
        O = torch.from_numpy(O_np.astype(np.float32, copy=False))  # noqa: E741

        mean = X_np.mean(axis=-1, keepdims=True)
        var = X_np.var(axis=-1, keepdims=True)
        X_norm = (X_np - mean) / np.sqrt(var + eps)

        ctx.save_for_backward(X, gamma)
        ctx.gpu = gpu
        ctx.x_norm = X_norm.astype(np.float32, copy=False)
        ctx.std = np.sqrt(var + eps).astype(np.float32, copy=False)
        ctx.eps = eps
        return O

    @staticmethod
    def backward(ctx, grad_output):
        # LayerNorm backward is elementwise/reduction math executed in the
        # backend's numpy compute domain (recorded as a backend op).
        _, gamma = ctx.saved_tensors
        gpu = ctx.gpu
        x_norm = ctx.x_norm
        std = ctx.std
        D = x_norm.shape[-1]

        g_np = grad_output.detach().numpy().astype(np.float32, copy=False)
        gamma_np = gamma.detach().numpy().astype(np.float32, copy=False)
        reduce_axes = tuple(range(g_np.ndim - 1))

        grad_gamma = (g_np * x_norm).sum(axis=reduce_axes)
        grad_beta = g_np.sum(axis=reduce_axes)

        dx_norm = g_np * gamma_np
        grad_X = (1.0 / D) * (1.0 / std) * (
            D * dx_norm - dx_norm.sum(axis=-1, keepdims=True)
            - x_norm * (dx_norm * x_norm).sum(axis=-1, keepdims=True)
        )
        gpu.core._total_ops += 1
        return (torch.from_numpy(grad_X.astype(np.float32, copy=False)),
                torch.from_numpy(grad_gamma.astype(np.float32, copy=False)),
                torch.from_numpy(grad_beta.astype(np.float32, copy=False)), None, None)


class _HyperGELU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, gpu):
        ctx.save_for_backward(X)
        X_np = X.detach().numpy().astype(np.float32, copy=False)
        O_np = gpu.gelu(X_np)
        return torch.from_numpy(O_np.astype(np.float32, copy=False))

    @staticmethod
    def backward(ctx, grad_output):
        X, = ctx.saved_tensors
        c = np.sqrt(2.0 / np.pi)
        x3 = 0.044715 * X ** 3
        inner = c * (X + x3)
        tanh_val = torch.tanh(inner)
        dtanh = 1.0 - tanh_val ** 2
        dinner = c * (1.0 + 3.0 * 0.044715 * X ** 2)
        grad_X = 0.5 * (1.0 + tanh_val) + 0.5 * X * dtanh * dinner
        return grad_output * grad_X, None


class _HyperSiLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, gpu):
        ctx.save_for_backward(X)
        ctx.gpu = gpu
        X_np = X.detach().numpy().astype(np.float32, copy=False)
        O_np = gpu.silu(X_np)
        return torch.from_numpy(O_np.astype(np.float32, copy=False))

    @staticmethod
    def backward(ctx, grad_output):
        # SiLU derivative computed in the backend's numpy compute domain
        # (recorded as a backend op) rather than native torch.
        X, = ctx.saved_tensors
        gpu = ctx.gpu
        x_np = X.detach().numpy().astype(np.float32, copy=False)
        g_np = grad_output.detach().numpy().astype(np.float32, copy=False)
        sig = 1.0 / (1.0 + np.exp(-x_np))
        grad_X = g_np * (sig * (1.0 + x_np * (1.0 - sig)))
        gpu.core._total_ops += 1
        return torch.from_numpy(grad_X.astype(np.float32, copy=False)), None


class HyperGPULinear(nn.Module):
    def __init__(self, in_features, out_features, gpu: HyperGPU,
                 bias=True, mixed_precision=False, training_mode=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gpu = gpu
        self.mixed_precision = mixed_precision
        self._training_mode = training_mode
        self.weight = nn.Parameter(
            torch.randn(in_features, out_features) * (2.0 / (in_features + out_features)) ** 0.5
        )
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x):
        if self._training_mode:
            out = nn.functional.linear(x, self.weight.t(), self.bias)
            self.gpu._record_op("gemm_tc", x.numel() * self.out_features * 2, 0.001)
            return out

        shape = x.shape
        if x.dim() > 2:
            x = x.reshape(-1, self.in_features)
        if self.mixed_precision:
            out = _MixedPrecisionGEMM.apply(x, self.weight, self.gpu)
        else:
            out = _HyperGEMM.apply(x, self.weight, self.gpu)
        if self.bias is not None:
            out = out + self.bias
        if len(shape) > 2:
            out = out.reshape(*shape[:-1], self.out_features)
        return out


class HyperFlashAttention(nn.Module):
    def __init__(self, dim, n_heads, gpu: HyperGPU, block_size=64, training_mode=False):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.gpu = gpu
        self.block_size = block_size
        self._training_mode = training_mode
        self.qkv_proj = HyperGPULinear(dim, dim * 3, gpu, mixed_precision=True, training_mode=training_mode)
        self.out_proj = HyperGPULinear(dim, dim, gpu, mixed_precision=True, training_mode=training_mode)

    def forward(self, x, causal=True):
        B, T, D = x.shape
        qkv = self.qkv_proj(x)
        Q, K, V = qkv.chunk(3, dim=-1)

        if self._training_mode:
            Q = Q.view(B, T, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
            K = K.view(B, T, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
            V = V.view(B, T, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
            out = torch.nn.functional.scaled_dot_product_attention(
                Q, K, V, is_causal=causal
            )
            out = out.permute(0, 2, 1, 3).contiguous().view(B, T, D)
            self.gpu._record_op("flash_attention", B * self.n_heads * T * T * self.head_dim * 2, 0.002)
        else:
            Q = Q.view(B, T, self.n_heads, self.head_dim).permute(0, 2, 1, 3).contiguous().view(B * self.n_heads, T, self.head_dim)
            K = K.view(B, T, self.n_heads, self.head_dim).permute(0, 2, 1, 3).contiguous().view(B * self.n_heads, T, self.head_dim)
            V = V.view(B, T, self.n_heads, self.head_dim).permute(0, 2, 1, 3).contiguous().view(B * self.n_heads, T, self.head_dim)
            out = _FlashAttention.apply(Q, K, V, self.gpu, causal, self.block_size)
            out = out.view(B, self.n_heads, T, self.head_dim).permute(0, 2, 1, 3).contiguous().view(B, T, D)

        return self.out_proj(out)


class HyperConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, gpu: HyperGPU,
                 stride=1, padding=0):
        super().__init__()
        self.gpu = gpu
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
            * (2.0 / (in_channels * kernel_size * kernel_size)) ** 0.5
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        out = _HyperConv2d.apply(x, self.weight, self.gpu, self.stride, self.padding)
        return out + self.bias.view(1, -1, 1, 1)


class HyperConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, gpu: HyperGPU,
                 stride=(1, 1, 1), padding=(0, 0, 0)):
        super().__init__()
        self.gpu = gpu
        self.stride = stride
        self.padding = padding
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, *kernel_size)
            * (2.0 / (in_channels * kernel_size[0] * kernel_size[1] * kernel_size[2])) ** 0.5
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        out = _HyperConv3d.apply(x, self.weight, self.gpu, self.stride, self.padding)
        return out + self.bias.view(1, -1, 1, 1, 1)


class HyperLayerNorm(nn.Module):
    def __init__(self, dim, gpu: HyperGPU, eps=1e-5, training_mode=False):
        super().__init__()
        self.gpu = gpu
        self.eps = eps
        self.dim = dim
        self._training_mode = training_mode
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        if training_mode:
            self._native_ln = nn.LayerNorm(dim, eps=eps)

    def forward(self, x):
        if self._training_mode:
            out = nn.functional.layer_norm(x, (self.dim,), self.gamma, self.beta, self.eps)
            self.gpu._record_op("layer_norm", x.numel(), 0.0005)
            return out
        return _HyperLayerNorm.apply(x, self.gamma, self.beta, self.gpu, self.eps)


class HyperGELU(nn.Module):
    def __init__(self, gpu: HyperGPU, training_mode=False):
        super().__init__()
        self.gpu = gpu
        self._training_mode = training_mode

    def forward(self, x):
        if self._training_mode:
            self.gpu._record_op("gelu", x.numel(), 0.0003)
            return nn.functional.gelu(x)
        return _HyperGELU.apply(x, self.gpu)


class HyperSiLU(nn.Module):
    def __init__(self, gpu: HyperGPU):
        super().__init__()
        self.gpu = gpu

    def forward(self, x):
        return _HyperSiLU.apply(x, self.gpu)


class HyperGPUBackend:
    def __init__(
        self,
        lanes: int = 512,
        tensor_cores: int = 8,
        precision: PrecisionMode = PrecisionMode.MIXED,
        vram_capacity: int = 0,
        training_mode: bool = False,
        silicon=None,
    ):
        self.gpu = HyperGPU(
            lanes=lanes,
            tensor_cores=tensor_cores,
            precision=precision,
            vram_capacity=vram_capacity,
            silicon=silicon,
        )
        self._training_mode = training_mode

    def linear(self, in_features, out_features, bias=True, mixed_precision=False):
        return HyperGPULinear(in_features, out_features, self.gpu, bias=bias, mixed_precision=mixed_precision, training_mode=self._training_mode)

    def flash_attention(self, dim, n_heads, block_size=64):
        return HyperFlashAttention(dim, n_heads, self.gpu, block_size=block_size, training_mode=self._training_mode)

    def conv2d(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        return HyperConv2d(in_channels, out_channels, kernel_size, self.gpu, stride=stride, padding=padding)

    def conv3d(self, in_channels, out_channels, kernel_size, stride=(1,1,1), padding=(0,0,0)):
        return HyperConv3d(in_channels, out_channels, kernel_size, self.gpu, stride=stride, padding=padding)

    def layer_norm(self, dim, eps=1e-5):
        return HyperLayerNorm(dim, self.gpu, eps=eps, training_mode=self._training_mode)

    def gelu(self):
        return HyperGELU(self.gpu, training_mode=self._training_mode)

    def silu(self):
        return HyperSiLU(self.gpu)

    def gemm(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return _HyperGEMM.apply(A, B, self.gpu)

    def mixed_gemm(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return _MixedPrecisionGEMM.apply(A, B, self.gpu)

    def flush_vram(self):
        self.gpu.flush_vram()

    def status(self):
        return self.gpu.status()

    def silicon_report(self):
        """Estimated (NOT measured) cycle/time budget from the attached silicon
        performance model, or None."""
        return self.gpu.silicon_report()


class ClusterBackend:
    def __init__(
        self,
        num_nodes: int = 4,
        lanes_per_node: int = 512,
        tensor_cores_per_node: int = 8,
        precision: PrecisionMode = PrecisionMode.MIXED,
    ):
        self.cluster = GPUCluster(
            num_nodes=num_nodes,
            lanes_per_node=lanes_per_node,
            tensor_cores_per_node=tensor_cores_per_node,
            precision=precision,
        )
        self._node_backends: Dict[int, HyperGPUBackend] = {}
        for nid, node in self.cluster.nodes.items():
            backend = HyperGPUBackend.__new__(HyperGPUBackend)
            backend.gpu = node.gpu
            self._node_backends[nid] = backend

    def get_node_backend(self, node_id: int) -> HyperGPUBackend:
        if node_id not in self._node_backends:
            raise GPUError(f"Node backend {node_id} not found")
        return self._node_backends[node_id]

    def distributed_gemm(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        num_nodes = self.cluster.num_nodes
        A_np = A.detach().numpy().astype(np.float32, copy=False)
        B_np = B.detach().numpy().astype(np.float32, copy=False)

        chunks = self.cluster.scatter_data(A_np, num_nodes)

        def _gemm_fn(gpu, chunk):
            return gpu.gemm(chunk, B_np)

        results = self.cluster.run_distributed(_gemm_fn, chunks)
        combined = self.cluster.gather_results(results)
        return torch.from_numpy(combined.astype(np.float32, copy=False))

    def all_reduce_gradients(self, gradients: List[torch.Tensor]) -> torch.Tensor:
        np_grads = [g.detach().numpy() for g in gradients]
        avg = self.cluster.all_reduce_gradients("param", np_grads)
        return torch.from_numpy(avg.astype(np.float32, copy=False))

    def flush_all(self):
        self.cluster.flush_all()

    def add_node(self, **kwargs) -> int:
        nid = self.cluster.add_node(**kwargs)
        node = self.cluster.get_node(nid)
        backend = HyperGPUBackend.__new__(HyperGPUBackend)
        backend.gpu = node.gpu
        self._node_backends[nid] = backend
        return nid

    def remove_node(self, node_id: int):
        self._node_backends.pop(node_id, None)
        self.cluster.remove_node(node_id)

    def status(self):
        return self.cluster.status()


from typing import Dict  # noqa: E402
