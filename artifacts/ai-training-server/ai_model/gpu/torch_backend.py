import torch
import torch.nn as nn
import numpy as np
from ai_model.gpu.digital_gpu import DigitalGPU


class _DigitalGEMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, gpu):
        ctx.save_for_backward(A, B)
        ctx.gpu = gpu
        A_np = A.detach().numpy().astype(np.float64)
        B_np = B.detach().numpy().astype(np.float64)
        C_np = gpu.gemm(A_np, B_np)
        return torch.from_numpy(C_np.astype(np.float32))

    @staticmethod
    def backward(ctx, grad_output):
        A, B = ctx.saved_tensors
        gpu = ctx.gpu
        grad_np = grad_output.numpy().astype(np.float64)
        A_np = A.detach().numpy().astype(np.float64)
        B_np = B.detach().numpy().astype(np.float64)
        grad_A = gpu.gemm(grad_np, B_np.T)
        grad_B = gpu.gemm(A_np.T, grad_np)
        return torch.from_numpy(grad_A.astype(np.float32)), torch.from_numpy(grad_B.astype(np.float32)), None


class _DigitalAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, gpu):
        A_np = A.detach().numpy().astype(np.float64)
        B_np = B.detach().numpy().astype(np.float64)
        C_np = gpu.add(A_np, B_np)
        return torch.from_numpy(C_np.astype(np.float32))

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output, None


class _DigitalSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, axis, gpu):
        X_np = X.detach().numpy().astype(np.float64)
        Y_np = gpu.softmax(X_np, axis=axis)
        Y = torch.from_numpy(Y_np.astype(np.float32))
        ctx.save_for_backward(Y)
        return Y

    @staticmethod
    def backward(ctx, grad_output):
        Y, = ctx.saved_tensors
        gY = grad_output * Y
        return gY - Y * gY.sum(dim=-1, keepdim=True), None, None


class _DigitalAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, gpu, causal):
        Q_np = Q.detach().numpy().astype(np.float64)
        K_np = K.detach().numpy().astype(np.float64)
        V_np = V.detach().numpy().astype(np.float64)
        O_np = gpu.attention(Q_np, K_np, V_np, causal=causal)
        O = torch.from_numpy(O_np.astype(np.float32))
        ctx.save_for_backward(Q, K, V)
        ctx.gpu = gpu
        ctx.causal = causal
        return O

    @staticmethod
    def backward(ctx, grad_output):
        Q, K, V = ctx.saved_tensors
        causal = ctx.causal
        B, T, D = Q.shape
        scale = 1.0 / (D ** 0.5)
        scores = torch.bmm(Q, K.transpose(1, 2)) * scale
        if causal:
            causal_mask = torch.triu(torch.full((T, T), float('-inf')), diagonal=1)
            scores = scores + causal_mask.unsqueeze(0)
        attn = torch.softmax(scores, dim=-1)
        grad_V = torch.bmm(attn.transpose(1, 2), grad_output)
        grad_attn = torch.bmm(grad_output, V.transpose(1, 2))
        grad_scores = grad_attn * attn
        grad_scores = grad_scores - attn * grad_scores.sum(dim=-1, keepdim=True)
        grad_scores = grad_scores * scale
        grad_Q = torch.bmm(grad_scores, K)
        grad_K = torch.bmm(grad_scores.transpose(1, 2), Q)
        return grad_Q, grad_K, grad_V, None, None


class _DigitalGEMMBiasReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, bias, gpu):
        A_np = A.detach().numpy().astype(np.float64)
        B_np = B.detach().numpy().astype(np.float64)
        bias_np = bias.detach().numpy().astype(np.float64)
        O_np = gpu.gemm_bias_relu(A_np, B_np, bias_np)
        O = torch.from_numpy(O_np.astype(np.float32))
        ctx.save_for_backward(A, B, bias, O)
        ctx.gpu = gpu
        return O

    @staticmethod
    def backward(ctx, grad_output):
        A, B, bias, O = ctx.saved_tensors
        gpu = ctx.gpu
        relu_mask = (O > 0).float()
        grad_pre_relu = grad_output * relu_mask
        grad_np = grad_pre_relu.numpy().astype(np.float64)
        A_np = A.detach().numpy().astype(np.float64)
        B_np = B.detach().numpy().astype(np.float64)
        grad_A = gpu.gemm(grad_np, B_np.T)
        grad_B = gpu.gemm(A_np.T, grad_np)
        grad_bias_np = grad_np.sum(axis=0)
        return (
            torch.from_numpy(grad_A.astype(np.float32)),
            torch.from_numpy(grad_B.astype(np.float32)),
            torch.from_numpy(grad_bias_np.astype(np.float32)),
            None,
        )


class DigitalGPULinear(nn.Module):
    def __init__(self, in_features, out_features, gpu, bias=True, fused_relu=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gpu = gpu
        self.fused_relu = fused_relu
        self.weight = nn.Parameter(torch.randn(in_features, out_features) * (2.0 / (in_features + out_features)) ** 0.5)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    def forward(self, x):
        shape = x.shape
        if x.dim() > 2:
            x = x.reshape(-1, self.in_features)

        if self.fused_relu and self.bias is not None:
            out = _DigitalGEMMBiasReLU.apply(x, self.weight, self.bias, self.gpu)
        else:
            out = _DigitalGEMM.apply(x, self.weight, self.gpu)
            if self.bias is not None:
                out = out + self.bias

        if len(shape) > 2:
            out = out.reshape(*shape[:-1], self.out_features)
        return out


class DigitalGPUAttention(nn.Module):
    def __init__(self, dim, n_heads, gpu):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.gpu = gpu
        self.qkv_proj = DigitalGPULinear(dim, dim * 3, gpu)
        self.out_proj = DigitalGPULinear(dim, dim, gpu)

    def forward(self, x, mask=None, causal=True):
        B, T, D = x.shape
        qkv = self.qkv_proj(x)
        Q, K, V = qkv.chunk(3, dim=-1)
        Q = Q.view(B, T, self.n_heads, self.head_dim).permute(0, 2, 1, 3).contiguous().view(B * self.n_heads, T, self.head_dim)
        K = K.view(B, T, self.n_heads, self.head_dim).permute(0, 2, 1, 3).contiguous().view(B * self.n_heads, T, self.head_dim)
        V = V.view(B, T, self.n_heads, self.head_dim).permute(0, 2, 1, 3).contiguous().view(B * self.n_heads, T, self.head_dim)
        out = _DigitalAttention.apply(Q, K, V, self.gpu, causal)
        out = out.view(B, self.n_heads, T, self.head_dim).permute(0, 2, 1, 3).contiguous().view(B, T, D)
        return self.out_proj(out)


class DigitalGPUSoftmax(nn.Module):
    def __init__(self, gpu, dim=-1):
        super().__init__()
        self.gpu = gpu
        self.dim = dim

    def forward(self, x):
        return _DigitalSoftmax.apply(x, self.dim, self.gpu)


class DigitalGPUBackend:
    def __init__(self, lanes=32):
        self.gpu = DigitalGPU(lanes=lanes)
        self._profile_history = []

    def linear(self, in_features, out_features, bias=True, fused_relu=False):
        return DigitalGPULinear(in_features, out_features, self.gpu, bias=bias, fused_relu=fused_relu)

    def attention(self, dim, n_heads):
        return DigitalGPUAttention(dim, n_heads, self.gpu)

    def softmax(self, dim=-1):
        return DigitalGPUSoftmax(self.gpu, dim=dim)

    def gemm(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return _DigitalGEMM.apply(A, B, self.gpu)

    def profile(self):
        return self.gpu.last_profile()

    def flush_vram(self):
        self.gpu.vram._store.clear()
        self.gpu.vram._meta.clear()

    def status(self):
        vram_count = len(self.gpu.vram._store)
        vram_bytes = sum(a.nbytes for a in self.gpu.vram._store.values())
        return {
            "lanes": self.gpu.core.lanes,
            "tile_size": f"{self.gpu.core.tile_m}x{self.gpu.core.tile_n}x{self.gpu.core.tile_k}",
            "vram_handles": vram_count,
            "vram_bytes": vram_bytes,
            "vram_mb": round(vram_bytes / (1024 * 1024), 2),
        }
