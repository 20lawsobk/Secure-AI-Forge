from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from ai_model.gpu.hyper_core import (
    HyperGPU, HyperSIMDCore, GPUCluster, GPUClusterNode,
    PrecisionMode, GPUError,
)
from ai_model.gpu.torch_backend import (
    _DigitalGEMM, _DigitalAttention, _DigitalSoftmax, _DigitalGEMMBiasReLU,
    DigitalGPULinear, DigitalGPUAttention, DigitalGPUSoftmax,
)


class _HyperGEMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, gpu):
        ctx.save_for_backward(A, B)
        ctx.gpu = gpu
        A_np = A.detach().numpy().astype(np.float32)
        B_np = B.detach().numpy().astype(np.float32)
        C_np = gpu.gemm(A_np, B_np)
        return torch.from_numpy(C_np.astype(np.float32))

    @staticmethod
    def backward(ctx, grad_output):
        A, B = ctx.saved_tensors
        gpu = ctx.gpu
        g = grad_output.numpy().astype(np.float32)
        A_np = A.detach().numpy().astype(np.float32)
        B_np = B.detach().numpy().astype(np.float32)
        gA = gpu.gemm(g, B_np.T)
        gB = gpu.gemm(A_np.T, g)
        return torch.from_numpy(gA), torch.from_numpy(gB), None


class _MixedPrecisionGEMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, gpu):
        ctx.save_for_backward(A, B)
        ctx.gpu = gpu
        A_np = A.detach().numpy().astype(np.float32)
        B_np = B.detach().numpy().astype(np.float32)
        C_np = gpu.mixed_gemm(A_np, B_np)
        return torch.from_numpy(C_np.astype(np.float32))

    @staticmethod
    def backward(ctx, grad_output):
        A, B = ctx.saved_tensors
        gpu = ctx.gpu
        g = grad_output.numpy().astype(np.float32)
        A_np = A.detach().numpy().astype(np.float32)
        B_np = B.detach().numpy().astype(np.float32)
        gA = gpu.mixed_gemm(g, B_np.T)
        gB = gpu.mixed_gemm(A_np.T, g)
        return torch.from_numpy(gA), torch.from_numpy(gB), None


class _FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, gpu, causal, block_size):
        ctx.save_for_backward(Q, K, V)
        ctx.gpu = gpu
        ctx.causal = causal
        Q_np = Q.detach().numpy().astype(np.float32)
        K_np = K.detach().numpy().astype(np.float32)
        V_np = V.detach().numpy().astype(np.float32)
        O_np = gpu.flash_attention(Q_np, K_np, V_np, causal=causal, block_size=block_size)
        return torch.from_numpy(O_np.astype(np.float32))

    @staticmethod
    def backward(ctx, grad_output):
        Q, K, V = ctx.saved_tensors
        causal = ctx.causal
        B, T, D = Q.shape
        scale = 1.0 / (D ** 0.5)
        scores = torch.bmm(Q, K.transpose(1, 2)) * scale
        if causal:
            mask = torch.triu(torch.full((T, T), float('-inf')), diagonal=1)
            scores = scores + mask.unsqueeze(0)
        attn = torch.softmax(scores, dim=-1)
        grad_V = torch.bmm(attn.transpose(1, 2), grad_output)
        grad_attn = torch.bmm(grad_output, V.transpose(1, 2))
        grad_scores = grad_attn * attn
        grad_scores = grad_scores - attn * grad_scores.sum(dim=-1, keepdim=True)
        grad_scores = grad_scores * scale
        grad_Q = torch.bmm(grad_scores, K)
        grad_K = torch.bmm(grad_scores.transpose(1, 2), Q)
        return grad_Q, grad_K, grad_V, None, None, None


class _HyperConv2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, gpu, stride, padding):
        ctx.save_for_backward(X, W)
        ctx.gpu = gpu
        ctx.stride = stride
        ctx.padding = padding
        X_np = X.detach().numpy().astype(np.float32)
        W_np = W.detach().numpy().astype(np.float32)
        O_np = gpu.conv2d(X_np, W_np, stride=stride, padding=padding)
        return torch.from_numpy(O_np.astype(np.float32))

    @staticmethod
    def backward(ctx, grad_output):
        X, W = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding

        grad_X = torch.nn.functional.conv_transpose2d(
            grad_output, W, stride=stride, padding=padding,
        )
        if grad_X.shape != X.shape:
            grad_X = grad_X[:, :, :X.shape[2], :X.shape[3]]

        B, C_out, H_out, W_out = grad_output.shape
        _, C_in, kH, kW = W.shape
        if padding > 0:
            X_padded = torch.nn.functional.pad(X, (padding, padding, padding, padding))
        else:
            X_padded = X

        grad_W = torch.zeros_like(W)
        for b in range(B):
            for i in range(H_out):
                for j in range(W_out):
                    patch = X_padded[b, :, i*stride:i*stride+kH, j*stride:j*stride+kW]
                    for co in range(C_out):
                        grad_W[co] += grad_output[b, co, i, j] * patch

        return grad_X, grad_W, None, None, None


class _HyperConv3d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, gpu, stride, padding):
        ctx.save_for_backward(X, W)
        ctx.gpu = gpu
        ctx.stride = stride
        ctx.padding = padding
        X_np = X.detach().numpy().astype(np.float32)
        W_np = W.detach().numpy().astype(np.float32)
        O_np = gpu.conv3d(X_np, W_np, stride=stride, padding=padding)
        return torch.from_numpy(O_np.astype(np.float32))

    @staticmethod
    def backward(ctx, grad_output):
        X, W = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        grad_X = torch.nn.functional.conv_transpose3d(
            grad_output, W, stride=stride, padding=padding,
        )
        if grad_X.shape != X.shape:
            slices = tuple(slice(0, s) for s in X.shape)
            grad_X = grad_X[slices]

        grad_W = torch.zeros_like(W)
        B = grad_output.shape[0]
        C_out = W.shape[0]
        kD, kH, kW = W.shape[2], W.shape[3], W.shape[4]
        sd, sh, sw = stride

        pd, ph, pw = padding
        if any(p > 0 for p in padding):
            X_padded = torch.nn.functional.pad(X, (pw, pw, ph, ph, pd, pd))
        else:
            X_padded = X

        D_out, H_out, W_out = grad_output.shape[2], grad_output.shape[3], grad_output.shape[4]

        for b in range(B):
            for d in range(D_out):
                for i in range(H_out):
                    for j in range(W_out):
                        patch = X_padded[b, :, d*sd:d*sd+kD, i*sh:i*sh+kH, j*sw:j*sw+kW]
                        for co in range(C_out):
                            grad_W[co] += grad_output[b, co, d, i, j] * patch

        return grad_X, grad_W, None, None, None


class _HyperLayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, gamma, beta, gpu, eps):
        X_np = X.detach().numpy().astype(np.float32)
        g_np = gamma.detach().numpy().astype(np.float32)
        b_np = beta.detach().numpy().astype(np.float32)
        O_np = gpu.layer_norm(X_np, g_np, b_np, eps=eps)
        O = torch.from_numpy(O_np.astype(np.float32))

        mean = X_np.mean(axis=-1, keepdims=True)
        var = X_np.var(axis=-1, keepdims=True)
        X_norm = (X_np - mean) / np.sqrt(var + eps)

        ctx.save_for_backward(X, gamma)
        ctx.x_norm = torch.from_numpy(X_norm.astype(np.float32))
        ctx.std = torch.from_numpy(np.sqrt(var + eps).astype(np.float32))
        ctx.eps = eps
        return O

    @staticmethod
    def backward(ctx, grad_output):
        X, gamma = ctx.saved_tensors
        x_norm = ctx.x_norm
        std = ctx.std
        D = X.shape[-1]

        grad_gamma = (grad_output * x_norm).sum(dim=tuple(range(grad_output.dim() - 1)))
        grad_beta = grad_output.sum(dim=tuple(range(grad_output.dim() - 1)))

        dx_norm = grad_output * gamma
        grad_X = (1.0 / D) * (1.0 / std) * (
            D * dx_norm - dx_norm.sum(dim=-1, keepdim=True)
            - x_norm * (dx_norm * x_norm).sum(dim=-1, keepdim=True)
        )
        return grad_X, grad_gamma, grad_beta, None, None


class _HyperGELU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, gpu):
        ctx.save_for_backward(X)
        X_np = X.detach().numpy().astype(np.float32)
        O_np = gpu.gelu(X_np)
        return torch.from_numpy(O_np.astype(np.float32))

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
        X_np = X.detach().numpy().astype(np.float32)
        O_np = gpu.silu(X_np)
        return torch.from_numpy(O_np.astype(np.float32))

    @staticmethod
    def backward(ctx, grad_output):
        X, = ctx.saved_tensors
        sig = torch.sigmoid(X)
        grad_X = sig * (1.0 + X * (1.0 - sig))
        return grad_output * grad_X, None


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
    ):
        self.gpu = HyperGPU(
            lanes=lanes,
            tensor_cores=tensor_cores,
            precision=precision,
            vram_capacity=vram_capacity,
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
        A_np = A.detach().numpy().astype(np.float32)
        B_np = B.detach().numpy().astype(np.float32)

        chunks = self.cluster.scatter_data(A_np, num_nodes)

        def _gemm_fn(gpu, chunk):
            return gpu.gemm(chunk, B_np)

        results = self.cluster.run_distributed(_gemm_fn, chunks)
        combined = self.cluster.gather_results(results)
        return torch.from_numpy(combined.astype(np.float32))

    def all_reduce_gradients(self, gradients: List[torch.Tensor]) -> torch.Tensor:
        np_grads = [g.detach().numpy() for g in gradients]
        avg = self.cluster.all_reduce_gradients("param", np_grads)
        return torch.from_numpy(avg.astype(np.float32))

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


from typing import Dict
