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
        O = torch.from_numpy(O_np.astype(np.float32))  # noqa: E741
        ctx.save_for_backward(Q, K, V)
        ctx.gpu = gpu
        ctx.causal = causal
        return O

    @staticmethod
    def backward(ctx, grad_output):
        Q, K, V = ctx.saved_tensors
        gpu = ctx.gpu
        causal = ctx.causal
        B, T, D = Q.shape
        scale = 1.0 / (D ** 0.5)

        # Route all compute through the digital GPU to keep the full
        # forward+backward execution on the custom silicon path.
        # DigitalGPU.gemm expects 2D (M, K) × (K, N), so we loop over the
        # batch dimension; softmax handles any shape via h_softmax/np_like.
        Q_np = Q.detach().numpy().astype(np.float64)   # (B, T, D)
        K_np = K.detach().numpy().astype(np.float64)   # (B, T, D)
        V_np = V.detach().numpy().astype(np.float64)   # (B, T, D)
        grad_np = grad_output.numpy().astype(np.float64)  # (B, T, D)

        grad_Q_np = np.empty_like(Q_np)
        grad_K_np = np.empty_like(K_np)
        grad_V_np = np.empty_like(V_np)

        causal_mask = np.triu(np.full((T, T), -1e9), k=1) if causal else None

        for b in range(B):
            # scores (T, T) = Q[b] @ K[b]^T × scale — digital GPU 2-D GEMM
            scores = gpu.gemm(Q_np[b], K_np[b].T) * scale
            if causal_mask is not None:
                scores = scores + causal_mask
            # attn (T, T) via digital GPU softmax (shape-agnostic)
            attn = gpu.softmax(scores, axis=-1)

            # grad_V[b] = attn^T @ grad[b]
            grad_V_np[b] = gpu.gemm(attn.T, grad_np[b])

            # grad_attn (T, T) = grad[b] @ V[b]^T
            grad_attn = gpu.gemm(grad_np[b], V_np[b].T)

            # Softmax Jacobian: dL/d_scores = (dL/d_attn - sum) * attn
            grad_scores = grad_attn * attn
            grad_scores = grad_scores - attn * grad_scores.sum(axis=-1, keepdims=True)
            grad_scores = grad_scores * scale

            # grad_Q[b] = grad_scores @ K[b], grad_K[b] = grad_scores^T @ Q[b]
            grad_Q_np[b] = gpu.gemm(grad_scores, K_np[b])
            grad_K_np[b] = gpu.gemm(grad_scores.T, Q_np[b])

        return (
            torch.from_numpy(grad_Q_np.astype(np.float32)),
            torch.from_numpy(grad_K_np.astype(np.float32)),
            torch.from_numpy(grad_V_np.astype(np.float32)),
            None,
            None,
        )


class _DigitalGEMMBiasReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, bias, gpu):
        A_np = A.detach().numpy().astype(np.float64)
        B_np = B.detach().numpy().astype(np.float64)
        bias_np = bias.detach().numpy().astype(np.float64)
        O_np = gpu.gemm_bias_relu(A_np, B_np, bias_np)
        O = torch.from_numpy(O_np.astype(np.float32))  # noqa: E741
        ctx.save_for_backward(A, B, bias, O)
        ctx.gpu = gpu
        return O

    @staticmethod
    def backward(ctx, grad_output):
        A, B, bias, O = ctx.saved_tensors  # noqa: E741
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
    # Real numpy kernels this backend can dispatch by name via run_kernel().
    # Each maps to an actual DigitalGPU op that computes on numpy. There is
    # deliberately NO fp8 / tensor-core / sm_102 kernel here — those need real
    # silicon, so requesting one raises instead of masquerading plain numpy
    # under a hardware-implying name.
    _KERNELS = ("gemm", "add", "softmax", "attention", "gemm_bias_relu")

    def __init__(self, lanes=32, silicon=None):
        self.gpu = DigitalGPU(lanes=lanes, silicon=silicon)
        self._profile_history = []

    # ── raw allocation (byte-sized handles, like a device allocator) ──────────
    def alloc(self, size: int) -> int:
        """Allocate ``size`` bytes of digital VRAM and return an integer handle.

        This is a *real* allocation — a numpy byte buffer tracked in VRAM — not a
        simulated counter. Free it with ``free(handle)``. (For typed data, prefer
        ``from_tensor`` which stores the array directly; this exists so a
        device-style caller that thinks in byte sizes and handles has a working
        allocator.)
        """
        # Strict integer bytes — never silently truncate a fractional size.
        if isinstance(size, bool):
            raise TypeError("alloc size must be an int number of bytes, not bool")
        if isinstance(size, float):
            if not size.is_integer():
                raise ValueError(
                    f"alloc size must be a whole number of bytes, got {size}")
            size = int(size)
        if not isinstance(size, (int, np.integer)):
            raise TypeError(
                f"alloc size must be an int, got {type(size).__name__}")
        n = int(size)
        if n < 0:
            raise ValueError(f"alloc size must be non-negative, got {size}")
        return self.gpu.vram.alloc(np.zeros(n, dtype=np.uint8))

    def free(self, handle: int) -> None:
        """Free a handle previously returned by ``alloc`` (or any VRAM handle)."""
        self.gpu.vram.free(handle)

    # ── tensor ↔ handle boundary (tinygrad Buffer.fromCPU/toCPU pattern) ─────

    def from_tensor(self, arr: np.ndarray) -> int:
        """Store a numpy array in VRAM and return an integer handle.

        This is the entry-point from the *tensor world* into the *handle world*:
        the returned handle can be passed to ``run_kernel_h`` or retrieved later
        with ``to_tensor``. The array is stored by reference (not copied) unless
        dtype conversion is needed — call ``arr.copy()`` first if you need
        isolation.

        Example full handle pipeline::

            h_A = backend.from_tensor(A)
            h_B = backend.from_tensor(B)
            h_C = backend.run_kernel_h("gemm", [h_A, h_B])
            C   = backend.to_tensor(h_C)
            backend.free(h_A); backend.free(h_B); backend.free(h_C)
        """
        return self.gpu.vram.alloc(np.asarray(arr))

    def to_tensor(self, handle: int) -> np.ndarray:
        """Retrieve the numpy array stored at ``handle``.

        Returns a *live view* into the VRAM store — not a copy. Mutating the
        returned array mutates the stored buffer. Call ``.copy()`` for an
        independent array.
        """
        return self.gpu.vram.get(handle)

    def run_kernel_h(self, name: str, input_handles: list, **kwargs) -> int:
        """Handle-based kernel dispatch — the unified interface that composes
        with ``alloc`` / ``from_tensor`` / ``free``.

        Reads each input array from VRAM by handle, runs the named kernel, stores
        the result in a *new* VRAM handle, and returns that handle. The caller is
        responsible for freeing input handles (and the returned output handle)
        when they are no longer needed.

        This closes the interface gap where ``alloc()`` returns handles but
        ``run_kernel()`` requires tensor arrays: both sides now speak handles.
        The existing ``run_kernel()`` method (takes and returns arrays directly)
        is preserved for callers that already have arrays.

        Only kernels in ``_KERNELS`` are accepted — same safeguard as
        ``run_kernel()``; hardware-implying names raise rather than masquerade.

        Args:
            name: kernel name (e.g. ``"gemm"``, ``"attention"``).
            input_handles: list of integer VRAM handles, in the order the kernel
                expects its positional arguments (A, B for gemm; Q, K, V for
                attention; etc.).
            **kwargs: forwarded verbatim to the underlying kernel (e.g.
                ``causal=True`` for attention).

        Returns:
            Integer VRAM handle for the output array.

        Raises:
            ValueError: unknown kernel name or handle not in VRAM.
        """
        key = str(name).lower()
        if key not in self._KERNELS:
            raise ValueError(
                f"digital GPU has no kernel named {name!r}. Supported: "
                f"{list(self._KERNELS)}. Hardware-implying names (fp8, sm102, "
                f"tensor-core variants) require real silicon — use the numpy "
                f"equivalent here, or a real CUDA backend."
            )
        input_arrays = [self.gpu.vram.get(h) for h in input_handles]
        result = getattr(self.gpu, key)(*input_arrays, **kwargs)
        return self.gpu.vram.alloc(np.asarray(result))

    # ── named-kernel dispatch to the real numpy ops ──────────────────────────
    def run_kernel(self, name: str, *args, **kwargs):
        """Dispatch a named kernel to its real numpy implementation on the
        digital GPU and return the actual result array.

        Only genuine ops are supported (see ``_KERNELS``); e.g.
        ``run_kernel("gemm", A, B)`` or
        ``run_kernel("attention", Q, K, V, causal=True)``. A name implying
        hardware this backend doesn't have (``*_fp8``, ``*_sm102``, tensor-core
        variants, ...) raises a clear error rather than quietly running plain
        numpy and reporting it as an FP8/tensor-core kernel.
        """
        key = str(name).lower()
        if key not in self._KERNELS:
            raise ValueError(
                f"digital GPU has no kernel named {name!r}. Supported numpy "
                f"kernels: {list(self._KERNELS)}. There is no FP8 / tensor-core "
                f"/ sm_102 kernel on this backend — that requires real silicon; "
                f"use the numpy op (e.g. 'attention', 'gemm') here, or the torch "
                f"GPUBackend on an actual CUDA host."
            )
        return getattr(self.gpu, key)(*args, **kwargs)

    def silicon_report(self):
        """Estimated (NOT measured) cycle/time budget from the attached silicon
        performance model, or None."""
        return self.gpu.silicon_report()

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
        # Snapshot: .copy() is atomic under the GIL; iterating the live dict
        # could race a concurrent alloc/free.
        store = self.gpu.vram._store.copy()
        vram_count = len(store)
        vram_bytes = sum(a.nbytes for a in store.values())
        return {
            "lanes": self.gpu.core.lanes,
            "tile_size": f"{self.gpu.core.tile_m}x{self.gpu.core.tile_n}x{self.gpu.core.tile_k}",
            "vram_handles": vram_count,
            "vram_bytes": vram_bytes,
            "vram_mb": round(vram_bytes / (1024 * 1024), 2),
        }
