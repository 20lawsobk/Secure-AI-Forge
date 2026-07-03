"""Digital-GPU-routed creative transformer.

This mirrors the architecture of ``ai_model.model.transformer.TransformerLM``
(RoPE + SwiGLU + pre-norm + weight-tied head + KV-cache), but every heavy compute
op — linear projections, attention, layer-norm, SiLU — is routed through the
in-house Digital GPU (HyperGPU) autograd functions in ``hyper_backend``.

One intentional deviation: because the fused Digital-GPU flash-attention kernel
computes softmax internally, attention dropout is applied to the attention
*output* here rather than to the attention-probability matrix as in
``TransformerLM``. All other layers match exactly. This does not affect weight
compatibility (dropout has no parameters).

The point: the training run's real forward+backward compute genuinely executes on
the Digital GPU backend (``training_mode=False`` → true NumPy tensor-core kernels
with hand-written autograd), NOT plain PyTorch ops.

Weight compatibility: parameter names and shapes are byte-for-byte identical to
``TransformerLM``, so a model trained here can be transferred into the fast
KV-cache ``TransformerLM`` for production serving via ``load_state_dict``.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ai_model.gpu.hyper_core import HyperGPU, PrecisionMode
from ai_model.gpu.hyper_backend import (
    _MixedPrecisionGEMM,
    _FlashAttention,
    _HyperLayerNorm,
    _HyperSiLU,
)
from ai_model.model.transformer import (
    precompute_rope_freqs,
    apply_rope,
    apply_rope_offset,
)


# ─── nn.Linear-compatible linear routed through the Digital GPU ────────────────

class HyperLinearNL(nn.Module):
    """Linear layer with ``nn.Linear`` weight convention ``(out_features, in_features)``
    (so state-dict transfers 1:1 to ``nn.Linear``), but the matmul + its backward
    run on the Digital GPU via ``_MixedPrecisionGEMM``.
    """

    def __init__(self, in_features: int, out_features: int, gpu: HyperGPU,
                 bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gpu = gpu
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x2d = x.reshape(-1, self.in_features) if x.dim() > 2 else x
        # weight is (out, in); GEMM computes A @ B, so pass weight.t() -> (in, out)
        out = _MixedPrecisionGEMM.apply(x2d, self.weight.t().contiguous(), self.gpu)
        if self.bias is not None:
            out = out + self.bias
        if len(shape) > 2:
            out = out.reshape(*shape[:-1], self.out_features)
        return out


class HyperLN(nn.Module):
    """LayerNorm routed through the Digital GPU. Params named ``weight``/``bias``
    to match ``nn.LayerNorm`` for state-dict transfer."""

    def __init__(self, dim: int, gpu: HyperGPU, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.gpu = gpu
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _HyperLayerNorm.apply(x, self.weight, self.bias, self.gpu, self.eps)


# ─── RoPE self-attention, Digital-GPU routed ──────────────────────────────────

class HyperRoPESelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, gpu: HyperGPU,
                 dropout: float = 0.1, block_size: int = 64):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.gpu = gpu
        self.block_size = block_size

        self.qkv = HyperLinearNL(dim, 3 * dim, gpu, bias=False)
        self.out = HyperLinearNL(dim, dim, gpu, bias=False)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        """Training / causal forward — attention math runs on the Digital GPU."""
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # each [B, T, H, D_h]

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # -> [B*H, T, D_h] for the flash-attention kernel (causal only)
        q = q.permute(0, 2, 1, 3).contiguous().view(B * self.n_heads, T, self.head_dim)
        k = k.permute(0, 2, 1, 3).contiguous().view(B * self.n_heads, T, self.head_dim)
        v = v.permute(0, 2, 1, 3).contiguous().view(B * self.n_heads, T, self.head_dim)

        out = _FlashAttention.apply(q, k, v, self.gpu, True, self.block_size)
        out = out.view(B, self.n_heads, T, self.head_dim).permute(0, 2, 1, 3).contiguous().view(B, T, C)
        # NOTE (intentional deviation from TransformerLM): the fused Digital-GPU
        # flash kernel computes softmax internally, so attention-matrix dropout
        # cannot be injected there. We instead apply dropout to the attention
        # output to preserve attention-path regularization strength during
        # Digital-GPU training. Inference (KV-cache) runs under eval() so dropout
        # is a no-op regardless.
        out = self.attn_drop(out)
        return self.out(out)

    # ── Inference paths (KV-cache). No grad needed; attention math in torch, but
    #    the linear projections still route through the Digital GPU. ────────────

    def forward_with_kv(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                        mask: torch.Tensor | None = None,
                        key_padding_mask: torch.Tensor | None = None,
                        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn + mask.unsqueeze(0).unsqueeze(0)
        if key_padding_mask is not None:
            attn = attn.masked_fill(key_padding_mask[:, None, None, :], -1e9)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).contiguous().reshape(B, T, C)
        return self.out(out), k, v

    def decode_one(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                   past_k: torch.Tensor, past_v: torch.Tensor,
                   key_padding_mask: torch.Tensor | None = None,
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        offset = past_k.shape[2]
        B, T, C = x.shape  # T == 1

        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)

        q = apply_rope_offset(q, cos, sin, offset)
        k = apply_rope_offset(k, cos, sin, offset)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        k_cat = torch.cat([past_k, k], dim=2)
        v_cat = torch.cat([past_v, v], dim=2)

        attn = (q @ k_cat.transpose(-2, -1)) * self.scale
        if key_padding_mask is not None:
            attn = attn.masked_fill(key_padding_mask[:, None, None, :], -1e9)
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v_cat).transpose(1, 2).contiguous().reshape(B, T, C)
        return self.out(out), k_cat, v_cat


# ─── SwiGLU FFN, Digital-GPU routed ───────────────────────────────────────────

class HyperSwiGLUFFN(nn.Module):
    def __init__(self, dim: int, gpu: HyperGPU, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden = int(dim * expansion * 2 / 3)
        hidden = ((hidden + 63) // 64) * 64  # round to multiple of 64
        self.gpu = gpu
        self.gate = HyperLinearNL(dim, hidden * 2, gpu, bias=False)
        self.down = HyperLinearNL(hidden, dim, gpu, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g, v = self.gate(x).chunk(2, dim=-1)
        act = _HyperSiLU.apply(g, self.gpu)
        return self.down(self.drop(act * v))


# ─── Decoder layer ────────────────────────────────────────────────────────────

class HyperTransformerDecoderLayer(nn.Module):
    def __init__(self, dim: int, n_heads: int, gpu: HyperGPU, dropout: float = 0.1):
        super().__init__()
        self.ln1 = HyperLN(dim, gpu)
        self.attn = HyperRoPESelfAttention(dim, n_heads, gpu, dropout)
        self.ln2 = HyperLN(dim, gpu)
        self.ffn = HyperSwiGLUFFN(dim, gpu, dropout=dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, cos, sin, mask=None):
        x = x + self.drop(self.attn(self.ln1(x), cos, sin, mask))
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x

    def forward_with_kv(self, x, cos, sin, mask=None, key_padding_mask=None):
        attn_out, k, v = self.attn.forward_with_kv(self.ln1(x), cos, sin, mask, key_padding_mask)
        x = x + self.drop(attn_out)
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x, k, v

    def decode_one(self, x, cos, sin, past_k, past_v, key_padding_mask=None):
        attn_out, new_k, new_v = self.attn.decode_one(
            self.ln1(x), cos, sin, past_k, past_v, key_padding_mask)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, new_k, new_v


# ─── Full model ───────────────────────────────────────────────────────────────

class HyperCreativeTransformerLM(nn.Module):
    """RoPE + SwiGLU decoder-only LM whose compute runs on the Digital GPU.

    Parameter names/shapes match ``TransformerLM`` exactly, so a trained
    checkpoint transfers into the fast KV-cache serving model.
    """

    def __init__(self, vocab_size: int, dim: int = 512, n_layers: int = 8,
                 n_heads: int = 8, max_len: int = 1024, dropout: float = 0.1,
                 gpu: HyperGPU | None = None):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.gpu = gpu if gpu is not None else HyperGPU(
            lanes=512, tensor_cores=8, precision=PrecisionMode.MIXED)

        self.token_emb = nn.Embedding(vocab_size, dim)
        self.emb_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            HyperTransformerDecoderLayer(dim, n_heads, self.gpu, dropout)
            for _ in range(n_layers)
        ])
        self.ln_final = HyperLN(dim, self.gpu)

        # Weight-tied output head (shares token_emb.weight, exactly like TransformerLM).
        rope_cos, rope_sin = precompute_rope_freqs(dim // n_heads, max_len)
        self.register_buffer("rope_cos", rope_cos)
        self.register_buffer("rope_sin", rope_sin)
        causal_mask = torch.triu(torch.full((max_len, max_len), float('-inf')), diagonal=1)
        self.register_buffer("causal_mask", causal_mask)

        self._init_weights(n_layers)

    def _init_weights(self, n_layers: int):
        std = 0.02
        residual_std = std / math.sqrt(2 * n_layers)
        for name, param in self.named_parameters():
            if param.dim() >= 2:
                if "out.weight" in name or "down.weight" in name:
                    nn.init.normal_(param, mean=0.0, std=residual_std)
                else:
                    nn.init.normal_(param, mean=0.0, std=std)
            elif "bias" in name:
                nn.init.zeros_(param)
            elif param.dim() == 1 and ("ln" in name or "ln_final" in name) and "weight" in name:
                nn.init.ones_(param)
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=std)

    def _head(self, h: torch.Tensor) -> torch.Tensor:
        """Tied output projection routed through the Digital GPU."""
        shape = h.shape
        h2d = h.reshape(-1, self.dim)
        logits = _MixedPrecisionGEMM.apply(h2d, self.token_emb.weight.t().contiguous(), self.gpu)
        return logits.reshape(*shape[:-1], self.token_emb.num_embeddings)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        assert T <= self.max_len, f"Sequence length {T} exceeds max_len {self.max_len}"
        h = self.emb_dropout(self.token_emb(x))
        mask = self.causal_mask[:T, :T]
        cos = self.rope_cos[:T]
        sin = self.rope_sin[:T]
        for layer in self.layers:
            h = layer(h, cos, sin, mask)
        h = self.ln_final(h)
        return self._head(h)

    def prefill(self, x: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None,
                ) -> tuple[torch.Tensor, list]:
        B, T = x.shape
        h = self.emb_dropout(self.token_emb(x))
        mask = self.causal_mask[:T, :T]
        cos = self.rope_cos[:T]
        sin = self.rope_sin[:T]
        kv_cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer in self.layers:
            h, k, v = layer.forward_with_kv(h, cos, sin, mask, key_padding_mask)
            kv_cache.append((k, v))
        h = self.ln_final(h)
        return self._head(h), kv_cache

    def decode_one(self, x_new: torch.Tensor,
                   kv_cache: list[tuple[torch.Tensor, torch.Tensor]],
                   key_padding_mask: torch.Tensor | None = None,
                   ) -> tuple[torch.Tensor, list]:
        h = self.token_emb(x_new)
        new_cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i, layer in enumerate(self.layers):
            past_k, past_v = kv_cache[i]
            h, new_k, new_v = layer.decode_one(
                h, self.rope_cos, self.rope_sin, past_k, past_v, key_padding_mask)
            new_cache.append((new_k, new_v))
        h = self.ln_final(h)
        return self._head(h), new_cache

    @property
    def pos_emb(self):
        max_len = self.max_len

        class _FakeEmb:
            num_embeddings = max_len
        return _FakeEmb()
