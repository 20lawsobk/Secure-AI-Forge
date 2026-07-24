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

import hashlib
import math
import threading
import time
import zlib
import pickle
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
from typing import Optional

from ai_model.model.transformer import (
    KVCache,
    precompute_rope_freqs,
    apply_rope,
    apply_rope_offset,
)

import logging as _logging
_hct_logger = _logging.getLogger("hyper_creative_transformer")

# ── Digital-GPU attention GEMM counter ───────────────────────────────────────
# Incremented each time q@k or attn@v successfully runs through gpu.matmul.
# Tests can read this to verify the Digital GPU path is taken.
_gpu_attn_calls: int = 0


def get_gpu_attn_calls() -> int:
    """Return the number of successful Digital GPU attention GEMM calls."""
    return _gpu_attn_calls


# ── Prefix KV cache ───────────────────────────────────────────────────────────
# Stores the KV state + hidden state at the end of a prompt prefix so that
# subsequent requests that share the same prefix skip prefill entirely for
# those tokens.  The payload is zlib-compressed to keep memory footprint small.
#
# Key: SHA-256 of the token IDs up to min(T, _PREFIX_CACHE_LEN)
# Value: {"h": tensor, "kv": list[(k,v)], "ts": float, "prefix_len": int}
#
# Thread-safe: writes use _PREFIX_KV_LOCK; reads are checked under the lock too
# because dict.get() is only GIL-safe for CPython — this is safer cross-version.

_PREFIX_CACHE_LEN   = 256    # max tokens hashed as the "prefix"
_PREFIX_CACHE_MAX   = 32     # max entries (KV states are large)
_PREFIX_CACHE_TTL   = 600.0  # seconds
_PREFIX_KV_CACHE: dict[str, bytes] = {}   # key → zlib-pickled payload
_PREFIX_KV_LOCK  = threading.Lock()
_PREFIX_KV_STATS = {"hits": 0, "misses": 0, "evictions": 0}


def _prefix_key(ids: torch.Tensor) -> str:
    prefix = ids[0, :_PREFIX_CACHE_LEN].tolist()
    return hashlib.sha256(str(prefix).encode()).hexdigest()


def _prefix_get(key: str) -> Optional[dict]:
    with _PREFIX_KV_LOCK:
        raw = _PREFIX_KV_CACHE.get(key)
        if raw is None:
            _PREFIX_KV_STATS["misses"] += 1
            return None
        payload = pickle.loads(zlib.decompress(raw))
        if time.monotonic() - payload["ts"] > _PREFIX_CACHE_TTL:
            del _PREFIX_KV_CACHE[key]
            _PREFIX_KV_STATS["misses"] += 1
            return None
        _PREFIX_KV_STATS["hits"] += 1
        return payload


def _prefix_put(key: str, h: torch.Tensor,
                kv: list[tuple[torch.Tensor, torch.Tensor]],
                prefix_len: int) -> None:
    payload = {
        "h": h.detach().cpu(),
        "kv": [(k.detach().cpu(), v.detach().cpu()) for k, v in kv],
        "ts": time.monotonic(),
        "prefix_len": prefix_len,
    }
    raw = zlib.compress(pickle.dumps(payload, protocol=4), level=1)
    with _PREFIX_KV_LOCK:
        # Evict oldest entry if at capacity
        if len(_PREFIX_KV_CACHE) >= _PREFIX_CACHE_MAX:
            oldest = next(iter(_PREFIX_KV_CACHE))
            del _PREFIX_KV_CACHE[oldest]
            _PREFIX_KV_STATS["evictions"] += 1
        _PREFIX_KV_CACHE[key] = raw


def get_prefix_kv_stats() -> dict:
    with _PREFIX_KV_LOCK:
        n = len(_PREFIX_KV_CACHE)
        s = dict(_PREFIX_KV_STATS)
    total = s["hits"] + s["misses"]
    s["cache_size"] = n
    s["hit_rate"] = round(s["hits"] / total, 4) if total else 0.0
    s["max_entries"] = _PREFIX_CACHE_MAX
    s["ttl_seconds"] = _PREFIX_CACHE_TTL
    return s


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

        # ── Scores: q @ k.T  — routed through Digital GPU ───────────────
        # gpu.matmul expects 2D inputs; reshape BH×T×D_h to (BH*T)×D_h then
        # back to BH×T×T.  Fall back to plain torch if gpu.matmul raises.
        global _gpu_attn_calls
        BH, T2, D_h = q.shape
        try:
            import numpy as _np
            _q_np = q.reshape(BH * T2, D_h).detach().float().numpy()
            _kt_np = k.transpose(-2, -1).contiguous().reshape(D_h, BH * T2).detach().float().numpy()
            # matmul: (BH*T, D_h) @ (D_h, BH*T) won't work — need per-head batching.
            # Use gemm on the 2D slices: for each head, (T, D_h) @ (D_h, T)
            _scores_list = []
            for _h in range(BH):
                _qh = q[_h].detach().float().numpy()          # (T, D_h)
                _kh = k[_h].transpose(0, 1).detach().float().numpy()  # (D_h, T)
                _s = self.gpu.gemm(_qh, _kh)                  # (T, T)
                _scores_list.append(torch.from_numpy(_s))
            attn = torch.stack(_scores_list, dim=0) * self.scale  # (BH, T, T)
            _gpu_attn_calls += 1
        except Exception as _e:
            _hct_logger.warning("[gpu-attn] forward_with_kv scores fallback: %s", _e)
            attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn = attn + mask.unsqueeze(0).unsqueeze(0)
        if key_padding_mask is not None:
            attn = attn.masked_fill(key_padding_mask[:, None, None, :], -1e9)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # ── Context: attn @ v  — routed through Digital GPU ─────────────
        try:
            _ctx_list = []
            for _h in range(BH):
                _ah = attn[_h].detach().float().numpy()   # (T, T)
                _vh = v[_h].detach().float().numpy()       # (T, D_h)
                _c = self.gpu.gemm(_ah, _vh)              # (T, D_h)
                _ctx_list.append(torch.from_numpy(_c))
            out = torch.stack(_ctx_list, dim=0).to(q.dtype)  # (BH, T, D_h)
            _gpu_attn_calls += 1
        except Exception as _e:
            _hct_logger.warning("[gpu-attn] forward_with_kv context fallback: %s", _e)
            out = attn @ v

        out = out.transpose(1, 2).contiguous().reshape(B, T, C)
        return self.out(out), k, v

    def decode_one(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                   past_k: torch.Tensor, past_v: torch.Tensor,
                   key_padding_mask: torch.Tensor | None = None,
                   _kv_buf: Optional[tuple] = None,
                   _write_pos: Optional[int] = None,
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

        if _kv_buf is not None and _write_pos is not None:
            # In-place write into the pre-allocated KVCache buffers — same
            # optimisation as TransformerLM: no torch.cat allocation.
            k_buf, v_buf = _kv_buf
            k_buf[:, :, _write_pos:_write_pos + 1, :] = k
            v_buf[:, :, _write_pos:_write_pos + 1, :] = v
            k_cat = k_buf[:, :, :_write_pos + 1, :]
            v_cat = v_buf[:, :, :_write_pos + 1, :]
        else:
            k_cat = torch.cat([past_k, k], dim=2)
            v_cat = torch.cat([past_v, v], dim=2)

        # ── Scores: q @ k_cat.T  — routed through Digital GPU ───────────
        global _gpu_attn_calls
        BH_d = q.shape[0]   # B * n_heads
        T_kv = k_cat.shape[2]
        try:
            _scores_list = []
            for _h in range(BH_d):
                _qh = q[_h].detach().float().numpy()                        # (1, D_h)
                _kh = k_cat[_h].transpose(0, 1).detach().float().numpy()   # (D_h, T_kv)
                _s = self.gpu.gemm(_qh, _kh)                                # (1, T_kv)
                _scores_list.append(torch.from_numpy(_s))
            attn = torch.stack(_scores_list, dim=0) * self.scale  # (BH, 1, T_kv)
            _gpu_attn_calls += 1
        except Exception as _e:
            _hct_logger.warning("[gpu-attn] decode_one scores fallback: %s", _e)
            attn = (q @ k_cat.transpose(-2, -1)) * self.scale

        if key_padding_mask is not None:
            attn = attn.masked_fill(key_padding_mask[:, None, None, :], -1e9)
        attn = F.softmax(attn, dim=-1)

        # ── Context: attn @ v_cat  — routed through Digital GPU ─────────
        try:
            _ctx_list = []
            for _h in range(BH_d):
                _ah = attn[_h].detach().float().numpy()      # (1, T_kv)
                _vh = v_cat[_h].detach().float().numpy()     # (T_kv, D_h)
                _c = self.gpu.gemm(_ah, _vh)                 # (1, D_h)
                _ctx_list.append(torch.from_numpy(_c))
            out = torch.stack(_ctx_list, dim=0).to(q.dtype)  # (BH, 1, D_h)
            _gpu_attn_calls += 1
        except Exception as _e:
            _hct_logger.warning("[gpu-attn] decode_one context fallback: %s", _e)
            out = attn @ v_cat

        out = out.transpose(1, 2).contiguous().reshape(B, T, C)
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

    def decode_one(self, x, cos, sin, past_k, past_v, key_padding_mask=None,
                   _kv_buf=None, _write_pos=None):
        attn_out, new_k, new_v = self.attn.decode_one(
            self.ln1(x), cos, sin, past_k, past_v, key_padding_mask,
            _kv_buf=_kv_buf, _write_pos=_write_pos)
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

    def warmup_pocket(self) -> dict:
        """Pre-register every weight matrix in the pocket accelerator.

        Walks all ``HyperLinearNL`` parameters and issues one synthetic GEMM per
        weight so the per-array digest is cached and the pocket accelerator's
        adaptive gate starts with a warmup credit.  After this call, the very
        first real forward pass sees pocket hits for all weight GEMMs — not just
        after the second request.

        Returns a summary dict: {"layers_warmed": int, "weight_matrices": int}.
        """
        self.eval()
        layers_warmed = 0
        weight_count  = 0
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, HyperLinearNL):
                    W = module.weight.detach().float()           # (out, in)
                    x_syn = torch.zeros(1, module.in_features)  # [1, in]
                    try:
                        _MixedPrecisionGEMM.apply(
                            x_syn.contiguous(), W.t().contiguous(), self.gpu)
                        weight_count += 1
                    except Exception:
                        pass
                    layers_warmed += 1
        return {"layers_warmed": layers_warmed, "weight_matrices": weight_count}

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

        # ── Prefix KV cache fast-path ─────────────────────────────────────
        # If the first min(T, _PREFIX_CACHE_LEN) tokens match a cached entry,
        # restore the KV state and hidden activations from that checkpoint and
        # only run the forward pass on the remaining suffix tokens.  In eval
        # mode there is no dropout, so the restored activations are exact.
        if not self.training and T > 1:
            pk = _prefix_key(x)
            cached = _prefix_get(pk)
            if cached is not None and cached["prefix_len"] <= T:
                p_len = cached["prefix_len"]
                h_pfx = cached["h"].to(x.device)   # [B, p_len, dim]
                kv_pfx = [(k.to(x.device), v.to(x.device))
                           for k, v in cached["kv"]]
                if p_len == T:
                    # Full prompt already cached — return immediately
                    h_out = self.ln_final(h_pfx)
                    return self._head(h_out), kv_pfx

                # Partial prefix — compute suffix only
                suffix = x[:, p_len:]                   # [B, T-p_len]
                sfx_len = suffix.shape[1]
                h_sfx = self.emb_dropout(self.token_emb(suffix))
                # Build extended h and causal mask for the suffix positions
                cos_sfx = self.rope_cos[p_len:p_len + sfx_len]
                sin_sfx = self.rope_sin[p_len:p_len + sfx_len]
                mask_sfx = self.causal_mask[p_len:p_len + sfx_len, :p_len + sfx_len]
                kv_full: list[tuple[torch.Tensor, torch.Tensor]] = []
                h = h_sfx
                for i, layer in enumerate(self.layers):
                    past_k, past_v = kv_pfx[i]           # [B, H, p_len, D_h]
                    # Run forward_with_kv on suffix; KV cache for suffix only
                    h_out_layer, new_k, new_v = layer.forward_with_kv(
                        h, cos_sfx, sin_sfx, mask_sfx, key_padding_mask)
                    # Concatenate prefix and suffix KV along the time axis
                    full_k = torch.cat([past_k, new_k], dim=2)
                    full_v = torch.cat([past_v, new_v], dim=2)
                    kv_full.append((full_k, full_v))
                    h = h_out_layer
                # Merge prefix activations with suffix activations
                h_merged = torch.cat([h_pfx, h], dim=1)  # [B, T, dim]
                h_final = self.ln_final(h_merged[:, -1:, :])
                # Cache the full result as a new prefix entry
                _prefix_put(pk, h_merged, kv_full, T)
                return self._head(h_final), kv_full

        # ── Full prefill (no cache hit) ───────────────────────────────────
        h = self.emb_dropout(self.token_emb(x))
        mask = self.causal_mask[:T, :T]
        cos = self.rope_cos[:T]
        sin = self.rope_sin[:T]
        kv_cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer in self.layers:
            h, k, v = layer.forward_with_kv(h, cos, sin, mask, key_padding_mask)
            kv_cache.append((k, v))
        h_out = self.ln_final(h)
        logits = self._head(h_out)
        # Store this prompt as a prefix for future requests
        if not self.training and T >= 4:
            pk = _prefix_key(x)
            _prefix_put(pk, h_out, kv_cache, min(T, _PREFIX_CACHE_LEN))
        return logits, kv_cache

    def decode_one(self, x_new: torch.Tensor,
                   kv_cache,
                   key_padding_mask: torch.Tensor | None = None,
                   ) -> tuple[torch.Tensor, object]:
        h = self.token_emb(x_new)

        if isinstance(kv_cache, KVCache):
            # Fast path: in-place KVCache — eliminates torch.cat per layer.
            write_pos = kv_cache._len
            for i, layer in enumerate(self.layers):
                past_k, past_v = kv_cache[i]
                h, _, _ = layer.decode_one(
                    h, self.rope_cos, self.rope_sin, past_k, past_v,
                    key_padding_mask,
                    _kv_buf=(kv_cache._k[i], kv_cache._v[i]),
                    _write_pos=write_pos,
                )
            kv_cache._len += 1
            h = self.ln_final(h)
            return self._head(h), kv_cache

        # Original list-of-tuples path — backward compatible.
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
