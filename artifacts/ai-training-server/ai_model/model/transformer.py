import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

_log = logging.getLogger(__name__)

# ─── Digital GPU — module-level singleton (never-raise) ───────────────────────
#
# Every linear projection and layer-norm in this file routes through the in-house
# HyperGPU kernel stack when it is available.  If the backend fails to import or
# initialise for any reason the code transparently falls back to standard
# torch.nn.Linear / nn.LayerNorm so generation never breaks.
#
# The global is lazily resolved on first use to avoid circular imports during
# server start-up.

_hyper_gpu = None      # HyperGPU instance, or None
_gpu_ready  = False    # True once the backend initialised successfully
_gpu_init_attempted = False


def _ensure_gpu() -> bool:
    """Initialise the HyperGPU singleton once; return whether it is available."""
    global _hyper_gpu, _gpu_ready, _gpu_init_attempted
    if _gpu_init_attempted:
        return _gpu_ready
    _gpu_init_attempted = True
    try:
        from ai_model.gpu.hyper_core import HyperGPU, PrecisionMode
        _hyper_gpu = HyperGPU(lanes=512, tensor_cores=8,
                               precision=PrecisionMode.MIXED)
        _gpu_ready = True
        _log.info("transformer: DigitalGPU backend active (HyperGPU tensor-core kernels)")
    except Exception as exc:
        _log.warning("transformer: DigitalGPU unavailable – falling back to torch "
                     "(reason: %s)", exc)
        _gpu_ready = False
    return _gpu_ready


def _make_linear(in_features: int, out_features: int,
                 bias: bool = False) -> nn.Module:
    """Return a HyperLinearNL (Digital GPU) if available, else nn.Linear."""
    if _ensure_gpu():
        try:
            from ai_model.gpu.hyper_creative_transformer import HyperLinearNL
            from ai_model.maxcore.observability import METRICS
            lin = HyperLinearNL(in_features, out_features, _hyper_gpu, bias=bias)
            # Wrap forward to count ops in the shared METRICS registry so that
            # probe_gpu_routing() can verify routing via snapshot() delta.
            _orig_fwd = lin.forward
            def _tracked_forward(x: torch.Tensor) -> torch.Tensor:
                out = _orig_fwd(x)
                METRICS.incr("transformer.hyper_gemm_ops")
                return out
            lin.forward = _tracked_forward  # type: ignore[method-assign]
            return lin
        except Exception as exc:
            _log.warning("transformer: HyperLinearNL unavailable (%s); "
                         "using nn.Linear fallback", exc)
    return nn.Linear(in_features, out_features, bias=bias)


def _make_ln(dim: int) -> nn.Module:
    """Return a HyperLN (Digital GPU) if available, else nn.LayerNorm."""
    if _ensure_gpu():
        try:
            from ai_model.gpu.hyper_creative_transformer import HyperLN
            from ai_model.maxcore.observability import METRICS
            ln = HyperLN(dim, _hyper_gpu)
            _orig_fwd = ln.forward
            def _tracked_forward(x: torch.Tensor) -> torch.Tensor:
                out = _orig_fwd(x)
                METRICS.incr("transformer.hyper_layernorm_ops")
                return out
            ln.forward = _tracked_forward  # type: ignore[method-assign]
            return ln
        except Exception as exc:
            _log.warning("transformer: HyperLN unavailable (%s); "
                         "using nn.LayerNorm fallback", exc)
    return nn.LayerNorm(dim)


def _head_gemm(h: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Weight-tied output projection through Digital GPU (never-raise).

    Equivalent to ``F.linear(h, weight)`` but the matmul routes through the
    in-house HyperGPU mixed-precision GEMM kernel.  Falls back to F.linear
    transparently if the backend is unavailable or raises.

    Args:
        h:      [*, dim] hidden states (any leading dims).
        weight: [vocab_size, dim] — shared with token_emb.weight.
    Returns:
        [*, vocab_size] logits.
    """
    if _ensure_gpu():
        try:
            from ai_model.gpu.hyper_backend import _MixedPrecisionGEMM
            from ai_model.maxcore.observability import METRICS
            shape = h.shape
            h2d = h.reshape(-1, shape[-1])
            # weight is (vocab, dim); GEMM needs (dim, vocab)
            out = _MixedPrecisionGEMM.apply(
                h2d.float(), weight.t().contiguous().float(), _hyper_gpu
            )
            METRICS.incr("transformer.hyper_head_ops")
            return out.reshape(*shape[:-1], weight.shape[0])
        except Exception as exc:
            _log.warning("transformer: _head_gemm GPU path failed (%s); "
                         "using F.linear fallback", exc)
    return F.linear(h, weight)


# ─── Warm-start routing probe ─────────────────────────────────────────────────

def probe_gpu_routing(model: "TransformerLM", prompt_len: int = 4) -> dict:
    """Verify that all generation-critical heavy math routes through the Digital GPU.

    Runs a single prefill pass on a synthetic prompt, then compares
    METRICS.snapshot() before and after to confirm HyperGPU ops were dispatched
    across all three critical paths:

    * ``transformer.hyper_gemm_ops``     — linear projections (qkv, out, ffn gate/down)
    * ``transformer.hyper_layernorm_ops`` — pre-norm layer norms
    * ``transformer.hyper_head_ops``     — weight-tied output projection (logits GEMM)

    Returns a dict with:
    - ``routed`` (bool): True only when ALL three paths recorded delta > 0
    - ``partial`` (bool): True when some but not all paths routed
    - ``fallback_paths`` (list[str]): names of paths still on torch fallback
    - Per-counter deltas for inspection

    Never raises — any exception is caught and reported in the return value.
    """
    try:
        from ai_model.maxcore.observability import METRICS
        snap_before = METRICS.snapshot()
        before_gemm = snap_before["counters"].get("transformer.hyper_gemm_ops", 0)
        before_ln   = snap_before["counters"].get("transformer.hyper_layernorm_ops", 0)
        before_head = snap_before["counters"].get("transformer.hyper_head_ops", 0)

        # Single-batch synthetic prefill (no grad needed, eval mode preserved)
        was_training = model.training
        model.eval()
        with torch.no_grad():
            ids = torch.zeros(1, prompt_len, dtype=torch.long)
            try:
                model.prefill(ids)
            finally:
                if was_training:
                    model.train()

        snap_after = METRICS.snapshot()
        delta_gemm = snap_after["counters"].get("transformer.hyper_gemm_ops", 0) - before_gemm
        delta_ln   = snap_after["counters"].get("transformer.hyper_layernorm_ops", 0) - before_ln
        delta_head = snap_after["counters"].get("transformer.hyper_head_ops", 0) - before_head

        # All three critical generation paths must register ops for full routing.
        path_results = {
            "linear_projections": delta_gemm > 0,
            "layer_norms":        delta_ln   > 0,
            "output_projection":  delta_head > 0,
        }
        fallback_paths = [name for name, ok in path_results.items() if not ok]
        fully_routed   = len(fallback_paths) == 0
        partially      = bool(fallback_paths) and any(path_results.values())

        result = {
            "routed":                   fully_routed,
            "partial":                  partially,
            "gpu_available":            _gpu_ready,
            "fallback_paths":           fallback_paths,
            "delta_hyper_gemm_ops":     delta_gemm,
            "delta_hyper_layernorm_ops": delta_ln,
            "delta_hyper_head_ops":     delta_head,
        }

        if fully_routed:
            _log.info(
                "transformer probe: full DigitalGPU routing confirmed "
                "(linear+%d, ln+%d, head+%d)",
                delta_gemm, delta_ln, delta_head,
            )
        elif partially:
            _log.warning(
                "transformer probe: PARTIAL routing – fallback torch paths: %s "
                "(linear+%d, ln+%d, head+%d)",
                fallback_paths, delta_gemm, delta_ln, delta_head,
            )
        else:
            _log.warning(
                "transformer probe: NO DigitalGPU ops detected – "
                "all paths on torch fallback"
            )
        return result
    except Exception as exc:
        _log.warning("transformer probe: error during routing check: %s", exc)
        return {"routed": False, "partial": False, "gpu_available": _gpu_ready,
                "error": str(exc)}


# ─── Static KV Cache ──────────────────────────────────────────────────────────

class KVCache:
    """Pre-allocated KV cache that eliminates per-step O(T) allocations.

    Standard decode uses ``torch.cat([past_k, new_k], dim=2)`` on every token,
    creating a brand-new tensor each step.  Over T decode steps that is O(T²)
    total memory traffic.

    This class pre-allocates ``[B, H, max_len, D_h]`` buffers once and writes
    new K/V slices in-place (O(1) per step).  It implements the same index
    protocol as the original ``list[tuple[Tensor, Tensor]]`` cache so callers
    can pass it into ``TransformerLM.decode_one`` transparently.

    Usage::

        # After prefill:
        logits, kv = model.prefill(prompt_ids)
        static_kv = KVCache.from_prefill(kv, max_new_tokens=200)

        # Decode loop — no allocation overhead:
        for _ in range(200):
            logits, static_kv = model.decode_one(next_id, static_kv)
    """

    def __init__(self, n_layers: int, batch: int, n_heads: int,
                 max_len: int, head_dim: int,
                 device: Optional[torch.device] = None) -> None:
        self._n_layers = n_layers
        self._max_len = max_len
        self._len = 0  # tokens filled so far
        self._k = [torch.zeros(batch, n_heads, max_len, head_dim, device=device)
                   for _ in range(n_layers)]
        self._v = [torch.zeros(batch, n_heads, max_len, head_dim, device=device)
                   for _ in range(n_layers)]

    # ── list-like interface (compatible with the old list[tuple] cache) ────────

    def __len__(self) -> int:
        return self._n_layers

    def __getitem__(self, i: int):  # → (k_view, v_view) truncated to current len
        return (self._k[i][:, :, :self._len, :],
                self._v[i][:, :, :self._len, :])

    # ── construction helpers ──────────────────────────────────────────────────

    @classmethod
    def from_prefill(cls, kv_list, max_new_tokens: int,
                     device: Optional[torch.device] = None) -> "KVCache":
        """Build a ``KVCache`` pre-loaded with the result of ``model.prefill``."""
        k0, v0 = kv_list[0]
        B, H, T_prompt, D_h = k0.shape
        n_layers = len(kv_list)
        max_len = T_prompt + max_new_tokens
        cache = cls(n_layers, B, H, max_len, D_h,
                    device=device or k0.device)
        for i, (k, v) in enumerate(kv_list):
            cache._k[i][:, :, :T_prompt, :] = k
            cache._v[i][:, :, :T_prompt, :] = v
        cache._len = T_prompt
        return cache


# ─── Rotary Position Embedding (RoPE) ────────────────────────────────────────

def precompute_rope_freqs(dim: int, max_len: int, base: float = 10000.0, device=None):
    """Pre-compute cos/sin tensors for rotary position embeddings."""
    half = dim // 2
    theta = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    t = torch.arange(max_len, device=device).float()
    freqs = torch.outer(t, theta)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to a query or key tensor [B, T, H, D_h].
    Positions are 0..T-1 (standard prefill / training path).
    """
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    c = cos[: x.shape[-3], :].unsqueeze(0).unsqueeze(2)  # [1, T, 1, D/2]
    s = sin[: x.shape[-3], :].unsqueeze(0).unsqueeze(2)
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)


def apply_rope_offset(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                      offset: int) -> torch.Tensor:
    """Apply rotary embeddings starting at `offset` (KV-cache decode path).
    x: [B, T, H, D_h]  —  T==1 during single-token decode steps.
    """
    T = x.shape[-3]
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    c = cos[offset : offset + T, :].unsqueeze(0).unsqueeze(2)
    s = sin[offset : offset + T, :].unsqueeze(0).unsqueeze(2)
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)


# ─── RoPE-aware Multi-head Self-Attention ─────────────────────────────────────

class RoPESelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        # Route through Digital GPU when available; fall back to nn.Linear.
        self.qkv = _make_linear(dim, 3 * dim, bias=False)
        self.out = _make_linear(dim, dim, bias=False)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # each [B, T, H, D_h]

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn + mask.unsqueeze(0).unsqueeze(0)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).contiguous().reshape(B, T, C)
        return self.out(out)

    def forward_with_kv(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                        mask: torch.Tensor | None = None,
                        key_padding_mask: torch.Tensor | None = None,
                        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prefill pass — identical to forward() but also returns K, V for cache seeding.

        ``key_padding_mask`` (optional): bool [B, T], True at PAD key positions to
        exclude from attention. Used for batched generation of unequal-length
        prompts (left-padded). None preserves the original single-sequence path.
        """
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
            # Large finite negative (not -inf) so fully-masked PAD query rows
            # produce a finite (discarded) output instead of NaN.
            attn = attn.masked_fill(key_padding_mask[:, None, None, :], -1e9)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).contiguous().reshape(B, T, C)
        return self.out(out), k, v  # k, v: [B, H, T, D_h]

    def decode_one(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                   past_k: torch.Tensor, past_v: torch.Tensor,
                   key_padding_mask: torch.Tensor | None = None,
                   _kv_buf: Optional[tuple] = None,
                   _write_pos: Optional[int] = None,
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-token KV-cache decode step.
        x: [B, 1, C]  —  past_k / past_v: [B, H, T_past, D_h]
        Returns: (out [B, 1, C], new_k [B, H, T_past+1, D_h], new_v)

        ``key_padding_mask`` (optional): bool [B, T_past+1] over the full cache
        (incl. the new token) marking PAD positions to exclude. None preserves
        the original single-sequence path.

        ``_kv_buf`` / ``_write_pos`` (optional): pass the pre-allocated full-length
        KV buffers from a ``KVCache`` instance to enable in-place writes.  When
        provided, new K/V is written directly into the buffer at ``_write_pos``
        and a view is returned, eliminating the ``torch.cat`` allocation.
        """
        offset = past_k.shape[2]
        B, T, C = x.shape  # T == 1

        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # each [B, 1, H, D_h]

        q = apply_rope_offset(q, cos, sin, offset)
        k = apply_rope_offset(k, cos, sin, offset)

        q = q.transpose(1, 2)  # [B, H, 1, D_h]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if _kv_buf is not None and _write_pos is not None:
            # In-place path: write new K/V at _write_pos in the pre-allocated
            # buffer and return a view of [B, H, :_write_pos+1, D_h].
            # This eliminates the O(T) torch.cat allocation on every step.
            k_buf, v_buf = _kv_buf
            k_buf[:, :, _write_pos:_write_pos + 1, :] = k
            v_buf[:, :, _write_pos:_write_pos + 1, :] = v
            k_cat = k_buf[:, :, :_write_pos + 1, :]   # view — zero allocation
            v_cat = v_buf[:, :, :_write_pos + 1, :]
        else:
            k_cat = torch.cat([past_k, k], dim=2)  # [B, H, T_past+1, D_h]
            v_cat = torch.cat([past_v, v], dim=2)

        # Single query attends to entire causal context.
        attn = (q @ k_cat.transpose(-2, -1)) * self.scale  # [B, H, 1, T_past+1]
        if key_padding_mask is not None:
            attn = attn.masked_fill(key_padding_mask[:, None, None, :], -1e9)
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v_cat).transpose(1, 2).contiguous().reshape(B, T, C)
        return self.out(out), k_cat, v_cat


# ─── Feed-forward with SwiGLU activation ─────────────────────────────────────

class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward: uses 2/3 the parameters for same effective width."""
    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden = int(dim * expansion * 2 / 3)
        hidden = ((hidden + 63) // 64) * 64  # round to multiple of 64
        # Route gate and down projections through Digital GPU.
        self.gate = _make_linear(dim, hidden * 2, bias=False)
        self.down = _make_linear(hidden, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g, v = self.gate(x).chunk(2, dim=-1)
        return self.down(self.drop(F.silu(g) * v))


# ─── Decoder Layer ─────────────────────────────────────────────────────────────

class TransformerDecoderLayer(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        # Route layer norms through Digital GPU.
        self.ln1 = _make_ln(dim)
        self.attn = RoPESelfAttention(dim, n_heads, dropout)
        self.ln2 = _make_ln(dim)
        self.ffn = SwiGLUFFN(dim, dropout=dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.drop(self.attn(self.ln1(x), cos, sin, mask))
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x

    def forward_with_kv(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                        mask: torch.Tensor | None = None,
                        key_padding_mask: torch.Tensor | None = None,
                        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prefill: returns (output, k, v) for KV cache seeding."""
        attn_out, k, v = self.attn.forward_with_kv(self.ln1(x), cos, sin, mask, key_padding_mask)
        x = x + self.drop(attn_out)
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x, k, v

    def decode_one(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                   past_k: torch.Tensor, past_v: torch.Tensor,
                   key_padding_mask: torch.Tensor | None = None,
                   _kv_buf: Optional[tuple] = None,
                   _write_pos: Optional[int] = None,
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-token decode with KV cache. No dropout at inference."""
        attn_out, new_k, new_v = self.attn.decode_one(
            self.ln1(x), cos, sin, past_k, past_v, key_padding_mask,
            _kv_buf=_kv_buf, _write_pos=_write_pos)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, new_k, new_v


# ─── Full Language Model ───────────────────────────────────────────────────────

class TransformerLM(nn.Module):
    """
    Decoder-only transformer with:
    - Rotary Position Embeddings (RoPE)
    - SwiGLU feed-forward networks
    - Pre-norm (LayerNorm before each sub-layer)
    - Weight tying between token embedding and output head
    - Scaled initialization (GPT-2 style)
    - KV-cache support via prefill() + decode_one()
    - Digital GPU routing: all linear projections, layer norms, and the
      weight-tied output projection use HyperGPU kernels when available, with a
      transparent torch fallback (never-raise).  Use probe_gpu_routing() to
      confirm all three paths are active after model construction.
    """
    def __init__(self, vocab_size: int, dim: int = 512, n_layers: int = 8,
                 n_heads: int = 8, max_len: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.max_len = max_len

        self.token_emb = nn.Embedding(vocab_size, dim)
        self.emb_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TransformerDecoderLayer(dim, n_heads, dropout)
            for _ in range(n_layers)
        ])
        # Final layer norm routes through Digital GPU.
        self.ln_final = _make_ln(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

        # Weight tying: output head shares weights with token embedding
        self.head.weight = self.token_emb.weight

        # Pre-compute RoPE frequencies up to max_len
        rope_cos, rope_sin = precompute_rope_freqs(dim // n_heads, max_len)
        self.register_buffer("rope_cos", rope_cos)
        self.register_buffer("rope_sin", rope_sin)

        # Pre-compute causal mask once — sliced to [T,T] on each forward pass
        causal_mask = torch.triu(torch.full((max_len, max_len), float('-inf')), diagonal=1)
        self.register_buffer("causal_mask", causal_mask)

        self._init_weights(n_layers)

    def _init_weights(self, n_layers: int):
        """GPT-2 style scaled initialization."""
        std = 0.02
        residual_std = std / math.sqrt(2 * n_layers)
        for name, param in self.named_parameters():
            if "head" in name:
                continue  # tied, initialized with emb
            if param.dim() >= 2:
                if "out.weight" in name or "down.weight" in name:
                    nn.init.normal_(param, mean=0.0, std=residual_std)
                else:
                    nn.init.normal_(param, mean=0.0, std=std)
            elif "bias" in name:
                nn.init.zeros_(param)
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard training / non-cached forward pass."""
        B, T = x.shape
        assert T <= self.max_len, f"Sequence length {T} exceeds max_len {self.max_len}"

        h = self.emb_dropout(self.token_emb(x))

        mask = self.causal_mask[:T, :T]
        cos = self.rope_cos[:T]
        sin = self.rope_sin[:T]

        for layer in self.layers:
            h = layer(h, cos, sin, mask)

        h = self.ln_final(h)
        return _head_gemm(h, self.token_emb.weight)

    def prefill(self, x: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None,
                ) -> tuple[torch.Tensor, list]:
        """
        KV-cache prefill: process the full prompt in one batched pass.
        Returns (logits [B, T, vocab], kv_cache).
        kv_cache is a list of (k, v) per layer — [B, H, T, D_h] each.
        Use kv_cache with decode_one() for O(1)-per-step generation.

        ``key_padding_mask`` (optional): bool [B, T], True at PAD positions. Pass
        when batching left-padded unequal-length prompts. None → original path.
        """
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
        return _head_gemm(h, self.token_emb.weight), kv_cache

    def decode_one(self, x_new: torch.Tensor,
                   kv_cache,
                   key_padding_mask: torch.Tensor | None = None,
                   ) -> tuple[torch.Tensor, object]:
        """
        Single-token KV-cache decode step — O(1) per token (constant context cost).
        x_new: [B, 1] — single new token id.
        Returns (logits [B, 1, vocab], new_kv_cache).

        ``kv_cache`` may be either the original ``list[tuple[Tensor, Tensor]]``
        returned by ``prefill()`` or a ``KVCache`` instance created via
        ``KVCache.from_prefill()``.  The ``KVCache`` path is ~2× faster for
        long sequences because it writes new K/V in-place (no ``torch.cat``).

        ``key_padding_mask`` (optional): bool [B, T_cache] over the full cache
        (incl. the new token). Pass when batching left-padded prompts. None →
        original path.
        """
        h = self.token_emb(x_new)  # [B, 1, dim] — no dropout at inference

        if isinstance(kv_cache, KVCache):
            # Fast path: in-place writes into pre-allocated buffers.
            # torch.cat([past_k, new_k], dim=2) is eliminated entirely;
            # each layer writes its new slice directly at write_pos.
            write_pos = kv_cache._len
            for i, layer in enumerate(self.layers):
                past_k, past_v = kv_cache[i]  # views: [B, H, :len, D_h]
                h, _, _ = layer.decode_one(
                    h, self.rope_cos, self.rope_sin, past_k, past_v,
                    key_padding_mask,
                    _kv_buf=(kv_cache._k[i], kv_cache._v[i]),
                    _write_pos=write_pos,
                )
            kv_cache._len += 1
            h = self.ln_final(h)
            return _head_gemm(h, self.token_emb.weight), kv_cache

        # Original list-of-tuples path — fully backward compatible.
        new_cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i, layer in enumerate(self.layers):
            past_k, past_v = kv_cache[i]
            h, new_k, new_v = layer.decode_one(
                h, self.rope_cos, self.rope_sin, past_k, past_v, key_padding_mask)
            new_cache.append((new_k, new_v))

        h = self.ln_final(h)
        return _head_gemm(h, self.token_emb.weight), new_cache

    # Legacy compatibility: some code checks pos_emb.num_embeddings
    @property
    def pos_emb(self):
        class _FakeEmb:
            num_embeddings = self.max_len
        return _FakeEmb()
