import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
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
                        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prefill pass — identical to forward() but also returns K, V for cache seeding."""
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
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).contiguous().reshape(B, T, C)
        return self.out(out), k, v  # k, v: [B, H, T, D_h]

    def decode_one(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                   past_k: torch.Tensor, past_v: torch.Tensor,
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-token KV-cache decode step.
        x: [B, 1, C]  —  past_k / past_v: [B, H, T_past, D_h]
        Returns: (out [B, 1, C], new_k [B, H, T_past+1, D_h], new_v)
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

        k_cat = torch.cat([past_k, k], dim=2)  # [B, H, T_past+1, D_h]
        v_cat = torch.cat([past_v, v], dim=2)

        # Single query attends to entire causal context — no mask needed
        attn = (q @ k_cat.transpose(-2, -1)) * self.scale  # [B, H, 1, T_past+1]
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
        self.gate = nn.Linear(dim, hidden * 2, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g, v = self.gate(x).chunk(2, dim=-1)
        return self.down(self.drop(F.silu(g) * v))


# ─── Decoder Layer ─────────────────────────────────────────────────────────────

class TransformerDecoderLayer(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = RoPESelfAttention(dim, n_heads, dropout)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = SwiGLUFFN(dim, dropout=dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.drop(self.attn(self.ln1(x), cos, sin, mask))
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x

    def forward_with_kv(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                        mask: torch.Tensor | None = None,
                        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prefill: returns (output, k, v) for KV cache seeding."""
        attn_out, k, v = self.attn.forward_with_kv(self.ln1(x), cos, sin, mask)
        x = x + self.drop(attn_out)
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x, k, v

    def decode_one(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                   past_k: torch.Tensor, past_v: torch.Tensor,
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-token decode with KV cache. No dropout at inference."""
        attn_out, new_k, new_v = self.attn.decode_one(self.ln1(x), cos, sin, past_k, past_v)
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
        self.ln_final = nn.LayerNorm(dim)
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
        return self.head(h)

    def prefill(self, x: torch.Tensor) -> tuple[torch.Tensor, list]:
        """
        KV-cache prefill: process the full prompt in one batched pass.
        Returns (logits [B, T, vocab], kv_cache).
        kv_cache is a list of (k, v) per layer — [B, H, T, D_h] each.
        Use kv_cache with decode_one() for O(1)-per-step generation.
        """
        B, T = x.shape
        h = self.emb_dropout(self.token_emb(x))
        mask = self.causal_mask[:T, :T]
        cos = self.rope_cos[:T]
        sin = self.rope_sin[:T]

        kv_cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer in self.layers:
            h, k, v = layer.forward_with_kv(h, cos, sin, mask)
            kv_cache.append((k, v))

        h = self.ln_final(h)
        return self.head(h), kv_cache

    def decode_one(self, x_new: torch.Tensor,
                   kv_cache: list[tuple[torch.Tensor, torch.Tensor]],
                   ) -> tuple[torch.Tensor, list]:
        """
        Single-token KV-cache decode step — O(1) per token (constant context cost).
        x_new: [B, 1] — single new token id.
        Returns (logits [B, 1, vocab], new_kv_cache).
        """
        h = self.token_emb(x_new)  # [B, 1, dim] — no dropout at inference

        new_cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i, layer in enumerate(self.layers):
            past_k, past_v = kv_cache[i]
            h, new_k, new_v = layer.decode_one(h, self.rope_cos, self.rope_sin, past_k, past_v)
            new_cache.append((new_k, new_v))

        h = self.ln_final(h)
        return self.head(h), new_cache

    # Legacy compatibility: some code checks pos_emb.num_embeddings
    @property
    def pos_emb(self):
        class _FakeEmb:
            num_embeddings = self.max_len
        return _FakeEmb()