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
    """Apply rotary embeddings to a query or key tensor  [B, T, H, D]."""
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    c = cos[: x.shape[-3], :].unsqueeze(0).unsqueeze(2)  # [1,T,1,D/2]
    s = sin[: x.shape[-3], :].unsqueeze(0).unsqueeze(2)
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
        q, k, v = qkv.unbind(2)  # each [B,T,H,D_h]

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # [B,H,T,D_h]
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


# ─── Full Language Model ───────────────────────────────────────────────────────

class TransformerLM(nn.Module):
    """
    Decoder-only transformer with:
    - Rotary Position Embeddings (RoPE)
    - SwiGLU feed-forward networks
    - Pre-norm (LayerNorm before each sub-layer)
    - Weight tying between token embedding and output head
    - Scaled initialization (GPT-2 style)
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
        B, T = x.shape
        assert T <= self.max_len, f"Sequence length {T} exceeds max_len {self.max_len}"

        h = self.emb_dropout(self.token_emb(x))

        # Slice pre-computed causal mask — no recomputation per forward pass
        mask = self.causal_mask[:T, :T]

        cos = self.rope_cos[:T]
        sin = self.rope_sin[:T]

        for layer in self.layers:
            h = layer(h, cos, sin, mask)

        h = self.ln_final(h)
        return self.head(h)

    # Legacy compatibility: some code checks pos_emb.num_embeddings
    @property
    def pos_emb(self):
        class _FakeEmb:
            num_embeddings = self.max_len
        return _FakeEmb()
