from __future__ import annotations
import math
import torch
import torch.nn as nn
from ai_model.gpu.hyper_backend import HyperGPUBackend
from ai_model.gpu.hyper_core import PrecisionMode


class HyperFeedForward(nn.Module):
    def __init__(self, dim, ff_dim, backend: HyperGPUBackend, dropout=0.1):
        super().__init__()
        self.linear1 = backend.linear(dim, ff_dim, mixed_precision=True)
        self.gelu = backend.gelu()
        self.linear2 = backend.linear(ff_dim, dim, mixed_precision=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class HyperDecoderLayer(nn.Module):
    def __init__(self, dim, n_heads, backend: HyperGPUBackend, dropout=0.1):
        super().__init__()
        self.norm1 = backend.layer_norm(dim)
        self.norm2 = backend.layer_norm(dim)
        self.attn = backend.flash_attention(dim=dim, n_heads=n_heads, block_size=32)
        self.ff = HyperFeedForward(dim, 4 * dim, backend, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        residual = x
        x = self.norm1(x)
        x = self.attn(x, causal=True)
        x = self.dropout1(x) + residual

        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout2(x) + residual
        return x


class HyperTransformerLM(nn.Module):
    def __init__(self, vocab_size, dim=256, n_layers=4, n_heads=4,
                 max_len=512, dropout=0.1, backend=None):
        super().__init__()
        if backend is None:
            backend = HyperGPUBackend(
                lanes=512, tensor_cores=8, precision=PrecisionMode.MIXED
            )
        self.backend = backend
        self.dim = dim

        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_len, dim)
        self.emb_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            HyperDecoderLayer(dim, n_heads, backend, dropout)
            for _ in range(n_layers)
        ])

        self.ln = backend.layer_norm(dim)
        self.head = backend.linear(dim, vocab_size, mixed_precision=True)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

        for layer in self.layers:
            for name, param in layer.named_parameters():
                if 'weight' in name and param.dim() >= 2:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(0, T, device=x.device).unsqueeze(0)
        h = self.token_emb(x) + self.pos_emb(pos)
        h = self.emb_dropout(h)

        for layer in self.layers:
            h = layer(h)

        h = self.ln(h)
        logits = self.head(h)
        return logits

    def gpu_status(self):
        return self.backend.status()
