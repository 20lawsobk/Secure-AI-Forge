import torch
import torch.nn as nn
from ai_model.gpu.torch_backend import DigitalGPUBackend


class DigitalGPUFeedForward(nn.Module):
    def __init__(self, dim, ff_dim, gpu_backend, dropout=0.1):
        super().__init__()
        self.linear1 = gpu_backend.linear(dim, ff_dim, fused_relu=True)
        self.linear2 = gpu_backend.linear(ff_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class DigitalGPUDecoderLayer(nn.Module):
    def __init__(self, dim, n_heads, gpu_backend, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = gpu_backend.attention(dim, n_heads)
        self.ff = DigitalGPUFeedForward(dim, 4 * dim, gpu_backend, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        residual = x
        x = self.norm1(x)
        x = self.attn(x, mask=mask)
        x = self.dropout1(x) + residual

        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout2(x) + residual
        return x


class AcceleratedTransformerLM(nn.Module):
    def __init__(self, vocab_size, dim=256, n_layers=4, n_heads=4, max_len=512, dropout=0.1, gpu_backend=None):
        super().__init__()
        if gpu_backend is None:
            gpu_backend = DigitalGPUBackend(lanes=32)
        self.gpu_backend = gpu_backend
        self.dim = dim

        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_len, dim)
        self.emb_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            DigitalGPUDecoderLayer(dim, n_heads, gpu_backend, dropout)
            for _ in range(n_layers)
        ])

        self.ln = nn.LayerNorm(dim)
        self.head = gpu_backend.linear(dim, vocab_size)

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
        return self.gpu_backend.status()

    def gpu_profile(self):
        return self.gpu_backend.profile()
