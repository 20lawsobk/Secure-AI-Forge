from __future__ import annotations
import torch
import torch.nn as nn
from typing import Dict, Optional, List
from ai_model.gpu.multi_backend import StreamBackend, MultiStreamBackend
from ai_model.gpu.accelerated_transformer import DigitalGPUDecoderLayer


AGENT_TYPES = ["script", "distribution", "visual_spec", "optimization"]


class AgentHead(nn.Module):
    def __init__(self, agent_type: str, dim: int, vocab_size: int, stream_backend: StreamBackend):
        super().__init__()
        self.agent_type = agent_type
        self.dim = dim
        self.vocab_size = vocab_size

        self.adapter = nn.Sequential(
            stream_backend.linear(dim, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
        )
        self.head = stream_backend.linear(dim, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight is not None and m.weight.dim() >= 2:
                nn.init.xavier_uniform_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, backbone_output: torch.Tensor) -> torch.Tensor:
        h = self.adapter(backbone_output)
        return self.head(h)


class SharedBackbone(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        max_len: int = 128,
        dropout: float = 0.1,
        stream_backend: Optional[StreamBackend] = None,
    ):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size

        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_len, dim)
        self.emb_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            DigitalGPUDecoderLayer(dim, n_heads, stream_backend, dropout)
            for _ in range(n_layers)
        ])

        self.ln = nn.LayerNorm(dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
        for layer in self.layers:
            for name, param in layer.named_parameters():
                if 'weight' in name and param.dim() >= 2:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        pos = torch.arange(0, T, device=x.device).unsqueeze(0)
        h = self.token_emb(x) + self.pos_emb(pos)
        h = self.emb_dropout(h)
        for layer in self.layers:
            h = layer(h)
        h = self.ln(h)
        return h


class MultiHeadModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        max_len: int = 128,
        dropout: float = 0.1,
        multi_backend: Optional[MultiStreamBackend] = None,
        agent_types: Optional[List[str]] = None,
    ):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.agent_types = agent_types or AGENT_TYPES

        if multi_backend is None:
            multi_backend = MultiStreamBackend(total_lanes=32)
        self.multi_backend = multi_backend

        self._backbone_stream = multi_backend.create_stream(
            name="backbone", lanes=0
        )
        self.backbone = SharedBackbone(
            vocab_size=vocab_size,
            dim=dim,
            n_layers=n_layers,
            n_heads=n_heads,
            max_len=max_len,
            dropout=dropout,
            stream_backend=self._backbone_stream,
        )

        self.agent_heads = nn.ModuleDict()
        self._head_streams: Dict[str, StreamBackend] = {}

        for agent_type in self.agent_types:
            stream = multi_backend.create_stream(name=f"head_{agent_type}", lanes=0)
            self._head_streams[agent_type] = stream
            self.agent_heads[agent_type] = AgentHead(
                agent_type=agent_type,
                dim=dim,
                vocab_size=vocab_size,
                stream_backend=stream,
            )

        multi_backend.rebalance()

    def forward(self, x: torch.Tensor, agent_type: str = "script") -> torch.Tensor:
        if agent_type not in self.agent_heads:
            raise ValueError(
                f"Unknown agent type '{agent_type}'. "
                f"Available: {list(self.agent_heads.keys())}"
            )
        h = self.backbone(x)
        logits = self.agent_heads[agent_type](h)
        return logits

    def forward_all(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.backbone(x)
        results = {}
        for agent_type in self.agent_types:
            results[agent_type] = self.agent_heads[agent_type](h)
        return results

    def flush_all_vram(self):
        self._backbone_stream.flush_vram()
        for stream in self._head_streams.values():
            stream.flush_vram()

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def backbone_params(self):
        return list(self.backbone.parameters())

    def head_params(self, agent_type: str):
        if agent_type not in self.agent_heads:
            raise ValueError(f"Unknown agent type: {agent_type}")
        return list(self.agent_heads[agent_type].parameters())

    def all_head_params(self):
        params = []
        for head in self.agent_heads.values():
            params.extend(head.parameters())
        return params

    def gpu_status(self) -> dict:
        return self.multi_backend.status()

    def gpu_profile(self) -> dict:
        return self.multi_backend.profile_all()

    def param_summary(self) -> dict:
        backbone_count = sum(p.numel() for p in self.backbone.parameters())
        head_counts = {}
        for name, head in self.agent_heads.items():
            head_counts[name] = sum(p.numel() for p in head.parameters())
        total = backbone_count + sum(head_counts.values())
        return {
            "total_params": total,
            "backbone_params": backbone_count,
            "head_params": head_counts,
            "agent_types": self.agent_types,
            "backbone_shared": True,
        }

    @classmethod
    def from_pretrained_backbone(
        cls,
        checkpoint_path: str,
        multi_backend: Optional[MultiStreamBackend] = None,
        agent_types: Optional[List[str]] = None,
    ) -> "MultiHeadModel":
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("config", {})
        vocab_size = ckpt.get("next_id", 1000)

        model = cls(
            vocab_size=vocab_size,
            dim=cfg.get("dim", 128),
            n_layers=cfg.get("layers", 3),
            n_heads=cfg.get("heads", 4),
            max_len=cfg.get("max_len", 128),
            multi_backend=multi_backend,
            agent_types=agent_types,
        )

        state = ckpt.get("model_state_dict", {})
        backbone_state = {}
        for k, v in state.items():
            mapped = None
            if k.startswith("token_emb."):
                mapped = k
            elif k.startswith("pos_emb."):
                mapped = k
            elif k.startswith("layers."):
                mapped = k
            elif k.startswith("ln."):
                mapped = k
            if mapped:
                backbone_state[mapped] = v

        if backbone_state:
            missing, unexpected = model.backbone.load_state_dict(
                backbone_state, strict=False
            )
            loaded = len(backbone_state) - len(unexpected)
            print(f"Loaded {loaded} backbone weights from checkpoint")

        return model
