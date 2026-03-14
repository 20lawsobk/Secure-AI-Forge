from __future__ import annotations
from typing import Any, Dict, Optional


class TrainConfig:
    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        model_cfg = {}
        train_cfg = {}
        if cfg:
            model_cfg = cfg.get("model", {})
            train_cfg = cfg.get("train", {})

        self.dim: int = model_cfg.get("dim", 512)
        self.layers: int = model_cfg.get("layers", 8)
        self.heads: int = model_cfg.get("heads", 8)
        self.max_len: int = model_cfg.get("max_len", 1024)

        self.lr: float = train_cfg.get("lr", 3e-4)
        self.batch_size: int = train_cfg.get("batch_size", 8)
        self.epochs: int = train_cfg.get("epochs", 3)
        self.data_path: str = train_cfg.get("data_path", "training/boostsheet_samples.json")

        self.weight_decay: float = 0.01
        self.label_smoothing: float = 0.1
        self.gradient_accumulation_steps: int = 4
        self.warmup_ratio: float = 0.06
        self.min_lr_ratio: float = 0.1
        self.max_grad_norm: float = 1.0

    def __repr__(self):
        return (
            f"TrainConfig(dim={self.dim}, layers={self.layers}, heads={self.heads}, "
            f"max_len={self.max_len}, lr={self.lr}, batch_size={self.batch_size}, "
            f"epochs={self.epochs})"
        )
