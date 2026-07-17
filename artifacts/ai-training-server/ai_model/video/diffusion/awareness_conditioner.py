"""
Awareness Conditioner for MaxCore Diffusion.

Converts the live awareness context string + Visual DNA parameters into a
fixed-size conditioning tensor that the Temporal DiT cross-attends over.

Design:
  - Keyword vocabulary of 64 music/content domain terms → one-hot presence vector
  - Visual DNA float params (energy, darkness, warmth, saturation) appended
  - Linear projection to (N_TOKENS, D_MODEL) conditioning sequence
  - Same awareness → deterministically same conditioning (no randomness here)

The N_TOKENS conditioning tokens are treated as "style memory" — the DiT
cross-attends over them on every denoising step, so they continuously bias the
generated frames toward the current industry moment.
"""
from __future__ import annotations

import re
import torch
import torch.nn as nn
from typing import List, Optional

N_TOKENS: int = 8
D_MODEL: int = 256

VOCAB: List[str] = [
    "tiktok", "instagram", "youtube", "facebook", "twitter", "linkedin",
    "phonk", "trap", "drill", "afrobeats", "latin", "hyperpop", "rnb",
    "soul", "indie", "lofi", "jazz", "pop", "reggaeton", "tropical",
    "viral", "trending", "algorithm", "shorts", "reels", "playlist",
    "editorial", "spotify", "apple", "growth", "engagement", "conversion",
    "dark", "bright", "energetic", "calm", "melancholic", "euphoric",
    "hype", "cinematic", "moody", "warm", "cool", "gritty", "clean",
    "808", "bass", "hook", "drop", "build", "chorus", "verse",
    "collab", "feature", "duet", "challenge", "fyp", "freestyle", "release",
    "morning", "evening", "night", "peak", "weekend", "summer",
    "high", "medium", "low", "urgent",
]

_VOCAB_IDX = {w: i for i, w in enumerate(VOCAB)}
_VOCAB_SIZE = len(VOCAB)
_DNA_SIZE = 4
_INPUT_SIZE = _VOCAB_SIZE + _DNA_SIZE


def _presence_vector(awareness: str) -> torch.Tensor:
    """Build a float32 one-hot presence vector over VOCAB from the awareness string."""
    vec = torch.zeros(_VOCAB_SIZE, dtype=torch.float32)
    text_lower = awareness.lower()
    for word, idx in _VOCAB_IDX.items():
        if word in text_lower:
            vec[idx] = 1.0
    # Upweight HIGH-signal keywords
    for line in awareness.splitlines():
        m = re.match(r"\[HIGH\]\s+(.+)", line.strip())
        if m:
            headline = m.group(1).lower()
            for word, idx in _VOCAB_IDX.items():
                if word in headline:
                    vec[idx] = min(vec[idx] + 0.5, 2.0)
    return vec


class AwarenessConditioner(nn.Module):
    """Projects awareness presence + Visual DNA → N_TOKENS conditioning tokens."""

    def __init__(self, d_model: int = D_MODEL, n_tokens: int = N_TOKENS) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_tokens = n_tokens
        self.proj = nn.Linear(_INPUT_SIZE, n_tokens * d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        awareness: str,
        dna_params: Optional[List[float]] = None,
        batch_size: int = 1,
    ) -> torch.Tensor:
        """Return conditioning tensor of shape [batch_size, N_TOKENS, D_MODEL]."""
        pres = _presence_vector(awareness)

        if dna_params is not None and len(dna_params) >= 4:
            dna = torch.tensor(dna_params[:4], dtype=torch.float32)
        else:
            dna = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float32)

        feat = torch.cat([pres, dna], dim=0)
        tokens = self.proj(feat).view(self.n_tokens, self.d_model)
        tokens = self.norm(tokens)
        return tokens.unsqueeze(0).expand(batch_size, -1, -1)

    @torch.no_grad()
    def encode(
        self,
        awareness: str,
        dna_params: Optional[List[float]] = None,
        batch_size: int = 1,
    ) -> torch.Tensor:
        """Inference-mode encode — no grad."""
        return self.forward(awareness, dna_params, batch_size)
