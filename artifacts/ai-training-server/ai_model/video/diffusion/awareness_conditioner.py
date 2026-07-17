"""
Awareness Conditioner for MaxCore Diffusion.

Converts the live awareness context string + Visual DNA parameters into a
fixed-size conditioning tensor that the Temporal DiT cross-attends over.

Design:
  - Keyword vocabulary of music/content domain terms → weighted presence vector
  - Visual DNA float params (energy, darkness, warmth, saturation) appended
  - Linear projection to (N_TOKENS, D_MODEL) conditioning sequence
  - Same awareness → deterministically same conditioning (no randomness here)

The N_TOKENS conditioning tokens are treated as "style memory" — the DiT
cross-attends over them on every denoising step, so they continuously bias the
generated frames toward the current industry moment.

Vocab coverage (129 terms, no duplicates):
  Platforms    — all major + emerging (Threads, Bluesky, Pinterest, Snapchat…)
  Genres       — current landscape incl. amapiano, citypop, bedroom, rage,
                 pluggnb, synthwave, grime, jersey, emo, shoegaze, funk…
  Distribution — viral, fyp, shorts, reels, editorial, exclusive, debut…
  Production   — 808, bass, hook, drop, beat, sample, bpm, mix, master…
  Mood & Tone  — cinematic, soulful, nostalgic, aggressive, smooth, polished…
  Visual       — neon, aesthetic, retro, vintage, futuristic, pastel, grunge…
  Timing       — morning, night, peak, weekend, summer, winter, midnight…
  Signal       — high, medium, low, urgent
"""
from __future__ import annotations

import re
import torch
import torch.nn as nn
from typing import List, Optional

N_TOKENS: int = 8
D_MODEL: int = 256

VOCAB: List[str] = [
    # ── Platforms ──────────────────────────────────────────────────────────────
    "tiktok", "instagram", "youtube", "facebook", "twitter", "linkedin",
    "threads", "bluesky", "pinterest", "snapchat", "discord", "twitch",
    "soundcloud", "bereal", "spotify", "apple",

    # ── Genres & Subgenres ─────────────────────────────────────────────────────
    "phonk", "trap", "drill", "afrobeats", "latin", "hyperpop", "rnb",
    "soul", "indie", "lofi", "jazz", "pop", "reggaeton", "tropical",
    "amapiano", "citypop", "bedroom", "rage", "pluggnb", "synthwave",
    "grime", "gospel", "dancehall", "country", "edm", "house", "techno",
    "hiphop", "cumbia", "baile", "jersey", "emo", "shoegaze", "funk",

    # ── Distribution & Discovery ───────────────────────────────────────────────
    "viral", "trending", "algorithm", "shorts", "reels", "playlist",
    "editorial", "growth", "engagement", "conversion", "fyp", "exclusive",
    "debut", "premiere", "release", "challenge",

    # ── Music Production ───────────────────────────────────────────────────────
    "808", "bass", "hook", "drop", "build", "chorus", "verse",
    "beat", "sample", "wave", "bop", "bridge", "tempo", "bpm",
    "mix", "master", "prod", "collab", "feature", "duet", "freestyle",

    # ── Mood & Tone ────────────────────────────────────────────────────────────
    "dark", "bright", "energetic", "calm", "melancholic", "euphoric",
    "hype", "cinematic", "moody", "warm", "cool", "gritty", "clean",
    "soulful", "nostalgic", "raw", "polished", "aggressive", "smooth",

    # ── Visual / Aesthetic ────────────────────────────────────────────────────
    "neon", "aesthetic", "retro", "vintage", "minimal", "vibrant",
    "futuristic", "pastel", "grunge",

    # ── Timing & Context ─────────────────────────────────────────────────────
    "morning", "evening", "night", "peak", "weekend", "summer",
    "winter", "midnight", "spring",

    # ── Signal Strength ───────────────────────────────────────────────────────
    "high", "medium", "low", "urgent",
]

# Sanity-check at import time: no duplicate terms allowed.
_seen: set[str] = set()
for _term in VOCAB:
    assert _term not in _seen, f"Duplicate VOCAB entry: {_term!r}"
    _seen.add(_term)
del _seen, _term

_VOCAB_IDX = {w: i for i, w in enumerate(VOCAB)}
_VOCAB_SIZE = len(VOCAB)
_DNA_SIZE = 4
_INPUT_SIZE = _VOCAB_SIZE + _DNA_SIZE


def _presence_vector(awareness: str) -> torch.Tensor:
    """Build a float32 weighted presence vector over VOCAB from the awareness string.

    Base match  → 1.0
    [HIGH] line → +0.5 (capped at 2.0), so HIGH-signal terms bias diffusion more
    """
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
