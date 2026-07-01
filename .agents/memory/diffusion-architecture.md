---
name: MaxCore Diffusion Architecture
description: Neural video diffusion system built on HyperGPU — SDEdit prior, Temporal DiT, awareness conditioning, VAE. Integration pattern and key constraints.
---

# MaxCore Diffusion Architecture

## Files
All 7 new files live in `ai_model/video/diffusion/`:
- `noise_scheduler.py` — DDPMScheduler + DDIMScheduler (pure NumPy, no torch dep)
- `music_vae.py` — MusicVAE encoder/decoder (256×256×3 ↔ 32×32×4, standard PyTorch conv)
- `awareness_conditioner.py` — keyword vocab presence + DNA params → [N_TOKENS=8, D_MODEL=256] conditioning tensor
- `temporal_dit.py` — TemporalDiT with spatial HyperFlashAttention, temporal HyperFlashAttention, awareness cross-attention, AdaLayerNorm
- `sdedit_prior.py` — SDEditPrior: RCGS real frame → partial noise (40%) as denoising starting point
- `maxcore_diffusion.py` — MaxCoreDiffusionPipeline singleton + `get_diffusion_frame()` public API
- `__init__.py` — exports MaxCoreDiffusionPipeline, get_diffusion_frame

## Key parameters
- Resolution: 256×256 → 32×32×4 latent (8× compression, 4 latent channels)
- Patch size: 4×4 → 64 patches per frame
- d_model=256, n_heads=4, n_layers=4, DDIM steps=20, noise_fraction=0.40

## HyperGPU usage
- Spatial self-attention: HyperFlashAttention over 64 spatial patches
- Temporal self-attention: HyperFlashAttention over T frames per spatial position
- FFN: HyperGPULinear (4× expansion)
- VAE uses standard PyTorch (HyperGPU reserved for DiT — compute-intensive path)
- Backend: `HyperGPUBackend(lanes=256, tensor_cores=4)` — smaller than main model to avoid contention

## Integration (additive, non-breaking)
- `SceneConfig.diffusion_meta: Optional[dict]` — None = skip diffusion (existing pipeline unaffected)
- `_pil_bg_frame()` in scenes.py tries diffusion FIRST, falls through to procedural on any error
- `ai_scene_builder.build_scenes(..., awareness="")` — awareness="" disables diffusion (old call sites unaffected)
- `video_agent.build_open_scenes()` passes `awareness=req.awareness` → scenes get diffusion_meta populated
- Singleton pattern in maxcore_diffusion.py: `_get_pipeline()` lazily initialises once, reuses

## torch import pattern in maxcore_diffusion.py
- Top-level: `import torch as _torch  # availability probe`
- Inside methods that actually need torch: `import torch` + `import torch.nn as nn` locally
- This avoids F821 (name not defined) + F401 (imported but unused) conflict

**Why SDEdit prior matters:** Without it, the DiT (randomly initialized) generates pure noise. With it, every frame starts from a RCGS-grounded real frame at 40% noise — output is always visually grounded even before training. As training runs on pdim data, the DiT learns to actually stylise the frames meaningfully.

## Checkpoint persistence
- Weights saved/loaded from `uploads/diffusion/{vae,dit,conditioner}.pt`
- `MaxCoreDiffusionPipeline.save_checkpoints()` — call from training loop when ready
- Missing checkpoints → random init (silent, non-fatal)
