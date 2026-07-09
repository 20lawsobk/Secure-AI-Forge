---
name: RTA-1 rendering fabric
description: The unified node-graph render fabric (image path tracer / video grade / audio spectral) and how it routes through the Digital GPU
---

# RTA-1 (Unified Rendering Trinity Architecture)

A node-graph render fabric in `ai_model/rta/` that unifies the three generation
mediums behind one UMRF graph + deterministic scheduler, then lifts fidelity per
medium. Built because the three renderers were pure PIL/ffmpeg/sample-selection.

## Shape
- `fabric/` — MediaState/FrameState, NodeGraph + `topological_sort` (Kahn, cycle-raising),
  `RTACompute` (compute), `UMRFScheduler` + `register_node`, bounded `NodeCache`.
- Domain node sets register into the scheduler at import: **IRC** (image path tracer),
  **VRC** (video colour grade), **ARC** (audio spectral). `ai_model.rta.api` builds the
  graphs and is the entrypoint the server calls; importing it registers all nodes.

## The non-negotiable rule: everything heavy runs on the self-contained Digital GPU
All heavy batched linear algebra routes through `RTACompute.gemm`, which dispatches to
`ai_model.gpu.digital_gpu.DigitalGPU.gemm` (tiled SIMD, NumPy in/out) — **never** raw
`np.matmul`/BLAS. A process-wide GEMM counter (`global_op_counts`) proves it; verify via
`GET /api/rta/status?selftest=1` (watch `digital_gpu_ops` climb).
**Why:** the whole project runs on the in-house Digital GPU; a BLAS shortcut here would
break that contract and make the fabric inconsistent with the rest of the system.
- Path tracer: camera-basis transform + batched ray/sphere dot products (`D·C`, `O·C`) are GEMMs.
- Video grade: the 3×3 creative colour matrix is a GEMM (`pixels[N,3] @ M.T`).
- Audio: STFT forward + inverse DFT are GEMMs against precomputed cos/sin bases.

## Path tracer noise control (learned the hard way)
Small, intensely-bright emitters = Monte-Carlo fireflies/grain. Fix was NOT brute-force
samples but: **large, dimmer area lights + a non-trivial sky dome as the primary
illuminant** (low variance), plus a per-sample firefly **clamp** and a 3×3 **median**
denoise before tonemap. ~20 spp @ 384px then reads clean (~4-5s CPU). Endpoint default is
20 spp. See `image/scene_builder.py` (lighting) and `image/path_tracer.py` (clamp/median).

## Determinism
Never seed with the built-in `hash()` — it is salted per process (`PYTHONHASHSEED`) and
breaks cross-restart determinism. Use `scene_builder.stable_seed(*parts)` (blake2b-derived).
**How to apply:** any new RTA seed derivation (scenes, endpoint seed) must go through a
stable hash, not `hash()`.

## ARC STFT/OLA tail
Frame the signal so the final partial window is covered (pad to
`(n_frames-1)*hop+win`), overlap-add over the padded length, then trim to the original
length — otherwise the reconstructed tail decays toward zero on non-hop-aligned inputs.

## Wiring (all additive, explicit fallbacks, no silent fakes)
- Image `/api/generate/image`: opt-in `render_engine="pathtraced"` (or env `RTA_IMAGE_ENGINE`);
  renders a path-traced background then composites the existing PIL poster typography over it
  via `ImageRequest.background`. Falls back to procedural PIL on any failure (logged). Meta
  engine tag `rta-irc-pathtraced-v1`.
- Video: VRC grade default-on (env `RTA_VIDEO_GRADE`, `1`), applied to the background still in
  `ai_model/video/scenes.py::_render_pil_based`; explicit fallback to the ffmpeg grade preset.
- Audio: ARC opt-in (env `RTA_AUDIO_SPECTRAL=1`) as a post-encode pass in
  `_render_audio_from_dataset` (`_arc_spectral_clean_file`); dataset must be seeded to run.
