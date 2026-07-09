---
name: RTA path tracer next-event estimation
description: How direct light sampling (NEE) is done in the IRC path tracer and the estimator rules that keep it unbiased
---

# NEE in the RTA-1 image path tracer

The path tracer (`ai_model/rta/image/path_tracer.py`) samples emitter spheres
directly each bounce (next-event estimation) instead of relying on diffuse
bounces randomly striking them. This cut the production sample count from 20 spp
to 6 spp at cleaner-or-equal quality.

## The estimator contract (do not break — it keeps NEE unbiased)
- **Emission is added ONLY on the primary (camera) ray** (`_bounce == 0`). For
  every deeper bounce a BSDF ray hitting an emitter must NOT add emission — that
  light–path is already counted by the NEE shadow ray at the *previous* surface.
  Adding it again double-counts.
- **Area lights are sampled ONLY by NEE; the sky/environment ONLY by BSDF
  sampling** (the miss term). Each light is sampled by exactly one strategy, so
  no MIS is needed and there is no double count.
- Emitter = any sphere with non-zero emission; NEE loops over those.

## Sampling math
Cone / solid-angle sampling of each emitter sphere (the sphere exactly subtends
a cone of half-angle θmax where sinθmax = r/d, so every in-cone direction hits
it). With uniform-cone pdf `1/(2π(1-cosθmax))` the Lambertian direct estimate
collapses to `albedo · Lᵉ · cosθ_surface · 2·(1-cosθmax) · V` — no 1/dist²
geometry term (it is folded into the solid-angle pdf). `V` = shadow-ray
visibility: cast from `P + N·eps`, the sample counts only if the ray's first
hit is that light's own sphere id.

**Why:** the scene was already engineered for low variance (large dim area
lights + sky dome), so NEE's win is real but modest (~4× fewer samples for equal
RMSE, not orders of magnitude). Verified unbiased: old (no-NEE) vs new high-spp
signed-mean diff sits at the old-vs-old noise floor.

## How to apply
- All shadow-ray intersection stays on `_intersect` (Digital-GPU GEMMs) — never
  raw np.matmul. Batch all lights' shadow rays into ONE `_intersect` per bounce.
- Determinism: NEE draws from the same per-sample `np.random.Generator` in fixed
  order (loop lights in order); seed via `stable_seed`, never `hash()`.
- Production call site is `server.py` `/api/generate/image` (samples=6). Node/api
  defaults (samples=4) are unchanged.
