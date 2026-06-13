---
name: Video scene text generation
description: How scene text is generated for multi-scene videos and why the dataset sampler replaced model inference
---

## Rule
Use `dataset_sampler.sample_all_scenes()` as the primary (and only) path for video scene text in `plan()`. Do NOT use `model.generate()` or `model.generate_batch()` for this.

**Why:** The transformer is undertrained — it echoes the idea snippet followed by training-vocabulary noise. Dataset sampling from `synthetic.py` phrase banks produces immediately meaningful, personalised text that matches the "datasets" source contract the product expects.

**How to apply:** `plan()` in `video_agent.py` calls `_sample_all_scenes(scene_sequence, idea, genre, tone, platform, artist_name)` which returns `(Dict[int,str], "datasets")`. No inference, no memory spike, always succeeds for any scene count.

## generate_batch() — exists but not in the hot path
`CreativeModel.generate_batch()` (micro-batched, chunk_size=4, max_new_tokens=30) was implemented and works correctly. It is the right inference method when the model is well-trained. Parameters were chosen to keep KV-cache peak < 110 MB per chunk (safe on the 8 GB host with 2 uvicorn workers at 84% idle baseline). Do NOT use max_new_tokens=200 for batched generation — KV-cache grows to 1.3 GB and OOM-kills the worker.

## 2-worker cap
`server.py` caps uvicorn at `min(cpu_count, 2)` workers. Each loaded model instance ≈ 1.7 GB; 2 × 1.7 + 1 GB OS = 4.4 GB idle on 8 GB host, leaving 1.3 GB headroom. This cap must be preserved.
