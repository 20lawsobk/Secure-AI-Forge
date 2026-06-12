---
name: KV-cache inference
description: TransformerLM inference is O(T²) per call without KV-cache; prefill+decode_one pattern needed for any generation > 20 tokens
---

## Rule
Always use `model.prefill()` + `model.decode_one()` for generation (KV-cache path), not the naïve `model(full_context)` loop.

**Why:** Without KV-cache, each of the 200 autoregressive steps re-runs all 8 transformer layers over the full growing context. On CPU this is ~0.45s/step × 200 steps = 90s per generate() call. With KV-cache (prefill once, decode_one per token), it's ~0.1s/step = 20s for 200 tokens — 4.5× faster.

**How to apply:** `creative_model.py generate()` now uses `self.model.prefill(prompt_tensor)` for the initial pass, then `self.model.decode_one(new_token, kv_cache)` per generated token. The transformer modules (`transformer.py`) expose `prefill()`, `decode_one()`, and layer-level `forward_with_kv()` / `decode_one()` methods. Training still uses the standard `forward()` path unchanged.
