---
name: Generation speed stack
description: All optimizations layered for minimum-latency generation; constraints to preserve when touching creative_model.py or hyper_core.py
---

# Generation Speed Stack

## L1 in-process generation cache (creative_model.py)
- Module-level ordered-dict LRU, 512 entries, 120s TTL, thread-safe
- Key = SHA-256(prompt | max_new_tokens | temperature | top_p | top_k | rep_penalty)
- `_gen_cache_get` / `_gen_cache_put` / `get_gen_cache_stats()` — stats exposed at `/api/gpu/gen-cache/stats`
- Sits in front of the pdim fleet dedup (L2) — cache hit returns in microseconds before any model inference

**Why:** pdim hit is still a network round-trip (~1ms+); in-process dict lookup is ~0.5µs.

**How to apply:** `generate()` checks cache first. Do NOT bypass the cache for admin/internal calls unless you need guaranteed fresh output (add a `skip_cache=True` param if needed).

## Pure-numpy decode loop (creative_model.py `generate`)
- Logits extracted from the model as numpy once at the prefill boundary: `logits_all[:, -1, :].float().numpy().copy()`
- `_sample_next_np(logits_np, ...)` operates entirely in numpy — no `torch.from_numpy` / `.numpy()` inside the loop
- `_sample_next(tensor, ...)` is a thin tensor wrapper kept for external callers (beam search etc.)
- Repetition penalty is the only per-step tensor op (uses vectorized gather); result converted back to numpy immediately

**Why:** The old `_gpu_softmax` did `tensor → .detach().float().numpy() → SIMD → torch.from_numpy()` on every single decode step. That's 3 redundant conversions per token.

**How to apply:** Keep logits as numpy through the decode loop. Any new per-token ops should work on the numpy array, not a torch tensor, unless they require torch grad.

## Flash attention Tq=1 fast-path (hyper_core.py `flash_attention`)
- When `Tq == 1` (every KV-cache decode step): skip the Python tile loop entirely
- Does: `scores = batched_gemm(Q, Kᵀ) * scale` → `softmax via _native.softmax_rows` → `batched_gemm(probs, V)`
- Two GEMM dispatches + one native SIMD softmax vs the old O(Tk/bs) Python iterations

**Why:** During generation, every decode step calls flash_attention with Tq=1. The tile loop had Python overhead proportional to Tk/bs even when there was nothing to tile.

**How to apply:** The shortcut is inside `flash_attention` — callers don't need to change. Don't add a separate `flash_attention_decode` method; keep it unified with a branch on Tq.

## Fused linear+activation (hyper_core.py)
- `linear_gelu(X, W, bias)` — single Python call: GEMM → in-place native SIMD GeLU
- `linear_silu(X, W, bias)` — same pattern with SiLU
- `linear_relu(X, W, bias)` — same pattern with ReLU (numpy maximum, no separate kernel)
- Weight must be passed as W (not Wᵀ); the method handles `.T` internally

**Why:** Feedforward blocks call matmul then activation as two separate Python dispatches. Fusing them keeps intermediate results hot in L2/L3 cache and halves the Python-layer overhead per FF layer.

**How to apply:** Replace `core.gemm(X, W.T) + bias; core.gelu(out)` call pairs with `core.linear_gelu(X, W, bias)` at any new feedforward block. Existing code using separate calls still works; migrate opportunistically.

## Stats endpoint
- `/api/gpu/gen-cache/stats` (GET, requires API key) — returns L1 cache size, hits, total, hit_rate, TTL
- Proxy route: `router.get("/gpu/gen-cache/stats", ...)` in model-proxy.ts
