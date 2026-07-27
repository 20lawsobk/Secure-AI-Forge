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

## Per-array digest cache (pocket_multiply.py)
- `_array_digest(a)` caches SHA-256 per array by `(ctypes.data, shape, dtype.str)` — memory address + shape + dtype
- Weight matrix bytes (e.g. 3 MB QKV matrix) hashed ONCE at first call; all subsequent forward passes use the cached hex string
- Combined digest = SHA-256 of "|".join(per_array_digests) — still hash-secure but fast even for large weights
- `get_digest_cache_stats()` exposed at `/api/gpu/digest-cache/stats`
- NOT valid for in-place training ops (activations vary each call → cache miss → normal hash path → no correctness risk)

## Prefix KV cache (hyper_creative_transformer.py)
- Module-level dict `_PREFIX_KV_CACHE`: key = SHA-256 of first min(T, 256) token IDs
- Value = zlib-compressed pickle of {h: tensor, kv: list[(K,V)], ts: float, prefix_len: int}
- Max 32 entries, 600s TTL, zlib level 1 compression (~4-8x smaller than raw tensors)
- Two fast-paths in `prefill()`:
  1. Full match (p_len == T): return cached h and kv immediately — zero forward passes
  2. Partial match (p_len < T): load prefix KV, run suffix only, cat prefix+suffix KV along time axis
- Store after every full prefill (min 4 tokens) keyed to min(T, 256) prefix length
- `get_prefix_kv_stats()` exposed at `/api/gpu/prefix-kv/stats`
- EVAL MODE ONLY: `if not self.training` guard prevents cache use during training

## Infinite replica namespace pool (replica_scaler.py)
- `ReplicaPool`: N `PocketDimension` instances sharing one orchestrator (cross-replica dedup is free)
- Default N = max(2, min(cpu_count, 8)) — scales to core count
- Round-robin dispatches GEMM requests across replicas; seed log (last 64 GEMMs) warms new replicas
- `get_replica_pool()` singleton; `scale_to(target)` grows pool; never shrinks
- `/api/gpu/replica-pool/stats` (GET) + `/api/gpu/replica-pool/grow` (POST, admin)

## Weight pre-registration (warmup_pocket on HyperCreativeTransformerLM)
- `warmup_pocket()` walks all `HyperLinearNL` modules, issues one synthetic GEMM per weight
- Called during `_warm_digital_gpu()` at startup — first real forward pass sees pocket hits
- Returns `{layers_warmed: int, weight_matrices: int}`

## Stats endpoints (all require API key unless noted)
- `/api/gpu/gen-cache/stats` — L1 in-process generation output cache
- `/api/gpu/digest-cache/stats` — per-array SHA-256 digest cache hit count
- `/api/gpu/prefix-kv/stats` — prefix KV cache hits/misses/hit_rate
- `/api/gpu/replica-pool/stats` — replica count, round-robin position, hit rate
- `/api/gpu/replica-pool/grow` (POST, admin) — grow pool to target replica count

## Image L1 response cache + warm pools
- /api/generate/image has an in-process L1 response cache (TTL 600s, bounded, key = canonical JSON of request fields + awareness digest). Hits validate backing PNGs exist (evict+re-render if deleted) and return deep copies. Renders are deterministic per inputs, so caching loses nothing.
- Boot warmers: content (25 slots), audio (5 genres × 30s+180s), image (3 platforms). Image warm calls the handler directly via asyncio.run in a daemon thread — safe alongside uvicorn.
- Cached image hits still measure ~500ms end-to-end through the Node proxy — remaining latency is proxy/cluster overhead, not Python.
