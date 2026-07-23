---
name: Digital GPU routing — inference, training, sampling
description: Rules for keeping ALL math on the MaxCore self-contained stack; no Replit base-env CPU paths allowed
---

# Digital GPU Routing Contract

The MaxCore stack is self-contained and must handle every compute-heavy operation. No raw `torch.matmul`, `F.softmax`, `nn.Linear`, or `np.matmul` is permitted outside these approved wrappers.

## Inference model hierarchy (server.py `_init_ai_model`)
1. **Preferred**: `HyperCreativeTransformerLM` (has prefill + decode_one KV-cache) — gated on `_hyper_backend is not None`
2. **Secondary fallback**: `HyperCreativeTransformerLM` with a freshly created `HyperGPU` (covers transient init errors)
3. **No CPU fallback**: the old `TransformerLM + torch.compile(aot_eager)` path has been removed

**Why:** `model/status` must show `"device": "digital_gpu"` on every boot. `TransformerLM` routes through Replit's base torch CPU kernels — that is the base env and is banned.

## Training paths
- **BPE scale-up** (`/training/bpe-scale-up`): `HyperTransformerLM` for training, checkpoint saved as `trained_on: hyper_gpu`
- **Digital GPU scale-up** (`/training/digital-gpu-scale-up`): `HyperCreativeTransformerLM` trains AND serves (no weight-transfer to `TransformerLM` step — removed)

**Why:** The old digital GPU path trained on `HyperCreativeTransformerLM` then copied weights into a plain `TransformerLM` for "serving". That CPU transfer step defeated the purpose — now trained model serves directly.

## Sampling softmax (creative_model.py)
- `_gpu_softmax()` and `_gpu_log_softmax()` are module-level helpers routing through `HyperSIMDCore.softmax` (SIMD kernel)
- Lazy singleton via `_get_hyper_core()` — created once, reused for every token sample
- F.softmax is the never-raise fallback ONLY inside those helpers, not at call sites

## Audio awareness bridge (server.py `api_generate_audio`)
- `_aw_str_handler` must be set from `_merged_awareness_for(req)` (captured before `build_brief()`), not from raw `req.awareness`
- This ensures live Deezer BPM/genre signals reach genre/mood extraction, not just caller-supplied text

## Drop-in wrappers (all in `ai_model/gpu/hyper_backend.py`)
- `HyperGPULinear` — replaces `nn.Linear` (routes GEMM through HyperSIMDCore)
- `HyperFlashAttention` — replaces manual Q@K softmax@V
- `HyperLayerNorm` — replaces `nn.LayerNorm`
- `HyperGELU`, `HyperSiLU` — activation functions

**How to apply:** Any new model or training code must use these wrappers instead of raw torch.nn equivalents. Create a `HyperGPUBackend` instance and use `backend.linear(in, out)`, `backend.flash_attention(dim, heads)`, etc.
