---
name: HyperGPU training routing (forward + backward)
description: To genuinely route training compute through the in-house Digital GPU, both forward AND backward autograd Functions must call gpu ops; verify via op-counter delta in backward.
---

# Routing real training compute through the Digital GPU (HyperGPU)

To train a model whose *actual* forward+backward compute runs on the in-house
Digital GPU (not native torch), build a torch model out of custom
`torch.autograd.Function`s whose **both** `forward` and `backward` call the
`gpu.*` kernels (gemm / mixed_gemm / softmax / flash_attention / layer_norm /
silu). A common mistake: only the GEMM Function routes backward through the gpu
while attention/layernorm/SiLU `backward` quietly fall back to `torch.bmm /
torch.softmax / torch.sigmoid`. That means the backward pass is NOT backend-routed.

**Why:** the user's requirement is "actual backend functionality, not
simulation" — a review will (correctly) fail the feature if backward runs on
torch even when forward is routed.

**How to apply / verify:**
- The real-path op counter is `gpu.core._total_ops` (NOT `gpu._training_ops`,
  which only increments in the `training_mode=True` shortcut).
- Assert `_total_ops` increases during **backward**, not just forward. At 90M
  params (dim=768,L=12,H=12) a B=1,T=64 step gives ~98 fwd ops and ~855 backward
  ops. A counter-delta test is the honest proof.
- Attention backward: recompute scores with `gpu.gemm` per batch over the
  `[B*H, T, Dh]` flattened tensors, softmax via `gpu.softmax`, then dV = A^T·dO,
  dS = (dA − sum(dA*A))·A, dQ/dK via more `gpu.gemm`. Causal mask uses `-1e9`
  (matches the fused flash forward kernel's finite-mask convention, not `-inf`).
- LayerNorm/SiLU backward are elementwise/reduction — no matmul kernel exists, so
  compute in the backend's numpy domain and bump `gpu.core._total_ops += 1` to
  keep it measurable; save `ctx.gpu` in forward so backward can reach it.

## Flash kernel can't do attention-matrix dropout
The fused Digital-GPU flash-attention kernel computes softmax internally, so you
cannot inject dropout on the attention-probability matrix (as `TransformerLM`
does). Apply dropout to the attention **output** instead and document it as an
intentional deviation. It does not affect weight compatibility (dropout has no
params), so a hyper-trained checkpoint still transfers 1:1 into `TransformerLM`.

## Perf reality
Real routing (tiled python-loop tensor-core matmul) is ~100–1000x slower than
torch. A 90M-param fwd+bwd step at B=1,T=64 is ~30s, peak RSS ~1.2GB. Full-scale
training is HOURS/epoch — keep bounded defaults + a `max_samples` cap, and be
transparent about the tradeoff. Serving still uses the fast KV-cache
`TransformerLM` with the transferred weights.
