---
name: GPU kernel redesign (flash attention + im2col conv)
description: HyperSIMDCore's flash_attention and conv2d/conv3d are real tiled/vectorized algorithms; the honest CPU tradeoffs and the regressions to avoid re-introducing.
---

`HyperSIMDCore.flash_attention` is a TRUE tiled FlashAttention (online softmax:
running max `m`, denom `l`, unnormalized accumulator `O`; blocks over K/V; the
full `[B,Tq,Tk]` score matrix is never materialized). `conv2d`/`conv3d` use
vectorized im2col via `np.lib.stride_tricks.as_strided` + one batched `matmul`.

**Measured on this CPU host (not a claim — a benchmark):**
- Flash vs naive attention: numerically identical (~1e-15); score memory
  O(T·block) vs O(T²) → 4–16× less as T grows; wall-clock comparable, and
  actually *faster* at long T (e.g. T=2048: ~293ms vs ~417ms) because the full
  score matrix thrashes cache while blocks stay cache-resident.
- Strided im2col vs the old Python patch-loop im2col: 1.4–2.3× faster, same result.

**Why this matters:** these are genuine, measurable wins over the previous
kernels — the earlier `flash_attention` was mislabeled (it materialized the full
score matrix) and conv used a Python patch loop. The improvement is real; the
speed of the *host* is unchanged (still numpy/BLAS on CPU).

**Regressions to never re-introduce (caught in review):**
- Flash must iterate over `Tk = K.shape[1]`, NOT `Tq = Q.shape[1]` — iterating
  over Tq breaks cross-attention (silently drops keys or hits empty-block
  `max` when Tk≠Tq). Validate `K.len == V.len`.
- `block_size` is now operational (was a no-op) — clamp `max(1, min(bs, Tk))` so
  legacy callers passing 0 don't hit an invalid `range` step.
- Conv output dtype must follow the input (`W_col.astype(X.dtype)`), not be
  forced to float32 — direct callers depend on the dtype contract.
- First-block `m=-inf` gives `exp(m-m_new)=0` (not NaN); causal uses finite -1e9
  (matches prior behavior), which is fine at real score scales.

**How to apply:** guard all four with `ai_model/gpu/tests/test_kernels.py`
(cross-attention Tk≠Tq, T%block≠0, causal, block_size clamp, conv dtype). Kernel
edits need a model-server reload to go live.

**Rejected: Winograd F(2,3) 3x3 conv.** Measured on this CPU it is 11-18x SLOWER
than im2col+BLAS (vectorized ~552ms, per-tile ~870ms vs im2col ~49ms), despite
2.25x fewer theoretical multiplies. Fewer FLOPs ≠ faster: a single large BLAS
GEMM already saturates the CPU, and Winograd's 4 transform passes add memory
traffic that dominates. Winograd only wins on hardware where multiplies dominate
AND transforms are fused (GPU/DSP/hand-tuned). Do NOT integrate it here.

**Accepted: fused row-dot via einsum.** `np.einsum("ij,ij->i", A, B)` is ~3.5x
faster than `np.sum(A*B, axis=1)` (and `np.dot` ~4.6x faster than `np.sum(x*y)`
for 1D) because it skips the temporary and hits a fused/BLAS reduction. Applied
to the path tracer's per-bounce batched dot products (ai_model/rta/image).
**Why:** the temp array + separate reduction is pure overhead; einsum fuses them.
**How to apply:** use it for hot batched dot products; it's numerically identical
(RTA render-contract determinism test still byte-identical).
