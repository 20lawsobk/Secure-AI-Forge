---
name: Dynamic batching coalescer
description: How/why cross-request generate() batching works on the in-house CPU transformer, the parity guarantee, the threading contract, and how to enable it.
---

# Cross-request dynamic batching (generate coalescer)

Opt-in feature that collapses concurrent `CreativeModel.generate` calls into ONE
batched forward to raise CPU throughput. Lives in
`ai_model/dynamic_batching.py`; batched path is `CreativeModel.generate_batch_rows`;
padding-mask support is threaded through `transformer.py`
(prefill/decode_one + attention) as an optional `key_padding_mask`.

## The rules (be consistent with these)

- **One batcher thread owns the model.** A single daemon thread is the only
  thing that runs batched inference. It therefore must NOT acquire
  `INFERENCE_GATE`. Callers (already on run_in_executor worker threads) submit a
  row and block on an Event WITHOUT holding the gate.
  **Why:** if the batcher took the gate it would deadlock against gate-holding
  callers / oversubscribe the 4 cores. **How to apply:** any *new* model-touching
  path you add must either route through the coalescer or stay single-stream —
  do not run a second concurrent forward alongside the batcher.

- **Parity is logits-exact, not always text-exact.** Left-padding + a
  `key_padding_mask` that drops PAD keys at every layer makes each row's per-step
  logits identical to running it standalone (RoPE is relative, so a uniform
  left-shift is invisible). Verified ~1.6e-6 fp32. Text is byte-identical only
  for B=1 / greedy; for stochastic B>1 it's the SAME sampling distribution, not
  the same string (batched rows consume the global RNG in batch order).
  **How to apply:** never advertise "identical output" for B>1; say
  distribution-parity.

- **Mask with -1e9, not -inf.** A fully-masked PAD query row under softmax(-inf)
  yields NaN; -1e9 keeps it finite. **Why:** ragged batches always have such rows.

## Safety model

- Flag OFF (`AI_DYNAMIC_BATCHING` unset) ⇒ `generate` is left untouched and
  behavior is byte-for-byte identical. `install()` returns None.
- Total fallback: any batch error / count mismatch ⇒ per-row retry through the
  original single-sequence `generate`. A timed-out caller marks its request
  `abandoned` and drops it from the queue under the same lock the batcher uses to
  claim a batch, so the batcher never spends a forward (or a fallback) on a row
  nobody is waiting for.

## Enabling

`AI_DYNAMIC_BATCHING=1` (plus optional `AI_BATCH_MAX`, `AI_BATCH_WINDOW_MS`,
`AI_BATCH_TIMEOUT_S`). Requires a server restart to load. Keep conservative
defaults (max_batch 8, small window); high `AI_BATCH_MAX` can OOM since the
memory cap is a heuristic (~0.15 GB/slot), not length-aware.
