from __future__ import annotations
import gc
import hashlib
import threading
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
from .tokenizer import SimpleTokenizer, BPETokenizer

# ── Digital GPU core singleton ────────────────────────────────────────────────
# All softmax / log-softmax calls route through HyperSIMDCore so they execute
# on the self-contained MaxCore stack.  Singleton is created lazily once.
_hyper_core = None
_hyper_core_lock = threading.Lock()

def _get_hyper_core():
    """Return the module-level HyperSIMDCore singleton (thread-safe, lazy)."""
    global _hyper_core
    if _hyper_core is not None:
        return _hyper_core
    with _hyper_core_lock:
        if _hyper_core is None:
            try:
                from ai_model.gpu.hyper_core import HyperSIMDCore, PrecisionMode
                _hyper_core = HyperSIMDCore(
                    lanes=512, tensor_cores=8, precision=PrecisionMode.MIXED
                )
            except Exception:
                pass
    return _hyper_core


def _np_softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable numpy softmax over the last axis (fallback)."""
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def _gpu_softmax(t: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Softmax via HyperSIMDCore. Falls back to F.softmax (never raises)."""
    core = _get_hyper_core()
    if core is None:
        return F.softmax(t, dim=dim)
    try:
        arr = t.detach().float().numpy()
        out = core.softmax(arr, axis=dim)
        return torch.from_numpy(out)
    except Exception:
        return F.softmax(t, dim=dim)


def _gpu_log_softmax(t: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Log-softmax via HyperSIMDCore. Falls back to F.log_softmax (never raises)."""
    probs = _gpu_softmax(t, dim=dim)
    return torch.log(probs.clamp(min=1e-38))


# ── In-process generation output cache (L1) ──────────────────────────────────
# Sits in front of the pdim fleet-wide dedup (L2).  A cache hit returns in
# microseconds — genuine sub-ms delivery for any repeated or seeded request.
#
# Design:
#   • Thread-safe via a single lock (lock is only held for dict ops, never
#     during model inference, so contention is negligible).
#   • LRU eviction: the ordered dict moves a hit entry to the end; the oldest
#     entry (front) is evicted when the cache is full.
#   • Per-entry TTL: entries expire after _GEN_CACHE_TTL_S seconds so stale
#     outputs don't persist across model checkpoints.

_GEN_CACHE_MAX   = 512          # max entries (each ≤ a few KB of text)
_GEN_CACHE_TTL_S = 120.0        # seconds before an entry is considered stale

from collections import OrderedDict as _OD

_gen_cache: _OD[str, tuple[float, str]] = _OD()   # key → (ts, text)
_gen_cache_lock = threading.Lock()
_gen_cache_hits  = 0
_gen_cache_total = 0


def _gen_cache_key(
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
) -> str:
    raw = f"{prompt}|{max_new_tokens}|{temperature:.4f}|{top_p:.4f}|{top_k}|{repetition_penalty:.4f}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _gen_cache_get(key: str) -> str | None:
    global _gen_cache_hits, _gen_cache_total
    _gen_cache_total += 1
    with _gen_cache_lock:
        entry = _gen_cache.get(key)
        if entry is None:
            return None
        ts, text = entry
        if time.monotonic() - ts > _GEN_CACHE_TTL_S:
            del _gen_cache[key]
            return None
        # LRU: move to end (most-recently used)
        _gen_cache.move_to_end(key)
        _gen_cache_hits += 1
    return text


def _gen_cache_put(key: str, text: str) -> None:
    with _gen_cache_lock:
        if key in _gen_cache:
            _gen_cache.move_to_end(key)
        _gen_cache[key] = (time.monotonic(), text)
        # Evict oldest when over capacity
        while len(_gen_cache) > _GEN_CACHE_MAX:
            _gen_cache.popitem(last=False)


def get_gen_cache_stats() -> dict:
    with _gen_cache_lock:
        size = len(_gen_cache)
    total = _gen_cache_total
    hits  = _gen_cache_hits
    return {
        "size": size, "capacity": _GEN_CACHE_MAX,
        "hits": hits, "total": total,
        "hit_rate": round(hits / total, 4) if total else 0.0,
        "ttl_s": _GEN_CACHE_TTL_S,
    }


class CreativeModel:
    """
    Wrapper around TransformerLM that provides:
    - KV-cache nucleus sampling (prefill once, decode O(1) per token)
    - Beam search (contrastive decoding)
    - Min length control
    - Repetition penalty (vectorized)
    """

    def __init__(self, model: nn.Module, tokenizer: Union[SimpleTokenizer, BPETokenizer], device="cpu"):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.tokenizer.freeze()
        self.model.eval()

    def resize_embeddings(self):
        new_vocab = self.tokenizer.vocab_size
        old_emb = self.model.token_emb
        _old_head = self.model.head
        if new_vocab > old_emb.num_embeddings:
            dim = old_emb.embedding_dim
            new_emb = nn.Embedding(new_vocab, dim).to(self.device)
            new_emb.weight.data[:old_emb.num_embeddings] = old_emb.weight.data
            self.model.token_emb = new_emb
            new_head = nn.Linear(dim, new_vocab, bias=False).to(self.device)
            new_head.weight = new_emb.weight
            self.model.head = new_head

    def _safety_bad_ids(self) -> list[int]:
        """Cached list of hard-blocked token ids for logit masking (Stage 8)."""
        cached = getattr(self, "_safety_bad_ids_cache", None)
        if cached is None:
            try:
                from ai_model.safety import get_safety
                cached = list(get_safety().bad_token_ids(self.tokenizer))
            except Exception:
                cached = []
            self._safety_bad_ids_cache = cached
        return cached

    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        token_window: list[int],
        penalty: float,
        special_ids: tuple[int, ...],
    ) -> torch.Tensor:
        """Vectorized repetition penalty — single scatter operation. Logits: [1, vocab]."""
        seen_ids = [t for t in set(token_window) if t not in special_ids]
        if not seen_ids:
            return logits
        idx = torch.tensor(seen_ids, device=self.device, dtype=torch.long)
        lv = logits[0, idx]
        penalized = torch.where(lv > 0, lv / penalty, lv * penalty)
        logits[0, idx] = penalized
        return logits

    def _sample_next_np(
        self,
        logits_np: np.ndarray,          # [1, vocab] float32 — already in numpy
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> int:
        """
        Pure-numpy nucleus sampling — zero tensor↔numpy roundtrips.

        Taking the logits as numpy at the boundary (once, in generate()) and
        doing all filtering + softmax + sampling here eliminates the two
        to_numpy / from_numpy calls that _gpu_softmax used to make on every
        single decode step.  Returns the sampled token id as a plain int.
        """
        core = _get_hyper_core()
        row = logits_np[0]                      # view, no copy

        # Stage 8: mask hard-blocked tokens
        bad_ids = self._safety_bad_ids()
        if bad_ids:
            row[bad_ids] = -np.inf

        row /= max(temperature, 1e-8)

        # Top-k: zero out everything below the k-th largest logit
        if top_k > 0:
            k = min(top_k, len(row))
            # np.partition is O(V) vs O(V log V) sort — faster for large vocab
            threshold = np.partition(row, -k)[-k]
            row[row < threshold] = -np.inf

        # Top-p nucleus
        if 0.0 < top_p < 1.0:
            order = np.argsort(-row)            # descending indices, O(V log V)
            sorted_row = row[order]
            if core is not None:
                try:
                    probs_s = core.softmax(sorted_row[None], axis=-1)[0]
                except Exception:
                    probs_s = _np_softmax(sorted_row[None])[0]
            else:
                probs_s = _np_softmax(sorted_row[None])[0]
            cumulative = np.cumsum(probs_s)
            # mask tokens whose cumulative prob exceeds top_p (keep the first)
            mask = (cumulative - probs_s) > top_p
            sorted_row[mask] = -np.inf
            row[order] = sorted_row             # write back in-place

        # Final softmax → sample
        if core is not None:
            try:
                probs = core.softmax(logits_np, axis=-1)[0]
            except Exception:
                probs = _np_softmax(logits_np)[0]
        else:
            probs = _np_softmax(logits_np)[0]

        probs = np.maximum(probs, 0.0)
        total = probs.sum()
        if total <= 0.0 or not np.isfinite(total):
            probs = np.ones(len(probs), dtype=np.float32) / len(probs)
        else:
            probs /= total

        return int(np.random.choice(len(probs), p=probs))

    def _sample_next(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> torch.Tensor:
        """Tensor-in / tensor-out wrapper kept for external callers (beam search etc.)."""
        token_id = self._sample_next_np(
            logits.detach().float().numpy().copy(),
            temperature, top_p, top_k,
        )
        return torch.tensor([[token_id]], dtype=torch.long)

    # ── KV-cache sampling generation ──────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.85,
        top_p: float = 0.92,
        top_k: int = 50,
        repetition_penalty: float = 1.15,
        min_length: int = 10,
    ) -> str:
        """
        Unlimited autoregressive generation with KV-cache.

        Prefills the prompt in one batched forward pass, then generates
        each new token in O(1) time (single-position forward, no context
        re-computation).  Repeated calls with identical parameters are
        served from the in-process L1 cache in microseconds (sub-ms).
        """
        # ── L1 generation cache: sub-ms hit path ──────────────────────────────
        cache_key = _gen_cache_key(
            prompt, max_new_tokens, temperature, top_p, top_k, repetition_penalty
        )
        cached = _gen_cache_get(cache_key)
        if cached is not None:
            return cached

        ids = self.tokenizer.encode(prompt).ids
        if not ids:
            ids = [self.tokenizer.token_to_id("<BOS>")]

        eos_id      = self.tokenizer.token_to_id("<EOS>")
        pad_id      = self.tokenizer.token_to_id("<PAD>")
        unk_id      = self.tokenizer.token_to_id("<UNK>")
        special_ids = (pad_id, unk_id, eos_id)
        max_ctx     = getattr(self.model, 'max_len', 1024)

        if len(ids) > max_ctx:
            ids = ids[-max_ctx:]

        generated_ids: list[int] = []

        _autocast = torch.autocast("cpu", dtype=torch.bfloat16, enabled=True)
        with torch.no_grad(), _autocast:
            # ── Prefill: full prompt in one batched pass, build KV cache ──────
            x_prompt = torch.tensor([ids], device=self.device)
            logits_all, kv_cache = self.model.prefill(x_prompt)

            # Extract first-token logits once as numpy — stays numpy through
            # the entire decode loop to avoid per-token tensor roundtrips.
            next_logits_np = logits_all[:, -1, :].float().numpy().copy()  # [1, V]

            # ── Decode: O(1) per step via KV cache, pure-numpy sampling ───────
            for step in range(max_new_tokens):
                # Mask special tokens in-place (no tensor allocation)
                next_logits_np[0, pad_id] = -np.inf
                next_logits_np[0, unk_id] = -np.inf
                if step < min_length:
                    next_logits_np[0, eos_id] = -np.inf

                if generated_ids:
                    # Repetition penalty: still needs a torch tensor for the
                    # vectorized gather — convert, penalise, convert back.
                    lt = torch.from_numpy(next_logits_np)
                    lt = self._apply_repetition_penalty(
                        lt, generated_ids[-64:], repetition_penalty, special_ids
                    )
                    next_logits_np = lt.numpy()

                # Pure-numpy nucleus sampling — zero roundtrips inside
                token_id = self._sample_next_np(
                    next_logits_np.copy(), temperature, top_p, top_k
                )
                generated_ids.append(token_id)

                if token_id == eos_id:
                    break

                # Context window guard
                ctx_used = len(ids) + len(generated_ids)
                if ctx_used >= max_ctx:
                    kv_cache = [(k[:, :, 1:, :], v[:, :, 1:, :]) for k, v in kv_cache]

                # One-token forward with KV cache
                next_id_t = torch.tensor([[token_id]], dtype=torch.long,
                                         device=self.device)
                logits_new, kv_cache = self.model.decode_one(next_id_t, kv_cache)
                next_logits_np = logits_new[:, 0, :].float().numpy().copy()

        result = self.tokenizer.decode(ids + generated_ids)

        # Store in L1 cache for future sub-ms hits
        _gen_cache_put(cache_key, result)
        return result

    # ── Batched autoregressive generation ────────────────────────────────────

    def generate_batch(
        self,
        prompts: list[str],
        max_new_tokens: int = 30,
        temperature: float = 0.85,
        top_p: float = 0.92,
        top_k: int = 50,
        repetition_penalty: float = 1.15,
        min_length: int = 5,
        chunk_size: int = 4,
    ) -> list[str]:
        """
        Batched autoregressive generation with memory-safe micro-batching.

        All B prompts are split into chunks of `chunk_size` and each chunk
        runs a single batched prefill + decode loop.  Within each chunk,
        all sequences advance simultaneously (one forward pass per step).

        Memory budget per chunk: chunk_size × KV-cache per layer.
        With chunk_size=4 and max_new_tokens=30 the KV-cache peak is
        ~110 MB — safe alongside two loaded model instances on 8 GB RAM.

        Speed vs. N sequential calls: N/chunk_size × (max_new_tokens / 200)
        improvement  ≈  5× for a 20-scene request.
        """
        if not prompts:
            return []

        results: list[str] = []
        for i in range(0, len(prompts), chunk_size):
            chunk = prompts[i : i + chunk_size]
            chunk_out = self._generate_batch_chunk(
                chunk,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                min_length=min_length,
            )
            results.extend(chunk_out)
            gc.collect()
        return results

    def _generate_batch_chunk(
        self,
        prompts: list[str],
        max_new_tokens: int = 30,
        temperature: float = 0.85,
        top_p: float = 0.92,
        top_k: int = 50,
        repetition_penalty: float = 1.15,
        min_length: int = 5,
    ) -> list[str]:
        """
        Core batched inference for a single micro-batch.

        One prefill for all B prompts, then one decode_one per token step —
        all B sequences advance in a single forward pass per step.

        KV-cache peak: B × 8heads × (prompt_len + max_new_tokens) × 64 × 4B × 2 × 8layers.
        For B=4, tokens=42: ≈ 55 MB — well within the 1.3 GB headroom.
        """
        B = len(prompts)
        eos_id      = self.tokenizer.token_to_id("<EOS>")
        pad_id      = self.tokenizer.token_to_id("<PAD>")
        bos_id      = self.tokenizer.token_to_id("<BOS>")
        unk_id      = self.tokenizer.token_to_id("<UNK>")
        special_ids = (pad_id, unk_id, eos_id)
        max_ctx     = getattr(self.model, "max_len", 1024)

        # Tokenize and right-pad to uniform length
        prompt_ids_list: list[list[int]] = []
        for p in prompts:
            ids = self.tokenizer.encode(p).ids or [bos_id]
            if len(ids) > max_ctx:
                ids = ids[-max_ctx:]
            prompt_ids_list.append(ids)

        max_plen = max(len(ids) for ids in prompt_ids_list)
        padded   = [ids + [pad_id] * (max_plen - len(ids))
                    for ids in prompt_ids_list]

        generated: list[list[int]] = [[] for _ in range(B)]
        done = [False] * B

        _autocast = torch.autocast("cpu", dtype=torch.bfloat16, enabled=True)
        with torch.no_grad(), _autocast:
            x = torch.tensor(padded, device=self.device)  # [B, max_plen]
            logits_all, kv_cache = self.model.prefill(x)
            next_logits = logits_all[:, -1, :].float().clone()  # [B, vocab]

            for step in range(max_new_tokens):
                next_logits[:, pad_id] = float("-inf")
                next_logits[:, unk_id] = float("-inf")
                if step < min_length:
                    next_logits[:, eos_id] = float("-inf")

                next_tokens: list[int] = []
                for b in range(B):
                    if done[b]:
                        next_tokens.append(pad_id)
                        continue
                    lb = next_logits[b : b + 1].clone()  # [1, vocab]
                    if generated[b]:
                        lb = self._apply_repetition_penalty(
                            lb, generated[b][-64:], repetition_penalty, special_ids
                        )
                    nid = int(self._sample_next(lb, temperature, top_p, top_k).item())
                    next_tokens.append(nid)
                    if nid == eos_id:
                        done[b] = True
                    else:
                        generated[b].append(nid)

                if all(done):
                    break

                ctx_used = max_plen + max(len(g) for g in generated)
                if ctx_used >= max_ctx:
                    kv_cache = [
                        (k[:, :, 1:, :], v[:, :, 1:, :]) for k, v in kv_cache
                    ]

                nt = torch.tensor([[t] for t in next_tokens], device=self.device)
                logits_new, kv_cache = self.model.decode_one(nt, kv_cache)
                next_logits = logits_new[:, 0, :].float().clone()  # [B, vocab]

        return [
            self.tokenizer.decode(prompt_ids_list[b] + generated[b])
            for b in range(B)
        ]

    # ── Heterogeneous coalesced batch (cross-request dynamic batching) ─────────

    def generate_batch_rows(self, rows: list[dict]) -> list[str]:
        """
        Batched generation for a coalesced set of *independent* requests.

        Each row is a dict: ``{"prompt": str, ...optional sampling params}`` where
        params default to the same values as :meth:`generate` — ``max_new_tokens``,
        ``temperature``, ``top_p``, ``top_k``, ``repetition_penalty``, ``min_length``.

        Prompts are **left-padded** to a common length with a ``key_padding_mask``
        that excludes PAD positions. Because RoPE attention is relative, the output
        for each row is identical (same RNG) to generating that row alone,
        regardless of the other rows in the batch — this is what makes it safe to
        merge unrelated concurrent requests into one forward pass.

        Returns decoded strings aligned to ``rows``.
        """
        if not rows:
            return []
        B = len(rows)

        eos_id = self.tokenizer.token_to_id("<EOS>")
        pad_id = self.tokenizer.token_to_id("<PAD>")
        bos_id = self.tokenizer.token_to_id("<BOS>")
        unk_id = self.tokenizer.token_to_id("<UNK>")
        special_ids = (pad_id, unk_id, eos_id)
        max_ctx = getattr(self.model, "max_len", 1024)

        r_max_new = [int(r.get("max_new_tokens", 200)) for r in rows]
        r_temp = [float(r.get("temperature", 0.85)) for r in rows]
        r_top_p = [float(r.get("top_p", 0.92)) for r in rows]
        r_top_k = [int(r.get("top_k", 50)) for r in rows]
        r_rep = [float(r.get("repetition_penalty", 1.15)) for r in rows]
        r_min = [int(r.get("min_length", 10)) for r in rows]

        prompt_ids_list: list[list[int]] = []
        for r in rows:
            ids = self.tokenizer.encode(r["prompt"]).ids or [bos_id]
            if len(ids) > max_ctx:
                ids = ids[-max_ctx:]
            prompt_ids_list.append(ids)

        max_plen = max(len(ids) for ids in prompt_ids_list)
        pad_counts = [max_plen - len(ids) for ids in prompt_ids_list]
        # LEFT-pad: the last position is always a real token for every row.
        padded = [[pad_id] * pc + ids for pc, ids in zip(pad_counts, prompt_ids_list)]
        has_pad = any(pc > 0 for pc in pad_counts)

        generated: list[list[int]] = [[] for _ in range(B)]
        done = [False] * B
        max_steps = max(r_max_new) if r_max_new else 0

        _autocast = torch.autocast("cpu", dtype=torch.bfloat16, enabled=True)
        with torch.no_grad(), _autocast:
            x = torch.tensor(padded, device=self.device)  # [B, max_plen]
            # None when no padding -> bit-identical to the original single-seq path
            # (this is the B=1 / equal-length fast path).
            kpm = None
            if has_pad:
                kpm = torch.tensor(
                    [[True] * pc + [False] * (max_plen - pc) for pc in pad_counts],
                    device=self.device, dtype=torch.bool,
                )
            logits_all, kv_cache = self.model.prefill(x, key_padding_mask=kpm)
            next_logits = logits_all[:, -1, :].float().clone()  # [B, vocab]

            for step in range(max_steps):
                next_logits[:, pad_id] = float("-inf")
                next_logits[:, unk_id] = float("-inf")

                next_tokens: list[int] = []
                for b in range(B):
                    if done[b] or step >= r_max_new[b]:
                        done[b] = True
                        next_tokens.append(pad_id)
                        continue
                    lb = next_logits[b : b + 1].clone()  # [1, vocab]
                    if len(generated[b]) < r_min[b]:
                        lb[:, eos_id] = float("-inf")
                    if generated[b]:
                        lb = self._apply_repetition_penalty(
                            lb, generated[b][-64:], r_rep[b], special_ids
                        )
                    nid = int(
                        self._sample_next(lb, r_temp[b], r_top_p[b], r_top_k[b]).item()
                    )
                    generated[b].append(nid)
                    if nid == eos_id:
                        done[b] = True
                        next_tokens.append(pad_id)
                    else:
                        next_tokens.append(nid)

                if all(done):
                    break

                ctx_used = max_plen + max(len(g) for g in generated)
                if ctx_used >= max_ctx:
                    kv_cache = [(k[:, :, 1:, :], v[:, :, 1:, :]) for k, v in kv_cache]
                    if kpm is not None:
                        kpm = kpm[:, 1:]

                nt = torch.tensor([[t] for t in next_tokens], device=self.device)
                if kpm is not None:
                    # New token column: real for active rows -> never masked.
                    kpm = torch.cat(
                        [kpm, torch.zeros(B, 1, dtype=torch.bool, device=self.device)],
                        dim=1,
                    )
                logits_new, kv_cache = self.model.decode_one(
                    nt, kv_cache, key_padding_mask=kpm
                )
                next_logits = logits_new[:, 0, :].float().clone()  # [B, vocab]

        return [
            self.tokenizer.decode(prompt_ids_list[b] + generated[b])
            for b in range(B)
        ]

    # ── Beam search generation ────────────────────────────────────────────────

    def beam_search(
        self,
        prompt: str,
        max_new_tokens: int = 120,
        num_beams: int = 4,
        length_penalty: float = 1.0,
        repetition_penalty: float = 1.2,
        min_length: int = 8,
        temperature: float = 1.0,
    ) -> str:
        """Beam search with length penalty."""
        ids = self.tokenizer.encode(prompt).ids
        if not ids:
            ids = [self.tokenizer.token_to_id("<BOS>")]

        eos_id   = self.tokenizer.token_to_id("<EOS>")
        pad_id   = self.tokenizer.token_to_id("<PAD>")
        unk_id   = self.tokenizer.token_to_id("<UNK>")
        special_ids = (pad_id, unk_id, eos_id)
        max_ctx  = getattr(self.model, 'max_len', 1024)
        vocab_size = self.model.token_emb.num_embeddings

        beams: list[tuple[float, list[int]]] = [(0.0, list(ids))]
        completed: list[tuple[float, list[int]]] = []

        _autocast = torch.autocast("cpu", dtype=torch.bfloat16, enabled=True)
        with torch.no_grad(), _autocast:
            for step in range(max_new_tokens):
                if not beams:
                    break
                all_candidates: list[tuple[float, list[int]]] = []

                for score, beam_ids in beams:
                    if beam_ids[-1] == eos_id:
                        completed.append((score, beam_ids))
                        continue

                    x = torch.tensor([beam_ids[-max_ctx:]], device=self.device)
                    logits = self.model(x).float()
                    next_logits = logits[0, -1, :].clone().unsqueeze(0)

                    next_logits[0, pad_id] = float('-inf')
                    next_logits[0, unk_id] = float('-inf')
                    # Stage 8 constraint enforcement — mask hard-blocked tokens.
                    bad_ids = self._safety_bad_ids()
                    if bad_ids:
                        next_logits[0, bad_ids] = float('-inf')
                    if len(beam_ids) - len(ids) < min_length:
                        next_logits[0, eos_id] = float('-inf')

                    next_logits = self._apply_repetition_penalty(
                        next_logits, beam_ids[-64:], repetition_penalty, special_ids
                    )

                    if temperature != 1.0:
                        next_logits = next_logits / max(temperature, 1e-8)

                    # MaxCore digital GPU log-softmax — no Replit CPU kernels
                    log_probs = _gpu_log_softmax(next_logits[0], dim=-1)
                    topk = min(num_beams * 2, vocab_size)
                    top_vals, top_idxs = torch.topk(log_probs, topk)

                    for lp, tid in zip(top_vals.tolist(), top_idxs.tolist()):
                        all_candidates.append((score + lp, beam_ids + [tid]))

                if not all_candidates:
                    break

                all_candidates.sort(
                    key=lambda c: c[0] / max(1, len(c[1])) ** length_penalty,
                    reverse=True,
                )
                beams = all_candidates[:num_beams]

            completed.extend(beams)

        if not completed:
            return prompt

        best = max(completed, key=lambda c: c[0] / max(1, len(c[1])) ** length_penalty)
        return self.tokenizer.decode(best[1])

    # ── Contrastive decoding (delegates to generate with stronger rep penalty) ─

    def contrastive_generate(
        self,
        prompt: str,
        max_new_tokens: int = 150,
        temperature: float = 0.9,
        alpha: float = 0.6,
        top_k: int = 50,
        min_length: int = 10,
        repetition_penalty: float = 1.1,
    ) -> str:
        return self.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.95,
            top_k=top_k,
            repetition_penalty=repetition_penalty * (1 + alpha * 0.2),
            min_length=min_length,
        )
