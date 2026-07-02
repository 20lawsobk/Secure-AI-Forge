from __future__ import annotations
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
from .tokenizer import SimpleTokenizer, BPETokenizer


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

    def _sample_next(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> torch.Tensor:
        """Apply temperature + top-k + nucleus sampling. Returns next token id [B, 1]."""
        logits = logits / max(temperature, 1e-8)

        if top_k > 0:
            top_k_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < top_k_vals[:, -1].unsqueeze(-1)] = float('-inf')

        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cumulative = torch.cumsum(probs, dim=-1)
            mask = (cumulative - probs) > top_p
            sorted_logits[mask] = float('-inf')
            logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

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
        re-computation). Generation stops when the model emits <EOS> or
        max_new_tokens is reached — no artificial cap otherwise.
        """
        ids = self.tokenizer.encode(prompt).ids
        if not ids:
            ids = [self.tokenizer.token_to_id("<BOS>")]

        eos_id   = self.tokenizer.token_to_id("<EOS>")
        pad_id   = self.tokenizer.token_to_id("<PAD>")
        unk_id   = self.tokenizer.token_to_id("<UNK>")
        special_ids = (pad_id, unk_id, eos_id)
        max_ctx  = getattr(self.model, 'max_len', 1024)

        # Truncate prompt if needed
        if len(ids) > max_ctx:
            ids = ids[-max_ctx:]

        generated_ids: list[int] = []

        _autocast = torch.autocast("cpu", dtype=torch.bfloat16, enabled=True)
        with torch.no_grad(), _autocast:
            # ── Prefill: full prompt in one batched pass, build KV cache ──────
            x_prompt = torch.tensor([ids], device=self.device)
            logits_all, kv_cache = self.model.prefill(x_prompt)

            # Logits for the first token to generate (after the prompt)
            next_logits = logits_all[:, -1, :].float().clone()  # [1, vocab]

            # ── Decode: O(1) per step via KV cache ────────────────────────────
            for step in range(max_new_tokens):
                next_logits[:, pad_id] = float('-inf')
                next_logits[:, unk_id] = float('-inf')
                if step < min_length:
                    next_logits[:, eos_id] = float('-inf')

                if generated_ids:
                    next_logits = self._apply_repetition_penalty(
                        next_logits, generated_ids[-64:], repetition_penalty, special_ids
                    )

                next_id   = self._sample_next(next_logits, temperature, top_p, top_k)
                token_id  = int(next_id.item())
                generated_ids.append(token_id)

                if token_id == eos_id:
                    break

                # Context window guard: drop oldest KV entries when full
                ctx_used = len(ids) + len(generated_ids)
                if ctx_used >= max_ctx:
                    kv_cache = [(k[:, :, 1:, :], v[:, :, 1:, :]) for k, v in kv_cache]

                # One-token forward with KV cache
                logits_new, kv_cache = self.model.decode_one(next_id, kv_cache)
                next_logits = logits_new[:, 0, :].float().clone()

        return self.tokenizer.decode(ids + generated_ids)

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
                    if len(beam_ids) - len(ids) < min_length:
                        next_logits[0, eos_id] = float('-inf')

                    next_logits = self._apply_repetition_penalty(
                        next_logits, beam_ids[-64:], repetition_penalty, special_ids
                    )

                    if temperature != 1.0:
                        next_logits = next_logits / max(temperature, 1e-8)

                    log_probs = F.log_softmax(next_logits[0], dim=-1)
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
