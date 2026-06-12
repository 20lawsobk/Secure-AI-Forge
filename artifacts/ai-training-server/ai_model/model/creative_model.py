from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from .tokenizer import SimpleTokenizer


class CreativeModel:
    """
    Wrapper around TransformerLM that provides:
    - KV-cache nucleus sampling (prefill once, decode O(1) per token)
    - Beam search (contrastive decoding)
    - Min length control
    - Repetition penalty (vectorized)
    """

    def __init__(self, model: nn.Module, tokenizer: SimpleTokenizer, device="cpu"):
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
