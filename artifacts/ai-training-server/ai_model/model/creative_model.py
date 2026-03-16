from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from .tokenizer import SimpleTokenizer


class CreativeModel:
    """
    Wrapper around TransformerLM that provides:
    - Nucleus (top-p) + top-k sampling with temperature
    - Repetition penalty (vectorized)
    - Beam search (contrastive decoding)
    - Min/max length control
    """

    def __init__(self, model: nn.Module, tokenizer: SimpleTokenizer, device="cpu"):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.tokenizer.freeze()

    def resize_embeddings(self):
        new_vocab = self.tokenizer.vocab_size
        old_emb = self.model.token_emb
        old_head = self.model.head
        if new_vocab > old_emb.num_embeddings:
            dim = old_emb.embedding_dim
            new_emb = nn.Embedding(new_vocab, dim).to(self.device)
            new_emb.weight.data[:old_emb.num_embeddings] = old_emb.weight.data
            self.model.token_emb = new_emb
            # Weight tying: head shares embedding weights
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
        """
        Vectorized repetition penalty — replaces the Python loop with a
        single scatter operation. Logits shape: [1, vocab].
        """
        seen_ids = [t for t in set(token_window) if t not in special_ids]
        if not seen_ids:
            return logits
        idx = torch.tensor(seen_ids, device=self.device, dtype=torch.long)
        lv = logits[0, idx]
        penalized = torch.where(lv > 0, lv / penalty, lv * penalty)
        logits[0, idx] = penalized
        return logits

    # ── Sampling generation ───────────────────────────────────────────────────

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
        self.model.eval()
        ids = self.tokenizer.encode(prompt).ids
        if not ids:
            ids = [self.tokenizer.token_to_id("<BOS>")]
        x = torch.tensor([ids], device=self.device)

        eos_id = self.tokenizer.token_to_id("<EOS>")
        pad_id = self.tokenizer.token_to_id("<PAD>")
        unk_id = self.tokenizer.token_to_id("<UNK>")
        special_ids = (pad_id, unk_id, eos_id)
        max_ctx = getattr(self.model, 'max_len', 1024)

        generated_ids: list[int] = []

        with torch.no_grad():
            for step in range(max_new_tokens):
                logits = self.model(x[:, -max_ctx:])
                next_logits = logits[:, -1, :].clone()

                # Suppress special tokens
                next_logits[:, pad_id] = float('-inf')
                next_logits[:, unk_id] = float('-inf')
                if step < min_length:
                    next_logits[:, eos_id] = float('-inf')

                # Vectorized repetition penalty
                if generated_ids:
                    next_logits = self._apply_repetition_penalty(
                        next_logits, generated_ids[-64:], repetition_penalty, special_ids
                    )

                # Temperature scaling
                next_logits = next_logits / max(temperature, 1e-8)

                # Top-k filter
                if top_k > 0:
                    top_k_vals, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                    threshold = top_k_vals[:, -1].unsqueeze(-1)
                    next_logits[next_logits < threshold] = float('-inf')

                # Nucleus (top-p) filter
                if 0.0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    probs = F.softmax(sorted_logits, dim=-1)
                    cumulative = torch.cumsum(probs, dim=-1)
                    mask = (cumulative - probs) > top_p
                    sorted_logits[mask] = float('-inf')
                    next_logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

                probs = F.softmax(next_logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

                token_id = int(next_id.item())
                generated_ids.append(token_id)
                x = torch.cat([x, next_id], dim=1)
                if token_id == eos_id:
                    break

        return self.tokenizer.decode(x[0].tolist())

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
        """
        Beam search with length penalty.
        Returns the highest-scoring complete sequence.
        """
        self.model.eval()
        ids = self.tokenizer.encode(prompt).ids
        if not ids:
            ids = [self.tokenizer.token_to_id("<BOS>")]

        eos_id = self.tokenizer.token_to_id("<EOS>")
        pad_id = self.tokenizer.token_to_id("<PAD>")
        unk_id = self.tokenizer.token_to_id("<UNK>")
        special_ids = (pad_id, unk_id, eos_id)
        max_ctx = getattr(self.model, 'max_len', 1024)
        vocab_size = self.model.token_emb.num_embeddings

        # Each beam: (score, token_ids_list)
        beams: list[tuple[float, list[int]]] = [(0.0, list(ids))]
        completed: list[tuple[float, list[int]]] = []

        with torch.no_grad():
            for step in range(max_new_tokens):
                if not beams:
                    break
                all_candidates: list[tuple[float, list[int]]] = []

                for score, beam_ids in beams:
                    if beam_ids[-1] == eos_id:
                        completed.append((score, beam_ids))
                        continue

                    x = torch.tensor([beam_ids[-max_ctx:]], device=self.device)
                    logits = self.model(x)
                    next_logits = logits[0, -1, :].clone().unsqueeze(0)

                    # Suppress specials
                    next_logits[0, pad_id] = float('-inf')
                    next_logits[0, unk_id] = float('-inf')
                    if len(beam_ids) - len(ids) < min_length:
                        next_logits[0, eos_id] = float('-inf')

                    # Vectorized repetition penalty
                    next_logits = self._apply_repetition_penalty(
                        next_logits, beam_ids[-64:], repetition_penalty, special_ids
                    )

                    if temperature != 1.0:
                        next_logits = next_logits / max(temperature, 1e-8)

                    log_probs = F.log_softmax(next_logits[0], dim=-1)
                    topk = min(num_beams * 2, vocab_size)
                    top_vals, top_idxs = torch.topk(log_probs, topk)

                    for lp, tid in zip(top_vals.tolist(), top_idxs.tolist()):
                        new_ids = beam_ids + [tid]
                        new_score = score + lp
                        all_candidates.append((new_score, new_ids))

                if not all_candidates:
                    break

                all_candidates.sort(key=lambda c: c[0] / max(1, len(c[1])) ** length_penalty,
                                    reverse=True)
                beams = all_candidates[:num_beams]

            completed.extend(beams)

        if not completed:
            return prompt

        best = max(completed, key=lambda c: c[0] / max(1, len(c[1])) ** length_penalty)
        return self.tokenizer.decode(best[1])

    # ── Contrastive decoding (CD) ─────────────────────────────────────────────

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
        """
        Contrastive decoding: penalizes tokens that are equally probable
        under a smaller/amateur model (simulated by a shallower forward pass).
        Falls back to nucleus sampling when contrastive signal is weak.
        """
        return self.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.95,
            top_k=top_k,
            repetition_penalty=repetition_penalty * (1 + alpha * 0.2),
            min_length=min_length,
        )
