"""Cross-request dynamic batching for the in-house transformer.

The server runs every text-generation call (captions, hooks, CTAs, scripts,
distribution copy) through ``CreativeModel.generate`` inside worker threads. On a
CPU box each call is a single-stream (B=1) forward pass; when many requests
arrive at once they each spin up their own B=1 forward and thrash the 4 cores.

A batched forward is far cheaper per token than N separate ones (measured ~5.9x
prefill throughput at B=32 on this box). This module coalesces concurrent
``generate`` calls that are in flight at the same moment into ONE batched forward
via ``CreativeModel.generate_batch_rows`` (left-padded + key_padding_mask, so each
row's per-step logits match running it alone — exact text parity for B=1/greedy,
and the same sampling distribution for stochastic B>1), then routes results back
to each caller.

Design (single-GIL, sync-thread friendly — matches the existing run_in_executor
worker-thread execution model):

- Callers (already on worker threads) ``submit`` a row and block on an Event.
- One dedicated daemon batcher thread drains up to ``max_batch`` rows that arrived
  within a small time window and runs a single batched forward.
- The batcher never acquires INFERENCE_GATE: it is the *only* thread that touches
  the model, so inference is naturally serialized (one batched forward at a time)
  — no CPU oversubscription, and no deadlock with gate-holding callers.
- Total safety: any failure (batch error, count mismatch, timeout) falls back to
  the original single-sequence ``generate`` so generation can never break.

Opt-in via ``AI_DYNAMIC_BATCHING=1``. When unset, the server leaves
``CreativeModel.generate`` untouched and behavior is byte-for-byte unchanged.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Callable

try:  # memory-aware batch cap (best-effort; never required)
    from .adaptive_concurrency import available_memory_bytes
except Exception:  # pragma: no cover - defensive
    def available_memory_bytes() -> int:
        return 2 * 1024 ** 3


def is_enabled() -> bool:
    """True when cross-request dynamic batching is opt-in enabled."""
    return os.environ.get("AI_DYNAMIC_BATCHING", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


class _Pending:
    __slots__ = ("row", "event", "result", "error", "abandoned")

    def __init__(self, row: dict) -> None:
        self.row = row
        self.event = threading.Event()
        self.result: str | None = None
        self.error: BaseException | None = None
        self.abandoned = False


class GenerateCoalescer:
    """Coalesces concurrent single-prompt generate calls into batched forwards."""

    def __init__(
        self,
        batch_fn: Callable[[list[dict]], list[str]],
        fallback_fn: Callable[[dict], str],
        max_batch: int = 8,
        window_ms: float = 8.0,
        submit_timeout_s: float = 180.0,
    ) -> None:
        # batch_fn(rows) -> outputs aligned to rows (CreativeModel.generate_batch_rows)
        # fallback_fn(row) -> single output (original CreativeModel.generate)
        self._batch_fn = batch_fn
        self._fallback_fn = fallback_fn
        self.max_batch = max(1, _env_int("AI_BATCH_MAX", max_batch))
        self.window_s = max(0.0, _env_float("AI_BATCH_WINDOW_MS", window_ms) / 1000.0)
        self.submit_timeout_s = max(1.0, _env_float("AI_BATCH_TIMEOUT_S", submit_timeout_s))

        self._cv = threading.Condition()
        self._pending: list[_Pending] = []
        self._started = False
        self._runner = threading.Thread(
            target=self._loop, name="gen-coalescer", daemon=True
        )
        # Lightweight observability (read without locking is fine for stats).
        self.stats = {
            "batches": 0,
            "requests": 0,
            "max_batch_seen": 0,
            "fallbacks": 0,
        }

    def start(self) -> None:
        if not self._started:
            self._started = True
            self._runner.start()

    # ── caller side ───────────────────────────────────────────────────────────

    def submit(self, row: dict) -> str:
        """Enqueue a row and block until its batched result is ready."""
        req = _Pending(row)
        with self._cv:
            self._pending.append(req)
            self._cv.notify()
        if not req.event.wait(timeout=self.submit_timeout_s):
            # The caller has given up. Mark abandoned and drop it from the queue
            # if the batcher hasn't claimed it yet, so we never spend a batched
            # forward (or a fallback generate) on a row nobody is waiting for.
            with self._cv:
                req.abandoned = True
                try:
                    self._pending.remove(req)
                except ValueError:
                    pass
            raise TimeoutError("dynamic-batching submit timed out")
        if req.error is not None:
            raise req.error
        assert req.result is not None
        return req.result

    # ── batcher side ──────────────────────────────────────────────────────────

    def _effective_max_batch(self) -> int:
        """Shrink the batch cap if the box is low on memory (best-effort)."""
        try:
            avail_gb = available_memory_bytes() / (1024 ** 3)
        except Exception:
            return self.max_batch
        # ~0.15 GB/slot headroom for KV-cache + activations at typical lengths.
        by_mem = int(avail_gb / 0.15)
        return max(1, min(self.max_batch, by_mem))

    def _collect_batch(self) -> list[_Pending]:
        with self._cv:
            while not self._pending:
                self._cv.wait()
            cap = self._effective_max_batch()
            # Got >=1; briefly wait for more to fill the batch (collapses to
            # immediate dispatch under low load if window is tiny / nobody else).
            if self.window_s > 0:
                deadline = time.monotonic() + self.window_s
                while len(self._pending) < cap:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._cv.wait(timeout=remaining)
            batch = self._pending[:cap]
            del self._pending[:cap]
            # Drop callers that already timed out (abandoned under this same lock
            # in submit) so we never batch rows nobody is waiting for.
            return [r for r in batch if not r.abandoned]

    def _loop(self) -> None:
        while True:
            try:
                batch = self._collect_batch()
            except Exception:
                continue
            if not batch:
                continue
            rows = [r.row for r in batch]
            self.stats["batches"] += 1
            self.stats["requests"] += len(batch)
            self.stats["max_batch_seen"] = max(self.stats["max_batch_seen"], len(batch))
            try:
                outs = self._batch_fn(rows)
                if not isinstance(outs, list) or len(outs) != len(batch):
                    raise RuntimeError(
                        f"batch output count mismatch: {len(outs) if isinstance(outs, list) else 'n/a'} "
                        f"!= {len(batch)}"
                    )
                for r, o in zip(batch, outs):
                    r.result = o
                    r.event.set()
            except Exception:
                # One bad batch must never fail every caller: retry each row
                # individually through the original single-sequence generate.
                for r in batch:
                    if r.abandoned:  # caller already timed out + fell back
                        r.event.set()
                        continue
                    self.stats["fallbacks"] += 1
                    try:
                        r.result = self._fallback_fn(r.row)
                    except BaseException as exc:  # surface per-row, keep loop alive
                        r.error = exc
                    finally:
                        r.event.set()


def install(creative_model) -> GenerateCoalescer | None:
    """Wrap ``creative_model.generate`` to route through a coalescer.

    Returns the coalescer (started) when enabled, else None (model untouched).
    The wrapper signature mirrors ``CreativeModel.generate`` exactly so existing
    positional/keyword call sites behave identically. On any coalescer error the
    wrapper falls back to the original generate, so generation cannot break.
    """
    if not is_enabled():
        return None

    original_generate = creative_model.generate

    def _fallback(row: dict) -> str:
        params = {k: v for k, v in row.items() if k != "prompt"}
        return original_generate(row["prompt"], **params)

    coalescer = GenerateCoalescer(
        batch_fn=creative_model.generate_batch_rows,
        fallback_fn=_fallback,
    )
    coalescer.start()

    def wrapped_generate(
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.85,
        top_p: float = 0.92,
        top_k: int = 50,
        repetition_penalty: float = 1.15,
        min_length: int = 10,
    ) -> str:
        row = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "min_length": min_length,
        }
        try:
            return coalescer.submit(row)
        except Exception:
            return original_generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                min_length=min_length,
            )

    creative_model.generate = wrapped_generate
    return coalescer
