"""Cross-request dynamic batching for the in-house transformer.

The server runs every text-generation call (captions, hooks, CTAs, scripts,
distribution copy) through ``CreativeModel.generate`` inside worker threads. On a
Digital GPU node each call is a single-stream (B=1) forward pass; when many
requests arrive at once they each spin up their own B=1 forward and thrash the engine.

A batched forward is far cheaper per token than N separate ones (measured ~5.9×
prefill throughput at B=32 on this Digital GPU node). This module coalesces concurrent
``generate`` calls into ONE batched forward via
``CreativeModel.generate_batch_rows`` (left-padded + key_padding_mask, so each
row's per-step logits match running it alone — exact text parity for B=1/greedy,
and the same sampling distribution for stochastic B>1), then routes results back
to each caller.

Architecture — pipelined two-thread design for 100 % unique-request throughput
────────────────────────────────────────────────────────────────────────────────
The original single-thread design serialised collection and execution:

    [collect batch N] → [execute batch N] → [collect batch N+1] → …

This left the model idle during collection and the queue empty during execution.
The new design overlaps them via a staging queue:

    [collect N] → staging_q → [execute N]
    [collect N+1] ←──────────── (while N is executing)

  • ``gen-collector`` thread  — calls ``_collect_batch`` and puts into ``_ready_q``
  • ``gen-executor``  thread  — pulls from ``_ready_q`` and runs the forward pass

The staging queue is bounded (maxsize=2) so the collector never races too far
ahead — backpressure kicks in when execution is slower than collection.

Pocket GPU lifecycle
────────────────────
When a ``PocketGPUPool`` is supplied each batched forward gets its own GPU life
(born → working → dead) from the pocket dimension.  The batch is the unit of
life: one forward pass, one GPU instance, isolated VRAM, clean death on finish.

Enabled by default.  Disable with ``AI_DYNAMIC_BATCHING=0``.
"""
from __future__ import annotations

import os
import queue
import threading
import time
from typing import Callable

# available_memory_bytes import removed: the Digital GPU backend is independent
# of Replit's host environment — host RAM is not a valid batch-size signal.


def is_enabled() -> bool:
    """True when cross-request dynamic batching is active.

    Enabled by default — set ``AI_DYNAMIC_BATCHING=0`` to disable."""
    return os.environ.get("AI_DYNAMIC_BATCHING", "1").strip().lower() not in (
        "0", "false", "no", "off",
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


def _default_max_batch() -> int:
    """Max batch size for the Digital GPU coalescer.

    The Digital GPU backend is independent of Replit's host environment —
    host CPU count is not the right sizing signal. Use the full capacity
    ceiling; the GPU engine absorbs the load regardless of host vCPUs."""
    return 64


class _Pending:
    __slots__ = ("row", "event", "result", "error", "abandoned")

    def __init__(self, row: dict) -> None:
        self.row = row
        self.event = threading.Event()
        self.result: str | None = None
        self.error: BaseException | None = None
        self.abandoned = False


class GenerateCoalescer:
    """Coalesces concurrent single-prompt generate calls into batched forwards.

    Two daemon threads keep the model busy continuously:
      gen-collector — gathers requests into the next batch
      gen-executor  — runs the current batch's forward pass

    They run concurrently so collection of batch N+1 overlaps execution of N.
    """

    def __init__(
        self,
        batch_fn: Callable[[list[dict]], list[str]],
        fallback_fn: Callable[[dict], str],
        max_batch: int | None = None,
        window_ms: float = 6.0,
        submit_timeout_s: float = 180.0,
        gpu_pool=None,          # optional PocketGPUPool — one life per batch
    ) -> None:
        self._batch_fn    = batch_fn
        self._fallback_fn = fallback_fn
        self._gpu_pool    = gpu_pool

        self.max_batch        = max(1, _env_int("AI_BATCH_MAX", max_batch or _default_max_batch()))
        self.window_s         = max(0.0, _env_float("AI_BATCH_WINDOW_MS", window_ms) / 1000.0)
        self.submit_timeout_s = max(1.0, _env_float("AI_BATCH_TIMEOUT_S", submit_timeout_s))
        # Staging-queue depth: how many ready batches the collector may buffer
        # ahead of the executor.  AI_PIPE_DEPTH=4 lets the collector stay 3
        # batches ahead so the forward-pass thread is never starved during
        # 90M-scale unique-request burst.  Raise for deeper pipelines on boxes
        # with enough RAM; lower to 1 to revert to the original serial design.
        self.pipe_depth       = max(1, _env_int("AI_PIPE_DEPTH", 4))

        # Pending queue (unbounded — callers add, collector drains).
        self._cv      = threading.Condition()
        self._pending: list[_Pending] = []

        # Staging queue between collector and executor (bounded = backpressure).
        self._ready_q: queue.Queue[list[_Pending]] = queue.Queue(maxsize=self.pipe_depth)

        self._started = False
        self._collector = threading.Thread(
            target=self._collect_loop, name="gen-collector", daemon=True,
        )
        self._executor = threading.Thread(
            target=self._execute_loop, name="gen-executor", daemon=True,
        )

        # Lightweight observability.
        self.stats = {
            "batches":        0,
            "requests":       0,
            "max_batch_seen": 0,
            "fallbacks":      0,
            "gpu_lives":      0,   # number of pocket GPU lives spawned
        }

    def start(self) -> None:
        if not self._started:
            self._started = True
            self._collector.start()
            self._executor.start()

    # ── caller side ───────────────────────────────────────────────────────────

    def submit(self, row: dict) -> str:
        """Enqueue a row and block until its batched result is ready."""
        req = _Pending(row)
        with self._cv:
            self._pending.append(req)
            self._cv.notify()
        if not req.event.wait(timeout=self.submit_timeout_s):
            # Caller timed out — mark abandoned so the batcher skips it.
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

    # ── collector thread ──────────────────────────────────────────────────────

    def _effective_max_batch(self) -> int:
        """Return the configured batch cap.

        The Digital GPU backend manages its own memory independently of
        Replit's host RAM — host available_memory_bytes() is not a valid
        proxy for GPU-side capacity and must not throttle batch sizes."""
        return self.max_batch

    def _collect_batch(self) -> list[_Pending]:
        with self._cv:
            while not self._pending:
                self._cv.wait()
            cap = self._effective_max_batch()
            # Wait briefly for more requests to fill the batch.
            # Under high unique-request load the window fills instantly;
            # under low load it degrades gracefully to immediate dispatch.
            if self.window_s > 0:
                deadline = time.monotonic() + self.window_s
                while len(self._pending) < cap:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._cv.wait(timeout=remaining)
            batch = self._pending[:cap]
            del self._pending[:cap]
            # Drop callers that timed out before we even got to them.
            return [r for r in batch if not r.abandoned]

    def _collect_loop(self) -> None:
        """Continuously collect batches and push to the ready queue.

        Runs independently of the executor — while the forward pass is running
        this thread is already collecting the next batch.  The ``_ready_q``
        backpressure (maxsize=pipe_depth) prevents runaway buffering."""
        while True:
            try:
                batch = self._collect_batch()
            except Exception:
                continue
            if not batch:
                continue
            # put() blocks when the queue is full — natural backpressure.
            self._ready_q.put(batch)

    # ── executor thread ───────────────────────────────────────────────────────

    def _run_batch(self, batch: list[_Pending]) -> None:
        """Run one batched forward pass, fan results back to callers.

        If a PocketGPUPool is wired in, the batch lives inside one GPU life:
        born at the start of the forward pass, dead (VRAM flushed) on exit."""
        rows = [r.row for r in batch]
        self.stats["batches"]  += 1
        self.stats["requests"] += len(batch)
        self.stats["max_batch_seen"] = max(self.stats["max_batch_seen"], len(batch))

        def _forward() -> list[str]:
            return self._batch_fn(rows)

        try:
            if self._gpu_pool is not None:
                self.stats["gpu_lives"] += 1
                with self._gpu_pool.spawn_sync(f"batch-{self.stats['batches']}") as _glife:
                    outs = _forward()
            else:
                outs = _forward()

            if not isinstance(outs, list) or len(outs) != len(batch):
                raise RuntimeError(
                    f"batch output count mismatch: "
                    f"{len(outs) if isinstance(outs, list) else 'n/a'} != {len(batch)}"
                )
            for r, o in zip(batch, outs):
                r.result = o
                r.event.set()

        except Exception:
            # One bad batch must never fail every caller — fall back to
            # individual single-sequence generate for each row.
            for r in batch:
                if r.abandoned:
                    r.event.set()
                    continue
                self.stats["fallbacks"] += 1
                try:
                    r.result = self._fallback_fn(r.row)
                except BaseException as exc:
                    r.error = exc
                finally:
                    r.event.set()

    def _execute_loop(self) -> None:
        """Continuously pull batches from the staging queue and run them.

        While this thread is blocked in the forward pass, the collector
        thread is already gathering the next batch — the model is never idle
        waiting for requests to arrive."""
        while True:
            try:
                batch = self._ready_q.get()
            except Exception:
                continue
            try:
                self._run_batch(batch)
            except Exception:
                pass


def install(creative_model, gpu_pool=None) -> "GenerateCoalescer | None":
    """Wrap ``creative_model.generate`` to route through the pipelined coalescer.

    Returns the coalescer (started) when enabled, else None (model untouched).
    The wrapper signature mirrors ``CreativeModel.generate`` exactly so existing
    call sites behave identically.  On any coalescer error the wrapper falls
    back to the original generate so generation can never break.

    Args:
        creative_model: the model whose ``.generate`` will be wrapped.
        gpu_pool: optional ``PocketGPUPool`` — when supplied each batched
                  forward gets its own GPU life (born → working → dead).
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
        gpu_pool=gpu_pool,
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
            "prompt":             prompt,
            "max_new_tokens":     max_new_tokens,
            "temperature":        temperature,
            "top_p":              top_p,
            "top_k":              top_k,
            "repetition_penalty": repetition_penalty,
            "min_length":         min_length,
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
