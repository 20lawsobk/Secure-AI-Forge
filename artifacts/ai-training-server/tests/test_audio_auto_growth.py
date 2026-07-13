"""Focused integration checks for audio dataset auto-growth.

Tests the three behaviours the reviewer required:
  1. Successful auto-seed updates last_audio_seed_at and audio_auto_seed_count.
  2. Rate-limit throttle: a second call within the interval is a no-op.
  3. AlreadySeedingError from a concurrent seed is caught gracefully.

All tests are deterministic — no real storage, no network, no ffmpeg.
"""

from __future__ import annotations

import threading
import time
import types
import unittest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------

def _make_storage(num_chunks: int = 5) -> MagicMock:
    """Minimal storage stub that satisfies DataPuller._audio_chunk_count()."""
    s = MagicMock()
    s.is_available = True
    s.disk_store_available = True
    s.get = MagicMock(return_value={"num_chunks": num_chunks})
    return s


def _make_puller(num_chunks: int = 5):
    """DataPuller wired to a stub storage that looks below threshold."""
    from workers.data_puller import DataPuller

    dp = DataPuller(_make_storage(num_chunks))
    # Reset auto-growth counters explicitly (mirror __init__)
    dp.state["last_audio_seed_at"] = None
    dp.state["audio_auto_seed_count"] = 0
    dp.state["audio_chunks"] = num_chunks
    return dp


_FAKE_SUMMARY = {"stored": 3, "total": 8, "source": "librosa:bundled-examples"}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAutoGrowthStateUpdates(unittest.TestCase):
    """Successful auto-seed MUST record timestamp and increment counter."""

    def test_state_updated_on_success(self):
        dp = _make_puller(num_chunks=5)  # below AUDIO_GROWTH_THRESHOLD=20

        with patch("workers.data_puller.DataPuller.AUDIO_GROWTH_THRESHOLD", 20), \
             patch("workers.data_puller.DataPuller.AUDIO_GROWTH_INTERVAL_SEC", 0), \
             patch("workers.seed_audio_dataset.seed", return_value=_FAKE_SUMMARY):

            dp._auto_seed_audio()

        self.assertIsNotNone(
            dp.state["last_audio_seed_at"],
            "last_audio_seed_at must be set after a successful auto-seed",
        )
        self.assertEqual(
            dp.state["audio_auto_seed_count"], 1,
            "audio_auto_seed_count must be incremented to 1 after first auto-seed",
        )
        self.assertEqual(
            dp.state["audio_chunks"], _FAKE_SUMMARY["total"],
            "audio_chunks must reflect the new total from the seed summary",
        )

    def test_counter_increments_across_multiple_seeds(self):
        dp = _make_puller(num_chunks=5)

        with patch("workers.data_puller.DataPuller.AUDIO_GROWTH_THRESHOLD", 20), \
             patch("workers.data_puller.DataPuller.AUDIO_GROWTH_INTERVAL_SEC", 0), \
             patch("workers.seed_audio_dataset.seed", return_value=_FAKE_SUMMARY):

            dp._auto_seed_audio()
            dp._auto_seed_audio()

        self.assertEqual(
            dp.state["audio_auto_seed_count"], 2,
            "Counter must reach 2 after two successful seeds (interval=0)",
        )


class TestRateLimitThrottle(unittest.TestCase):
    """Second call within AUDIO_GROWTH_INTERVAL_SEC must be a no-op."""

    def test_throttled_within_interval(self):
        dp = _make_puller(num_chunks=5)
        seed_call_count = {"n": 0}

        def _fake_seed(*args, **kwargs):
            seed_call_count["n"] += 1
            return _FAKE_SUMMARY

        with patch("workers.data_puller.DataPuller.AUDIO_GROWTH_THRESHOLD", 20), \
             patch("workers.data_puller.DataPuller.AUDIO_GROWTH_INTERVAL_SEC", 9999), \
             patch("workers.seed_audio_dataset.seed", side_effect=_fake_seed):

            dp._auto_seed_audio()   # first call — should seed
            dp._auto_seed_audio()   # second call within interval — throttled

        self.assertEqual(
            seed_call_count["n"], 1,
            "seed() must be called exactly once when interval throttle is active",
        )

    def test_not_throttled_after_interval_expires(self):
        dp = _make_puller(num_chunks=5)
        seed_call_count = {"n": 0}

        def _fake_seed(*args, **kwargs):
            seed_call_count["n"] += 1
            return _FAKE_SUMMARY

        with patch("workers.data_puller.DataPuller.AUDIO_GROWTH_THRESHOLD", 20), \
             patch("workers.data_puller.DataPuller.AUDIO_GROWTH_INTERVAL_SEC", 0), \
             patch("workers.seed_audio_dataset.seed", side_effect=_fake_seed):

            dp._auto_seed_audio()   # first call
            dp._auto_seed_audio()   # second call — interval=0, so not throttled

        self.assertEqual(
            seed_call_count["n"], 2,
            "seed() must be called twice when interval=0 (no throttle)",
        )


class TestAlreadySeedingHandled(unittest.TestCase):
    """AlreadySeedingError must be caught — auto-growth must never raise."""

    def test_already_seeding_is_graceful(self):
        from workers.seed_audio_dataset import AlreadySeedingError

        dp = _make_puller(num_chunks=5)

        with patch("workers.data_puller.DataPuller.AUDIO_GROWTH_THRESHOLD", 20), \
             patch("workers.data_puller.DataPuller.AUDIO_GROWTH_INTERVAL_SEC", 0), \
             patch("workers.seed_audio_dataset.seed",
                   side_effect=AlreadySeedingError("already seeding")):

            # Must not raise
            dp._auto_seed_audio()

        # State must not be updated when seeding was skipped
        self.assertIsNone(
            dp.state["last_audio_seed_at"],
            "last_audio_seed_at must remain None when seed was skipped",
        )
        self.assertEqual(
            dp.state["audio_auto_seed_count"], 0,
            "counter must remain 0 when seed was skipped",
        )

    def test_concurrent_lock_held_no_raise(self):
        """Simulate the real lock being held by another thread."""
        from workers.seed_audio_dataset import _SEED_LOCK

        dp = _make_puller(num_chunks=5)

        # Hold the module-level seed lock from another thread
        _SEED_LOCK.acquire()
        try:
            with patch("workers.data_puller.DataPuller.AUDIO_GROWTH_THRESHOLD", 20), \
                 patch("workers.data_puller.DataPuller.AUDIO_GROWTH_INTERVAL_SEC", 0):
                # _auto_seed_audio calls seed() → raises AlreadySeedingError → caught
                dp._auto_seed_audio()
        finally:
            _SEED_LOCK.release()

        # State unchanged
        self.assertIsNone(dp.state["last_audio_seed_at"])


class TestAboveThresholdSkipped(unittest.TestCase):
    """No seeding when dataset already meets the threshold."""

    def test_above_threshold_no_seed(self):
        dp = _make_puller(num_chunks=25)  # above threshold of 20
        seed_call_count = {"n": 0}

        def _fake_seed(*args, **kwargs):
            seed_call_count["n"] += 1
            return _FAKE_SUMMARY

        with patch("workers.data_puller.DataPuller.AUDIO_GROWTH_THRESHOLD", 20), \
             patch("workers.data_puller.DataPuller.AUDIO_GROWTH_INTERVAL_SEC", 0), \
             patch("workers.seed_audio_dataset.seed", side_effect=_fake_seed):

            dp._auto_seed_audio()

        self.assertEqual(seed_call_count["n"], 0, "seed() must not be called when above threshold")


if __name__ == "__main__":
    unittest.main(verbosity=2)
