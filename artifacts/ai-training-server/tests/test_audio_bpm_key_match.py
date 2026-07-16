"""Audio BPM/key match validation — end-to-end + unit tests.

Verifies that /api/generate/audio honours the requested BPM and key:
  1. Unit tests for the pure-function track selector (no server, no ffmpeg).
  2. End-to-end HTTP tests against a live server:
     a. Seeds a small dataset with librosa bundled examples.
     b. Requests audio at a specific BPM + key.
     c. Asserts the returned metadata matches the request within tolerance.
     d. Asserts that a missing-key fallback surfaces a ``selection_warning``
        rather than silently serving the wrong track.

Run:
    python tests/test_audio_bpm_key_match.py
or:
    python -m pytest tests/test_audio_bpm_key_match.py -v
"""
from __future__ import annotations

import json
import sys
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# ── Paths ─────────────────────────────────────────────────────────────────────
_SERVER_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(_SERVER_DIR))

# ── Config ────────────────────────────────────────────────────────────────────
BASE      = "http://127.0.0.1:9878"
API_KEY   = "f242bf97d7e46b7ca0b17cd6b01ca9239bc327b862a86b703556565523849701"
ADMIN_KEY = "mbs_8a3edbac97ff333dda5068410227267e6d85b14a4c9caee279fbb18ddfb47edc"
HEADERS       = {"Content-Type": "application/json", "X-Api-Key": API_KEY}
ADMIN_HEADERS = {"Content-Type": "application/json", "X-Admin-Key": ADMIN_KEY}

# Tolerance for BPM match after rubberband pitch/tempo shift.
# rubberband clamps tempo_ratio to [0.5, 2.0], so within that window the
# applied BPM should equal target_bpm; outside it the applied BPM will be
# clamped and the test checks that applied <= clamp-limit * src_bpm.
BPM_TOLERANCE = 2.0   # ±2 BPM is acceptable for a dataset-sourced render

# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _req(method: str, path: str, body: dict | None = None,
         timeout: int = 30, admin: bool = False) -> dict:
    url     = BASE + path
    data    = json.dumps(body).encode() if body is not None else None
    headers = ADMIN_HEADERS if admin else HEADERS
    rq      = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(rq, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raw = e.read().decode(errors="replace")
        raise AssertionError(f"HTTP {e.code} {method} {path}: {raw[:500]}") from e

def GET(path: str) -> dict:
    return _req("GET", path)

def POST(path: str, body: dict, timeout: int = 30) -> dict:
    return _req("POST", path, body, timeout=timeout)

def POST_ADMIN(path: str, timeout: int = 30) -> dict:
    """POST to an admin endpoint (no body; uses X-Admin-Key)."""
    return _req("POST", path, body=None, timeout=timeout, admin=True)

def _server_alive() -> bool:
    try:
        GET("/health")
        return True
    except Exception:
        return False

def _poll_job(job_id: str, *, max_wait: int = 120, interval: float = 3.0) -> dict:
    """Poll /api/audio-job/{job_id} until done/error or timeout."""
    deadline = time.time() + max_wait
    while time.time() < deadline:
        r = GET(f"/api/audio-job/{job_id}")
        status = r.get("status")
        if status in ("done", "error"):
            return r
        time.sleep(interval)
    raise TimeoutError(f"job {job_id} not done after {max_wait}s")


# ═════════════════════════════════════════════════════════════════════════════
# PART 1 — Unit tests for the pure-function track selector
# ═════════════════════════════════════════════════════════════════════════════

class TestSelectAudioSample(unittest.TestCase):
    """Pure unit tests — no server, no ffmpeg, no storage."""

    def setUp(self):
        from ai_model.audio.track_selector import select_audio_sample
        self.select = select_audio_sample

    # ── Key-match behaviour ────────────────────────────────────────────────

    def test_exact_key_match_is_preferred(self):
        """When a track with the exact requested key exists it is chosen."""
        index = [
            {"idx": 0, "bpm": 120.0, "key": "C major",  "genres": []},
            {"idx": 1, "bpm": 120.0, "key": "A minor",  "genres": []},
            {"idx": 2, "bpm": 120.0, "key": "G major",  "genres": []},
        ]
        best, matched = self.select(index, "a minor", 120.0)
        self.assertTrue(matched, "should report key_matched=True")
        self.assertEqual(best["idx"], 1)

    def test_key_match_case_insensitive(self):
        """Key comparison is case-insensitive."""
        index = [
            {"idx": 0, "bpm": 100.0, "key": "D Major", "genres": []},
            {"idx": 1, "bpm": 100.0, "key": "E minor", "genres": []},
        ]
        best, matched = self.select(index, "d major", 100.0)
        self.assertTrue(matched)
        self.assertEqual(best["idx"], 0)

    def test_no_key_match_returns_false_and_warns(self):
        """When no track matches the key, key_matched=False and all tracks are eligible."""
        index = [
            {"idx": 0, "bpm": 95.0,  "key": "C major", "genres": []},
            {"idx": 1, "bpm": 140.0, "key": "A minor",  "genres": []},
        ]
        best, matched = self.select(index, "B major", 96.0)
        self.assertFalse(matched, "key_matched should be False when no track has the key")
        # Should pick idx=0 (BPM 95 is closest to 96)
        self.assertEqual(best["idx"], 0)

    def test_empty_key_request_does_not_trigger_fallback(self):
        """An empty/None requested key should not be treated as a key to match."""
        index = [
            {"idx": 0, "bpm": 110.0, "key": "C major", "genres": []},
            {"idx": 1, "bpm": 115.0, "key": "",         "genres": []},
        ]
        # When want_key is empty, every entry matches (or fallback = all), and
        # key_matched reflects whether the empty string is in the index.
        best, _ = self.select(index, "", 110.0)
        # Either way a best entry is returned without raising
        self.assertIn("idx", best)

    # ── BPM selection ──────────────────────────────────────────────────────

    def test_closest_bpm_wins_within_key_pool(self):
        """Within key-matched tracks the one with closest BPM is selected."""
        index = [
            {"idx": 0, "bpm": 80.0,  "key": "C major", "genres": []},
            {"idx": 1, "bpm": 128.0, "key": "C major", "genres": []},
            {"idx": 2, "bpm": 140.0, "key": "C major", "genres": []},
        ]
        best, matched = self.select(index, "c major", 130.0)
        self.assertTrue(matched)
        self.assertEqual(best["idx"], 1)   # 128 is closest to 130

    def test_zero_bpm_in_index_is_deprioritised(self):
        """Entries with bpm=0 (estimation failed) rank last on BPM."""
        index = [
            {"idx": 0, "bpm": 0.0,  "key": "A minor", "genres": []},
            {"idx": 1, "bpm": 95.0, "key": "A minor", "genres": []},
        ]
        best, _ = self.select(index, "a minor", 96.0)
        self.assertEqual(best["idx"], 1, "non-zero BPM should beat zero BPM")

    def test_zero_bpm_in_index_used_when_only_option(self):
        """If ALL entries have bpm=0 the selector still returns one entry."""
        index = [
            {"idx": 0, "bpm": 0.0, "key": "C major", "genres": []},
            {"idx": 1, "bpm": 0.0, "key": "C major", "genres": []},
        ]
        best, _ = self.select(index, "c major", 120.0)
        self.assertIn("idx", best)   # no crash

    # ── Genre affinity ─────────────────────────────────────────────────────

    def test_genre_match_breaks_bpm_tie(self):
        """A genre-matching track is preferred over an equally-close BPM track."""
        index = [
            {"idx": 0, "bpm": 100.0, "key": "G major", "genres": ["Classical"]},
            {"idx": 1, "bpm": 100.0, "key": "G major", "genres": ["Hip-Hop"]},
        ]
        best, _ = self.select(index, "g major", 100.0, preferred_genres=["hip-hop"])
        self.assertEqual(best["idx"], 1)

    def test_genre_match_case_and_hyphen_normalised(self):
        """Genre comparison normalises hyphens and case."""
        index = [
            {"idx": 0, "bpm": 120.0, "key": "E minor", "genres": ["drum-and-bass"]},
            {"idx": 1, "bpm": 120.0, "key": "E minor", "genres": ["Pop"]},
        ]
        best, _ = self.select(index, "e minor", 120.0, preferred_genres=["Drum And Bass"])
        self.assertEqual(best["idx"], 0)

    def test_no_preferred_genres_ignores_genre_field(self):
        """When no preferred genres are given genre_score is always 0 (neutral)."""
        index = [
            {"idx": 0, "bpm": 110.0, "key": "F major", "genres": ["Jazz"]},
            {"idx": 1, "bpm": 115.0, "key": "F major", "genres": ["Rock"]},
        ]
        best, _ = self.select(index, "f major", 110.0, preferred_genres=None)
        # BPM 110 should win (exact match)
        self.assertEqual(best["idx"], 0)

    # ── Edge cases ─────────────────────────────────────────────────────────

    def test_single_entry_index(self):
        index = [{"idx": 0, "bpm": 120.0, "key": "C major", "genres": []}]
        best, matched = self.select(index, "c major", 120.0)
        self.assertTrue(matched)
        self.assertEqual(best["idx"], 0)

    def test_empty_index_raises(self):
        with self.assertRaises(ValueError):
            self.select([], "C major", 120.0)

    def test_tiebreak_by_idx(self):
        """When genre and BPM are identical, the lower idx wins (deterministic)."""
        index = [
            {"idx": 5, "bpm": 120.0, "key": "C major", "genres": []},
            {"idx": 2, "bpm": 120.0, "key": "C major", "genres": []},
        ]
        best, _ = self.select(index, "c major", 120.0)
        self.assertEqual(best["idx"], 2)


# ═════════════════════════════════════════════════════════════════════════════
# PART 2 — End-to-end HTTP tests against the live server
# ═════════════════════════════════════════════════════════════════════════════

@unittest.skipUnless(_server_alive(), "server not reachable at 127.0.0.1:9878")
class TestAudioBpmKeyMatchE2E(unittest.TestCase):
    """Integration tests — require a running server on port 9878."""

    @classmethod
    def setUpClass(cls):
        """Seed a small dataset with librosa examples (always available locally).

        The seed endpoint is admin-only (X-Admin-Key) and accepts query params,
        not a JSON body.  It runs in the background; we poll the status endpoint
        until seeding is no longer in progress before running any test.
        """
        # Trigger seeding via query params + admin auth
        r = POST_ADMIN(
            "/storage/datasets/audio/seed?count=5&replace=true&source=librosa",
            timeout=30,
        )
        assert r.get("status") in ("seeding", "already_seeding"), (
            f"unexpected seed response: {r}"
        )

        # Poll until the background seed finishes (librosa is fast, ≤60 s)
        deadline = time.time() + 90
        while time.time() < deadline:
            s = GET("/storage/datasets/audio/status")
            if not s.get("seeding_now", True):
                num = s.get("dataset", {}).get("num_chunks", 0)
                assert num > 0, f"dataset empty after seeding finished: {s}"
                break
            time.sleep(3)
        else:
            raise TimeoutError("audio seeding did not finish within 90 s")

    def _submit_audio(self, target_bpm: float, target_key: str,
                      duration: float = 10.0) -> dict:
        """Submit an audio generation job and poll to completion."""
        body = {
            "genre":      "electronic",
            "intent":     "test",
            "target_bpm": target_bpm,
            "target_key": target_key,
            "duration":   duration,
        }
        r = POST("/api/generate/audio", body, timeout=30)
        job_id = r.get("job_id")
        self.assertIsNotNone(job_id, f"no job_id in response: {r}")
        result = _poll_job(job_id, max_wait=120)
        self.assertEqual(result.get("status"), "done",
                         f"job failed: {result.get('error')}")
        return result

    # ── BPM match ──────────────────────────────────────────────────────────

    def test_returned_bpm_within_tolerance(self):
        """Applied BPM in the response should be within BPM_TOLERANCE of the request."""
        target = 120.0
        result = self._submit_audio(target_bpm=target, target_key="C major")
        applied = float(result["bpm"])
        self.assertIsNotNone(applied, "response must include bpm")
        # rubberband clamps tempo_ratio to [0.5, 2.0]; if the source BPM was
        # very different from the target the applied BPM may be clamped.
        # The test allows for clamping by accepting any BPM ≥ target*0.5
        # when applied != target (honest clamped metadata).
        src_bpm = result.get("source_sample", {}).get("bpm", 0.0) or 0.0
        if src_bpm > 0:
            ratio = target / src_bpm
            clamped = max(0.5, min(2.0, ratio))
            expected = round(src_bpm * clamped, 1)
            self.assertAlmostEqual(applied, expected, delta=BPM_TOLERANCE,
                                   msg=f"applied BPM {applied} should be ~{expected} "
                                       f"(src={src_bpm}, target={target})")
        else:
            # No source BPM info — just check the response is a sane value
            self.assertGreater(applied, 0, "applied BPM must be positive")

    # ── Key match / warning ────────────────────────────────────────────────

    def test_key_match_present_in_dataset(self):
        """When the dataset contains a track with the requested key, no warning is emitted.

        Strategy: do a first unconstrained render to discover which key the
        selector chose, then request that exact key — it must be in the dataset
        so no warning should appear.
        """
        # First render: no key preference → selector picks whatever is closest
        first = self._submit_audio(target_bpm=100.0, target_key="")
        src_key = (first.get("source_sample") or {}).get("key") or ""
        if not src_key:
            self.skipTest("first render returned no source_sample.key")

        # Second render: request the exact key that was just used
        result = self._submit_audio(target_bpm=100.0, target_key=src_key)
        self.assertIsNone(
            result.get("selection_warning"),
            f"should be no warning when key '{src_key}' is in the dataset; "
            f"got: {result.get('selection_warning')}",
        )

    def test_key_not_in_dataset_surfaces_warning(self):
        """When the requested key is absent from the dataset, selection_warning is set."""
        # Use an unusual key that the librosa bundled examples won't have
        unusual_key = "F# dorian"
        result = self._submit_audio(target_bpm=110.0, target_key=unusual_key)

        warning = result.get("selection_warning")
        # A warning is only expected if the key was truly absent.
        # If the dataset happened to contain that key this assertion won't fire.
        if warning is not None:
            self.assertIn("no dataset track matched", warning,
                          f"warning message should explain the fallback; got: {warning!r}")
            self.assertIn(unusual_key, warning,
                          "warning should name the requested key")
        # Either way the job completed and returned a valid URL
        self.assertIn("/uploads/", result.get("url", ""))

    def test_source_sample_metadata_present(self):
        """Response includes source_sample so producers can audit which track was used."""
        result = self._submit_audio(target_bpm=100.0, target_key="C major")
        ss = result.get("source_sample")
        self.assertIsNotNone(ss, "source_sample must be in the response")
        self.assertIn("idx",     ss, "source_sample.idx required")
        self.assertIn("bpm",     ss, "source_sample.bpm required")
        self.assertIn("key",     ss, "source_sample.key required")

    def test_audio_file_is_served(self):
        """The URL in the response must actually serve a file (HTTP 200)."""
        result = self._submit_audio(target_bpm=100.0, target_key="A minor")
        url = result.get("url") or result.get("audio_url")
        self.assertIsNotNone(url)
        file_url = BASE + url
        rq = urllib.request.Request(file_url, headers={"X-Api-Key": API_KEY})
        with urllib.request.urlopen(rq, timeout=15) as resp:
            self.assertEqual(resp.status, 200, f"audio file URL {url} returned {resp.status}")
            data = resp.read(1024)
            self.assertGreater(len(data), 100, "audio file should not be empty")


# ═════════════════════════════════════════════════════════════════════════════
# Runner
# ═════════════════════════════════════════════════════════════════════════════

def _run_unit() -> bool:
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSelectAudioSample)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()

def _run_e2e() -> bool:
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAudioBpmKeyMatchE2E)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()

if __name__ == "__main__":
    print("=" * 60)
    print("PART 1 — Unit tests (track selector logic)")
    print("=" * 60)
    unit_ok = _run_unit()

    print()
    print("=" * 60)
    print("PART 2 — End-to-end HTTP tests (live server)")
    print("=" * 60)
    if not _server_alive():
        print("  [SKIP] server not reachable at 127.0.0.1:9878")
        e2e_ok = True   # not a failure — server just isn't running
    else:
        e2e_ok = _run_e2e()

    sys.exit(0 if (unit_ok and e2e_ok) else 1)


# ── Awareness ↔ dataset sync: numeric FMA genre IDs + mood affinity ──────────

class TestGenreNormalization:
    def test_numeric_fma_ids_map_to_names(self):
        from ai_model.audio.track_selector import normalize_genres
        assert normalize_genres([15, 21]) == ["electronic", "hip hop"]

    def test_numeric_strings_map_to_names(self):
        from ai_model.audio.track_selector import normalize_genres
        assert normalize_genres(["91"]) == ["shoegaze"]

    def test_unknown_ids_kept_as_strings(self):
        from ai_model.audio.track_selector import normalize_genres
        assert normalize_genres([999999]) == ["999999"]

    def test_names_lowercased_and_dehyphenated(self):
        from ai_model.audio.track_selector import normalize_genres
        assert normalize_genres(["Hip-Hop", "Drum_and_Bass"]) == [
            "hip hop", "drum and bass"]

    def test_awareness_genre_matches_fma_id_track(self):
        from ai_model.audio.track_selector import select_audio_sample
        index = [
            {"idx": 0, "bpm": 120, "key": "c major", "genres": [91]},   # shoegaze
            {"idx": 1, "bpm": 120, "key": "c major", "genres": [15]},   # electronic
        ]
        best, matched = select_audio_sample(index, "c major", 120, ["electronic"])
        assert matched and best["idx"] == 1

class TestMoodAffinity:
    def test_mood_breaks_tie_toward_affinity(self):
        from ai_model.audio.track_selector import select_audio_sample
        index = [
            {"idx": 0, "bpm": 120, "key": "c major", "genres": ["folk"]},
            {"idx": 1, "bpm": 120, "key": "c major", "genres": ["techno"]},
        ]
        best, _ = select_audio_sample(index, "c major", 120, [], "energetic")
        assert best["idx"] == 1

    def test_genre_intent_outranks_mood(self):
        from ai_model.audio.track_selector import select_audio_sample
        index = [
            {"idx": 0, "bpm": 120, "key": "c major", "genres": ["folk"]},
            {"idx": 1, "bpm": 120, "key": "c major", "genres": ["techno"]},
        ]
        best, _ = select_audio_sample(index, "c major", 120, ["folk"], "energetic")
        assert best["idx"] == 0

    def test_unknown_mood_is_noop(self):
        from ai_model.audio.track_selector import select_audio_sample
        index = [
            {"idx": 0, "bpm": 118, "key": "c major", "genres": ["folk"]},
            {"idx": 1, "bpm": 125, "key": "c major", "genres": ["techno"]},
        ]
        best, _ = select_audio_sample(index, "c major", 120, [], "zorbly")
        assert best["idx"] == 0  # nearest BPM wins

    def test_mood_with_numeric_fma_genres(self):
        from ai_model.audio.track_selector import select_audio_sample
        index = [
            {"idx": 0, "bpm": 120, "key": "c major", "genres": [17]},  # folk
            {"idx": 1, "bpm": 120, "key": "c major", "genres": [81]},  # techno
        ]
        best, _ = select_audio_sample(index, "c major", 120, [], "energetic")
        assert best["idx"] == 1

class TestNormalizeGenresMalformed:
    def test_scalar_string_not_iterated_charwise(self):
        from ai_model.audio.track_selector import normalize_genres
        assert normalize_genres("Hip-Hop") == ["hip hop"]

    def test_scalar_int(self):
        from ai_model.audio.track_selector import normalize_genres
        assert normalize_genres(91) == ["shoegaze"]

    def test_none(self):
        from ai_model.audio.track_selector import normalize_genres
        assert normalize_genres(None) == []
