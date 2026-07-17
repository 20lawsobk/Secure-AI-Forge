"""
Tests for the expanded AwarenessConditioner vocab.

Verifies:
  1. No duplicate VOCAB entries
  2. New platforms (threads, bluesky, snapchat, etc.) shift the vector
  3. New genres (amapiano, citypop, bedroom, rage, grime, etc.) shift the vector
  4. [HIGH] upweighting produces a stronger signal than base presence
  5. Identical awareness strings produce identical conditioning tensors
  6. Different awareness strings produce different conditioning tensors
  7. genre detection in script_agent picks up new genre vocabulary
"""
from __future__ import annotations

import sys
import os

# Run from repo root or tests/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest
from ai_model.video.diffusion.awareness_conditioner import (
    VOCAB,
    _VOCAB_IDX,
    _presence_vector,
    AwarenessConditioner,
)
from ai_model.agents.script_agent import _detect_genre_from_awareness


# ── 1. Vocab integrity ─────────────────────────────────────────────────────────

def test_no_duplicates():
    assert len(VOCAB) == len(set(VOCAB)), "VOCAB contains duplicate entries"


def test_vocab_size_at_least_120():
    assert len(VOCAB) >= 120, f"Expected ≥120 terms, got {len(VOCAB)}"


def test_new_platforms_in_vocab():
    new_platforms = ["threads", "bluesky", "pinterest", "snapchat", "discord", "twitch", "bereal"]
    missing = [p for p in new_platforms if p not in _VOCAB_IDX]
    assert not missing, f"New platforms missing from VOCAB: {missing}"


def test_new_genres_in_vocab():
    new_genres = ["amapiano", "citypop", "bedroom", "rage", "pluggnb", "synthwave",
                  "grime", "gospel", "emo", "shoegaze", "funk", "jersey", "cumbia"]
    missing = [g for g in new_genres if g not in _VOCAB_IDX]
    assert not missing, f"New genres missing from VOCAB: {missing}"


def test_new_production_terms_in_vocab():
    new_terms = ["beat", "sample", "bpm", "mix", "master", "prod", "bridge"]
    missing = [t for t in new_terms if t not in _VOCAB_IDX]
    assert not missing, f"New production terms missing from VOCAB: {missing}"


def test_new_mood_terms_in_vocab():
    new_terms = ["soulful", "nostalgic", "raw", "polished", "aggressive", "smooth"]
    missing = [t for t in new_terms if t not in _VOCAB_IDX]
    assert not missing, f"New mood terms missing from VOCAB: {missing}"


def test_new_visual_terms_in_vocab():
    new_terms = ["neon", "aesthetic", "retro", "futuristic", "pastel", "grunge", "minimal", "vibrant"]
    missing = [t for t in new_terms if t not in _VOCAB_IDX]
    assert not missing, f"New visual terms missing from VOCAB: {missing}"


# ── 2. Presence vector — novel terms fire correctly ────────────────────────────

def test_amapiano_fires():
    vec = _presence_vector("This amapiano track is taking over")
    assert vec[_VOCAB_IDX["amapiano"]] == 1.0


def test_bluesky_fires():
    vec = _presence_vector("Going viral on bluesky right now")
    assert vec[_VOCAB_IDX["bluesky"]] == 1.0


def test_threads_fires():
    vec = _presence_vector("threads is blowing up with this drop")
    assert vec[_VOCAB_IDX["threads"]] == 1.0


def test_grime_fires():
    vec = _presence_vector("UK grime scene exploding in 2026")
    assert vec[_VOCAB_IDX["grime"]] == 1.0


def test_synthwave_fires():
    vec = _presence_vector("retro synthwave aesthetic dominating playlists")
    assert vec[_VOCAB_IDX["synthwave"]] == 1.0


def test_neon_aesthetic_fires():
    vec = _presence_vector("neon aesthetic trending on shorts")
    assert vec[_VOCAB_IDX["neon"]] == 1.0
    assert vec[_VOCAB_IDX["aesthetic"]] == 1.0


def test_empty_awareness_is_zero():
    vec = _presence_vector("")
    assert vec.sum().item() == 0.0


def test_unrecognised_term_is_zero():
    """A completely novel term not in VOCAB produces no signal (not a crash)."""
    vec = _presence_vector("xyzzy_unknown_microgenre_2099")
    assert vec.sum().item() == 0.0


# ── 3. HIGH upweighting ────────────────────────────────────────────────────────

def test_high_signal_upweights():
    base = _presence_vector("amapiano trending")
    high = _presence_vector("[HIGH] amapiano trending this week")
    assert high[_VOCAB_IDX["amapiano"]] > base[_VOCAB_IDX["amapiano"]]
    assert high[_VOCAB_IDX["amapiano"]] == pytest.approx(1.5)


def test_high_upweight_capped_at_2():
    vec = _presence_vector("amapiano\n[HIGH] amapiano at the top\n[HIGH] amapiano again")
    assert vec[_VOCAB_IDX["amapiano"]] <= 2.0


# ── 4. Conditioning tensor properties ─────────────────────────────────────────

def _conditioner():
    torch.manual_seed(42)
    return AwarenessConditioner()


def test_output_shape():
    model = _conditioner()
    out = model.encode("viral tiktok drop", batch_size=2)
    assert out.shape == (2, 8, 256)


def test_deterministic():
    model = _conditioner()
    a = model.encode("amapiano trending on bluesky").clone()
    b = model.encode("amapiano trending on bluesky").clone()
    assert torch.allclose(a, b), "Same awareness should produce identical tensors"


def test_different_awareness_different_tensor():
    model = _conditioner()
    a = model.encode("amapiano trending on bluesky")
    b = model.encode("synthwave neon aesthetic on youtube")
    assert not torch.allclose(a, b), "Different awareness must produce different tensors"


def test_novel_term_shifts_tensor():
    """Adding a previously-absent vocab term must measurably shift the output."""
    model = _conditioner()
    base = model.encode("viral trending").clone()
    novel = model.encode("viral trending amapiano").clone()  # 'amapiano' is new in VOCAB
    diff = (novel - base).abs().max().item()
    assert diff > 0.0, "Adding a vocab term should shift the conditioning tensor"


def test_platform_term_shifts_tensor():
    """threads / bluesky (new platforms) must shift the tensor."""
    model = _conditioner()
    base = model.encode("trending viral").clone()
    with_threads = model.encode("trending viral threads").clone()
    assert not torch.allclose(base, with_threads)


# ── 5. Genre detection in script_agent ────────────────────────────────────────

def test_detect_amapiano():
    assert _detect_genre_from_awareness("Heavy amapiano influence this week") == "amapiano"


def test_detect_city_pop():
    result = _detect_genre_from_awareness("city pop is resurging on youtube")
    assert result in ("city pop", "citypop")


def test_detect_grime():
    assert _detect_genre_from_awareness("UK grime scene") == "grime"


def test_detect_gospel():
    assert _detect_genre_from_awareness("gospel-influenced soul record") == "gospel"


def test_detect_synthwave():
    assert _detect_genre_from_awareness("synthwave retro aesthetic") == "synthwave"


def test_detect_bedroom():
    assert _detect_genre_from_awareness("bedroom pop artist going viral") == "bedroom pop"


def test_explicit_genre_wins():
    result = _detect_genre_from_awareness("amapiano trending", explicit_genre="drill")
    assert result == "drill"


if __name__ == "__main__":
    # Quick self-test without pytest
    import traceback
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed, failed = 0, 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {fn.__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(failed)
