"""Voice-over synthesis regression tests (Task: birdcall narration fix).

Guards the failure class where "voice-over" videos shipped unintelligible
audio: narration must be real speech at the pipeline sample rate, not the
arpeggio synth and not a pitch-shifted (wrong-rate) render.
"""
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_model.audio.voiceover import (  # noqa: E402
    ESPEAK_BIN,
    VO_SAMPLE_RATE,
    clean_vo_text,
    mix_voiceover_over_music,
    synthesize_voiceover,
    voiceover_track,
)


def _read_wav(path):
    with wave.open(str(path)) as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        data = np.frombuffer(wf.readframes(n), dtype=np.int16).astype(np.float64)
    return sr, data


def _spectral_centroid(sr, data):
    spec = np.abs(np.fft.rfft(data))
    freqs = np.fft.rfftfreq(len(data), 1.0 / sr)
    return float((spec * freqs).sum() / (spec.sum() or 1.0))


class TestCleanVoText:
    def test_strips_emoji_hashtags_urls_tokens(self):
        raw = ("Stop scrolling — Midnight Drive is here! 🔥 #newmusic "
               "<STAGE_CTA> https://example.com/track stream it now")
        out = clean_vo_text(raw)
        assert "🔥" not in out and "#" not in out and "<" not in out
        assert "http" not in out
        assert "Midnight Drive is here" in out

    def test_empty_and_symbol_only(self):
        assert clean_vo_text("") == ""
        assert clean_vo_text("🔥🔥🔥 ✨") == ""


needs_espeak = pytest.mark.skipif(
    not ESPEAK_BIN, reason="espeak-ng not installed in this environment"
)


@needs_espeak
class TestSynthesizeVoiceover:
    def test_produces_speech_at_pipeline_rate(self, tmp_path):
        out = tmp_path / "vo.wav"
        ok = synthesize_voiceover(
            "This is your sign to stop and play Midnight Drive immediately.",
            str(out),
        )
        assert ok, "espeak-ng synthesis must succeed"
        sr, data = _read_wav(out)
        # The birdcall bug class: narration muxed at a rate it wasn't
        # synthesized at. The artifact itself must already be 44100 Hz.
        assert sr == VO_SAMPLE_RATE
        assert len(data) > sr * 1.0, "narration should be > 1s of audio"
        # Speech energy concentrates low; the arpeggio synth (birdcalls)
        # centres far higher. Centroid must sit in the speech band.
        centroid = _spectral_centroid(sr, data)
        assert 100.0 <= centroid <= 3000.0, f"centroid {centroid:.0f} Hz not speech-like"

    def test_empty_text_returns_false(self, tmp_path):
        assert synthesize_voiceover("🔥🔥", str(tmp_path / "x.wav")) is False


@needs_espeak
class TestMixAndTrack:
    def _music(self, tmp_path):
        """A bright arpeggio-ish tone — stands in for the synth soundtrack."""
        sr = VO_SAMPLE_RATE
        t = np.arange(sr * 3) / sr
        tone = 0.8 * np.sin(2 * np.pi * 1760.0 * t)
        pcm = (tone * 32767).astype(np.int16)
        p = tmp_path / "music.wav"
        with wave.open(str(p), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm.tobytes())
        return p

    def test_voice_dominates_ducked_music(self, tmp_path):
        vo = tmp_path / "vo.wav"
        assert synthesize_voiceover(
            "Midnight Drive is out now. Stream it everywhere and tell a friend.",
            str(vo),
        )
        music = self._music(tmp_path)
        mixed = tmp_path / "mixed.wav"
        assert mix_voiceover_over_music(str(vo), str(music), str(mixed), 3.0)
        sr, data = _read_wav(mixed)
        assert sr == VO_SAMPLE_RATE
        # Ducking check: energy in the speech band (100-1000 Hz) must exceed
        # the energy in the music-tone band (1760 Hz ± 60). If ducking failed
        # the continuous 0.8-amplitude tone dominates by an order of magnitude.
        spec = np.abs(np.fft.rfft(data)) ** 2
        freqs = np.fft.rfftfreq(len(data), 1.0 / sr)
        speech_e = spec[(freqs >= 100) & (freqs <= 1000)].sum()
        tone_e = spec[(freqs >= 1700) & (freqs <= 1820)].sum()
        assert speech_e > tone_e, (
            f"music not ducked (speech band {speech_e:.3g} <= tone band {tone_e:.3g})"
        )

    def test_voiceover_track_end_to_end(self, tmp_path):
        music = self._music(tmp_path)
        path = voiceover_track(
            "New single out Friday. Do not miss it.",
            out_dir=str(tmp_path), job_id="testjob12345",
            duration_sec=3.0, music_path=str(music),
        )
        assert path and Path(path).exists()
        sr, _ = _read_wav(path)
        assert sr == VO_SAMPLE_RATE

    def test_never_raises_on_bad_music_path(self, tmp_path):
        path = voiceover_track(
            "Hello world narration.", out_dir=str(tmp_path),
            job_id="badmusicjob1", duration_sec=2.0,
            music_path=str(tmp_path / "missing.wav"),
        )
        # Falls back to narration-only, never raises
        assert path and Path(path).exists()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
