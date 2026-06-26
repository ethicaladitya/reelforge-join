"""Tests for audio processing helpers."""

from __future__ import annotations

from pathlib import Path

from reelforge.audio_processor import audio_fade_filter, build_music_filter
from reelforge.config import MusicConfig


def test_audio_fade_filter_contains_afade() -> None:
    f = audio_fade_filter(10.0, fade_ms=80)
    assert "afade=t=in" in f
    assert "afade=t=out" in f


def test_audio_fade_filter_very_short_clip() -> None:
    # Should not produce negative st value
    f = audio_fade_filter(0.05, fade_ms=80)
    assert "afade=t=in" in f


def test_build_music_filter_contains_amix() -> None:
    cfg = MusicConfig(enabled=True, volume=0.05, duck_volume=0.02)
    speech_intervals = [(0.5, 3.0), (4.0, 7.0)]
    f = build_music_filter(10.0, Path("music.mp3"), speech_intervals, cfg)
    assert "amix" in f


def test_build_music_filter_no_speech() -> None:
    cfg = MusicConfig(enabled=True, volume=0.05, duck_volume=0.02)
    f = build_music_filter(10.0, Path("music.mp3"), [], cfg)
    assert "amix" in f
    # No ducking expressions needed
    assert "between" not in f
