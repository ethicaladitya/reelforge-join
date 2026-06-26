"""Tests for transition filter builders."""

from __future__ import annotations

from reelforge.config import TransitionConfig
from reelforge.transitions import build_audio_crossfade, build_xfade_filter


def test_no_transition_returns_empty() -> None:
    cfg = TransitionConfig(type="none", duration=0.25)
    fc, label = build_xfade_filter(3, [5.0, 5.0, 5.0], cfg)
    assert fc == ""
    assert label == ""


def test_single_clip_returns_empty() -> None:
    cfg = TransitionConfig(type="fade", duration=0.25)
    fc, label = build_xfade_filter(1, [5.0], cfg)
    assert fc == ""
    assert label == ""


def test_fade_transition_produces_xfade() -> None:
    cfg = TransitionConfig(type="fade", duration=0.25)
    fc, label = build_xfade_filter(2, [5.0, 5.0], cfg)
    assert "xfade" in fc
    assert "fade" in fc
    assert label != ""


def test_dip_to_black_uses_fadeblack() -> None:
    cfg = TransitionConfig(type="dip_to_black", duration=0.5)
    fc, _ = build_xfade_filter(2, [5.0, 5.0], cfg)
    assert "fadeblack" in fc


def test_dissolve_transition() -> None:
    cfg = TransitionConfig(type="dissolve", duration=0.3)
    fc, _ = build_xfade_filter(2, [5.0, 5.0], cfg)
    assert "dissolve" in fc


def test_audio_crossfade_single_clip() -> None:
    fc, label = build_audio_crossfade(1, [5.0], 0.25)
    assert label == "aout"


def test_audio_crossfade_multiple_clips() -> None:
    fc, label = build_audio_crossfade(3, [5.0, 5.0, 5.0], 0.25)
    assert "acrossfade" in fc
    assert label == "aout"
