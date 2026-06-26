"""Tests for FFmpeg utility functions."""

from __future__ import annotations

import pytest

from reelforge.ffmpeg_utils import fps_fraction, loudnorm_filter, scale_pad_filter


def test_fps_fraction_common_values() -> None:
    assert fps_fraction(30.0) == "30/1"
    assert fps_fraction(60.0) == "60/1"
    assert fps_fraction(25.0) == "25/1"
    assert fps_fraction(24.0) == "24/1"


def test_fps_fraction_generic() -> None:
    result = fps_fraction(29.97)
    assert "/" in result


def test_loudnorm_filter_contains_params() -> None:
    f = loudnorm_filter(-14.0, -1.0, 11.0)
    assert "I=-14.0" in f
    assert "TP=-1.0" in f
    assert "LRA=11.0" in f


def test_scale_pad_filter_format() -> None:
    f = scale_pad_filter(1080, 1920)
    assert "scale=1080:1920" in f
    assert "pad=1080:1920" in f
