"""Tests for clip discovery and sorting."""

from __future__ import annotations

from pathlib import Path

import pytest

from reelforge.clip_discovery import discover_clips, _natural_key
from reelforge.config import ClipsConfig
from reelforge.exceptions import NoClipsFoundError


def _make_clips(directory: Path, names: list[str]) -> None:
    for name in names:
        (directory / name).touch()


def test_discovers_mp4_files(tmp_path: Path) -> None:
    _make_clips(tmp_path, ["01_hook.mp4", "02_problem.mp4", "03_solution.mp4"])
    cfg = ClipsConfig(directory=str(tmp_path))
    clips = discover_clips(tmp_path, cfg)
    assert len(clips) == 3
    assert clips[0].name == "01_hook.mp4"


def test_sorts_alphabetically(tmp_path: Path) -> None:
    _make_clips(tmp_path, ["03_c.mp4", "01_a.mp4", "02_b.mp4"])
    cfg = ClipsConfig(directory=str(tmp_path), sort="alphabetical")
    clips = discover_clips(tmp_path, cfg)
    names = [c.name for c in clips]
    assert names == sorted(names)


def test_ignores_non_video_files(tmp_path: Path) -> None:
    _make_clips(tmp_path, ["clip.mp4", "notes.txt", "image.png"])
    cfg = ClipsConfig()
    clips = discover_clips(tmp_path, cfg)
    assert len(clips) == 1
    assert clips[0].name == "clip.mp4"


def test_raises_if_directory_missing() -> None:
    cfg = ClipsConfig()
    with pytest.raises(NoClipsFoundError, match="does not exist"):
        discover_clips(Path("/nonexistent/path/xyz"), cfg)


def test_raises_if_no_clips(tmp_path: Path) -> None:
    (tmp_path / "readme.txt").touch()
    cfg = ClipsConfig()
    with pytest.raises(NoClipsFoundError, match="No video files"):
        discover_clips(tmp_path, cfg)


def test_natural_key_orders_correctly() -> None:
    paths = [Path("clip10.mp4"), Path("clip2.mp4"), Path("clip1.mp4")]
    sorted_paths = sorted(paths, key=_natural_key)
    assert [p.name for p in sorted_paths] == ["clip1.mp4", "clip2.mp4", "clip10.mp4"]
