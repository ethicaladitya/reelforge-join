"""Tests for config loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from reelforge.config import ReelForgeConfig, load_config, _deep_merge


def test_default_config_loads() -> None:
    cfg = load_config()
    assert isinstance(cfg, ReelForgeConfig)
    assert cfg.output.width == 1080
    assert cfg.output.height == 1920
    assert cfg.output.fps == 30


def test_transition_type_validation() -> None:
    from reelforge.config import TransitionConfig
    with pytest.raises(Exception):
        TransitionConfig(type="wipe")  # invalid


def test_deep_merge_overrides_leaf() -> None:
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    override = {"a": {"b": 99}, "d": 10}
    result = _deep_merge(base, override)
    assert result["a"]["b"] == 99
    assert result["a"]["c"] == 2  # preserved
    assert result["d"] == 10


def test_deep_merge_adds_new_key() -> None:
    base = {"a": 1}
    override = {"b": 2}
    result = _deep_merge(base, override)
    assert result["a"] == 1
    assert result["b"] == 2


def test_load_config_from_file(tmp_path: Path) -> None:
    cfg_file = tmp_path / "test.yaml"
    cfg_file.write_text("output:\n  fps: 60\n  crf: 22\n")
    cfg = load_config(cfg_file)
    assert cfg.output.fps == 60
    assert cfg.output.crf == 22
    # defaults preserved
    assert cfg.output.width == 1080


def test_load_config_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nonexistent.yaml")


def test_highlight_keywords_loaded() -> None:
    cfg = load_config()
    if cfg.captions.highlight.keywords:
        assert any(k.lower() == "wordpress" for k in [k.lower() for k in cfg.captions.highlight.keywords])
