"""Configuration models and loader for ReelForge."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class FontConfig(BaseModel):
    family: str = "Montserrat-Bold"
    size: int = 68
    color: str = "#FFFFFF"
    stroke_color: str = "#000000"
    stroke_width: int = 4


class HighlightConfig(BaseModel):
    enabled: bool = True
    color: str = "#FFD400"
    keywords: list[str] = Field(default_factory=list)


class CaptionConfig(BaseModel):
    enabled: bool = True
    model: str = "base"
    language: str = "en"
    font: FontConfig = Field(default_factory=FontConfig)
    position: float = 0.78
    max_chars_per_line: int = 22
    words_per_block: int = 4
    highlight: HighlightConfig = Field(default_factory=HighlightConfig)


class MusicConfig(BaseModel):
    enabled: bool = False
    path: str | None = None
    volume: float = 0.05
    duck_volume: float = 0.02
    fade_in: float = 1.0
    fade_out: float = 2.0


class ZoomConfig(BaseModel):
    enabled: bool = True
    scale: float = 1.05
    interval: float = 4.0
    duration: float = 2.0


class TransitionConfig(BaseModel):
    type: str = "fade"
    duration: float = 0.25

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        allowed = {"none", "fade", "dissolve", "dip_to_black"}
        if v not in allowed:
            raise ValueError(f"transition.type must be one of {allowed}")
        return v


class SilenceConfig(BaseModel):
    trim_enabled: bool = True
    threshold_db: float = -40.0
    min_silence_ms: int = 300
    pad_ms: int = 80


class WatermarkConfig(BaseModel):
    enabled: bool = True
    path: str = "assets/watermarks/watermark.png"
    position: str = "bottom_left"
    opacity: float = 0.6
    margin: int = 40


class LogoConfig(BaseModel):
    enabled: bool = False
    path: str = "assets/logos/logo.png"
    position: str = "top_right"
    scale: float = 0.08
    opacity: float = 0.9
    margin: int = 30


class OutputConfig(BaseModel):
    width: int = 1080
    height: int = 1920
    fps: int = 30
    codec: str = "libx264"
    crf: int = 18
    preset: str = "slow"
    pixel_format: str = "yuv420p"
    audio_codec: str = "aac"
    audio_bitrate: str = "192k"


class AudioConfig(BaseModel):
    loudness_target: float = -14.0
    true_peak: float = -1.0
    lra: float = 11.0


class EndCardConfig(BaseModel):
    enabled: bool = True
    text: str = "Follow for more."
    duration: float = 2.5
    font_size: int = 54


class BrandConfig(BaseModel):
    handle: str = "@handle"
    end_card: EndCardConfig = Field(default_factory=EndCardConfig)


class ClipsConfig(BaseModel):
    directory: str = "clips"
    sort: str = "alphabetical"
    extensions: list[str] = Field(default_factory=lambda: [".mp4", ".mov", ".webm", ".mkv"])


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------


class ReelForgeConfig(BaseModel):
    brand: BrandConfig = Field(default_factory=BrandConfig)
    captions: CaptionConfig = Field(default_factory=CaptionConfig)
    music: MusicConfig = Field(default_factory=MusicConfig)
    zoom: ZoomConfig = Field(default_factory=ZoomConfig)
    transitions: TransitionConfig = Field(default_factory=TransitionConfig)
    silence: SilenceConfig = Field(default_factory=SilenceConfig)
    watermark: WatermarkConfig = Field(default_factory=WatermarkConfig)
    logo: LogoConfig = Field(default_factory=LogoConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    clips: ClipsConfig = Field(default_factory=ClipsConfig)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_config(path: Path | None = None) -> ReelForgeConfig:
    """Load config from YAML file, falling back to defaults."""
    base: dict[str, Any] = {}

    default_path = Path(__file__).parent.parent / "config" / "default.yaml"
    if default_path.exists():
        with default_path.open() as f:
            base = yaml.safe_load(f) or {}

    if path is not None:
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with path.open() as f:
            overrides: dict[str, Any] = yaml.safe_load(f) or {}
        base = _deep_merge(base, overrides)

    return ReelForgeConfig.model_validate(base)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
