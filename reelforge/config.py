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
    numbers: bool = False  # also highlight any word containing a digit (₹5000, 20%, 2025)


class CaptionConfig(BaseModel):
    enabled: bool = True
    model: str = "base"
    language: str = "en"
    font: FontConfig = Field(default_factory=FontConfig)
    position: float = 0.78
    max_chars_per_line: int = 22
    words_per_block: int = 4
    highlight: HighlightConfig = Field(default_factory=HighlightConfig)
    bold: bool = True
    uppercase: bool = False
    shadow: int = 0              # drop-shadow depth in px (0 = none)
    box: bool = False            # draw a translucent box behind each caption
    box_color: str = "#000000"
    box_opacity: float = 0.55
    animation: str = "none"      # none | pop | fade
    fonts_dir: str = "assets/fonts"  # drop .ttf/.otf files here to use them by family name

    @field_validator("animation")
    @classmethod
    def validate_animation(cls, v: str) -> str:
        allowed = {"none", "pop", "fade"}
        if v not in allowed:
            raise ValueError(f"captions.animation must be one of {allowed}")
        return v


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
    punch_in: float = 1.0  # >1.0 zooms every other clip by this much, so each cut "punches in"


class LookConfig(BaseModel):
    grade: str = "none"    # none | warm | cream | punchy | cool | mono
    vignette: bool = False
    grain: int = 0         # film grain strength, 0–20

    @field_validator("grade")
    @classmethod
    def validate_grade(cls, v: str) -> str:
        allowed = {"none", "warm", "cream", "punchy", "cool", "mono"}
        if v not in allowed:
            raise ValueError(f"look.grade must be one of {allowed}")
        return v


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
    fit: str = "crop"  # pad | crop | blur — how non-vertical clips fill the 9:16 frame

    @field_validator("fit")
    @classmethod
    def validate_fit(cls, v: str) -> str:
        allowed = {"pad", "crop", "blur"}
        if v not in allowed:
            raise ValueError(f"output.fit must be one of {allowed}")
        return v


class AudioConfig(BaseModel):
    loudness_target: float = -14.0
    true_peak: float = -1.0
    lra: float = 11.0
    voice_cleanup: bool = False  # de-rumble, denoise, EQ, compress and de-ess the voice


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
    style: str = "classic"
    look: LookConfig = Field(default_factory=LookConfig)
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


STYLES_PATH = Path(__file__).parent.parent / "config" / "styles.yaml"


def load_styles() -> dict[str, dict[str, Any]]:
    """Return the style presets from config/styles.yaml."""
    with STYLES_PATH.open() as f:
        return {k: v or {} for k, v in (yaml.safe_load(f) or {}).items()}


def build_config(
    base: dict[str, Any], overrides: dict[str, Any], style: str | None = None
) -> ReelForgeConfig:
    """Merge defaults < style preset < overrides and validate.

    The style comes from the argument, else an override's ``style:`` key,
    else the base's, else ``classic``.
    """
    name = style or overrides.get("style") or base.get("style") or "classic"
    styles = load_styles()
    if name not in styles:
        raise ValueError(f"Unknown style {name!r}. Choose from: {', '.join(styles)}")
    merged = _deep_merge(_deep_merge(base, styles[name]), overrides)
    merged["style"] = name
    return ReelForgeConfig.model_validate(merged)


def load_config(path: Path | None = None, style: str | None = None) -> ReelForgeConfig:
    """Load config from YAML file and optional style preset, falling back to defaults."""
    base: dict[str, Any] = {}
    overrides: dict[str, Any] = {}

    default_path = Path(__file__).parent.parent / "config" / "default.yaml"
    if default_path.exists():
        with default_path.open() as f:
            base = yaml.safe_load(f) or {}

    if path is not None:
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with path.open() as f:
            overrides = yaml.safe_load(f) or {}

    return build_config(base, overrides, style)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
