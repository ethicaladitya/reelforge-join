"""Colour grade, vignette and film grain as FFmpeg video filters."""

from __future__ import annotations

from .config import LookConfig

_GRADES = {
    "none": "",
    # Gentle warmth: a touch of red in, blue out.
    "warm": "colorbalance=rs=.04:bs=-.04:rm=.03:bm=-.03,eq=saturation=1.05",
    # Cream editorial: lifted blacks, softened whites, warm mids, calmer blues.
    "cream": (
        "curves=all='0/0.055 0.5/0.52 1/0.96',"
        "colorbalance=rm=.035:bm=-.065:bh=-.03,eq=saturation=0.9"
    ),
    # Punchy: more contrast and colour for high-energy edits.
    "punchy": "eq=contrast=1.12:saturation=1.22:gamma=0.97",
    # Cool neutral: slight blue shadows, tidy phone-camera feel.
    "cool": "colorbalance=rs=-.03:bs=.05:bm=.03,eq=saturation=0.95",
    # Monochrome with a little bite.
    "mono": "hue=s=0,eq=contrast=1.08",
}


def build_look_filter(cfg: LookConfig) -> str:
    """Return a comma-joined filter chain (empty string when the look is off)."""
    parts = [_GRADES[cfg.grade]]
    if cfg.vignette:
        parts.append("vignette=PI/5")
    if cfg.grain > 0:
        parts.append(f"noise=alls={min(cfg.grain, 20)}")
    return ",".join(p for p in parts if p)
