"""Watermark and logo overlay filter builders."""

from __future__ import annotations

import logging
from pathlib import Path

from .config import LogoConfig, WatermarkConfig

log = logging.getLogger("reelforge")


def _position_expr(position: str, margin: int, item: str = "overlay") -> tuple[str, str]:
    """Return (x_expr, y_expr) for the given position string."""
    m = margin
    positions = {
        "bottom_left": (f"{m}", f"H-h-{m}"),
        "bottom_right": (f"W-w-{m}", f"H-h-{m}"),
        "top_left": (f"{m}", f"{m}"),
        "top_right": (f"W-w-{m}", f"{m}"),
        "center": ("(W-w)/2", "(H-h)/2"),
    }
    return positions.get(position, (f"{m}", f"H-h-{m}"))


def watermark_filter_inputs(cfg: WatermarkConfig, base_dir: Path) -> tuple[list[str], str] | None:
    """Return (extra_inputs, filter_graph_chain) for watermark overlay, or None if disabled."""
    if not cfg.enabled:
        return None

    wm_path = base_dir / cfg.path
    if not wm_path.exists():
        log.warning("Watermark file not found: %s — skipping watermark", wm_path)
        return None

    x, y = _position_expr(cfg.position, cfg.margin)
    alpha = cfg.opacity

    # [wm] input stream: scale to reasonable size, apply opacity
    filter_chain = (
        f"[wm_in]format=rgba,"
        f"colorchannelmixer=aa={alpha:.2f}"
        f"[wm];"
        f"[base][wm]overlay=x={x}:y={y}:format=auto[with_wm]"
    )

    return (["-i", str(wm_path)], filter_chain)


def logo_filter_inputs(cfg: LogoConfig, base_dir: Path, video_width: int) -> tuple[list[str], str] | None:
    """Return (extra_inputs, filter_graph_chain) for logo overlay, or None if disabled."""
    if not cfg.enabled:
        return None

    logo_path = base_dir / cfg.path
    if not logo_path.exists():
        log.warning("Logo file not found: %s — skipping logo", logo_path)
        return None

    x, y = _position_expr(cfg.position, cfg.margin)
    alpha = cfg.opacity
    scale_w = int(video_width * cfg.scale)

    filter_chain = (
        f"[logo_in]scale={scale_w}:-1,"
        f"format=rgba,"
        f"colorchannelmixer=aa={alpha:.2f}"
        f"[logo];"
        f"[prev][logo]overlay=x={x}:y={y}:format=auto[with_logo]"
    )

    return (["-i", str(logo_path)], filter_chain)
