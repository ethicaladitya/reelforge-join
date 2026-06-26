"""Generate a branded end-card video clip using FFmpeg."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from .config import BrandConfig, OutputConfig
from .ffmpeg_utils import run_ffmpeg

log = logging.getLogger("reelforge")

# System fonts that are reliably present on macOS / common Linux distros
_FONT_CANDIDATES = [
    # macOS
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/Library/Fonts/Arial Bold.ttf",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/SFNS.ttf",
    # Linux
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]


def _find_system_font() -> str | None:
    for p in _FONT_CANDIDATES:
        if Path(p).exists():
            return p
    # Try fc-list as last resort
    try:
        import subprocess
        r = subprocess.run(["fc-list", ":style=Bold", "--format=%{file}\n"],
                           capture_output=True, text=True, timeout=3)
        for line in r.stdout.splitlines():
            if line.strip() and Path(line.strip()).exists():
                return line.strip()
    except Exception:
        pass
    return None


def render_end_card(cfg: BrandConfig, out_cfg: OutputConfig, output_path: Path) -> None:
    """Render a black end card with handle text.  Skips gracefully if no font found."""
    if not cfg.end_card.enabled:
        return

    font_path = _find_system_font()
    if not font_path:
        log.warning("No system font found — skipping end card")
        return

    duration = cfg.end_card.duration
    w, h, fps = out_cfg.width, out_cfg.height, out_cfg.fps
    font_size = cfg.end_card.font_size
    handle = cfg.handle.replace("'", "\\'").replace(":", "\\:")
    body_text = cfg.end_card.text.replace("'", "\\'").replace(":", "\\:")

    y_body = h // 2 - font_size - 24
    y_handle = h // 2 + 12

    # Escape font path for FFmpeg (colons are path separators in filter args)
    safe_font = font_path.replace("\\", "/").replace(":", "\\:")

    dt_body = (
        f"drawtext=fontfile='{safe_font}':"
        f"text='{body_text}':"
        f"fontsize={font_size}:fontcolor=white:"
        f"x=(w-text_w)/2:y={y_body}:"
        f"shadowcolor=black:shadowx=2:shadowy=2"
    )
    dt_handle = (
        f"drawtext=fontfile='{safe_font}':"
        f"text='{handle}':"
        f"fontsize={font_size + 8}:fontcolor=#FFD400:"
        f"x=(w-text_w)/2:y={y_handle}:"
        f"shadowcolor=black:shadowx=2:shadowy=2"
    )

    run_ffmpeg(
        [
            "-f", "lavfi",
            "-i", f"color=black:s={w}x{h}:r={fps}:d={duration}",
            "-f", "lavfi",
            "-i", f"anullsrc=channel_layout=stereo:sample_rate=48000",
            "-t", str(duration),
            "-vf", f"{dt_body},{dt_handle}",
            "-c:v", out_cfg.codec,
            "-crf", str(out_cfg.crf),
            "-preset", "fast",
            "-c:a", out_cfg.audio_codec,
            "-b:a", out_cfg.audio_bitrate,
            "-pix_fmt", out_cfg.pixel_format,
            str(output_path),
        ],
        description="Render end card",
    )
    log.info("End card rendered: %s", output_path.name)
