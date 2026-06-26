"""Low-level FFmpeg helpers: probe, run, filter building."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .exceptions import FFmpegNotFoundError, RenderError

log = logging.getLogger("reelforge")


# ---------------------------------------------------------------------------
# Availability checks
# ---------------------------------------------------------------------------


def require_ffmpeg() -> tuple[str, str]:
    """Return (ffmpeg_path, ffprobe_path), preferring ffmpeg-full on macOS.

    ffmpeg-full (brew install ffmpeg-full) includes libfreetype, libass and
    other filters stripped from the standard formula.  We check its keg-only
    path first so the server works regardless of which PATH is active.
    """
    _FULL_BINS = [
        "/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg",
        "/usr/local/opt/ffmpeg-full/bin/ffmpeg",
    ]

    ffmpeg: str | None = None
    ffprobe: str | None = None

    for candidate in _FULL_BINS:
        p = Path(candidate)
        if p.exists():
            ffmpeg = str(p)
            ffprobe = str(p.parent / "ffprobe")
            break

    if not ffmpeg:
        ffmpeg = shutil.which("ffmpeg")
        ffprobe_found = shutil.which("ffprobe")
        if ffmpeg:
            ffprobe = ffprobe_found

    if not ffmpeg or not ffprobe:
        raise FFmpegNotFoundError(
            "FFmpeg is not installed or not on PATH.\n"
            "  macOS:  brew install ffmpeg-full\n"
            "  Linux:  apt install ffmpeg\n"
        )
    return ffmpeg, ffprobe


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------


@dataclass
class StreamInfo:
    index: int
    codec_type: str
    codec_name: str
    width: int | None
    height: int | None
    duration: float | None
    r_frame_rate: str | None
    sample_rate: int | None
    channels: int | None


@dataclass
class ProbeResult:
    streams: list[StreamInfo]
    format_duration: float | None
    format_name: str

    @property
    def video(self) -> StreamInfo | None:
        return next((s for s in self.streams if s.codec_type == "video"), None)

    @property
    def audio(self) -> StreamInfo | None:
        return next((s for s in self.streams if s.codec_type == "audio"), None)

    @property
    def duration(self) -> float:
        if self.format_duration is not None:
            return self.format_duration
        v = self.video
        if v and v.duration is not None:
            return v.duration
        return 0.0

    @property
    def fps(self) -> float:
        v = self.video
        if not v or not v.r_frame_rate:
            return 30.0
        try:
            num, den = v.r_frame_rate.split("/")
            return float(num) / float(den)
        except Exception:
            return 30.0


def probe(path: Path) -> ProbeResult:
    """Run ffprobe on *path* and return structured result."""
    _, ffprobe = require_ffmpeg()
    cmd = [
        ffprobe,
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RenderError(f"ffprobe failed on {path}: {result.stderr}")

    data: dict[str, Any] = json.loads(result.stdout)

    streams = []
    for s in data.get("streams", []):
        streams.append(
            StreamInfo(
                index=int(s.get("index", 0)),
                codec_type=s.get("codec_type", ""),
                codec_name=s.get("codec_name", ""),
                width=s.get("width"),
                height=s.get("height"),
                duration=float(s["duration"]) if "duration" in s else None,
                r_frame_rate=s.get("r_frame_rate"),
                sample_rate=int(s["sample_rate"]) if "sample_rate" in s else None,
                channels=s.get("channels"),
            )
        )

    fmt = data.get("format", {})
    return ProbeResult(
        streams=streams,
        format_duration=float(fmt["duration"]) if "duration" in fmt else None,
        format_name=fmt.get("format_name", ""),
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_ffmpeg(
    args: list[str],
    *,
    description: str = "FFmpeg",
    verbose: bool = False,
) -> None:
    """Run FFmpeg with the given argument list.  Raises RenderError on failure."""
    ffmpeg, _ = require_ffmpeg()
    cmd = [ffmpeg, "-y"] + args
    if not verbose:
        cmd = [ffmpeg, "-y", "-loglevel", "error"] + args

    log.debug("Running: %s", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=not verbose, text=True)
    if result.returncode != 0:
        stderr = result.stderr if not verbose else ""
        raise RenderError(f"{description} failed (exit {result.returncode}):\n{stderr}")


# ---------------------------------------------------------------------------
# Filter-graph helpers
# ---------------------------------------------------------------------------


def fps_fraction(fps: float) -> str:
    """Convert float fps to a clean fraction string suitable for FFmpeg."""
    if fps == 30.0:
        return "30/1"
    if fps == 60.0:
        return "60/1"
    if fps == 25.0:
        return "25/1"
    if fps == 24.0:
        return "24/1"
    # Generic: multiply by 1000 to handle 29.97 etc.
    num = round(fps * 1000)
    return f"{num}/1000"


def loudnorm_filter(target: float = -14.0, tp: float = -1.0, lra: float = 11.0) -> str:
    """Return an FFmpeg loudnorm filter string."""
    return f"loudnorm=I={target}:TP={tp}:LRA={lra}:print_format=none"


def scale_pad_filter(width: int, height: int) -> str:
    """Scale video to fit within (width x height) with letterbox/pillarbox."""
    return (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black"
    )
