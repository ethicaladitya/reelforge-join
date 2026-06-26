"""Silence detection and trimming using FFmpeg silencedetect filter."""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .config import SilenceConfig
from .ffmpeg_utils import require_ffmpeg

log = logging.getLogger("reelforge")


@dataclass
class SilenceInterval:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class TrimPoints:
    start: float  # seconds to trim from the front
    end: float    # total duration to keep (i.e. end point in original)


def detect_silence(path: Path, cfg: SilenceConfig) -> list[SilenceInterval]:
    """Run FFmpeg silencedetect and return silence intervals."""
    ffmpeg, _ = require_ffmpeg()
    noise = cfg.threshold_db
    duration = cfg.min_silence_ms / 1000.0

    cmd = [
        ffmpeg,
        "-i", str(path),
        "-af", f"silencedetect=n={noise}dB:d={duration}",
        "-f", "null",
        "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stderr  # silencedetect writes to stderr

    intervals: list[SilenceInterval] = []
    starts: list[float] = []

    for line in output.splitlines():
        m_start = re.search(r"silence_start:\s*([\d.]+)", line)
        m_end = re.search(r"silence_end:\s*([\d.]+)", line)
        if m_start:
            starts.append(float(m_start.group(1)))
        if m_end and starts:
            intervals.append(SilenceInterval(start=starts[-1], end=float(m_end.group(1))))

    return intervals


def compute_trim_points(path: Path, cfg: SilenceConfig, total_duration: float) -> TrimPoints:
    """Compute start/end trim points by inspecting leading/trailing silence."""
    if not cfg.trim_enabled:
        return TrimPoints(start=0.0, end=total_duration)

    intervals = detect_silence(path, cfg)
    pad = cfg.pad_ms / 1000.0

    trim_start = 0.0
    trim_end = total_duration

    # Trim leading silence: if first interval starts at 0
    if intervals and intervals[0].start < 0.05:
        trim_start = max(0.0, intervals[0].end - pad)

    # Trim trailing silence: if last interval ends near the file end
    if intervals and abs(intervals[-1].end - total_duration) < 0.2:
        trim_end = min(total_duration, intervals[-1].start + pad)

    if trim_start > 0 or trim_end < total_duration:
        log.debug(
            "Trim %s: %.3fs → %.3fs (saved %.3fs)",
            path.name,
            trim_start,
            trim_end,
            (total_duration - trim_end) + trim_start,
        )

    return TrimPoints(start=trim_start, end=trim_end)
