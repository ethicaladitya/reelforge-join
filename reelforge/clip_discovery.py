"""Discover and sort video clips from the input directory."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from .config import ClipsConfig
from .exceptions import NoClipsFoundError

log = logging.getLogger("reelforge")


def _natural_key(path: Path) -> list[int | str]:
    """Sort key that handles embedded numbers naturally (01, 02, 10, …)."""
    parts: list[int | str] = []
    for chunk in re.split(r"(\d+)", path.stem):
        parts.append(int(chunk) if chunk.isdigit() else chunk.lower())
    return parts


def discover_clips(directory: Path, cfg: ClipsConfig) -> list[Path]:
    """Return a sorted list of video clip paths found in *directory*.

    Raises NoClipsFoundError if the directory is empty or does not exist.
    """
    if not directory.exists():
        raise NoClipsFoundError(
            f"Clips directory does not exist: {directory}\n"
            "Create it and add your .mp4 files, then re-run."
        )

    exts = {e.lower() for e in cfg.extensions}
    clips = [p for p in directory.iterdir() if p.suffix.lower() in exts and p.is_file()]

    if not clips:
        raise NoClipsFoundError(
            f"No video files found in {directory}.\n"
            f"Supported extensions: {', '.join(sorted(exts))}"
        )

    sort_mode = cfg.sort
    if sort_mode in ("alphabetical", "natural"):
        clips.sort(key=_natural_key)
    elif sort_mode == "numeric":
        # Extract leading integer from filename for sorting
        def _num_key(p: Path) -> int:
            m = re.match(r"(\d+)", p.stem)
            return int(m.group(1)) if m else 0

        clips.sort(key=_num_key)
    else:
        clips.sort(key=_natural_key)

    log.info("Discovered %d clip(s):", len(clips))
    for i, c in enumerate(clips, 1):
        log.info("  %2d. %s", i, c.name)

    return clips
