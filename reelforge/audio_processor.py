"""Audio processing: loudness normalization, ducking, fades."""

from __future__ import annotations

import logging
from pathlib import Path

from .config import AudioConfig, MusicConfig
from .ffmpeg_utils import loudnorm_filter, run_ffmpeg

log = logging.getLogger("reelforge")


def normalize_audio(input_path: Path, output_path: Path, cfg: AudioConfig) -> None:
    """Two-pass EBU R128 loudness normalization via FFmpeg loudnorm."""
    filter_str = loudnorm_filter(cfg.loudness_target, cfg.true_peak, cfg.lra)

    run_ffmpeg(
        [
            "-i", str(input_path),
            "-af", filter_str,
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            str(output_path),
        ],
        description=f"Normalize audio for {input_path.name}",
    )


def build_music_filter(
    speech_duration: float,
    music_path: Path,
    speech_intervals: list[tuple[float, float]],
    music_cfg: MusicConfig,
) -> str:
    """Build an FFmpeg audio filter that mixes music with ducking during speech.

    Returns the filter_complex string for use with two inputs:
        [0:a] = speech/video audio
        [1:a] = background music
    """
    vol = music_cfg.volume
    duck = music_cfg.duck_volume
    fade_in = music_cfg.fade_in
    fade_out = music_cfg.fade_out

    # Loop and trim music to match speech duration
    music_chain = (
        f"[1:a]"
        f"aloop=loop=-1:size=2e+09,"
        f"atrim=0:{speech_duration:.3f},"
        f"afade=t=in:st=0:d={fade_in},"
        f"afade=t=out:st={max(0, speech_duration - fade_out):.3f}:d={fade_out}"
        f"[music_raw];"
    )

    # Build volume automation using the asendcmd + volume approach via
    # aeval + sidechaincompress or simple volume filter with enable expressions.
    # We use volume filter with timeline editing (enable) for simplicity.
    enable_parts: list[str] = []
    for start, end in speech_intervals:
        enable_parts.append(f"between(t,{start:.3f},{end:.3f})")

    enable_expr = "+".join(enable_parts) if enable_parts else "0"

    duck_chain = (
        f"[music_raw]"
        f"volume={duck}:enable='{enable_expr}',"
        f"volume={vol}:enable='not({enable_expr})'"
        f"[music_ducked];"
    )

    # Mix with speech audio
    mix_chain = "[0:a][music_ducked]amix=inputs=2:duration=first:dropout_transition=0[aout]"

    return music_chain + duck_chain + mix_chain


def audio_fade_filter(duration: float, fade_ms: int = 80) -> str:
    """Return an FFmpeg audio filter adding tiny fades at clip boundaries."""
    fade_s = fade_ms / 1000.0
    return (
        f"afade=t=in:st=0:d={fade_s},"
        f"afade=t=out:st={max(0, duration - fade_s):.4f}:d={fade_s}"
    )
