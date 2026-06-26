"""Transition filter-graph builders for FFmpeg xfade."""

from __future__ import annotations

from .config import TransitionConfig


def build_xfade_filter(
    num_clips: int,
    durations: list[float],
    cfg: TransitionConfig,
) -> tuple[str, str]:
    """Build FFmpeg filter_complex for chained xfade transitions.

    Returns (filter_complex, output_label).
    If transition type is 'none', returns empty strings (use concat instead).
    """
    if cfg.type == "none" or num_clips < 2:
        return "", ""

    xfade_type = _xfade_type(cfg.type)
    td = cfg.duration  # transition duration in seconds

    # Each xfade output feeds the next. Accumulate offsets.
    chains: list[str] = []
    current_offset = 0.0
    last_label = "[v0]"

    for i in range(num_clips):
        chains.append(f"[{i}:v]setpts=PTS-STARTPTS[v{i}]")

    video_labels = [f"[v{i}]" for i in range(num_clips)]
    out_label = "[vout]"

    xfade_parts: list[str] = []
    prev = video_labels[0]

    for i in range(1, num_clips):
        current_offset += durations[i - 1] - td
        next_v = video_labels[i]
        out = f"[xf{i}]" if i < num_clips - 1 else out_label

        xfade_parts.append(
            f"{prev}{next_v}xfade=transition={xfade_type}"
            f":duration={td}:offset={current_offset:.4f}{out}"
        )
        prev = out

    filter_complex = ";".join(chains) + ";" + ";".join(xfade_parts)
    return filter_complex, out_label.strip("[]")


def _xfade_type(transition_type: str) -> str:
    mapping = {
        "fade": "fade",
        "dissolve": "dissolve",
        "dip_to_black": "fadeblack",
    }
    return mapping.get(transition_type, "fade")


def build_audio_crossfade(
    num_clips: int,
    durations: list[float],
    fade_duration: float,
) -> tuple[str, str]:
    """Build audio acrossfade chain for smooth audio transitions.

    Returns (filter_complex_audio_section, output_label).
    """
    if num_clips < 2:
        return f"[0:a][aout]", "aout"

    parts: list[str] = []
    prev = "[0:a]"

    for i in range(1, num_clips):
        next_a = f"[{i}:a]"
        out = f"[af{i}]" if i < num_clips - 1 else "[aout]"
        d = min(fade_duration, durations[i - 1] * 0.5)
        parts.append(f"{prev}{next_a}acrossfade=d={d:.4f}:c1=tri:c2=tri{out}")
        prev = out

    return ";".join(parts), "aout"
