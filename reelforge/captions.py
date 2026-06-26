"""Caption rendering: generate ASS subtitle file from Transcript."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from .config import CaptionConfig
from .transcriber import Transcript, Word

log = logging.getLogger("reelforge")


# ---------------------------------------------------------------------------
# ASS helpers
# ---------------------------------------------------------------------------


def _hex_to_ass(color: str) -> str:
    """Convert #RRGGBB to ASS &H00BBGGRR format."""
    color = color.lstrip("#")
    r, g, b = color[0:2], color[2:4], color[4:6]
    return f"&H00{b}{g}{r}"


def _seconds_to_ass_time(secs: float) -> str:
    """Convert seconds to ASS timestamp H:MM:SS.cc"""
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    s = secs % 60
    return f"{h}:{m:02d}:{s:05.2f}"


@dataclass
class CaptionBlock:
    """A group of words displayed together on screen."""
    words: list[Word]
    start: float
    end: float

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)


def build_caption_blocks(transcript: Transcript, cfg: CaptionConfig) -> list[CaptionBlock]:
    """Group transcript words into display blocks (N words per block)."""
    all_words = transcript.all_words
    if not all_words:
        return []

    blocks: list[CaptionBlock] = []
    n = cfg.words_per_block

    for i in range(0, len(all_words), n):
        group = all_words[i : i + n]
        blocks.append(
            CaptionBlock(
                words=group,
                start=group[0].start,
                end=group[-1].end,
            )
        )
    return blocks


# ---------------------------------------------------------------------------
# ASS file generation
# ---------------------------------------------------------------------------


_ASS_HEADER = """\
[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_family},{font_size},{primary},{secondary},{outline},{back},1,0,0,0,100,100,0,0,1,{stroke_w},0,2,40,40,{margin_v},1
Style: Highlight,{font_family},{font_size},{highlight},{secondary},{outline},{back},1,0,0,0,100,100,0,0,1,{stroke_w},0,2,40,40,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def generate_ass(
    blocks: list[CaptionBlock],
    cfg: CaptionConfig,
    output_path: Path,
    width: int,
    height: int,
) -> None:
    """Write an ASS subtitle file implementing word-by-word karaoke-style captions."""
    font = cfg.font
    primary = _hex_to_ass(font.color)
    highlight = _hex_to_ass(cfg.highlight.color)
    outline = _hex_to_ass(font.stroke_color)
    back = "&H00000000"
    secondary = "&H00000000"

    margin_v = int(height * (1.0 - cfg.position))

    header = _ASS_HEADER.format(
        width=width,
        height=height,
        font_family=font.family,
        font_size=font.size,
        primary=primary,
        secondary=secondary,
        outline=outline,
        back=back,
        stroke_w=font.stroke_width,
        highlight=highlight,
        margin_v=margin_v,
    )

    keywords = {k.lower() for k in cfg.highlight.keywords}
    lines: list[str] = []

    for block in blocks:
        # Build one dialogue line per block; words that are keywords get
        # inline override tags to switch to highlight colour.
        parts: list[str] = []
        for word in block.words:
            clean = re.sub(r"[^\w]", "", word.text).lower()
            if cfg.highlight.enabled and clean in keywords:
                parts.append(
                    f"{{\\c{highlight}}}{word.text}{{\\c{primary}}}"
                )
            else:
                parts.append(word.text)

        text = " ".join(parts)
        t_start = _seconds_to_ass_time(block.start)
        t_end = _seconds_to_ass_time(block.end)

        # Alignment 8 = top-centre; 2 = bottom-centre
        # We use MarginV to push up from the bottom.
        lines.append(
            f"Dialogue: 0,{t_start},{t_end},Default,,0,0,0,,{text}"
        )

    output_path.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")
    log.debug("Wrote ASS file: %s (%d events)", output_path.name, len(lines))
