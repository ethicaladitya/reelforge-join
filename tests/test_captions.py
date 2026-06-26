"""Tests for caption block generation and ASS file output."""

from __future__ import annotations

from pathlib import Path

from reelforge.captions import (
    _hex_to_ass,
    _seconds_to_ass_time,
    build_caption_blocks,
    generate_ass,
)
from reelforge.config import CaptionConfig, FontConfig, HighlightConfig
from reelforge.transcriber import Segment, Transcript, Word


def _make_transcript(words: list[tuple[str, float, float]]) -> Transcript:
    w_objs = [Word(t, s, e) for t, s, e in words]
    seg = Segment(" ".join(t for t, *_ in words), w_objs[0].start, w_objs[-1].end, w_objs)
    return Transcript([seg])


def test_hex_to_ass_conversion() -> None:
    assert _hex_to_ass("#FFD400") == "&H0000D4FF"
    assert _hex_to_ass("#FFFFFF") == "&H00FFFFFF"
    assert _hex_to_ass("#000000") == "&H00000000"


def test_seconds_to_ass_time() -> None:
    assert _seconds_to_ass_time(0.0) == "0:00:00.00"
    assert _seconds_to_ass_time(61.5) == "0:01:01.50"
    assert _seconds_to_ass_time(3661.0) == "1:01:01.00"


def test_build_caption_blocks_groups_words() -> None:
    words = [(f"word{i}", float(i), float(i + 1)) for i in range(8)]
    transcript = _make_transcript(words)
    cfg = CaptionConfig(words_per_block=4)
    blocks = build_caption_blocks(transcript, cfg)
    assert len(blocks) == 2
    assert len(blocks[0].words) == 4
    assert len(blocks[1].words) == 4


def test_build_caption_blocks_empty_transcript() -> None:
    cfg = CaptionConfig()
    blocks = build_caption_blocks(Transcript(), cfg)
    assert blocks == []


def test_caption_block_text() -> None:
    words = [("Hello", 0.0, 0.5), ("world", 0.5, 1.0)]
    transcript = _make_transcript(words)
    cfg = CaptionConfig(words_per_block=2)
    blocks = build_caption_blocks(transcript, cfg)
    assert blocks[0].text == "Hello world"


def test_generate_ass_creates_file(tmp_path: Path) -> None:
    words = [("Your", 0.0, 0.3), ("website", 0.3, 0.7), ("can", 0.7, 1.0)]
    transcript = _make_transcript(words)
    cfg = CaptionConfig(
        words_per_block=3,
        highlight=HighlightConfig(enabled=True, color="#FFD400", keywords=["website"]),
    )
    blocks = build_caption_blocks(transcript, cfg)
    ass_path = tmp_path / "test.ass"
    generate_ass(blocks, cfg, ass_path, 1080, 1920)

    assert ass_path.exists()
    content = ass_path.read_text()
    assert "[Events]" in content
    assert "Dialogue:" in content
    # keyword highlight tag should appear
    assert "website" in content


def test_transcript_offset() -> None:
    words = [("Hello", 0.0, 0.5)]
    t = _make_transcript(words)
    offset_t = t.offset(10.0)
    assert offset_t.all_words[0].start == pytest.approx(10.0)
    assert offset_t.all_words[0].end == pytest.approx(10.5)


import pytest
