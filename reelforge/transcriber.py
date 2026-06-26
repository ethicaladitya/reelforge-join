"""Speech transcription using faster-whisper with word-level timestamps."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from .config import CaptionConfig
from .exceptions import WhisperError

log = logging.getLogger("reelforge")


@dataclass
class Word:
    text: str
    start: float   # seconds
    end: float     # seconds
    probability: float = 1.0


@dataclass
class Segment:
    text: str
    start: float
    end: float
    words: list[Word] = field(default_factory=list)


@dataclass
class Transcript:
    segments: list[Segment] = field(default_factory=list)

    @property
    def all_words(self) -> list[Word]:
        words: list[Word] = []
        for seg in self.segments:
            words.extend(seg.words)
        return words

    def offset(self, seconds: float) -> "Transcript":
        """Return a new Transcript with all timestamps shifted by *seconds*."""
        new_segs: list[Segment] = []
        for seg in self.segments:
            new_words = [
                Word(w.text, w.start + seconds, w.end + seconds, w.probability)
                for w in seg.words
            ]
            new_segs.append(
                Segment(seg.text, seg.start + seconds, seg.end + seconds, new_words)
            )
        return Transcript(new_segs)


def transcribe(path: Path, cfg: CaptionConfig) -> Transcript:
    """Transcribe *path* using faster-whisper; return word-level Transcript."""
    try:
        from faster_whisper import WhisperModel  # type: ignore[import-untyped]
    except ImportError as exc:
        raise WhisperError(
            "faster-whisper is not installed.\n"
            "  Run: uv add faster-whisper"
        ) from exc

    log.info("Loading Whisper model '%s'…", cfg.model)
    try:
        model = WhisperModel(cfg.model, device="cpu", compute_type="int8")
    except Exception as exc:
        raise WhisperError(f"Failed to load Whisper model '{cfg.model}': {exc}") from exc

    log.info("Transcribing %s…", path.name)
    try:
        segments_iter, _ = model.transcribe(
            str(path),
            language=cfg.language if cfg.language != "auto" else None,
            word_timestamps=True,
            vad_filter=True,
        )
    except Exception as exc:
        raise WhisperError(f"Transcription failed for {path}: {exc}") from exc

    transcript = Transcript()
    for seg in segments_iter:
        words: list[Word] = []
        if seg.words:
            for w in seg.words:
                words.append(Word(w.word.strip(), w.start, w.end, w.probability))
        transcript.segments.append(Segment(seg.text.strip(), seg.start, seg.end, words))

    word_count = sum(len(s.words) for s in transcript.segments)
    log.info("Transcribed %d word(s) in %d segment(s)", word_count, len(transcript.segments))
    return transcript
