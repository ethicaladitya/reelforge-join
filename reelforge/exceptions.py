"""Custom exceptions for ReelForge."""

from __future__ import annotations


class ReelForgeError(Exception):
    """Base exception for all ReelForge errors."""


class FFmpegNotFoundError(ReelForgeError):
    """FFmpeg binary is not installed or not on PATH."""


class WhisperError(ReelForgeError):
    """faster-whisper failed to load or transcribe."""


class NoClipsFoundError(ReelForgeError):
    """No valid video clips were found in the input directory."""


class InvalidConfigError(ReelForgeError):
    """The provided configuration is invalid."""


class FontNotFoundError(ReelForgeError):
    """A required font could not be located."""


class UnsupportedCodecError(ReelForgeError):
    """A clip uses a codec that cannot be processed."""


class ResolutionMismatchError(ReelForgeError):
    """Clip resolution does not match the expected output resolution."""


class RenderError(ReelForgeError):
    """FFmpeg rendering pipeline failed."""
