"""Rich-powered logger for ReelForge."""

from __future__ import annotations

import logging
from typing import ClassVar

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

_THEME = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "bold red",
        "success": "bold green",
        "step": "bold magenta",
    }
)

console = Console(theme=_THEME)


def get_logger(name: str = "reelforge", verbose: bool = False) -> logging.Logger:
    """Return a configured logger with Rich output."""
    level = logging.DEBUG if verbose else logging.INFO

    handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
    )
    handler.setLevel(level)

    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


class StepLogger:
    """Context-manager helper that prints a step header and timing."""

    _instances: ClassVar[list["StepLogger"]] = []

    def __init__(self, label: str, logger: logging.Logger) -> None:
        self._label = label
        self._log = logger

    def __enter__(self) -> "StepLogger":
        import time

        self._start = time.perf_counter()
        self._log.info(f"[step]▶  {self._label}[/step]")
        return self

    def __exit__(self, *_: object) -> None:
        import time

        elapsed = time.perf_counter() - self._start
        self._log.info(f"[success]✔  {self._label}[/success] ({elapsed:.1f}s)")
