"""CLI entry point for ReelForge."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import click
from rich.panel import Panel

from . import __version__
from .config import load_config
from .exceptions import ReelForgeError
from .ffmpeg_utils import require_ffmpeg
from .logger import console, get_logger
from .pipeline import ReelPipeline


@click.group(context_settings={"help_option_names": ["-h", "--help"]}, invoke_without_command=True)
@click.version_option(__version__, "-V", "--version")
@click.pass_context
def main(ctx: click.Context) -> None:
    """ReelForge — convert AI talking-head clips into polished vertical reels.

    Run without a subcommand to render directly from the command line,
    or use 'reelforge ui' to open the web interface.
    """
    if ctx.invoked_subcommand is None:
        ctx.invoke(render)


# ---------------------------------------------------------------------------
# UI subcommand
# ---------------------------------------------------------------------------


@main.command(name="ui")
@click.option("--host", default="127.0.0.1", show_default=True, help="Host to bind to.")
@click.option("--port", default=7433, show_default=True, help="Port to listen on.")
@click.option("--open/--no-open", "open_browser", default=True, help="Open browser automatically.")
def ui_cmd(host: str, port: int, open_browser: bool) -> None:
    """Launch the ReelForge web UI."""
    import threading
    import webbrowser

    import uvicorn

    url = f"http://{host}:{port}"
    get_logger("reelforge")

    console.print(
        Panel.fit(
            f"[bold cyan]ReelForge UI[/bold cyan]\n"
            f"[dim]Open →[/dim] [cyan]{url}[/cyan]\n"
            "[dim]Press Ctrl+C to stop[/dim]",
            border_style="cyan",
        )
    )

    if open_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    uvicorn.run(
        "reelforge.server:app",
        host=host,
        port=port,
        log_level="warning",
    )


# ---------------------------------------------------------------------------
# Render subcommand (also called when no subcommand given)
# ---------------------------------------------------------------------------


@main.command(name="render")
@click.option("--config", "-c", type=click.Path(path_type=Path), default=None,
              help="Path to a YAML config file (defaults to config/default.yaml).")
@click.option("--input", "-i", "input_dir",
              type=click.Path(path_type=Path, exists=True, file_okay=False), default=None,
              help="Directory containing input clips (overrides config).")
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None,
              help="Output video path (default: output/final_reel.mp4).")
@click.option("--music", "-m", type=click.Path(path_type=Path, exists=True), default=None,
              help="Optional background music file.")
@click.option("--verbose", "-v", is_flag=True, default=False, help="Show verbose FFmpeg output.")
def render(
    config: Path | None,
    input_dir: Path | None,
    output: Path | None,
    music: Path | None,
    verbose: bool,
) -> None:
    """Render a reel from the command line.

    \b
    Examples:
      reelforge render
      reelforge render --config my.yaml
      reelforge render --input clips/ --output output/reel.mp4
      reelforge render --music music/song.mp3 --verbose
    """
    log = get_logger("reelforge", verbose=verbose)

    console.print(
        Panel.fit(
            f"[bold cyan]ReelForge[/bold cyan] v{__version__}\n"
            "[dim]AI clips → polished vertical reels[/dim]",
            border_style="cyan",
        )
    )

    try:
        ffmpeg_path, _ = require_ffmpeg()
        log.debug("FFmpeg found: %s", ffmpeg_path)
    except ReelForgeError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)

    try:
        cfg = load_config(config)
    except Exception as exc:
        console.print(f"[bold red]Config error:[/bold red] {exc}")
        sys.exit(1)

    t0 = time.perf_counter()
    try:
        pipeline = ReelPipeline(
            cfg,
            clips_dir=input_dir,
            output_path=output,
            music_path=music,
            verbose=verbose,
        )
        pipeline.run()
    except ReelForgeError as exc:
        console.print(f"\n[bold red]✘ Error:[/bold red] {exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
        sys.exit(130)
    except Exception as exc:
        console.print(f"\n[bold red]✘ Unexpected error:[/bold red] {exc}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    elapsed = time.perf_counter() - t0
    console.print(f"\n[dim]Total time: {elapsed:.1f}s[/dim]")


if __name__ == "__main__":
    main()
