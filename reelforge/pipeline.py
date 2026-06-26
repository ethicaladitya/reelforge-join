"""Main rendering pipeline — orchestrates all processing stages."""

from __future__ import annotations

import logging
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from .audio_processor import audio_fade_filter, build_music_filter
from .captions import build_caption_blocks, generate_ass
from .clip_discovery import discover_clips
from .config import ReelForgeConfig
from .end_card import render_end_card
from .exceptions import RenderError
from .ffmpeg_utils import loudnorm_filter, probe, run_ffmpeg, scale_pad_filter
from .logger import StepLogger, console
from .silence import compute_trim_points
from .transcriber import Transcript, transcribe
from .transitions import build_audio_crossfade, build_xfade_filter
from .watermark import logo_filter_inputs, watermark_filter_inputs

log = logging.getLogger("reelforge")


@dataclass
class ClipMeta:
    path: Path
    duration: float
    trim_start: float
    trim_end: float
    transcript: Transcript | None = None

    @property
    def trimmed_duration(self) -> float:
        return self.trim_end - self.trim_start


@dataclass
class PipelineResult:
    output_path: Path
    total_duration: float
    clip_count: int
    word_count: int


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class ReelPipeline:
    """Orchestrates the full clip-to-reel rendering pipeline."""

    def __init__(
        self,
        cfg: ReelForgeConfig,
        *,
        clips_dir: Path | None = None,
        output_path: Path | None = None,
        music_path: Path | None = None,
        verbose: bool = False,
        base_dir: Path | None = None,
    ) -> None:
        self.cfg = cfg
        self.verbose = verbose
        self.base_dir = base_dir or Path.cwd()
        self.clips_dir = clips_dir or (self.base_dir / cfg.clips.directory)
        self.output_path = output_path or (self.base_dir / "output" / "final_reel.mp4")
        self.music_path = music_path

        if music_path and not cfg.music.enabled:
            cfg.music.enabled = True
            cfg.music.path = str(music_path)

    # ------------------------------------------------------------------

    def run(self) -> PipelineResult:
        with tempfile.TemporaryDirectory(prefix="reelforge_") as tmpdir:
            tmp = Path(tmpdir)
            self._tmp = tmp

            with StepLogger("Discover clips", log):
                clips_paths = discover_clips(self.clips_dir, self.cfg.clips)

            with StepLogger("Probe & trim silence", log):
                metas = self._probe_and_trim(clips_paths)

            with StepLogger("Transcribe audio", log) if self.cfg.captions.enabled else _noop():
                if self.cfg.captions.enabled:
                    metas = self._transcribe_clips(metas, tmp)

            with StepLogger("Normalize audio", log):
                normalized = self._normalize_clips(metas, tmp)

            with StepLogger("Render end card", log) if self.cfg.brand.end_card.enabled else _noop():
                if self.cfg.brand.end_card.enabled:
                    ec_path = tmp / "end_card.mp4"
                    render_end_card(self.cfg.brand, self.cfg.output, ec_path)
                    if ec_path.exists():
                        ec_meta = ClipMeta(
                            path=ec_path,
                            duration=self.cfg.brand.end_card.duration,
                            trim_start=0.0,
                            trim_end=self.cfg.brand.end_card.duration,
                        )
                        normalized.append(ec_meta)

            with StepLogger("Concatenate & composite", log):
                merged = self._concat_and_composite(normalized, tmp)

            with StepLogger("Burn captions", log) if self.cfg.captions.enabled else _noop():
                if self.cfg.captions.enabled:
                    captioned = self._burn_captions(merged, metas, tmp)
                else:
                    captioned = merged

            with StepLogger("Mix music & finalize", log):
                final = self._finalize(captioned, metas, tmp)

            # Move to output
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(final, self.output_path)

        total_dur = sum(m.trimmed_duration for m in metas)
        word_count = sum(
            len(m.transcript.all_words) for m in metas if m.transcript is not None
        )

        console.print(f"\n[bold green]✔ Reel rendered:[/bold green] {self.output_path}")
        console.print(
            f"   Clips: {len(metas)}  |  Duration: {total_dur:.1f}s  |  Words: {word_count}"
        )

        return PipelineResult(
            output_path=self.output_path,
            total_duration=total_dur,
            clip_count=len(metas),
            word_count=word_count,
        )

    # ------------------------------------------------------------------
    # Stage implementations
    # ------------------------------------------------------------------

    def _probe_and_trim(self, paths: list[Path]) -> list[ClipMeta]:
        metas: list[ClipMeta] = []
        for path in paths:
            info = probe(path)
            dur = info.duration
            trim = compute_trim_points(path, self.cfg.silence, dur)
            metas.append(ClipMeta(path=path, duration=dur, trim_start=trim.start, trim_end=trim.end))
            log.debug("%s  →  %.2fs trimmed (was %.2fs)", path.name, trim.end - trim.start, dur)
        return metas

    def _transcribe_clips(self, metas: list[ClipMeta], tmp: Path) -> list[ClipMeta]:
        offset = 0.0
        for meta in metas:
            t = transcribe(meta.path, self.cfg.captions)
            meta.transcript = t.offset(offset)
            offset += meta.trimmed_duration
        return metas

    def _normalize_clips(self, metas: list[ClipMeta], tmp: Path) -> list[ClipMeta]:
        """Normalize audio loudness for each clip and apply tiny audio fades."""
        result: list[ClipMeta] = []
        out_cfg = self.cfg.output
        audio_cfg = self.cfg.audio

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Normalizing…", total=len(metas))

            for i, meta in enumerate(metas):
                out_path = tmp / f"norm_{i:03d}.mp4"
                dur = meta.trimmed_duration
                fade_filter = audio_fade_filter(dur)
                norm_filter = loudnorm_filter(
                    audio_cfg.loudness_target, audio_cfg.true_peak, audio_cfg.lra
                )
                scale_filter = scale_pad_filter(out_cfg.width, out_cfg.height)

                run_ffmpeg(
                    [
                        "-ss", str(meta.trim_start),
                        "-to", str(meta.trim_end),
                        "-i", str(meta.path),
                        "-vf", scale_filter,
                        "-af", f"{norm_filter},{fade_filter}",
                        "-r", str(out_cfg.fps),
                        "-c:v", out_cfg.codec,
                        "-crf", str(out_cfg.crf),
                        "-preset", out_cfg.preset,
                        "-c:a", out_cfg.audio_codec,
                        "-b:a", out_cfg.audio_bitrate,
                        "-pix_fmt", out_cfg.pixel_format,
                        str(out_path),
                    ],
                    description=f"Normalize {meta.path.name}",
                    verbose=self.verbose,
                )
                new_meta = ClipMeta(
                    path=out_path,
                    duration=dur,
                    trim_start=0.0,
                    trim_end=dur,
                    transcript=meta.transcript,
                )
                result.append(new_meta)
                progress.advance(task)

        return result

    def _concat_and_composite(self, metas: list[ClipMeta], tmp: Path) -> Path:
        """Concatenate clips with transitions and apply zoom effect."""
        out_cfg = self.cfg.output
        trans_cfg = self.cfg.transitions
        zoom_cfg = self.cfg.zoom
        output = tmp / "concat.mp4"

        durations = [m.trimmed_duration for m in metas]
        n = len(metas)

        # Build input list
        inputs: list[str] = []
        for m in metas:
            inputs.extend(["-i", str(m.path)])

        if trans_cfg.type == "none" or n == 1:
            # Simple concat using concat demuxer
            list_file = tmp / "concat_list.txt"
            list_file.write_text(
                "\n".join(f"file '{m.path}'" for m in metas)
            )
            run_ffmpeg(
                [
                    "-f", "concat",
                    "-safe", "0",
                    "-i", str(list_file),
                    "-c:v", out_cfg.codec,
                    "-crf", str(out_cfg.crf),
                    "-preset", out_cfg.preset,
                    "-c:a", out_cfg.audio_codec,
                    "-b:a", out_cfg.audio_bitrate,
                    "-pix_fmt", out_cfg.pixel_format,
                    str(output),
                ],
                description="Concatenate clips",
                verbose=self.verbose,
            )
        else:
            # xfade transitions
            v_fc, v_out = build_xfade_filter(n, durations, trans_cfg)
            a_fc, a_out = build_audio_crossfade(n, durations, trans_cfg.duration)

            filter_complex = f"{v_fc};{a_fc}"

            run_ffmpeg(
                inputs + [
                    "-filter_complex", filter_complex,
                    "-map", f"[{v_out}]",
                    "-map", f"[{a_out}]",
                    "-c:v", out_cfg.codec,
                    "-crf", str(out_cfg.crf),
                    "-preset", out_cfg.preset,
                    "-c:a", out_cfg.audio_codec,
                    "-b:a", out_cfg.audio_bitrate,
                    "-pix_fmt", out_cfg.pixel_format,
                    str(output),
                ],
                description="Concatenate with transitions",
                verbose=self.verbose,
            )

        # Apply zoom if enabled
        if zoom_cfg.enabled:
            zoomed = tmp / "zoomed.mp4"
            from .zoom import build_zoom_filter
            total_dur = sum(durations)
            zoom_vf = build_zoom_filter(total_dur, zoom_cfg, out_cfg.width, out_cfg.height)
            run_ffmpeg(
                [
                    "-i", str(output),
                    "-vf", zoom_vf,
                    "-c:a", "copy",
                    "-c:v", out_cfg.codec,
                    "-crf", str(out_cfg.crf),
                    "-preset", "fast",  # zoompan is slow; use fast preset here
                    "-pix_fmt", out_cfg.pixel_format,
                    str(zoomed),
                ],
                description="Apply zoom effect",
                verbose=self.verbose,
            )
            return zoomed

        return output

    def _burn_captions(self, video: Path, metas: list[ClipMeta], tmp: Path) -> Path:
        """Generate ASS file and burn subtitles into video."""
        all_transcripts = [m.transcript for m in metas if m.transcript is not None]
        if not all_transcripts:
            log.warning("No transcripts available — skipping captions")
            return video

        from .transcriber import Transcript as T
        combined = T(segments=[s for t in all_transcripts for s in t.segments])
        blocks = build_caption_blocks(combined, self.cfg.captions)

        if not blocks:
            log.warning("No caption blocks generated")
            return video

        ass_path = tmp / "captions.ass"
        generate_ass(blocks, self.cfg.captions, ass_path, self.cfg.output.width, self.cfg.output.height)

        out_cfg = self.cfg.output
        output = tmp / "captioned.mp4"

        run_ffmpeg(
            [
                "-i", str(video),
                "-vf", f"ass={ass_path}",
                "-c:a", "copy",
                "-c:v", out_cfg.codec,
                "-crf", str(out_cfg.crf),
                "-preset", out_cfg.preset,
                "-pix_fmt", out_cfg.pixel_format,
                str(output),
            ],
            description="Burn captions",
            verbose=self.verbose,
        )
        return output

    def _apply_overlays(self, video: Path, tmp: Path) -> Path:
        """Apply watermark and logo overlays."""
        out_cfg = self.cfg.output
        extra_inputs: list[str] = []
        filter_parts: list[str] = []
        current_label = "v_base"
        out_label = current_label

        filter_parts.append(f"[0:v]{current_label}")

        wm = watermark_filter_inputs(self.cfg.watermark, self.base_dir)
        if wm:
            wm_inputs, wm_chain = wm
            idx = 1 + len(extra_inputs) // 2
            wm_chain = wm_chain.replace("[wm_in]", f"[{idx}:v]")
            wm_chain = wm_chain.replace("[base]", f"[{current_label}]")
            out_label = "with_wm"
            wm_chain = wm_chain.replace("[with_wm]", f"[{out_label}]")
            filter_parts.append(wm_chain)
            extra_inputs.extend(wm_inputs)
            current_label = out_label

        logo = logo_filter_inputs(self.cfg.logo, self.base_dir, out_cfg.width)
        if logo:
            logo_inputs, logo_chain = logo
            idx = 1 + len(extra_inputs) // 2
            logo_chain = logo_chain.replace("[logo_in]", f"[{idx}:v]")
            logo_chain = logo_chain.replace("[prev]", f"[{current_label}]")
            out_label = "with_logo"
            logo_chain = logo_chain.replace("[with_logo]", f"[{out_label}]")
            filter_parts.append(logo_chain)
            extra_inputs.extend(logo_inputs)
            current_label = out_label

        if not extra_inputs:
            return video

        output = tmp / "overlayed.mp4"
        filter_complex = ";".join(filter_parts)

        run_ffmpeg(
            ["-i", str(video)]
            + extra_inputs
            + [
                "-filter_complex", filter_complex,
                "-map", f"[{out_label}]",
                "-map", "0:a",
                "-c:v", out_cfg.codec,
                "-crf", str(out_cfg.crf),
                "-preset", out_cfg.preset,
                "-c:a", "copy",
                "-pix_fmt", out_cfg.pixel_format,
                str(output),
            ],
            description="Apply overlays",
            verbose=self.verbose,
        )
        return output

    def _finalize(self, video: Path, metas: list[ClipMeta], tmp: Path) -> Path:
        """Apply overlays, mix music, and produce the final output file."""
        # Overlays (watermark / logo)
        overlayed = self._apply_overlays(video, tmp)

        if not self.cfg.music.enabled:
            return overlayed

        # Background music
        music_src = self.music_path or (
            Path(self.cfg.music.path) if self.cfg.music.path else None
        )
        if not music_src or not music_src.exists():
            log.warning("Music enabled but file not found — skipping music")
            return overlayed

        total_dur = sum(m.trimmed_duration for m in metas)

        # Collect speech intervals for ducking
        speech_intervals: list[tuple[float, float]] = []
        for m in metas:
            if m.transcript:
                for seg in m.transcript.segments:
                    speech_intervals.append((seg.start, seg.end))

        music_filter = build_music_filter(
            total_dur, music_src, speech_intervals, self.cfg.music
        )

        out_cfg = self.cfg.output
        output = tmp / "with_music.mp4"

        run_ffmpeg(
            [
                "-i", str(overlayed),
                "-i", str(music_src),
                "-filter_complex", music_filter,
                "-map", "0:v",
                "-map", "[aout]",
                "-c:v", "copy",
                "-c:a", out_cfg.audio_codec,
                "-b:a", out_cfg.audio_bitrate,
                "-t", str(total_dur),
                str(output),
            ],
            description="Mix background music",
            verbose=self.verbose,
        )
        return output


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _noop:
    """No-op context manager used to skip optional stages cleanly."""

    def __enter__(self) -> "_noop":
        return self

    def __exit__(self, *_: object) -> None:
        pass
