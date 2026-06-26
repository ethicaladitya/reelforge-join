# ReelForge Architecture

## Overview

ReelForge is a pipeline-based video processing application. Each stage is implemented as a separate, testable module. The pipeline is coordinated by `ReelPipeline` in `pipeline.py`.

All heavy lifting is delegated to FFmpeg via subprocess calls. Python orchestrates the pipeline, builds filter graphs, and manages temporary files.

## Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `cli.py` | Click CLI; parses args, loads config, runs pipeline |
| `config.py` | Pydantic models for all configuration; YAML loader with deep merge |
| `pipeline.py` | Orchestrates all stages; owns the temp directory |
| `clip_discovery.py` | Finds and sorts clips in the input directory |
| `ffmpeg_utils.py` | `probe()`, `run_ffmpeg()`, filter string helpers |
| `silence.py` | `silencedetect` filter; computes per-clip trim points |
| `transcriber.py` | faster-whisper wrapper; returns `Transcript` with word timestamps |
| `captions.py` | Builds caption blocks; generates ASS subtitle file |
| `audio_processor.py` | `loudnorm` filter strings; music ducking filter graph |
| `transitions.py` | `xfade` and `acrossfade` filter graph builders |
| `zoom.py` | `zoompan` filter string for periodic zoom pulses |
| `watermark.py` | Overlay filter strings for watermark and logo |
| `end_card.py` | Renders branded end card via FFmpeg `lavfi` source |
| `exceptions.py` | Custom exception hierarchy |
| `logger.py` | Rich-powered logger; `StepLogger` context manager |

## Data Flow

```
clips/
  01_hook.mp4     ─┐
  02_problem.mp4  ─┤─ ClipMeta (path, duration, trim_start, trim_end, transcript)
  03_solution.mp4 ─┘
         │
         ▼
   [silence trim]  → TrimPoints per clip
         │
         ▼
   [transcribe]    → Transcript (Segments → Words with timestamps)
         │
         ▼
   [normalize]     → loudnorm + afade → tmp/norm_NNN.mp4
         │
         ▼
   [end card]      → tmp/end_card.mp4 (optional)
         │
         ▼
   [concat+xfade]  → tmp/concat.mp4
         │
         ▼
   [zoompan]       → tmp/zoomed.mp4
         │
         ▼
   [ASS captions]  → tmp/captions.ass → tmp/captioned.mp4
         │
         ▼
   [overlay]       → watermark + logo → tmp/overlayed.mp4
         │
         ▼
   [music mix]     → ducking + amix → tmp/with_music.mp4
         │
         ▼
   output/final_reel.mp4
```

## FFmpeg Filter Strategy

ReelForge prefers single-pass FFmpeg invocations with complex filter graphs over chained Python loops. Each stage that can be expressed as a filter graph is implemented that way:

- **Silence trim**: `silencedetect` filter on stderr
- **Scale + pad**: `scale + pad` in one `-vf`
- **Normalize**: `loudnorm` filter (single-pass; two-pass would require JSON parsing)
- **Audio fades**: `afade` filter
- **Transitions**: `xfade` (video) + `acrossfade` (audio) chained
- **Zoom**: `zoompan` filter with expression-based zoom curve
- **Watermark**: `overlay` filter with `colorchannelmixer` for opacity
- **Music ducking**: `volume` filter with `enable` expression + `amix`
- **Captions**: `ass` filter (hardsubbed)

## Configuration Design

Configuration uses Pydantic v2 models with nested validation. The YAML loader performs a deep merge of the default config with any user-provided overrides, meaning users only need to specify the keys they want to change.

## Temporary Files

All intermediate files are written to a `tempfile.TemporaryDirectory` that is automatically cleaned up after the pipeline completes (or fails). Only the final output is written to the user-specified path.

## Error Handling

Each stage raises a specific subclass of `ReelForgeError` on failure. The CLI catches all `ReelForgeError` instances and prints a helpful message. Unexpected exceptions propagate and are shown as a traceback in `--verbose` mode.
