# ReelForge

> Convert AI-generated talking-head clips into polished Instagram Reels, TikTok videos, and YouTube Shorts — fully automated.

---

## What it does

You drop a folder of short clips (e.g. from Gemini or any AI video tool) and ReelForge automatically:

- Sorts and concatenates them
- Trims dead air between clips
- Normalizes loudness (EBU R128)
- Transcribes speech with [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- Burns animated, word-highlighted captions (Alex Hormozi style)
- Applies cinematic punch-in zooms
- Adds configurable transitions (fade, dissolve, dip to black)
- Overlays your watermark and logo
- Mixes optional background music with auto-ducking
- Renders a branded end card
- Outputs H.264 1080×1920 @ 30fps — ready for Instagram, TikTok, and YouTube Shorts

---

## Architecture

```
reelforge/
├── cli.py              ← Click CLI entry point
├── config.py           ← Pydantic config models + YAML loader
├── pipeline.py         ← Main orchestration pipeline
├── clip_discovery.py   ← Discover & sort input clips
├── ffmpeg_utils.py     ← FFmpeg probe, run, filter helpers
├── silence.py          ← Silence detection & trim points
├── transcriber.py      ← faster-whisper speech-to-text
├── captions.py         ← ASS subtitle file generator
├── audio_processor.py  ← Loudness normalization & music ducking
├── transitions.py      ← xfade & acrossfade filter builders
├── zoom.py             ← zoompan cinematic zoom filter
├── watermark.py        ← Watermark & logo overlay filters
└── end_card.py         ← Branded end card renderer
```

Pipeline stages (in order):

```
Discover clips
    ↓
Probe & silence trim
    ↓
Transcribe (faster-whisper)
    ↓
Normalize audio (EBU R128)
    ↓
Render end card
    ↓
Concatenate + transitions + zoom
    ↓
Burn ASS captions
    ↓
Apply watermark / logo
    ↓
Mix background music (with ducking)
    ↓
output/final_reel.mp4
```

---

## Requirements

- macOS (Apple Silicon M1+ recommended) or Linux
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- FFmpeg (with libx264 support)

```bash
# macOS
brew install ffmpeg uv

# Linux
sudo apt install ffmpeg
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## Installation

```bash
git clone https://github.com/yourhandle/reelforge.git
cd reelforge

# Install dependencies
uv sync

# Install with dev tools
uv sync --extra dev
```

---

## Quick Start

```bash
# 1. Drop your clips in
cp ~/gemini_clips/*.mp4 clips/

# 2. Run with defaults
uv run reelforge

# 3. Find your reel at
open output/final_reel.mp4
```

---

## Usage

```
Usage: reelforge [OPTIONS]

  ReelForge — convert AI talking-head clips into polished vertical reels.

Options:
  -c, --config PATH   Path to a YAML config file (defaults to config/default.yaml).
  -i, --input PATH    Directory containing input clips (overrides config).
  -o, --output PATH   Output video path (default: output/final_reel.mp4).
  -m, --music PATH    Optional background music file.
  -v, --verbose       Show verbose FFmpeg output.
  -V, --version       Show the version and exit.
  -h, --help          Show this message and exit.
```

### Examples

```bash
# Use defaults
reelforge

# Custom config
reelforge --config my_brand.yaml

# Custom input/output
reelforge --input clips/ --output output/reel_v2.mp4

# With background music
reelforge --music music/lofi_beat.mp3

# Verbose FFmpeg output for debugging
reelforge --verbose
```

---

## Configuration

Everything is controlled through YAML. Start from `config/default.yaml`.

```yaml
brand:
  handle: "@yourhandle"
  end_card:
    enabled: true
    text: "Follow for practical tips."
    duration: 2.5

captions:
  enabled: true
  model: "base"          # tiny | base | small | medium | large-v3
  font:
    family: "Montserrat-Bold"
    size: 68
    color: "#FFFFFF"
    stroke_color: "#000000"
    stroke_width: 4
  words_per_block: 4
  highlight:
    enabled: true
    color: "#FFD400"
    keywords:
      - WordPress
      - Security
      - Backup

music:
  enabled: false
  path: null
  volume: 0.05
  duck_volume: 0.02

zoom:
  enabled: true
  scale: 1.05
  interval: 4.0

transitions:
  type: "fade"           # none | fade | dissolve | dip_to_black
  duration: 0.25

watermark:
  enabled: true
  path: "assets/watermarks/watermark.png"
  position: "bottom_left"
  opacity: 0.6

output:
  width: 1080
  height: 1920
  fps: 30
  crf: 18
  preset: "slow"
```

### Whisper model sizes

| Model    | VRAM  | Speed   | Accuracy |
|----------|-------|---------|----------|
| tiny     | ~1GB  | Fast    | Low      |
| base     | ~1GB  | Fast    | Good     |
| small    | ~2GB  | Medium  | Better   |
| medium   | ~5GB  | Slow    | High     |
| large-v3 | ~10GB | Slowest | Best     |

For talking-head clips with clear speech, `base` is usually sufficient.

---

## Folder Structure

```
reelforge/
├── assets/
│   ├── fonts/           ← Custom fonts (.ttf / .otf)
│   ├── logos/           ← Logo PNGs
│   └── watermarks/      ← Watermark PNGs
├── clips/               ← Drop your input .mp4 files here
│   ├── 01_hook.mp4
│   ├── 02_problem.mp4
│   └── 03_solution.mp4
├── music/               ← Optional background music
├── output/              ← Rendered reels appear here
├── config/
│   └── default.yaml     ← Default configuration
├── reelforge/           ← Source code
├── tests/               ← pytest test suite
└── docs/
    └── architecture.md
```

---

## Adding Your Watermark

1. Create a transparent PNG: `assets/watermarks/watermark.png`
2. Enable in config:

```yaml
watermark:
  enabled: true
  path: "assets/watermarks/watermark.png"
  position: "bottom_left"
  opacity: 0.6
  margin: 40
```

---

## Caption Highlighting

Words matching your keyword list render in the highlight color (default yellow `#FFD400`). Everything else stays white.

```yaml
captions:
  highlight:
    enabled: true
    color: "#FFD400"
    keywords:
      - WordPress
      - Backup
      - Security
      - Hack
      - Malware
```

The matching is case-insensitive.

---

## Running Tests

```bash
uv run pytest
uv run pytest --cov=reelforge --cov-report=html
```

---

## Troubleshooting

### FFmpeg not found

```
Error: FFmpeg is not installed or not on PATH.
```

Install with: `brew install ffmpeg` (macOS) or `apt install ffmpeg` (Linux)

### No clips found

```
Error: No video files found in clips/
```

Ensure your clips are in the `clips/` directory and have a supported extension (`.mp4`, `.mov`, `.webm`, `.mkv`).

### Whisper not installed

```
Error: faster-whisper is not installed.
```

Run: `uv add faster-whisper`

### Captions are misaligned

- Try a larger Whisper model: `model: "small"` or `"medium"`
- Ensure your clips have clear speech without background noise

### Video codec issues

ReelForge will scale and re-encode all clips to `libx264/yuv420p` by default. If you encounter issues with a specific clip, try converting it first:

```bash
ffmpeg -i problem_clip.mp4 -c:v libx264 -pix_fmt yuv420p -c:a aac clips/01_fixed.mp4
```

---

## Roadmap

- [ ] GPU acceleration (VideoToolbox on Apple Silicon)
- [ ] Auto thumbnail generation
- [ ] B-roll overlay support
- [ ] Animated emoji overlays
- [ ] Batch rendering (multiple reels in parallel)
- [ ] Watch mode (`--watch` flag, re-renders on file changes)
- [ ] Auto hashtag generation via LLM
- [ ] Web UI

---

## License

MIT
