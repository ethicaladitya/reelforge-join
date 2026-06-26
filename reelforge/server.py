"""FastAPI server for the ReelForge web UI."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import aiofiles
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from .config import load_config
from .exceptions import ReelForgeError
from .ffmpeg_utils import probe, require_ffmpeg

log = logging.getLogger("reelforge")

# ---------------------------------------------------------------------------
# Job registry (in-memory; single-server use)
# ---------------------------------------------------------------------------

_jobs: dict[str, dict[str, Any]] = {}
_job_lock = threading.Lock()


def _new_job() -> str:
    jid = str(uuid.uuid4())
    with _job_lock:
        _jobs[jid] = {
            "id": jid,
            "status": "pending",   # pending | running | done | error
            "progress": 0,
            "stage": "",
            "log": [],
            "output": None,
            "error": None,
            "started_at": None,
            "finished_at": None,
        }
    return jid


def _update_job(jid: str, **kwargs: Any) -> None:
    with _job_lock:
        if jid in _jobs:
            _jobs[jid].update(kwargs)


def _get_job(jid: str) -> dict[str, Any]:
    with _job_lock:
        job = _jobs.get(jid)
        if job is None:
            raise KeyError(jid)
        return dict(job)


# ---------------------------------------------------------------------------
# Progress-capturing pipeline runner
# ---------------------------------------------------------------------------


class ProgressLogger(logging.Handler):
    """Capture log records and append to a job's log list."""

    def __init__(self, jid: str) -> None:
        super().__init__()
        self._jid = jid

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        with _job_lock:
            if self._jid in _jobs:
                _jobs[self._jid]["log"].append(msg)


_STAGES = [
    ("Discover clips", 5),
    ("Probe & trim silence", 10),
    ("Transcribe audio", 30),
    ("Normalize audio", 50),
    ("Render end card", 58),
    ("Concatenate & composite", 70),
    ("Burn captions", 83),
    ("Mix music & finalize", 93),
]

_STAGE_PROGRESS = {name: pct for name, pct in _STAGES}


def _run_pipeline_thread(
    jid: str,
    clips_dir: Path,
    output_path: Path,
    config_overrides: dict[str, Any],
    music_path: Path | None,
    base_dir: Path,
) -> None:
    """Run the pipeline in a background thread and update job state."""
    _update_job(jid, status="running", started_at=time.time())

    # Attach progress logger to reelforge logger
    handler = ProgressLogger(jid)
    handler.setFormatter(logging.Formatter("%(message)s"))
    rf_log = logging.getLogger("reelforge")
    rf_log.addHandler(handler)

    # Monkey-patch StepLogger to emit progress updates
    import reelforge.logger as _rl

    _original_enter = _rl.StepLogger.__enter__
    _original_exit = _rl.StepLogger.__exit__

    def _patched_enter(self: _rl.StepLogger) -> _rl.StepLogger:
        pct = _STAGE_PROGRESS.get(self._label, None)
        if pct is not None:
            _update_job(jid, stage=self._label, progress=pct)
        return _original_enter(self)

    _rl.StepLogger.__enter__ = _patched_enter  # type: ignore[method-assign]

    try:
        from .config import ReelForgeConfig, _deep_merge
        import yaml

        base_cfg_path = base_dir / "config" / "default.yaml"
        base_raw: dict[str, Any] = {}
        if base_cfg_path.exists():
            with base_cfg_path.open() as f:
                base_raw = yaml.safe_load(f) or {}

        merged = _deep_merge(base_raw, config_overrides)
        cfg = ReelForgeConfig.model_validate(merged)

        from .pipeline import ReelPipeline

        pipeline = ReelPipeline(
            cfg,
            clips_dir=clips_dir,
            output_path=output_path,
            music_path=music_path,
            verbose=False,
            base_dir=base_dir,
        )
        result = pipeline.run()

        _update_job(
            jid,
            status="done",
            progress=100,
            stage="Complete",
            output=str(result.output_path),
            finished_at=time.time(),
        )

    except ReelForgeError as exc:
        _update_job(jid, status="error", error=str(exc), finished_at=time.time())
    except Exception as exc:
        import traceback
        _update_job(
            jid,
            status="error",
            error=f"{exc}\n{traceback.format_exc()}",
            finished_at=time.time(),
        )
    finally:
        rf_log.removeHandler(handler)
        _rl.StepLogger.__enter__ = _original_enter  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


BASE_DIR = Path(__file__).parent.parent
UPLOADS_DIR = BASE_DIR / "output" / "_uploads"
OUTPUTS_DIR = BASE_DIR / "output"


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="ReelForge", version="0.1.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Static HTML UI (single-file, no build step)
# ---------------------------------------------------------------------------

UI_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>ReelForge</title>
<style>
  :root {
    --bg: #0f0f13;
    --surface: #1a1a23;
    --surface2: #23232f;
    --accent: #FFD400;
    --accent2: #ff6b35;
    --text: #f0f0f5;
    --muted: #7a7a90;
    --border: #2e2e3e;
    --success: #4ade80;
    --error: #f87171;
    --radius: 12px;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    display: flex;
    flex-direction: column;
  }

  /* Header */
  header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 18px 32px;
    border-bottom: 1px solid var(--border);
    background: var(--surface);
  }
  header .logo {
    font-size: 22px;
    font-weight: 800;
    letter-spacing: -0.5px;
  }
  header .logo span { color: var(--accent); }
  header .tagline { font-size: 13px; color: var(--muted); margin-left: 4px; }
  header nav { margin-left: auto; display: flex; gap: 8px; align-items: center; }
  header nav a {
    font-size: 13px; font-weight: 600; color: var(--muted);
    text-decoration: none; padding: 6px 14px; border-radius: 8px;
    border: 1px solid var(--border); transition: all 0.15s;
  }
  header nav a:hover { color: var(--accent); border-color: var(--accent); }
  header .badge {
    font-size: 11px;
    background: var(--accent);
    color: #000;
    padding: 3px 8px;
    border-radius: 20px;
    font-weight: 700;
    letter-spacing: 0.3px;
  }

  /* Layout */
  main {
    flex: 1;
    display: grid;
    grid-template-columns: 1fr 380px;
    gap: 0;
    height: calc(100vh - 63px);
  }
  .left-panel { padding: 28px 32px; overflow-y: auto; display: flex; flex-direction: column; gap: 24px; }
  .right-panel { border-left: 1px solid var(--border); background: var(--surface); padding: 28px 24px; overflow-y: auto; display: flex; flex-direction: column; gap: 20px; }

  /* Cards */
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px;
  }
  .card-title {
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: var(--muted);
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 6px;
  }

  /* Drop zone */
  #drop-zone {
    border: 2px dashed var(--border);
    border-radius: var(--radius);
    padding: 40px 24px;
    text-align: center;
    cursor: pointer;
    transition: all 0.2s;
    background: var(--surface2);
  }
  #drop-zone.drag-over { border-color: var(--accent); background: rgba(255,212,0,0.05); }
  #drop-zone .drop-icon { font-size: 40px; margin-bottom: 10px; }
  #drop-zone p { color: var(--muted); font-size: 14px; }
  #drop-zone strong { color: var(--text); }
  #file-input { display: none; }

  /* Clip list */
  #clip-list { display: flex; flex-direction: column; gap: 8px; margin-top: 14px; }
  .clip-item {
    display: flex;
    align-items: center;
    gap: 10px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 12px;
    font-size: 13px;
  }
  .clip-item .clip-icon { font-size: 18px; flex-shrink: 0; }
  .clip-item .clip-name { flex: 1; font-weight: 500; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--text); }
  .clip-item .clip-size { color: var(--muted); font-size: 11px; flex-shrink: 0; }
  .clip-item .clip-remove { cursor: pointer; color: var(--muted); font-size: 16px; padding: 2px 6px; border-radius: 4px; border: none; background: none; color: var(--error); flex-shrink: 0; }
  .clip-item .clip-remove:hover { background: rgba(248,113,113,0.12); }
  .clip-item .clip-order {
    width: 22px;
    height: 22px;
    border-radius: 50%;
    background: var(--accent);
    color: #000;
    font-size: 11px;
    font-weight: 800;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
  }

  /* Settings */
  .setting-row { display: flex; flex-direction: column; gap: 4px; margin-bottom: 12px; }
  .setting-row label { font-size: 12px; color: var(--muted); font-weight: 600; }
  .setting-row input[type=text],
  .setting-row input[type=number],
  .setting-row select {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--text);
    padding: 8px 10px;
    font-size: 13px;
    width: 100%;
    outline: none;
    transition: border-color 0.15s;
  }
  .setting-row input:focus,
  .setting-row select:focus { border-color: var(--accent); }
  .setting-row input[type=checkbox] { width: 16px; height: 16px; accent-color: var(--accent); }
  .toggle-row { flex-direction: row; align-items: center; justify-content: space-between; }

  /* Music */
  #music-drop {
    border: 2px dashed var(--border);
    border-radius: 8px;
    padding: 16px;
    text-align: center;
    cursor: pointer;
    font-size: 13px;
    color: var(--muted);
    transition: all 0.2s;
  }
  #music-drop.has-file { border-color: var(--success); color: var(--success); }
  #music-drop.drag-over { border-color: var(--accent); }

  /* Render button */
  #render-btn {
    width: 100%;
    padding: 14px;
    background: var(--accent);
    color: #000;
    border: none;
    border-radius: var(--radius);
    font-size: 15px;
    font-weight: 800;
    cursor: pointer;
    letter-spacing: 0.3px;
    transition: all 0.15s;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
  }
  #render-btn:hover { background: #ffe033; transform: translateY(-1px); }
  #render-btn:disabled { background: var(--border); color: var(--muted); transform: none; cursor: not-allowed; }

  /* Progress */
  #progress-panel { display: none; }
  #progress-panel.visible { display: block; }
  .progress-bar-wrap { background: var(--surface2); border-radius: 999px; height: 8px; overflow: hidden; margin: 10px 0; }
  .progress-bar-fill { height: 100%; background: linear-gradient(90deg, var(--accent), var(--accent2)); border-radius: 999px; transition: width 0.4s ease; width: 0%; }
  .stage-label { font-size: 13px; color: var(--muted); }
  .log-box {
    background: #0a0a0e;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px;
    height: 180px;
    overflow-y: auto;
    font-family: "SF Mono", "Fira Code", monospace;
    font-size: 11px;
    color: #a0a0b8;
    line-height: 1.6;
  }
  .log-box .log-line { white-space: pre-wrap; word-break: break-all; }
  .log-box .log-error { color: var(--error); }
  .log-box .log-success { color: var(--success); }

  /* Output video */
  #output-panel { display: none; }
  #output-panel.visible { display: block; }
  #output-video {
    width: 100%;
    border-radius: var(--radius);
    background: #000;
    max-height: 400px;
  }
  .output-actions { display: flex; gap: 8px; margin-top: 10px; }
  .btn-secondary {
    flex: 1;
    padding: 10px;
    background: var(--surface2);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 8px;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    text-align: center;
    text-decoration: none;
    transition: all 0.15s;
  }
  .btn-secondary:hover { border-color: var(--accent); color: var(--accent); }

  /* Status pill */
  .status-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-size: 12px;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 999px;
  }
  .status-pill.running { background: rgba(255,212,0,0.15); color: var(--accent); }
  .status-pill.done { background: rgba(74,222,128,0.15); color: var(--success); }
  .status-pill.error { background: rgba(248,113,113,0.15); color: var(--error); }
  .pulse { animation: pulse 1.2s ease-in-out infinite; }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }

  /* Drag handles */
  .clip-item { cursor: grab; }
  .clip-item.dragging { opacity: 0.4; }
  .clip-item.drag-target { border-color: var(--accent); }

  /* Empty state */
  .empty-state { text-align: center; padding: 32px 0; color: var(--muted); font-size: 13px; }

  /* Responsive */
  @media (max-width: 900px) {
    main { grid-template-columns: 1fr; }
    .right-panel { border-left: none; border-top: 1px solid var(--border); }
  }
</style>
</head>
<body>

<header>
  <div class="logo">Reel<span>Forge</span></div>
  <div class="tagline">AI clips → polished vertical reels</div>
  <nav>
    <a href="/" style="color:var(--accent);border-color:var(--accent);">Render</a>
    <a href="/library">Library</a>
  </nav>
  <div class="badge">v0.1.0</div>
</header>

<main>
  <!-- LEFT: Clips + Settings -->
  <div class="left-panel">

    <!-- Drop zone -->
    <div>
      <div class="card-title">🎬 Input Clips</div>
      <div id="drop-zone">
        <div class="drop-icon">📁</div>
        <p><strong>Drop video clips here</strong></p>
        <p style="margin-top:6px">or click to browse · MP4, MOV, WebM, MKV</p>
        <p style="margin-top:8px;font-size:12px">Files appear in the order you drop them — drag rows to reorder</p>
      </div>
      <input type="file" id="file-input" multiple accept=".mp4,.mov,.webm,.mkv"/>
      <div style="display:flex;align-items:center;justify-content:space-between;margin-top:12px;margin-bottom:4px;min-height:28px;" id="clip-list-header" style="display:none">
        <span style="font-size:11px;color:var(--muted);" id="clip-count"></span>
        <button id="sort-btn" onclick="toggleSort()" style="font-size:11px;font-weight:600;padding:4px 10px;border-radius:6px;border:1px solid var(--border);background:var(--surface2);color:var(--muted);cursor:pointer;transition:all 0.15s;">⇅ Auto-sort</button>
      </div>
      <div id="clip-list"></div>
    </div>

    <!-- Settings -->
    <div class="card">
      <div class="card-title">⚙️ Settings</div>

      <div class="setting-row">
        <label>Brand Handle</label>
        <input type="text" id="s-handle" value="@ethicaladitya" placeholder="@yourhandle"/>
      </div>

      <div class="setting-row">
        <label>Whisper Model</label>
        <select id="s-model">
          <option value="tiny">tiny — fastest, lower accuracy</option>
          <option value="base" selected>base — fast, good accuracy ✓</option>
          <option value="small">small — medium speed, better accuracy</option>
          <option value="medium">medium — slower, high accuracy</option>
          <option value="large-v3">large-v3 — best accuracy, slow</option>
        </select>
      </div>

      <div class="setting-row">
        <label>Transition</label>
        <select id="s-transition">
          <option value="fade" selected>Fade</option>
          <option value="dissolve">Cross Dissolve</option>
          <option value="dip_to_black">Dip to Black</option>
          <option value="none">None</option>
        </select>
      </div>

      <div class="setting-row toggle-row">
        <label>Captions</label>
        <input type="checkbox" id="s-captions" checked/>
      </div>

      <div class="setting-row toggle-row">
        <label>Zoom Effects</label>
        <input type="checkbox" id="s-zoom" checked/>
      </div>

      <div class="setting-row toggle-row">
        <label>Watermark</label>
        <input type="checkbox" id="s-watermark" checked/>
      </div>

      <div class="setting-row toggle-row">
        <label>End Card</label>
        <input type="checkbox" id="s-endcard" checked/>
      </div>

      <div class="setting-row">
        <label>Highlight Color</label>
        <input type="text" id="s-highlight-color" value="#FFD400" placeholder="#FFD400"/>
      </div>

      <div class="setting-row">
        <label>Highlight Keywords (comma-separated)</label>
        <input type="text" id="s-keywords" value="WordPress,Backup,Security,Hack,Malware,Plugin,Speed,SSL,Database"/>
      </div>

      <!-- Music -->
      <div class="setting-row" style="margin-top:6px">
        <label>Background Music (optional)</label>
        <div id="music-drop">🎵 Drop MP3/WAV here or click to browse</div>
        <input type="file" id="music-input" accept=".mp3,.wav,.aac,.m4a" style="display:none"/>
      </div>

    </div>

  </div>

  <!-- RIGHT: Render + Output -->
  <div class="right-panel">

    <div>
      <button id="render-btn" onclick="startRender()">
        ▶ Render Reel
      </button>
    </div>

    <!-- Progress -->
    <div id="progress-panel">
      <div class="card-title">
        <span id="status-pill" class="status-pill running">
          <span class="pulse">●</span> <span id="status-text">Running</span>
        </span>
      </div>
      <div class="stage-label" id="stage-label">Starting…</div>
      <div class="progress-bar-wrap">
        <div class="progress-bar-fill" id="progress-fill"></div>
      </div>
      <div style="font-size:12px;color:var(--muted);margin-bottom:8px" id="progress-pct">0%</div>
      <div class="log-box" id="log-box"></div>
    </div>

    <!-- Output -->
    <div id="output-panel">
      <div class="card-title">✅ Output</div>
      <video id="output-video" controls playsinline></video>
      <div class="output-actions">
        <a id="download-btn" class="btn-secondary" download="final_reel.mp4">⬇ Download</a>
        <a href="/library" class="btn-secondary">📚 Library</a>
        <button class="btn-secondary" onclick="resetUI()">🔄 New Reel</button>
      </div>
    </div>

    <!-- Help -->
    <div class="card" id="help-card">
      <div class="card-title">💡 Quick Start</div>
      <ol style="font-size:13px;color:var(--muted);padding-left:18px;line-height:2">
        <li>Drop your clips (01_hook.mp4, 02_…)</li>
        <li>Adjust settings if needed</li>
        <li>Click <strong style="color:var(--accent)">Render Reel</strong></li>
        <li>Watch the magic happen</li>
        <li>Download your reel</li>
      </ol>
    </div>

  </div>
</main>

<script>
// ─────────────────────────────────────────────
// State
// ─────────────────────────────────────────────
let clips = [];       // {file, name, size, uploadedName}
let musicFile = null;
let currentJobId = null;
let ws = null;
let sessionId = crypto.randomUUID(); // fresh session per page load
let autoSort = false;

// ─────────────────────────────────────────────
// Drop zone
// ─────────────────────────────────────────────
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');

dropZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', e => addFiles([...e.target.files]));

dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  addFiles([...e.dataTransfer.files]);
});

function naturalKey(name) {
  return name.replace(/(\d+)/g, n => n.padStart(10, '0')).toLowerCase();
}

function addFiles(files) {
  const videoExts = ['.mp4','.mov','.webm','.mkv'];
  const valid = files.filter(f => videoExts.some(e => f.name.toLowerCase().endsWith(e)));
  valid.forEach(f => {
    if (!clips.find(c => c.name === f.name))
      clips.push({ file: f, name: f.name, size: f.size, uploadedName: null });
  });
  if (autoSort) clips.sort((a, b) => naturalKey(a.name).localeCompare(naturalKey(b.name)));
  renderClipList();
}

function toggleSort() {
  autoSort = !autoSort;
  const btn = document.getElementById('sort-btn');
  if (autoSort) {
    clips.sort((a, b) => naturalKey(a.name).localeCompare(naturalKey(b.name)));
    btn.style.color = 'var(--accent)';
    btn.style.borderColor = 'var(--accent)';
    btn.textContent = '⇅ Auto-sort ON';
  } else {
    btn.style.color = 'var(--muted)';
    btn.style.borderColor = 'var(--border)';
    btn.textContent = '⇅ Auto-sort';
  }
  renderClipList();
}

function renderClipList() {
  const list = document.getElementById('clip-list');
  const hdr = document.getElementById('clip-list-header');
  if (clips.length === 0) {
    list.innerHTML = '';
    hdr.style.display = 'none';
    return;
  }
  hdr.style.display = 'flex';
  document.getElementById('clip-count').textContent = `${clips.length} clip${clips.length > 1 ? 's' : ''}`;

  list.innerHTML = clips.map((c, i) => `
    <div class="clip-item" draggable="true" data-idx="${i}"
         ondragstart="dragStart(event,${i})" ondragover="dragOver(event,${i})"
         ondrop="dragDrop(event,${i})" ondragleave="dragLeave(event)">
      <div class="clip-order">${i+1}</div>
      <div class="clip-icon">🎬</div>
      <div class="clip-name" title="${c.name}">${c.name}</div>
      <div class="clip-size">${fmtSize(c.size)}</div>
      <button class="clip-remove" onclick="removeClip(${i})">✕</button>
    </div>
  `).join('');
}

function removeClip(idx) {
  clips.splice(idx, 1);
  renderClipList();
}

function fmtSize(bytes) {
  if (bytes > 1024*1024) return (bytes/1024/1024).toFixed(1)+'MB';
  return (bytes/1024).toFixed(0)+'KB';
}

// ─────────────────────────────────────────────
// Drag-to-reorder
// ─────────────────────────────────────────────
let dragIdx = null;
function dragStart(e, idx) { dragIdx = idx; e.currentTarget.classList.add('dragging'); }
function dragOver(e, idx) { e.preventDefault(); e.currentTarget.classList.add('drag-target'); }
function dragLeave(e) { e.currentTarget.classList.remove('drag-target'); }
function dragDrop(e, idx) {
  e.currentTarget.classList.remove('drag-target');
  if (dragIdx === null || dragIdx === idx) return;
  const moved = clips.splice(dragIdx, 1)[0];
  clips.splice(idx, 0, moved);
  dragIdx = null;
  renderClipList();
}

// ─────────────────────────────────────────────
// Music
// ─────────────────────────────────────────────
const musicDrop = document.getElementById('music-drop');
const musicInput = document.getElementById('music-input');

musicDrop.addEventListener('click', () => musicInput.click());
musicInput.addEventListener('change', e => setMusic(e.target.files[0]));
musicDrop.addEventListener('dragover', e => { e.preventDefault(); musicDrop.classList.add('drag-over'); });
musicDrop.addEventListener('dragleave', () => musicDrop.classList.remove('drag-over'));
musicDrop.addEventListener('drop', e => {
  e.preventDefault(); musicDrop.classList.remove('drag-over');
  setMusic(e.dataTransfer.files[0]);
});

function setMusic(file) {
  if (!file) return;
  musicFile = file;
  musicDrop.textContent = `🎵 ${file.name}`;
  musicDrop.classList.add('has-file');
}

// ─────────────────────────────────────────────
// Render
// ─────────────────────────────────────────────
async function startRender() {
  if (clips.length === 0) {
    alert('Add at least one video clip first.');
    return;
  }

  _jobFinished = false;
  if (_pollInterval) { clearInterval(_pollInterval); _pollInterval = null; }

  const btn = document.getElementById('render-btn');
  btn.disabled = true;
  btn.textContent = '⏳ Uploading clips…';

  // Show progress panel
  document.getElementById('progress-panel').classList.add('visible');
  document.getElementById('output-panel').classList.remove('visible');
  document.getElementById('help-card').style.display = 'none';
  clearLog();
  setProgress(0, 'Uploading…');
  setStatus('running');

  try {
    // 1. Upload clips
    const orderedNames = await uploadClips();

    // 2. Upload music (if any)
    let musicName = null;
    if (musicFile) {
      btn.textContent = '⏳ Uploading music…';
      musicName = await uploadMusic();
    }

    // 3. Build config overrides
    const config = buildConfigOverrides();

    // 4. Submit job
    btn.textContent = '⏳ Starting render…';
    const label = clips.map(c => c.name.replace(/\.\w+$/, '')).join(' · ').slice(0, 60);
    const resp = await fetch('/api/render', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        clips: orderedNames,
        session_id: sessionId,
        music: musicName,
        config,
        label,
      })
    });
    const job = await resp.json();
    if (!resp.ok) throw new Error(job.detail || 'Failed to start job');

    currentJobId = job.job_id;
    btn.textContent = '🎬 Rendering…';

    // 5. Connect WebSocket for live updates
    connectWS(job.job_id);

  } catch (err) {
    setStatus('error');
    appendLog(`ERROR: ${err.message}`, 'error');
    btn.disabled = false;
    btn.textContent = '▶ Render Reel';
  }
}

async function uploadClips() {
  // New session for each render so clips never bleed across jobs
  sessionId = crypto.randomUUID();
  const names = [];
  for (let i = 0; i < clips.length; i++) {
    const c = clips[i];
    setProgress(Math.round((i / clips.length) * 10), `Uploading ${c.name}…`);

    const fd = new FormData();
    fd.append('file', c.file, c.name);        // keep original name
    fd.append('session_id', sessionId);        // scoped to this render
    const r = await fetch('/api/upload/clip', { method: 'POST', body: fd });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail);
    names.push(d.filename);                    // names in user-specified order
  }
  return names;
}

async function uploadMusic() {
  const fd = new FormData();
  fd.append('file', musicFile, musicFile.name);
  fd.append('session_id', sessionId);
  const r = await fetch('/api/upload/music', { method: 'POST', body: fd });
  const d = await r.json();
  if (!r.ok) throw new Error(d.detail);
  return d.filename;
}

function buildConfigOverrides() {
  const handle = document.getElementById('s-handle').value;
  const model = document.getElementById('s-model').value;
  const transition = document.getElementById('s-transition').value;
  const captions = document.getElementById('s-captions').checked;
  const zoom = document.getElementById('s-zoom').checked;
  const watermark = document.getElementById('s-watermark').checked;
  const endcard = document.getElementById('s-endcard').checked;
  const highlightColor = document.getElementById('s-highlight-color').value;
  const keywords = document.getElementById('s-keywords').value
    .split(',').map(k => k.trim()).filter(Boolean);

  return {
    brand: { handle, end_card: { enabled: endcard } },
    captions: {
      enabled: captions,
      model,
      highlight: { enabled: true, color: highlightColor, keywords }
    },
    zoom: { enabled: zoom },
    transitions: { type: transition },
    watermark: { enabled: watermark }
  };
}

// ─────────────────────────────────────────────
// WebSocket
// ─────────────────────────────────────────────
function connectWS(jobId) {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/api/ws/${jobId}`);

  ws.onmessage = (e) => {
    const data = JSON.parse(e.data);
    handleUpdate(data);
  };

  ws.onerror = () => {
    appendLog('WebSocket error — polling for status…', 'error');
    startPolling(jobId);
  };

  ws.onclose = () => {
    // Always fall back to polling on close — the job may still be running
    if (!_jobFinished) startPolling(jobId);
  };
}

let _pollInterval = null;
let _jobFinished = false;

function startPolling(jobId) {
  if (_pollInterval) return; // already polling
  _pollInterval = setInterval(async () => {
    try {
      const r = await fetch(`/api/jobs/${jobId}`);
      const data = await r.json();
      handleUpdate(data);
      if (data.status === 'done' || data.status === 'error') {
        clearInterval(_pollInterval);
        _pollInterval = null;
      }
    } catch(e) { /* network error — keep polling */ }
  }, 1500);
}

function handleUpdate(data) {
  if (!data || !data.status) return;

  setProgress(data.progress || 0, data.stage || '');
  setStatus(data.status);

  // Append only new log lines (server sends full log, we track count)
  if (data.log && data.log.length) {
    const box = document.getElementById('log-box');
    const lastCount = parseInt(box.dataset.count || '0');
    data.log.slice(lastCount).forEach(line => appendLog(line));
    box.dataset.count = data.log.length;
  }

  if (data.status === 'done') {
    _jobFinished = true;
    if (data.output) {
      onRenderDone(data.output);
    } else {
      appendLog('Render complete but output path missing.', 'error');
    }
    const btn = document.getElementById('render-btn');
    btn.disabled = false;
    btn.textContent = '▶ Render Another';
  }

  if (data.status === 'error') {
    _jobFinished = true;
    if (data.error) appendLog(data.error, 'error');
    const btn = document.getElementById('render-btn');
    btn.disabled = false;
    btn.textContent = '▶ Render Reel';
  }
}

// ─────────────────────────────────────────────
// UI helpers
// ─────────────────────────────────────────────
function setProgress(pct, stage) {
  document.getElementById('progress-fill').style.width = pct + '%';
  document.getElementById('progress-pct').textContent = pct + '%';
  if (stage) document.getElementById('stage-label').textContent = stage;
}

function setStatus(s) {
  const pill = document.getElementById('status-pill');
  pill.className = 'status-pill ' + s;
  const icons = { running: '●', done: '✓', error: '✗', pending: '⏳' };
  const labels = { running: 'Running', done: 'Done', error: 'Error', pending: 'Pending' };
  const icon = icons[s] || '●';
  const label = labels[s] || s;
  const pulse = s === 'running' ? 'class="pulse"' : '';
  pill.innerHTML = `<span ${pulse}>${icon}</span> <span>${label}</span>`;
}

function appendLog(line, type) {
  const box = document.getElementById('log-box');
  const div = document.createElement('div');
  div.className = 'log-line' + (type ? ' log-'+type : '');
  div.textContent = line;
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
}

function clearLog() {
  const box = document.getElementById('log-box');
  box.innerHTML = '';
  box.dataset.count = '0';
}

function onRenderDone(outputPath) {
  const panel = document.getElementById('output-panel');
  const video = document.getElementById('output-video');
  const dlBtn = document.getElementById('download-btn');

  // Extract just filename for URL
  const fname = outputPath.split('/').pop();
  const videoUrl = `/api/output/${fname}`;

  video.src = videoUrl;
  dlBtn.href = videoUrl;
  dlBtn.download = fname;
  panel.classList.add('visible');

  const btn = document.getElementById('render-btn');
  btn.disabled = false;
  btn.textContent = '▶ Render Another';
  setProgress(100, 'Complete!');
}

function resetUI() {
  clips = [];
  musicFile = null;
  currentJobId = null;
  if (ws) { ws.close(); ws = null; }
  renderClipList();
  document.getElementById('output-panel').classList.remove('visible');
  document.getElementById('progress-panel').classList.remove('visible');
  document.getElementById('help-card').style.display = '';
  document.getElementById('render-btn').disabled = false;
  document.getElementById('render-btn').textContent = '▶ Render Reel';
  document.getElementById('music-drop').textContent = '🎵 Drop MP3/WAV here or click to browse';
  document.getElementById('music-drop').classList.remove('has-file');
}
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse(UI_HTML)


# ---------------------------------------------------------------------------
# Upload endpoints
# ---------------------------------------------------------------------------


@app.post("/api/upload/clip")
async def upload_clip(
    file: UploadFile = File(...),
    session_id: str = Form(""),
) -> dict[str, str]:
    """Upload a clip into a session-scoped directory to preserve order."""
    sid = session_id or "default"
    session_dir = UPLOADS_DIR / sid / "clips"
    session_dir.mkdir(parents=True, exist_ok=True)

    safe_name = Path(file.filename or "clip.mp4").name
    dest = session_dir / safe_name

    async with aiofiles.open(dest, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            await f.write(chunk)

    return {"filename": safe_name, "session_id": sid}


@app.post("/api/upload/music")
async def upload_music(
    file: UploadFile = File(...),
    session_id: str = Form(""),
) -> dict[str, str]:
    sid = session_id or "default"
    music_dir = UPLOADS_DIR / sid / "music"
    music_dir.mkdir(parents=True, exist_ok=True)

    safe_name = Path(file.filename or "music.mp3").name
    dest = music_dir / safe_name

    async with aiofiles.open(dest, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            await f.write(chunk)

    return {"filename": safe_name}


# ---------------------------------------------------------------------------
# Render job endpoint
# ---------------------------------------------------------------------------


from pydantic import BaseModel


class RenderPayload(BaseModel):
    clips: list[str]        # ordered list of filenames as uploaded
    session_id: str = "default"
    music: str | None = None
    config: dict[str, Any] = {}
    label: str = ""         # optional human label shown in library


@app.post("/api/render")
async def start_render(payload: RenderPayload) -> dict[str, str]:
    sid = payload.session_id or "default"
    clips_dir = UPLOADS_DIR / sid / "clips"

    if not clips_dir.exists():
        raise HTTPException(status_code=400, detail="No clips uploaded for this session")

    # Build ordered clip paths exactly as the client specified
    ordered_clips = []
    for name in payload.clips:
        p = clips_dir / name
        if p.exists():
            ordered_clips.append(p)

    if not ordered_clips:
        raise HTTPException(status_code=400, detail="None of the specified clips were found")

    jid = _new_job()

    # Store label + clip names for library display
    _update_job(
        jid,
        label=payload.label or f"Reel {jid[:6]}",
        clip_names=[p.name for p in ordered_clips],
    )

    output_path = OUTPUTS_DIR / f"reel_{jid[:8]}.mp4"

    music_path: Path | None = None
    if payload.music:
        music_path = UPLOADS_DIR / sid / "music" / payload.music

    # Pass ordered clips via a symlinked staging dir so pipeline sorts correctly
    staging_dir = UPLOADS_DIR / sid / f"stage_{jid[:8]}"
    staging_dir.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(ordered_clips):
        dst = staging_dir / f"{i+1:03d}_{src.name}"
        if not dst.exists():
            import shutil as _sh
            _sh.copy2(src, dst)

    thread = threading.Thread(
        target=_run_pipeline_thread,
        args=(jid, staging_dir, output_path, payload.config, music_path, BASE_DIR),
        daemon=True,
    )
    thread.start()

    return {"job_id": jid}


# ---------------------------------------------------------------------------
# Job status + WebSocket
# ---------------------------------------------------------------------------


@app.get("/api/jobs")
async def list_jobs() -> list[dict[str, Any]]:
    with _job_lock:
        return list(_jobs.values())


@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str) -> dict[str, Any]:
    try:
        return _get_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found")


@app.websocket("/api/ws/{job_id}")
async def websocket_progress(websocket: WebSocket, job_id: str) -> None:
    await websocket.accept()
    try:
        last_log_count = 0
        while True:
            try:
                job = _get_job(job_id)
            except KeyError:
                await websocket.send_text(json.dumps({"error": "Job not found"}))
                break

            # Only send new log lines to avoid resending everything
            new_logs = job["log"][last_log_count:]
            last_log_count = len(job["log"])
            update = {**job, "log": new_logs}
            await websocket.send_text(json.dumps(update))

            if job["status"] in ("done", "error"):
                # Send full log one last time then close
                await asyncio.sleep(0.2)
                final = _get_job(job_id)
                await websocket.send_text(json.dumps(final))
                break

            await asyncio.sleep(0.8)

    except WebSocketDisconnect:
        pass


# ---------------------------------------------------------------------------
# Serve output video
# ---------------------------------------------------------------------------


@app.get("/api/output/{filename}")
async def serve_output(filename: str) -> FileResponse:
    path = OUTPUTS_DIR / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Output not found")
    return FileResponse(path, media_type="video/mp4", filename=filename)


# ---------------------------------------------------------------------------
# Library — list all completed reels on disk
# ---------------------------------------------------------------------------


@app.get("/api/library")
async def list_library() -> list[dict[str, Any]]:
    """Return metadata for every completed reel in the output directory."""
    import time as _time

    reels = []
    for p in sorted(OUTPUTS_DIR.glob("reel_*.mp4"), key=lambda f: f.stat().st_mtime, reverse=True):
        stat = p.stat()
        # Try to match to an in-memory job for extra metadata
        job_meta: dict[str, Any] = {}
        short = p.stem.replace("reel_", "")
        with _job_lock:
            for jid, job in _jobs.items():
                if jid[:8] == short:
                    job_meta = job
                    break

        reels.append({
            "filename": p.name,
            "size_mb": round(stat.st_size / 1024 / 1024, 1),
            "created_at": stat.st_mtime,
            "label": job_meta.get("label", p.stem),
            "clip_names": job_meta.get("clip_names", []),
            "duration_s": job_meta.get("total_duration"),
        })
    return reels


@app.get("/library", response_class=HTMLResponse)
async def library_page() -> HTMLResponse:
    return HTMLResponse(LIBRARY_HTML)


# ---------------------------------------------------------------------------
# Library HTML
# ---------------------------------------------------------------------------

LIBRARY_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>ReelForge — Library</title>
<style>
  :root {
    --bg: #0f0f13; --surface: #1a1a23; --surface2: #23232f;
    --accent: #FFD400; --text: #f0f0f5; --muted: #7a7a90;
    --border: #2e2e3e; --success: #4ade80; --error: #f87171;
    --radius: 12px;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: var(--bg); color: var(--text); min-height: 100vh; }

  header {
    display: flex; align-items: center; gap: 12px;
    padding: 18px 32px; border-bottom: 1px solid var(--border);
    background: var(--surface);
  }
  header .logo { font-size: 22px; font-weight: 800; }
  header .logo span { color: var(--accent); }
  header nav { margin-left: auto; display: flex; gap: 8px; }
  header nav a {
    font-size: 13px; font-weight: 600; color: var(--muted);
    text-decoration: none; padding: 6px 14px; border-radius: 8px;
    border: 1px solid var(--border); transition: all 0.15s;
  }
  header nav a:hover, header nav a.active { color: var(--accent); border-color: var(--accent); }

  .page { max-width: 1100px; margin: 0 auto; padding: 32px 24px; }
  .page-title { font-size: 24px; font-weight: 800; margin-bottom: 6px; }
  .page-sub { font-size: 14px; color: var(--muted); margin-bottom: 28px; }

  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
    gap: 20px;
  }

  .reel-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); overflow: hidden;
    transition: border-color 0.15s, transform 0.15s;
  }
  .reel-card:hover { border-color: var(--accent); transform: translateY(-2px); }

  .reel-thumb {
    width: 100%; aspect-ratio: 9/16; background: #000;
    display: flex; align-items: center; justify-content: center;
    position: relative; overflow: hidden; cursor: pointer;
  }
  .reel-thumb video { width: 100%; height: 100%; object-fit: cover; }
  .reel-thumb .play-btn {
    position: absolute; inset: 0; display: flex;
    align-items: center; justify-content: center;
    background: rgba(0,0,0,0.35); transition: opacity 0.15s;
  }
  .reel-thumb:hover .play-btn { opacity: 0; }
  .play-icon {
    width: 48px; height: 48px; border-radius: 50%;
    background: rgba(255,212,0,0.9);
    display: flex; align-items: center; justify-content: center;
    font-size: 20px; color: #000;
  }

  .reel-info { padding: 14px; }
  .reel-label { font-size: 14px; font-weight: 700; margin-bottom: 4px;
                white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .reel-meta { font-size: 12px; color: var(--muted); margin-bottom: 10px; }
  .reel-clips { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 12px; }
  .clip-pill {
    font-size: 10px; background: var(--surface2); border: 1px solid var(--border);
    border-radius: 4px; padding: 2px 6px; color: var(--muted);
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 120px;
  }

  .reel-actions { display: flex; gap: 8px; }
  .btn { flex: 1; padding: 8px; border-radius: 8px; border: 1px solid var(--border);
         background: var(--surface2); color: var(--text); font-size: 12px;
         font-weight: 600; cursor: pointer; text-align: center; text-decoration: none;
         transition: all 0.15s; }
  .btn:hover { border-color: var(--accent); color: var(--accent); }
  .btn-primary { background: var(--accent); color: #000; border-color: var(--accent); }
  .btn-primary:hover { background: #ffe033; color: #000; }

  .empty {
    grid-column: 1/-1; text-align: center; padding: 80px 0;
    color: var(--muted); font-size: 15px;
  }
  .empty a { color: var(--accent); text-decoration: none; font-weight: 600; }

  .badge {
    display: inline-block; font-size: 10px; font-weight: 700;
    padding: 2px 7px; border-radius: 999px; margin-left: 6px;
    background: var(--accent); color: #000; vertical-align: middle;
  }

  /* Modal */
  .modal-bg {
    display: none; position: fixed; inset: 0;
    background: rgba(0,0,0,0.85); z-index: 100;
    align-items: center; justify-content: center;
  }
  .modal-bg.open { display: flex; }
  .modal {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 20px;
    max-width: 420px; width: 90%;
  }
  .modal video { width: 100%; border-radius: 8px; max-height: 70vh; }
  .modal-title { font-size: 15px; font-weight: 700; margin-bottom: 10px; }
  .modal-close {
    float: right; background: none; border: none; color: var(--muted);
    font-size: 20px; cursor: pointer; line-height: 1;
  }
</style>
</head>
<body>

<header>
  <div class="logo">Reel<span>Forge</span></div>
  <nav>
    <a href="/">Render</a>
    <a href="/library" class="active">Library</a>
  </nav>
</header>

<div class="page">
  <div class="page-title">Library <span class="badge" id="count">0</span></div>
  <div class="page-sub">All your rendered reels — click to preview, download, or share.</div>

  <div class="grid" id="grid">
    <div class="empty">Loading…</div>
  </div>
</div>

<!-- Modal player -->
<div class="modal-bg" id="modal" onclick="closeModal(event)">
  <div class="modal">
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;">
      <div class="modal-title" id="modal-title">Preview</div>
      <button class="modal-close" onclick="closeModal()">✕</button>
    </div>
    <video id="modal-video" controls playsinline></video>
    <div style="display:flex;gap:8px;margin-top:12px;">
      <a id="modal-dl" class="btn btn-primary" download>⬇ Download</a>
      <button class="btn" onclick="closeModal()">Close</button>
    </div>
  </div>
</div>

<script>
async function loadLibrary() {
  const r = await fetch('/api/library');
  const reels = await r.json();
  const grid = document.getElementById('grid');
  document.getElementById('count').textContent = reels.length;

  if (reels.length === 0) {
    grid.innerHTML = '<div class="empty">No reels yet. <a href="/">Render your first one →</a></div>';
    return;
  }

  grid.innerHTML = reels.map(reel => {
    const date = new Date(reel.created_at * 1000).toLocaleDateString(undefined, {
      month: 'short', day: 'numeric', year: 'numeric'
    });
    const clipPills = (reel.clip_names || []).map(n =>
      `<span class="clip-pill" title="${n}">${n.replace(/^\d+_/,'').replace(/\.mp4$/i,'')}</span>`
    ).join('');

    return `
    <div class="reel-card">
      <div class="reel-thumb" onclick="openModal('${reel.filename}','${reel.label}')">
        <video src="/api/output/${reel.filename}" muted preload="metadata"
               onmouseenter="this.play()" onmouseleave="this.pause();this.currentTime=0"></video>
        <div class="play-btn"><div class="play-icon">▶</div></div>
      </div>
      <div class="reel-info">
        <div class="reel-label">${reel.label}</div>
        <div class="reel-meta">${reel.size_mb} MB · ${date}</div>
        ${clipPills ? `<div class="reel-clips">${clipPills}</div>` : ''}
        <div class="reel-actions">
          <a class="btn btn-primary" href="/api/output/${reel.filename}" download="${reel.filename}">⬇ Download</a>
          <button class="btn" onclick="openModal('${reel.filename}','${reel.label}')">▶ Preview</button>
        </div>
      </div>
    </div>`;
  }).join('');
}

function openModal(filename, label) {
  const v = document.getElementById('modal-video');
  const dl = document.getElementById('modal-dl');
  document.getElementById('modal-title').textContent = label;
  v.src = `/api/output/${filename}`;
  dl.href = `/api/output/${filename}`;
  dl.download = filename;
  document.getElementById('modal').classList.add('open');
  v.play();
}

function closeModal(e) {
  if (e && e.target !== document.getElementById('modal')) return;
  const v = document.getElementById('modal-video');
  v.pause();
  v.src = '';
  document.getElementById('modal').classList.remove('open');
}

loadLibrary();
</script>
</body>
</html>
"""
