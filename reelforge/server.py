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
import hashlib
import hmac
import os
import secrets

import httpx
from authlib.integrations.httpx_client import AsyncOAuth2Client
from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

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
USERS_DIR = BASE_DIR / "output" / "users"  # per-user data root

# Google OAuth config — read from env or fall back to stored config
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
# Facebook/Instagram OAuth
FACEBOOK_APP_ID = os.environ.get("FACEBOOK_APP_ID", "")
FACEBOOK_APP_SECRET = os.environ.get("FACEBOOK_APP_SECRET", "")
# PUBLIC_URL must be set when running behind a reverse proxy / tunnel
# e.g. PUBLIC_URL=https://reeleditor.ariham.com
PUBLIC_URL = os.environ.get("PUBLIC_URL", "").rstrip("/")
_GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
_GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
_GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"
_FB_AUTH_URL = "https://www.facebook.com/v19.0/dialog/oauth"
_FB_TOKEN_URL = "https://graph.facebook.com/v19.0/oauth/access_token"


def _app_base(request: Request) -> str:
    """Return the public base URL, falling back to request.base_url."""
    return PUBLIC_URL if PUBLIC_URL else str(request.base_url).rstrip("/")

# Session secret — persist across restarts
_SECRET_FILE = BASE_DIR / "output" / "_secret.key"


def _get_session_secret() -> str:
    if _SECRET_FILE.exists():
        return _SECRET_FILE.read_text().strip()
    secret = secrets.token_hex(32)
    _SECRET_FILE.parent.mkdir(parents=True, exist_ok=True)
    _SECRET_FILE.write_text(secret)
    return secret


# ---------------------------------------------------------------------------
# Per-user path helpers
# ---------------------------------------------------------------------------


def user_dir(uid: str) -> Path:
    return USERS_DIR / uid


def user_outputs(uid: str) -> Path:
    return user_dir(uid) / "reels"


def user_uploads(uid: str) -> Path:
    return user_dir(uid) / "uploads"


def user_settings_file(uid: str) -> Path:
    return user_dir(uid) / "settings.json"


def user_yt_token(uid: str) -> Path:
    return user_dir(uid) / "yt_token.json"


def user_yt_creds(uid: str) -> Path:
    return user_dir(uid) / "yt_credentials.json"


def _load_settings(uid: str) -> dict[str, Any]:
    f = user_settings_file(uid)
    return json.loads(f.read_text()) if f.exists() else {}


def _save_settings(uid: str, data: dict[str, Any]) -> None:
    user_settings_file(uid).write_text(json.dumps(data, indent=2))


def _ensure_user_dirs(uid: str) -> None:
    user_outputs(uid).mkdir(parents=True, exist_ok=True)
    user_uploads(uid).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------


def _get_user(request: Request) -> dict[str, Any] | None:
    return request.session.get("user")


def _require_user(request: Request) -> dict[str, Any]:
    user = _get_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


def _require_user_page(request: Request) -> dict[str, Any]:
    """For HTML pages — raises redirect instead of 401."""
    user = _get_user(request)
    if not user:
        raise _LoginRedirect(str(request.url))
    return user


class _LoginRedirect(Exception):
    def __init__(self, next_url: str = "/") -> None:
        self.next_url = next_url


def _google_configured() -> bool:
    return bool(GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET)


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    USERS_DIR.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="Reel Editor", version="0.1.0", lifespan=lifespan)
app.add_middleware(SessionMiddleware, secret_key=_get_session_secret(), max_age=30 * 24 * 3600)


# ---------------------------------------------------------------------------
# Exception handler for login redirects
# ---------------------------------------------------------------------------


from starlette.exceptions import HTTPException as StarletteHTTPException  # noqa: E402


@app.exception_handler(_LoginRedirect)
async def _login_redirect_handler(request: Request, exc: _LoginRedirect) -> RedirectResponse:
    return RedirectResponse(f"/auth/login?next={exc.next_url}")


# ---------------------------------------------------------------------------
# Static HTML UI (single-file, no build step)
# ---------------------------------------------------------------------------

UI_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Reel Editor</title>
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
    <a href="/render" style="color:var(--accent);border-color:var(--accent);">Render</a>
    <a href="/library">Library</a>
    <a href="/settings">Settings</a>
    <a href="/auth/logout" id="nav-user" title="Sign out" style="display:flex;align-items:center;gap:6px;">
      <img id="nav-avatar" src="" width="22" height="22" style="border-radius:50%;display:none"/>
      <span id="nav-name"></span> · Sign out
    </a>
  </nav>
</header>
<script>
fetch('/api/me').then(r=>r.json()).then(u=>{
  document.getElementById('nav-name').textContent = u.name.split(' ')[0];
  const img = document.getElementById('nav-avatar');
  if(u.picture){img.src=u.picture;img.style.display='';}
}).catch(()=>{});
</script>

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

function uploadWithProgress(url, formData, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', url);
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && onProgress) {
        onProgress(e.loaded / e.total);
      }
    };
    xhr.onload = () => {
      try {
        const d = JSON.parse(xhr.responseText);
        if (xhr.status >= 200 && xhr.status < 300) resolve(d);
        else reject(new Error(d.detail || `Upload failed (${xhr.status})`));
      } catch { reject(new Error(`Upload failed (${xhr.status})`)); }
    };
    xhr.onerror = () => reject(new Error('Network error during upload'));
    xhr.ontimeout = () => reject(new Error('Upload timed out'));
    xhr.timeout = 600000; // 10 min
    xhr.send(formData);
  });
}

async function uploadClips() {
  // New session for each render so clips never bleed across jobs
  sessionId = crypto.randomUUID();
  const names = [];
  for (let i = 0; i < clips.length; i++) {
    const c = clips[i];

    const fd = new FormData();
    fd.append('file', c.file, c.name);
    fd.append('session_id', sessionId);

    const d = await uploadWithProgress('/api/upload/clip', fd, (frac) => {
      // each clip gets an equal slice of the 0-10% band
      const base = (i / clips.length) * 10;
      const slice = (1 / clips.length) * 10;
      const pct = Math.round(base + slice * frac);
      setProgress(pct, `Uploading ${c.name}… ${Math.round(frac * 100)}%`);
    });
    names.push(d.filename);
  }
  return names;
}

async function uploadMusic() {
  const fd = new FormData();
  fd.append('file', musicFile, musicFile.name);
  fd.append('session_id', sessionId);
  const d = await uploadWithProgress('/api/upload/music', fd, (frac) => {
    setProgress(10, `Uploading music… ${Math.round(frac * 100)}%`);
  });
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


@app.get("/render", response_class=HTMLResponse)
async def render_page(request: Request) -> HTMLResponse:
    _require_user_page(request)
    return HTMLResponse(UI_HTML)


# ---------------------------------------------------------------------------
# Google OAuth routes
# ---------------------------------------------------------------------------

LOGIN_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Reel Editor — Sign in</title>
<style>
  :root { --bg:#0f0f13; --surface:#1a1a23; --accent:#FFD400; --text:#f0f0f5; --muted:#7a7a90; --border:#2e2e3e; --radius:12px; }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
         background:var(--bg); color:var(--text); min-height:100vh;
         display:flex; align-items:center; justify-content:center; }
  .card { background:var(--surface); border:1px solid var(--border); border-radius:var(--radius);
          padding:48px 40px; max-width:380px; width:90%; text-align:center; }
  .logo { font-size:32px; font-weight:900; margin-bottom:8px; }
  .logo span { color:var(--accent); }
  .tagline { font-size:14px; color:var(--muted); margin-bottom:36px; }
  .signin-btn {
    display:inline-flex; align-items:center; gap:12px;
    padding:14px 24px; border-radius:10px;
    background:#fff; color:#1f1f1f; font-size:15px; font-weight:600;
    border:none; cursor:pointer; text-decoration:none; transition:opacity 0.15s;
    width:100%; justify-content:center;
  }
  .signin-btn:hover { opacity:0.9; }
  .signin-btn svg { flex-shrink:0; }
  .not-configured { font-size:13px; color:var(--muted); margin-top:24px;
                    padding:14px; border:1px solid var(--border); border-radius:8px; }
  .not-configured code { color:var(--accent); font-size:12px; }
</style>
</head>
<body>
<div class="card">
  <div class="logo">Reel<span>Forge</span></div>
  <div class="tagline">AI clips → polished vertical reels</div>
  {body}
</div>
</body>
</html>"""

_SIGNIN_BTN = """<a class="signin-btn" href="/auth/google">
  <svg width="20" height="20" viewBox="0 0 48 48">
    <path fill="#EA4335" d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z"/>
    <path fill="#4285F4" d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z"/>
    <path fill="#FBBC05" d="M10.53 28.59c-.48-1.45-.76-2.99-.76-4.59s.27-3.14.76-4.59l-7.98-6.19C.92 16.46 0 20.12 0 24c0 3.88.92 7.54 2.56 10.78l7.97-6.19z"/>
    <path fill="#34A853" d="M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.18 1.48-4.97 2.31-8.16 2.31-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z"/>
  </svg>
  Sign in with Google
</a>"""

_NOT_CONFIGURED = """<div class="not-configured">
  Google OAuth not configured.<br/>
  Set <code>GOOGLE_CLIENT_ID</code> and <code>GOOGLE_CLIENT_SECRET</code> env vars<br/>
  then restart the server.
</div>"""


@app.get("/auth/login", response_class=HTMLResponse)
async def login_page(request: Request) -> HTMLResponse:
    if _get_user(request):
        return RedirectResponse("/")  # type: ignore[return-value]
    body = _SIGNIN_BTN if _google_configured() else _NOT_CONFIGURED
    return HTMLResponse(LOGIN_HTML.replace("{body}", body))


@app.get("/auth/google")
async def google_login(request: Request) -> RedirectResponse:
    if not _google_configured():
        raise HTTPException(status_code=503, detail="Google OAuth not configured")
    next_url = request.query_params.get("next", "/render")
    state = secrets.token_urlsafe(16) + "|" + next_url
    request.session["oauth_state"] = state
    redirect_uri = _app_base(request) + "/auth/google/callback"
    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "access_type": "offline",
        "prompt": "select_account",
    }
    from urllib.parse import urlencode
    return RedirectResponse(f"{_GOOGLE_AUTH_URL}?{urlencode(params)}")


@app.get("/auth/google/callback")
async def google_callback(request: Request) -> RedirectResponse:
    state = request.query_params.get("state", "")
    saved_state = request.session.pop("oauth_state", "")
    next_url = "/render"
    if "|" in state:
        _, next_url = state.split("|", 1)
        if next_url in ("/", "/library"):
            next_url = "/render"

    code = request.query_params.get("code")
    if not code:
        raise HTTPException(status_code=400, detail="Missing auth code")

    redirect_uri = _app_base(request) + "/auth/google/callback"

    async with httpx.AsyncClient() as client:
        token_resp = await client.post(_GOOGLE_TOKEN_URL, data={
            "code": code,
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        })
        if token_resp.status_code != 200:
            raise HTTPException(status_code=502, detail="Token exchange failed")
        tokens = token_resp.json()
        access_token = tokens.get("access_token")

        user_resp = await client.get(_GOOGLE_USERINFO_URL,
                                     headers={"Authorization": f"Bearer {access_token}"})
        if user_resp.status_code != 200:
            raise HTTPException(status_code=502, detail="Userinfo fetch failed")
        user_info = user_resp.json()

    request.session["user"] = {
        "sub": user_info["sub"],
        "email": user_info.get("email", ""),
        "name": user_info.get("name", ""),
        "picture": user_info.get("picture", ""),
    }
    _ensure_user_dirs(user_info["sub"])
    return RedirectResponse(next_url if next_url.startswith("/") else "/")


@app.get("/auth/logout")
async def logout(request: Request) -> RedirectResponse:
    request.session.clear()
    return RedirectResponse("/auth/login")


@app.get("/api/me")
async def me(request: Request) -> dict[str, Any]:
    user = _require_user(request)
    return {"email": user["email"], "name": user["name"], "picture": user["picture"]}


# ---------------------------------------------------------------------------
# Upload endpoints
# ---------------------------------------------------------------------------


@app.post("/api/upload/clip")
async def upload_clip(
    request: Request,
    file: UploadFile = File(...),
    session_id: str = Form(""),
) -> dict[str, str]:
    """Upload a clip into a session-scoped directory to preserve order."""
    user = _require_user(request)
    uid = user["sub"]
    sid = session_id or "default"
    session_dir = user_uploads(uid) / sid / "clips"
    session_dir.mkdir(parents=True, exist_ok=True)

    safe_name = Path(file.filename or "clip.mp4").name
    dest = session_dir / safe_name

    async with aiofiles.open(dest, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            await f.write(chunk)

    return {"filename": safe_name, "session_id": sid}


@app.post("/api/upload/music")
async def upload_music(
    request: Request,
    file: UploadFile = File(...),
    session_id: str = Form(""),
) -> dict[str, str]:
    user = _require_user(request)
    uid = user["sub"]
    sid = session_id or "default"
    music_dir = user_uploads(uid) / sid / "music"
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
async def start_render(request: Request, payload: RenderPayload) -> dict[str, str]:
    user = _require_user(request)
    uid = user["sub"]
    _ensure_user_dirs(uid)

    sid = payload.session_id or "default"
    clips_dir = user_uploads(uid) / sid / "clips"

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
    _update_job(
        jid,
        label=payload.label or f"Reel {jid[:6]}",
        clip_names=[p.name for p in ordered_clips],
        user_id=uid,
    )

    output_path = user_outputs(uid) / f"reel_{jid[:8]}.mp4"

    music_path: Path | None = None
    if payload.music:
        music_path = user_uploads(uid) / sid / "music" / payload.music

    staging_dir = user_uploads(uid) / sid / f"stage_{jid[:8]}"
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
async def serve_output(filename: str, request: Request) -> Response:
    user = _require_user(request)
    path = user_outputs(user["sub"]) / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Output not found")

    # Use nginx X-Accel-Redirect for efficient static file serving.
    # FastAPI handles auth; nginx serves the actual bytes (zero-copy, sendfile).
    # Falls back to Python streaming if X-Accel is not available (local dev).
    uid = user["sub"]
    internal_path = f"/_internal_output/{uid}/reels/{filename}"

    # Check if we're behind nginx (X-Accel-capable)
    if request.headers.get("x-forwarded-for") or request.headers.get("x-forwarded-proto"):
        return Response(
            status_code=200,
            headers={
                "X-Accel-Redirect": internal_path,
                "Content-Type": "video/mp4",
                "Accept-Ranges": "bytes",
            },
        )

    # Fallback for local dev (no nginx): serve directly via Python
    file_size = path.stat().st_size
    range_header = request.headers.get("range")

    if range_header:
        try:
            range_val = range_header.strip().replace("bytes=", "")
            start_str, _, end_str = range_val.partition("-")
            start = int(start_str) if start_str else 0
            end = int(end_str) if end_str else file_size - 1
        except ValueError:
            raise HTTPException(status_code=416, detail="Invalid range")

        end = min(end, file_size - 1)
        if start > end or start >= file_size:
            raise HTTPException(status_code=416, detail="Range not satisfiable")

        chunk_size = end - start + 1

        def iterfile():
            with open(path, "rb") as f:
                f.seek(start)
                remaining = chunk_size
                while remaining:
                    data = f.read(min(65536, remaining))
                    if not data:
                        break
                    remaining -= len(data)
                    yield data

        return StreamingResponse(
            iterfile(),
            status_code=206,
            media_type="video/mp4",
            headers={
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(chunk_size),
            },
        )

    return FileResponse(
        path,
        media_type="video/mp4",
        headers={"Accept-Ranges": "bytes", "Content-Length": str(file_size)},
    )


# ---------------------------------------------------------------------------
# Temporary signed public URLs (for Instagram/external access)
# ---------------------------------------------------------------------------

_SHARE_TOKEN_EXPIRY = 600  # 10 minutes


def _sign_share_token(uid: str, filename: str, expires: int) -> str:
    """Create an HMAC-signed token for temporary public video access."""
    msg = f"{uid}:{filename}:{expires}"
    sig = hmac.new(_get_session_secret().encode(), msg.encode(), hashlib.sha256).hexdigest()[:32]
    return f"{expires}.{sig}"


def _verify_share_token(uid: str, filename: str, token: str) -> bool:
    """Verify an HMAC-signed share token and check expiry."""
    try:
        parts = token.split(".", 1)
        if len(parts) != 2:
            return False
        expires = int(parts[0])
        if time.time() > expires:
            return False
        expected = _sign_share_token(uid, filename, expires)
        return hmac.compare_digest(token, expected)
    except (ValueError, TypeError):
        return False


@app.get("/api/output/public/{uid}/{filename}")
async def serve_output_public(uid: str, filename: str, token: str = "") -> Response:
    """Serve a video file without auth, validated by a signed token.
    Used by Instagram/external services that need to download the video."""
    if not token or not _verify_share_token(uid, filename, token):
        raise HTTPException(status_code=403, detail="Invalid or expired share link")

    if not filename.startswith("reel_") or not filename.endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Invalid filename")

    path = user_outputs(uid) / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Output not found")

    return FileResponse(
        path,
        media_type="video/mp4",
        headers={"Accept-Ranges": "bytes", "Content-Length": str(path.stat().st_size)},
    )


@app.delete("/api/output/{filename}")
async def delete_output(filename: str, request: Request) -> dict[str, str]:
    user = _require_user(request)
    if not filename.startswith("reel_") or not filename.endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = user_outputs(user["sub"]) / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    path.unlink()
    return {"deleted": filename}


# ---------------------------------------------------------------------------
# Settings API (OAuth-only — no manual token entry)
# ---------------------------------------------------------------------------


@app.get("/api/settings")
async def get_settings(request: Request) -> dict[str, Any]:
    user = _require_user(request)
    uid = user["sub"]
    s = _load_settings(uid)
    return {
        "ig_connected": bool(s.get("ig_access_token") and s.get("ig_user_id")),
        "ig_username": s.get("ig_username", ""),
        "yt_connected": user_yt_token(uid).exists(),
        "yt_channel": _load_settings(uid).get("yt_channel", ""),
        "ig_available": bool(FACEBOOK_APP_ID and FACEBOOK_APP_SECRET),
        "yt_available": bool(GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET),
    }


# ---------------------------------------------------------------------------
# YouTube OAuth (reuses server's GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET)
# ---------------------------------------------------------------------------

_YT_SCOPES = ["https://www.googleapis.com/auth/youtube.upload",
               "https://www.googleapis.com/auth/youtube.readonly"]


@app.get("/api/auth/youtube")
async def youtube_auth_start(request: Request) -> RedirectResponse:
    user = _require_user(request)
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        raise HTTPException(status_code=503, detail="YouTube OAuth not configured on server")
    from urllib.parse import urlencode
    redirect_uri = _app_base(request) + "/api/auth/youtube/callback"
    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": " ".join(_YT_SCOPES),
        "access_type": "offline",
        "prompt": "consent",
        "state": user["sub"],
    }
    return RedirectResponse(f"{_GOOGLE_AUTH_URL}?{urlencode(params)}")


@app.get("/api/auth/youtube/callback")
async def youtube_auth_callback(request: Request) -> HTMLResponse:
    uid = request.query_params.get("state", "")
    code = request.query_params.get("code", "")
    if not uid or not code:
        raise HTTPException(status_code=400, detail="Missing state or code")
    redirect_uri = _app_base(request) + "/api/auth/youtube/callback"
    async with httpx.AsyncClient() as client:
        r = await client.post(_GOOGLE_TOKEN_URL, data={
            "code": code,
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        })
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Token exchange failed: {r.text}")
        tokens = r.json()
        # Fetch channel name
        ch_r = await client.get(
            "https://www.googleapis.com/youtube/v3/channels",
            params={"part": "snippet", "mine": "true"},
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
        )
        channel_name = ""
        if ch_r.status_code == 200:
            items = ch_r.json().get("items", [])
            if items:
                channel_name = items[0]["snippet"]["title"]

    _ensure_user_dirs(uid)
    user_yt_token(uid).write_text(json.dumps({
        "token": tokens["access_token"],
        "refresh_token": tokens.get("refresh_token", ""),
        "token_uri": _GOOGLE_TOKEN_URL,
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "scopes": _YT_SCOPES,
    }))
    s = _load_settings(uid)
    s["yt_channel"] = channel_name
    _save_settings(uid, s)
    return HTMLResponse("<script>window.close();opener&&opener.location.reload();</script>"
                        f"<p>YouTube connected ({channel_name}). You can close this tab.</p>")


@app.delete("/api/auth/youtube")
async def youtube_revoke(request: Request) -> dict[str, str]:
    user = _require_user(request)
    uid = user["sub"]
    t = user_yt_token(uid)
    if t.exists():
        t.unlink()
    s = _load_settings(uid)
    s.pop("yt_channel", None)
    _save_settings(uid, s)
    return {"status": "revoked"}


# ---------------------------------------------------------------------------
# Instagram OAuth (via Facebook Login)
# ---------------------------------------------------------------------------

_IG_SCOPES = "instagram_business_basic,instagram_business_content_publish,instagram_business_manage_comments,pages_show_list"


@app.get("/api/auth/instagram")
async def instagram_auth_start(request: Request) -> RedirectResponse:
    user = _require_user(request)
    if not FACEBOOK_APP_ID or not FACEBOOK_APP_SECRET:
        raise HTTPException(status_code=503, detail="Instagram OAuth not configured on server")
    from urllib.parse import urlencode
    redirect_uri = _app_base(request) + "/api/auth/instagram/callback"
    params = {
        "client_id": FACEBOOK_APP_ID,
        "redirect_uri": redirect_uri,
        "scope": _IG_SCOPES,
        "response_type": "code",
        "state": user["sub"],
    }
    return RedirectResponse(f"{_FB_AUTH_URL}?{urlencode(params)}")


@app.get("/api/auth/instagram/callback")
async def instagram_auth_callback(request: Request) -> HTMLResponse:
    uid = request.query_params.get("state", "")
    code = request.query_params.get("code", "")
    if not uid or not code:
        raise HTTPException(status_code=400, detail="Missing state or code")
    redirect_uri = _app_base(request) + "/api/auth/instagram/callback"
    base = "https://graph.facebook.com/v21.0"

    async with httpx.AsyncClient(timeout=30) as client:
        # Exchange code for short-lived token
        r = await client.get(f"{base}/oauth/access_token", params={
            "client_id": FACEBOOK_APP_ID,
            "client_secret": FACEBOOK_APP_SECRET,
            "redirect_uri": redirect_uri,
            "code": code,
        })
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail=f"FB token exchange failed: {r.text}")
        short_token = r.json().get("access_token", "")

        # Exchange for long-lived token (60 days)
        ll = await client.get(f"{base}/oauth/access_token", params={
            "grant_type": "fb_exchange_token",
            "client_id": FACEBOOK_APP_ID,
            "client_secret": FACEBOOK_APP_SECRET,
            "fb_exchange_token": short_token,
        })
        long_token = ll.json().get("access_token", short_token) if ll.status_code == 200 else short_token

        # Get connected Instagram Professional account
        pages_r = await client.get(f"{base}/me/accounts", params={
            "fields": "instagram_business_account,name",
            "access_token": long_token,
        })
        ig_user_id = ""
        ig_username = ""
        if pages_r.status_code == 200:
            for page in pages_r.json().get("data", []):
                iba = page.get("instagram_business_account", {})
                if iba.get("id"):
                    ig_user_id = iba["id"]
                    # Get username
                    ig_r = await client.get(f"{base}/{ig_user_id}", params={
                        "fields": "username",
                        "access_token": long_token,
                    })
                    if ig_r.status_code == 200:
                        ig_username = ig_r.json().get("username", ig_user_id)
                    break

        if not ig_user_id:
            return HTMLResponse(
                "<script>window.close();</script>"
                "<p style='color:red'>No Instagram Professional account found linked to your Facebook pages. "
                "Make sure your Instagram account is set to Creator or Business and linked to a Facebook Page.</p>"
            )

    _ensure_user_dirs(uid)
    s = _load_settings(uid)
    s["ig_access_token"] = long_token
    s["ig_user_id"] = ig_user_id
    s["ig_username"] = ig_username
    _save_settings(uid, s)
    return HTMLResponse("<script>window.close();opener&&opener.location.reload();</script>"
                        f"<p>Instagram connected (@{ig_username}). You can close this tab.</p>")


@app.post("/api/auth/instagram/manual")
async def instagram_manual_token(request: Request) -> dict[str, Any]:
    """Accept a manually-generated Instagram API token from developers.facebook.com."""
    user = _require_user(request)
    uid = user["sub"]
    body = await request.json()
    access_token = body.get("access_token", "").strip()
    ig_user_id = body.get("ig_user_id", "").strip()

    if not access_token:
        raise HTTPException(status_code=400, detail="access_token is required")

    base = "https://graph.facebook.com/v21.0"
    ig_username = ""

    async with httpx.AsyncClient(timeout=30) as client:
        if not ig_user_id:
            # Try to discover the IG user ID from the token
            # First try: direct /me call (works for IG Business tokens)
            me_r = await client.get(f"{base}/me", params={
                "fields": "id,username,name,account_type",
                "access_token": access_token,
            })
            if me_r.status_code == 200:
                me_data = me_r.json()
                # If this is an Instagram-scoped token, it returns the IG user directly
                if me_data.get("username"):
                    ig_user_id = me_data["id"]
                    ig_username = me_data.get("username", ig_user_id)

            if not ig_user_id:
                # Fallback: look through Facebook pages for linked IG account
                pages_r = await client.get(f"{base}/me/accounts", params={
                    "fields": "instagram_business_account,name",
                    "access_token": access_token,
                })
                if pages_r.status_code == 200:
                    for page in pages_r.json().get("data", []):
                        iba = page.get("instagram_business_account", {})
                        if iba.get("id"):
                            ig_user_id = iba["id"]
                            ig_r = await client.get(f"{base}/{ig_user_id}", params={
                                "fields": "username",
                                "access_token": access_token,
                            })
                            if ig_r.status_code == 200:
                                ig_username = ig_r.json().get("username", ig_user_id)
                            break

        if not ig_user_id:
            raise HTTPException(
                status_code=400,
                detail="Could not find an Instagram Business account with this token. "
                       "Please also provide ig_user_id, or ensure the token has the right permissions.",
            )

        # If we have ig_user_id but not username yet, fetch it
        if not ig_username:
            ig_r = await client.get(f"{base}/{ig_user_id}", params={
                "fields": "username",
                "access_token": access_token,
            })
            if ig_r.status_code == 200:
                ig_username = ig_r.json().get("username", ig_user_id)
            else:
                ig_username = ig_user_id

    _ensure_user_dirs(uid)
    s = _load_settings(uid)
    s["ig_access_token"] = access_token
    s["ig_user_id"] = ig_user_id
    s["ig_username"] = ig_username
    _save_settings(uid, s)
    return {"status": "connected", "ig_username": ig_username, "ig_user_id": ig_user_id}


@app.delete("/api/auth/instagram")
async def instagram_revoke(request: Request) -> dict[str, str]:
    user = _require_user(request)
    uid = user["sub"]
    s = _load_settings(uid)
    s.pop("ig_access_token", None)
    s.pop("ig_user_id", None)
    s.pop("ig_username", None)
    _save_settings(uid, s)
    return {"status": "revoked"}


# ---------------------------------------------------------------------------
# Publish API
# ---------------------------------------------------------------------------


@app.post("/api/publish/instagram")
async def publish_instagram(request: Request) -> dict[str, Any]:
    user = _require_user(request)
    uid = user["sub"]
    body = await request.json()
    filename: str = body.get("filename", "")
    caption: str = body.get("caption", "")
    s = _load_settings(uid)

    ig_user_id = s.get("ig_user_id", "").strip()
    access_token = s.get("ig_access_token", "").strip()
    public_base = PUBLIC_URL or _app_base(request)

    if not ig_user_id or not access_token:
        raise HTTPException(status_code=400, detail="Connect Instagram in Settings first")

    if not filename.startswith("reel_") or not filename.endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Invalid filename")
    if not (user_outputs(uid) / filename).exists():
        raise HTTPException(status_code=404, detail="File not found")

    # Generate a signed temporary public URL so Instagram can download the video
    expires = int(time.time()) + _SHARE_TOKEN_EXPIRY
    token = _sign_share_token(uid, filename, expires)
    video_url = f"{public_base}/api/output/public/{uid}/{filename}?token={token}"
    base = "https://graph.facebook.com/v21.0"

    async with httpx.AsyncClient(timeout=60) as client:
        # Step 1: create container
        r = await client.post(f"{base}/{ig_user_id}/media", data={
            "media_type": "REELS",
            "video_url": video_url,
            "caption": caption,
            "share_to_feed": "true",
            "access_token": access_token,
        })
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail=f"IG container create failed: {r.text}")
        container_id = r.json().get("id")

        # Step 2: poll until ready
        for _ in range(30):
            await asyncio.sleep(5)
            s2 = await client.get(f"{base}/{container_id}", params={
                "fields": "status_code,status",
                "access_token": access_token,
            })
            status = s2.json().get("status_code", "")
            if status == "FINISHED":
                break
            if status in ("ERROR", "EXPIRED"):
                raise HTTPException(status_code=502, detail=f"IG video processing failed: {s2.text}")

        # Step 3: publish
        r2 = await client.post(f"{base}/{ig_user_id}/media_publish", data={
            "creation_id": container_id,
            "access_token": access_token,
        })
        if r2.status_code != 200:
            raise HTTPException(status_code=502, detail=f"IG publish failed: {r2.text}")

        media_id = r2.json().get("id")
        return {"status": "published", "media_id": media_id, "platform": "instagram"}


@app.post("/api/publish/youtube")
async def publish_youtube(request: Request) -> dict[str, Any]:
    user = _require_user(request)
    uid = user["sub"]
    body = await request.json()
    filename: str = body.get("filename", "")
    title: str = body.get("title", "My Reel")
    description: str = body.get("description", "")
    privacy: str = body.get("privacy", "public")

    yt_tok = user_yt_token(uid)
    if not yt_tok.exists():
        raise HTTPException(status_code=400, detail="YouTube not authorized — connect in Settings first")
    if not filename.startswith("reel_") or not filename.endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = user_outputs(uid) / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request as GRequest
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    token_data = json.loads(yt_tok.read_text())
    creds = Credentials(
        token=token_data["token"],
        refresh_token=token_data["refresh_token"],
        token_uri=token_data["token_uri"],
        client_id=token_data["client_id"],
        client_secret=token_data["client_secret"],
        scopes=token_data["scopes"],
    )
    if creds.expired and creds.refresh_token:
        creds.refresh(GRequest())
        token_data["token"] = creds.token
        yt_tok.write_text(json.dumps(token_data))

    def _upload() -> dict[str, Any]:
        youtube = build("youtube", "v3", credentials=creds)
        body_yt = {
            "snippet": {
                "title": title,
                "description": description,
                "categoryId": "22",  # People & Blogs
            },
            "status": {"privacyStatus": privacy, "selfDeclaredMadeForKids": False},
        }
        media = MediaFileUpload(str(path), mimetype="video/mp4", resumable=True, chunksize=10 * 1024 * 1024)
        req = youtube.videos().insert(part="snippet,status", body=body_yt, media_body=media)
        response = None
        while response is None:
            _, response = req.next_chunk()
        return response

    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, _upload)
    video_id = response.get("id", "")
    return {
        "status": "published",
        "video_id": video_id,
        "url": f"https://www.youtube.com/shorts/{video_id}",
        "platform": "youtube",
    }


# ---------------------------------------------------------------------------
# Settings page
# ---------------------------------------------------------------------------


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request) -> HTMLResponse:
    _require_user_page(request)
    return HTMLResponse(SETTINGS_HTML)


SETTINGS_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Reel Editor — Settings</title>
<style>
  :root {
    --bg:#0f0f13; --surface:#1a1a23; --surface2:#23232f;
    --accent:#FFD400; --text:#f0f0f5; --muted:#7a7a90;
    --border:#2e2e3e; --success:#4ade80; --error:#f87171; --radius:12px;
  }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
         background:var(--bg); color:var(--text); min-height:100vh; }
  header { display:flex; align-items:center; gap:12px; padding:18px 32px;
           border-bottom:1px solid var(--border); background:var(--surface); }
  header .logo { font-size:22px; font-weight:800; }
  header .logo span { color:var(--accent); }
  header nav { margin-left:auto; display:flex; gap:8px; }
  header nav a { font-size:13px; font-weight:600; color:var(--muted);
    text-decoration:none; padding:6px 14px; border-radius:8px;
    border:1px solid var(--border); transition:all 0.15s;
    display:flex; align-items:center; gap:6px; }
  header nav a:hover, header nav a.active { color:var(--accent); border-color:var(--accent); }
  .page { max-width:680px; margin:0 auto; padding:40px 24px; }
  h1 { font-size:22px; font-weight:800; margin-bottom:6px; }
  .page-sub { font-size:14px; color:var(--muted); margin-bottom:36px; }
  .section { background:var(--surface); border:1px solid var(--border);
             border-radius:var(--radius); padding:24px; margin-bottom:20px; }
  .section-title { font-size:16px; font-weight:700; margin-bottom:6px; display:flex; align-items:center; gap:8px; }
  .section-sub { font-size:13px; color:var(--muted); margin-bottom:20px; line-height:1.5; }
  .btn { display:inline-flex; align-items:center; gap:8px; padding:11px 20px;
         border-radius:9px; border:1px solid var(--border); background:var(--surface2);
         color:var(--text); font-size:14px; font-weight:600; cursor:pointer; transition:all 0.15s; }
  .btn:hover { border-color:var(--accent); color:var(--accent); }
  .btn-primary { background:var(--accent); color:#000; border-color:var(--accent); font-size:15px; padding:13px 24px; }
  .btn-primary:hover { background:#ffe033; color:#000; }
  .btn-danger { color:var(--error); border-color:var(--error); background:none; padding:8px 14px; font-size:13px; }
  .connect-btn { width:100%; justify-content:center; }
  .tag { display:inline-block; font-size:12px; font-weight:700; padding:4px 12px;
         border-radius:999px; background:var(--success); color:#000; }
  .connected-row { display:flex; align-items:center; gap:12px; flex-wrap:wrap; }
  .unavail { font-size:13px; color:var(--muted); padding:14px;
             border:1px dashed var(--border); border-radius:8px; line-height:1.6; }
  .unavail code { color:var(--accent); }
</style>
</head>
<body>
<header>
  <div class="logo">Reel<span>Forge</span></div>
  <nav>
    <a href="/render">Render</a>
    <a href="/library">Library</a>
    <a href="/settings" class="active">Settings</a>
    <a href="/auth/logout" id="nav-user">
      <img id="nav-avatar" src="" width="22" height="22" style="border-radius:50%;display:none"/>
      <span id="nav-name"></span> &middot; Sign out
    </a>
  </nav>
</header>

<div class="page">
  <h1>Settings</h1>
  <div class="page-sub">Connect your social accounts — via OAuth or manual token.</div>

  <div class="section">
    <div class="section-title">📸 Instagram</div>
    <div class="section-sub">Requires a Creator or Business account linked to a Facebook Page.</div>
    <div id="ig-section">Loading…</div>
  </div>

  <div class="section">
    <div class="section-title">▶ YouTube</div>
    <div class="section-sub">Uses the same Google account as your Reel Editor login.</div>
    <div id="yt-section">Loading…</div>
  </div>
</div>

<script>
async function connect(platform) {
  const w = window.open(`/api/auth/${platform}`, '_blank', 'width=600,height=700');
  const t = setInterval(() => { if (w.closed) { clearInterval(t); loadSettings(); } }, 500);
}
async function disconnect(platform) {
  if (!confirm(`Disconnect ${platform === 'youtube' ? 'YouTube' : 'Instagram'}?`)) return;
  await fetch(`/api/auth/${platform}`, { method: 'DELETE' });
  loadSettings();
}
async function loadSettings() {
  const r = await fetch('/api/settings');
  const s = await r.json();
  renderIG(s); renderYT(s);
}
function renderIG(s) {
  const el = document.getElementById('ig-section');
  if (s.ig_connected) {
    el.innerHTML = `<div class="connected-row"><span class="tag">✓ @${s.ig_username}</span><button class="btn btn-danger" onclick="disconnect('instagram')">Disconnect</button></div>`;
    return;
  }
  let html = '';
  if (s.ig_available) {
    html += '<button class="btn btn-primary connect-btn" onclick="connect(\'instagram\')"><svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2.163c3.204 0 3.584.012 4.85.07 3.252.148 4.771 1.691 4.919 4.919.058 1.265.069 1.645.069 4.849 0 3.205-.012 3.584-.069 4.849-.149 3.225-1.664 4.771-4.919 4.919-1.266.058-1.644.07-4.85.07-3.204 0-3.584-.012-4.849-.07-3.26-.149-4.771-1.699-4.919-4.92-.058-1.265-.07-1.644-.07-4.849 0-3.204.013-3.583.07-4.849.149-3.227 1.664-4.771 4.919-4.919 1.266-.057 1.645-.069 4.849-.069zm0-2.163c-3.259 0-3.667.014-4.947.072-4.358.2-6.78 2.618-6.98 6.98-.059 1.281-.073 1.689-.073 4.948 0 3.259.014 3.668.072 4.948.2 4.358 2.618 6.78 6.98 6.98 1.281.058 1.689.072 4.948.072 3.259 0 3.668-.014 4.948-.072 4.354-.2 6.782-2.618 6.979-6.98.059-1.28.073-1.689.073-4.948 0-3.259-.014-3.667-.072-4.947-.196-4.354-2.617-6.78-6.979-6.98-1.281-.059-1.69-.073-4.949-.073zm0 5.838c-3.403 0-6.162 2.759-6.162 6.162s2.759 6.163 6.162 6.163 6.162-2.759 6.162-6.163c0-3.403-2.759-6.162-6.162-6.162zm0 10.162c-2.209 0-4-1.79-4-4 0-2.209 1.791-4 4-4s4 1.791 4 4c0 2.21-1.791 4-4 4zm6.406-11.845c-.796 0-1.441.645-1.441 1.44s.645 1.44 1.441 1.44c.795 0 1.439-.645 1.439-1.44s-.644-1.44-1.439-1.44z"/></svg> Connect Instagram</button>';
    html += '<div style="text-align:center;color:var(--muted);font-size:12px;margin:12px 0 8px">— or paste a token from developers.facebook.com —</div>';
  }
  html += `<div style="display:flex;flex-direction:column;gap:10px">
    <input id="ig-token" type="text" placeholder="Access token from developers.facebook.com"
      style="width:100%;padding:10px 14px;border-radius:8px;border:1px solid var(--border);background:var(--surface2);color:var(--text);font-size:13px;font-family:monospace;outline:none;" />
    <input id="ig-userid" type="text" placeholder="Instagram User ID (optional — auto-detected from token)"
      style="width:100%;padding:10px 14px;border-radius:8px;border:1px solid var(--border);background:var(--surface2);color:var(--text);font-size:13px;font-family:monospace;outline:none;" />
    <button class="btn btn-primary connect-btn" onclick="saveManualIG()" id="ig-save-btn">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M5 13l4 4L19 7"/></svg>
      Save Token
    </button>
    <div id="ig-manual-status" style="font-size:13px;min-height:20px;"></div>
  </div>`;
  el.innerHTML = html;
}
async function saveManualIG() {
  const token = document.getElementById('ig-token').value.trim();
  const userId = document.getElementById('ig-userid').value.trim();
  const status = document.getElementById('ig-manual-status');
  const btn = document.getElementById('ig-save-btn');
  if (!token) { status.innerHTML = '<span style="color:var(--error)">Please paste an access token.</span>'; return; }
  btn.disabled = true; btn.style.opacity = '0.5';
  status.innerHTML = '<span style="color:var(--muted)">Verifying token…</span>';
  try {
    const r = await fetch('/api/auth/instagram/manual', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ access_token: token, ig_user_id: userId }),
    });
    const d = await r.json();
    if (r.ok) {
      status.innerHTML = `<span style="color:var(--success)">✓ Connected @${d.ig_username}</span>`;
      setTimeout(loadSettings, 1000);
    } else {
      status.innerHTML = `<span style="color:var(--error)">${d.detail || 'Failed to connect'}</span>`;
    }
  } catch (e) {
    status.innerHTML = `<span style="color:var(--error)">Error: ${e.message}</span>`;
  }
  btn.disabled = false; btn.style.opacity = '1';
}
function renderYT(s) {
  const el = document.getElementById('yt-section');
  if (!s.yt_available) {
    el.innerHTML = '<div class="unavail">YouTube OAuth not configured.<br/><code>GOOGLE_CLIENT_ID</code> and <code>GOOGLE_CLIENT_SECRET</code> env vars needed on the server.</div>';
    return;
  }
  el.innerHTML = s.yt_connected
    ? `<div class="connected-row"><span class="tag">✓ ${s.yt_channel || 'YouTube connected'}</span><button class="btn btn-danger" onclick="disconnect('youtube')">Disconnect</button></div>`
    : '<button class="btn btn-primary connect-btn" onclick="connect(\'youtube\')"><svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M23.495 6.205a3.007 3.007 0 0 0-2.088-2.088c-1.87-.501-9.396-.501-9.396-.501s-7.507-.01-9.396.501A3.007 3.007 0 0 0 .527 6.205a31.247 31.247 0 0 0-.522 5.805 31.247 31.247 0 0 0 .522 5.783 3.007 3.007 0 0 0 2.088 2.088c1.868.502 9.396.502 9.396.502s7.506 0 9.396-.502a3.007 3.007 0 0 0 2.088-2.088 31.247 31.247 0 0 0 .5-5.783 31.247 31.247 0 0 0-.5-5.805zM9.609 15.601V8.408l6.264 3.602z"/></svg> Connect YouTube</button>';
}
fetch('/api/me').then(r=>r.json()).then(u=>{
  document.getElementById('nav-name').textContent = u.name.split(' ')[0];
  const img = document.getElementById('nav-avatar');
  if(u.picture){img.src=u.picture;img.style.display='';}
}).catch(()=>{});
loadSettings();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Library — list all completed reels on disk
# ---------------------------------------------------------------------------


@app.get("/api/library")
async def list_library(request: Request) -> list[dict[str, Any]]:
    user = _require_user(request)
    uid = user["sub"]
    out_dir = user_outputs(uid)
    if not out_dir.exists():
        return []

    reels = []
    for p in sorted(out_dir.glob("reel_*.mp4"), key=lambda f: f.stat().st_mtime, reverse=True):
        stat = p.stat()
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


@app.get("/editor")
async def editor_entry(request: Request) -> RedirectResponse:
    if request.session.get("user"):
        return RedirectResponse("/render")
    return RedirectResponse("/auth/login?next=/render")


@app.get("/library", response_class=HTMLResponse)
async def library_page(request: Request) -> HTMLResponse:
    _require_user_page(request)
    return HTMLResponse(LIBRARY_HTML)


# ---------------------------------------------------------------------------
# Library HTML
# ---------------------------------------------------------------------------

LIBRARY_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Reel Editor — Library</title>
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

  /* Publish modal */
  .pub-modal-bg {
    display: none; position: fixed; inset: 0;
    background: rgba(0,0,0,0.85); z-index: 200;
    align-items: center; justify-content: center;
  }
  .pub-modal-bg.open { display: flex; }
  .pub-modal {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 24px; max-width: 460px; width: 90%;
  }
  .pub-title { font-size: 16px; font-weight: 800; margin-bottom: 6px; }
  .pub-sub { font-size: 13px; color: var(--muted); margin-bottom: 18px; }
  .platform-toggle { display: flex; gap: 10px; margin-bottom: 18px; }
  .plat-btn {
    flex: 1; padding: 12px; border-radius: 10px;
    border: 2px solid var(--border); background: var(--surface2);
    color: var(--muted); font-size: 13px; font-weight: 700;
    cursor: pointer; text-align: center; transition: all 0.15s;
  }
  .plat-btn.on { border-color: var(--accent); color: var(--accent); background: rgba(255,212,0,0.08); }
  .pub-field { margin-bottom: 14px; }
  .pub-field label { display: block; font-size: 11px; font-weight: 700; color: var(--muted);
                     text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 5px; }
  .pub-field input, .pub-field textarea, .pub-field select {
    width: 100%; padding: 9px 12px; background: var(--surface2);
    border: 1px solid var(--border); border-radius: 8px; color: var(--text);
    font-size: 13px; outline: none; transition: border-color 0.15s;
    font-family: inherit; resize: vertical;
  }
  .pub-field input:focus, .pub-field textarea:focus, .pub-field select:focus { border-color: var(--accent); }
  .pub-actions { display: flex; gap: 10px; margin-top: 20px; }
  .pub-status { font-size: 13px; color: var(--muted); margin-top: 10px; min-height: 20px; }
  .pub-status.ok { color: var(--success); }
  .pub-status.err { color: var(--error); }

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
  .btn-danger { color: var(--error); border-color: transparent; background: none; flex: 0; padding: 8px 10px; }
  .btn-danger:hover { background: rgba(248,113,113,0.1); border-color: var(--error); color: var(--error); }

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
    <a href="/render">Render</a>
    <a href="/library" class="active">Library</a>
    <a href="/settings">Settings</a>
    <a href="/auth/logout" id="nav-user" style="display:flex;align-items:center;gap:6px;">
      <img id="nav-avatar" src="" width="22" height="22" style="border-radius:50%;display:none"/>
      <span id="nav-name"></span> · Sign out
    </a>
  </nav>
</header>
<script>
fetch('/api/me').then(r=>r.json()).then(u=>{
  document.getElementById('nav-name').textContent = u.name.split(' ')[0];
  const img = document.getElementById('nav-avatar');
  if(u.picture){img.src=u.picture;img.style.display='';}
}).catch(()=>{});
</script>

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

<!-- Publish modal -->
<div class="pub-modal-bg" id="pub-modal">
  <div class="pub-modal">
    <div class="pub-title">📤 Publish Reel</div>
    <div class="pub-sub" id="pub-filename"></div>

    <div class="platform-toggle">
      <button class="plat-btn" id="plat-ig" onclick="togglePlat('ig')">📸 Instagram</button>
      <button class="plat-btn" id="plat-yt" onclick="togglePlat('yt')">▶ YouTube</button>
    </div>

    <div class="pub-field">
      <label>Caption / Description</label>
      <textarea id="pub-caption" rows="3" placeholder="Write your caption…"></textarea>
    </div>
    <div class="pub-field" id="pub-title-field">
      <label>YouTube Title</label>
      <input type="text" id="pub-yt-title" placeholder="My Reel"/>
    </div>
    <div class="pub-field" id="pub-privacy-field">
      <label>YouTube Privacy</label>
      <select id="pub-privacy">
        <option value="public">Public</option>
        <option value="unlisted">Unlisted</option>
        <option value="private">Private</option>
      </select>
    </div>

    <div class="pub-status" id="pub-status"></div>

    <div class="pub-actions">
      <button class="btn btn-primary" id="pub-go-btn" onclick="doPublish()">Publish</button>
      <button class="btn" onclick="closePubModal()">Cancel</button>
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
    grid.innerHTML = '<div class="empty">No reels yet. <a href="/render">Render your first one →</a></div>';
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
          <button class="btn" onclick="openModal('${reel.filename}','${reel.label}')">▶</button>
          <button class="btn" onclick="openPubModal('${reel.filename}','${reel.label}')" title="Publish">📤</button>
          <button class="btn btn-danger" onclick="deleteReel('${reel.filename}', this)" title="Delete">🗑</button>
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

async function deleteReel(filename, btn) {
  if (!confirm(`Delete ${filename}?\nThis cannot be undone.`)) return;
  btn.disabled = true;
  btn.textContent = '…';
  const r = await fetch(`/api/output/${filename}`, { method: 'DELETE' });
  if (r.ok) {
    btn.closest('.reel-card').style.transition = 'opacity 0.3s';
    btn.closest('.reel-card').style.opacity = '0';
    setTimeout(() => { btn.closest('.reel-card').remove(); updateCount(); }, 300);
  } else {
    btn.textContent = '🗑';
    btn.disabled = false;
    alert('Delete failed');
  }
}

function updateCount() {
  const n = document.querySelectorAll('.reel-card').length;
  document.getElementById('count').textContent = n;
  if (n === 0) document.getElementById('grid').innerHTML =
    '<div class="empty">No reels yet. <a href="/render">Render your first one →</a></div>';
}

// ---- Publish modal ----
let _pubFile = '';
let _pubPlatforms = new Set();

function openPubModal(filename, label) {
  _pubFile = filename;
  _pubPlatforms = new Set();
  document.getElementById('pub-filename').textContent = label || filename;
  document.getElementById('pub-caption').value = '';
  document.getElementById('pub-yt-title').value = label || '';
  document.getElementById('pub-status').textContent = '';
  document.getElementById('pub-status').className = 'pub-status';
  document.getElementById('pub-go-btn').disabled = false;
  document.getElementById('pub-go-btn').textContent = 'Publish';
  ['plat-ig','plat-yt'].forEach(id => document.getElementById(id).classList.remove('on'));
  updatePubFields();
  document.getElementById('pub-modal').classList.add('open');
}

function closePubModal() { document.getElementById('pub-modal').classList.remove('open'); }

function togglePlat(p) {
  if (_pubPlatforms.has(p)) _pubPlatforms.delete(p); else _pubPlatforms.add(p);
  document.getElementById('plat-ig').className = 'plat-btn' + (_pubPlatforms.has('ig') ? ' on' : '');
  document.getElementById('plat-yt').className = 'plat-btn' + (_pubPlatforms.has('yt') ? ' on' : '');
  updatePubFields();
}

function updatePubFields() {
  const hasYT = _pubPlatforms.has('yt');
  document.getElementById('pub-title-field').style.display = hasYT ? '' : 'none';
  document.getElementById('pub-privacy-field').style.display = hasYT ? '' : 'none';
}

async function doPublish() {
  if (_pubPlatforms.size === 0) {
    const el = document.getElementById('pub-status');
    el.textContent = 'Pick at least one platform';
    el.className = 'pub-status err';
    return;
  }
  const btn = document.getElementById('pub-go-btn');
  btn.disabled = true;
  btn.textContent = 'Publishing…';
  const caption = document.getElementById('pub-caption').value;
  const title = document.getElementById('pub-yt-title').value || _pubFile;
  const privacy = document.getElementById('pub-privacy').value;
  const results = [];

  for (const plat of _pubPlatforms) {
    const el = document.getElementById('pub-status');
    el.textContent = `Posting to ${plat === 'ig' ? 'Instagram' : 'YouTube'}…`;
    el.className = 'pub-status';
    try {
      const body = plat === 'ig'
        ? { filename: _pubFile, caption }
        : { filename: _pubFile, title, description: caption, privacy };
      const r = await fetch(`/api/publish/${plat === 'ig' ? 'instagram' : 'youtube'}`, {
        method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body),
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data.detail || 'Unknown error');
      if (plat === 'yt' && data.url) results.push(`YouTube: ✓ <a href="${data.url}" target="_blank">View Reel</a>`);
      else results.push(`${plat === 'ig' ? 'Instagram' : 'YouTube'}: ✓ Published`);
    } catch(e) {
      results.push(`${plat === 'ig' ? 'Instagram' : 'YouTube'}: ✗ ${e.message}`);
    }
  }

  const allOk = results.every(r => r.includes('✓'));
  const el = document.getElementById('pub-status');
  el.innerHTML = results.join('<br>');
  el.className = 'pub-status ' + (allOk ? 'ok' : 'err');
  btn.textContent = 'Done';
}

document.getElementById('pub-modal').addEventListener('click', e => {
  if (e.target === document.getElementById('pub-modal')) closePubModal();
});
updatePubFields();

loadLibrary();
</script>
</body>
</html>
"""
