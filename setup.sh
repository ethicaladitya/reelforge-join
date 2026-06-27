#!/usr/bin/env bash
# ReelForge setup + launcher
# Usage:
#   ./setup.sh          → first-time setup (checks deps, creates .env)
#   ./setup.sh start    → start server + Cloudflare tunnel
#   ./setup.sh stop     → stop everything
#   ./setup.sh status   → show what's running

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"
PID_DIR="$SCRIPT_DIR/.pids"
LOG_DIR="$SCRIPT_DIR/.logs"

# ─── colours ────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}▸${RESET} $*"; }
success() { echo -e "${GREEN}✓${RESET} $*"; }
warn()    { echo -e "${YELLOW}⚠${RESET} $*"; }
error()   { echo -e "${RED}✗${RESET} $*"; }
header()  { echo -e "\n${BOLD}$*${RESET}"; }

# ─── load .env ───────────────────────────────────────────────────────────────
load_env() {
  if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
  fi
}

# ─── setup (first-time) ──────────────────────────────────────────────────────
cmd_setup() {
  header "ReelForge Setup"

  # 1. Check uv
  if ! command -v uv &>/dev/null; then
    error "uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
  fi
  success "uv found ($(uv --version))"

  # 2. Check ffmpeg-full
  FFMPEG_FULL="/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg"
  if [[ -f "$FFMPEG_FULL" ]]; then
    success "ffmpeg-full found at $FFMPEG_FULL"
  elif command -v ffmpeg &>/dev/null; then
    warn "Standard ffmpeg found but ffmpeg-full is preferred (adds caption/subtitle support)"
    warn "Install with: brew install ffmpeg-full"
  else
    error "ffmpeg not found. Install with: brew install ffmpeg-full"
    exit 1
  fi

  # 3. Check cloudflared (optional)
  if command -v cloudflared &>/dev/null; then
    success "cloudflared found"
  else
    warn "cloudflared not found (optional, needed for public URL via Cloudflare Tunnel)"
    warn "Install with: brew install cloudflared"
  fi

  # 4. Install Python deps
  info "Installing Python dependencies..."
  cd "$SCRIPT_DIR"
  uv sync --quiet
  success "Dependencies installed"

  # 5. Create .env if missing
  if [[ ! -f "$ENV_FILE" ]]; then
    cp "$SCRIPT_DIR/.env.example" "$ENV_FILE"
    success "Created .env from .env.example"
    echo ""
    warn "Fill in your credentials in .env before starting:"
    warn "  Required: GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, PUBLIC_URL"
    warn "  Optional: FACEBOOK_APP_ID, FACEBOOK_APP_SECRET (for Instagram)"
    echo ""
    info "Then run: ./setup.sh start"
  else
    success ".env already exists"

    # Validate required fields
    load_env
    MISSING=()
    [[ -z "${GOOGLE_CLIENT_ID:-}" ]]     && MISSING+=("GOOGLE_CLIENT_ID")
    [[ -z "${GOOGLE_CLIENT_SECRET:-}" ]] && MISSING+=("GOOGLE_CLIENT_SECRET")
    [[ -z "${PUBLIC_URL:-}" ]]           && MISSING+=("PUBLIC_URL")

    if [[ ${#MISSING[@]} -gt 0 ]]; then
      warn "Missing required values in .env: ${MISSING[*]}"
    else
      success "All required env vars are set"
      echo ""
      info "Run: ./setup.sh start"
    fi
  fi
}

# ─── start ───────────────────────────────────────────────────────────────────
cmd_start() {
  load_env
  mkdir -p "$PID_DIR" "$LOG_DIR"

  # Validate required env vars
  MISSING=()
  [[ -z "${GOOGLE_CLIENT_ID:-}" ]]     && MISSING+=("GOOGLE_CLIENT_ID")
  [[ -z "${GOOGLE_CLIENT_SECRET:-}" ]] && MISSING+=("GOOGLE_CLIENT_SECRET")
  [[ -z "${PUBLIC_URL:-}" ]]           && MISSING+=("PUBLIC_URL")
  if [[ ${#MISSING[@]} -gt 0 ]]; then
    error "Missing in .env: ${MISSING[*]}"
    error "Run ./setup.sh first"
    exit 1
  fi

  HOST="${HOST:-127.0.0.1}"
  PORT="${PORT:-7433}"
  CF_TUNNEL="${CF_TUNNEL_NAME:-reelforge}"

  header "Starting ReelForge"

  # Stop any existing processes
  cmd_stop_silent

  # Start server
  info "Starting ReelForge server on $HOST:$PORT..."
  cd "$SCRIPT_DIR"
  nohup uv run reelforge ui \
    --host "$HOST" \
    --port "$PORT" \
    --no-open \
    > "$LOG_DIR/server.log" 2>&1 &
  echo $! > "$PID_DIR/server.pid"
  success "Server started (PID $(cat "$PID_DIR/server.pid"))"

  # Wait for server to be ready
  for i in {1..15}; do
    if curl -sf "http://$HOST:$PORT/auth/login" -o /dev/null 2>/dev/null; then
      break
    fi
    sleep 1
  done

  # Start Cloudflare tunnel (if cloudflared installed and tunnel name set)
  if command -v cloudflared &>/dev/null && [[ -n "$CF_TUNNEL" ]]; then
    info "Starting Cloudflare tunnel '$CF_TUNNEL'..."
    nohup cloudflared tunnel run "$CF_TUNNEL" \
      > "$LOG_DIR/cloudflared.log" 2>&1 &
    echo $! > "$PID_DIR/cloudflared.pid"
    sleep 2
    success "Tunnel started (PID $(cat "$PID_DIR/cloudflared.pid"))"
  else
    warn "Skipping Cloudflare tunnel (cloudflared not found or CF_TUNNEL_NAME not set)"
  fi

  echo ""
  echo -e "  ${BOLD}Local:${RESET}  http://$HOST:$PORT"
  [[ -n "${PUBLIC_URL:-}" ]] && echo -e "  ${BOLD}Public:${RESET} $PUBLIC_URL"
  echo ""
  echo -e "  Logs:  tail -f $LOG_DIR/server.log"
  echo -e "  Stop:  ./setup.sh stop"
  echo ""
}

# ─── stop ────────────────────────────────────────────────────────────────────
cmd_stop_silent() {
  for name in server cloudflared; do
    PID_FILE="$PID_DIR/$name.pid"
    if [[ -f "$PID_FILE" ]]; then
      PID=$(cat "$PID_FILE")
      kill "$PID" 2>/dev/null || true
      rm -f "$PID_FILE"
    fi
  done
  # Also kill by port/name as fallback
  lsof -ti :${PORT:-7433} 2>/dev/null | xargs kill -9 2>/dev/null || true
}

cmd_stop() {
  header "Stopping ReelForge"
  load_env
  for name in server cloudflared; do
    PID_FILE="$PID_DIR/$name.pid"
    if [[ -f "$PID_FILE" ]]; then
      PID=$(cat "$PID_FILE")
      if kill "$PID" 2>/dev/null; then
        success "Stopped $name (PID $PID)"
      else
        warn "$name was not running (stale PID $PID)"
      fi
      rm -f "$PID_FILE"
    else
      warn "$name PID file not found (may not be running)"
    fi
  done
  lsof -ti :"${PORT:-7433}" 2>/dev/null | xargs kill -9 2>/dev/null || true
}

# ─── status ──────────────────────────────────────────────────────────────────
cmd_status() {
  load_env
  HOST="${HOST:-127.0.0.1}"
  PORT="${PORT:-7433}"

  header "ReelForge Status"

  # Server
  SERVER_PID_FILE="$PID_DIR/server.pid"
  if [[ -f "$SERVER_PID_FILE" ]] && kill -0 "$(cat "$SERVER_PID_FILE")" 2>/dev/null; then
    success "Server running (PID $(cat "$SERVER_PID_FILE"))"
  else
    error "Server not running"
  fi

  # Tunnel
  CF_PID_FILE="$PID_DIR/cloudflared.pid"
  if [[ -f "$CF_PID_FILE" ]] && kill -0 "$(cat "$CF_PID_FILE")" 2>/dev/null; then
    success "Cloudflare tunnel running (PID $(cat "$CF_PID_FILE"))"
  else
    warn "Cloudflare tunnel not running"
  fi

  # Health check
  if curl -sf "http://$HOST:$PORT/auth/login" -o /dev/null 2>/dev/null; then
    success "Server is responding at http://$HOST:$PORT"
  else
    error "Server is not responding"
  fi
}

# ─── logs ────────────────────────────────────────────────────────────────────
cmd_logs() {
  TARGET="${2:-server}"
  LOG_FILE="$LOG_DIR/$TARGET.log"
  if [[ -f "$LOG_FILE" ]]; then
    tail -f "$LOG_FILE"
  else
    error "No log file found at $LOG_FILE"
    info  "Available: server, cloudflared"
  fi
}

# ─── dispatch ────────────────────────────────────────────────────────────────
case "${1:-setup}" in
  setup)   cmd_setup ;;
  start)   cmd_start ;;
  stop)    cmd_stop ;;
  restart) cmd_stop; sleep 1; cmd_start ;;
  status)  cmd_status ;;
  logs)    cmd_logs "$@" ;;
  *)
    echo "Usage: $0 {setup|start|stop|restart|status|logs [server|cloudflared]}"
    exit 1
    ;;
esac
