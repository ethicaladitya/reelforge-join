#!/usr/bin/env bash
# Reel Editor — setup, deploy, and management script
#
# LOCAL (macOS):
#   ./setup.sh              → first-time local setup
#   ./setup.sh start        → start server + Cloudflare tunnel
#   ./setup.sh stop         → stop everything
#   ./setup.sh restart      → stop + start
#   ./setup.sh status       → show status
#   ./setup.sh logs         → tail logs
#
# SERVER (Ubuntu, run as root):
#   sudo ./setup.sh deploy [--domain example.com] [--email admin@example.com] [--app-dir /opt/reelforge]
#     → full install on fresh server, OR just updates code if already deployed
#   sudo ./setup.sh update
#     → pull latest code + restart (alias for deploy when already set up)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"
PID_DIR="$SCRIPT_DIR/.pids"
LOG_DIR="$SCRIPT_DIR/.logs"
SERVICE_NAME="reelforge"
DEFAULT_APP_DIR="/var/www/html/reeleditor"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}▸${RESET} $*"; }
success() { echo -e "${GREEN}✓${RESET} $*"; }
warn()    { echo -e "${YELLOW}⚠${RESET} $*"; }
error()   { echo -e "${RED}✗${RESET} $*"; }
header()  { echo -e "\n${BOLD}── $* ──${RESET}"; }
skip()    { echo -e "  ${BOLD}[skip]${RESET} $* (already done)"; }

load_env() {
  local f="${1:-$ENV_FILE}"
  if [[ -f "$f" ]]; then
    set -a; source "$f"; set +a
  fi
}

# ─────────────────────────────────────────────────────────────────────────────
# DEPLOY  (idempotent — safe to re-run; updates code if already installed)
# ─────────────────────────────────────────────────────────────────────────────
cmd_deploy() {
  # ── parse flags ────────────────────────────────────────────────────────────
  DEPLOY_DOMAIN=""
  DEPLOY_EMAIL=""
  APP_DIR="$DEFAULT_APP_DIR"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --domain)  DEPLOY_DOMAIN="$2"; shift 2 ;;
      --email)   DEPLOY_EMAIL="$2";  shift 2 ;;
      --app-dir) APP_DIR="$2";       shift 2 ;;
      *) shift ;;
    esac
  done

  if [[ "$EUID" -ne 0 ]]; then
    error "deploy must run as root: sudo ./setup.sh deploy"
    exit 1
  fi

  # Load .env from app dir (or script dir) to get defaults
  load_env "$APP_DIR/.env" 2>/dev/null || load_env "$SCRIPT_DIR/.env" 2>/dev/null || true

  # Resolve domain
  if [[ -z "$DEPLOY_DOMAIN" ]]; then
    RAW="${PUBLIC_URL:-}"
    DEPLOY_DOMAIN="${RAW#https://}"; DEPLOY_DOMAIN="${DEPLOY_DOMAIN#http://}"; DEPLOY_DOMAIN="${DEPLOY_DOMAIN%%/*}"
  fi
  [[ -z "$DEPLOY_DOMAIN" ]] && { error "Set --domain or PUBLIC_URL in .env"; exit 1; }

  [[ -z "$DEPLOY_EMAIL" ]] && DEPLOY_EMAIL="admin@${DEPLOY_DOMAIN}"
  APP_PORT="${PORT:-7433}"

  # ── detect existing install ─────────────────────────────────────────────────
  IS_UPDATE=false
  systemctl is-active --quiet "$SERVICE_NAME" 2>/dev/null && IS_UPDATE=true

  if $IS_UPDATE; then
    echo -e "
${BOLD}Reel Editor Update${RESET}  (existing install detected)"
  else
    echo -e "
${BOLD}Reel Editor Server Deploy${RESET}"
  fi
  info "Domain:  $DEPLOY_DOMAIN"
  info "App dir: $APP_DIR"
  info "Port:    $APP_PORT"

  # ── 1. system packages ──────────────────────────────────────────────────────
  header "System packages"
  NEED_PKGS=()
  command -v ffmpeg    &>/dev/null || NEED_PKGS+=(ffmpeg)
  command -v nginx     &>/dev/null || NEED_PKGS+=(nginx)
  command -v certbot   &>/dev/null || NEED_PKGS+=(certbot python3-certbot-nginx)
  command -v git       &>/dev/null || NEED_PKGS+=(git)
  command -v rsync     &>/dev/null || NEED_PKGS+=(rsync)
  dpkg -s build-essential &>/dev/null 2>&1 || NEED_PKGS+=(build-essential python3-dev)

  if [[ ${#NEED_PKGS[@]} -gt 0 ]]; then
    info "Installing: ${NEED_PKGS[*]}"
    DEBIAN_FRONTEND=noninteractive apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "${NEED_PKGS[@]}"
    success "Packages installed"
  else
    skip "All system packages"
  fi

  # ── 2. uv ────────────────────────────────────────────────────────────────────
  header "uv (Python package manager)"
  export PATH="/root/.local/bin:/usr/local/bin:$PATH"
  if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | env HOME=/root sh
    # symlink so it's available system-wide
    ln -sf /root/.local/bin/uv  /usr/local/bin/uv
    ln -sf /root/.local/bin/uvx /usr/local/bin/uvx
    success "uv installed"
  else
    skip "uv ($(uv --version))"
  fi
  UV_BIN="$(command -v uv)"

  # ── 3. copy / update app code ────────────────────────────────────────────────
  header "Application code → $APP_DIR"
  mkdir -p "$APP_DIR"
  rsync -a --delete \
    --exclude='.git/' \
    --exclude='output/' \
    --exclude='.pids/' \
    --exclude='.logs/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.venv/' \
    --exclude='.env' \
    "$SCRIPT_DIR/" "$APP_DIR/"
  mkdir -p "$APP_DIR/output"
  success "Code deployed to $APP_DIR"

  # ── 4. .env ──────────────────────────────────────────────────────────────────
  header ".env configuration"
  if [[ ! -f "$APP_DIR/.env" ]]; then
    if [[ -f "$APP_DIR/.env.example" ]]; then
      cp "$APP_DIR/.env.example" "$APP_DIR/.env"
      warn "Created $APP_DIR/.env from .env.example — fill in credentials!"
    fi
  else
    skip ".env (already exists — preserving)"
  fi
  # Always ensure PUBLIC_URL matches the deploy domain in .env
  if [[ -f "$APP_DIR/.env" ]]; then
    if grep -q "^PUBLIC_URL=" "$APP_DIR/.env"; then
      sed -i "s|^PUBLIC_URL=.*|PUBLIC_URL=https://$DEPLOY_DOMAIN|" "$APP_DIR/.env"
    else
      echo "PUBLIC_URL=https://$DEPLOY_DOMAIN" >> "$APP_DIR/.env"
    fi
    success "PUBLIC_URL set to https://$DEPLOY_DOMAIN"
  fi

  # ── 5. Python dependencies ────────────────────────────────────────────────────
  header "Python dependencies"
  cd "$APP_DIR"
  "$UV_BIN" sync --quiet
  success "Dependencies up to date"

  # ── 6. systemd service ────────────────────────────────────────────────────────
  header "systemd service"
  SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
  cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=Reel Editor Video Tool
After=network.target

[Service]
Type=simple
WorkingDirectory=$APP_DIR
EnvironmentFile=$APP_DIR/.env
ExecStart=$UV_BIN run reelforge ui --host 127.0.0.1 --port $APP_PORT --no-open
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
  systemctl daemon-reload
  systemctl enable "$SERVICE_NAME" --quiet

  # Check credentials before starting
  load_env "$APP_DIR/.env"
  if [[ -z "${GOOGLE_CLIENT_ID:-}" || -z "${GOOGLE_CLIENT_SECRET:-}" ]]; then
    warn "GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET not set in $APP_DIR/.env"
    warn "Fill them in, then: systemctl start $SERVICE_NAME"
  else
    systemctl restart "$SERVICE_NAME"
    sleep 2
    if systemctl is-active --quiet "$SERVICE_NAME"; then
      success "Service running"
    else
      error "Service failed to start — check: journalctl -u $SERVICE_NAME -n 30 --no-pager"
      journalctl -u "$SERVICE_NAME" -n 20 --no-pager || true
    fi
  fi

  # ── 7. SSL certificate ────────────────────────────────────────────────────────
  header "SSL certificate"
  CERT_PATH="/etc/letsencrypt/live/$DEPLOY_DOMAIN/fullchain.pem"
  if [[ -f "$CERT_PATH" ]]; then
    skip "SSL cert for $DEPLOY_DOMAIN (expires: $(openssl x509 -noout -enddate -in "$CERT_PATH" | cut -d= -f2))"
  else
    info "Obtaining SSL cert for $DEPLOY_DOMAIN via Let's Encrypt..."
    # Temp nginx config for ACME webroot challenge
    mkdir -p /var/www/certbot
    cat > /etc/nginx/sites-available/reelforge_tmp <<NGINXEOF
server {
    listen 80;
    listen [::]:80;
    server_name $DEPLOY_DOMAIN;
    location /.well-known/acme-challenge/ {
        root /var/www/certbot;
        try_files \$uri =404;
    }
    location / { return 200 'ok'; add_header Content-Type text/plain; }
}
NGINXEOF
    ln -sf /etc/nginx/sites-available/reelforge_tmp /etc/nginx/sites-enabled/reelforge
    rm -f /etc/nginx/sites-enabled/default
    nginx -t -q && systemctl reload nginx

    certbot certonly --webroot -w /var/www/certbot \
      -d "$DEPLOY_DOMAIN" \
      --non-interactive --agree-tos \
      --email "$DEPLOY_EMAIL" \
      --expand --quiet
    rm -f /etc/nginx/sites-available/reelforge_tmp
    success "SSL certificate obtained"
  fi

  # ── 8. nginx vhost ────────────────────────────────────────────────────────────
  header "nginx vhost"
  NGINX_CONF="/etc/nginx/sites-available/reelforge"
  # Write (or overwrite) the full HTTPS config
  cat > "$NGINX_CONF" <<NGINXEOF
server {
    listen 80;
    listen [::]:80;
    server_name $DEPLOY_DOMAIN;
    location /.well-known/acme-challenge/ {
        root /var/www/certbot;
        try_files \$uri =404;
    }
    location / {
        return 301 https://\$host\$request_uri;
    }
}

server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name $DEPLOY_DOMAIN;

    ssl_certificate     /etc/letsencrypt/live/$DEPLOY_DOMAIN/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/$DEPLOY_DOMAIN/privkey.pem;
    include /etc/letsencrypt/options-ssl-nginx.conf;
    ssl_dhparam /etc/letsencrypt/ssl-dhparams.pem;

    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;

    client_max_body_size 2G;
    proxy_request_buffering off;

    # Marketing website — serve docs/ as static files at root
    root $APP_DIR/docs;

    location = / {
        try_files /index.html =404;
    }

    location ~ ^/(privacy|terms|support)\.html$ {
        try_files \$uri =404;
    }

    # App routes — everything else proxies to FastAPI
    location / {
        proxy_pass http://127.0.0.1:$APP_PORT;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Range \$http_range;
        proxy_set_header If-Range \$http_if_range;
        proxy_pass_header Content-Range;
        proxy_read_timeout 600s;
        proxy_send_timeout 600s;
        proxy_connect_timeout 60s;
    }
}
NGINXEOF

  ln -sf "$NGINX_CONF" /etc/nginx/sites-enabled/reelforge
  rm -f /etc/nginx/sites-enabled/default

  if nginx -t -q 2>/dev/null; then
    systemctl reload nginx
    success "nginx configured for https://$DEPLOY_DOMAIN"
  else
    error "nginx config has errors:"
    nginx -t
    exit 1
  fi

  # ── certbot auto-renew ───────────────────────────────────────────────────────
  if ! systemctl is-enabled --quiet certbot.timer 2>/dev/null; then
    systemctl enable certbot.timer --quiet 2>/dev/null || true
    success "Certbot auto-renew enabled"
  fi

  # ── done ─────────────────────────────────────────────────────────────────────
  echo ""
  echo -e "  ${BOLD}URL:${RESET}      https://$DEPLOY_DOMAIN"
  echo -e "  ${BOLD}Config:${RESET}   $APP_DIR/.env"
  echo -e "  ${BOLD}Logs:${RESET}     journalctl -u $SERVICE_NAME -f"
  echo -e "  ${BOLD}Restart:${RESET}  systemctl restart $SERVICE_NAME"
  echo -e "  ${BOLD}Update:${RESET}   sudo ./setup.sh deploy"
  echo ""
  if $IS_UPDATE; then
    success "Update complete!"
  else
    success "Deploy complete!"
  fi
}

# ─────────────────────────────────────────────────────────────────────────────
# LOCAL SETUP  (macOS)
# ─────────────────────────────────────────────────────────────────────────────
cmd_setup() {
  header "Reel Editor Local Setup"

  if ! command -v uv &>/dev/null; then
    error "uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
  fi
  success "uv ($(uv --version))"

  FFMPEG_FULL="/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg"
  if [[ -f "$FFMPEG_FULL" ]]; then
    success "ffmpeg-full at $FFMPEG_FULL"
  elif command -v ffmpeg &>/dev/null; then
    warn "Standard ffmpeg found; ffmpeg-full preferred: brew install ffmpeg-full"
  else
    error "ffmpeg not found: brew install ffmpeg-full"; exit 1
  fi

  command -v cloudflared &>/dev/null \
    && success "cloudflared found" \
    || warn "cloudflared not found (optional): brew install cloudflared"

  cd "$SCRIPT_DIR" && uv sync --quiet
  success "Python dependencies installed"

  if [[ ! -f "$ENV_FILE" ]]; then
    cp "$SCRIPT_DIR/.env.example" "$ENV_FILE"
    success "Created .env from .env.example"
    echo ""
    warn "Fill in .env before starting:"
    warn "  Required: GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, PUBLIC_URL"
    warn "  Optional: FACEBOOK_APP_ID, FACEBOOK_APP_SECRET"
    info "Then: ./setup.sh start"
  else
    success ".env exists"
    load_env
    MISSING=()
    [[ -z "${GOOGLE_CLIENT_ID:-}" ]]     && MISSING+=("GOOGLE_CLIENT_ID")
    [[ -z "${GOOGLE_CLIENT_SECRET:-}" ]] && MISSING+=("GOOGLE_CLIENT_SECRET")
    [[ -z "${PUBLIC_URL:-}" ]]           && MISSING+=("PUBLIC_URL")
    [[ ${#MISSING[@]} -gt 0 ]] \
      && warn "Missing in .env: ${MISSING[*]}" \
      || { success "All required env vars set"; info "Run: ./setup.sh start"; }
  fi
}

# ─────────────────────────────────────────────────────────────────────────────
# LOCAL START / STOP / STATUS / LOGS
# ─────────────────────────────────────────────────────────────────────────────
cmd_start() {
  load_env; mkdir -p "$PID_DIR" "$LOG_DIR"
  MISSING=()
  [[ -z "${GOOGLE_CLIENT_ID:-}" ]]     && MISSING+=("GOOGLE_CLIENT_ID")
  [[ -z "${GOOGLE_CLIENT_SECRET:-}" ]] && MISSING+=("GOOGLE_CLIENT_SECRET")
  [[ -z "${PUBLIC_URL:-}" ]]           && MISSING+=("PUBLIC_URL")
  [[ ${#MISSING[@]} -gt 0 ]] && { error "Missing in .env: ${MISSING[*]}"; exit 1; }

  HOST="${HOST:-127.0.0.1}"; PORT="${PORT:-7433}"; CF_TUNNEL="${CF_TUNNEL_NAME:-}"
  header "Starting Reel Editor"; cmd_stop_silent

  info "Server on $HOST:$PORT..."
  cd "$SCRIPT_DIR"
  nohup uv run reelforge ui --host "$HOST" --port "$PORT" --no-open \
    > "$LOG_DIR/server.log" 2>&1 &
  echo $! > "$PID_DIR/server.pid"
  success "Server started (PID $(cat "$PID_DIR/server.pid"))"

  for i in {1..15}; do
    curl -sf "http://$HOST:$PORT/auth/login" -o /dev/null 2>/dev/null && break
    sleep 1
  done

  if command -v cloudflared &>/dev/null && [[ -n "$CF_TUNNEL" ]]; then
    info "Tunnel '$CF_TUNNEL'..."
    nohup cloudflared tunnel run "$CF_TUNNEL" > "$LOG_DIR/cloudflared.log" 2>&1 &
    echo $! > "$PID_DIR/cloudflared.pid"; sleep 2
    success "Tunnel started (PID $(cat "$PID_DIR/cloudflared.pid"))"
  fi

  echo ""
  echo -e "  ${BOLD}Local:${RESET}  http://$HOST:$PORT"
  [[ -n "${PUBLIC_URL:-}" ]] && echo -e "  ${BOLD}Public:${RESET} $PUBLIC_URL"
  echo -e "  Logs: tail -f $LOG_DIR/server.log  |  Stop: ./setup.sh stop"
  echo ""
}

cmd_stop_silent() {
  for name in server cloudflared; do
    f="$PID_DIR/$name.pid"
    [[ -f "$f" ]] && { kill "$(cat "$f")" 2>/dev/null || true; rm -f "$f"; }
  done
  lsof -ti :"${PORT:-7433}" 2>/dev/null | xargs kill -9 2>/dev/null || true
}

cmd_stop() {
  header "Stopping Reel Editor"; load_env
  for name in server cloudflared; do
    f="$PID_DIR/$name.pid"
    if [[ -f "$f" ]]; then
      PID=$(cat "$f")
      kill "$PID" 2>/dev/null && success "Stopped $name (PID $PID)" || warn "$name not running (stale PID $PID)"
      rm -f "$f"
    else
      warn "$name PID file not found"
    fi
  done
  lsof -ti :"${PORT:-7433}" 2>/dev/null | xargs kill -9 2>/dev/null || true
}

cmd_status() {
  load_env; HOST="${HOST:-127.0.0.1}"; PORT="${PORT:-7433}"
  header "Reel Editor Status"

  if systemctl is-active --quiet "$SERVICE_NAME" 2>/dev/null; then
    success "Server running (systemd: $SERVICE_NAME)"
  elif [[ -f "$PID_DIR/server.pid" ]] && kill -0 "$(cat "$PID_DIR/server.pid")" 2>/dev/null; then
    success "Server running (PID $(cat "$PID_DIR/server.pid"))"
  else
    error "Server not running"
  fi

  [[ -f "$PID_DIR/cloudflared.pid" ]] && kill -0 "$(cat "$PID_DIR/cloudflared.pid")" 2>/dev/null \
    && success "Cloudflare tunnel running" || warn "Cloudflare tunnel not running"

  curl -sf "http://$HOST:$PORT/auth/login" -o /dev/null 2>/dev/null \
    && success "Responding at http://$HOST:$PORT" || error "Not responding on port $PORT"
}

cmd_logs() {
  TARGET="${1:-server}"
  if systemctl is-active --quiet "$SERVICE_NAME" 2>/dev/null && [[ "$TARGET" == "server" ]]; then
    journalctl -u "$SERVICE_NAME" -f
  elif [[ -f "$LOG_DIR/$TARGET.log" ]]; then
    tail -f "$LOG_DIR/$TARGET.log"
  else
    error "No log for '$TARGET'"; info "Try: journalctl -u $SERVICE_NAME -f"
  fi
}

# ─────────────────────────────────────────────────────────────────────────────
# DISPATCH
# ─────────────────────────────────────────────────────────────────────────────
CMD="${1:-setup}"; shift || true

case "$CMD" in
  setup)         cmd_setup ;;
  deploy|update) cmd_deploy "$@" ;;
  start)         cmd_start ;;
  stop)          cmd_stop ;;
  restart)       cmd_stop; sleep 1; cmd_start ;;
  status)        cmd_status ;;
  logs)          cmd_logs "${1:-server}" ;;
  *)
    echo "Usage: $0 {setup|deploy|update|start|stop|restart|status|logs}"
    echo ""
    echo "  setup             First-time local setup (macOS)"
    echo "  deploy            Full server deploy or update (Ubuntu, run with sudo)"
    echo "    --domain        Domain name (default: from PUBLIC_URL in .env)"
    echo "    --email         Let's Encrypt email (default: admin@<domain>)"
    echo "    --app-dir       Install path (default: /opt/reelforge)"
    echo "  update            Alias for deploy (code update + restart)"
    echo "  start             Start locally (macOS)"
    echo "  stop              Stop locally"
    echo "  restart           Restart locally"
    echo "  status            Show status"
    echo "  logs [server|cf]  Tail logs"
    exit 1 ;;
esac
