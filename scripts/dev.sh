#!/usr/bin/env bash
# One command for this lab. Stops every other copy of this app, then serves
# the same database every time: data/devlab/learnloop.db.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DATA_DIR="$ROOT/data/devlab"
DB_FILE="$DATA_DIR/learnloop.db"
API_PORT=8000
APP_PORT=5174

export LEARNLOOP_DATA_DIR="$DATA_DIR"
export LEARNLOOP_DATABASE_URL="sqlite:///$DB_FILE"
export LEARNLOOP_AUTO_SEED=1
export LEARNLOOP_WORKER_ENABLED=1
export LEARNLOOP_CORS_ORIGINS="http://127.0.0.1:${APP_PORT},http://localhost:${APP_PORT}"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy \
  SOCKS_PROXY SOCKS5_PROXY socks_proxy socks5_proxy \
  GIT_HTTP_PROXY GIT_HTTPS_PROXY NO_PROXY no_proxy \
  __CURSOR_SANDBOX_ENV_RESTORE || true

if [[ ! -x backend/.venv/bin/uvicorn ]]; then
  echo "Create the backend environment first: /usr/local/bin/python3.12 -m venv backend/.venv && backend/.venv/bin/pip install -e './backend[dev]'" >&2
  exit 1
fi
if [[ ! -d frontend/node_modules ]]; then
  npm install --prefix frontend
fi
mkdir -p "$DATA_DIR"

is_this_app() {
  local pid=$1 cmd=$2 cwd
  cmd=${cmd:-$(ps -p "$pid" -o command= 2>/dev/null || true)}
  [[ -n "$cmd" ]] || return 1
  cwd=$(lsof -a -p "$pid" -d cwd -Fn 2>/dev/null | sed -n 's/^n//p' | head -1 || true)
  [[ "$cwd" == "$ROOT" || "$cwd" == "$ROOT"/* || "$cmd" == *"$ROOT"* ]] || return 1
  case "$cmd" in
    *app.main:app*|*multiprocessing.spawn\ import\ spawn_main*|*node_modules/.bin/vite*|*node_modules/@esbuild*|*npm\ run\ dev*|*npm\ --prefix\ frontend*|*scripts/dev.sh*)
      return 0
      ;;
  esac
  return 1
}

stop_other_copies() {
  local pid cmd
  local -a victims=()
  while read -r pid; do
    pid="${pid// /}"
    [[ -n "$pid" && "$pid" != "$$" && "$pid" != "$PPID" ]] || continue
    cmd=$(ps -p "$pid" -o command= 2>/dev/null || true)
    if is_this_app "$pid" "$cmd"; then
      victims+=("$pid")
    fi
  done < <(ps -ax -o pid=)
  if ((${#victims[@]} == 0)); then
    echo "No other LearnLoop servers were running."
    return
  fi
  echo "Stopping ${#victims[@]} existing LearnLoop process(es)."
  kill "${victims[@]}" 2>/dev/null || true
  local attempt
  for attempt in 1 2 3 4 5 6 7 8 9 10; do
    local alive=0
    for pid in "${victims[@]}"; do
      if ps -p "$pid" >/dev/null 2>&1; then
        alive=1
        break
      fi
    done
    if ((alive == 0)); then
      return
    fi
    sleep 0.3
  done
  for pid in "${victims[@]}"; do
    if ps -p "$pid" >/dev/null 2>&1; then
      kill -9 "$pid" 2>/dev/null || true
    fi
  done
}

port_owner() {
  lsof -nP -iTCP:"$1" -sTCP:LISTEN -t 2>/dev/null | head -1 || true
}

require_port_free() {
  local port=$1 owner
  owner=$(port_owner "$port")
  [[ -z "$owner" ]] && return
  echo "Port ${port} is still in use by pid ${owner}: $(ps -p "$owner" -o command= 2>/dev/null || true)" >&2
  echo "That process is not this app, so it was left running." >&2
  exit 1
}

stop_other_copies
require_port_free "$API_PORT"
require_port_free "$APP_PORT"

echo "Database  $DB_FILE"
echo "API       http://127.0.0.1:${API_PORT}"
echo "App       http://127.0.0.1:${APP_PORT}"

# Job control would put the API in the background and the terminal would freeze it,
# so the page would load while every data request hung.
backend/.venv/bin/uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port "$API_PORT" --reload --timeout-graceful-shutdown 5 &
API_PID=$!
trap 'kill "$API_PID" 2>/dev/null || true' EXIT INT TERM

ready=0
for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25; do
  if curl -sf -m 1 "http://127.0.0.1:${API_PORT}/api/projects" >/dev/null; then
    ready=1
    break
  fi
  if ! ps -p "$API_PID" >/dev/null 2>&1; then
    echo "The API process exited before it could serve." >&2
    exit 1
  fi
  sleep 0.2
done
if ((ready != 1)); then
  echo "The API did not answer on port ${API_PORT}." >&2
  exit 1
fi
if ! ps eww -p "$API_PID" -o command= | grep -F "LEARNLOOP_DATABASE_URL=sqlite:///$DB_FILE" >/dev/null; then
  echo "The API is not using $DB_FILE." >&2
  exit 1
fi

npm --prefix frontend run dev -- --host 127.0.0.1 --port "$APP_PORT" --strictPort
