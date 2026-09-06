#!/usr/bin/env bash
# =============================================================================
#  serve_flux.sh — start the warm FLUX.1-schnell server ON THE GPU HOST.
#
#  Keeps the ~31 GB pipeline resident so a board costs ~10 s instead of ~70 s
#  (batched) or ~6 min (the original per-fact calls). See serve_flux.py.
#
#  ON THE GPU HOST (unicorn):
#      bash scripts/serve_flux.sh              # foreground, Ctrl-C to stop
#      bash scripts/serve_flux.sh --daemon     # background + PID file
#      bash scripts/serve_flux.sh --stop
#
#  FROM THE LAPTOP, once it is up:
#      ssh -N -L 8500:localhost:8500 unicorn &
#      export FLUX_SERVER_URL=http://localhost:8500
#      # ui/orchestrator.py now uses it instead of shelling out per board.
#
#  The server binds to LOOPBACK only and is reached through that tunnel — this
#  box is shared with other users, and an unauthenticated GPU endpoint has no
#  business on its network interface.
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs"
# Where the venv, weights and code live. See that file — the short version is
# DFKI's rule: code in $HOME, virtualenvs and model weights on scratch.
source "$(dirname "${BASH_SOURCE[0]}")/lib/env_paths.sh"
PORT="${FLUX_PORT:-8500}"
PID_FILE="${LOG_DIR}/serve_flux_${PORT}.pid"   # port-scoped, like serve_vllm.sh

mkdir -p "$LOG_DIR"

if [[ "${1:-}" == "--stop" ]]; then
  if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    kill "$(cat "$PID_FILE")" && rm -f "$PID_FILE"
    echo "🛑 stopped FLUX server on port ${PORT}"
  else
    echo "ℹ️  no running server for port ${PORT}"
  fi
  exit 0
fi

[[ -x "$PY" ]] || { echo "❌ no venv — run scripts/setup_remote_env.sh first"; exit 1; }

if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
  echo "ℹ️  already serving on port ${PORT}:"
  curl -s "http://localhost:${PORT}/health"; echo
  exit 0
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/serve_flux_${PORT}_${STAMP}.log"

echo "🚀 FLUX server → http://127.0.0.1:${PORT}"
echo "   HF_HOME : ${HF_HOME}"
echo "   log     : ${LOG_FILE}"

if [[ "${1:-}" == "--daemon" ]]; then
  # setsid so it survives the ssh session that launched it closing.
  FLUX_PORT="$PORT" PYTHONUNBUFFERED=1 \
    setsid nohup "$PY" -u "${REPO_ROOT}/serve_flux.py" > "$LOG_FILE" 2>&1 &
  echo $! > "$PID_FILE"
  echo "   pid     : $(cat "$PID_FILE")"

  # The model takes ~40-90 s to load; poll rather than guess.
  echo -n "   waiting for warm"
  for _ in $(seq 1 120); do
    if curl -sf "http://localhost:${PORT}/health" 2>/dev/null | grep -q '"status":"warm"'; then
      echo; echo "✅ warm and serving"; curl -s "http://localhost:${PORT}/health"; echo; exit 0
    fi
    # If the process died, stop waiting and show why.
    if ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
      echo; echo "❌ server exited — tail of ${LOG_FILE}:"; tail -20 "$LOG_FILE"; exit 1
    fi
    echo -n "."; sleep 3
  done
  echo; echo "⚠️  still not warm after 6 min — check ${LOG_FILE}"
  exit 1
fi

FLUX_PORT="$PORT" PYTHONUNBUFFERED=1 exec "$PY" -u "${REPO_ROOT}/serve_flux.py" 2>&1 | tee "$LOG_FILE"
