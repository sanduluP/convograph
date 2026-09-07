#!/usr/bin/env bash
# =============================================================================
#  run_content_map.sh — window of a temporal KG  ->  a CONTENT MAP on Excalidraw.
#
#  The whole redesigned module 3, end to end, with no browser in the way:
#
#      hop 1  cypher: a contiguous run of episodes + their facts   (module 2)
#      hop 2  an LLM plans the WHOLE board from that window        (board_plan)
#      hop 3  FLUX draws one wordless pictogram per anchor         (unicorn H100)
#      hop 4  anchors -> nodes, links -> labelled arrows           (render_board)
#
#  Every run lands in its OWN folder, ui/output/<provider>-<model>_w<N>_<stamp>/,
#  holding plan.json, board.excalidraw, preview.png and images/. Ablations
#  therefore never collide, and a result stays attributable to the model that
#  produced it — which flat output did not, so yesterday's three-model run left
#  nothing behind that could be told apart.
#
#  USAGE
#      bash scripts/run_content_map.sh --group-id gmb_finance_full
#      bash scripts/run_content_map.sh --group-id gmb_finance_full --windows 3
#      bash scripts/run_content_map.sh --group-id gmb_finance_full \
#           --provider ollama --model qwen3:4b-instruct        # the ablation
#
#  PREREQUISITES  (both are on unicorn, behind ONE tunnel from this laptop)
#      ssh -f -N -o ServerAliveInterval=20 \
#          -L 11435:localhost:11435 -L 8500:localhost:8500 unicorn
#  and, for the default SAIA planner, SAIA_API_KEY in a git-ignored .env.
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs"
LOG_FILE="${LOG_DIR}/content_map.log"
VENV_PY="${REPO_ROOT}/ui/.venv/bin/python"

mkdir -p "${LOG_DIR}"

# Everything below is tee'd, so the log holds the banner as well as the run.
{
  echo "═══════════════════════════════════════════════════════════════════"
  echo "🧭 CONTENT MAP — $(date '+%Y-%m-%d %H:%M:%S')"
  echo "   args: $*"
  echo "═══════════════════════════════════════════════════════════════════"

  if [[ ! -x "${VENV_PY}" ]]; then
    echo "❌ ui venv missing at ${VENV_PY}"
    echo "   create it:  python3 -m venv ui/.venv && \\"
    echo "               ui/.venv/bin/pip install -r ui/requirements.txt"
    exit 1
  fi

  # Open the tunnels rather than complaining that they are shut. tunnels.sh is
  # idempotent and probes the SERVICES (a dead tunnel keeps its port bound, so a
  # port check reports healthy while every request hangs), so the normal path is
  # that nobody ever types an -L line.
  bash "${REPO_ROOT}/scripts/tunnels.sh" start || {
    echo "⚠️  continuing anyway — FLUX will fall back to rsync+ssh, and the run"
    echo "    will fail outright if the embedder is needed."
  }

  echo "⚙️  running..."
  # -u so the log updates live and `tail -f` is useful mid-run.
  PYTHONUNBUFFERED=1 "${VENV_PY}" -u "${REPO_ROOT}/ui/content_map_cli.py" "$@"

  echo "🧹 done — $(date '+%H:%M:%S')"
} 2>&1 | tee -a "${LOG_FILE}"

echo
echo "📝 log: ${LOG_FILE}"
