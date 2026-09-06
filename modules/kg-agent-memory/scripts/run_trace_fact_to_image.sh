#!/usr/bin/env bash
# =============================================================================
#  run_trace_fact_to_image.sh — follow ONE real fact from the KG to a PNG,
#  printing the exact value at every hop. -> logs/trace_fact_to_image.log
#
#  This is a DOCUMENTATION tool: it answers "what does the data actually look
#  like between module 2 and module 3", with real strings instead of a diagram.
#
#  Needs:
#    - .env with NEO4J_* (git-ignored, next to this module)
#    - ollama running locally (the caption model)
#    - a tunnel to the FLUX server on unicorn:
#          ssh -N -L 8500:localhost:8500 unicorn &
#
#  Usage:
#    bash scripts/run_trace_fact_to_image.sh ui_aura_smoke
#    FACT_INDEX=2 bash scripts/run_trace_fact_to_image.sh ui_aura_smoke
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

GROUP_ID="${1:-}"
[[ -n "$GROUP_ID" ]] || { echo "❌ usage: $0 <group_id>"; exit 1; }

LOG_DIR="${HERE}/logs";  mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/trace_fact_to_image.log"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-${HERE}/traces/${GROUP_ID}_${STAMP}}"

PY="${HERE}/.venv/bin/python"
[[ -x "$PY" ]] || { echo "❌ no venv at ${PY}"; exit 1; }

FLUX_SERVER_URL="${FLUX_SERVER_URL:-http://localhost:8500}"

{
  echo "📝 tracing one fact: KG → caption → FLUX"
  echo "   group   : ${GROUP_ID}"
  echo "   out     : ${OUT_DIR}"
  echo "   flux    : ${FLUX_SERVER_URL}"
  echo "   log     : ${LOG}"
  echo ""

  # Fail early and clearly if the tunnel is not open — otherwise the failure
  # surfaces two minutes later as an opaque connection error from inside python.
  if ! curl -sf --max-time 5 "${FLUX_SERVER_URL}/health" >/dev/null 2>&1; then
    echo "❌ no FLUX server at ${FLUX_SERVER_URL}"
    echo "   open the tunnel first:  ssh -N -L 8500:localhost:8500 unicorn &"
    exit 1
  fi
  echo "✅ FLUX server reachable"
  echo ""

  PYTHONUNBUFFERED=1 "$PY" -u "${HERE}/analysis/trace_fact_to_image.py" \
    --group-id "${GROUP_ID}" \
    --out-dir "${OUT_DIR}" \
    --flux-server "${FLUX_SERVER_URL}" \
    ${FACT_INDEX:+--fact-index "${FACT_INDEX}"}

  echo ""
  echo "✅ done"
} 2>&1 | tee "$LOG"
