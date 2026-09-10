#!/usr/bin/env bash
# =============================================================================
#  run_probe_saia_concurrency.sh — how many PARALLEL calls will SAIA accept?
#
#  Graphiti fans its extraction calls out per episode, so the limit that bites
#  during an end-to-end run is requests-in-flight, not tokens-per-day. This
#  finds that wall with a couple of dozen four-token replies and stops at the
#  first rejection — the number it prints is what goes in the rate-limit
#  increase email, instead of asking SAIA for an unspecified "more".
#
#  USAGE
#      bash scripts/run_probe_saia_concurrency.sh              # ramp to 12
#      bash scripts/run_probe_saia_concurrency.sh --max 4      # stay small
# =============================================================================
set -euo pipefail

MODULE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${MODULE_ROOT}/logs"
LOG_FILE="${LOG_DIR}/probe_saia_concurrency.log"
VENV_PY="${MODULE_ROOT}/.venv/bin/python"

mkdir -p "${LOG_DIR}"

{
  echo "═══════════════════════════════════════════════════════════════════"
  echo "📡 SAIA CONCURRENCY PROBE — $(date '+%Y-%m-%d %H:%M:%S')"
  echo "═══════════════════════════════════════════════════════════════════"
  [[ -x "${VENV_PY}" ]] || { echo "❌ venv missing at ${VENV_PY}"; exit 1; }
  PYTHONUNBUFFERED=1 "${VENV_PY}" -u \
      "${MODULE_ROOT}/analysis/probe_saia_concurrency.py" "$@"
  echo "🧹 done — $(date '+%H:%M:%S')"
} 2>&1 | tee -a "${LOG_FILE}"

echo
echo "📝 log: ${LOG_FILE}"
