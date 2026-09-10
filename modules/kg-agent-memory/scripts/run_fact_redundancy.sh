#!/usr/bin/env bash
# =============================================================================
#  run_fact_redundancy.sh — how many DISTINCT ideas are in a set of facts?
#
#  Hop 2's planner drops most of the facts it is given. This answers whether
#  that is the planner being too selective or the extraction being redundant.
#  The distinction matters beyond the board: a retrieval benchmark over a fact
#  set that repeats itself is measuring something other than what it claims.
#
#  USAGE
#      bash scripts/run_fact_redundancy.sh --plan ../../ui/output/<run>/plan.json
#      bash scripts/run_fact_redundancy.sh --group-id gmb_finance_full --windows 2
# =============================================================================
set -euo pipefail

MODULE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${MODULE_ROOT}/logs"
LOG_FILE="${LOG_DIR}/fact_redundancy.log"
VENV_PY="${MODULE_ROOT}/.venv/bin/python"

mkdir -p "${LOG_DIR}"
{
  echo "═══════════════════════════════════════════════════════════════════"
  echo "♻️  FACT REDUNDANCY — $(date '+%Y-%m-%d %H:%M:%S')"
  echo "   args: $*"
  echo "═══════════════════════════════════════════════════════════════════"
  [[ -x "${VENV_PY}" ]] || { echo "❌ venv missing at ${VENV_PY}"; exit 1; }
  PYTHONUNBUFFERED=1 "${VENV_PY}" -u "${MODULE_ROOT}/analysis/fact_redundancy.py" "$@"
  echo "🧹 done"
} 2>&1 | tee -a "${LOG_FILE}"
echo
echo "📝 log: ${LOG_FILE}"
