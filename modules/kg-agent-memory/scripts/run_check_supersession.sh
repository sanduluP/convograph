#!/usr/bin/env bash
# =============================================================================
#  run_check_supersession.sh — validate the "what replaced this fact?" join.
#
#  Graphiti stores no pointer from a superseded fact to its replacement, only
#  the timestamps. The successor is recoverable because invalidation copies the
#  new fact's valid_at into the old fact's invalid_at (edge_operations.py:569),
#  but the join is only useful if it lands on ONE candidate. This measures that
#  before the query goes into eyeball_kg.cypher.
#
#  USAGE
#      bash scripts/run_check_supersession.sh --group-id gmb_finance_full
#      bash scripts/run_check_supersession.sh --limit 500
# =============================================================================
set -euo pipefail

MODULE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${MODULE_ROOT}/logs"
LOG_FILE="${LOG_DIR}/check_supersession.log"
VENV_PY="${MODULE_ROOT}/.venv/bin/python"

mkdir -p "${LOG_DIR}"
{
  echo "═══════════════════════════════════════════════════════════════════"
  echo "🔗 SUPERSESSION JOIN CHECK — $(date '+%Y-%m-%d %H:%M:%S')"
  echo "   args: $*"
  echo "═══════════════════════════════════════════════════════════════════"
  [[ -x "${VENV_PY}" ]] || { echo "❌ venv missing at ${VENV_PY}"; exit 1; }
  PYTHONUNBUFFERED=1 "${VENV_PY}" -u "${MODULE_ROOT}/analysis/check_supersession.py" "$@"
  echo "🧹 done"
} 2>&1 | tee -a "${LOG_FILE}"
echo
echo "📝 log: ${LOG_FILE}"
