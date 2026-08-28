#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_diagnose_retrieval.sh — split a retriever's losses into RETRIEVAL failures
# (the evidence never reached the agent) and ANSWERING failures (it did, and the
# agent still got it wrong). Those two have opposite fixes, so an aggregate
# accuracy number cannot tell you what to work on next.
#
# Optional arg: coverage threshold (default 0.5).
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
LOG_DIR="${REPO_ROOT}/logs"; mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/diagnose_retrieval.log"
echo "📝 logging to ${LOG_FILE}"
{
  echo ""
  echo "════════════════════════════════════════════════════════════════"
  echo "🔬 retrieval diagnosis   $(date '+%Y-%m-%d %H:%M:%S')"
  echo "════════════════════════════════════════════════════════════════"
  PY="${REPO_ROOT}/.venv/bin/python"
  [ -x "${PY}" ] || { echo "❌ no venv at ${REPO_ROOT}/.venv"; exit 1; }
  PYTHONUNBUFFERED=1 "${PY}" -u analysis/diagnose_retrieval.py "$@"
  echo "✅ diagnosis complete"
} 2>&1 | tee -a "${LOG_FILE}"
exit "${PIPESTATUS[0]}"
