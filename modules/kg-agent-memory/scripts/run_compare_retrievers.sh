#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_compare_retrievers.sh — paired comparison of two retriever runs.
#
# A raw percentage gap on n=32 is not a result. This wraps the McNemar exact test
# (the right test for two systems scored on the SAME questions) plus Wilson CIs,
# and prints the questions where the two actually disagree — which is what you
# read next, rather than the average.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
LOG_DIR="${REPO_ROOT}/logs"; mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/compare_retrievers.log"
echo "📝 logging to ${LOG_FILE}"
{
  echo ""
  echo "════════════════════════════════════════════════════════════════"
  echo "⚖️  paired retriever comparison   $(date '+%Y-%m-%d %H:%M:%S')"
  echo "════════════════════════════════════════════════════════════════"
  PY="${REPO_ROOT}/.venv/bin/python"
  [ -x "${PY}" ] || { echo "❌ no venv at ${REPO_ROOT}/.venv"; exit 1; }
  PYTHONUNBUFFERED=1 "${PY}" -u analysis/compare_retrievers.py "$@"
  echo "✅ comparison complete"
} 2>&1 | tee -a "${LOG_FILE}"
exit "${PIPESTATUS[0]}"
