#!/usr/bin/env bash
# =============================================================================
#  run_score_digest.sh — is the digest actually about this meeting?
#
#  Scores a digest against a phase's GROUND TRUTH — the labels GroupMemBench
#  ships, not ones we wrote: the phase's `topic`, and its one
#  decision_type=="changed" pair with original_decision -> changed_to spelled
#  out. That last one is exactly what Q1's revision spine claims to recover, so
#  "the digest board looks better" becomes "the ground-truth revision is at
#  rank N of 25".
#
#  Matching is lexical token-set F1: no GPU, no API key, no drift between runs.
#  A paraphrase sharing no vocabulary scores 0, so every number is a FLOOR.
#
#  USAGE
#      bash scripts/run_score_digest.sh \
#        --digest ../../ui/output/<run>/digest/digest.json \
#        --ground-truth tmp/phase_prod_deploy.ground_truth.json
# =============================================================================
set -euo pipefail

MODULE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${MODULE_ROOT}/logs"
LOG_FILE="${LOG_DIR}/score_digest.log"
VENV_PY="${MODULE_ROOT}/.venv/bin/python"

mkdir -p "${LOG_DIR}"
{
  echo "═══════════════════════════════════════════════════════════════════"
  echo "🎯 DIGEST SCORE — $(date '+%Y-%m-%d %H:%M:%S')"
  echo "   args: $*"
  echo "═══════════════════════════════════════════════════════════════════"
  [[ -x "${VENV_PY}" ]] || { echo "❌ venv missing at ${VENV_PY}"; exit 1; }
  PYTHONUNBUFFERED=1 "${VENV_PY}" -u "${MODULE_ROOT}/analysis/score_digest.py" "$@"
  echo "🧹 done"
} 2>&1 | tee -a "${LOG_FILE}"
echo
echo "📝 log: ${LOG_FILE}"
