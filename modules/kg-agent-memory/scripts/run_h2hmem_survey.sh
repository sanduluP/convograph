#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_h2hmem_survey.sh — measure the H2HMem multi-party subset before we commit
# to it as a second evaluation corpus.
#
# Answers the four questions that actually decide it: how big the corpus is (=
# would one GPU reservation do, or are we sharding again), how many questions
# target superseded/evolving facts (our differentiator), how many need an image,
# and how many span multiple sessions.
#
# Reads the public HuggingFace dataset over HTTPS. No clone, no GPU, no key.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/h2hmem_survey.log"

echo "📝 logging to ${LOG_FILE}"

{
  echo ""
  echo "════════════════════════════════════════════════════════════════"
  echo "🔎 H2HMem multi-party survey   $(date '+%Y-%m-%d %H:%M:%S')"
  echo "════════════════════════════════════════════════════════════════"

  # Rule 1: this repo's OWN venv, never another project's.
  PY="${REPO_ROOT}/.venv/bin/python"
  [ -x "${PY}" ] || { echo "❌ no venv at ${REPO_ROOT}/.venv"; exit 1; }

  echo "⚙️  surveying …"
  PYTHONUNBUFFERED=1 "${PY}" -u analysis/h2hmem_survey.py
  echo "✅ survey complete"
} 2>&1 | tee -a "${LOG_FILE}"

exit "${PIPESTATUS[0]}"
