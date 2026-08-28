#!/usr/bin/env bash
# =============================================================================
#  run_mine_answers_in_kg.sh — "is the answer even IN the graph?"
#
#  Pairs every GroupMemBench knowledge_update question with its gold answer and
#  the closest facts in the merged Finance KG, then writes an Obsidian note we
#  read BY HAND. No judge LLM involved — that is the point.
#
#  Requires the local merged-KG container to be running:
#      docker start neo4j-gmb-full          # bolt on :7688
#  (or `bash scripts/run_open_merged_kg.sh` if it was never downloaded).
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs"
LOG_FILE="${LOG_DIR}/mine_answers_in_kg.log"
PY="${REPO_ROOT}/.venv/bin/python"          # this repo's OWN venv (house rule 1)

mkdir -p "${LOG_DIR}"

{
  echo "════════════════════════════════════════════════════════════════"
  echo "🔬 Mining the KG for each question's answer   ($(date '+%F %T'))"
  echo "════════════════════════════════════════════════════════════════"

  # Fail early and loudly if the graph is not up — otherwise the script would
  # produce an empty, misleading audit.
  if ! docker ps --format '{{.Names}}' | grep -qx "neo4j-gmb-full"; then
    echo "❌ container 'neo4j-gmb-full' is not running."
    echo "   start it with:  docker start neo4j-gmb-full"
    exit 1
  fi
  echo "✅ merged-KG container is up"

  # PYTHONUNBUFFERED so the log updates live and can be tailed mid-run.
  PYTHONUNBUFFERED=1 "${PY}" -u "${REPO_ROOT}/analysis/mine_answers_in_kg.py"

  echo "🎉 done — open the note in Obsidian and read it row by row"
} 2>&1 | tee -a "${LOG_FILE}"
