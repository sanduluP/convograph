#!/usr/bin/env bash
# =============================================================================
#  run_tkg_digest.sh — distil a whole meeting out of the temporal KG, cypher only.
#
#  Five queries between module 2 and the board planner, so the planner sees what
#  the MEETING was about instead of whichever 2-window slice it was handed:
#
#      Q1 revisions     what the meeting changed its mind about
#      Q2 topics        what it was about
#      Q3 participants  who was in the room  (read from text, not the graph)
#      Q4 decisions     what was settled
#      Q5 open threads  asked for, never confirmed
#
#  RUN IT ON A MEETING, NOT A CORPUS. finance_speaker_free is six weeks of
#  several parallel projects; digesting all of it summarises a corpus. Use
#  --episode-limit to take one meeting-sized slice.
#
#  USAGE
#      # the speaker-free graph, served locally by run_open_merged_kg.sh
#      bash scripts/run_tkg_digest.sh --group-id finance_speaker_free \
#          --uri bolt://localhost:7689 --user neo4j --password graphiti123 \
#          --episode-limit 80 --out ../../ui/output/digest_speaker_free
#
#      # a graph that IS one meeting (no slicing needed)
#      bash scripts/run_tkg_digest.sh --group-id treasury_prod_deploy_speaker_free
# =============================================================================
set -euo pipefail

MODULE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${MODULE_ROOT}/logs"
LOG_FILE="${LOG_DIR}/tkg_digest.log"
VENV_PY="${MODULE_ROOT}/.venv/bin/python"

mkdir -p "${LOG_DIR}"
{
  echo "═══════════════════════════════════════════════════════════════════"
  echo "🧪 TKG DIGEST — $(date '+%Y-%m-%d %H:%M:%S')"
  echo "   args: $*"
  echo "═══════════════════════════════════════════════════════════════════"
  [[ -x "${VENV_PY}" ]] || { echo "❌ venv missing at ${VENV_PY}"; exit 1; }
  PYTHONUNBUFFERED=1 "${VENV_PY}" -u "${MODULE_ROOT}/analysis/tkg_digest.py" "$@"
  echo "🧹 done"
} 2>&1 | tee -a "${LOG_FILE}"
echo
echo "📝 log: ${LOG_FILE}"
