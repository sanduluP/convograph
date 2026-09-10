#!/usr/bin/env bash
# =============================================================================
#  run_rank_entities.sh — does this graph rank CONCEPTS, or does it rank PEOPLE?
#
#  A board digest has to answer "what was this meeting about", which means
#  ranking entities. On gmb_finance_full the naive answer fails outright: the
#  top 12 entities by degree are all participants, because 94% of extracted
#  facts are person-rooted. The speaker-free corpus is the same conversations
#  re-ingested WITHOUT speaker names, so the extractor could not root every fact
#  at a person — this runs the same four rankings on either graph so the two are
#  directly comparable.
#
#  USAGE
#      bash scripts/run_rank_entities.sh --group-id gmb_finance_full
#
#      # the speaker-free merge, served locally by run_open_merged_kg.sh:
#      bash scripts/run_rank_entities.sh --group-id finance_speaker_free \
#          --uri bolt://localhost:7689 --user neo4j --password <pw>
#
#      # one MEETING-sized slice instead of a six-week corpus:
#      bash scripts/run_rank_entities.sh --group-id gmb_finance_full --episode-limit 60
# =============================================================================
set -euo pipefail

MODULE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${MODULE_ROOT}/logs"
LOG_FILE="${LOG_DIR}/rank_entities.log"
VENV_PY="${MODULE_ROOT}/.venv/bin/python"

mkdir -p "${LOG_DIR}"
{
  echo "═══════════════════════════════════════════════════════════════════"
  echo "🏅 ENTITY RANKING — $(date '+%Y-%m-%d %H:%M:%S')"
  echo "   args: $*"
  echo "═══════════════════════════════════════════════════════════════════"
  [[ -x "${VENV_PY}" ]] || { echo "❌ venv missing at ${VENV_PY}"; exit 1; }
  PYTHONUNBUFFERED=1 "${VENV_PY}" -u "${MODULE_ROOT}/analysis/rank_entities.py" "$@"
  echo "🧹 done"
} 2>&1 | tee -a "${LOG_FILE}"
echo
echo "📝 log: ${LOG_FILE}"
