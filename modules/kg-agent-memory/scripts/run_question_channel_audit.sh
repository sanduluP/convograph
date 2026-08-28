#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  run_question_channel_audit.sh
#
#  Answers: "how many of the 32 knowledge_update questions were even ANSWERABLE
#  from the 4,877-message AML slice we ingested?"
#
#  Context: the 2026-08-03 run scored Graphiti 5/32 and BM25 6/32 on a corpus
#  truncated to ONE channel, while the questions still covered all six. This
#  audit measures the resulting ceiling so we know whether that 5-vs-6 gap means
#  anything at all.
#
#  CPU-only, no LLM, seconds to run.
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# Rule 1: this repo's OWN venv, never another repo's. Honour an inherited $PY so
# the cluster job (which exports its own interpreter) is not clobbered — the bug
# that silently voided the 2026-07-22 baseline.
PY="${PY:-${ROOT_DIR}/.venv/bin/python}"
[[ -x "${PY}" ]] || { echo "❌ no python at ${PY} — run scripts/setup_env.sh" >&2; exit 1; }

DOMAIN="${DOMAIN:-Finance}"
QTYPE="${QTYPE:-knowledge_update}"
CHANNEL="${CHANNEL:-AML (Anti-Money Laundering) Project}"

LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/question_channel_audit.log"

# Everything below is tee'd so the run is inspectable after the fact (rule 3).
{
  echo "═══════════════════════════════════════════════════════════════"
  echo "🔍 question → channel audit"
  echo "   domain  : ${DOMAIN}"
  echo "   qtype   : ${QTYPE}"
  echo "   channel : ${CHANNEL}"
  echo "   started : $(date '+%Y-%m-%d %H:%M:%S')"
  echo "═══════════════════════════════════════════════════════════════"

  PYTHONUNBUFFERED=1 "${PY}" -u analysis/question_channel_audit.py \
    --conversation-json "data/final/${DOMAIN}/synthetic_domain_channels_rolevariants_${DOMAIN}.json" \
    --questions-jsonl "questions/${DOMAIN}/${QTYPE}.jsonl" \
    --truncated-channel "${CHANNEL}"

  echo "✅ done  $(date '+%Y-%m-%d %H:%M:%S')"
} 2>&1 | tee "${LOG_FILE}"
