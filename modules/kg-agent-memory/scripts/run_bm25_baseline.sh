#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_bm25_baseline.sh — FULL BM25 baseline over one domain, all 6 question types.
#
# BM25 here is only the RETRIEVER (CPU, instant). The time cost is the two LLM
# calls per question (agent answers + judge grades) on serv-3306 — unavoidable,
# it's how GroupMemBench scores any system. Sequential on purpose so the number
# is a faithful baseline (vLLM concurrency comes later for speed).
#
# One results file + one accuracy line per question type; everything tee'd to
# logs/bm25_baseline.log (rule 3), Python unbuffered so the log updates live.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# Interpreter: honour an inherited $PY so the CLUSTER job can point us at its own
# venv (/fscratch/abuali/venvs/groupmembench). Symlinking .venv/bin/python at the
# cluster venv does NOT work — Python looks for pyvenv.cfg next to the symlink,
# finds none, and falls back to the BASE interpreter with no site-packages
# (that's the "No module named 'openai'" that silently N/A'd the 2026-07-22 run).
PY="${PY:-${ROOT_DIR}/.venv/bin/python}"
[[ -x "${PY}" ]] || { echo "❌ no python at ${PY} — run scripts/setup_env.sh" >&2; exit 1; }

DOMAIN=${DOMAIN:-Finance}
CONVERSATION_JSON="${ROOT_DIR}/data/final/${DOMAIN}/synthetic_domain_channels_rolevariants_${DOMAIN}.json"
AGENT_MODEL=${AGENT_MODEL:-"qwen2.5:32b"}
JUDGE_MODEL=${JUDGE_MODEL:-"qwen2.5:32b"}
RETRIEVE_TOP_K=${RETRIEVE_TOP_K:-10}

mkdir -p logs results "results/bm25_${DOMAIN}"
LOG="${ROOT_DIR}/logs/bm25_baseline.log"

# Six question types shipped per domain.
TYPES=(multi_hop knowledge_update temporal user_implicit term_ambiguity abstention)

{
  echo "🚀 [bm25-baseline] domain=${DOMAIN}  agent=${AGENT_MODEL}  judge=${JUDGE_MODEL}  (needs DFKI VPN)"
  echo "📚 [bm25-baseline] corpus=${CONVERSATION_JSON##*/}"
  echo "🗂️  [bm25-baseline] types: ${TYPES[*]}"
  echo

  SUMMARY=()
  for t in "${TYPES[@]}"; do
    QJSON="${ROOT_DIR}/questions/${DOMAIN}/${t}.jsonl"
    OUT="${ROOT_DIR}/results/bm25_${DOMAIN}/${t}.jsonl"
    if [[ ! -f "${QJSON}" ]]; then
      echo "⚠️  [bm25-baseline] missing ${QJSON} — skipping"; continue
    fi
    echo "▶️  [bm25-baseline] type=${t}  ($(wc -l < "${QJSON}") questions)"
    # LIMIT=0 → run ALL questions of this type. Reuse the baseline entrypoint.
    ACC_LINE=$(LIMIT=0 \
      CONVERSATION_JSON="${CONVERSATION_JSON}" \
      QUESTIONS_JSONL="${QJSON}" \
      OUTPUT_JSONL="${OUT}" \
      AGENT_MODEL="${AGENT_MODEL}" JUDGE_MODEL="${JUDGE_MODEL}" \
      RETRIEVE_TOP_K="${RETRIEVE_TOP_K}" \
      bash baselines/graphiti/run_eval.sh 2>&1 | tee /dev/stderr | grep -E "^Accuracy:" || true)
    echo "✅ [bm25-baseline] ${t} → ${ACC_LINE:-<no accuracy line>}"
    SUMMARY+=("${t}: ${ACC_LINE:-N/A}")
    echo
  done

  echo "════════════════════════════════════════════════"
  echo "📊 BM25 baseline summary — ${DOMAIN}"
  for s in "${SUMMARY[@]}"; do echo "   • ${s}"; done
  echo "════════════════════════════════════════════════"
} 2>&1 | tee "${LOG}"
