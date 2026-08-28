#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_eval.sh — run the Graphiti baseline (step-1 = BM25 stub) on GroupMemBench.
#
# Mirrors baselines/bm25/run_eval.sh so the comparison is apples-to-apples, plus:
#   • defaults to our LOCAL serv-3306 qwen2.5:32b agent+judge (via ../../.env),
#   • defaults to a SMALL --limit for a fast plumbing smoke test,
#   • tee's all output to logs/graphiti_eval.log (rule 3) and runs unbuffered.
#
# Override any knob via env, e.g.:
#   QUESTIONS_JSONL=questions/Finance/temporal.jsonl LIMIT=5 bash baselines/graphiti/run_eval.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

# Use THIS repo's own venv (rule 1), unless a caller (the cluster job) exported
# $PY pointing at the cluster venv. See the note in scripts/run_bm25_baseline.sh:
# a bare symlink to another venv's python loses that venv's site-packages.
PY="${PY:-${ROOT_DIR}/.venv/bin/python}"
if [[ ! -x "${PY}" ]]; then
  echo "❌ no python at ${PY} — run: bash scripts/setup_env.sh" >&2
  exit 1
fi

CONVERSATION_JSON=${CONVERSATION_JSON:-"${ROOT_DIR}/data/final/Finance/synthetic_domain_channels_rolevariants_Finance.json"}
QUESTIONS_JSONL=${QUESTIONS_JSONL:-"${ROOT_DIR}/questions/Finance/knowledge_update.jsonl"}
ENV_FILE=${ENV_FILE:-".env"}
AGENT_MODEL=${AGENT_MODEL:-"qwen2.5:32b"}
JUDGE_MODEL=${JUDGE_MODEL:-"qwen2.5:32b"}
AGENT_PROMPT=${AGENT_PROMPT:-"prompts/hipporag_agent_system.txt"}
JUDGE_PROMPT=${JUDGE_PROMPT:-"prompts/hipporag_judge_system.txt"}
RETRIEVE_TOP_K=${RETRIEVE_TOP_K:-10}
OUTPUT_JSONL=${OUTPUT_JSONL:-"results/graphiti_smoke.jsonl"}
LLM_PROVIDER=${LLM_PROVIDER:-"openai"}   # matches .env → serv-3306 OpenAI-compatible
LIMIT=${LIMIT:-3}                        # smoke test: only 3 questions by default

mkdir -p logs results
LOG="${ROOT_DIR}/logs/graphiti_eval.log"

echo "🚀 [graphiti] conversation=${CONVERSATION_JSON##*/}  questions=${QUESTIONS_JSONL##*/}  limit=${LIMIT}"
echo "🤖 [graphiti] agent=${AGENT_MODEL}  judge=${JUDGE_MODEL}  provider=${LLM_PROVIDER}  (needs DFKI VPN)"

ARGS=(
  --conversation-json "${CONVERSATION_JSON}"
  --questions-jsonl "${QUESTIONS_JSONL}"
  --env-file "${ENV_FILE}"
  --agent-model "${AGENT_MODEL}"
  --judge-model "${JUDGE_MODEL}"
  --agent-prompt "${AGENT_PROMPT}"
  --judge-prompt "${JUDGE_PROMPT}"
  --retrieve-top-k "${RETRIEVE_TOP_K}"
  --output-jsonl "${OUTPUT_JSONL}"
  --llm-provider "${LLM_PROVIDER}"
  --limit "${LIMIT}"
)
if [[ "${INGEST_ONLY:-0}" == "1" ]]; then
  ARGS+=(--ingest-only)
fi

# Unbuffered + tee'd so the log updates live and every run is inspectable.
PYTHONUNBUFFERED=1 PYTHONPATH="${ROOT_DIR}" "${PY}" \
  baselines/graphiti/eval_benchmark.py "${ARGS[@]}" "$@" 2>&1 | tee "${LOG}"
