#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

CONVERSATION_JSON=${CONVERSATION_JSON:-"${ROOT_DIR}/data/final/Finance/synthetic_domain_channels_rolevariants_Finance.json"}
QUESTIONS_JSONL=${QUESTIONS_JSONL:-""}
ENV_FILE=${ENV_FILE:-".env"}
AGENT_MODEL=${AGENT_MODEL:-"gpt-5"}
JUDGE_MODEL=${JUDGE_MODEL:-"gpt-5"}
AGENT_PROMPT=${AGENT_PROMPT:-"prompts/hipporag_agent_system.txt"}
JUDGE_PROMPT=${JUDGE_PROMPT:-"prompts/hipporag_judge_system.txt"}
RETRIEVE_TOP_K=${RETRIEVE_TOP_K:-10}
OUTPUT_JSONL=${OUTPUT_JSONL:-"results/bm25_eval_results.jsonl"}
LLM_PROVIDER=${LLM_PROVIDER:-""}

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
)
if [[ -n "${LLM_PROVIDER}" ]]; then
  ARGS+=(--llm-provider "${LLM_PROVIDER}")
fi
if [[ "${INGEST_ONLY:-0}" == "1" ]]; then
  ARGS+=(--ingest-only)
fi

PYTHONPATH="${ROOT_DIR}" python baselines/bm25/eval_benchmark.py "${ARGS[@]}" "$@"
