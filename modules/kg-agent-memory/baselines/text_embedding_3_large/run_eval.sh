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
OUTPUT_JSONL=${OUTPUT_JSONL:-"results/text_embedding_3_large_eval_results.jsonl"}
LLM_PROVIDER=${LLM_PROVIDER:-""}
EMBEDDING_PROVIDER=${EMBEDDING_PROVIDER:-"azure_openai"}
EMBEDDING_MODEL=${EMBEDDING_MODEL:-"text-embedding-3-large"}
EMBEDDING_DIMS=${EMBEDDING_DIMS:-3072}
EMBEDDING_BATCH_SIZE=${EMBEDDING_BATCH_SIZE:-64}
STORE_DIR=${STORE_DIR:-""}

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
  --embedding-provider "${EMBEDDING_PROVIDER}"
  --embedding-model "${EMBEDDING_MODEL}"
  --embedding-dims "${EMBEDDING_DIMS}"
  --embedding-batch-size "${EMBEDDING_BATCH_SIZE}"
)
if [[ -n "${LLM_PROVIDER}" ]]; then
  ARGS+=(--llm-provider "${LLM_PROVIDER}")
fi
if [[ -n "${STORE_DIR}" ]]; then
  ARGS+=(--store-dir "${STORE_DIR}")
fi
if [[ "${INGEST_ONLY:-0}" == "1" ]]; then
  ARGS+=(--ingest-only)
fi
if [[ "${FORCE_REBUILD:-0}" == "1" && -n "${STORE_DIR}" ]]; then
  rm -rf "${STORE_DIR}"
fi

PYTHONPATH="${ROOT_DIR}" python baselines/text_embedding_3_large/eval_benchmark.py "${ARGS[@]}" "$@"
