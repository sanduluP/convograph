#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_graphiti_smoke.sh — Step 2b PLUMBING smoke test for the REAL Graphiti
# temporal-KG retriever (baselines/graphiti/graphiti_retriever.py).
#
# GOAL = prove the whole path runs end-to-end WITHOUT the cluster:
#   windowed ingest → Neo4j temporal KG → graphiti.search() → invalid_at-aware
#   ranking → precise message indices → shared run_qa (agent + judge) → score.
#
# It is deliberately TINY (a handful of ingested messages + a few questions) so
# it finishes in minutes on the slow serv-3306 endpoint. Accuracy is NOT the
# point here — the point is "does the pipeline work and return sane indices".
# The real number comes later at scale on the cluster vLLM.
#
# Prereqs: local Neo4j up (`docker start neo4j-graphiti`) + DFKI VPN (serv-3306).
#
# Override any knob via env, e.g.:
#   MAX_INGEST=80 WINDOW=10 LIMIT=5 bash scripts/run_graphiti_smoke.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# Honour an inherited $PY (cluster venv); default to this repo's venv (rule 1).
PY="${PY:-${ROOT_DIR}/.venv/bin/python}"
if [[ ! -x "${PY}" ]]; then
  echo "❌ no python at ${PY} — run: bash scripts/setup_env.sh" >&2
  exit 1
fi

# --- knobs (small defaults for a fast plumbing check) ------------------------
DOMAIN=${DOMAIN:-"Finance"}
CONVERSATION_JSON=${CONVERSATION_JSON:-"${ROOT_DIR}/data/final/${DOMAIN}/synthetic_domain_channels_rolevariants_${DOMAIN}.json"}
QUESTIONS_JSONL=${QUESTIONS_JSONL:-"${ROOT_DIR}/questions/${DOMAIN}/knowledge_update.jsonl"}
ENV_FILE=${ENV_FILE:-".env"}
AGENT_MODEL=${AGENT_MODEL:-"qwen2.5:32b"}
JUDGE_MODEL=${JUDGE_MODEL:-"qwen2.5:32b"}
AGENT_PROMPT=${AGENT_PROMPT:-"prompts/hipporag_agent_system.txt"}
JUDGE_PROMPT=${JUDGE_PROMPT:-"prompts/hipporag_judge_system.txt"}
RETRIEVE_TOP_K=${RETRIEVE_TOP_K:-10}
OUTPUT_JSONL=${OUTPUT_JSONL:-"results/graphiti_smoke.jsonl"}
LLM_PROVIDER=${LLM_PROVIDER:-"openai"}

# Graphiti-specific:
WINDOW=${WINDOW:-10}                 # messages per episode
MAX_INGEST=${MAX_INGEST:-40}         # cap ingested messages (0 = all) — small for smoke
GROUP_ID=${GROUP_ID:-"gmb_smoke"}    # wiped + rebuilt each run
LIMIT=${LIMIT:-3}                    # only a few questions
STRICT_PROMPT=${STRICT_PROMPT:-1}    # install the meeting-aware resolve_edge prompt

mkdir -p logs results
LOG="${ROOT_DIR}/logs/graphiti_smoke.log"

echo "🚀 [graphiti-smoke] domain=${DOMAIN}  window=${WINDOW}  max_ingest=${MAX_INGEST}  limit=${LIMIT}"
echo "🤖 [graphiti-smoke] agent=${AGENT_MODEL}  judge=${JUDGE_MODEL}  (Graphiti extraction on serv-3306; needs DFKI VPN)"
echo "🧠 [graphiti-smoke] Neo4j group='${GROUP_ID}'  (wiped + rebuilt this run)"

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
  --retriever graphiti
  --window "${WINDOW}"
  --group-id "${GROUP_ID}"
  --max-ingest-messages "${MAX_INGEST}"
)
if [[ "${STRICT_PROMPT}" == "1" ]]; then
  ARGS+=(--strict-prompt)
fi
if [[ "${INGEST_ONLY:-0}" == "1" ]]; then
  ARGS+=(--ingest-only)
fi

# Unbuffered + tee'd so the log updates live and every run is inspectable (rule 3).
PYTHONUNBUFFERED=1 PYTHONPATH="${ROOT_DIR}" "${PY}" \
  baselines/graphiti/eval_benchmark.py "${ARGS[@]}" "$@" 2>&1 | tee "${LOG}"
