#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_graphiti_slice.sh — SCOPE-MATCHED prototype run (Step 2b, first real number)
#
# Ingests a bounded SLICE of one channel into a Graphiti temporal KG and answers
# GroupMemBench questions from it — then (optionally) runs BM25 over the EXACT
# SAME slice as a control.
#
# WHY the control matters: our headline baseline (52.8%) was BM25 over all 30,000
# messages. Graphiti on a 1,000-message slice cannot be compared to that — most
# questions' evidence isn't in the slice, so it would look bad for a reason that
# has nothing to do with the method. `--max-corpus-messages` truncates the corpus
# for BOTH retrievers, so the only difference left is the RETRIEVAL METHOD.
#
# TOPOLOGY (all local, no cluster job needed beyond the served model):
#   • extraction/QA LLM → cluster vLLM, reached through an SSH tunnel on :8000
#   • embeddings        → serv-3306 bge-m3 (needs DFKI VPN)
#   • graph store       → local Neo4j docker (so you can eyeball it afterwards)
#
# Prereqs:
#   docker start neo4j-graphiti
#   ssh -N -L 8000:<node>:8000 pegasus      # node printed by the serve job log
#
# Usage:
#   bash scripts/run_graphiti_slice.sh              # graphiti run
#   RETRIEVER=bm25 bash scripts/run_graphiti_slice.sh   # the scope-matched control
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PY="${PY:-${ROOT_DIR}/.venv/bin/python}"
[[ -x "${PY}" ]] || { echo "❌ no python at ${PY} — run scripts/setup_env.sh" >&2; exit 1; }

# --- knobs -------------------------------------------------------------------
DOMAIN=${DOMAIN:-"Finance"}
QTYPE=${QTYPE:-"knowledge_update"}
CORPUS=${CORPUS:-1000}          # scope-matched corpus size (BOTH retrievers)
WINDOW=${WINDOW:-10}            # messages per episode
RETRIEVER=${RETRIEVER:-graphiti}
GROUP_ID=${GROUP_ID:-"gmb_slice1k"}
LIMIT=${LIMIT:-0}               # 0 = all questions of this type
STRICT_PROMPT=${STRICT_PROMPT:-1}
RETRIEVE_TOP_K=${RETRIEVE_TOP_K:-10}

# Agent + judge run on the SAME model as the 52.8% baseline, reached through the
# tunnel, so the QA/judging half is held constant and only retrieval varies.
MODEL_ID=${MODEL_ID:-"Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"}

CONVERSATION_JSON="${ROOT_DIR}/data/final/${DOMAIN}/synthetic_domain_channels_rolevariants_${DOMAIN}.json"
QUESTIONS_JSONL="${ROOT_DIR}/questions/${DOMAIN}/${QTYPE}.jsonl"
OUTPUT_JSONL="${ROOT_DIR}/results/graphiti_slice/${RETRIEVER}_${QTYPE}_${CORPUS}.jsonl"

mkdir -p logs results "results/graphiti_slice"
LOG="${ROOT_DIR}/logs/graphiti_slice_${RETRIEVER}.log"

# --- preflight: fail EARLY and loudly, not 20 minutes into an ingest ----------
echo "🔎 [preflight] tunnel → cluster vLLM on localhost:8000 …"
curl -sf -m 15 http://localhost:8000/v1/models >/dev/null \
  || { echo "❌ no vLLM on localhost:8000 — is the SSH tunnel up?" >&2; exit 1; }
echo "   ✅ vLLM reachable"

if [[ "${RETRIEVER}" == "graphiti" ]]; then
  echo "🔎 [preflight] local Neo4j …"
  docker ps --filter name=neo4j-graphiti --format '{{.Names}}' | grep -q neo4j-graphiti \
    || { echo "❌ neo4j-graphiti container not running — docker start neo4j-graphiti" >&2; exit 1; }
  echo "   ✅ Neo4j up"
  echo "🔎 [preflight] embeddings on serv-3306 (needs DFKI VPN) …"
  curl -sf -m 15 "${GRAPHITI_EMBED_BASE_URL:-http://serv-3306.kl.dfki.de:8000/v1}/models" >/dev/null \
    || { echo "❌ serv-3306 unreachable — is the DFKI VPN on?" >&2; exit 1; }
  echo "   ✅ embeddings reachable"
fi

echo "🚀 [slice] retriever=${RETRIEVER}  corpus=${CORPUS}  window=${WINDOW}  qtype=${QTYPE}"
echo "🤖 [slice] agent=judge=${MODEL_ID}  (via tunnel)  top_k=${RETRIEVE_TOP_K}"
echo "🧠 [slice] neo4j group='${GROUP_ID}' (wiped + rebuilt)"

ARGS=(
  --conversation-json "${CONVERSATION_JSON}"
  --questions-jsonl "${QUESTIONS_JSONL}"
  --env-file ".env"
  --agent-model "${MODEL_ID}"
  --judge-model "${MODEL_ID}"
  --agent-prompt "prompts/hipporag_agent_system.txt"
  --judge-prompt "prompts/hipporag_judge_system.txt"
  --retrieve-top-k "${RETRIEVE_TOP_K}"
  --output-jsonl "${OUTPUT_JSONL}"
  --llm-provider "openai"
  --retriever "${RETRIEVER}"
  --max-corpus-messages "${CORPUS}"
  --window "${WINDOW}"
  --group-id "${GROUP_ID}"
)
[[ "${LIMIT}" != "0" ]] && ARGS+=(--limit "${LIMIT}")
[[ "${STRICT_PROMPT}" == "1" && "${RETRIEVER}" == "graphiti" ]] && ARGS+=(--strict-prompt)

# SEMAPHORE_LIMIT above = Graphiti's internal concurrency (graphiti_core.helpers).
# It defaults to 20; we had pinned it to 5 back when extraction ran on serv-3306,
# whose Ollama fell over under load. vLLM is built for concurrency (that is the
# entire point of the MoE), so 5 was throttling us ~4x for no reason.
#
# Point agent/judge/extraction at the tunnel for THIS process only — the .env on
# disk stays untouched (it still describes the serv-3306 setup).
PYTHONUNBUFFERED=1 PYTHONPATH="${ROOT_DIR}" \
OPENAI_BASE_URL="http://localhost:8000/v1" \
GRAPHITI_LLM_BASE_URL="http://localhost:8000/v1" \
GRAPHITI_LLM_MODEL="${MODEL_ID}" \
SEMAPHORE_LIMIT="${SEMAPHORE_LIMIT:-20}" \
  "${PY}" baselines/graphiti/eval_benchmark.py "${ARGS[@]}" "$@" 2>&1 | tee "${LOG}"
