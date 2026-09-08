#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
#  cluster_phase_ingest_job.sh — ingest ONE meeting-sized phase, speaker-free,
#  writing DIRECTLY to AuraDB.
#
#  WHY A SEPARATE JOB FROM cluster_ingest_job.sh
#  ---------------------------------------------
#  That job ingests the CORPUS: it loads the whole domain JSON, windows it
#  globally, and takes --window-range to do a slice. A phase cannot be expressed
#  that way. The loader sorts by (channel, timestamp, msg_node) and phases
#  INTERLEAVE inside a channel — measured for "Production Deployment Readiness":
#  its 401 messages span global indices 25129..27436, and that span holds 1,907
#  messages from other phases, 82.6% contamination. A window range would ingest
#  five phases and call it one.
#
#  So this takes the phase as a TEXT FILE (analysis/extract_phase.py writes it)
#  and runs ui_ingest.py, which windows exactly what it is given.
#
#  WHY IT WRITES TO AURADB AND STARTS NO NEO4J
#  -------------------------------------------
#  The point of this graph is that Rahul and Priyabanta can pick it in the UI.
#  Ingesting into a local store on /fscratch would mean a 3 GB rsync down and a
#  separate upload afterwards; AuraDB is reachable from the cluster, so the
#  ingest lands where the UI already reads. The LLM and embedder still run
#  LOCALLY on the node — that is the whole speed argument for being here.
#
#  Usage (submit from the login node):
#      TEXT_FILE=tmp/phase_prod_deploy.txt GROUP_ID=finance_phase_prod_deploy \
#        bash scripts/srun_submit.sh all phase_ingest_prod_deploy 8 1 96G 4 \
#          scripts/cluster_phase_ingest_job.sh
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
PY="${REPO_ROOT}/.venv/bin/python"

TEXT_FILE="${TEXT_FILE:?set TEXT_FILE to the phase transcript (analysis/extract_phase.py)}"
GROUP_ID="${GROUP_ID:?set GROUP_ID — it is how the UI will name this graph}"
FS_ROOT="${FS_ROOT:-/fscratch/abuali}"
CHAT_PORT="${CHAT_PORT:-8000}"
EMBED_PORT="${EMBED_PORT:-8001}"
CHAT_MODEL_DIR="${CHAT_MODEL_DIR:-${FS_ROOT}/models/Qwen3-30B-A3B-Instruct-2507-FP8}"
EMBED_MODEL_DIR="${EMBED_MODEL_DIR:-${FS_ROOT}/models/bge-m3}"
MODEL_ID="${MODEL_ID:-Qwen/$(basename "${CHAT_MODEL_DIR}")}"
EMBED_MODEL_ID="${EMBED_MODEL_ID:-BAAI/bge-m3}"
CHAT_GPU_FRAC="${CHAT_GPU_FRAC:-0.72}"
EMBED_GPU_FRAC="${EMBED_GPU_FRAC:-0.12}"

# Speaker exclusion is the POINT of this graph, so it defaults ON here — the
# opposite of cluster_ingest_job.sh, where it defaults off. Measured 2026-09-08:
# excluding speakers takes concept->concept facts from 6.6% to 68.9%, and a
# content map is made of exactly those edges.
EXCLUDE_SPEAKERS="${GRAPHITI_EXCLUDE_SPEAKERS:-1}"

echo "🖥️  [job] node=$(hostname)"
echo "📄 [job] text     : ${TEXT_FILE}"
echo "🏷️  [job] group_id : ${GROUP_ID}"
echo "🚫👤 [job] speakers : $([[ "${EXCLUDE_SPEAKERS}" == "1" ]] && echo EXCLUDED || echo included)"

[[ -f "${TEXT_FILE}" ]] || { echo "❌ no such file: ${TEXT_FILE}"; exit 1; }
LINES=$(grep -cve '^[[:space:]]*$' "${TEXT_FILE}")
echo "💬 [job] ${LINES} utterances → ~$(( (LINES + 4) / 5 )) windows"

# ── credentials ──────────────────────────────────────────────────────────────
# .env is git-ignored and carries the AuraDB connection. It is NOT synced by
# sync_to_cluster.sh (secrets never go over that path), so it must already be on
# the cluster — say so plainly rather than failing later inside the driver.
ENV_FILE="${ENV_FILE:-${REPO_ROOT}/.env}"
[[ -f "${ENV_FILE}" ]] || {
  echo "❌ ${ENV_FILE} not found on the cluster."
  echo "   Copy it once:  scp modules/kg-agent-memory/.env pegasus:${REPO_ROOT}/.env"
  exit 1
}
set -a; source "${ENV_FILE}"; set +a
: "${NEO4J_URI:?NEO4J_URI missing from ${ENV_FILE}}"

# ── fail fast on an unreachable database ─────────────────────────────────────
# Compute nodes do not always have outbound internet. Discovering that AFTER a
# ~30 GB model load wastes the whole GPU allocation, so check first: it costs a
# second and turns a 40-minute mystery into an immediate, legible failure.
echo "🔌 [job] checking AuraDB reachability…"
"${PY}" - <<PYEOF || { echo "❌ cannot reach ${NEO4J_URI} from $(hostname)."; \
  echo "   The compute node has no route to AuraDB. Ingest into a local store"; \
  echo "   with cluster_ingest_job.sh and upload afterwards instead."; exit 1; }
import os
from neo4j import GraphDatabase
d = GraphDatabase.driver(os.environ["NEO4J_URI"],
                         auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]))
d.verify_connectivity(); d.close()
print("✅ [job] AuraDB reachable")
PYEOF

# ── refuse to write into a group that already exists ─────────────────────────
# ui_ingest names episodes "<group>_w<N>" and would happily interleave a second
# ingest into the same group, producing a graph that is two runs blended with
# nothing recording that it happened.
"${PY}" - <<PYEOF
import os, sys
from neo4j import GraphDatabase
d = GraphDatabase.driver(os.environ["NEO4J_URI"],
                         auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]))
with d.session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as s:
    n = list(s.run("MATCH (e:Episodic) WHERE e.group_id=\$g RETURN count(*) AS n",
                   g="${GROUP_ID}"))[0]["n"]
d.close()
if n:
    sys.exit(f"❌ group '${GROUP_ID}' already has {n} episode(s). "
             f"Pick a new GROUP_ID, or delete that group first.")
print("✅ [job] group '${GROUP_ID}' is empty — safe to write")
PYEOF

cleanup() {
  echo "🧹 [job] stopping vLLM …"
  for p in "${CHAT_PORT}" "${EMBED_PORT}"; do
    PIDF="${REPO_ROOT}/logs/serve_vllm/serve_vllm_${p}.pid"
    [ -f "$PIDF" ] && kill "$(cat "$PIDF")" 2>/dev/null || true
  done
}
trap cleanup EXIT

# ── 1/3  chat model ──────────────────────────────────────────────────────────
# Same model and same context length as the corpus ingest, deliberately: a graph
# built by a different extractor is not comparable with the ones we already have.
echo "════════ 1/3  serving chat model ════════"
VLLM_PORT="${CHAT_PORT}" \
VLLM_MODEL_DIR="${CHAT_MODEL_DIR}" \
VLLM_SERVED_NAME="${MODEL_ID}" \
VLLM_MAX_LEN="${CHAT_MAX_LEN:-65536}" \
VLLM_EXTRA_ARGS="--gpu-memory-utilization ${CHAT_GPU_FRAC}" \
  bash scripts/serve_vllm.sh

# ── 2/3  embedder ────────────────────────────────────────────────────────────
echo "════════ 2/3  serving embedder (bge-m3) ════════"
VLLM_PORT="${EMBED_PORT}" \
VLLM_MODEL_DIR="${EMBED_MODEL_DIR}" \
VLLM_SERVED_NAME="${EMBED_MODEL_ID}" \
VLLM_MAX_LEN=8192 \
VLLM_EXTRA_ARGS="--runner pooling --convert embed --gpu-memory-utilization ${EMBED_GPU_FRAC}" \
  bash scripts/serve_vllm.sh

# ── 3/3  the ingest ──────────────────────────────────────────────────────────
echo "════════ 3/3  ingesting ════════"
# SEMAPHORE_LIMIT is 20 here, not the 4 the laptop uses: that cap exists for
# SAIA's rate limit, and vLLM on this node has none. The writes go to AuraDB,
# which is the only remote hop and is not the bottleneck.
GRAPHITI_LLM_BASE_URL="http://localhost:${CHAT_PORT}/v1" \
GRAPHITI_LLM_MODEL="${MODEL_ID}" \
GRAPHITI_LLM_API_KEY="dummy" \
GRAPHITI_EMBED_BASE_URL="http://localhost:${EMBED_PORT}/v1" \
GRAPHITI_EMBED_MODEL="${EMBED_MODEL_ID}" \
GRAPHITI_EMBED_API_KEY="dummy" \
GRAPHITI_EMBED_DIM=1024 \
GRAPHITI_MAX_TOKENS="${GRAPHITI_MAX_TOKENS:-8192}" \
GRAPHITI_EXCLUDE_SPEAKERS="${EXCLUDE_SPEAKERS}" \
SEMAPHORE_LIMIT="${SEMAPHORE_LIMIT:-20}" \
  "${PY}" -u ui_ingest.py --text-file "${TEXT_FILE}" --group-id "${GROUP_ID}" \
  > "${REPO_ROOT}/logs/phase_ingest_${GROUP_ID}.json" \
  2> >(tee "${REPO_ROOT}/logs/phase_ingest_${GROUP_ID}.progress.log" >&2)

echo ""
echo "🎉 [job] ALL DONE — group '${GROUP_ID}' is in AuraDB"
echo "   facts JSON : logs/phase_ingest_${GROUP_ID}.json"
echo "   next       : it appears in the UI's graph list automatically"
