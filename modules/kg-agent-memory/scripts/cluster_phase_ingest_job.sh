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

# The cluster venv lives on SCRATCH, not in the repo: $HOME is source code only
# (DFKI storage guidelines) and a 10 GB quota that a torch install would eat.
# There IS a stale ./.venv directory in $HOME from July with nothing in it, so
# defaulting to the repo-local path finds a directory and fails on the missing
# binary. Same definition as cluster_ingest_job.sh, deliberately.
FS_ROOT_EARLY="${FS_ROOT:-/fscratch/abuali}"
VENV="${VENV:-${FS_ROOT_EARLY}/venvs/groupmembench}"
PY="${VENV}/bin/python"
[ -x "${PY}" ] || { echo "❌ no cluster venv at ${VENV}"; \
  echo "   build it with scripts/setup_cluster_env.sh on the LOGIN node"; exit 1; }

TEXT_FILE="${TEXT_FILE:?set TEXT_FILE to the phase transcript (analysis/extract_phase.py)}"
GROUP_ID="${GROUP_ID:?set GROUP_ID — it is how the UI will name this graph}"
FS_ROOT="${FS_ROOT:-/fscratch/abuali}"
CHAT_PORT="${CHAT_PORT:-8000}"
# 8100, NOT 8001. vLLM's V1 engine opens an internal engine-core socket on the
# port just above its API port, so a chat server on 8000 takes 8001 for itself
# and the embedder then dies with "OSError: [Errno 98] Address already in use" —
# a collision between two halves of the SAME job, which reads like a stale
# process from someone else's run. Verified on serv-3313:
#     :8000  vllm             (the API server)
#     :8001  VLLM::EngineCor  (its engine core)
# A wide separation costs nothing and removes the whole class.
EMBED_PORT="${EMBED_PORT:-8100}"
CHAT_MODEL_DIR="${CHAT_MODEL_DIR:-${FS_ROOT}/models/Qwen3-30B-A3B-Instruct-2507-FP8}"
EMBED_MODEL_DIR="${EMBED_MODEL_DIR:-${FS_ROOT}/models/bge-m3}"
MODEL_ID="${MODEL_ID:-Qwen/$(basename "${CHAT_MODEL_DIR}")}"
EMBED_MODEL_ID="${EMBED_MODEL_ID:-BAAI/bge-m3}"
# ── size vLLM from the CARD WE ACTUALLY GOT, not from a constant ─────────────
# The first attempt inherited the corpus job's H100 numbers - 0.72 utilisation,
# 65,536 context - and landed on an L40S. The 30B FP8 weights are 29.1 GiB; on a
# 48 GB card 0.72 leaves 34.5 GiB, so after weights and CUDA graphs vLLM reported
# "Available KV cache memory: 1.08 GiB" and the engine refused to start, since
# 65k tokens of context needs far more than that. On an 80 GB H100 the same
# numbers are comfortable, which is why the corpus job never hit this.
#
# So: detect the card and size to it. A short context is cheap here - this job
# ingests ONE phase, ~81 windows, so Graphiti's node-dedup prompt (which grows
# with the graph) stays small. That is the assumption that would break on a
# corpus-sized run, and it is why the corpus job keeps 65k.
GPU_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo unknown)
if [[ "${GPU_MB}" -ge 70000 ]]; then          # H100 / H200 / B200 (80 GB+)
  CHAT_GPU_FRAC="${CHAT_GPU_FRAC:-0.72}"; CHAT_MAX_LEN="${CHAT_MAX_LEN:-65536}"
else
  # An earlier version tried 16,384 context on a 48 GB L40S, reasoning that one
  # phase makes a small graph so Graphiti's prompts stay short. That was wrong in
  # two ways, and both showed up on the FIRST window:
  #
  #   * graphiti's edge-extraction prompt is 15,168 characters before any graph
  #     exists to grow it — the length comes from the instructions, not the data;
  #   * OpenAIGenericClient.__init__ defaults max_tokens to 16384 and OVERRIDES
  #     the config with it (llm_client/openai_generic_client.py:65,94), so the
  #     request asks for the entire context as OUTPUT and leaves nothing for input.
  #
  # Either alone would fit in 65k. Together they need a context this card cannot
  # give alongside 29.1 GiB of weights, so refuse rather than half-work.
  echo "❌ ${GPU_NAME} has ${GPU_MB} MB — not enough for this job."
  echo "   The 30B FP8 weights are 29.1 GiB and graphiti needs a 65k context:"
  echo "   its edge-extraction prompt is ~15k characters and the client requests"
  echo "   16,384 output tokens regardless of what the config says."
  echo "   Resubmit pinned to an 80 GB card:"
  echo "       bash scripts/srun_submit.sh H100,H100-RP,H100-PCI,H200,H200-PCI,B200 ..."
  exit 1
fi
EMBED_GPU_FRAC="${EMBED_GPU_FRAC:-0.10}"
echo "🎮 [job] ${GPU_NAME} (${GPU_MB} MB) → chat util ${CHAT_GPU_FRAC}, ctx ${CHAT_MAX_LEN}"

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
# The check reports the driver's OWN error rather than assuming what went wrong.
# An earlier version wrapped this in `|| echo "cannot reach AuraDB"`, which fires
# on any non-zero exit — including 127 from a missing interpreter. It did, and
# the job confidently blamed the network while the container could resolve,
# connect and TLS-verify the database without trouble.
set +e
REACH_OUT="$("${PY}" - <<'PYEOF' 2>&1
import os, sys
try:
    from neo4j import GraphDatabase
    d = GraphDatabase.driver(os.environ["NEO4J_URI"],
                             auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]))
    d.verify_connectivity(); d.close()
    print("REACHABLE")
except Exception as exc:
    print(f"UNREACHABLE {type(exc).__name__}: {exc}")
    sys.exit(1)
PYEOF
)"
REACH_RC=$?
set -e
if [[ ${REACH_RC} -ne 0 ]]; then
  echo "❌ [job] the reachability check failed on $(hostname):"
  echo "${REACH_OUT}" | sed 's/^/     /'
  echo "   If this says UNREACHABLE, the node has no route to AuraDB — ingest to a"
  echo "   local store with cluster_ingest_job.sh and upload afterwards. Anything"
  echo "   else is a problem with this job, not with the network."
  exit 1
fi
echo "✅ [job] AuraDB reachable"

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

for port in "${CHAT_PORT}" "${EMBED_PORT}"; do
  if ss -ltn 2>/dev/null | grep -q ":${port} "; then
    echo "❌ [job] port ${port} is already in use on $(hostname):"
    ss -ltnp 2>/dev/null | grep ":${port} " | sed 's/^/     /' || true
    echo "   Another job is on this node, or a previous run leaked a server."
    echo "   Resubmit with CHAT_PORT/EMBED_PORT set to free ports."
    exit 1
  fi
done

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
VLLM_MAX_LEN="${CHAT_MAX_LEN}" \
VLLM_EXTRA_ARGS="--gpu-memory-utilization ${CHAT_GPU_FRAC}" \
  bash scripts/serve_vllm.sh

# If the engine still cannot fit a KV cache, say so in the terms the log used —
# "Available KV cache memory: X GiB" is the line that explains the failure, and
# it is 40 lines above the traceback that actually gets printed.
# NOTE THE `|| true` ON BOTH LINES. This block killed the job it was written to
# help debug: under `set -euo pipefail`, `grep … | tail -1` makes tail close the
# pipe early, grep dies of SIGPIPE, pipefail turns that into exit 141, and set -e
# ends the run — right after vLLM had come up healthy. A diagnostic must not be
# able to fail the thing it is diagnosing.
VLLM_LOG="$(ls -t "${REPO_ROOT}/logs/serve_vllm/"serve_vllm_*.log 2>/dev/null | head -1 || true)"
KV_LINE="$(grep "Available KV cache memory" "${VLLM_LOG:-/dev/null}" 2>/dev/null | tail -1 || true)"
[[ -n "${KV_LINE}" ]] && echo "   ${KV_LINE}" || true

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
