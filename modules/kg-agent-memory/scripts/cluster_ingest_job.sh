#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# cluster_ingest_job.sh — the SELF-CONTAINED Graphiti ingest job.
#
# THE POINT: everything runs on ONE compute node, talking over localhost, so the
# run does NOT depend on Faris's laptop, the VPN, or an SSH tunnel. Submit it,
# close the laptop, go to sleep. Previously the ingest was orchestrated from the
# laptop (Neo4j local, LLM remote) and a VPN drop killed a 200-window run at
# window 37 — that is the failure this job exists to make impossible.
#
# It starts, in order, all on this node:
#   1. Neo4j          (from /fscratch, plain process + conda JRE — no container)
#   2. vLLM chat      :8000  — extraction/QA model
#   3. vLLM embedder  :8001  — bge-m3, because serv-3306 returns HTTP 403 from
#                              the cluster (it is VPN-gated, interactive-use-only)
#   4. the ingest itself, then stops everything.
#
# The Neo4j DATA DIR lives on /fscratch, so the KG PERSISTS after the job ends and
# can be rsynced to the laptop for eyeballing in Neo4j Browser.
#
# Submit (wide FP8-safe partitions, rule 8):
#   bash scripts/srun_submit.sh all gmb_ingest 8 1 96G 10 scripts/cluster_ingest_job.sh
#
# Knobs: CORPUS, WINDOW, GROUP_ID, DOMAIN, MODEL_ID (see below).
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

FS_ROOT=${FS_ROOT:-/fscratch/abuali}
VENV="${FS_ROOT}/venvs/groupmembench"
PY="${VENV}/bin/python"
[ -x "$PY" ] || { echo "❌ no cluster venv at ${VENV}"; exit 1; }

# --- Neo4j (installed by scripts/setup_cluster_neo4j.sh) ---------------------
NEO4J_VERSION=${NEO4J_VERSION:-5.26.0}
NEO4J_HOME="${FS_ROOT}/neo4j/neo4j-community-${NEO4J_VERSION}"
JAVA_HOME="${FS_ROOT}/conda_envs/java21"
NEO4J_PASSWORD=${NEO4J_PASSWORD:-graphiti123}
export JAVA_HOME
export PATH="${JAVA_HOME}/bin:${PATH}"

# --- models -------------------------------------------------------------------
CHAT_MODEL_DIR=${CHAT_MODEL_DIR:-"${FS_ROOT}/models/Qwen3-30B-A3B-Instruct-2507-FP8"}
MODEL_ID=${MODEL_ID:-"Qwen/$(basename "${CHAT_MODEL_DIR}")"}
EMBED_MODEL_DIR=${EMBED_MODEL_DIR:-"${FS_ROOT}/models/bge-m3"}
EMBED_MODEL_ID=${EMBED_MODEL_ID:-"BAAI/bge-m3"}
# Ports are picked at RUNTIME, not hardcoded: compute nodes are shared, and another
# user's process already owned 8001 on serv-3342 ("Address already in use"), which
# killed the job at step 3. free_port() asks the OS for an unused port instead.
free_port() {
  "${PY}" - <<'PYEOF'
import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))   # port 0 = let the kernel choose a free one
print(s.getsockname()[1])
s.close()
PYEOF
}
CHAT_PORT=${CHAT_PORT:-$(free_port)}
EMBED_PORT=${EMBED_PORT:-$(free_port)}
BOLT_PORT=${BOLT_PORT:-$(free_port)}
HTTP_PORT=${HTTP_PORT:-$(free_port)}
echo "🔌 [job] ports: chat=${CHAT_PORT}  embed=${EMBED_PORT}  bolt=${BOLT_PORT}"

# --- GPU budget must be sized to the CARD, not to one lucky node ---------------
# We submit to a wide partition list (rule 8), so the job may land on an 80 GB
# H100/H200 or on a 48 GB L40S. Both servers share ONE card, so their utilisation
# fractions must sum to < 1.0 — but the chat model's ~30 GB of FP8 weights are a
# FIXED cost, so what is left over for KV cache differs enormously:
#   80 GB card @ 0.78  →  ~30 GB KV  (317,552 tokens, measured on serv-3342)
#   48 GB card @ 0.78  →  ~3.7 GB KV (measured on serv-3304)
# Two lessons paid for by dead shards on 2026-08-03:
#   1. A hardcoded max-model-len of 65,536 needs 6.0 GB of KV for a single
#      sequence — fine on the H100, instant death on the L40S.
#   2. Lowering it to 40,960 was NOT enough: KV cache size is set by
#      gpu_memory_utilization, not by the context length, so the card still
#      offered exactly 3.74 GB against the 3.75 GB that 40,960 needs. The fix is
#      to also hand the chat model a BIGGER SLICE of the small card, taking it
#      from the embedder (bge-m3 is ~2.3 GB and needs very little).
if [[ -z "${CHAT_MAX_LEN:-}" ]]; then
  GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)
  if (( GPU_MEM_MB >= 70000 )); then
    # 80 GB class: the proven configuration from the 12 h 29 m serv-3342 run.
    CHAT_MAX_LEN=65536; CHAT_GPU_FRAC_DEFAULT=0.78; EMBED_GPU_FRAC_DEFAULT=0.10
  elif (( GPU_MEM_MB >= 44000 )); then
    # 48 GB class: 0.86 leaves ~7.4 GB of KV — comfortably above the 3.75 GB that
    # a single 40,960-token request needs, so short prompts still batch well.
    CHAT_MAX_LEN=40960; CHAT_GPU_FRAC_DEFAULT=0.86; EMBED_GPU_FRAC_DEFAULT=0.08
  else
    CHAT_MAX_LEN=32768; CHAT_GPU_FRAC_DEFAULT=0.86; EMBED_GPU_FRAC_DEFAULT=0.08
  fi
  echo "🧠 [job] GPU has ${GPU_MEM_MB} MiB → max-model-len=${CHAT_MAX_LEN}," \
       "chat_frac=${CHAT_GPU_FRAC_DEFAULT} embed_frac=${EMBED_GPU_FRAC_DEFAULT}"
else
  CHAT_GPU_FRAC_DEFAULT=0.78; EMBED_GPU_FRAC_DEFAULT=0.10
fi
export CHAT_MAX_LEN
CHAT_GPU_FRAC=${CHAT_GPU_FRAC:-${CHAT_GPU_FRAC_DEFAULT}}
EMBED_GPU_FRAC=${EMBED_GPU_FRAC:-${EMBED_GPU_FRAC_DEFAULT}}

# --- ingest knobs -------------------------------------------------------------
DOMAIN=${DOMAIN:-Finance}
CORPUS=${CORPUS:-0}            # 0 = whole corpus; set e.g. 50 for the probe
WINDOW=${WINDOW:-5}
GROUP_ID=${GROUP_ID:-gmb_cluster}
STRICT_PROMPT=${STRICT_PROMPT:-1}
QTYPE=${QTYPE:-knowledge_update}
# INGEST_ONLY=1 builds the KG and stops (the probe). Set 0 to also run QA and get
# an accuracy number.
INGEST_ONLY=${INGEST_ONLY:-1}
# CONTROL_BM25=1 re-runs the SAME questions over the SAME corpus with BM25, so the
# only difference between the two numbers is the retrieval method. Without this,
# a Graphiti score on one channel is not comparable to the 52.8% BM25 figure,
# which was measured over all 30,000 messages.
CONTROL_BM25=${CONTROL_BM25:-0}

# --- sharding (parallel ingest across jobs) -----------------------------------
# The cluster caps a reservation at 1 day, but ingesting all 30,000 Finance
# messages costs ~38 h of GPU. So we shard: N jobs, each ingesting a slice of the
# GLOBAL window list, each with its OWN Neo4j instance (own data dir, own ports)
# so the jobs never fight over a database lock. The partial graphs are merged
# afterwards by scripts/run_merge_shards.sh — safe because every job numbers its
# windows over the full corpus, so episode names and message indices agree.
#   SHARD        : short name, e.g. "s1". Empty = the old single-instance mode.
#   WINDOW_RANGE : "START:END", 1-based, END exclusive. Empty = all windows.
SHARD=${SHARD:-}
WINDOW_RANGE=${WINDOW_RANGE:-}
# STORE_ROOT points the job at an ARBITRARY Neo4j store instead of a shard's.
# The reason it exists: scoring the full corpus means talking to the MERGED store
# (${FS_ROOT}/neo4j/merged/<group>), which is neither the default data dir nor
# any shard. Setting it reuses the whole per-shard isolation mechanism (own conf,
# own data dir, own ports) for a store that is not a shard.
STORE_ROOT=${STORE_ROOT:-}
# RETRIEVE_ONLY=1 scores an EXISTING graph: ingest nothing, just map episodes back
# to message indices and run QA. This is what makes a scoring run minutes instead
# of hours — without it the job would ingest every window the original run skipped.
RETRIEVE_ONLY=${RETRIEVE_ONLY:-0}
# RESULT_TAG distinguishes result files from runs that differ only in CONFIG
# (e.g. stitched vs no-stitch merge). Without it both write the same path and
# the second run silently destroys the first run's evidence — which is exactly
# the kind of loss you notice only when you go to compare them.
RESULT_TAG=${RESULT_TAG:-}
# CHANNEL_SCOPE filters retrieval to ONE channel before the QA agent sees it.
#   none     = every run so far (no filtering)
#   oracle   = pick the channel using the gold answer → the CEILING
#   majority = pick the most-represented channel → the buildable version
# RETRIEVER selects which retrieval strategy is scored against the SAME graph.
#   graphiti = entity/embedding search (the real system)
#   kgfacts  = BM25 over the graph's fact TEXT (the extraction-vs-retrieval
#              ablation — same facts, different index)
RETRIEVER=${RETRIEVER:-graphiti}
CHANNEL_SCOPE=${CHANNEL_SCOPE:-none}
CHANNEL_OVERSAMPLE=${CHANNEL_OVERSAMPLE:-3}
TAG_SUFFIX=""
[[ -n "${RESULT_TAG}" ]] && TAG_SUFFIX="_${RESULT_TAG}"
# RESUME=1 keeps whatever is already in the group instead of wiping it. This is
# what makes a job restartable after a walltime kill: resubmit the SAME command
# and it picks up at the first window it never finished.
RESUME=${RESUME:-0}

# Each shard gets an isolated Neo4j: same read-only install, but its own conf,
# data, logs, run dir and ports. NEO4J_CONF is how Neo4j 5 is told to read a conf
# directory other than $NEO4J_HOME/conf.
if [[ -n "${SHARD}" || -n "${STORE_ROOT}" ]]; then
  # STORE_ROOT wins when given (the merged store); otherwise derive it from SHARD.
  SHARD_ROOT="${STORE_ROOT:-${FS_ROOT}/neo4j/shards/${SHARD}}"
  mkdir -p "${SHARD_ROOT}"/{conf,data,logs,run}
  # Start from the base conf (so JVM settings etc. stay in sync), then replace
  # our block with shard-specific paths and ports.
  sed '/^# --- groupmembench settings ---/,$d' "${NEO4J_HOME}/conf/neo4j.conf" \
    > "${SHARD_ROOT}/conf/neo4j.conf"
  cat >> "${SHARD_ROOT}/conf/neo4j.conf" <<EOF
# --- groupmembench settings (shard ${SHARD}) ---
server.default_listen_address=127.0.0.1
server.bolt.listen_address=127.0.0.1:${BOLT_PORT}
server.http.listen_address=127.0.0.1:${HTTP_PORT}
server.directories.data=${SHARD_ROOT}/data
server.directories.logs=${SHARD_ROOT}/logs
server.directories.run=${SHARD_ROOT}/run
server.memory.heap.initial_size=2g
server.memory.heap.max_size=8g
server.memory.pagecache.size=4g
EOF
  # Neo4j copies these two files out of the install dir at first start; without
  # them a fresh conf dir refuses to boot.
  cp -n "${NEO4J_HOME}/conf/server-logs.xml" "${SHARD_ROOT}/conf/" 2>/dev/null || true
  cp -n "${NEO4J_HOME}/conf/user-logs.xml"   "${SHARD_ROOT}/conf/" 2>/dev/null || true
  export NEO4J_CONF="${SHARD_ROOT}/conf"
  NEO4J_DATA_DIR="${SHARD_ROOT}/data"
  echo "🧩 [job] store=${SHARD:-${STORE_ROOT}}  data=${NEO4J_DATA_DIR}  bolt=${BOLT_PORT}"
else
  # Unsharded mode keeps the original single instance, whose conf hardcodes 7687.
  NEO4J_DATA_DIR="${FS_ROOT}/neo4j/data"
  BOLT_PORT=7687
fi
NEO4J_BOLT_URI="bolt://localhost:${BOLT_PORT}"

cleanup() {
  echo "🧹 [job] stopping services …"
  for p in "${CHAT_PORT}" "${EMBED_PORT}"; do
    PIDF="${REPO_ROOT}/logs/serve_vllm/serve_vllm_${p}.pid"
    [ -f "$PIDF" ] && kill "$(cat "$PIDF")" 2>/dev/null || true
  done
  "${NEO4J_HOME}/bin/neo4j" stop >/dev/null 2>&1 || true
  echo "🧹 [job] done"
}
# Stop everything on ANY exit path (success, error, or SLURM timeout) so we never
# leak a Neo4j holding a lock on the shared /fscratch store.
trap cleanup EXIT

echo "🖥️  [job] node=$(hostname)  domain=${DOMAIN}  corpus=${CORPUS}  window=${WINDOW}"

# ══ 1/4  Neo4j ═══════════════════════════════════════════════════════════════
echo "════════ 1/4  starting Neo4j ════════"
[ -x "${NEO4J_HOME}/bin/neo4j" ] || { echo "❌ Neo4j missing — run scripts/setup_cluster_neo4j.sh on the LOGIN node"; exit 1; }
# A shard's data dir starts empty, so its password has never been initialised.
# set-initial-password only works before the DB first starts; afterwards it exits
# non-zero, which is fine and expected on a resume.
if [[ -n "${SHARD}" || -n "${STORE_ROOT}" ]]; then
  "${NEO4J_HOME}/bin/neo4j-admin" dbms set-initial-password "${NEO4J_PASSWORD}" 2>/dev/null \
    && echo "🔐 [job] store password initialised" \
    || echo "ℹ️  [job] store password already set (existing store)"
fi
"${NEO4J_HOME}/bin/neo4j" start
# Wait for bolt to actually accept connections (neo4j start returns before ready).
for i in $(seq 1 60); do
  if "${PY}" - <<PYEOF 2>/dev/null
from neo4j import GraphDatabase
d = GraphDatabase.driver("${NEO4J_BOLT_URI}", auth=("neo4j", "${NEO4J_PASSWORD}"))
d.verify_connectivity(); d.close()
PYEOF
  then echo "✅ [job] Neo4j ready after ${i} checks"; break; fi
  [ "$i" = "60" ] && { echo "❌ Neo4j never became ready"; tail -30 "$(dirname "${NEO4J_DATA_DIR}")/logs/neo4j.log" 2>/dev/null; exit 1; }
  sleep 5
done

# ══ 2/4  vLLM chat model ═════════════════════════════════════════════════════
echo "════════ 2/4  serving chat model ════════"
# Context length: 32768 was NOT enough. Graphiti's node-dedup prompt
# (resolve_extracted_nodes) embeds the existing entity candidates, so the PROMPT
# grows as the graph grows. On the 2026-08-02 run it crossed 16,385 input tokens
# by window 61 and — with 16,384 tokens reserved for output — blew the 32k budget
# with HTTP 400 on 96 of 628 windows. vLLM reported 317,552 tokens of KV cache on
# that GPU, so raising max-model-len is essentially free: at 65,536 the engine
# still serves ~4.8 concurrent full-length requests, and our real prompts are
# ~10-16k. Combined with GRAPHITI_MAX_TOKENS=8192 below this leaves ~57k of input
# headroom — 3.5x the size that broke the last run.
VLLM_PORT="${CHAT_PORT}" \
VLLM_MODEL_DIR="${CHAT_MODEL_DIR}" \
VLLM_SERVED_NAME="${MODEL_ID}" \
VLLM_MAX_LEN="${CHAT_MAX_LEN:-65536}" \
VLLM_EXTRA_ARGS="--gpu-memory-utilization ${CHAT_GPU_FRAC}" \
  bash scripts/serve_vllm.sh

# ══ 3/4  vLLM embedder ═══════════════════════════════════════════════════════
# Two model-specific flags, both verified against THIS vLLM (0.25.1):
#   --runner pooling  : vLLM 0.25 replaced the old `--task embed` with
#                       --runner {generate,pooling,draft} + --convert {none,embed,
#                       classify}. Passing `--task embed` fails with
#                       "unrecognized arguments".
#   (no --load-format): bge-m3 ships pytorch_model.bin. `--load-format pt` matches
#                       only "*.pt" and fails with "Cannot find any model weights";
#                       the format that matches "*.bin" is "hf", which the default
#                       "auto" already selects when no safetensors are present.
echo "════════ 3/4  serving embedder (bge-m3) ════════"
VLLM_PORT="${EMBED_PORT}" \
VLLM_MODEL_DIR="${EMBED_MODEL_DIR}" \
VLLM_SERVED_NAME="${EMBED_MODEL_ID}" \
VLLM_MAX_LEN=8192 \
VLLM_EXTRA_ARGS="--runner pooling --convert embed --gpu-memory-utilization ${EMBED_GPU_FRAC}" \
  bash scripts/serve_vllm.sh

# ══ 4/4  the ingest ══════════════════════════════════════════════════════════
echo "════════ 4/4  running the ingest ════════"
CONVERSATION_JSON="${REPO_ROOT}/data/final/${DOMAIN}/synthetic_domain_channels_rolevariants_${DOMAIN}.json"
mkdir -p "${REPO_ROOT}/results/cluster"

# QTYPE may be a SPACE-SEPARATED LIST, and every entry is scored inside THIS job.
#
# Why it has to work that way: Neo4j takes an EXCLUSIVE lock on its data
# directory. Submitting one job per question type against the same store means
# the first job to start wins the lock and every other job dies with
# "Neo4j never became ready" — which is exactly what happened on 2026-08-12 when
# 5 concurrent jobs were pointed at merged/gmb_finance_full and 4 of them died.
# Looping here instead shares one Neo4j and one vLLM across all the types, which
# is also cheaper: the ~30 GB model is loaded once rather than N times.
for QT in ${QTYPE}; do
echo ""
echo "──────── question type: ${QT} ────────"
QUESTIONS_JSONL="${REPO_ROOT}/questions/${DOMAIN}/${QT}.jsonl"
if [[ ! -f "${QUESTIONS_JSONL}" ]]; then
  echo "⚠️  [job] missing ${QUESTIONS_JSONL} — skipping ${QT}"
  continue
fi

ARGS=(
  --conversation-json "${CONVERSATION_JSON}"
  --questions-jsonl "${QUESTIONS_JSONL}"
  --env-file ".env.cluster"
  --agent-model "${MODEL_ID}"
  --judge-model "${MODEL_ID}"
  --agent-prompt "prompts/hipporag_agent_system.txt"
  --judge-prompt "prompts/hipporag_judge_system.txt"
  --retrieve-top-k 10
  --output-jsonl "${REPO_ROOT}/results/cluster/${RETRIEVER}_${GROUP_ID}_${QT}${TAG_SUFFIX}.jsonl"
  --llm-provider openai
  --retriever "${RETRIEVER}"
  --window "${WINDOW}"
  --group-id "${GROUP_ID}"
)
[[ "${CORPUS}" != "0" ]] && ARGS+=(--max-corpus-messages "${CORPUS}")
[[ "${STRICT_PROMPT}" == "1" ]] && ARGS+=(--strict-prompt)
[[ "${INGEST_ONLY}" == "1" ]] && ARGS+=(--ingest-only)
[[ -n "${WINDOW_RANGE}" ]] && ARGS+=(--window-range "${WINDOW_RANGE}")
[[ "${RESUME}" == "1" && "${RETRIEVER}" == "graphiti" ]] && ARGS+=(--resume)
# --retrieve-only implies --resume inside the retriever, so a scoring run can
# never wipe the graph it was asked to score.
[[ "${RETRIEVE_ONLY}" == "1" && "${RETRIEVER}" == "graphiti" ]] && ARGS+=(--retrieve-only)
[[ "${CHANNEL_SCOPE}" != "none" ]] && ARGS+=(--channel-scope "${CHANNEL_SCOPE}" --channel-oversample "${CHANNEL_OVERSAMPLE}")

# Everything points at localhost — that is the whole reason this job is immune to
# VPN/tunnel failure.
PYTHONUNBUFFERED=1 PYTHONPATH="${REPO_ROOT}" \
OPENAI_BASE_URL="http://localhost:${CHAT_PORT}/v1" \
OPENAI_API_KEY="dummy" \
GRAPHITI_LLM_BASE_URL="http://localhost:${CHAT_PORT}/v1" \
GRAPHITI_LLM_MODEL="${MODEL_ID}" \
GRAPHITI_LLM_API_KEY="dummy" \
GRAPHITI_EMBED_BASE_URL="http://localhost:${EMBED_PORT}/v1" \
GRAPHITI_EMBED_MODEL="${EMBED_MODEL_ID}" \
GRAPHITI_EMBED_API_KEY="dummy" \
GRAPHITI_EMBED_DIM=1024 \
GRAPHITI_MAX_TOKENS="${GRAPHITI_MAX_TOKENS:-8192}" \
GRAPHITI_BFS="${GRAPHITI_BFS:-0}" \
GRAPHITI_SPEAKER_FRAMING="${GRAPHITI_SPEAKER_FRAMING:-speaker}" \
GRAPHITI_EXCLUDE_SPEAKERS="${GRAPHITI_EXCLUDE_SPEAKERS:-0}" \
GRAPHITI_BFS_DEPTH="${GRAPHITI_BFS_DEPTH:-1}" \
NEO4J_URI="${NEO4J_BOLT_URI}" \
NEO4J_USER="neo4j" \
NEO4J_PASSWORD="${NEO4J_PASSWORD}" \
SEMAPHORE_LIMIT="${SEMAPHORE_LIMIT:-20}" \
  "${PY}" baselines/graphiti/eval_benchmark.py "${ARGS[@]}"

echo "✅ [job] graphiti run finished — KG persists at ${NEO4J_DATA_DIR} (group '${GROUP_ID}')"

# ══ 5/5  scope-matched BM25 control (optional) ═══════════════════════════════
# BM25 is CPU-only and instant to index; the cost is the same agent+judge calls.
# Running it here, in the same job against the same vLLM and the same truncated
# corpus, is what makes the Graphiti number interpretable.
if [[ "${CONTROL_BM25}" == "1" && "${INGEST_ONLY}" != "1" ]]; then
  echo "════════ 5/5  scope-matched BM25 control ════════"
  CTRL_ARGS=(
    --conversation-json "${CONVERSATION_JSON}"
    --questions-jsonl "${QUESTIONS_JSONL}"
    --env-file ".env.cluster"
    --agent-model "${MODEL_ID}"
    --judge-model "${MODEL_ID}"
    --agent-prompt "prompts/hipporag_agent_system.txt"
    --judge-prompt "prompts/hipporag_judge_system.txt"
    --retrieve-top-k 10
    --output-jsonl "${REPO_ROOT}/results/cluster/bm25_${GROUP_ID}_${QT}${TAG_SUFFIX}.jsonl"
    --llm-provider openai
    --retriever bm25
  )
  [[ "${CORPUS}" != "0" ]] && CTRL_ARGS+=(--max-corpus-messages "${CORPUS}")
  [[ "${CHANNEL_SCOPE}" != "none" ]] && CTRL_ARGS+=(--channel-scope "${CHANNEL_SCOPE}" --channel-oversample "${CHANNEL_OVERSAMPLE}")
  PYTHONUNBUFFERED=1 PYTHONPATH="${REPO_ROOT}" \
  OPENAI_BASE_URL="http://localhost:${CHAT_PORT}/v1" \
  OPENAI_API_KEY="dummy" \
    "${PY}" baselines/graphiti/eval_benchmark.py "${CTRL_ARGS[@]}"
  echo "✅ [job] BM25 control finished"
fi

echo "✔️  [job] finished question type: ${QT}"
done   # ← end of the per-question-type loop opened before step 4/4

echo "🎉 [job] ALL DONE (corpus=${CORPUS}, window=${WINDOW}, qtypes=${QTYPE})"
