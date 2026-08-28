#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_merge_shards.sh — submit the shard-merge job (run this on the Pegasus
# LOGIN node, not on your laptop).
#
# It does two things the merge job itself cannot do:
#   1. REFUSES to submit while any gmb_s* ingest job is still running. Two Neo4j
#      processes on two nodes opening the same /fscratch store would corrupt it,
#      and NFS does not reliably enforce Neo4j's store lock — so the guard is the
#      only thing standing between us and a silently broken graph.
#   2. Requests GPUS=0 on a BROAD partition list. The merge is pure Neo4j I/O
#      plus Python (embeddings ride along inside the dumps, nothing is
#      re-embedded), so a 0-GPU job schedules on whatever node is free — including
#      the plain `batch` partition, which the FP8-safe "all" list excludes.
#
# Usage:
#   bash scripts/run_merge_shards.sh                    # merge all 8 shards
#   SHARD_LIST="s1 s5" bash scripts/run_merge_shards.sh # merge a subset
#   FORCE=1 bash scripts/run_merge_shards.sh            # bypass the squeue guard
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/merge_shards.log"

# tee EVERYTHING from here on, so this submission is inspectable after the fact
# exactly like every other run in this repo.
exec > >(tee -a "${LOG_FILE}") 2>&1

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "🧩 merge shards → one graph   $(date '+%Y-%m-%d %H:%M:%S')"
echo "════════════════════════════════════════════════════════════════"

GROUP_ID=${GROUP_ID:-gmb_finance_full}
SHARD_LIST=${SHARD_LIST:-"s1 s2 s3 s4 s5 s6 s7 s8"}
EXPECTED_WINDOWS=${EXPECTED_WINDOWS:-6002}
WALLTIME=${WALLTIME:-6}
FORCE=${FORCE:-0}

# ── Guard: no ingest job may still hold a shard store ────────────────────────
if command -v squeue >/dev/null 2>&1; then
  RUNNING=$(squeue -u "${USER}" -h -o '%j %T' 2>/dev/null | grep -E '^gmb_s[0-9]+ ' || true)
  if [[ -n "${RUNNING}" ]]; then
    echo "🚫 refusing to submit — these ingest jobs still hold shard stores:"
    echo "${RUNNING}" | sed 's/^/     /'
    if [[ "${FORCE}" != "1" ]]; then
      echo ""
      echo "   Wait for them to finish (or scancel them), then re-run."
      echo "   Merging now risks corrupting a store that is being written to."
      exit 1
    fi
    echo "⚠️  FORCE=1 — submitting anyway, on your head be it."
  else
    echo "✅ no gmb_s* ingest jobs running — safe to merge"
  fi
else
  echo "⚠️  squeue not found — cannot verify that ingest jobs have stopped."
  echo "   Run this on the Pegasus LOGIN node."
  [[ "${FORCE}" == "1" ]] || exit 1
fi

# ── Submit ───────────────────────────────────────────────────────────────────
# `batch` first: it is the CPU-oriented partition and this job needs no GPU, so
# it should take a plain node before it takes a scarce accelerator (rule 8 —
# submit broad, and never leave a CPU-only job pending behind GPU work).
PARTITIONS=${PARTITIONS:-"batch,L40S-DSA,L40S,L40S-AV,H100,H100-RP,H100-PCI,H200,H200-PCI,B200"}

echo "🏷️  group    : ${GROUP_ID}"
echo "🧩 shards   : ${SHARD_LIST}"
echo "🪟 windows  : ${EXPECTED_WINDOWS} expected"
echo "⏱️  walltime : ${WALLTIME} h   (0 GPUs)"
echo "🌐 partitions: ${PARTITIONS}"
echo ""

GROUP_ID="${GROUP_ID}" \
SHARD_LIST="${SHARD_LIST}" \
EXPECTED_WINDOWS="${EXPECTED_WINDOWS}" \
MERGED_ROOT="${MERGED_ROOT:-}" \
STITCH="${STITCH:-1}" \
SKIP_EXPORT="${SKIP_EXPORT:-0}" \
  bash scripts/srun_submit.sh "${PARTITIONS}" "${JOB_NAME:-gmb_merge}" 8 0 96G "${WALLTIME}" \
    scripts/cluster_merge_job.sh

echo ""
echo "✅ submitted — follow it with:"
echo "     squeue -u ${USER}"
echo "     tail -f /fscratch/abuali/logs/gmb_merge_*.log"
