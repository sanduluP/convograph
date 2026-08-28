#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  submit_shards.sh — submit the FULL Finance ingest as N parallel SLURM jobs.
#
#  WHY SHARD AT ALL
#  ----------------
#  Ingesting all 30,000 Finance messages costs ~38 h of GPU (measured: ~44 s per
#  5-message window, 6,002 windows). The cluster caps a reservation at 1 day, so
#  a single job CANNOT finish — it would be killed at the walltime with a partial
#  graph. Eight jobs of ~751 windows each finish in ~9-13 h apiece and all run at
#  the same time, so wall-clock is one night instead of two days.
#
#  WHY THIS IS SAFE
#  ----------------
#  Every job loads the SAME full corpus and numbers its windows over the WHOLE
#  corpus before slicing (see --window-range in graphiti_retriever.py). So shard
#  s3's window 1503 covers exactly the messages a single serial run would have
#  put in window 1503, and episode names never collide. Each shard also gets its
#  OWN Neo4j instance (own data dir, own ports) so the jobs cannot corrupt each
#  other; scripts/run_merge_shards.sh combines them afterwards.
#
#  RESUMABILITY
#  ------------
#  Every job passes --resume, so re-running this script re-submits jobs that will
#  SKIP whatever is already in their shard's graph. If shard 5 dies at hour 12,
#  resubmit and it costs you hour 12, not hours 1-12.
#
#  Usage:
#    bash scripts/submit_shards.sh            # submit all 8
#    SHARDS="s3 s7" bash scripts/submit_shards.sh   # re-submit just these two
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# The shard plan. Ranges are 1-based, END exclusive, and were verified to cover
# all 6,002 windows exactly once (window count is 6,002 rather than 6,000 because
# _build_windows never lets an episode straddle two channels, so each of the 6
# channels ends with a short window).
SHARD_PLAN=(
  "s1 1:752"
  "s2 752:1503"
  "s3 1503:2254"
  "s4 2254:3005"
  "s5 3005:3756"
  "s6 3756:4507"
  "s7 4507:5258"
  "s8 5258:6003"
)

# Optional whitelist: re-submit only the named shards (e.g. after a failure).
SHARDS="${SHARDS:-}"

WALLTIME="${WALLTIME:-23}"       # hours; the cluster caps a reservation at 1 day
GROUP_ID="${GROUP_ID:-gmb_finance_full}"
WINDOW="${WINDOW:-5}"

echo "🚀 submitting Finance full-corpus ingest"
echo "   group    : ${GROUP_ID}"
echo "   window   : ${WINDOW}"
echo "   walltime : ${WALLTIME} h per job"
echo ""

for entry in "${SHARD_PLAN[@]}"; do
  shard="${entry%% *}"
  range="${entry##* }"

  # Skip anything not in the whitelist, when a whitelist was given.
  if [[ -n "${SHARDS}" && " ${SHARDS} " != *" ${shard} "* ]]; then
    continue
  fi

  echo "📤 ${shard}  windows ${range}"
  # RESUME=1 is mandatory here: without it each job would WIPE the group before
  # ingesting, and eight jobs wiping each other would leave one shard's worth of
  # graph at the end.
  SHARD="${shard}" \
  WINDOW_RANGE="${range}" \
  RESUME=1 \
  CORPUS=0 \
  WINDOW="${WINDOW}" \
  GROUP_ID="${GROUP_ID}" \
  INGEST_ONLY=1 \
  CONTROL_BM25=0 \
  GRAPHITI_SPEAKER_FRAMING="${GRAPHITI_SPEAKER_FRAMING:-speaker}" \
    nohup bash scripts/srun_submit.sh all "${JOB_PREFIX:-gmb}_${shard}" 8 1 96G "${WALLTIME}" \
      scripts/cluster_ingest_job.sh > "/tmp/submit_${JOB_PREFIX:-gmb}_${shard}.log" 2>&1 &
  # Stagger submissions: eight jobs racing to read the same 30,000-message JSON
  # and the same 30 GB of model weights off /fscratch at the identical instant is
  # a needless I/O spike.
  sleep 8
done

wait
echo ""
echo "✅ all submissions issued — check with: squeue -u abuali"
