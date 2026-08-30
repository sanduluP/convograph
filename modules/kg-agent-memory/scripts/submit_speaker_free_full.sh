#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  submit_speaker_free_full.sh — re-ingest the WHOLE Finance domain with speakers
#  excluded from the node set.
#
#  WHY, IN ONE NUMBER
#  ------------------
#  The 60-window A/B measured on 2026-08-30:
#
#                              baseline    speakers excluded
#    person-rooted facts         96.4 %          33.9 %
#    thing -> thing facts         3.4 %          60.5 %      ← 18x
#    User_N nodes surviving           8               0
#    facts per episode             19.1            16.8      ← recall held
#    biggest hub              User_2 (242)   compliance (252)
#
#  Six earlier interventions (entity stitching, group_id namespacing, channel
#  scoping, kgfacts, BFS traversal, "[said by User_N]" reframing) all measured
#  exactly zero. This one moved the defining metric by 18x, and the recall check
#  passed: the extractor did not give up when denied the speaker, it found two
#  concepts instead.
#
#  ⚠️ NOTHING EXISTING IS TOUCHED
#  ------------------------------
#  Own group_id (`finance_speaker_free`) and own store tree under
#  neo4j/shards_speaker_free/. The 40-hour `gmb_finance_full` graph, the merged
#  stores, the saidby stores and every score measured on them are untouched. The
#  script refuses to run if the group id or the store path looks like an existing
#  one.
#
#  COST: 8 parallel shards, ~9-13 h each, all at once. Same plan as the original.
#
#  Usage:  bash scripts/submit_speaker_free_full.sh
#          SHARDS="s3 s7" bash scripts/submit_speaker_free_full.sh   # resubmit
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail
cd "$(dirname "$0")/.."

GROUP_ID="${GROUP_ID:-finance_speaker_free}"
STORE_PREFIX="${STORE_PREFIX:-shards_speaker_free}"
FS_ROOT="${FS_ROOT:-/fscratch/abuali}"

# Hard guards. A typo here would mix two extraction configurations into one graph
# and silently destroy the comparison we just paid a GPU-day to establish.
for forbidden in gmb_finance_full gmb_finance_nostitch gmb_finance_saidby gmb_aml_full; do
  if [[ "${GROUP_ID}" == "${forbidden}" ]]; then
    echo "❌ refusing to write into the existing group '${forbidden}'."; exit 1
  fi
done
if [[ "${STORE_PREFIX}" == "shards" ]]; then
  echo "❌ refusing to reuse the original 'shards/' store tree."; exit 1
fi

echo "🚫👤 full-corpus re-ingest — SPEAKERS EXCLUDED FROM THE NODE SET"
echo "   group_id : ${GROUP_ID}      (new — nothing existing is touched)"
echo "   stores   : ${FS_ROOT}/neo4j/${STORE_PREFIX}/<shard>"
echo "   expect   : person-rooted 94 % → ~34 %,  thing→thing 0.4 % → ~60 %"
echo ""

# The shard plan covers all 6,002 windows exactly once. Window numbering is
# GLOBAL, so the partial graphs merge afterwards without collisions.
SHARD_PLAN=(
  "s1 1:752"     "s2 752:1503"   "s3 1503:2254"  "s4 2254:3005"
  "s5 3005:3756" "s6 3756:4507"  "s7 4507:5258"  "s8 5258:6003"
)
SHARDS="${SHARDS:-}"
WALLTIME="${WALLTIME:-23}"

for entry in "${SHARD_PLAN[@]}"; do
  shard="${entry%% *}"; range="${entry##* }"
  [[ -n "${SHARDS}" && " ${SHARDS} " != *" ${shard} "* ]] && continue

  echo "📤 ${shard}  windows ${range}"
  SHARD="${shard}" \
  STORE_ROOT="${FS_ROOT}/neo4j/${STORE_PREFIX}/${shard}" \
  WINDOW_RANGE="${range}" \
  GROUP_ID="${GROUP_ID}" \
  GRAPHITI_EXCLUDE_SPEAKERS=1 \
  RESUME=1 \
  CORPUS=0 \
  WINDOW=5 \
  DOMAIN=Finance \
  INGEST_ONLY=1 \
  CONTROL_BM25=0 \
  STRICT_PROMPT=1 \
    nohup bash scripts/srun_submit.sh all "speaker_free_${shard}" 8 1 96G "${WALLTIME}" \
      scripts/cluster_ingest_job.sh > "/tmp/submit_speaker_free_${shard}.log" 2>&1 &
  sleep 6
done

wait
echo ""
echo "✅ submitted — squeue -u abuali"
echo "   when all 8 finish:  GROUP_ID=${GROUP_ID} bash scripts/run_merge_shards.sh"
