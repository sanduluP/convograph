#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  submit_speaker_exclusion_smoke.sh — a 45-minute A/B on the star-graph problem.
#
#  THE QUESTION
#  ------------
#  94.1 % of the facts in our full-Finance graph start at a person, and only 0.4 %
#  join two domain concepts. For graphic recording that is fatal: the lines
#  BETWEEN concepts are what a graphic recording is made of, and we have 433 of
#  them out of 111,258.
#
#  Does dropping speakers from the node set fix the shape — and at what cost?
#
#  WHY A SMOKE TEST IS ENOUGH
#  --------------------------
#  Shape is a RATIO, not a total. It is stable long before the corpus is. Sixty
#  windows yield on the order of a thousand facts, which pins the percentages
#  well enough to decide whether a full re-ingest is worth a night of GPU. There
#  is no reason to spend 38 GPU-hours to learn something visible in 45 minutes.
#
#  THE TWO VARIANTS — identical except for one flag
#  ------------------------------------------------
#    baseline           what we run today
#    speaker_excluded   entity_types={"Speaker": ...} + excluded_entity_types
#
#  Both ingest THE SAME 60 windows, so the comparison is controlled.
#
#  ⚠️ NOTHING EXISTING IS TOUCHED
#  ------------------------------
#  Each variant gets its OWN store under neo4j/smoke/<name> and its OWN group_id.
#  The 40-hour `gmb_finance_full` graph, the merged store and every score measured
#  on them are untouched and unreadable from here. Nothing is deleted, nothing is
#  appended to an existing database.
#
#  WHAT TO LOOK AT AFTERWARDS
#  --------------------------
#    person-rooted %     94.1 % today  → want DOWN
#    concept -> concept   0.4 % today  → want UP
#    facts per episode                 → want ROUGHLY HELD
#  That third one is the one that can kill the idea: if the extractor cannot find
#  two concepts it may simply emit fewer facts, and we would have traded a star
#  for a sparse graph. A shape win bought with a recall collapse is not a win.
#
#  Usage:  bash scripts/submit_speaker_exclusion_smoke.sh
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT_DIR="$(pwd)"

WINDOWS="${WINDOWS:-60}"          # 60 windows = 300 messages ≈ 1,000+ facts
WALLTIME="${WALLTIME:-4}"
FS_ROOT="${FS_ROOT:-/fscratch/abuali}"

# Names are spelled out on purpose: `squeue` shows a wide NAME column and a store
# directory is read months later. "spkexcl_s1" saves six characters and costs
# minutes of guessing.
declare -A VARIANTS=(
  ["baseline"]=0
  ["speaker_excluded"]=1
)

echo "🔬 speaker-exclusion smoke test"
echo "   windows per variant : ${WINDOWS}  (1:$((WINDOWS + 1)))"
echo "   variants            : ${!VARIANTS[*]}"
echo "   walltime            : ${WALLTIME} h"
echo ""
echo "   ⚠️  writes ONLY to ${FS_ROOT}/neo4j/smoke/<variant>"
echo "      existing stores (gmb_finance_full, merged/*) are NOT touched"
echo ""

for variant in "${!VARIANTS[@]}"; do
  exclude="${VARIANTS[$variant]}"
  group="finance_smoke_${variant}"
  store="${FS_ROOT}/neo4j/smoke/${variant}"

  # Refuse to write anywhere near the real graphs.
  case "${store}" in
    *"/neo4j/smoke/"*) : ;;
    *) echo "❌ refusing to write outside neo4j/smoke: ${store}"; exit 1 ;;
  esac

  echo "📤 ${variant}   group_id=${group}"
  echo "     exclude speakers : ${exclude}"
  echo "     store            : ${store}"

  STORE_ROOT="${store}" \
  GROUP_ID="${group}" \
  WINDOW_RANGE="1:$((WINDOWS + 1))" \
  GRAPHITI_EXCLUDE_SPEAKERS="${exclude}" \
  RESUME=1 \
  CORPUS=0 \
  WINDOW=5 \
  DOMAIN=Finance \
  INGEST_ONLY=1 \
  CONTROL_BM25=0 \
  STRICT_PROMPT=1 \
    nohup bash scripts/srun_submit.sh all "smoke_${variant}" 8 1 96G "${WALLTIME}" \
      scripts/cluster_ingest_job.sh > "/tmp/submit_smoke_${variant}.log" 2>&1 &
  sleep 4
done

wait
echo ""
for variant in "${!VARIANTS[@]}"; do
  echo "──────── ${variant} ────────"
  grep -E "Log:|job=" "/tmp/submit_smoke_${variant}.log" 2>/dev/null || true
done
echo ""
echo "✅ submitted — compare with: bash scripts/run_speaker_exclusion_census.sh"
