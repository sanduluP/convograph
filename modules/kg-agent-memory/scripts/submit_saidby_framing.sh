#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  submit_saidby_framing.sh — re-extract the WHOLE Finance domain with the speaker
#  demoted out of subject position, to test whether that removes the star shape.
#
#  THE HYPOTHESIS
#  --------------
#  Our merged KG is a star around the 12 speakers: 94.1 % of 111,258 facts
#  originate at a person, only 0.4 % link two domain concepts, and the 12
#  highest-degree nodes ARE the 12 users (`User_13`: degree 17,814).
#
#  That is not a Graphiti defect. `_episode_body()` renders every window as
#      User_1: we should freeze v1 evidence fields now
#  so the speaker is the grammatical subject of every sentence in the corpus, and
#  `extract_edges` — correctly following its own instruction to anchor each detail
#  on a second entity — reaches for the speaker every time.
#
#  This run changes that ONE line to
#      [said by User_1] we should freeze v1 evidence fields now
#  and changes nothing else.
#
#  WHY THE WHOLE DOMAIN AND NOT ONE CHANNEL
#  ----------------------------------------
#  A single channel WOULD be enough to measure the graph SHAPE (person-rooted %,
#  concept→concept %) — those are Cypher censuses that need no questions. It would
#  NOT be enough to produce a SCORE: the 214 questions are spread across all six
#  channels, so a one-channel corpus caps the answerable set at a fraction of them.
#  That is exactly the trap that voided the 2026-08-03 AML-only run, where the
#  ceiling turned out to be 3 of 32 questions. Since we want both the shape AND a
#  comparable score, we re-extract everything.
#
#  WHY FINANCE AND NOT A FRESH DOMAIN
#  ----------------------------------
#  Finance is the only domain with a complete baseline (all 6 categories, Graphiti
#  + BM25 + BFS, measured 2026-08-12/13). A new domain would have nothing to
#  compare against. Same corpus, same questions, ONE variable changed.
#
#  ⚠️ IT WRITES TO A NEW GROUP — THE EXISTING GRAPH IS NOT TOUCHED
#  ---------------------------------------------------------------
#  GROUP_ID defaults to `gmb_finance_saidby`, so the 40-hour `gmb_finance_full`
#  graph and every score measured on it stay exactly where they are.
#
#  COST: 8 shards × ~9-13 h, running concurrently. Same plan as the original run.
#
#  Usage:
#    bash scripts/submit_saidby_framing.sh
#    SHARDS="s3 s7" bash scripts/submit_saidby_framing.sh    # re-submit failures
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

GROUP_ID="${GROUP_ID:-gmb_finance_saidby}"

# Guard: refuse to run against the baseline group. Re-extracting into
# `gmb_finance_full` with a different framing would irreversibly mix two
# extraction configurations inside one graph and destroy the comparison.
if [[ "${GROUP_ID}" == "gmb_finance_full" ]]; then
  echo "❌ refusing to write into the baseline group 'gmb_finance_full'."
  echo "   That graph is the control for this experiment. Pick another GROUP_ID."
  exit 1
fi

echo "🔀 re-extracting Finance with SAIDBY framing (speaker out of subject position)"
echo "   group     : ${GROUP_ID}   (baseline gmb_finance_full is untouched)"
echo "   framing   : saidby        →  [said by User_1] <text>"
echo "   baseline  : speaker       →  User_1: <text>"
echo ""
echo "   Expect: person-rooted share ↓ from 94.1 %, concept→concept ↑ from 0.4 %."
echo "   Those two numbers are the actual result — the score comes after."
echo ""

GRAPHITI_SPEAKER_FRAMING=saidby \
GROUP_ID="${GROUP_ID}" \
JOB_PREFIX=gmbsaidby \
SHARDS="${SHARDS:-}" \
  bash scripts/submit_shards.sh

echo ""
echo "📊 when the shards finish:"
echo "   1. merge  : GROUP_ID=${GROUP_ID} bash scripts/run_merge_shards.sh"
echo "   2. census : compare person-rooted % and concept→concept % vs the baseline"
echo "   3. score  : STORE=<merged> GROUP_ID=${GROUP_ID} bash scripts/submit_qtypes.sh"
