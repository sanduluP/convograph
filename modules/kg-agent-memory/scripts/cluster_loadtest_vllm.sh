#!/bin/bash
# ---------------------------------------------------------------------------
# cluster_loadtest_vllm.sh — ~5-minute SMOKE: can vLLM actually serve a given
# checkpoint, and does it answer sane + emit clean JSON?
#
# WHY this exists: before burning a long baseline/ingest run on a model, prove
# the two things Graphiti actually depends on:
#   1) the checkpoint LOADS in vLLM at all (arch/quant supported), and
#   2) it returns STRUCTURED JSON reliably — Graphiti's entity/fact extraction
#      parses JSON, so a model that chats nicely but leaks prose is useless here.
#
# Pick the model with VLLM_MODEL_DIR (defaults to the vetted Qwen3-30B-Instruct):
#   VLLM_MODEL_DIR=/fscratch/abuali/models/Qwen3.6-35B-A3B-FP8 \
#     bash scripts/srun_submit.sh all vllm_loadtest 8 1 96G 1 scripts/cluster_loadtest_vllm.sh
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."          # repo root

PORT="${VLLM_PORT:-8000}"
# DEFAULT = a small 4B model ON PURPOSE. This script tests PLUMBING, not answer
# quality, and a 30B FP8 checkpoint costs 6-10 min of weight loading per attempt
# — dead time when you're just checking a tunnel or a JSON parse. The 4B loads in
# well under a minute. Pass VLLM_MODEL_DIR explicitly when you truly need the big
# model (i.e. when the run's NUMBERS matter, not just whether it runs).
MODEL_DIR="${VLLM_MODEL_DIR:-/fscratch/abuali/models/Qwen3-4B-Instruct-2507}"
SERVED_NAME="${VLLM_SERVED_NAME:-Qwen/$(basename "$MODEL_DIR")}"

echo "════════ 1/3  serving vLLM ════════"
echo "🧪 load-test target: ${MODEL_DIR}"
# serve_vllm.sh blocks until the /v1/models endpoint answers, then backgrounds.
# If the checkpoint's architecture is unsupported it exits non-zero here — which
# IS the answer we came for, so let `set -e` surface it.
VLLM_MODEL_DIR="$MODEL_DIR" VLLM_SERVED_NAME="$SERVED_NAME" bash scripts/serve_vllm.sh

echo
echo "════════ 2/3  probing the endpoint ════════"

# --- probe A: plain chat — proves generation works end to end ----------------
echo "💬 probe A — plain chat:"
curl -sf "http://localhost:${PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${SERVED_NAME}\",\"max_tokens\":64,\"temperature\":0,
       \"messages\":[{\"role\":\"user\",\"content\":\"In one sentence: what is a knowledge graph?\"}]}" \
  | python3 -c 'import json,sys; print("   ↳", json.load(sys.stdin)["choices"][0]["message"]["content"].strip())'

# --- probe B: STRUCTURED JSON — the property Graphiti actually needs ---------
# Mirrors the shape of a Graphiti extraction call: give it a snippet of meeting
# chatter and demand a strict JSON object back, no prose, no markdown fence.
echo "🧱 probe B — structured JSON extraction (what Graphiti relies on):"
curl -sf "http://localhost:${PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${SERVED_NAME}\",\"max_tokens\":256,\"temperature\":0,
       \"messages\":[
         {\"role\":\"system\",\"content\":\"Reply with ONLY a JSON object, no markdown, no commentary.\"},
         {\"role\":\"user\",\"content\":\"Extract entities as {\\\"entities\\\":[{\\\"name\\\":str,\\\"type\\\":str}]} from: 'User_2: We moved the Model Feature Engineering deadline to July 9. User_5: Agreed, Sarah will own it.'\"}]}" \
  | python3 -c '
import json,sys
raw = json.load(sys.stdin)["choices"][0]["message"]["content"].strip()
print("   ↳ raw:", raw[:300])
# Strip a ```json fence if the model added one, then try to parse. Parsing is the
# actual pass/fail signal — Graphiti would choke on anything unparseable.
body = raw
if body.startswith("```"):
    body = body.split("\n", 1)[1].rsplit("```", 1)[0]
try:
    obj = json.loads(body)
    print("   ✅ JSON parsed OK →", len(obj.get("entities", [])), "entities")
except Exception as e:
    print("   ❌ JSON PARSE FAILED:", e)
    sys.exit(1)
'

echo
echo "════════ 3/3  stopping vLLM ════════"
PIDF="logs/serve_vllm/serve_vllm.pid"
[ -f "$PIDF" ] && kill "$(cat "$PIDF")" 2>/dev/null || true
echo "✅ load-test PASSED for ${SERVED_NAME}"
