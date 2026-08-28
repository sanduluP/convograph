#!/bin/bash
# ---------------------------------------------------------------------------
# cluster_serve_only.sh — serve vLLM on a compute node and STAY UP for the whole
# job walltime, so a laptop can reach it through an SSH tunnel:
#
#   ssh -N -L 8000:<node>:8000 pegasus
#   # then locally: GRAPHITI_LLM_BASE_URL=http://localhost:8000/v1
#
# WHY a separate script: cluster_baseline_job.sh serves, runs its eval, then
# exits — and when the job's main process exits SLURM tears down the node,
# killing vLLM with it. Here we serve and then block, keeping the endpoint alive
# for an interactive/local client to drive.
#
# Submit (wide FP8-safe partitions so it schedules fast, rule 8):
#   bash scripts/srun_submit.sh all gmb_serve 8 1 96G 8 scripts/cluster_serve_only.sh
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."

echo "════════ serving vLLM (stay-alive mode) ════════"
bash scripts/serve_vllm.sh

# Print the tunnel command with the REAL node name filled in, so it can be
# copy-pasted straight out of the job log.
NODE="$(hostname)"
PORT="${VLLM_PORT:-8000}"
echo
echo "🔌 TUNNEL FROM YOUR LAPTOP:"
echo "   ssh -N -L ${PORT}:${NODE}:${PORT} pegasus"
echo "🧪 THEN VERIFY LOCALLY:"
echo "   curl -s http://localhost:${PORT}/v1/models"
echo

# Block until SLURM reclaims the node at walltime. `wait` would return
# immediately (vLLM is nohup'd, not a child), so sleep in a loop and emit a
# heartbeat so the log shows the endpoint is still alive.
while true; do
  sleep 300
  if curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "💓 $(date '+%H:%M:%S') vLLM still serving on ${NODE}:${PORT}"
  else
    echo "❌ $(date '+%H:%M:%S') vLLM stopped responding — exiting job"
    exit 1
  fi
done
