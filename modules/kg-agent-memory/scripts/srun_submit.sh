#!/bin/bash
# srun_submit.sh — Submit a NON-interactive job to a Pegasus compute node
# (background, logs to /fscratch/abuali/logs). Use for long jobs like serving vLLM.
#
# Usage:
#   bash scripts/srun_submit.sh PARTITION JOB_NAME CPUS GPUS MEM HOURS SCRIPT [args...]
# Example (serve the model, grabbing whichever FP8-safe GPU frees first):
#   bash scripts/srun_submit.sh all serve_qwen 8 1 96G 8 scripts/serve_vllm.sh
#
# ─────────────────────────────────────────────────────────────────────────────
# PARTITION = "all"  → broad submit across a COMMA-LIST of partitions so SLURM
# grabs whichever node frees up first (no waiting in one congested queue).
#
# ‼️  This project serves FP8 (e4m3) models. Per Faris's rule 8, dynamic-activation
#     FP8 silently GARBLES on Ampere (A100 / RTXA6000) and Volta (V100). So our
#     "all" is a CURATED **FP8-safe** list — every Ada/Hopper/Blackwell partition,
#     Ampere/Volta EXCLUDED. This keeps "submit broad" while staying correct.
#     (If you ever run a NON-FP8 job here, pass an explicit partition instead.)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

if [ "$#" -lt 7 ]; then
  echo "Usage: $0 PARTITION JOB_NAME CPUS GPUS MEM HOURS SCRIPT [args...]"
  echo "Example: $0 all serve_qwen 8 1 96G 8 scripts/serve_vllm.sh"
  exit 1
fi

PARTITION=$1; JOB_NAME=$2; CPUS=$3; GPUS=$4; MEM=$5; HOURS=$6; TASK_SCRIPT=$7
shift 7
TASK_ARGS="$@"

[ -f "$TASK_SCRIPT" ] || { echo "Error: task script not found: $TASK_SCRIPT"; exit 1; }

# "all" → every FP8-safe (Ada/Hopper/Blackwell) partition. L40S-DSA first: it's
# our dedicated DSA allocation, so lowest contention. Ampere/Volta deliberately absent.
FP8_SAFE_PARTITIONS="L40S-DSA,L40S,L40S-AV,H100,H100-RP,H100-PCI,H200,H200-PCI,B200"
if [[ "$PARTITION" == "all" ]]; then
  PARTITION="$FP8_SAFE_PARTITIONS"
  echo "🌐 partition=all → FP8-safe list: $PARTITION"
fi

# Normalize memory ('G' / 'GB' / bare → 'G').
if   [[ $MEM == *GB ]]; then MEM="${MEM%GB}G"
elif [[ $MEM != *G  ]]; then MEM="${MEM}G"; fi

WORKDIR=$(pwd)
LOG_DIR="/fscratch/abuali/logs"; mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${JOB_NAME}_$(date +%Y%m%d_%H%M%S).log"

NVIDIA_CONTAINER_VERSION=25.02
CONTAINER="/netscratch/enroot/nvcr.io_nvidia_pytorch_${NVIDIA_CONTAINER_VERSION}-py3.sqsh"
[ -f "$CONTAINER" ] || { echo "🚨 Container not found: $CONTAINER"; exit 1; }

GPU_FLAGS=""
if [[ "${GPUS}" != "0" && -n "${GPUS}" ]]; then GPU_FLAGS="--gres=gpu:${GPUS}"; fi

# ⏰ Optional deferred start (BEGIN=…). Empty = start ASAP.
BEGIN="${BEGIN:-}"

echo "🚢 Container: $CONTAINER"
echo "💻 $PARTITION | job=$JOB_NAME cpus=$CPUS gpus=$GPUS mem=$MEM time=${HOURS}h"
echo "📜 Script: $TASK_SCRIPT $TASK_ARGS"
echo "📄 Log:    $LOG_FILE"
[[ -n "$BEGIN" ]] && echo "⏰ Begin:   $BEGIN  (job stays PENDING until then)"

nohup srun -K \
  -p "$PARTITION" \
  --job-name "$JOB_NAME" \
  --nodes 1 \
  --ntasks-per-node 1 \
  --cpus-per-task "$CPUS" \
  ${GPU_FLAGS} \
  --mem "$MEM" \
  ${BEGIN:+--begin="$BEGIN"} \
  --time="${HOURS}:00:00" \
  --chdir "$WORKDIR" \
  --container-image "$CONTAINER" \
  --container-mounts /home:/home,/netscratch:/netscratch,/ds:/ds,/fscratch:/fscratch \
  --container-workdir "$WORKDIR" \
  bash "$TASK_SCRIPT" $TASK_ARGS \
  > "$LOG_FILE" 2>&1 &

echo "Started background srun job. Monitor with:"
echo "  squeue -u abuali        # (PD = pending/waiting for the begin time or resources)"
echo "  tail -f $LOG_FILE"
