#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# download_embed_model.sh — fetch the EMBEDDING model onto /fscratch.
#
# WHY: Graphiti needs an embedder as well as a chat LLM. Until now embeddings came
# from serv-3306 (bge-m3), but serv-3306 is VPN-gated and returns **HTTP 403 from
# the cluster** ("interactive use only"). A self-contained SLURM job therefore has
# to serve its own embedder on the compute node.
#
# bge-m3 is the SAME model we have used all along (1024-dim), so switching to a
# local copy changes nothing about the vectors or any earlier comparison.
#
# RUN ON THE LOGIN NODE (compute nodes have restricted egress).
# Idempotent: skips the download if the model is already there.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

FS_ROOT=${FS_ROOT:-/fscratch/abuali}
MODEL_REPO=${MODEL_REPO:-BAAI/bge-m3}
MODEL_DIR="${FS_ROOT}/models/$(basename "${MODEL_REPO}")"
PY="${FS_ROOT}/venvs/groupmembench/bin/python"

mkdir -p "${FS_ROOT}/logs"
LOG="${FS_ROOT}/logs/download_embed_model.log"

{
echo "📥 [embed] repo=${MODEL_REPO} → ${MODEL_DIR}"

# Guard on the WEIGHTS, not on config.json. A config-only directory looks
# "present" but cannot load — that false positive cost a debug cycle already.
WEIGHTS_OK=0
if [[ -f "${MODEL_DIR}/pytorch_model.bin" || -f "${MODEL_DIR}/model.safetensors" ]]; then
  WEIGHTS_OK=1
fi
if [[ "${WEIGHTS_OK}" == "1" && "${FORCE:-0}" != "1" ]]; then
  echo "✅ [embed] weights already present, skipping download"
else
  # allow_patterns skips the redundant ONNX copy and the README images, but MUST
  # include *.bin: bge-m3 ships its weights as `pytorch_model.bin`, NOT safetensors
  # (filtering on *.safetensors alone silently downloads a 34 KB config-only dir
  # that then fails to load — exactly what happened on the first attempt).
  "${PY}" - <<PYEOF
from huggingface_hub import snapshot_download
p = snapshot_download(
    repo_id="${MODEL_REPO}",
    local_dir="${MODEL_DIR}",
    allow_patterns=["*.json", "*.safetensors", "*.bin", "*.txt", "*.model",
                    "sentencepiece*"],
    ignore_patterns=["onnx/*", "imgs/*"],
)
print("  downloaded to:", p)
PYEOF
  echo "✅ [embed] download complete"
fi

echo "📂 [embed] contents:"
ls -lh "${MODEL_DIR}" | head -12
du -sh "${MODEL_DIR}" | sed 's/^/   total: /'
echo "🎉 [embed] done"
} 2>&1 | tee "${LOG}"
