#!/usr/bin/env python3
"""
serve_flux.py — keep FLUX.1-schnell resident in VRAM behind a tiny HTTP API.

WHY A SERVER AND NOT JUST THE BATCH SCRIPT
------------------------------------------
Batching every caption of a board into one process (generate_image.py
--captions-file) already removed the worst waste: the ~31 GB pipeline is built
once per BOARD instead of once per FACT. This removes the last one — the
pipeline is built once per PROCESS LIFETIME instead of once per board.

    per fact   (original)   6 facts = ~6 min      6 loads
    per board  (batched)    6 facts = ~70 s       1 load
    per server (this file)  6 facts = ~10 s       0 loads (already warm)

WHY NOT vLLM
------------
vLLM is built for autoregressive decoding: paged KV-cache, continuous batching
of a token-generation loop. FLUX is a rectified-flow diffusion transformer —
no KV cache, no token loop, a fixed 4 denoising steps. Different architecture
class; vLLM does not serve diffusion models at all. Hence ~200 lines of FastAPI
rather than a serving framework.

API
---
    GET  /health   -> {"status","model","device","vram_gb","dtype"}
    POST /generate -> {"captions": [str, ...], "steps": 4, "seed": null}
                      returns {"images": [{"index","caption","png_b64","seed",
                                           "seconds"} | {"index","error"}]}

PNGs come back base64 in the JSON rather than as files on disk, because the
caller is on a laptop and the GPU is on another host: returning bytes over the
(tunnelled) HTTP connection avoids needing a shared filesystem or a second
rsync hop entirely.

RUN IT
------
    bash scripts/serve_flux.sh            # on the GPU host
    ssh -N -L 8500:localhost:8500 unicorn # from the laptop
    FLUX_SERVER_URL=http://localhost:8500 ...   # ui/orchestrator.py picks it up
"""
from __future__ import annotations

import base64
import io
import os
import threading
import time
from typing import Optional

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_ID = os.getenv("FLUX_MODEL_ID", "black-forest-labs/FLUX.1-schnell")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load the pipeline BEFORE the server accepts its first connection.

    This has to be eager. The point of the process is to be warm, and a lazy
    load makes "is it ready?" unanswerable: /health would report 200 the instant
    uvicorn binds, while the GPU is still empty, so the launcher polls for a
    warm status that nothing will ever produce until a real /generate arrives.
    (Observed exactly that: /health 200 OK repeatedly, nvidia-smi flat at the
    other tenant's 8 GB.)

    Loading here instead means: connection refused while loading, then warm.
    A failure to load kills the process, which the launcher already detects.
    """
    if os.getenv("FLUX_LAZY_LOAD", "0") != "1":
        _get_pipe()
    yield


app = FastAPI(title="FLUX.1-schnell", version="1.0", lifespan=lifespan)

_pipe = None
# Diffusers pipelines are NOT thread-safe and one H100 can only usefully run one
# denoise at a time anyway. Serialising here means two browser tabs hitting the
# UI produce a queue rather than a CUDA crash or interleaved garbage.
_lock = threading.Lock()


def _get_pipe():
    """Build the pipeline on first use and keep it. Guarded by the same lock so
    two simultaneous first-requests cannot both try to load 31 GB."""
    global _pipe
    if _pipe is None:
        import torch
        from diffusers import FluxPipeline

        t0 = time.time()
        print(f"⏳ loading {MODEL_ID} (~31 GB)...", flush=True)
        pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
        # ~36-38 GB resident at 1024x1024. This card is SHARED; if someone else
        # already holds most of it, FLUX_CPU_OFFLOAD=1 drops the peak to ~22 GB
        # at the cost of speed, rather than dying with a CUDA OOM.
        if os.getenv("FLUX_CPU_OFFLOAD", "0") == "1":
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
        _pipe = pipe
        print(f"✅ warm in {time.time() - t0:.1f}s", flush=True)
    return _pipe


class GenerateRequest(BaseModel):
    captions: list[str] = Field(..., min_length=1)
    steps: int = 4          # schnell is distilled; 4 is the standard setting
    seed: Optional[int] = None


@app.get("/health")
def health() -> dict:
    """Liveness probe. Does not touch the pipeline itself.

    Under the default eager load this only ever answers once the model is
    resident, so "status": "warm" is a genuine readiness signal rather than
    "uvicorn has bound a port"."""
    import torch

    return {
        "status": "warm" if _pipe is not None else "loading",
        "model": MODEL_ID,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
        if torch.cuda.is_available() else None,
        "dtype": "bfloat16",
    }


@app.post("/generate")
def generate(req: GenerateRequest) -> dict:
    """Render every caption. One failure does not sink the rest of the board."""
    import torch

    pipe = _get_pipe()
    out = []
    with _lock:
        for i, caption in enumerate(req.captions):
            t0 = time.time()
            try:
                # Vary the seed per image, or every pictogram on one board comes
                # back as a near-identical composition.
                s = None if req.seed is None else req.seed + i
                gen = torch.Generator("cuda").manual_seed(s) if s is not None else None
                image = pipe(
                    caption,
                    num_inference_steps=req.steps,
                    guidance_scale=0.0,   # schnell does not use CFG
                    generator=gen,
                ).images[0]

                buf = io.BytesIO()
                image.save(buf, format="PNG")
                out.append({
                    "index": i,
                    "caption": caption,
                    "seed": s,
                    "seconds": round(time.time() - t0, 2),
                    "png_b64": base64.b64encode(buf.getvalue()).decode("ascii"),
                })
                print(f"   [{i + 1}/{len(req.captions)}] {time.time() - t0:5.1f}s  "
                      f"{caption[:60]}", flush=True)
            except Exception as exc:                      # noqa: BLE001
                out.append({"index": i, "caption": caption,
                            "error": f"{type(exc).__name__}: {exc}"})
                print(f"   [{i + 1}/{len(req.captions)}] ❌ {exc}", flush=True)
    return {"images": out}


if __name__ == "__main__":
    import uvicorn

    # Bind to loopback only. Reached from the laptop through an ssh tunnel, the
    # same way module 2 reaches the cluster vLLM — this box is shared, and an
    # unauthenticated GPU endpoint should not be on its network interface.
    uvicorn.run(app, host=os.getenv("FLUX_HOST", "127.0.0.1"),
                port=int(os.getenv("FLUX_PORT", "8500")), log_level="info")
