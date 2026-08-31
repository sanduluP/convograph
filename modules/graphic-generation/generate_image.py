#!/usr/bin/env python3
"""
generate_image.py — caption -> FLUX.1-schnell -> PNG.

Rung 0 of the image side of Module 3. Input is a caption string, hardcoded to a
DUMMY default for now. The intent is that a later caller (something reading
compressed fact headlines out of modules/kg-agent-memory's temporal KG, or the
shared schemas/image-request.schema.json contract) passes a real caption to
generate() instead of relying on the default — this file does not need to
change for that, only its caller does.

Requires a CUDA GPU (FLUX.1-schnell is a 12B-parameter model; ~24GB+ in bf16).
Not intended to run on a laptop — point it at the department GPU cluster.

Usage:
    python generate_image.py --caption "Finance Ops locks the go/no-go date" --out image.png
    python generate_image.py                      # uses DEFAULT_CAPTION, writes out.png
"""
from __future__ import annotations

import argparse
import os

DEFAULT_CAPTION = (
    "A hand-drawn sticky-note style icon of a calendar with a red circle on one "
    "date, graphic-recording marker style, white background"
)
MODEL_ID = os.getenv("FLUX_MODEL_ID", "black-forest-labs/FLUX.1-schnell")


def generate(caption: str, out_path: str, steps: int = 4, seed: int | None = None) -> str:
    """Render `caption` with FLUX.1-schnell and save it to out_path. Returns out_path."""
    import torch
    from diffusers import FluxPipeline

    pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    pipe.to("cuda")

    generator = torch.Generator("cuda").manual_seed(seed) if seed is not None else None
    image = pipe(
        caption,
        num_inference_steps=steps,   # schnell is a distilled few-step model
        guidance_scale=0.0,          # schnell does not use classifier-free guidance
        generator=generator,
    ).images[0]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    image.save(out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--caption", default=DEFAULT_CAPTION, help="Text prompt (dummy default for now).")
    parser.add_argument("--out", default="out.png", help="Output PNG path.")
    parser.add_argument("--steps", type=int, default=4, help="Inference steps (schnell: 4 is standard).")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for reproducibility.")
    args = parser.parse_args()

    path = generate(args.caption, args.out, steps=args.steps, seed=args.seed)
    print(f"caption : {args.caption}")
    print(f"saved   : {path}")


if __name__ == "__main__":
    main()
