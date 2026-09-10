#!/usr/bin/env python3
"""
generate_image.py — caption(s) -> FLUX.1-schnell -> PNG(s).

TWO MODES, and the difference is the whole point of this file:

  --caption "..."              ONE image.   Original behaviour, unchanged.
  --captions-file captions.txt MANY images, ONE model load.

WHY THE BATCH MODE EXISTS
-------------------------
FLUX.1-schnell is ~31 GB of weights (transformer 22.2 + T5-XXL 8.9 + CLIP + VAE).
Constructing the pipeline costs ~40-90 s. Generating one 1024x1024 image at the
distilled 4 steps costs ~1-2 s on an H100.

So the load dominates by a factor of ~40. The caller used to invoke this script
once per caption, which meant a 6-fact board paid the load SIX times:

    before:  6 x (~60 s load + ~2 s render)  = ~6 minutes
    after:   1 x  ~60 s load + 6 x ~2 s      = ~70 seconds

Nothing about the model or the prompt changed — only how many times we pay to
put it on the GPU. (modules/graphic-generation/README.md already listed this
batching as NOT BUILT YET.)

THE MANIFEST
------------
Batch mode writes <out-dir>/manifest.jsonl, ONE JSON RECORD PER LINE:

    index    int     0-based position, matching the input file's line order
    caption  str     the exact prompt handed to FLUX
    path     str     absolute path to the PNG that caption produced
    seed     int|null the seed used, when --seed was passed (see below)
    error    str     present ONLY on a caption that failed; `path` is then null

JSONL rather than JSON so a partial run is still readable, and so a failure on
caption 4 does not cost us captions 0-3. The caller reads this instead of
guessing which PNG is which — the previous approach diffed the output directory
before/after and took the newest file, which cannot survive concurrent runs and
silently mismatches caption to image if anything else writes there.
"""
from __future__ import annotations

import argparse
import json
import os
import time

DEFAULT_CAPTION = (
    "A hand-drawn sticky-note style icon of a calendar with a red circle on one "
    "date, graphic-recording style, white background"
)
MODEL_ID = os.getenv("FLUX_MODEL_ID", "black-forest-labs/FLUX.1-schnell")


def _load_pipe():
    """Build the FluxPipeline once and move it to the GPU.

    Split out of generate() precisely so batch mode can call it a single time.
    Imports are deliberately inside the function: --dry-run must work on a
    laptop with no torch installed, and the module-level import would break that.
    """
    import torch
    from diffusers import FluxPipeline

    pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)

    # ~36-38 GB resident in bf16 at 1024x1024. If this card is shared and
    # someone else already holds most of it, fall back to CPU offload (~22 GB
    # peak, noticeably slower) rather than dying with a CUDA OOM.
    if os.getenv("FLUX_CPU_OFFLOAD", "0") == "1":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    return pipe


def _render(pipe, caption: str, out_path: str, steps: int, seed: int | None) -> str:
    """Render ONE caption with an ALREADY-LOADED pipeline."""
    import torch

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


def _render_placeholder(caption: str, out_path: str) -> str:
    """--dry-run: write a labelled grey PNG instead of calling FLUX.

    Exists so the whole caption -> image -> board pipeline can be exercised on a
    laptop with no GPU. Without it, every wiring change to the orchestrator
    needs a 31 GB model and a remote host just to find out whether an argument
    was passed through correctly.
    """
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (512, 512), (232, 232, 232))
    draw = ImageDraw.Draw(img)
    draw.rectangle([8, 8, 503, 503], outline=(150, 150, 150), width=3)
    # Wrap by character count - good enough for a placeholder, no font metrics.
    words, line, lines = caption.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > 34:
            lines.append(line); line = w
        else:
            line = f"{line} {w}".strip()
    lines.append(line)
    draw.text((24, 24), "DRY RUN — no FLUX\n\n" + "\n".join(lines), fill=(60, 60, 60))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    img.save(out_path)
    return out_path


def generate(caption: str, out_path: str, steps: int = 4, seed: int | None = None) -> str:
    """Single-caption entry point. Kept for callers that render exactly one image."""
    return _render(_load_pipe(), caption, out_path, steps, seed)


def generate_many(captions: list[str], out_dir: str, steps: int = 4,
                  seed: int | None = None, dry_run: bool = False) -> str:
    """Render every caption with ONE model load. Returns the manifest path.

    A caption that fails does NOT abort the batch: it is recorded with an
    `error` and the run continues. Losing one pictogram should not cost the
    other five, and the board composer can simply skip a missing entry.
    """
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, "manifest.jsonl")

    pipe = None
    if not dry_run:
        t0 = time.time()
        print(f"⏳ loading {MODEL_ID} (~31 GB, this is the expensive part)...", flush=True)
        pipe = _load_pipe()
        print(f"✅ model ready in {time.time() - t0:.1f}s — "
              f"now rendering {len(captions)} caption(s)", flush=True)

    with open(manifest_path, "w") as mf:
        for i, caption in enumerate(captions):
            out_path = os.path.join(out_dir, f"image_{i:03d}.png")
            record: dict = {"index": i, "caption": caption, "seed": seed}
            t0 = time.time()
            try:
                if dry_run:
                    record["path"] = _render_placeholder(caption, out_path)
                else:
                    # Vary the seed per image, or every pictogram on a board
                    # comes out as a near-identical composition.
                    s = None if seed is None else seed + i
                    record["seed"] = s
                    record["path"] = _render(pipe, caption, out_path, steps, s)
                print(f"   [{i + 1}/{len(captions)}] {time.time() - t0:5.1f}s  "
                      f"{caption[:60]}", flush=True)
            except Exception as exc:                      # noqa: BLE001
                record["path"] = None
                record["error"] = f"{type(exc).__name__}: {exc}"
                print(f"   [{i + 1}/{len(captions)}] ❌ {record['error']}", flush=True)

            # Flush per line so a killed run still leaves a usable manifest.
            mf.write(json.dumps(record) + "\n")
            mf.flush()

    print(f"📄 manifest: {manifest_path}", flush=True)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--caption", help="Single text prompt.")
    mode.add_argument("--captions-file",
                      help="UTF-8 file, ONE caption per line. Blank lines are "
                           "skipped. Renders all of them with one model load.")
    parser.add_argument("--out", default="out.png",
                        help="Output PNG path (single-caption mode only).")
    parser.add_argument("--out-dir",
                        help="Directory for the PNGs + manifest.jsonl "
                             "(--captions-file mode only).")
    parser.add_argument("--steps", type=int, default=4,
                        help="Inference steps (schnell: 4 is standard).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Base seed. In batch mode image i uses seed+i, so "
                             "the run is reproducible without every pictogram "
                             "coming out identical.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Write labelled placeholder PNGs instead of calling "
                             "FLUX. For testing the wiring without a GPU.")
    args = parser.parse_args()

    if args.captions_file:
        if not args.out_dir:
            parser.error("--captions-file requires --out-dir")
        with open(args.captions_file, encoding="utf-8") as fh:
            captions = [ln.strip() for ln in fh if ln.strip()]
        if not captions:
            parser.error(f"{args.captions_file} contains no non-blank lines")
        generate_many(captions, args.out_dir, steps=args.steps,
                      seed=args.seed, dry_run=args.dry_run)
        return

    caption = args.caption or DEFAULT_CAPTION
    path = (_render_placeholder(caption, args.out) if args.dry_run
            else generate(caption, args.out, steps=args.steps, seed=args.seed))
    # The shell wrapper greps these two lines - keep the exact prefixes.
    print(f"caption : {caption}")
    print(f"saved   : {path}")


if __name__ == "__main__":
    main()
