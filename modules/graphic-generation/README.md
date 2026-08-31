# graphic-generation

Turns a temporal-KG subgraph into a visual. Two pipelines live here, both "rung 0":

## 1. Fact-card board (copied from modules/kg-agent-memory)

`analysis/build_board.py` queries a temporal KG in Neo4j directly for facts that
were later superseded (`invalid_at` set), compresses each into a 3-6 word
headline (`analysis/compress_facts.py`), and renders them as an `.excalidraw`
board — struck-through superseded cards, replacement below in solid ink. See
[`modules/kg-agent-memory/CLAUDE.md`](../kg-agent-memory/CLAUDE.md) for the two
Excalidraw style regimes (`clean` vs `graphic-recording`) and why they're kept
separate.

> This content was **copied**, not moved, from `modules/kg-agent-memory` on
> 2026-08-31, to unblock Module 3 work without disturbing that module's active
> development. It still needs deduplication/cleanup — Module 2's author should
> decide what stays there vs. here.

Run: `bash scripts/run_build_board.sh` (needs this module's own venv + a
reachable Neo4j holding the merged KG — see the script header).

## 2. Caption -> image -> Excalidraw

The other half of the "flux → excalidraw" idea: an LLM-written caption becomes
a raster image via FLUX.1-schnell, then gets embedded as an image element in a
`.excalidraw` scene, with the caption as a text element underneath.

```
caption ──▶ generate_image.py ──▶ PNG ──▶ image_to_excalidraw.py ──▶ .excalidraw
(dummy for now, KG-derived later)   (FLUX.1-schnell, needs a CUDA GPU)
```

- `generate_image.py` — caption in, PNG out. Caption is `DEFAULT_CAPTION` unless
  `--caption` is passed; wiring a real caption from the KG (e.g. a compressed
  fact headline, or a caption written against
  [`schemas/image-request.schema.json`](../../schemas/image-request.schema.json))
  is a caller change, not a change to this file.
- `image_to_excalidraw.py` — wraps one PNG + its caption into a minimal, valid
  `.excalidraw` scene (base64-embedded image element + text element).
- `scripts/run_generate_image.sh` — runs both steps back to back, timestamped
  output under `output/`. **Test 1**: caption is either `DEFAULT_CAPTION` or a
  hand-typed `--caption`.

### Test 2 — KG-grounded caption instead of a hand-typed one

`kg_to_caption.py` is the caller change the note above pointed at: it reads a
temporal-KG subgraph (`schemas/temporal-kg.schema.json`), finds a decision
node, and asks a small local LLM (ollama, `qwen2.5:3b-instruct` — same
short-output-model choice as `modules/kg-agent-memory/analysis/compress_facts.py`,
and the same reason: a reasoning/hybrid build narrates instead of answering,
see that module's `CLAUDE.md`) to turn the decision into a FLUX-ready caption.
Output is one `ImageRequest` object (`schemas/image-request.schema.json`) —
caption plus the node ids it's grounded in, for traceability.

No live Neo4j / module-2 KG export (nodes+edges) was reachable while wiring
this, but module 2's own eval corpus was, so the default is **real data, not
invented data**: `samples/real_kg_finance_aml.json` reshapes three real
messages — a decision and the one that later superseded it — from
[`modules/kg-agent-memory/data/final/Finance/synthetic_domain_channels_rolevariants_Finance.json`](../kg-agent-memory/data/final/Finance/synthetic_domain_channels_rolevariants_Finance.json)
(project "AML (Anti-Money Laundering) Project") into `temporal-kg.schema.json`
shape; content/authors/roles/timestamps are copied verbatim. A fully
hand-written `samples/sample_kg.json` is kept as an offline-safe fallback.
Point `--kg` at a real Neo4j-sourced export once one exists — nothing
downstream changes.

```
KG subgraph ──▶ kg_to_caption.py ──▶ ImageRequest ──▶ generate_image.py ──▶ ... (same as Test 1)
(samples/real_kg_finance_aml.json)   (local ollama, qwen2.5:3b-instruct)   .caption field
```

```bash
bash scripts/run_kg_to_image.sh                        # samples/real_kg_finance_aml.json (default)
bash scripts/run_kg_to_image.sh --kg samples/sample_kg.json     # offline-safe hand-written fallback
bash scripts/run_kg_to_image.sh --kg path/to/real_kg.json       # a real Neo4j export, once reachable
```

This runs caption generation locally (no GPU needed) then calls
`scripts/run_on_unicorn.sh` under the hood — **Test 1's caption -> FLUX ->
Excalidraw pipeline, reused unchanged.** Output is separately timestamped
(`kg_request_<stamp>.json`, `image_<stamp>.png`, `image_<stamp>.excalidraw`),
so it never overwrites a Test 1 run.

Requires a CUDA GPU (FLUX.1-schnell is a 12B-parameter model) — this laptop
(M3 Pro, 18GB, no CUDA) can't run it. Runs instead on **unicorn**, a
department GPU host (`serv-7101.kl.dfki.de`) reached directly over SSH — not
behind SLURM like module 2's Pegasus cluster, so there's no `srun_submit.sh`
equivalent, just sync + ssh + run.

### Running on unicorn

Add to `~/.ssh/config` once:

```
Host unicorn
  HostName serv-7101.kl.dfki.de
  User sandulu
```

One-time env setup (installs torch/diffusers into this module's own venv,
mirroring module 2's "keep torch out of my env" isolation, in reverse):

```bash
bash scripts/sync_to_cluster.sh
ssh unicorn
cd /scratch/mpatil/sandulu/convograph-graphic-generation
bash scripts/setup_remote_env.sh
```

Then, from the laptop, one command does sync + run + copy the result back:

```bash
bash scripts/run_on_unicorn.sh
bash scripts/run_on_unicorn.sh --caption "Finance Ops locks go/no-go"
```

Output lands in `output/` both on unicorn and locally after copy-back.

## Not built yet

- Actually running `generate_image.py` against the real model — written and
  reviewed, not yet executed (no GPU available until unicorn is set up).
- Placing a generated image *inside* the fact-card board (same canvas) rather
  than in its own separate `.excalidraw` file.
- Wiring `kg_to_caption.py` to a real Neo4j-sourced node+edge export instead
  of `samples/real_kg_finance_aml.json` (which is real module-2 *message* data,
  reshaped by hand — not a live query result; waiting on a reachable Neo4j).
- The MCP bridge from an LLM client (GPT/Claude) to Excalidraw for interactive,
  incremental edits — everything above is a batch script, not an agent loop.
- A reusable style/image library for reproducible visuals across renders.
- Real-time, partial-region updates (the actual "editable graphic recording"
  end goal) — today everything is a one-shot full render.

## Status

Board rendering: copied from module 2, functionally working there. Caption→image
(Test 1): ran end-to-end on unicorn 2026-08-31, hand-typed caption. KG→caption
(Test 2): `kg_to_caption.py` tested end-to-end against `samples/sample_kg.json`
and local ollama; not yet run through to a FLUX image (that's just re-running
Test 1's already-verified path with a KG-derived caption). Image→Excalidraw:
implemented and tested with a synthetic image, and with a real FLUX output.
