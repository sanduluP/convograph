# ui — transcript in, one editable Excalidraw board out

Chains module 2 (temporal KG) and module 3 (caption → FLUX → Excalidraw) behind
one Streamlit page: paste a transcript, get back one `.excalidraw` file with
every decision (and what it superseded) laid out on a grid, ready to open and
rearrange by hand.

```
[browser mic / audio upload]──▶ ui/module1.py ──▶ modules/asr-diarization
    │                           (ffmpeg 16k mono → Docker: Sortformer +
    │                            multitalker ASR → speaker-tagged lines;
    │                            server needs NO microphone — st.audio_input
    │                            records in the browser. GPU: ~5.5 GB)
    ▼
[text box]
modules/kg-agent-memory/ui_ingest.py     (module 2's own venv)
    │  transcript -> Graphiti episode(s) -> facts (+ which got superseded)
    ▼
kg_to_caption.caption_from_fact()        (module 3, stdlib only)
    │  fact -> FLUX-ready caption, per decision
    ▼
modules/graphic-generation/scripts/run_on_unicorn.sh --caption "..."
    │  one remote FLUX call per decision (sequential — see LIMITATIONS)
    ▼
compose_board.compose()                  (module 3, needs PIL)
    │  all images + captions -> ONE scene, non-overlapping grid, time-ordered
    ▼
ui/output/board_<stamp>.excalidraw       — open in the VS Code extension or
                                            excalidraw.com; drag/resize/move
                                            freely, it's a normal scene.
```

## Run it

```bash
cd ui
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt   # once
./.venv/bin/streamlit run app.py
```

Needs, separately (see each module's own README for setup):
- `modules/kg-agent-memory/.venv` with its `requirements.txt` installed
- `modules/kg-agent-memory/.env` with `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD`
  (and `NEO4J_DATABASE` if not the driver's default — AuraDB needs this; see
  `ui_ingest.py`'s docstring for why)
- `~/.ssh/config`'s `unicorn` host set up per `modules/graphic-generation/README.md`
  (image generation still runs there, unchanged from that module's Test 1/2)

## Choosing the KG-extraction backend

Same env vars module 2's `baselines/graphiti/graphiti_retriever.py` already
uses — pointing this UI at local vs. cluster is a caller-side env change, no
code change:

```bash
# Fully local (no VPN/cluster needed) — what this was developed and tested against:
export GRAPHITI_LLM_BASE_URL=http://localhost:11434/v1
export GRAPHITI_LLM_MODEL=qwen2.5:3b-instruct
export GRAPHITI_EMBED_BASE_URL=http://localhost:11434/v1
export GRAPHITI_EMBED_MODEL=bge-m3:latest        # ollama pull bge-m3:latest

# Cluster vLLM (module 2's normal production setup, better extraction quality):
#   1. DFKI VPN on
#   2. submit the serving job: modules/kg-agent-memory/scripts/srun_submit.sh
#      (runs scripts/serve_vllm.sh — ~3-8 min to load)
#   3. tunnel from this laptop: ssh -N -L 8000:<node-from-job-log>:8000 pegasus
export GRAPHITI_LLM_BASE_URL=http://localhost:8000/v1
export GRAPHITI_LLM_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507-FP8
export GRAPHITI_EMBED_BASE_URL=http://serv-3306.kl.dfki.de:8000/v1   # needs VPN too
export GRAPHITI_EMBED_MODEL=bge-m3:latest
```

## LIMITATIONS (read before demoing)

- **Extraction quality with the local 3B model is uneven.** Verified 2026-08-31:
  a simple decision extracted cleanly (9 nodes/4 edges), but the *revision* of
  that decision — the exact "someone changed their mind" case this whole board
  concept exists to show — extracted 0 edges from the same model. The cluster's
  32B model is module 2's actual validated setup; the local path is for
  development/demos where cluster access isn't available, not a quality
  guarantee. If a board comes back with no struck-through/superseded cards, this
  is the first thing to suspect, not a bug in `ui_ingest.py`/`compose_board.py`.
- **Sequential remote image calls.** One `run_on_unicorn.sh` round-trip
  (sync + ssh + generate + copy back) per decision, not batched into one remote
  session. Fine for the handful of decisions a short meeting produces; would
  need batching for a board with many more images.
- **Audio input is a stub.** The uploader in `app.py` exists but does nothing —
  `modules/asr-diarization` has no code yet. Wiring it once module 1 exists is
  a callback change in `app.py` only.
- **No live in-browser Excalidraw canvas.** The UI writes a `.excalidraw` file
  and tells you how to open it (VS Code extension or excalidraw.com) — it does
  not embed a canvas in the Streamlit page itself.
