# convograph

A pipeline for turning multi-party spoken conversation into a temporal knowledge graph and an agent memory system (AMS), with an optional path to real-time visual output (graphic recording).

<p align="center">
  <img src="docs/figures/graphic-recording-teaser.png" alt="Four people talk; the conversation becomes a transcript, then a temporal knowledge graph, then a drawn canvas">
</p>

<p align="center">
  <sub><b>Where we are heading.</b> Four people talk. Diarization turns the audio into a
  speaker-tagged transcript; a temporal knowledge graph accumulates what was said and
  what has since been revised; a visual stage renders it onto a canvas that keeps
  updating as the meeting goes on — a <i>graphic recording</i>, produced live.<br>
  This is the target picture, not the current state: modules 1 (Sortformer
  diarization + multitalker transcription) and 2 are built, module 3 is in
  progress. FLUX.1-schnell and Excalidraw remain candidates under evaluation,
  not decisions.</sub></p>

## Setup

Everything heavy is hosted. You need **no GPU**, and you do not need to remember
any SSH incantation.

### 1. `~/.ssh/config`

Two services run on unicorn (DFKI's shared DSA box) and bind to `127.0.0.1`, so
they are reached through an SSH tunnel. Add this **with your own DFKI username**,
and be on the **DFKI VPN**:

```
Host unicorn
  HostName serv-7101.kl.dfki.de
  User <your-dfki-username>
  IdentityFile ~/.ssh/id_ed25519
```

### 2. Secrets

Copy the git-ignored `.env` into `modules/kg-agent-memory/` (ask Faris). It holds
the AuraDB connection and `SAIA_API_KEY`. **Never commit it** — a published key
gets auto-revoked and breaks the pipeline for everyone at once.

Apply for your own SAIA key at [academiccloud.de](https://academiccloud.de)
(federated login → RPTU, add Sebastian Vollmer / dsa.dfki.de). Approval can take
a week, so do it before you need it. Keys expire every 6 months.

### 3. Run it

**Two venvs**, and they are separate on purpose: module 2 (graphiti-core, neo4j)
and the UI have conflicting heavy pins, so the UI shells out to module 2's
interpreter rather than importing across them.

```bash
# the UI itself (streamlit + Pillow)
python3 -m venv ui/.venv
ui/.venv/bin/pip install -r ui/requirements.txt

# module 2 — the UI calls this interpreter to query the graph
python3 -m venv modules/kg-agent-memory/.venv
modules/kg-agent-memory/.venv/bin/pip install -r modules/kg-agent-memory/requirements.txt
```

Then either entry point:

```bash
bash scripts/run_content_map.sh --group-id gmb_finance_full --windows 2   # CLI
ui/.venv/bin/streamlit run ui/app.py                                      # UI
```

No venv is needed for module 3 locally — FLUX runs on unicorn.

**The tunnels open themselves.** Both entry points call `scripts/tunnels.sh`
before doing anything, so the normal path is that you never think about them.
(In the UI this happens when you press **Generate board**, not at startup.)

## Where things run, and what needs a tunnel

| what | where | tunnel? |
|---|---|---|
| Graphiti extraction LLM | SAIA `qwen3-30b-a3b-instruct-2507` | no — HTTPS API |
| Board planner | SAIA, same model | no — HTTPS API |
| Embedder `bge-m3` | unicorn `:11435` | **yes** |
| FLUX.1-schnell | unicorn `:8500` | **yes** |
| Neo4j | AuraDB | no — public endpoint |

The FLUX server is **shared** — one instance serves the whole team. The *tunnel*
is per-laptop: it is your key to that shared door, so each person opens their own.

### Health check

```bash
bash scripts/tunnels.sh status     # the only command worth remembering
bash scripts/tunnels.sh start      # idempotent, safe any time
bash scripts/tunnels.sh stop
```

```
🔎 Convograph tunnels via 'unicorn'
   ✅ FLUX      localhost:8500   (NVIDIA H100 PCIe)
   ✅ embedder  localhost:11435  (bge-m3)
   ℹ️  SAIA (all LLM work) and AuraDB need no tunnel.
```

By hand, if you prefer — both also open in a browser tab:

```bash
curl -s http://localhost:8500/health      # {"status":"warm","device":"NVIDIA H100 PCIe",...}
curl -s http://localhost:11435/api/tags   # JSON listing bge-m3
```

> **Do not use `ss -ltn` to check a tunnel.** A dead tunnel keeps its local port
> **bound**, so `ss` reports it listening while every request hangs. `tunnels.sh`
> probes the services instead, and tells the two cases apart: `not forwarded`
> versus `PORT BOUND BUT DEAD` (which it repairs automatically).

### Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `No usable 'unicorn' host` | missing SSH config | add the block above |
| ssh fails immediately | VPN down | connect to the DFKI VPN |
| tunnel up, service silent | the server on unicorn is down | `tunnels.sh` prints the restart command |
| Neo4j hostname will not resolve | AuraDB paused after ~3 days idle | resume at [console.neo4j.io](https://console.neo4j.io); data survives |
| extraction fails with empty `HTTP 500` | too many concurrent SAIA calls | `SEMAPHORE_LIMIT=4` (graphiti-core defaults to 20) |

Running *on* unicorn itself? Set `CONVOGRAPH_NO_TUNNEL=1` — the services are
already local.

## Research question

How well can domain adaptation from agent memory be done from multi-party human conversation? Concretely: given what person X said over N weeks of conversation, filter the relevant subgraph and use it to adapt/ground an agent's memory — and quantify how well that adaptation works.

## Pipeline

```
audio (4 speakers)
  -> [1] ASR + diarization  ---------------->  speaker-tagged transcript
  -> [2] temporal KG + agent memory system --> temporal KG, quantitative ablation results
  -> [3] graphic generation ----------------->  caption -> image -> editable canvas (future)
```

Modules are independent and communicate through the data contracts in [`schemas/`](schemas/). See [`docs/architecture.md`](docs/architecture.md) for the full picture.

## Modules

| Module | Path | Status | Description |
|---|---|---|---|
| ASR + diarization | [`modules/asr-diarization`](modules/asr-diarization) | **working** | Streaming 4-speaker diarization + speaker-attributed transcription (NVIDIA Streaming Sortformer + multitalker Parakeet, DGX GB10), offline + live mic, DER-scored on AMI-derived 2/3/4-speaker mixes. RTF ~0.06 → real-time capable. Exports schema-valid transcripts. |
| Temporal KG + agent memory | [`modules/kg-agent-memory`](modules/kg-agent-memory) | **working, measured** | Transcript → temporal KG (Graphiti + Neo4j) with a bi-temporal `invalid_at` layer, evaluated on GroupMemBench. Full Finance domain ingested: 5,810 episodes / 111,258 facts / 18,450 superseded. Browsable live — see the module README. |
| Graphic generation | [`modules/graphic-generation`](modules/graphic-generation) | **working end to end** | A window of the KG -> an LLM plans the whole board -> FLUX.1-schnell draws one wordless pictogram per anchor -> rendered as a **content map** (concepts as nodes, relations as labelled arrows) on an Excalidraw canvas. ~19 s per board against a warm FLUX server. Words are canvas text and drawings carry none — FLUX renders letters as gibberish, so the split is by construction. |

## Adding a new module

See [`scripts/import_module.sh`](scripts/import_module.sh) — handles both "here's a zip" and "here's a git repo" cases. Each module should stay self-contained (own README, own dependency file, own run scripts) and document how its inputs/outputs map to [`schemas/`](schemas/).

## Status

Early stage, modules land independently before full end-to-end assembly. The first inter-module hop (1 -> 2) is wired and verified — run `python3 pipeline/check_handoff.py` (no GPU/LLM needed). A unifying UI is planned once the pipeline stabilizes (see [`pipeline/`](pipeline)).

## License

MIT — see [`LICENSE`](LICENSE).
