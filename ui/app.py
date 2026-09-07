#!/usr/bin/env python3
"""
app.py — Streamlit front end: transcript in, one editable Excalidraw board out,
rendered live in this same page.

    streamlit run ui/app.py

Input is meant to be optional between audio and text, but module 1
(modules/asr-diarization) is a scaffold with no code yet (see its README) —
so the audio uploader below is shown but NOT wired to the pipeline. Only the
text box feeds orchestrator.run_pipeline(). This is a UI-shape decision, not
an oversight: wiring audio later is a callback change in this file only, once
module 1 exists.

THE LIVE CANVAS
----------------
Excalidraw ships a browser-ready ESM build specifically for embedding without
a bundler (see excalidraw/excalidraw's examples/with-script-in-browser on
GitHub) — loaded here via esm.sh with pinned React/ReactDOM deps, mounted
inside an iframe via st.components.v1.html. This has NOT been visually
verified in a real browser session (no browser tooling in the environment
this was built in) — if the canvas comes back blank, the "open externally"
fallback right below it still works regardless, and the browser console will
show whatever the CDN/mount step failed on.
"""
from __future__ import annotations

import json
import os
import sys

import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from orchestrator import PipelineError, run_content_map, run_pipeline  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "modules", "graphic-generation"))
import board_plan  # noqa: E402  (for the model picker's options)

EXCALIDRAW_VERSION = "0.18.0"
REACT_VERSION = "18.3.1"

EXAMPLE_TRANSCRIPT = (
    "User_9 (Business Analyst): Yes, lock it into the spec today. Compliance can "
    "add the threshold and escalation assumptions, and Data and Ops can flag any "
    "alert-population or routing dependency that would change build. Treat "
    "anything unresolved as open, not final. Confirm by Friday.\n"
    "User_2 (Client Services Lead): Agreed, routing-only channels can stay out of "
    "build-lock, but alert-volume movers need tighter treatment.\n"
    "User_9 (Business Analyst): Quick update, after the SME review we need to "
    "revisit the build-lock split for one payment scenario. The original "
    "build-lock-now, monitor-only-by-Friday decision no longer works. Compliance "
    "confirmed one high-risk payment scenario is required for the current "
    "regulatory reporting pack. Change: pull that channel into build-lock now. "
    "Compliance owns the finalized assumptions, and Data and Ops update the "
    "alert-population dependencies in the spec."
)

st.set_page_config(page_title="Convograph", page_icon="🗂️", layout="wide")


def render_excalidraw(scene: dict, height: int = 640) -> None:
    """Mount a live, editable Excalidraw canvas inline via esm.sh — the same
    CDN-embed approach Excalidraw's own repo documents for bundler-free use."""
    payload = json.dumps(scene).replace("</", "<\\/")
    html = f"""
    <style>
      html, body {{ margin: 0; height: 100%; background: #fff; }}
      #excalidraw-root {{ height: {height}px; width: 100%; }}
    </style>
    <div id="excalidraw-root"></div>
    <script>
      window.EXCALIDRAW_ASSET_PATH =
        "https://esm.sh/@excalidraw/excalidraw@{EXCALIDRAW_VERSION}/dist/prod/";
    </script>
    <link rel="stylesheet"
      href="https://unpkg.com/@excalidraw/excalidraw@{EXCALIDRAW_VERSION}/dist/prod/index.css" />
    <script type="module">
      import React from "https://esm.sh/react@{REACT_VERSION}";
      import {{ createRoot }} from "https://esm.sh/react-dom@{REACT_VERSION}/client";
      import {{ Excalidraw }} from
        "https://esm.sh/@excalidraw/excalidraw@{EXCALIDRAW_VERSION}?deps=react@{REACT_VERSION},react-dom@{REACT_VERSION}";

      const sceneData = {payload};
      const root = createRoot(document.getElementById("excalidraw-root"));
      root.render(React.createElement(Excalidraw, {{ initialData: sceneData }}));
    </script>
    """
    components.html(html, height=height, scrolling=False)


# ── sidebar: settings ────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    # Two board shapes, and they are genuinely different pipelines - not a
    # display option. See orchestrator.run_content_map's header comment.
    #
    #   Content map   a WINDOW of the conversation -> one LLM plan -> 2-4 nodes
    #                 joined by labelled arrows. Words are canvas text, drawings
    #                 are wordless. This is the graphic recording.
    #   Image grid    one fact -> one caption -> one image, laid out in a grid.
    #                 The original shape, kept as the thing to compare against.
    board_style = st.radio(
        "Board style",
        ["🧭 Content map", "🖼️ Image grid (original)"],
        help="A content map draws relations between ideas. The image grid draws "
             "one picture per fact with no relations — it is the baseline.",
    )
    is_map = board_style.startswith("🧭")

    if is_map:
        # Named for module 2's windowing on purpose: one window is one 5-message
        # episode today, and becomes "one meeting" once module 1 lands.
        windows = st.slider(
            "Windows fed to the planner", min_value=1, max_value=8, value=2,
            help="How many CONTIGUOUS 5-message episodes the planner sees. One "
                 "fact alone carries no context; a window does. The run with the "
                 "most superseded facts is chosen.",
        )
        max_facts = st.slider(
            "Max facts handed to the planner", min_value=5, max_value=120, value=40,
            help="Fact edges are reused across episodes, so 2 windows can pull "
                 "back 150+ facts without a cap. Superseded ones come first.",
        )

        # The planner model is an EXPERIMENT VARIABLE, so it belongs in the UI.
        # Measured 2026-09-06 on one window / 40 facts: the local 4B produced an
        # invalid link index and a reused fact; qwen3-30b was clean in 2.5 s;
        # gpt-oss-120b was clean in 28.6 s.
        MODEL_CHOICES = {
            "qwen3-30b-a3b-instruct-2507 (SAIA, default)":
                ("saia", "qwen3-30b-a3b-instruct-2507"),
            "openai-gpt-oss-120b (SAIA, slower)":
                ("saia", "openai-gpt-oss-120b"),
            "qwen3:4b-instruct (unicorn, no API key)":
                ("ollama", "qwen3:4b-instruct"),
        }
        choice = st.selectbox(
            "Planner model", list(MODEL_CHOICES),
            help="SAIA needs SAIA_API_KEY in a git-ignored .env. The unicorn "
                 "option needs only the tunnel on port 11435.",
        )
        provider, planner_model = MODEL_CHOICES[choice]
        columns = 3          # unused by the content map, kept for the call site
    else:
        provider = planner_model = None
        windows = 2
        max_facts = st.slider(
            "Max decisions to render", min_value=1, max_value=12, value=6,
            help="Caps how many decisions from the KG get turned into images. "
                 "Superseded/revised decisions are prioritized first.",
        )
        columns = st.slider(
            "Grid columns", min_value=1, max_value=6, value=3,
            help="How many images per row on the composed canvas.",
        )

    st.divider()
    # Two DIFFERENT models, routinely confused, so name both and say when each
    # one actually runs:
    #
    #   planner     picked above. Plans the board from a window. Always runs.
    #   Graphiti    builds the temporal KG. Runs ONLY on the "new transcript"
    #               path — reading an existing graph is cypher, no LLM at all.
    #
    # The extraction backend used to be shown unconditionally, which implied the
    # default path depends on it. It does not.
    if is_map:
        st.caption("Board planner")
        st.code(f"{board_plan.PROVIDERS[provider]['base_url']}\n{planner_model}",
                language=None)

    st.caption("KG extraction (Graphiti) — only on the *new transcript* path")
    st.code(
        os.environ.get("GRAPHITI_LLM_BASE_URL", "http://localhost:11435/v1")
        + "\n" + os.environ.get("GRAPHITI_LLM_MODEL", "qwen3:4b-instruct"),
        language=None,
    )
    st.caption(
        "⚠️ Not how the benchmark graphs were built. `gmb_finance_full` and the "
        "speaker-free shards were extracted on **Pegasus** by vLLM serving "
        "`Qwen3-30B-A3B-Instruct-2507-FP8` "
        "(`modules/kg-agent-memory/scripts/cluster_ingest_job.sh`). The model "
        "above is a laptop convenience for ad-hoc extraction only — a 4B is not "
        "good enough to build a graph worth measuring. Reading an existing "
        "graph calls no LLM at all."
    )

# ── header ────────────────────────────────────────────────────────────────────
st.title("Convograph")
st.markdown("##### Meeting → knowledge graph → graphic-recording board")
st.caption(
    "Text goes through module 2 (temporal KG) → module 3 (caption → FLUX) → "
    "one Excalidraw canvas, rendered live below — drag, resize, and reposition "
    "anything on it."
)

# ── inputs ────────────────────────────────────────────────────────────────────
with st.expander("🎙️ Meeting audio", expanded=False):
    audio_file = st.file_uploader(
        "Upload audio (not wired yet)", type=["wav", "mp3", "m4a"], key="audio",
    )
    if audio_file is not None:
        st.warning(
            "modules/asr-diarization is a scaffold — there's no transcription "
            "backend yet, so this file won't be used. Paste the transcript as "
            "text below instead."
        )

st.subheader("📝 Transcript")
if "transcript" not in st.session_state:
    st.session_state.transcript = ""

# ── where the facts come from ─────────────────────────────────────────────────
# Two sources, and the second is the one module 3 was always meant to have.
#
#   Existing graph  a cypher query against a KG module 2 already built. Seconds,
#                   and it is the real contract between the modules.
#   New transcript  runs module 2's extraction first. Minutes, and a
#                   benchmark-quality graph needs the 30B model on the cluster,
#                   not the 4B we serve for the UI.
#
# Existing is the DEFAULT because iterating on module 3 does not require
# re-deriving module 2's output every time.
source = st.radio(
    "Where should the facts come from?",
    ["🗄️ Existing knowledge graph", "📝 New transcript (runs extraction)"],
    horizontal=True,
    help="Module 3's real input is a cypher query over an existing temporal KG. "
         "Extraction is only needed when the graph does not exist yet.",
)
use_existing = source.startswith("🗄️")

if use_existing:
    existing_group_id = st.text_input(
        "group_id in Neo4j",
        value="gmb_finance_full",
        help="Facts the conversation later OVERTURNED are ranked first — those "
             "are the ones worth drawing.",
    )
    st.caption("No extraction. The query ranks superseded facts first, so "
               "'max decisions' selects rather than truncates.")
    text = ""
    ready = bool(existing_group_id.strip())
else:
    existing_group_id = None
    btn_col, count_col = st.columns([1, 4])
    with btn_col:
        if st.button("Load example"):
            st.session_state.transcript = EXAMPLE_TRANSCRIPT

    text = st.text_area(
        "Paste the meeting transcript here",
        height=200,
        key="transcript",
        placeholder="User_9 (Business Analyst): Yes, lock it into the spec today...",
    )
    with count_col:
        words = len(text.split())
        st.caption(f"{words} words" + ("" if words else " — paste a transcript or load the example"))
    ready = bool(text.strip())
    st.caption("⏳ Extraction is the slow stage — Graphiti makes ~10-20 LLM calls "
               "per episode.")

if st.button("Generate board", type="primary", disabled=not ready):
    with st.status("Running the pipeline...", expanded=True) as status:
        lines: list[str] = []
        log_area = st.empty()

        def _progress(msg: str) -> None:
            lines.append(msg)
            log_area.code("\n".join(lines), language=None)

        try:
            if is_map:
                result = run_content_map(
                    group_id=(existing_group_id or "") if use_existing else "",
                    text="" if use_existing else text,
                    windows=int(windows),
                    max_facts=int(max_facts),
                    provider=provider,
                    model=planner_model,
                    progress_cb=_progress,
                )
            else:
                result = run_pipeline(
                    text,
                    max_facts=int(max_facts),
                    columns=int(columns),
                    progress_cb=_progress,
                    existing_group_id=existing_group_id if use_existing else None,
                )
            status.update(label="Done", state="complete", expanded=False)
            st.session_state.last_result = result
        except PipelineError as exc:
            status.update(label="Failed", state="error", expanded=True)
            st.error(str(exc))

# ── output ────────────────────────────────────────────────────────────────────
if st.session_state.get("last_result"):
    result = st.session_state.last_result
    st.subheader("🗂️ Board")

    # A content-map result carries a plan; a grid result carries entries. The
    # tab is named for whichever it is rather than showing an empty pane.
    plan = result.get("plan")
    tab_canvas, tab_details = st.tabs(
        ["Canvas", "The plan" if plan else "Decisions used"])
    with tab_canvas:
        if plan:
            st.caption(
                f"{result['provider']}/{result['model']} · "
                f"{result['windows']} window(s) · planned in "
                f"{result['plan_seconds']}s · "
                + ("plan validates clean" if not result["validation"]
                   else f"⚠️ {len(result['validation'])} plan problem(s)")
            )
            st.caption(f"📁 `{result['run_dir']}`")
        with open(result["board_path"]) as fh:
            scene = json.load(fh)
        render_excalidraw(scene)
        with st.expander("If the canvas above is blank"):
            st.markdown(
                f"The live embed loads Excalidraw from a CDN in your browser — if "
                f"that failed to load (offline, blocked script, etc.), open the "
                f"file directly instead:\n\n"
                f"```\ncode {result['board_path']}\n```\n\n"
                f"or drag it into [excalidraw.com](https://excalidraw.com)."
            )

    with tab_details:
        if plan:
            # Show what the LLM decided, and — the part that makes it traceable
            # — which facts each anchor came from.
            st.markdown(f"### {plan.get('title', '')}")
            facts = plan.get("_facts", [])
            for i, anchor in enumerate(plan.get("anchors", [])):
                st.markdown(f"**{i}. {anchor.get('label', '')}**")
                st.caption(f"glyph sent to FLUX: *{anchor.get('glyph', '')}*")
                for fi in anchor.get("from_facts", []):
                    if isinstance(fi, int) and 0 <= fi < len(facts):
                        st.write(f"– [{fi}] {facts[fi]['fact']}")
                st.divider()
            if plan.get("links"):
                st.markdown("**Links**")
                for l in plan["links"]:
                    st.write(f"{l.get('from')} → {l.get('to')} · *{l.get('label','')}*")
            st.caption(f"{len(plan.get('dropped', []))} fact(s) dropped as not "
                       f"worth drawing")
            for problem in result.get("validation", []):
                st.warning(problem)
        else:
            for i, entry in enumerate(result["entries"], 1):
                st.markdown(f"**{entry.get('label', i)}**")
                st.write(entry["caption"])
                st.caption(entry.get("timestamp") or "no timestamp")
                st.divider()
