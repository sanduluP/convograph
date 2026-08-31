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
from orchestrator import PipelineError, run_pipeline  # noqa: E402

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
    st.caption("KG-extraction backend")
    st.code(
        os.environ.get("GRAPHITI_LLM_BASE_URL", "http://localhost:11434/v1")
        + "\n" + os.environ.get("GRAPHITI_LLM_MODEL", "qwen2.5:3b-instruct"),
        language=None,
    )
    st.caption(
        "Set via GRAPHITI_LLM_BASE_URL / GRAPHITI_LLM_MODEL env vars before "
        "launching — see ui/README.md for the local-vs-cluster tradeoff."
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

if st.button("Generate board", type="primary", disabled=not text.strip()):
    steps = ["Ingesting into the temporal KG", "Generating captions + images", "Composing the canvas"]
    with st.status("Running the pipeline...", expanded=True) as status:
        lines: list[str] = []
        log_area = st.empty()

        def _progress(msg: str) -> None:
            lines.append(msg)
            log_area.code("\n".join(lines), language=None)

        try:
            result = run_pipeline(
                text, max_facts=int(max_facts), columns=int(columns), progress_cb=_progress,
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

    tab_canvas, tab_details = st.tabs(["Canvas", "Decisions used"])
    with tab_canvas:
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
        for i, entry in enumerate(result["entries"], 1):
            st.markdown(f"**{entry.get('label', i)}**")
            st.write(entry["caption"])
            st.caption(entry.get("timestamp") or "no timestamp")
            st.divider()
