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
from orchestrator import (  # noqa: E402
    PipelineError, list_groups, list_previous_boards, next_episode_start,
    run_content_map, run_pipeline,
)

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
      // A canvas holding several meetings no longer fits the iframe at the
      // default viewport; ask Excalidraw to frame whatever is drawn.
      sceneData.scrollToContent = true;
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
        # Defaults are the measured-good settings, so the auto-run needs no
        # input at all; the knobs stay one click away for ablations.
        with st.expander("Advanced", expanded=False):
            # WHAT THE PLANNER SEES — the choice that matters most on this page.
            #
            #   window  2 episodes, ~40 facts, picked by POSITION in the graph.
            #           The board describes whichever five minutes it was handed.
            #   digest  five cypher queries over the whole meeting, picked by what
            #           the graph says MATTERED. Same prompt size (8.8 KB against
            #           9.2 KB), far more coverage.
            #
            # ON by default. The page auto-draws on load, before anyone can touch
            # a control, so an off-by-default toggle would mean the FIRST board —
            # the one people actually look at — is always the window one.
            use_digest = st.checkbox(
                "🧪 Plan from a whole-meeting digest",
                value=True,
                help="Five cypher queries — what changed, what it was about, who "
                     "was there, what was settled, what is still open — instead of "
                     "a two-window slice. Adds ~4 s.",
            )
            episode_limit = 0
            if use_digest:
                # A group can be a whole CORPUS. gmb_finance_full is six weeks of
                # several parallel projects, and digesting all of it summarises a
                # corpus rather than a meeting — which reads impressive and means
                # nothing. The cap is how you take one meeting-sized slice.
                episode_limit = st.slider(
                    "Episodes to digest", min_value=0, max_value=400, value=80,
                    step=20,
                    help="0 = the whole group. Use a cap when the group is a "
                         "corpus rather than one meeting.",
                )
                st.caption("The window sliders below are ignored in digest mode.")

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
        # The grid path has no planner and no digest, but run_key reads these
        # unconditionally — leaving them undefined here is a NameError the
        # moment someone picks the other board style.
        provider = planner_model = None
        use_digest = False
        episode_limit = 0
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
    # The graphs come FROM THE DATABASE, not from a hardcoded default. There
    # used to be one graph and a text box pre-filled with its name; every UI
    # ingest adds a ui_<stamp> group and module 1 will add meetings, so the
    # list has to be discovered. Ranked most-superseded-first by the backend,
    # and the FIRST entry is what auto-runs on page load, so that order is the
    # default board. Cached: this hits AuraDB, and Streamlit reruns this whole
    # script on every widget change.
    @st.cache_data(ttl=300, show_spinner="Listing graphs in Neo4j...")
    def _cached_groups() -> list[dict]:
        return list_groups()

    try:
        groups = _cached_groups()
    except PipelineError as exc:
        groups = []
        st.error(f"Could not list graphs in Neo4j:\n\n{exc}")

    if groups:
        # One tab per graph, drawn as a segmented control rather than st.tabs
        # ON PURPOSE: Streamlit executes the body of EVERY st.tabs pane on each
        # rerun, so real tabs would draw every graph's board on page load - N
        # planner calls and 4N FLUX images. A segmented control looks the same
        # but tells us which one is selected, so exactly one board is drawn:
        # the first (most superseded) by default, another only when clicked.
        gids = [g["group_id"] for g in groups]
        stats = {g["group_id"]: f"{g['episodes']} episodes · {g['facts']} facts · "
                                f"{g['superseded']} superseded" for g in groups}
        existing_group_id = st.segmented_control(
            "Knowledge graph", gids, default=gids[0], selection_mode="single",
            label_visibility="collapsed",
            help="Every group_id in Neo4j, most superseded facts first. The "
                 "first is drawn on load; click another to draw it.",
        ) or gids[0]     # deselecting everything falls back to the default
        st.caption(stats[existing_group_id])
        # A group_id is not a description. finance_speaker_free is SIX WEEKS of
        # several parallel projects; treasury_prod_deploy_* is one phase of one
        # channel. Whether the board summarises "a meeting" depends entirely on
        # which, and the name does not say.
        SCOPE = {
            "finance_speaker_free":
                "⚠️ the whole Finance corpus — 6 weeks, several parallel projects. "
                "Not one meeting. Cap the episodes in Advanced to take a slice.",
            "gmb_finance_full":
                "⚠️ the whole Finance corpus, ingested WITH speakers — "
                "concept→concept is only 6.6%, so there is little to draw arrows from.",
        }
        note = SCOPE.get(existing_group_id)
        if note:
            st.caption(note)
        elif existing_group_id.startswith("treasury_"):
            st.caption("✅ one phase of one channel — this is a meeting-sized unit.")
    else:
        # Fallback so a Neo4j hiccup does not leave the page with no controls.
        existing_group_id = st.text_input("group_id in Neo4j", value="gmb_finance_full")
    text = ""
    ready = bool((existing_group_id or "").strip())
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

# ── append onto a previous board ──────────────────────────────────────────────
# The renderer places one meeting's block clear of everything already drawn
# (render_board.append_scene). The "previous board" is a FILE from an earlier
# run: the embed below is one-way, so a card dragged in the browser is not
# seen here. The source file is never modified; the merged canvas lands in
# the new run's folder.
append_on = False
append_to = None
append_direction = "below"
episode_start = 0
if is_map:
    append_on = st.checkbox(
        "🧷 Append to a previous board",
        help="Place this meeting's block below or beside an earlier board instead "
             "of starting a fresh canvas. The earlier board file is left untouched.",
    )
    if append_on:
        boards = list_previous_boards()
        if not boards:
            st.info("No previous boards under ui/output yet — generate one first.")
            append_on = False
        else:
            by_dir = {b["run_dir"]: b for b in boards}
            dirs = list(by_dir)
            last = (st.session_state.get("last_result") or {}).get("run_dir")
            idx = dirs.index(last) if last in by_dir else 0
            a_col, d_col = st.columns([3, 1])
            with a_col:
                append_to = st.selectbox(
                    "Previous board", dirs, index=idx,
                    format_func=lambda d: f"{os.path.basename(d)}  ·  {by_dir[d]['label']}",
                )
            with d_col:
                append_direction = st.radio(
                    "Place the new meeting", ["below", "right"], horizontal=True,
                    help="Below keeps a page-like shape; right makes a timeline strip.",
                )
            if use_digest:
                # Advancing the episode start only makes sense when this board
                # digests the SAME graph the previous one did. A pasted
                # transcript becomes a brand-new group, and a different existing
                # group is a different corpus; both start at 0.
                base_gid = by_dir[append_to].get("group_id")
                same_graph = bool(use_existing and base_gid
                                  and existing_group_id == base_gid)
                suggested = next_episode_start(append_to) if same_graph else None
                episode_start = int(st.number_input(
                    "Start at episode", min_value=0, value=int(suggested or 0),
                    help="Prefilled from the previous run's digest.json as "
                         "episode_start + episodes_digested, so this board covers "
                         "the NEXT slice of the conversation, not the same one again.",
                ))
                if not same_graph:
                    st.caption(f"Different graph from the previous board "
                               f"({base_gid or 'unknown'}), so this one starts at "
                               f"episode 0 — a new transcript is a new group.")
                elif suggested is None:
                    st.caption("The previous run had no digest, so there is nothing "
                               "to advance from — starting at episode 0.")

# ── when to run ───────────────────────────────────────────────────────────────
# ONE BUTTON, both paths. Drawing on page load was tried and is worse: the run
# fires before anyone can touch a control, so every setting in Advanced - which
# graph, digest or window, how many episodes - could only ever affect the SECOND
# board. The first board, the one people actually look at and judge, was always
# built from defaults nobody chose. It also spends SAIA calls and GPU time on
# every page refresh, whether or not anyone wanted a board.
#
# The input tuple still guards against Streamlit's rerun-on-every-widget: the
# same inputs never run the pipeline twice, so touching a slider after a run
# does not silently redraw.
run_key = (
    is_map, use_existing,
    existing_group_id if use_existing else text,
    int(windows), int(max_facts), provider, planner_model, int(columns),
    bool(use_digest), int(episode_limit),
    bool(append_on), append_to, append_direction, int(episode_start),
)

clicked = st.button("🖍️ Generate board", type="primary", disabled=not ready)
# A click on inputs that already produced this exact board is a no-op rather
# than a re-run; the result is still on screen below.
should_run = clicked and st.session_state.get("last_key") != run_key
if clicked and not should_run:
    st.info("This board is already drawn below — change a setting to draw a new one.")

if should_run:
    with st.status("Drawing the board...", expanded=True) as status:
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
                    use_digest=bool(use_digest),
                    episode_limit=int(episode_limit),
                    episode_start=int(episode_start),
                    append_to=append_to if append_on else None,
                    append_direction=append_direction,
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
            st.session_state.last_key = run_key
            st.session_state.pop("last_failed_key", None)
        except PipelineError as exc:
            status.update(label="Failed", state="error", expanded=True)
            st.error(str(exc))
            # Remember the failure so this exact input does not auto-fire again
            # on the next widget touch; the last good board stays on screen.
            st.session_state.last_key = run_key
            st.session_state.last_failed_key = run_key

# ── output ────────────────────────────────────────────────────────────────────
if st.session_state.get("last_result"):
    result = st.session_state.last_result
    st.subheader("🗂️ Board")

    # Canvas only. The planner's decisions (which facts each node came from,
    # the glyph sent to FLUX, what was dropped, validation warnings) used to be
    # a second tab here; they are research-side detail, and every run already
    # writes the same information to plan.json in its run directory, so the
    # page shows the board and points at the folder.
    plan = result.get("plan")
    if plan:
        st.caption(
            f"{result['provider']}/{result['model']} · "
            f"{result['windows']} window(s) · planned in "
            f"{result['plan_seconds']}s · "
            + ("plan validates clean" if not result["validation"]
               else f"⚠️ {len(result['validation'])} plan problem(s)")
            + f" · 📁 `{result['run_dir']}`"
            + (f" · 🧷 appended {result['append_direction']} onto "
               f"`{os.path.basename(os.path.dirname(result['appended_from']))}` "
               f"({result.get('blocks', '?')} meetings)"
               if result.get("appended_from") else "")
        )
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
