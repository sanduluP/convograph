#!/usr/bin/env python3
"""
content_map_cli.py — run the content-map pipeline from the terminal.

The Streamlit UI (app.py) is the demo surface, but iterating on LAYOUT through a
browser is slow: every tweak to render_board.py means restarting the app,
re-clicking, re-waiting. This is the same orchestrator entry point with no UI in
front of it, so a layout round is one shell command and a look at preview.png.

Called by scripts/run_content_map.sh, which is what you should actually run —
it tees the whole thing into logs/ so a run is inspectable afterwards.

    python ui/content_map_cli.py --group-id gmb_finance_full --windows 2

APPENDING THE NEXT MEETING
--------------------------
    python ui/content_map_cli.py --group-id gmb_finance_full --digest \
        --episode-limit 80 --append-to ui/output/<previous run> --append-direction below

The previous board file is read, never written; the merged canvas is the new
run's board.excalidraw. With --digest and no explicit --episode-start, the start
is advanced to where the previous run's digest ended, so the new block covers
the NEXT slice of the conversation rather than the same one again.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import orchestrator  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group-id", required=True,
                    help="the Neo4j group_id to read — module 2's output, "
                         "treated as fixed input")
    ap.add_argument("--windows", type=int, default=2,
                    help="how many contiguous 5-message episodes to feed the "
                         "planner (default 2)")
    ap.add_argument("--max-facts", type=int, default=40,
                    help="cap on facts handed to the planner (default 40)")
    ap.add_argument("--digest", action="store_true",
                    help="plan from a whole-meeting DIGEST (five cypher queries) "
                         "instead of a 2-window slice")
    ap.add_argument("--episode-limit", type=int, default=0,
                    help="with --digest: digest only N episodes, to take one "
                         "meeting-sized slice out of a corpus (0 = all)")
    ap.add_argument("--episode-start", type=int, default=None,
                    help="with --digest: the first episode to digest. Default 0, "
                         "or, with --append-to, where the previous run's digest "
                         "ended (auto-advance). An explicit value always wins.")
    ap.add_argument("--append-to", default=None, metavar="RUN_DIR|BOARD",
                    help="place this meeting's block onto an existing canvas: a "
                         "previous run folder, or a board.excalidraw. That file "
                         "is never modified.")
    ap.add_argument("--append-direction", choices=["below", "right"], default="below",
                    help="where the new block goes relative to everything already "
                         "drawn (default below, which keeps a page-like shape)")
    ap.add_argument("--provider", default=None,
                    help="saia (default) or ollama — see board_plan.PROVIDERS")
    ap.add_argument("--model", default=None,
                    help="override the provider's default model")
    args = ap.parse_args()

    # Resolve the episode start BEFORE the run so the log says what will happen.
    episode_start = args.episode_start
    if episode_start is None:
        episode_start = 0
        if args.append_to and args.digest:
            base_board, base_run = orchestrator.resolve_board(args.append_to)
            base_gid = orchestrator.describe_run(base_run, base_board)["group_id"]
            nxt = orchestrator.next_episode_start(base_run)
            if base_gid != args.group_id:
                # A different group is a different corpus; where the previous
                # digest ended says nothing about where this one should start.
                print(f"⚠️  the base board is from group '{base_gid}', this run reads "
                      f"'{args.group_id}' — starting at episode 0; pass "
                      f"--episode-start to pick a slice")
            elif nxt is None:
                print("⚠️  the base run has no digest — starting at episode 0; "
                      "pass --episode-start to pick a slice")
            else:
                episode_start = nxt
                print(f"↪  episode-start auto-advanced to {nxt} "
                      f"(from {base_run}/digest/digest.json)")

    result = orchestrator.run_content_map(
        group_id=args.group_id,
        windows=args.windows,
        max_facts=args.max_facts,
        provider=args.provider,
        model=args.model,
        use_digest=args.digest,
        episode_limit=args.episode_limit,
        episode_start=episode_start,
        append_to=args.append_to,
        append_direction=args.append_direction,
        progress_cb=print,          # the shell IS the progress bar here
    )

    print()
    print("─" * 72)
    print(f"📁 run dir : {result['run_dir']}")
    print(f"🖼️  preview : {result['preview_path']}")
    print(f"🧭 model   : {result['provider']}/{result['model']}  "
          f"({result['plan_seconds']}s to plan)")
    if result.get("used_digest"):
        print(f"🎬 start   : episode {result['episode_start']}")
    else:
        print(f"🪟 windows : {result['windows']}")
    if result.get("appended_from"):
        print(f"🧷 appended: {result['append_direction']} onto "
              f"{result['appended_from']}  ({result['blocks']} meeting(s) on canvas)")
    status = "clean" if not result["validation"] else \
             f"{len(result['validation'])} problem(s)"
    print(f"✅ plan    : {status}")
    if result["layout_issues"]:
        print(f"⚠️  layout  : {len(result['layout_issues'])} issue(s)")


if __name__ == "__main__":
    main()
