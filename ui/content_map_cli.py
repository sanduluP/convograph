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
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import orchestrator  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--group-id", required=True,
                    help="the Neo4j group_id to read — module 2's output, "
                         "treated as fixed input")
    ap.add_argument("--windows", type=int, default=2,
                    help="how many contiguous 5-message episodes to feed the "
                         "planner (default 2)")
    ap.add_argument("--max-facts", type=int, default=40,
                    help="cap on facts handed to the planner (default 40)")
    ap.add_argument("--provider", default=None,
                    help="saia (default) or ollama — see board_plan.PROVIDERS")
    ap.add_argument("--model", default=None,
                    help="override the provider's default model")
    args = ap.parse_args()

    result = orchestrator.run_content_map(
        group_id=args.group_id,
        windows=args.windows,
        max_facts=args.max_facts,
        provider=args.provider,
        model=args.model,
        progress_cb=print,          # the shell IS the progress bar here
    )

    print()
    print("─" * 72)
    print(f"📁 run dir : {result['run_dir']}")
    print(f"🖼️  preview : {result['preview_path']}")
    print(f"🧭 model   : {result['provider']}/{result['model']}  "
          f"({result['plan_seconds']}s to plan)")
    print(f"🪟 windows : {result['windows']}")
    status = "clean" if not result["validation"] else \
             f"{len(result['validation'])} problem(s)"
    print(f"✅ plan    : {status}")
    if result["layout_issues"]:
        print(f"⚠️  layout  : {len(result['layout_issues'])} issue(s)")


if __name__ == "__main__":
    main()
