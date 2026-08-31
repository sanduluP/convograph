#!/usr/bin/env python3
"""
find_reversals.py — locate, in any domain, where a meeting CHANGED ITS MIND.

THE IDEA (Faris's, and it is better than grepping for keywords)
---------------------------------------------------------------
GroupMemBench ships a `knowledge_update` question set whose gold answers state
what the CURRENT approach is — and most of them phrase it as a contrast with what
came before: "Instead of keeping the split as-is …", "rather than parked
separately", "the original split is no longer safe".

That makes the question set a **human-curated index of every supersession in the
corpus**. Rather than hunting for reversals with regexes over 30,000 messages, we
take each answer that states a before→after, BM25 it back against the messages,
and read off where the evidence sits.

Two things fall out per question:
  • the CHANNEL it belongs to  → which channels are contentious
  • the WINDOW SPREAD of its evidence → how tightly the reversal is localised,
    i.e. how usable it is as a short demo slice

Output: a ranking of channels by how much genuine disagreement they contain, and
the tightest individual reversals, ready to extract as a transcript.
"""
from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rank_bm25 import BM25Okapi

from baselines.rag_common.eval_lib import load_conversation_messages, message_index_text
from baselines.graphiti.graphiti_retriever import _build_windows

_TOK = re.compile(r"[A-Za-z0-9_]+")
_tok = lambda t: [w.lower() for w in _TOK.findall(t or "")]

# Phrases that mark an answer as describing a CHANGE rather than a steady state.
# Deliberately conservative: a false positive costs us a bad demo slice.
REVERSAL = re.compile(
    r"(?i)\b(instead of|rather than|no longer|replaces?|superseded?|"
    r"moved from|changed from|revert|reopen|abandon|drop the)\b"
)


def scan(domain: str, window: int = 5, top_k: int = 10):
    conv = f"data/final/{domain}/synthetic_domain_channels_rolevariants_{domain}.json"
    qfile = f"questions/{domain}/knowledge_update.jsonl"
    if not (os.path.exists(conv) and os.path.exists(qfile)):
        return None

    msgs = load_conversation_messages(conv)
    wins = _build_windows(msgs, window)
    idx2win = {i: n for n, w in enumerate(wins, start=1) for i in w}
    bm25 = BM25Okapi([_tok(message_index_text(m)) for m in msgs])

    rows = [json.loads(l) for l in open(qfile) if l.strip()]
    reversals = [r for r in rows if REVERSAL.search(str(r.get("answer", "")))]

    found = []
    for r in reversals:
        scores = bm25.get_scores(_tok(f"{r['question']} {r.get('answer','')}"))
        top = sorted(range(len(msgs)), key=lambda i: scores[i], reverse=True)[:top_k]
        w = sorted(idx2win[i] for i in top)
        # Channel by majority of the evidence, not by the single top hit — one
        # stray high-scoring message should not decide attribution.
        chans = [msgs[i].get("_channel", "?") for i in top]
        channel = max(set(chans), key=chans.count)
        found.append({"id": r["id"], "channel": channel,
                      "w_lo": w[0], "w_hi": w[-1], "spread": w[-1] - w[0],
                      "answer": " ".join(str(r.get("answer", "")).split())})
    return {"domain": domain, "questions": len(rows),
            "reversals": len(reversals), "found": found}


def main() -> None:
    domains = sys.argv[1:] or ["Finance", "Technology", "Healthcare", "Manufacturing"]
    all_found = []
    print(f"{'domain':<16}{'questions':>10}{'with a reversal':>18}")
    print("-" * 46)
    for d in domains:
        res = scan(d)
        if not res:
            print(f"{d:<16}{'—  no data':>28}")
            continue
        pct = 100 * res["reversals"] / max(1, res["questions"])
        print(f"{d:<16}{res['questions']:>10}{res['reversals']:>13}  ({pct:.0f} %)")
        for f in res["found"]:
            f["domain"] = d
        all_found += res["found"]

    print(f"\n{'':-<92}\nMOST CONTENTIOUS CHANNELS  (most reversals located in them)\n")
    by_ch = {}
    for f in all_found:
        by_ch.setdefault((f["domain"], f["channel"]), []).append(f)
    for (d, ch), fs in sorted(by_ch.items(), key=lambda kv: -len(kv[1]))[:8]:
        print(f"  {len(fs):>2} reversals   {d:<14} {ch[:44]}")

    print(f"\n{'':-<92}\nTIGHTEST INDIVIDUAL REVERSALS  (best demo slices — small window spread)\n")
    for f in sorted(all_found, key=lambda x: x["spread"])[:8]:
        print(f"  {f['domain']:<14}{f['id']:<22}w{f['w_lo']}–{f['w_hi']}  "
              f"(spread {f['spread']})")
        print(f"      {f['channel'][:56]}")
        print(f"      {f['answer'][:100]}…\n")


if __name__ == "__main__":
    main()
