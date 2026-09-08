#!/usr/bin/env python3
"""
plot_graph_shape.py — the one figure that explains Module 2's central finding.

WHAT IT SHOWS
-------------
Two panels over the SAME knowledge graph:

  (1) A sampled subgraph, drawn with a force layout. Speaker nodes are coloured
      as a reserved status colour; everything else is a domain concept. The
      picture is a star: a handful of people in the middle, concepts hanging off
      them, almost nothing joining two concepts to each other.

  (2) The census behind that picture — what share of facts connect what.

WHY IT MATTERS
--------------
Graphiti retrieves by walking entities. If nearly every edge runs
speaker -> concept, then every path between two topics detours through a
`User_N` hub wired to thousands of things, so multi-hop retrieval has nothing to
walk. That is the measured reason six separate retrieval-side fixes all returned
null, and it is the argument for spending effort on extraction instead.

Reads the live Neo4j; writes figures/graph_shape.png.
"""
from __future__ import annotations

import os
import re
import random
import textwrap

import matplotlib
matplotlib.use("Agg")            # headless: no display on this machine
import matplotlib.pyplot as plt
import networkx as nx
from neo4j import GraphDatabase

# Reuse the SAME classifier the census uses. Defining a second, simpler rule here
# once produced a figure saying 8.7 % concept->concept while entity_quality.py
# reported 0.4 %, because the simpler rule counted sentence fragments and
# SharePoint URLs as "concepts". One classifier, one set of numbers.
from analysis.entity_quality import classify, _CONCEPT_KINDS, _PERSON_KINDS

# ── house style ───────────────────────────────────────────────────────────────
# Muted, desaturated palette. The speaker colour is deliberately a RESERVED
# status colour (this is the "problem" being shown), never a categorical hue.
C_PERSON   = "#c44e52"   # muted red   — speakers, the hubs
C_CONCEPT  = "#4c72b0"   # muted blue  — domain concepts
C_EDGE     = "#c8ccd4"
C_TEXT     = "#2f3337"
C_MUTED    = "#7a7f87"

PERSON_RE = re.compile(r"^User_\d+$")

URI  = os.getenv("NEO4J_URI", "bolt://localhost:7688")
USER = os.getenv("NEO4J_USER", "neo4j")
PWD  = os.getenv("NEO4J_PASSWORD", "graphiti123")

SAMPLE_CYPHER = """
MATCH (s:Entity)-[r:RELATES_TO]->(t:Entity)
WITH s, t, r, rand() AS x
ORDER BY x
LIMIT $limit
RETURN s.name AS src, t.name AS tgt
"""

CENSUS_CYPHER = """
MATCH (s:Entity)-[r:RELATES_TO]->(t:Entity)
RETURN s.name AS src, t.name AS tgt
"""


def kind(name: str) -> str:
    """person / concept / other — the three buckets the figure speaks in."""
    k = classify(name)
    if k in _PERSON_KINDS:
        return "person"
    if k in _CONCEPT_KINDS:
        return "concept"
    return "other"          # sentence fragments, single words, URLs, documents


def fetch(limit: int = 900):
    """Random edge sample for the picture; a full scan for the numbers.

    Drawing 111k edges would be an unreadable smear, so panel 1 shows a random
    slice and says so. Every percentage quoted comes from the full scan.
    """
    drv = GraphDatabase.driver(URI, auth=(USER, PWD))
    with drv.session() as s:
        edges = [(r["src"], r["tgt"]) for r in s.run(SAMPLE_CYPHER, limit=limit)]
        pairs = [(kind(r["src"]), kind(r["tgt"])) for r in s.run(CENSUS_CYPHER)]
        # Counted, never hardcoded: this script now runs against more than one
        # graph, and a subtitle quoting another graph's totals is a figure that
        # lies quietly.
        counts = {r["label"]: r["n"] for r in s.run(
            "MATCH (n) WHERE labels(n)[0] IN ['Entity','Episodic'] "
            "RETURN labels(n)[0] AS label, count(*) AS n")}
    drv.close()

    from collections import Counter
    c = Counter(pairs)
    total = sum(c.values())
    # Every row that carries weight, not a hand-picked four. The original list
    # showed 4 of the 9 possible (kind, kind) pairs, so on a graph where the
    # others are large the bars visibly failed to sum to the total — 24.2% of
    # facts shown out of 100% on the speaker-free graph. Anything at or above
    # 1% is drawn; the rest is honestly labelled "other".
    named = {
        ("person", "other"):    "person \u2192 fragment/doc",
        ("person", "concept"):  "person \u2192 concept",
        ("person", "person"):   "person \u2192 person",
        ("concept", "concept"): "concept \u2192 concept",
        ("concept", "other"):   "concept \u2192 fragment/doc",
        ("other", "other"):     "fragment \u2192 fragment",
        ("other", "concept"):   "fragment \u2192 concept",
        ("concept", "person"):  "concept \u2192 person",
        ("other", "person"):    "fragment \u2192 person",
    }
    rows = [(named.get(k, str(k)), n) for k, n in c.most_common()]
    shown = [(lbl, n) for lbl, n in rows if n / max(1, total) >= 0.01]
    rest = total - sum(n for _, n in shown)
    if rest > 0:
        shown.append(("other pairs", rest))

    census = {
        "facts": total,
        "person_rooted": sum(n for (a, _), n in c.items() if a == "person"),
        "concept_concept": c[("concept", "concept")],
        "entities": counts.get("Entity", 0),
        "episodes": counts.get("Episodic", 0),
        "rows": shown,
    }
    return edges, census


def main() -> None:
    random.seed(0)
    edges, census = fetch()

    G = nx.Graph()
    G.add_edges_from(edges)

    people = [n for n in G if PERSON_RE.match(n)]
    concepts = [n for n in G if not PERSON_RE.match(n)]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(15.5, 7.4), dpi=140,
        gridspec_kw={"width_ratios": [1.45, 1]},
    )
    fig.patch.set_facecolor("white")

    # ── panel 1: the shape ────────────────────────────────────────────────────
    # k pushes nodes apart; the hubs would otherwise collapse into one blob.
    pos = nx.spring_layout(G, k=0.42, iterations=60, seed=7)
    nx.draw_networkx_edges(G, pos, ax=ax1, edge_color=C_EDGE, width=0.5, alpha=0.65)
    nx.draw_networkx_nodes(G, pos, nodelist=concepts, ax=ax1, node_size=16,
                           node_color=C_CONCEPT, linewidths=0, alpha=0.85)
    # Hub size scales with degree so the disparity is visible, not just implied.
    nx.draw_networkx_nodes(
        G, pos, nodelist=people, ax=ax1,
        node_size=[70 + 5.5 * G.degree(n) for n in people],
        node_color=C_PERSON, linewidths=0.8, edgecolors="white",
    )
    # Label only the three biggest hubs, and place the text BELOW the node with a
    # white halo. Drawing all 12 labels inside the circles clipped them into
    # unreadable fragments ("ser_1", "User"), and every label reads "User_N"
    # anyway — the point is the SIZE of the hubs, not which person each one is.
    import matplotlib.patheffects as pe
    top3 = sorted(people, key=lambda n: G.degree(n), reverse=True)[:3]
    for n in top3:
        x, y = pos[n]
        ax1.text(x, y - 0.075, n, fontsize=9, fontweight="bold", color=C_PERSON,
                 ha="center", va="top",
                 path_effects=[pe.withStroke(linewidth=3, foreground="white")])
    ax1.set_axis_off()
    ax1.set_title("1  A random slice of the graph", fontsize=13.5, fontweight="bold",
                  color=C_TEXT, loc="left", pad=14)
    # Annotation sits OUTSIDE the data area, under the panel, never over the nodes.
    # Same rule as the headline: describe what is drawn, not what we expected.
    _pr = 100.0 * census["person_rooted"] / max(1, census["facts"])
    _note = (f"{len(people)} speakers (red) anchor almost everything.  "
             "Concepts rarely touch each other."
             if _pr >= 60 else
             f"{len(people)} speaker(s) appear in this sample.  "
             "Most edges now join concepts to each other.")
    ax1.text(0.5, -0.04, _note,
             transform=ax1.transAxes, ha="center", va="top",
             fontsize=10, color=C_MUTED)

    # ── panel 2: the census ───────────────────────────────────────────────────
    total = census["facts"]
    # Colour by whether a SPEAKER sits at the origin: red = person-rooted (the
    # problem), blue = a genuine concept-to-concept link (what we want more of).
    # Colour by WHAT THE BAR MEANS, not by the first word of its label. The
    # rule that matters here is "does this fact involve a speaker": red is our
    # reserved status colour and belongs to the thing the figure is about.
    # Keying on lab.startswith("concept") painted `fragment → concept` red and
    # `concept → fragment` blue, which keys nothing at all.
    def _bar_colour(lab: str) -> str:
        return C_PERSON if "person" in lab else (
            C_CONCEPT if "concept" in lab else C_MUTED)

    rows = [(lab, n, _bar_colour(lab))
            for lab, n in census["rows"]]
    labels = [f"{lab}   {n:,}  ({100*n/total:.1f} %)" for lab, n, _ in rows]
    ypos = range(len(rows))
    ax2.barh(list(ypos), [r[1] for r in rows], color=[r[2] for r in rows],
             height=0.55, alpha=0.9)
    ax2.set_yticks(list(ypos))
    ax2.set_yticklabels(labels, fontsize=11, color=C_TEXT)
    ax2.invert_yaxis()
    ax2.set_ylim(len(rows) - 0.45, -0.55)   # tighten: no dead space above/below
    ax2.set_xlim(0, max(r[1] for r in rows) * 1.16)
    ax2.set_xlabel("number of facts", fontsize=10.5, color=C_MUTED)
    ax2.tick_params(axis="x", colors=C_MUTED, labelsize=9.5)
    for sp in ("top", "right", "left"):
        ax2.spines[sp].set_visible(False)
    ax2.spines["bottom"].set_color("#dcdfe4")
    ax2.grid(axis="x", color="#eef0f3", linewidth=0.9)
    ax2.set_axisbelow(True)
    pr = 100.0 * census["person_rooted"] / total
    ax2.set_title(f"2  What a fact connects   ({pr:.1f} % start at a person)",
                  fontsize=13.5, fontweight="bold", color=C_TEXT, loc="left", pad=14)

    # ── titles: bold headline, recessive grey subtitle ────────────────────────
    # Gap specified in POINTS and converted from the figure's real height, so it
    # stays visually identical at any figure size (a hardcoded y= would not).
    fig_h_pt = fig.get_figheight() * 72.0
    top_pt, gap_pt = 20.0, 21.0
    y_title = 1.0 - top_pt / fig_h_pt
    y_sub = y_title - gap_pt / fig_h_pt
    # The headline STATES THE RESULT, so it cannot contradict the panel beneath
    # it. The original was hardcoded to "a star around its speakers" and stayed
    # that way when rendered against the speaker-free graph, where only 21% of
    # facts start at a person — a figure asserting the opposite of its own data.
    pr = 100.0 * census["person_rooted"] / max(1, census["facts"])
    cc = 100.0 * census["concept_concept"] / max(1, census["facts"])
    if pr >= 60:
        headline = "The knowledge graph is a star around its speakers"
    elif pr >= 30:
        headline = "Speakers still anchor much of the graph"
    else:
        headline = "Excluding speakers turns the star into a map of concepts"
    fig.suptitle(headline, fontsize=17, fontweight="bold", color=C_TEXT, x=0.045,
                 ha="left", y=y_title)
    sub = (f"Graphiti extraction over the GroupMemBench Finance domain — "
           f"{census['episodes']:,} episodes, "
           f"{census['entities']:,} entities, {census['facts']:,} facts. "
           f"{pr:.1f}% of facts start at a person; {cc:.1f}% join two concepts. "
           f"Panel 1 samples {len(edges)} random facts; panel 2 counts all of them.")
    fig.text(0.045, y_sub, textwrap.fill(sub, 128), fontsize=10.5,
             color=C_MUTED, ha="left", va="top")

    legend_y = y_sub - 30.0 / fig_h_pt
    # Draw the bullets in their real colours — a grey "●" keys nothing.
    fig.text(0.045, legend_y, "●", fontsize=13, color=C_PERSON, ha="left", va="top")
    fig.text(0.060, legend_y, "speaker (User_N)", fontsize=10.5, color=C_MUTED,
             ha="left", va="top")
    fig.text(0.150, legend_y, "●", fontsize=13, color=C_CONCEPT, ha="left", va="top")
    fig.text(0.165, legend_y, "domain concept", fontsize=10.5, color=C_MUTED,
             ha="left", va="top")

    fig.tight_layout(rect=[0, 0, 1, legend_y - 14.0 / fig_h_pt])
    fig.subplots_adjust(wspace=0.28)

    # Never hardcode the name: this script is now run against TWO graphs and a
    # fixed path makes the second render destroy the first (CLAUDE.md rule 6).
    out = os.getenv("SHAPE_OUT", "figures/graph_shape.png")
    # Archive any existing figure rather than silently replacing it.
    if os.path.exists(out):
        import shutil, time as _t
        arch = os.path.join("figures", "archive")
        os.makedirs(arch, exist_ok=True)
        stamp = _t.strftime("%Y-%m-%d-%H%M%S")
        base = os.path.basename(out).rsplit(".", 1)[0]
        shutil.copy2(out, os.path.join(arch, f"{base}_{stamp}.png"))
    os.makedirs("figures", exist_ok=True)
    if os.path.exists(out):                       # never overwrite a figure in place
        import shutil, datetime
        os.makedirs("figures/archive", exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
        shutil.copy(out, f"figures/archive/graph_shape_{stamp}.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"✅ wrote {out}")
    print(f"   sampled edges drawn : {G.number_of_edges():,}")
    print(f"   speakers in sample  : {len(people)}  ({', '.join(sorted(people)[:6])} …)")
    for lab, n, _ in rows:
        print(f"   {lab:<26} {n:>8,}  {100*n/total:5.1f} %")
    print(f"   {'person-rooted (any target)':<26} {census['person_rooted']:>8,}"
          f"  {100*census['person_rooted']/total:5.1f} %")


if __name__ == "__main__":
    main()
