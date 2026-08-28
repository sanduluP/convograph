#!/usr/bin/env python3
"""
build_serpentine_diagrams.py — produce the landscape v2 of both journey diagrams.

The row plans below are the ONLY editorial decision here; everything else (box
text, grey notes, colours) is carried over verbatim from v1. Rows are chosen so
that a logical stage never straddles a wrap:

  EXTRACTION   row 0  prepare the input
               row 1  inside graphiti.add_episode()   ← the dashed band
               row 2  persist and merge

  RETRIEVAL    row 0  question → search
               row 1  rank, map back, hand over the top-10
               row 2  answer, judge, score
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.serpentine_layout import relayout

relayout(
    src="excalidraw/extraction-journey.excalidraw",
    dst="excalidraw/extraction-journey-v2.excalidraw",
    rows=[
        ["d1", "a1", "d2"],                     # corpus → windows → one window
        ["a2", "d3", "a3", "d4", "a4"],         # the add_episode band
        ["d5", "d6", "a5", "d7"],               # episode → many → merge → TKG
    ],
    note_of={
        "d2": "note-window", "d3": "note-entities", "d4": "note-facts",
        "a4": "note-resolve", "d5": "note-episode", "d6": "note-shards",
        "d7": "note-out",
    },
    group={"row": 1, "label": "inside  graphiti.add_episode()   —  per window, LLM-driven"},
)

relayout(
    src="excalidraw/retrieval-journey.excalidraw",
    dst="excalidraw/retrieval-journey-v2.excalidraw",
    rows=[
        ["d1", "a1", "a2", "d2"],               # question → query → search → facts
        ["a3", "a4", "d3", "d4"],               # prefer valid → map back → top-10 → passages
        ["a5", "d5", "a6", "d6", "d7"],         # agent → answer → judge → verdict → accuracy
    ],
    note_of={
        "d1": "note-q", "a1": "note-a1", "a2": "note-a2", "d2": "note-d2",
        "a3": "note-a3", "a4": "note-a4", "d3": "note-d3", "d4": "note-d4",
        "a5": "note-a5", "a6": "note-a6", "d7": "note-d7",
    },
    # No dashed band here: the only natural group (graphiti.search) lands on
    # row 0, where the band would collide with the title block. The rows already
    # read as stages without it.
)
