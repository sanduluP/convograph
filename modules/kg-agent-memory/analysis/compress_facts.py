#!/usr/bin/env python3
"""
compress_facts.py — turn fact SENTENCES into board HEADLINES.

WHY THIS EXISTS
---------------
Measured on the first board: our cards carry a median of 18 words and up to 27.
A real graphic recording carries 3-6. That single difference is what makes the
board read as a Kanban wall, and it cannot be styled away:

    18-word sentences -> big boxes -> boxes must tile -> a grid -> a Kanban wall

Short cards are the precondition for everything else we want. Whitespace, a
spatial metaphor, pictograms sitting beside text rather than competing with it —
none of it is reachable while a card is a paragraph.

    in   "User_3 requires Finance Ops to own go/no-go cleanly by July 28."
    out  "Finance Ops owns go/no-go"

WHY THE RESULTS ARE CACHED ON DISK
----------------------------------
The same fact is compressed identically every time, so paying an LLM for it twice
is waste — and worse, a non-deterministic model would make the board change
between renders for no reason. The cache is keyed on a hash of the fact text plus
the model name, so changing the model or editing a fact invalidates only what it
should. This also means the board can be re-rendered offline, with no LLM
reachable at all.

BACKEND
-------
Any OpenAI-compatible endpoint. Defaults to a local ollama, which is enough for a
few dozen short calls; point OPENAI_BASE_URL at the cluster's vLLM for a big run.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys

CACHE = os.getenv("PHRASE_CACHE", "analysis/.phrase_cache.json")
BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "ollama")
# A REASONING model is the wrong tool here: qwen3:4b narrates its deliberation
# ("Hmm, the user wants me to…") and exhausts the token budget before answering,
# even with think=false and few-shot priming. An INSTRUCT model answers directly.
# The cluster mirrors this split — Qwen3-4B-Instruct-2507 vs -Thinking-2507; this
# job wants the Instruct side of that pair.
MODEL = os.getenv("PHRASE_MODEL", "qwen2.5:3b-instruct")
PROMPT_FILE = os.getenv("PHRASE_PROMPT", "prompts/board_phrase_system.txt")
MAX_WORDS = int(os.getenv("PHRASE_MAX_WORDS", "6"))


def _key(fact: str) -> str:
    return hashlib.sha1(f"{MODEL}::{fact}".encode()).hexdigest()[:16]


def _load_cache() -> dict:
    if os.path.exists(CACHE):
        try:
            return json.load(open(CACHE))
        except json.JSONDecodeError:
            return {}          # a truncated cache is not worth crashing over
    return {}


def _save_cache(c: dict) -> None:
    os.makedirs(os.path.dirname(CACHE) or ".", exist_ok=True)
    json.dump(c, open(CACHE, "w"), indent=2, ensure_ascii=False)


def _fallback(fact: str) -> str:
    """Deterministic trim used when no LLM is reachable.

    Deliberately crude: strip the leading speaker clause and keep the first few
    content words. It keeps the pipeline runnable offline and makes it obvious in
    the output when the model was not consulted, rather than silently producing a
    board that looks finished.
    """
    t = re.sub(r"^User_\d+\s+\w+s?\s+", "", fact).strip()
    t = re.sub(r"^(that|the|a|an)\s+", "", t, flags=re.I)
    return " ".join(t.split()[:MAX_WORDS]).rstrip(".,;:")


def _tidy(text: str) -> str:
    """Take the headline out of whatever the model wrapped it in."""
    t = text.strip()
    # thinking models emit a <think> block before the answer
    t = re.sub(r"(?s)<think>.*?</think>", "", t).strip()
    t = t.strip().strip('"').strip("'")
    t = t.split("\n")[0].strip()               # first line only
    t = re.sub(r"^(headline|answer)\s*:\s*", "", t, flags=re.I)
    words = t.split()
    if len(words) > MAX_WORDS:                 # enforce the budget ourselves
        words = words[:MAX_WORDS]
    # Truncating at a word boundary is not enough: cutting "Finance Ops owns
    # go/no-go & July 28" at six words leaves "… go/no-go &", and a headline
    # ending in a dangling connector reads as a rendering bug. Drop trailing
    # words that cannot end a phrase.
    DANGLING = {"&", "→", "and", "or", "to", "by", "for", "of", "on", "in",
                "with", "the", "a", "an", "at", "from", "as", "that", "-", "·"}
    while words and words[-1].lower().strip(".,;:") in DANGLING:
        words.pop()
    return " ".join(words).rstrip(".,;:")


def compress(facts: list[str], verbose: bool = True) -> dict[str, str]:
    """fact -> headline, using the cache and falling back if no LLM answers."""
    cache = _load_cache()
    todo = [f for f in facts if _key(f) not in cache]
    if verbose:
        print(f"   {len(facts)} facts · {len(facts) - len(todo)} cached · {len(todo)} to compress")

    if todo:
        try:
            from openai import OpenAI
            client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=60.0)
            system = open(PROMPT_FILE).read()
            for i, fact in enumerate(todo, 1):
                try:
                    r = client.chat.completions.create(
                        model=MODEL,
                        messages=[{"role": "system", "content": system},
                                  {"role": "user", "content": fact}],
                        temperature=0.2, max_tokens=200,
                    )
                    cache[_key(fact)] = _tidy(r.choices[0].message.content or "")
                except Exception as exc:                       # noqa: BLE001
                    if verbose:
                        print(f"   ⚠️  {type(exc).__name__} on {i}/{len(todo)} — using fallback")
                    cache[_key(fact)] = _fallback(fact)
                if verbose and i % 5 == 0:
                    print(f"   … {i}/{len(todo)}")
                    _save_cache(cache)          # checkpoint: a crash loses ≤5 calls
        except ImportError:
            if verbose:
                print("   ⚠️  openai package missing — falling back for all")
            for fact in todo:
                cache[_key(fact)] = _fallback(fact)
        _save_cache(cache)

    return {f: cache.get(_key(f)) or _fallback(f) for f in facts}


if __name__ == "__main__":
    # Smoke test: compress whatever is piped in, or three built-in examples.
    samples = [l.strip() for l in sys.stdin if l.strip()] if not sys.stdin.isatty() else [
        "User_3 requires Finance Ops to own go/no-go cleanly by July 28.",
        "User_2 requests Finance Ops and Integration Team to isolate the failure point, "
        "confirm impacted workflows, and post findings against the risk spec by EOD.",
        "The spec defers artifact naming to a separate follow-up after design review.",
    ]
    print(f"model={MODEL} endpoint={BASE_URL}")
    out = compress(samples)
    print()
    for fact, phrase in out.items():
        print(f"  {len(fact.split()):>2}w → {len(phrase.split()):>1}w   {phrase}")
        print(f"          from: {fact[:78]}")
