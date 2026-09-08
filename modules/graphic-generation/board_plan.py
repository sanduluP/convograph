#!/usr/bin/env python3
"""
board_plan.py — a WINDOW of the conversation -> a structured board plan.

WHY THIS REPLACES fact-by-fact captioning
-----------------------------------------
kg_to_caption.py turns ONE fact into ONE image prompt. Two things break there:

1. A single fact carries no context. "Waiting one week too long on a
   setup-question spike caused the fix to become a training scramble" is not
   interpretable alone - by a model or by a person - so the prompt built from
   it cannot be meaningful either.

2. It conflates two outputs with opposite requirements. Canvas text needs REAL
   WORDS; a diffusion prompt needs a WORDLESS noun phrase. Measured 2026-09-06:
   a narrative caption made FLUX draw a meeting scene captioned "Seclany /
   SeeLLine / Sronniinge", while "an icon of a closed padlock, no text" gave a
   clean pictogram. Same model, same seed - only the caption shape differed.

So this asks an LLM to plan the WHOLE board from a whole window, and to keep
the two kinds of output apart by construction:

    label / note / title  ->  drawn as Excalidraw TEXT (real, crisp, editable)
    glyph                 ->  sent to FLUX as a wordless object

If FLUX fails entirely, the board still works: labels, links and notes are text.

PROVIDERS
---------
Any OpenAI-compatible chat endpoint, so the model is an experiment variable:

    saia     GWDG Academic Cloud - qwen3-30b (default), gpt-oss-120b, ...
    ollama   local, or the one we host on unicorn's H100

This is deliberately swappable: planning a whole board is a much harder task
than compressing one sentence, so whether a 4B suffices is a question to
MEASURE, not assume.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

PROMPT_FILE = os.path.join(os.path.dirname(__file__), "prompts", "board_plan_system.txt")
# A digest is a different SHAPE of input from a window - sections of ranked
# findings rather than raw messages plus a flat fact list - so it gets its own
# instructions instead of one prompt hedging between the two.
DIGEST_PROMPT_FILE = os.path.join(os.path.dirname(__file__), "prompts",
                                  "board_plan_digest_system.txt")
# A digest covers a whole meeting and earns more anchors than a two-window
# slice. Kept next to the prompt file it must agree with: the validator used to
# hardcode 2-4 while the digest prompt asked for 3-5, so a plan that obeyed the
# prompt was reported as invalid.
DIGEST_ANCHOR_RANGE = (4, 6)

# The SAIA key lives ONLY in git-ignored .env files - never in a tracked file.
# The vault's .env is the canonical home (the Obsidian repo has a GitHub remote,
# so a key written into a NOTE would be published; the .env is git-ignored).
# Order matters: first hit wins, and _load_keys() never overwrites an exported
# value. The MODULE 2 .env is listed because that is the one file we hand to a
# new teammate — it already carries the Neo4j and Graphiti settings, so making it
# carry the planner's key too means one file to copy instead of three.
# The vault is Faris's own canonical copy and does not exist on anyone else's
# machine, so it must not be the only place this is found.
_ENV_CANDIDATES = [
    os.path.join(os.path.dirname(__file__), "..", "kg-agent-memory", ".env"),
    os.path.join(os.path.dirname(__file__), "..", "..", ".env"),
    os.path.expanduser("~/Documents/Obsidian Vault/.env"),
]


def _load_keys() -> None:
    """Populate missing API keys from a git-ignored .env. Never overwrites a
    value already exported, so a caller can always override."""
    for path in _ENV_CANDIDATES:
        if not os.path.exists(path):
            continue
        for line in open(path):
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

# ── providers ───────────────────────────────────────────────────────────────
# Both speak OpenAI's /chat/completions, so one call shape covers them. ollama
# also serves its native /api/chat, but the OpenAI route keeps the two paths
# identical and makes the ablation a one-word change.
PROVIDERS = {
    "ollama": {
        "base_url": os.getenv("BOARD_PLAN_BASE_URL", "http://localhost:11435/v1"),
        "api_key_env": None,
        "default_model": "qwen3:4b-instruct",
    },
    "saia": {
        "base_url": "https://chat-ai.academiccloud.de/v1",
        "api_key_env": "SAIA_API_KEY",
        "default_model": "qwen3-30b-a3b-instruct-2507",
    },
}

# Which provider a caller gets when it does not choose. Measured 2026-09-06 on
# one window of 40 facts: the local 4B returned a plan with an invalid link
# index and a reused fact (4.0 s); qwen3-30b-a3b-instruct-2507 validated clean
# in 2.5 s; openai-gpt-oss-120b validated clean but took 28.6 s. The 30B was
# both the fastest and correct, so it is the default. Planning a whole board
# is a harder task than compressing one sentence - a 4B does not cover it.
DEFAULT_PROVIDER = os.getenv("BOARD_PLAN_PROVIDER", "saia")


def _chat(messages: list[dict], provider: str, model: str,
          temperature: float = 0.2, timeout: int = 300,
          max_tokens: int = 4096) -> str:
    """One call. max_tokens is generous ON PURPOSE.

    Several of these models are REASONING builds: they emit a `reasoning`
    channel before any content. Measured 2026-09-06 on openai-gpt-oss-120b with
    max_tokens=20 — it returned finish_reason="length", content=None, and a
    `reasoning` field reading "The user says: 'Reply with exactly: OK'...". It
    had spent the whole budget deliberating. The same trap our CLAUDE.md records
    for ollama's hybrid qwen3:4b, one API away.
    """
    _load_keys()
    cfg = PROVIDERS[provider]
    headers = {"Content-Type": "application/json"}
    if cfg["api_key_env"]:
        key = os.getenv(cfg["api_key_env"], "")
        if not key:
            raise RuntimeError(
                f"{provider} needs {cfg['api_key_env']} in the environment. "
                f"It lives in a git-ignored .env - never in a tracked file."
            )
        headers["Authorization"] = f"Bearer {key}"
    else:
        headers["Authorization"] = "Bearer ollama"   # ignored, keeps shape identical

    payload = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }).encode()
    req = urllib.request.Request(f"{cfg['base_url']}/chat/completions",
                                 data=payload, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = json.loads(r.read())
        choice = body["choices"][0]
        content = (choice["message"].get("content") or "").strip()
        if content:
            return content
        # A reasoning model that ran out of budget leaves content empty and the
        # partial thinking in `reasoning`. Say so plainly rather than failing
        # with an opaque JSON error three frames later.
        reasoning = (choice["message"].get("reasoning") or "").strip()
        raise RuntimeError(
            f"{provider}/{model} returned no content "
            f"(finish_reason={choice.get('finish_reason')}). This is a reasoning "
            f"model that spent its budget thinking; raise max_tokens. "
            f"reasoning starts: {reasoning[:200]!r}"
        )
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")[:400]
        raise RuntimeError(f"{provider}/{model} HTTP {exc.code}: {body}") from exc


def _extract_json(text: str) -> dict:
    """Pull the JSON object out of a reply.

    Smaller models wrap it in prose or a ```json fence however firmly the prompt
    says otherwise, so parsing has to survive that rather than fail the run.
    """
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    else:
        first, last = text.find("{"), text.rfind("}")
        if first != -1 and last > first:
            text = text[first:last + 1]
    return json.loads(text)


def _render_input(episode_texts: list[dict], facts: list[dict]) -> str:
    """Build the user message: the raw conversation, then the indexed facts.

    Both are included on purpose. The MESSAGES give the model the context a
    lone fact cannot; the FACTS give it stable indices to refer to, so its plan
    can be checked against, and traced back to, the graph.
    """
    parts = ["## Meeting messages\n"]
    for ep in episode_texts:
        parts.append((ep.get("content") or "").strip() + "\n")
    parts.append("\n## Facts extracted from them\n")
    for i, f in enumerate(facts):
        mark = " [SUPERSEDED]" if f.get("invalid_at") else ""
        parts.append(f"[{i}] {f['fact']}{mark}")
    return "\n".join(parts)


def build_messages(episode_texts: list[dict], facts: list[dict]) -> list[dict]:
    """The EXACT messages the planner will receive.

    Split out of plan_board so a caller can persist the prompt verbatim. What
    the model was shown is the first thing you need when a board comes out wrong
    — how many facts it actually saw, whether they were readable, whether the
    conversation text made it in — and reconstructing it afterwards from the
    inputs is guesswork the moment anything upstream changes.
    """
    return [{"role": "system", "content": open(PROMPT_FILE).read()},
            {"role": "user", "content": _render_input(episode_texts, facts)}]


def flatten_digest(digest: dict) -> tuple[list[dict], str]:
    """A digest -> (indexed items, the rendered user message).

    Every citable line gets ONE index across the whole digest, so the plan's
    `from_facts` still points at something specific and a board element can be
    traced back to the query that produced it. Without that, a digest-planned
    board would be unauditable in a way the window-planned one is not.

    Topics and participants are context, not citable items: an anchor cites the
    revision or decision it is about, and naming a topic is what its `label` is
    for. Numbering them too would let the planner "cover" a fact by pointing at
    the word "compliance".
    """
    q = digest.get("queries", {})
    items: list[dict] = []
    parts: list[str] = []

    parts.append("## What this meeting was about\n")
    for t in q.get("topics", []):
        nb = ", ".join((t.get("neighbours") or [])[:4])
        parts.append(f"- {t['topic']} — in {t['episodes']} windows"
                     + (f" (with {nb})" if nb else ""))

    if q.get("artifacts"):
        parts.append("\n## Documents it kept returning to\n")
        for a in q["artifacts"]:
            parts.append(f"- {a['artifact']} — in {a['episodes']} windows")

    def _add(section: str, rows: list[dict], render) -> None:
        if not rows:
            return
        parts.append(f"\n## {section}\n")
        for row in rows:
            i = len(items)
            items.append({"index": i, "source": section, **row})
            parts.append(f"[{i}] {render(row)}")

    _add("What changed", q.get("revisions", []),
         lambda r: f"{r['was']}  →  {r.get('became') or '(replacement unknown)'}")
    _add("What was settled", q.get("decisions", []), lambda r: r["fact"])
    _add("Still open", q.get("open_threads", []), lambda r: r["fact"])

    if q.get("participants"):
        parts.append("\n## Who was in the room\n")
        for p_ in q["participants"]:
            parts.append(f"- {p_['speaker']} — {100 * p_['share']:.0f}% of utterances")

    return items, "\n".join(parts)


def build_messages_from_digest(digest: dict) -> list[dict]:
    """The exact messages a digest-planned board is built from."""
    return [{"role": "system", "content": open(DIGEST_PROMPT_FILE).read()},
            {"role": "user", "content": flatten_digest(digest)[1]}]


def plan_board_from_digest(digest: dict, provider: str = DEFAULT_PROVIDER,
                           model: str | None = None) -> tuple[dict, list[dict]]:
    """Plan a board from a whole-meeting digest. Returns (plan, cited items).

    The alternative this replaces is a 2-window slice: ~40 facts out of 82,165,
    chosen by position rather than by mattering. The digest is the same graph
    asked what deserves to be drawn.
    """
    model = model or PROVIDERS[provider]["default_model"]
    items, _ = flatten_digest(digest)
    raw = _chat(build_messages_from_digest(digest), provider, model)
    plan = _extract_json(raw)

    plan.setdefault("title", "")
    for k in ("anchors", "links", "notes", "dropped"):
        plan.setdefault(k, [])
    plan["_raw_reply"] = raw
    plan["_provider"] = provider
    plan["_model"] = model
    plan["_n_facts_in"] = len(items)
    plan["_source"] = "digest"
    plan["_anchor_range"] = DIGEST_ANCHOR_RANGE
    return plan, items


def plan_board(episode_texts: list[dict], facts: list[dict],
               provider: str = DEFAULT_PROVIDER, model: str | None = None) -> dict:
    """Return the board plan. Raises on a reply that is not usable JSON."""
    model = model or PROVIDERS[provider]["default_model"]
    messages = build_messages(episode_texts, facts)

    raw = _chat(messages, provider, model)
    plan = _extract_json(raw)

    # Normalise: downstream rendering should never have to guess whether a key
    # is missing or a different type.
    plan.setdefault("title", "")
    for k in ("anchors", "links", "notes", "dropped"):
        plan.setdefault(k, [])
    plan["_raw_reply"] = raw
    plan["_provider"] = provider
    plan["_model"] = model
    plan["_n_facts_in"] = len(facts)
    return plan


# Words that mean the drawing would contain WRITING. A glyph is a wordless
# object; any of these in it means FLUX will be asked to render letters, and
# what it renders is gibberish ("Seclany / SeeLLine / Sronniinge", 2026-09-06).
# "sign" is here bare, not just as "sign saying": a blank sign is an invitation
# for the model to invent text on it.
BANNED_IN_GLYPH = ("text", "texts", "word", "words", "label", "labels",
                   "writing", "written", "sign", "signs", "caption",
                   "captions", "letter", "letters")


WRITING_BEARING = ("calendar", "clock", "document", "paper", "book",
                   "newspaper", "receipt", "certificate", "invoice",
                   "contract", "form", "ticket", "stamp", "poster",
                   "banner", "screen", "monitor", "dashboard",
                   "spreadsheet", "chart", "graph", "note", "notebook")


def validate(plan: dict, n_facts: int,
             anchor_range: tuple[int, int] = (2, 4)) -> list[str]:
    """Return human-readable problems with a plan. Empty list = clean.

    Kept separate from plan_board so a flawed plan can still be rendered and
    LOOKED at - the whole point of this stage is eyeballing results, and a plan
    that is 90% right is more informative than an exception.
    """
    problems = []
    lo, hi = anchor_range
    n = len(plan.get("anchors", []))
    if not lo <= n <= hi:
        problems.append(f"{n} anchors (prompt asks for {lo}-{hi})")

    for i, a in enumerate(plan.get("anchors", [])):
        for key in ("label", "glyph"):
            if not a.get(key):
                problems.append(f"anchor {i} has no {key}")
        # The one rule that matters most: the drawing must carry no words.
        g = (a.get("glyph") or "").lower()
        for banned in BANNED_IN_GLYPH:
            # \b so "sign" does not fire on "design"/"assign"/"signature".
            if re.search(rf"\b{banned}\b", g):
                problems.append(f"anchor {i} glyph mentions '{banned}' — it must be wordless")
        for obj in WRITING_BEARING:
            if re.search(rf"\b{obj}s?\b", g):
                problems.append(f"anchor {i} glyph is a {obj} — objects defined by "
                                f"their markings come back with invented letters")

    for l in plan.get("links", []):
        for end in ("from", "to"):
            if not isinstance(l.get(end), int) or not 0 <= l[end] < n:
                problems.append(f"link {end}={l.get(end)} is not an anchor index")

    # Fact coverage: every fact accounted for exactly once.
    used = []
    for a in plan.get("anchors", []):
        used += list(a.get("from_facts", []))
    for nt in plan.get("notes", []):
        if isinstance(nt.get("fact"), int):
            used.append(nt["fact"])
    used += list(plan.get("dropped", []))
    dupes = {i for i in used if used.count(i) > 1}
    if dupes:
        problems.append(f"fact indices used more than once: {sorted(dupes)}")
    oob = [i for i in used if not 0 <= i < n_facts]
    if oob:
        problems.append(f"fact indices out of range: {sorted(oob)[:5]}")
    return problems


# DO NOT add "no shading, no texture" here. Tried 2026-09-07 to fix pictograms
# that looked cream-tinted in the preview; measurement showed the tint was never
# in the images (19/19 had a border mean of 254.8/255, stdev 0.5) and the added
# phrase produced the only real defect we have seen — FLUX filled the surround
# with black marker strokes, border mean 188 with stdev 110. The lesson is not
# about these two words: a pictogram defect judged by eye off a preview is a
# guess, and prompt edits at n=1 are roulette. Measure first
# (analysis/check_pictograms.py), then change one thing.
#
# The style tag appended to every glyph before it reaches FLUX. Short on purpose:
# the shape of the caption is what decided the outcome on 2026-09-06, not its
# length. "no text" is load-bearing - without it FLUX letters the image with
# gibberish. The word "marker" is deliberately absent: it made FLUX draw a
# marker PEN in the frame rather than adopt a marker-drawn LOOK.
GLYPH_STYLE = ("hand-drawn black ink pictogram, isolated on plain white, "
               "no text, no letters, no numbers")


def glyph_to_prompt(glyph: str) -> str:
    """One anchor's glyph -> the exact string sent to FLUX.

    Kept here rather than in the renderer because it is part of PLANNING what
    the board says: the plan owns both halves of the split (words -> canvas
    text, glyph -> diffusion prompt), so both halves are readable in one file.
    """
    return f"{' '.join(glyph.split())}, {GLYPH_STYLE}"
