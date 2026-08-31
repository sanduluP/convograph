"""
End-to-end check of the module 1 -> module 2 handoff. No GPU, no LLM, no cost.

What it proves: a real diarization output (the tracked AMI fixture
modules/asr-diarization/examples/3spk.rttm) flows through

    07_export_transcript.py   (module 1's output boundary)
      -> transcript_to_kg_input.py   (this pipeline's hop converter)
        -> load_conversation_messages / format_retrieved_message
           (module 2's OWN loader code, imported, not reimplemented)

and comes out the other side ordered, attributed, and formatted exactly like a
native GroupMemBench conversation. Since Sortformer produces no words, the
check fabricates per-segment ASR text — clearly marked as such — because the
thing under test is the plumbing, not a speech recognizer.

Run from anywhere:

    python3 pipeline/check_handoff.py
"""

import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULE1 = os.path.join(REPO_ROOT, "modules", "asr-diarization")
MODULE2 = os.path.join(REPO_ROOT, "modules", "kg-agent-memory")
FIXTURE_RTTM = os.path.join(MODULE1, "examples", "3spk.rttm")
SESSION_START = "2026-08-31T14:00:00+00:00"
SPEAKER_MAP = "speaker_1=Alice,speaker_2=Bob,speaker_3=Carol"


def run(cmd):
    shown = [os.path.relpath(c, REPO_ROOT) if c.startswith(REPO_ROOT)
             else os.path.basename(c) if os.path.sep in c else c
             for c in cmd]
    print(f"$ {' '.join(shown)}", flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    print(flush=True)


def fabricate_asr_json(rttm_path, out_path):
    """One text item per RTTM line, timed to sit inside that line's interval.
    The text is obviously synthetic on purpose. Items carry the model's
    speaker label, as a speaker-attributed ASR (multitalker-parakeet) would —
    AMI audio has overlapped speech, where time overlap alone cannot decide
    which concurrent speaker owns the words."""
    items = []
    with open(rttm_path) as fh:
        for i, line in enumerate(fh):
            parts = line.split()
            if not parts or parts[0] != "SPEAKER":
                continue
            start, dur, label = float(parts[3]), float(parts[4]), parts[7]
            items.append({
                "start": start,
                "end": start + dur,
                "speaker": label,
                "text": f"(synthetic ASR text for utterance {i}, model label {label})",
            })
    with open(out_path, "w") as fh:
        json.dump(items, fh)
    return len(items)


def import_module2_eval_lib():
    """Import module 2's shared loader. Its module pulls in LLM client deps at
    import time; if those are absent on this machine, stub them — this check
    only exercises the loader and the passage formatter, never an LLM call."""
    sys.path.insert(0, MODULE2)  # for `from llm_utils import ...` inside eval_lib
    sys.path.insert(0, os.path.join(MODULE2, "baselines", "rag_common"))
    try:
        import eval_lib
    except ImportError:
        import types
        stub = types.ModuleType("llm_utils")
        stub.chat_completion_text = None  # never called by this check
        sys.modules["llm_utils"] = stub
        tqdm_pkg = types.ModuleType("tqdm")
        tqdm_auto = types.ModuleType("tqdm.auto")
        tqdm_auto.tqdm = lambda x, **kw: x
        tqdm_pkg.auto = tqdm_auto
        sys.modules.setdefault("tqdm", tqdm_pkg)
        sys.modules["tqdm.auto"] = tqdm_auto
        import eval_lib
    return eval_lib


def main():
    tmp = tempfile.mkdtemp(prefix="convograph_handoff_")
    asr_json = os.path.join(tmp, "asr.json")
    transcript_json = os.path.join(tmp, "3spk.transcript.json")
    kg_input_json = os.path.join(tmp, "3spk.kg_input.json")

    n_rttm = fabricate_asr_json(FIXTURE_RTTM, asr_json)
    print(f"Fixture: {os.path.relpath(FIXTURE_RTTM, REPO_ROOT)} "
          f"({n_rttm} RTTM lines, synthetic text attached)\n")

    # Module 1's output boundary.
    run([sys.executable, os.path.join(MODULE1, "scripts", "07_export_transcript.py"),
         FIXTURE_RTTM,
         "--session-start", SESSION_START,
         "--asr-json", asr_json,
         "--conversation-id", "handoff_check",
         "--out", transcript_json])

    # The pipeline hop.
    run([sys.executable, os.path.join(REPO_ROOT, "pipeline", "transcript_to_kg_input.py"),
         transcript_json,
         "--speaker-map", SPEAKER_MAP,
         "--out", kg_input_json])

    # Module 2's own loader, on the result.
    eval_lib = import_module2_eval_lib()
    messages = eval_lib.load_conversation_messages(kg_input_json)

    assert messages, "module 2 loaded zero messages"
    assert all(m["_channel"] == "handoff_check" for m in messages), \
        "channel tag did not survive the loader"

    expected_authors = {"Alice", "Bob", "Carol"}
    authors = {m["author"] for m in messages}
    assert authors <= expected_authors, f"unexpected authors: {authors - expected_authors}"

    timestamps = [m["timestamp"] for m in messages]
    assert timestamps == sorted(timestamps), "loader ordering broke on our timestamps"
    datetime.fromisoformat(timestamps[0])  # ISO-parseable, as module 2's KG needs

    for m in messages:
        assert eval_lib.message_index_text(m), \
            f"empty indexable text in {m['msg_node']} — retriever would skip it"

    passage = eval_lib.format_retrieved_message(messages[0])
    assert passage.startswith("[user="), "passage header shape changed"
    assert "channel=handoff_check" in passage and "timestamp=" in passage

    print("Module 2 sees:")
    print(f"  {len(messages)} messages from {n_rttm} diarization segments, "
          f"authors: {', '.join(sorted(authors))}")
    print(f"  first passage as the QA agent would see it:\n")
    for line in passage.splitlines():
        print(f"    {line}")
    print(f"\nPASS — handoff verified. Scratch files in {tmp}")


if __name__ == "__main__":
    main()
