"""
module1.py — the UI's bridge to modules/asr-diarization (audio → transcript).

Same philosophy as orchestrator.py: talk to the module over subprocess, don't
import its internals. Module 1 runs inside its own Docker image (the GB10
ARM+Blackwell pinning lives there), so the chain here is:

  browser audio (mic recording or upload — the SERVER needs no microphone,
  st.audio_input records client-side and posts the bytes)
    │
    ▼  host ffmpeg                    16 kHz mono WAV (the only format the
    │                                 models accept)
    ▼  docker run <image> scripts/08_transcribe_multitalker.py
    │                                 who said what: words.json + _mt.rttm
    ▼  scripts/07_export_transcript.py (host python, stdlib)
    │                                 schema-valid DiarizedTranscript
    ▼  pipeline/transcript_to_ui_text.py's render_lines (imported, stdlib)
                                      "Sarah (PM): ..." lines — exactly what
                                      the transcript text box downstream eats

Renaming speakers after the fact is free: only the last step re-runs.

GPU note: transcription needs ~5.5 GB. If it fails with CUDA out-of-memory
while memory looks free, see the GB10 page-cache note in
modules/asr-diarization/README.md (Common errors).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

UI_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(UI_DIR)
MODULE1_DIR = os.path.join(REPO_ROOT, "modules", "asr-diarization")
DOCKER_IMAGE = os.environ.get("MODULE1_DOCKER_IMAGE", "sortformer-spark")

sys.path.insert(0, os.path.join(REPO_ROOT, "pipeline"))
from transcript_to_ui_text import render_lines  # noqa: E402


class Module1Error(RuntimeError):
    pass


FREE_GPU_HELPER = "/usr/local/sbin/convograph-free-gpu"


def _free_gpu_memory(progress_cb: Optional[Callable[[str], None]]) -> bool:
    """GB10 quirk: CUDA allocations degrade next to a big resident model
    server (page cache + fragmentation) until caches are dropped and memory
    compacted — root-only actions. modules/asr-diarization/scripts/
    setup_gpu_helper.sh installs a passwordless helper for exactly that.
    Returns True if the helper ran."""
    if not os.path.exists(FREE_GPU_HELPER):
        return False
    proc = subprocess.run(["sudo", "-n", FREE_GPU_HELPER],
                          capture_output=True, text=True, timeout=120)
    if proc.returncode == 0:
        if progress_cb:
            progress_cb("Freed reclaimable memory for CUDA (GB10 quirk)")
        return True
    return False


@dataclass
class TranscribeResult:
    transcript_json: str      # schema-valid DiarizedTranscript (abs path)
    speakers: List[str]       # canonical labels, e.g. ["speaker_1", "speaker_2"]
    duration_sec: float
    compute_sec: float


def _run(cmd, progress_cb: Optional[Callable[[str], None]], label: str,
         timeout: int = 600) -> str:
    if progress_cb:
        progress_cb(label)
    proc = subprocess.run(
        cmd, cwd=MODULE1_DIR, capture_output=True, text=True, timeout=timeout,
    )
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout + "\n" + proc.stderr).strip().splitlines()[-12:])
        raise Module1Error(f"{label} failed:\n{tail}")
    return proc.stdout


def transcribe(audio_bytes: bytes, orig_name: str,
               progress_cb: Optional[Callable[[str], None]] = None,
               session_start_iso: Optional[str] = None) -> TranscribeResult:
    """Audio bytes in, DiarizedTranscript JSON on disk out. Slow step (GPU)."""
    os.makedirs(os.path.join(MODULE1_DIR, "audio"), exist_ok=True)
    os.makedirs(os.path.join(MODULE1_DIR, "out"), exist_ok=True)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    ext = os.path.splitext(orig_name)[1] or ".wav"
    raw_rel = f"audio/ui_{stamp}{ext}"
    wav_rel = f"audio/ui_{stamp}_16k.wav"
    with open(os.path.join(MODULE1_DIR, raw_rel), "wb") as fh:
        fh.write(audio_bytes)

    _run(["ffmpeg", "-y", "-i", raw_rel, "-ar", "16000", "-ac", "1", wav_rel],
         progress_cb, "Converting to 16 kHz mono (ffmpeg)")

    docker_cmd = [
        "docker", "run", "--rm", "--gpus", "all",
        "--ipc=host", "--ulimit", "memlock=-1", "--ulimit", "stack=67108864",
        "-e", f"HF_TOKEN={os.environ.get('HF_TOKEN', '')}",
        "-v", os.path.expanduser("~/.cache/huggingface") + ":/root/.cache/huggingface",
        "-v", f"{MODULE1_DIR}:/work", "-w", "/work",
        DOCKER_IMAGE,
        "python", "scripts/08_transcribe_multitalker.py", wav_rel,
    ]
    _free_gpu_memory(progress_cb)  # pre-empt the OOM rather than hit it
    started = time.perf_counter()
    label = "Transcribing (Sortformer + multitalker Parakeet, ~5.5 GB GPU)"
    try:
        _run(docker_cmd, progress_cb, label)
    except Module1Error as exc:
        if "out of memory" not in str(exc):
            raise
        if not _free_gpu_memory(progress_cb):
            raise Module1Error(
                str(exc) + "\n\nCUDA out of memory on a GB10 usually means "
                "page-cache/fragmentation buildup. Install the self-heal "
                "helper once:\n  sudo modules/asr-diarization/scripts/"
                "setup_gpu_helper.sh"
            ) from exc
        _run(docker_cmd, progress_cb, label + " (retry)")
    compute = time.perf_counter() - started

    stem = f"ui_{stamp}_16k"
    transcript_rel = f"out/{stem}.transcript.json"
    export_cmd = [
        sys.executable, "scripts/07_export_transcript.py", f"out/{stem}_mt.rttm",
        "--asr-json", f"out/{stem}.words.json",
        "--conversation-id", f"ui_{stamp}",
        "--out", transcript_rel,
    ]
    if session_start_iso:
        export_cmd += ["--session-start", session_start_iso]
    _run(export_cmd, progress_cb, "Exporting schema-valid transcript")

    transcript_path = os.path.join(MODULE1_DIR, transcript_rel)
    with open(transcript_path) as fh:
        transcript = json.load(fh)

    segs = transcript["segments"]
    if segs:
        from datetime import datetime
        duration = (datetime.fromisoformat(segs[-1]["end_ts"])
                    - datetime.fromisoformat(segs[0]["start_ts"])).total_seconds()
    else:
        duration = 0.0

    if progress_cb:
        progress_cb(f"Done: {len(transcript['speakers'])} speaker(s), "
                    f"{len(segs)} segments, {compute:.0f}s compute")
    return TranscribeResult(
        transcript_json=transcript_path,
        speakers=transcript["speakers"],
        duration_sec=duration,
        compute_sec=compute,
    )


def to_text(transcript_json: str, speaker_names: Optional[Dict[str, str]] = None,
            merge_gap: float = 15.0) -> str:
    """Fast step: render (or re-render after renaming) the utterance lines."""
    with open(transcript_json) as fh:
        transcript = json.load(fh)
    # Empty rename fields fall back to the canonical label.
    name_map = {k: v.strip() for k, v in (speaker_names or {}).items() if v.strip()}
    lines = render_lines(transcript, name_map, merge_gap)
    if not lines:
        raise Module1Error(
            "The transcript has no text — the recording may contain no "
            "recognizable speech (or none in English, which the ASR is "
            "trained on)."
        )
    return "\n".join(lines)
