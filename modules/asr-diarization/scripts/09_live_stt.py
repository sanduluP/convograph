"""
Step 9: LIVE speaker-attributed transcription — a stream in, sentences out.

Where step 8 transcribes a finished file, this consumes raw 16 kHz mono S16_LE
PCM from stdin (a microphone, or anything ffmpeg can decode into that shape)
and prints the evolving speaker-tagged transcript as JSON lines on stdout,
one snapshot per model step (~1.12 s of audio at the default latency):

    {"ready": true}                            after models load (~30-60 s)
    {"t": 12.3, "seglst": [...], "dropped_before": 0}     per step
    {"t": 300.1, "seglst": [...], "eof": true}            final full flush

Each seglst entry is {"speaker": "speaker_0", "start_time", "end_time",
"words"} — and the LAST sentence per speaker is still growing: its words and
end_time keep changing until the speaker pauses or another sentence starts.
Consumers decide when a sentence is "settled"; this script just reports.

How it works: the same two models as step 8 (streaming Sortformer diarizer +
multitalker Parakeet), but driven chunk-by-chunk through NeMo's own streaming
step API with deploy_mode=True (its file-less serving mode) instead of a
prebuilt audio buffer over a file. Three rules keep it correct, each learned
the hard way from the NeMo 2.7.3 source:

  1. Only whole chunks. The streaming buffer's iterator advances even when it
     yields a short tail chunk, permanently skipping audio that arrives later
     — so we drain only while a full chunk is available, and flush the tail
     exactly once at EOF with is_buffer_empty=True.
  2. step_num starts at 0 exactly once (0 resets all streaming state).
  3. Read the evolving sentences from
     streamer.instance_manager.batch_asr_states[0].seglsts — NEVER call
     generate_seglst_dicts_from_parallel_streaming mid-stream (it extends
     internal state each call and corrupts the session).

Usage (inside the container):

    ... | python scripts/09_live_stt.py                # PCM on stdin
    python scripts/09_live_stt.py --wav examples/2spk.wav --fast   # simulate

stdout is exclusively the JSONL stream; all status goes to stderr.
"""

import argparse
import importlib.util
import json
import os
import sys
import time
import wave

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_RATE = 16000
READ_SECONDS = 0.25          # stdin granularity; small = smooth, any value works
SNAPSHOT_TAIL = 50           # sentences per snapshot line; older ones counted


def log(msg: str) -> None:
    print(f"STATUS {msg}", file=sys.stderr, flush=True)


def load_step8():
    """Reuse step 8's config + model loading (filename starts with a digit,
    so a plain import can't reach it)."""
    spec = importlib.util.spec_from_file_location(
        "mt08", os.path.join(HERE, "08_transcribe_multitalker.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pcm_source(args):
    """Yield float32 PCM arrays: stdin by default, or a wav for simulation."""
    if args.wav:
        with wave.open(args.wav, "rb") as w:
            assert w.getnchannels() == 1 and w.getframerate() == SAMPLE_RATE, \
                f"{args.wav} must be 16 kHz mono (run 02_prep_audio.sh)"
            step = int(READ_SECONDS * SAMPLE_RATE)
            while True:
                frames = w.readframes(step)
                if not frames:
                    return
                if not args.fast:
                    time.sleep(READ_SECONDS)   # pace at real time, like a mic
                yield np.frombuffer(frames, np.int16).astype(np.float32) / 32768.0
    else:
        read_bytes = int(READ_SECONDS * SAMPLE_RATE) * 2
        while True:
            raw = sys.stdin.buffer.read(read_bytes)
            if not raw:
                return
            if len(raw) % 2:                    # truncated final int16
                raw = raw[:-1]
            yield np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0


def emit_snapshot(streamer, audio_seconds: float, eof: bool = False) -> None:
    seglst = streamer.instance_manager.batch_asr_states[0].seglsts
    rows = [{"speaker": s["speaker"],
             "start_time": round(float(s["start_time"]), 3),
             "end_time": round(float(s["end_time"]), 3),
             "words": s["words"]}
            for s in seglst if s["words"].strip()]
    tail = rows if eof else rows[-SNAPSHOT_TAIL:]
    out = {"t": round(audio_seconds, 2), "seglst": tail,
           "dropped_before": len(rows) - len(tail)}
    if eof:
        out["eof"] = True
    print(json.dumps(out, ensure_ascii=False), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--att-context", default="70,13", metavar="LEFT,RIGHT",
                    help="latency knob in 80 ms frames (13 right ~ 1 s)")
    ap.add_argument("--sent-break-sec", type=float, default=4.0)
    ap.add_argument("--wav", default=None,
                    help="simulate live input from a 16 kHz mono wav")
    ap.add_argument("--fast", action="store_true",
                    help="with --wav: no real-time pacing")
    args = ap.parse_args()

    att_context = [int(x) for x in args.att_context.split(",")]
    mt08 = load_step8()

    # Step 8's config, switched into NeMo's file-less serving mode.
    cfg = mt08.default_cfg("streaming_session.wav", att_context,
                           args.sent_break_sec)
    cfg.deploy_mode = True
    cfg.audio_file = None
    cfg.generate_realtime_scripts = False

    log("loading models (diarizer + multitalker ASR)")
    asr_model, diar_model = mt08.load_models(cfg)

    from nemo.collections.asr.parts.utils.multispk_transcribe_utils import (
        SpeakerTaggedASR,
    )
    from nemo.collections.asr.parts.utils.streaming_utils import (
        CacheAwareStreamingAudioBuffer,
    )

    sc = asr_model.encoder.streaming_cfg
    pick = lambda v, i: v[i] if isinstance(v, (list, tuple)) else v  # noqa: E731
    chunk_first, chunk_later = pick(sc.chunk_size, 0), pick(sc.chunk_size, 1)
    shift = pick(sc.shift_size, 1)
    log(f"streaming_cfg: chunk_size={sc.chunk_size} shift_size={sc.shift_size} "
        f"pre_encode_cache={getattr(sc, 'pre_encode_cache_size', '?')} "
        f"drop_extra={sc.drop_extra_pre_encoded} valid_out_len={sc.valid_out_len}")

    buffer = CacheAwareStreamingAudioBuffer(
        model=asr_model, online_normalization=False,
        pad_and_drop_preencoded=False)
    streamer = SpeakerTaggedASR(cfg, asr_model, diar_model)
    autocast = torch.amp.autocast(asr_model.device.type, enabled=cfg.use_amp)

    print(json.dumps({"ready": True}), flush=True)
    log("READY — feed 16 kHz mono S16_LE PCM on stdin")

    step = 0
    appended = False
    samples_in = 0

    def run_step(chunk_audio, chunk_lengths, is_buffer_empty: bool) -> None:
        nonlocal step
        drop = 0 if step == 0 else sc.drop_extra_pre_encoded
        with torch.inference_mode(), autocast, torch.no_grad():
            streamer.perform_parallel_streaming_stt_spk(
                step_num=step, chunk_audio=chunk_audio,
                chunk_lengths=chunk_lengths, is_buffer_empty=is_buffer_empty,
                drop_extra_pre_encoded=drop)
        step += 1

    def frames_available() -> int:
        return int(buffer.streams_length[0]) - buffer.buffer_idx

    stream_iter = None
    for pcm in pcm_source(args):
        samples_in += len(pcm)
        if not appended:
            buffer.append_audio(pcm, stream_id=-1)
            appended = True
            # Rule check from the design notes: appending to stream 0 must
            # EXTEND the one batch row, not open a second one.
            if len(buffer.streams_length) != 1:
                log("FATAL unexpected buffer batch shape "
                    f"{buffer.streams_length} — aborting")
                sys.exit(3)
        else:
            before = int(buffer.streams_length[0])
            buffer.append_audio(pcm, stream_id=0)
            if len(buffer.streams_length) != 1 or \
                    int(buffer.streams_length[0]) <= before:
                log("FATAL append_audio(stream_id=0) did not extend the "
                    "stream — NeMo buffer semantics changed; aborting")
                sys.exit(3)
        if stream_iter is None:
            stream_iter = iter(buffer)

        # Rule 1: drain only whole chunks; the first step's chunk is larger.
        while frames_available() >= (chunk_first if step == 0 else chunk_later):
            chunk_audio, chunk_lengths = next(stream_iter)
            run_step(chunk_audio, chunk_lengths, is_buffer_empty=False)
            emit_snapshot(streamer, samples_in / SAMPLE_RATE)

    # EOF: flush the partial tail exactly once.
    if appended and stream_iter is not None and frames_available() > 0:
        try:
            chunk_audio, chunk_lengths = next(stream_iter)
            run_step(chunk_audio, chunk_lengths, is_buffer_empty=True)
        except StopIteration:
            pass
    emit_snapshot(streamer, samples_in / SAMPLE_RATE, eof=True)
    log(f"done: {samples_in / SAMPLE_RATE:.1f}s of audio, {step} steps")


if __name__ == "__main__":
    main()
