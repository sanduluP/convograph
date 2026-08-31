"""
Step 8: Speaker-attributed transcription — WHO said WHAT, in one pass.

This is the ASR half of the module. It runs two models together:

  - the same Streaming Sortformer diarizer the other scripts use
    (nvidia/diar_streaming_sortformer_4spk-v2.1), deciding who is talking, and
  - nvidia/multitalker-parakeet-streaming-0.6b-v1, a streaming ASR that NVIDIA
    built to pair with it: it spins up one recognizer instance per active
    speaker (\"speaker kernel injection\"), so it keeps transcribing correctly
    even while two people talk over each other.

The plumbing is adapted from NVIDIA's official example
(examples/asr/asr_cache_aware_streaming/speech_to_text_multitalker_streaming_infer.py
in the NeMo repo, at the NeMo version pinned in our image), with two changes:
both models load straight from Hugging Face instead of local .nemo files, and
the results are written in the shapes the rest of this module consumes.

Three files come out of ONE model run, so speaker labels are consistent by
construction (running 03 and an ASR separately could number speakers
differently):

  out/<stem>.seglst.json   NVIDIA's SegLST: sentence-level, speaker-tagged
  out/<stem>.words.json    the --asr-json input for 07_export_transcript.py
  out/<stem>_mt.rttm       diarization intervals from this same run

So the full path to a module-2-ready transcript is:

    make transcribe FILE=audio/meeting_16k.wav
    make transcript RTTM=out/meeting_16k_mt.rttm \\
        ASR=out/meeting_16k.words.json START=2026-08-31T14:00:00

Run inside the container (needs the GPU):

    python scripts/08_transcribe_multitalker.py audio/meeting_16k.wav
"""

import argparse
import json
import os
import time
import wave

import torch
from omegaconf import OmegaConf

from nemo.collections.asr.models import ASRModel, SortformerEncLabelModel
from nemo.collections.asr.parts.utils.multispk_transcribe_utils import SpeakerTaggedASR
from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer

DIAR_MODEL = "nvidia/diar_streaming_sortformer_4spk-v2.1"
ASR_MODEL = "nvidia/multitalker-parakeet-streaming-0.6b-v1"


def default_cfg(audio_file, att_context_size, sent_break_sec):
    """The configuration SpeakerTaggedASR expects. Values mirror the defaults
    of MultitalkerTranscriptionConfig in NVIDIA's example script; only the
    fields we deliberately set differ. att_context_size is [left, right] in
    80 ms frames — right context is the latency knob (13 frames ~ 1 s)."""
    return OmegaConf.create(dict(
        # input
        audio_file=audio_file,
        manifest_file=None,
        batch_size=1,
        # multi-instance ASR behavior (official defaults). Parallel strategy
        # only: NeMo also has a "serial" single-stream mode with finer
        # turn-level sentences, but at our NeMo version it collapsed all
        # speech onto one speaker when tested, so it is not offered here.
        max_num_of_spks=4,
        parallel_speaker_strategy=True,
        masked_asr=True,
        mask_preencode=False,
        cache_gating=True,
        cache_gating_buffer_size=2,
        single_speaker_mode=False,
        binary_diar_preds=False,
        spk_supervision="diar",
        # streaming geometry
        att_context_size=list(att_context_size),
        online_normalization=False,
        pad_and_drop_preencoded=False,
        # sentence assembly. NVIDIA's example uses sent_break_sec=30.0, which
        # on a real meeting glues a speaker's whole contribution into one
        # multi-minute block — bad timestamps for the knowledge graph. We
        # break at a few seconds of that speaker's silence instead.
        word_window=50,
        sent_break_sec=sent_break_sec,
        fix_prev_words_count=5,
        update_prev_words_sentence=5,
        left_frame_shift=-1,
        right_frame_shift=0,
        min_sigmoid_val=1e-2,
        discarded_frames=8,
        ignored_initial_frame_steps=5,
        # misc flags SpeakerTaggedASR may consult
        feat_len_sec=0.01,
        debug_mode=False,
        deploy_mode=False,
        verbose=False,
        log=False,
        colored_text=False,
        print_time=False,
        real_time_mode=False,
        use_amp=True,
        device="cuda",
    ))


def check_wav(path):
    with wave.open(path, "rb") as w:
        channels, rate = w.getnchannels(), w.getframerate()
        seconds = w.getnframes() / float(rate)
    if channels != 1 or rate != 16000:
        raise SystemExit(
            f"\nERROR: {path} is {channels}-channel at {rate} Hz.\n"
            f"Both models need 1-channel (mono) at 16000 Hz.\n"
            f"Fix it with:  ./scripts/02_prep_audio.sh {path}\n"
        )
    return seconds


def load_models(cfg):
    print(f"Loading diarizer   {DIAR_MODEL} ...")
    diar_model = SortformerEncLabelModel.from_pretrained(DIAR_MODEL).eval().to("cuda")
    # Streaming state geometry, as in the official example.
    diar_model.streaming_mode = True
    for key, value in dict(
        chunk_len=0, chunk_left_context=0, chunk_right_context=0,
        fifo_len=188, spkcache_len=188, spkcache_refresh_rate=0, log=False,
    ).items():
        setattr(diar_model.sortformer_modules, key, value)

    print(f"Loading ASR        {ASR_MODEL} (first run downloads ~2.5 GB) ...")
    asr_model = ASRModel.from_pretrained(model_name=ASR_MODEL).eval().to("cuda")
    asr_model.encoder.set_default_att_context_size(att_context_size=cfg.att_context_size)
    return asr_model, diar_model


def stream(cfg, asr_model, diar_model):
    """The chunk loop from the official example's launch_parallel_streaming."""
    streaming_buffer = CacheAwareStreamingAudioBuffer(
        model=asr_model,
        online_normalization=cfg.online_normalization,
        pad_and_drop_preencoded=cfg.pad_and_drop_preencoded,
    )
    streaming_buffer.append_audio_file(audio_filepath=cfg.audio_file, stream_id=-1)

    streamer = SpeakerTaggedASR(cfg, asr_model, diar_model)
    autocast = torch.amp.autocast(asr_model.device.type, enabled=cfg.use_amp)
    for step_num, (chunk_audio, chunk_lengths) in enumerate(iter(streaming_buffer)):
        drop_extra_pre_encoded = (
            0 if step_num == 0 and not cfg.pad_and_drop_preencoded
            else asr_model.encoder.streaming_cfg.drop_extra_pre_encoded
        )
        with torch.inference_mode(), autocast, torch.no_grad():
            streamer.perform_parallel_streaming_stt_spk(
                step_num=step_num,
                chunk_audio=chunk_audio,
                chunk_lengths=chunk_lengths,
                is_buffer_empty=streaming_buffer.is_buffer_empty(),
                drop_extra_pre_encoded=drop_extra_pre_encoded,
            )
    return streamer.generate_seglst_dicts_from_parallel_streaming(
        samples=[{"audio_filepath": cfg.audio_file}]
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", help="Path to a 16 kHz mono WAV file")
    ap.add_argument("--att-context", default="70,13", metavar="LEFT,RIGHT",
                    help="Attention context in 80 ms frames; RIGHT is the latency "
                         "knob (13 ~ 1 s of lookahead). Default: 70,13")
    ap.add_argument("--sent-break-sec", type=float, default=4.0,
                    help="Start a new sentence after this many seconds of the "
                         "speaker's own silence (default 4.0; NVIDIA's example "
                         "uses 30, which merges whole monologues)")
    ap.add_argument("--out-dir", default="out", help="Where to write the three outputs")
    args = ap.parse_args()

    att_context = [int(x) for x in args.att_context.split(",")]
    duration = check_wav(args.audio)
    print(f"Audio     : {args.audio}  ({duration:.1f} seconds)")

    cfg = default_cfg(args.audio, att_context, args.sent_break_sec)
    asr_model, diar_model = load_models(cfg)

    print("Transcribing...\n")
    started = time.perf_counter()
    seglst = stream(cfg, asr_model, diar_model)
    elapsed = time.perf_counter() - started

    if not seglst:
        raise SystemExit("ERROR: the model produced no speech segments — is there "
                         "audible speech in this file?")

    # The final flush of the streaming buffer can stamp words with times past
    # the end of the audio (padding frames); clamp so downstream timelines
    # never claim speech after the recording stopped.
    for seg in seglst:
        seg["start_time"] = min(seg["start_time"], duration)
        seg["end_time"] = min(seg["end_time"], duration)
    seglst = [s for s in seglst if s["end_time"] > s["start_time"]]
    seglst.sort(key=lambda s: (s["start_time"], s["end_time"]))

    for seg in seglst:
        print(f"  {seg['start_time']:8.2f}s -> {seg['end_time']:8.2f}s  "
              f"{seg['speaker']:<10} {seg['words']}")

    speakers = sorted({seg["speaker"] for seg in seglst})
    rtf = elapsed / duration if duration else float("nan")
    print(f"\nSpeakers found : {len(speakers)}  ({', '.join(speakers)})")
    print(f"Sentences      : {len(seglst)}")
    print(f"Processing time: {elapsed:.1f}s for {duration:.1f}s of audio "
          f"(RTF {rtf:.3f}, {'OK for live' if rtf < 1 else 'TOO SLOW for live'})")

    # ------------------------------------------------------------------
    # Write the three artifacts. All from this one run, so the speaker
    # labels in the RTTM and in the words file agree by construction.
    # ------------------------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.audio))[0]

    seglst_path = os.path.join(args.out_dir, f"{stem}.seglst.json")
    with open(seglst_path, "w") as fh:
        json.dump(seglst, fh, indent=2, ensure_ascii=False)

    words_path = os.path.join(args.out_dir, f"{stem}.words.json")
    with open(words_path, "w") as fh:
        json.dump([
            {"start": seg["start_time"], "end": seg["end_time"],
             "text": seg["words"], "speaker": seg["speaker"]}
            for seg in seglst
        ], fh, indent=2, ensure_ascii=False)

    rttm_path = os.path.join(args.out_dir, f"{stem}_mt.rttm")
    with open(rttm_path, "w") as fh:
        for seg in sorted(seglst, key=lambda s: s["start_time"]):
            dur = seg["end_time"] - seg["start_time"]
            if dur > 0:
                fh.write(f"SPEAKER {stem} 1 {seg['start_time']:.3f} {dur:.3f} "
                         f"<NA> <NA> {seg['speaker']} <NA> <NA>\n")

    print(f"\nSaved: {seglst_path}")
    print(f"Saved: {words_path}")
    print(f"Saved: {rttm_path}")
    print(f"\nNext (host-side): make transcript RTTM={rttm_path} "
          f"ASR={words_path} START=<recording start time>")


if __name__ == "__main__":
    main()
