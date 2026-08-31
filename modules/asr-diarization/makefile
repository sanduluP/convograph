# Simple shortcuts so you never have to remember the long docker commands.
# Run these from this folder on your DGX Spark.
#
#   make build    - build the image (once, ~15 min)
#   make verify   - check the GPU works
#   make shell    - open a terminal inside the container
#   make diarize FILE=audio/x_16k.wav
#   make diarize FILE=audio/x_16k.wav PRESET=realtime

IMAGE   := sortformer-spark
PRESET  ?= offline
FILE    ?=

# These flags are all needed:
#   --gpus all       give the container the GPU
#   --ipc=host       NeMo's data loaders need shared memory
#   --ulimit ...     raises stack/memory limits NVIDIA recommends
#   -v ~/.cache/...  reuse downloaded models instead of re-downloading
#   -v $(PWD):/work  make this folder visible inside the container
DOCKER_RUN = docker run --rm -it \
	--gpus all \
	--ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
	-e HF_TOKEN=$${HF_TOKEN} \
	-v $$HOME/.cache/huggingface:/root/.cache/huggingface \
	-v $(PWD):/work \
	-w /work \
	$(IMAGE)

# Same, but without -t, so audio can be piped into it.
DOCKER_PIPE = docker run --rm -i \
	--gpus all \
	--ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
	-e HF_TOKEN=$${HF_TOKEN} \
	-v $$HOME/.cache/huggingface:/root/.cache/huggingface \
	-v $(PWD):/work \
	-w /work \
	$(IMAGE)

.PHONY: build verify shell diarize getexamples examples example all-examples mics live transcript transcribe

build:
	docker build -t $(IMAGE) .

verify:
	$(DOCKER_RUN) python scripts/01_verify.py

shell:
	$(DOCKER_RUN) bash

# Downloads 4 headset channels from one AMI meeting (~200 MB). Host-side.
MINUTES ?= 5
getexamples:
	./scripts/04_get_examples.sh $(MINUTES)

# Builds examples/2spk.wav, 3spk.wav, 4spk.wav plus ground-truth labels.
examples:
	$(DOCKER_RUN) python scripts/05_make_examples.py

# Run one example and score it.  make example N=3
N ?= 2
example:
	$(DOCKER_RUN) python scripts/03_diarize.py examples/$(N)spk.wav \
		--preset $(PRESET) --ref examples/$(N)spk.rttm --rttm out/$(N)spk_pred.rttm

# Run all three, one after the other.
all-examples:
	@for n in 2 3 4; do \
		echo ""; echo "################  $$n SPEAKERS  ################"; \
		$(MAKE) --no-print-directory example N=$$n PRESET=$(PRESET); \
	done

# List microphones the host can see.
mics:
	@arecord -l || echo "arecord not found -- run: sudo apt install alsa-utils"

# Live diarization from the microphone.
#   make live                     use the default mic
#   make live MIC=plughw:1,0      use a specific one (see 'make mics')
#   make live WINDOW=20           less context, lower delay
MIC ?= default
WINDOW ?= 30
HOP ?= 1.5
LIVE_RTTM ?= out/live.rttm
live:
	@echo "Recording from '$(MIC)'. Ctrl-C to stop."
	@arecord -D $(MIC) -f S16_LE -r 16000 -c 1 -t raw -q | \
	$(DOCKER_PIPE) python scripts/06_live.py \
		--preset $(PRESET) --window $(WINDOW) --hop $(HOP) --rttm $(LIVE_RTTM)

# Speaker-attributed transcription: WHO said WHAT, in one pass (Sortformer +
# multitalker Parakeet). First run downloads ~2.5 GB. Writes three files to
# out/: <stem>.seglst.json, <stem>.words.json, <stem>_mt.rttm — feed the last
# two to `make transcript`.
#   make transcribe FILE=audio/meeting_16k.wav
transcribe:
ifndef FILE
	$(error Please pass a file, e.g. make transcribe FILE=audio/meeting_16k.wav)
endif
	$(DOCKER_RUN) python scripts/08_transcribe_multitalker.py $(FILE)

# Convert diarization output (RTTM) into the DiarizedTranscript JSON that the
# rest of convograph consumes (schemas/diarized-transcript.schema.json).
# Runs on the HOST — plain python3, no container, no GPU.
#   make transcript RTTM=out/meeting_16k.rttm START=2026-08-31T14:00:00
#   make transcript RTTM=out/live.rttm ASR=out/words.json
transcript:
ifndef RTTM
	$(error Please pass an RTTM, e.g. make transcript RTTM=out/meeting_16k.rttm)
endif
	python3 scripts/07_export_transcript.py $(RTTM) \
		$(if $(START),--session-start $(START)) \
		$(if $(ASR),--asr-json $(ASR))

diarize:
ifndef FILE
	$(error Please pass a file, e.g. make diarize FILE=audio/meeting_16k.wav)
endif
	$(DOCKER_RUN) python scripts/03_diarize.py $(FILE) --preset $(PRESET) --rttm out/$(notdir $(basename $(FILE))).rttm