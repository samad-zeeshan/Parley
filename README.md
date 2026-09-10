# Parley

A voice agent that books property viewings in the UAE. The caller can speak English, Gulf Arabic or
Modern Standard Arabic (MSA), or switch between them. Everything runs on one CPU-only machine after the
models are downloaded. No cloud API is in the request path.

All listings, agents, phone numbers and caller audio are synthetic. No real listing or person is in
this repository.

## How it works

```
caller audio (20 ms frames)
  -> VAD endpointing and barge-in           speech/vad.py, speech/duplex.py
  -> faster-whisper small on CPU            speech/asr.py
  -> language and dialect ID                speech/langid.py
  -> number, date, time, phone normalizer   speech/normalize.py
  -> NLU into a strict JSON schema          dialogue/nlu.py, dialogue/schema.py
  -> deterministic policy                   dialogue/policy.py
  -> schema-validated tool calls            dialogue/tools.py
  -> booking API (the oracle)               api/service.py
  -> reply template, optional model rewording, grounding check
                                            dialogue/phrasing.py, dialogue/grounding.py
  -> Piper TTS in the caller's last language speech/tts.py
```

Three rules hold the design together:

- The booking API owns every fact. The agent may only say a slot, price or address the API returned in
  this session. A model reply that states anything else is rejected and the template is spoken. See
  [ADR 0002](docs/adr/0002-oracle-owns-facts.md) and `tests/test_grounding.py`, which injects an invented
  slot into the model's reply and checks it never reaches the caller.
- The model never picks the next step. A pure function of the slot state does. The model fills a form
  and, optionally, rewords a reply.
- Every tool call is checked against its JSON schema and against what this session has seen. Every
  booking action, accepted or rejected, goes into a hash-chained audit log.

Model choices and the smoke test behind them are in [ADR 0001](docs/adr/0001-model-choices.md).
Turn taking and barge-in are in [ADR 0003](docs/adr/0003-turn-taking.md). The Jev decision heads
(one forward pass, softmax over option labels, a calibrated threshold cascade) are in
[ADR 0004](docs/adr/0004-jev-decision-heads.md).

## Results

`uv run --extra speech --extra jev python -m eval.run_eval` plays eight scripted calls (two each in
English, Gulf Arabic, MSA and code-switched) through the whole pipeline in five configs: the rule parser
(`rules`), the 9B local model for NLU (`llm`), the 9B model also rewording replies (`llm+phrasing`), Jev
decision heads on Qwen3-1.7B with rule escalation (`jev`), and Jev heads plus 9B rewording checked by the
Jev judge (`jev+llm`). It and writes `eval/results.json`,
`eval/results.md` and every transcript in `eval/transcripts.json`. The tables below are generated from
`eval/results.json` by `eval/report.py`, and CI fails if they drift (`uv run python -m eval.report --check`).

Read these limits before the numbers:

- All caller audio is TTS. No consented Gulf Arabic recordings were available. The only offline Arabic
  voice is a Jordanian Piper speaker, so the Gulf rows measure Gulf wording in a Levantine voice, not a
  Gulf accent.
- The sample is small. Each row rests on a few calls and a few dozen turns.
- Latency was measured on the machine named in `eval/results.json`, with no GPU.

<!-- results:begin -->
Calls booked by the scripted callers, rule parser:

| calls booked | v1 (8 calls) | local | local, fuzzy parser | mixed prompt | two-pass decode | hosted |
|---|---|---|---|---|---|---|
| English | 2/2 | 6/10 | 8/10 | 7/10 | 7/10 | not run |
| Gulf Arabic | 2/2 | 8/10 | 8/10 | 4/10 | 3/10 | not run |
| MSA | 1/2 | 3/10 | 4/10 | 1/10 | 1/10 | not run |
| Code-switched | 0/2 | 0/10 | 0/10 | 0/10 | 1/10 | not run |
| All | 5/8 | 17/40 | 20/40 | 12/40 | 12/40 | not run |
| Wrong actions | n/a | 5 | 6 | 5 | 3 | not run |
| Replies with an ungrounded fact | 0 | 0 | 0 | 0 | 0 | not run |

Code-switched lines scored per language as in arXiv 2605.19069. WER on the English words: local 0.500, mixed prompt 0.237, two-pass decode 0.108, hosted not run. WER on the Arabic words: local 0.695, mixed prompt 0.808, two-pass decode 0.350, hosted not run. English words kept in Latin script: local 0.552, mixed prompt 0.861, two-pass decode 0.990, hosted not run.

The dialogue layer alone on the same calls, MTVA style (arXiv 2609.20152):

| transcript | transcript WER | rules intent | rules slot F1 | rules booked | rules, no carry-over booked | Jev intent | Jev booked | 9B intent | 9B booked | reply language kept |
|---|---|---|---|---|---|---|---|---|---|---|
| reference text | 0.000 | 0.895 | 0.977 | 38/40 | 38/40 | not run | not run | 0.794 | 36/40 | 1.000 |
| local ASR text | 0.390 | 0.642 | 0.623 | 17/40 | 18/40 | not run | not run | not run | not run | 0.906 |
| hosted ASR text | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run |
| 5% injected errors | 0.050 | 0.800 | 0.891 | 34/40 | 35/40 | not run | not run | not run | not run | 0.991 |
| 10% injected errors | 0.097 | 0.724 | 0.807 | 34/40 | 34/40 | not run | not run | not run | not run | 0.982 |
| 20% injected errors | 0.202 | 0.534 | 0.620 | 21/40 | 25/40 | not run | not run | not run | not run | 0.980 |
| 30% injected errors | 0.293 | 0.425 | 0.509 | 9/40 | 9/40 | not run | not run | not run | not run | 0.978 |
| split turns | 0.000 | 0.867 | 0.859 | 29/40 | 0/40 | not run | not run | not run | not run | 0.945 |

With the 9B model rewording replies: on reference text, 265 model replies spoken, 51 rejected, reply language kept 1.000, 0 replies with an ungrounded fact; on local ASR text, 454 model replies spoken, 70 rejected, reply language kept 0.905, 0 replies with an ungrounded fact.

Acoustic stress on the local config, TRACE style (arXiv 2609.29452), plot in `eval/trace.svg`:

| arm (local) | English | Gulf Arabic | MSA | Code-switched | wrong actions | recovered after trouble | extra turns | hosted |
|---|---|---|---|---|---|---|---|---|
| clean | 6/10 | 8/10 | 3/10 | 0/10 | 5 | 14/37 | 0.0 | not run |
| white@20 | 7/10 | 5/10 | 4/10 | 0/10 | 6 | 13/37 | 1.4 | not run |
| white@10 | 6/10 | 2/10 | 3/10 | 0/10 | 7 | 7/36 | -2.6 | not run |
| white@5 | 7/10 | 0/10 | 4/10 | 0/10 | 9 | 8/37 | 0.9 | not run |
| babble@20 | 6/10 | 6/10 | 6/10 | 0/10 | 4 | 14/36 | 0.1 | not run |
| babble@10 | 6/10 | 5/10 | 3/10 | 0/10 | 3 | 11/37 | -0.7 | not run |
| babble@5 | 6/10 | 0/10 | 4/10 | 0/10 | 7 | 7/37 | 0.2 | not run |
| reverb@0.3 | 7/10 | 6/10 | 1/10 | 0/10 | 11 | 11/37 | -1.6 | not run |
| reverb@0.8 | 6/10 | 2/10 | 2/10 | 0/10 | 6 | 8/38 | -0.8 | not run |
| competing | 6/10 | 3/10 | 3/10 | 0/10 | 13 | 10/38 | 0.2 | not run |

Turn latency on the development CPU, all turns, target 1.5 s. Timed from when the caller's 0.5 s pause closes the turn; early decode starts recognition on the pause's first frame:

| speech config | booked | p50 (s) | p95 (s) | p95 without early decode (s) |
|---|---|---|---|---|
| v1 (whisper small) | 5/8 | 2.38 | n/a | 4.17 |
| local (whisper small) | 17/40 | 1.78 | 4.26 | 4.76 |
| mixed prompt | 12/40 | 1.55 | 3.82 | 4.32 |
| two-pass decode | 12/40 | 1.82 | 4.74 | 5.24 |
| whisper base | 7/40 | 0.33 | 1.44 | 1.94 |
| whisper tiny | 6/40 | 0.16 | 1.22 | 1.68 |
| hosted | not run | not run | not run | not run |

Booking API faults on reference text (arXiv 2608.02372, 2606.31307), calls booked: clean 38/40, slow 9/40, empty 0/40, contradictory 0/40, contradictory-confirm 38/40, down 0/40, down-at-confirm 0/40. False facts spoken in all modes: 0. Calls that hit a fault and were handled safely: 232/232. Callbacks queued: 78. Hosted: not run.

Gulf dialect rubric (arXiv 2608.29990), calibrated score out of 1: agent templates 0.917 over 19 replies, 9B rewordings 0.679 over 40 (penalties: register flattening 24, wrong dialect 22, hallucination 10, ambiguous framing 3). Judge qwen/qwen3.6-35b-a3b agrees with 13 labelled anchors on 0.815 of criteria; native-speaker spot check not done yet.

Barge-in, agent stopped within 0.2 s of the caller: English 77/82 (v1 5/6), Gulf 117/121 (v1 10/11), MSA 132/139 (v1 17/18), code-switched 136/146 (v1 22/22). Slowest stop 0.40 s; false stops on the agent's own echo: 0.
<!-- results:end -->

What the numbers say:

- English works end to end. Both English calls book in every config.
- Code switching fails. No code-switched call completed in any config. Whisper small drops or
  transliterates the English inside an Arabic sentence ("Sunday" came back as "سندي"), so the slots are
  never heard. On the reference text the same NLU gets every slot right, so the loss is in ASR.
- Gulf and MSA calls fail on the phone number and on short answers. Whisper writes spoken Arabic digits
  as a mix of digits and words and loses some ("صفر 5 صفر 7 6 4 3 2 واحي"). A one-word "نعم" came back as
  "لا" in one MSA call, so the caller said yes and the agent heard no.
- The local model does not beat the rule parser here. It is slower on every row and its intent accuracy
  is lower. It also proposed slot values nobody said on many turns; the evidence check dropped them.
- p95 turn latency is above the 1.5 second target in every config and every language. Most of it is
  CPU speech recognition; with the local model, the model call adds more.
- No spoken reply contained an ungrounded fact in any config. With model rewording on, the grounding
  check rejected some model replies and the template was spoken instead.
- Barge-in stopped the agent within the bound in most trials. English, Gulf and MSA each have one
  miss, printed above.
- The Jev configs complete the same calls as the rule parser (English 2/2, Gulf 2/2, MSA 1/2,
  code-switched 0/2), one more Gulf call than the 9B model. Each turn costs one Qwen3-1.7B forward pass,
  so p95 over all turns is 5.8 s against 4.17 s for the rule parser and 6.05 s for the 9B model.
- On ASR text, Jev intent accuracy is above the rule parser in every Arabic row (Gulf 0.824 against
  0.714, MSA 0.583 against 0.476, code-switched 0.750 against 0.667). On reference text it is below
  the rule parser in every Arabic row, where the rules score 1.000. The rules were written with these
  scripts in view, which flatters them on clean text; the samples are small either way.
- Jev probabilities are poorly calibrated on the evaluation turns: ECE 0.272 over all turns, 0.05 for
  English and about 0.3 for each Arabic row, after temperature scaling. The calibration set was small
  (ADR 0004).
- The Jev dialect head is bad. On reference text it gets 1 of 12 English lines and 0 of 11 MSA lines
  right. The lexical ID in `speech/langid.py` does far better and is the one
  the agent uses.
- The Jev judge rejected 48 of 66 model-reworded replies in `jev+llm`, against 8 of 60 rejected by the
  deterministic check alone in `llm+phrasing`. It errs toward rejecting, which is the safe side: the
  template is spoken instead. It can never let a reply through on its own, because every "grounded"
  verdict is checked against the API facts.

## Run it

Needs Python 3.12 and [uv](https://docs.astral.sh/uv/). ffmpeg on PATH for browser audio.

```
uv sync                               # API, dialogue, tests
uv run pytest                         # hermetic suite: no model, no LM Studio, no sound card
uv sync --extra speech                # faster-whisper and Piper
uv run --extra speech python -m piper.download_voices --data-dir models/piper en_US-lessac-medium ar_JO-kareem-medium
uv run --extra speech uvicorn api.server:app --port 8000
```

Open http://127.0.0.1:8000, hold the button and speak. The page shows each line in Arabic script with a
Latin reading underneath, and plays the reply. Pressing the button while the agent speaks stops it.
The first audio turn downloads the Whisper model if it is not cached.

To use the local model for NLU, start LM Studio with `qwen/qwen3.5-9b` loaded and set `PARLEY_NLU=llm`.
The Jev heads need `uv sync --extra jev` (transformers and torch, CPU); `uv run --extra jev python -m
eval.jev_calibrate Qwen/Qwen3-1.7B --write` refits the temperatures and threshold, and
`PARLEY_JEV_THRESHOLD` overrides the threshold.
Other settings are listed at the top of `api/server.py`.

## Deploy

`deploy/k8s` holds the manifests: the service, Prometheus scraping `/metrics`, and Grafana with a
provisioned dashboard. The layout follows Tally.

```
make k8s-up      # gate 1: image builds; gate 2: pods ready (kind cluster)
make k8s-gates   # gate 3: one scripted booking completes; gate 4: deployed config matches the manifest
```

The cluster image leaves out the speech models, so in the cluster the agent is reached by text and NLU
runs on the rule parser. The CI workflow in `.github/workflows/ci.yml` runs all four gates on kind.

## What is not done

- The CI deployment gates have not run yet. Docker and kind are not available on the development
  machine. The manifests pass strict schema validation offline, and the booking gate script passed
  against the service run locally, but the first real cluster run will be the first CI run.
- No recorded demo is included. The demo page was not recorded on this machine.
- Barge-in was tested on synthetic and TTS audio streams, not a real microphone and speaker. The echo
  path is modelled as a fixed gain.
- Streaming means VAD endpointing plus one decode per utterance, not incremental decoding.
- No model was fine-tuned. Gulf ASR would need consented Gulf recordings for that. The optional LoRA
  tuning of the Jev model was not done either (ADR 0004).
- The Jev heads are evaluated in the harness only. The HTTP service does not offer a Jev config yet.
- Qwen-Audio-3.1-Realtime was not tried: it needs a GPU runtime this machine does not have.

## Layout

```
api/        booking domain, SQLite schema, seed generator, audit chain, HTTP service
speech/     VAD, streaming ASR, TTS, normalizer, language ID, barge-in, transliteration
dialogue/   NLU schema, rule and model NLU, policy, tool validator, grounding, phrasing, agent loop
eval/       smoke test, scripted callers, scorer, results
deploy/     Kubernetes manifests, kind config, CI gate scripts
web/        push-to-talk demo page
docs/adr/   decisions
```

## License

MIT.
