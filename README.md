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
Turn taking and barge-in are in [ADR 0003](docs/adr/0003-turn-taking.md).

## Results

`uv run --extra speech python -m eval.run_eval` plays eight scripted calls (two each in English, Gulf
Arabic, MSA and code-switched) through the whole pipeline and writes `eval/results.json`,
`eval/results.md` and every transcript in `eval/transcripts.json`. The tables below are generated from
`eval/results.json` by `eval/report.py`, and CI fails if they drift (`uv run python -m eval.report --check`).

Read these limits before the numbers:

- All caller audio is TTS. No consented Gulf Arabic recordings were available. The only offline Arabic
  voice is a Jordanian Piper speaker, so the Gulf rows measure Gulf wording in a Levantine voice, not a
  Gulf accent.
- The sample is small. Each row rests on a few calls and a few dozen turns.
- Latency was measured on the machine named in `eval/results.json`, with no GPU.

<!-- results:begin -->
ASR (faster-whisper small, CPU) and dialect ID, per scripted line:

| language / dialect | lines | WER | WER after number normalization | digit WER | date WER | time WER | dialect ID on ASR text | dialect ID on reference text |
|---|---|---|---|---|---|---|---|---|
| English | 12 | 0.367 | 0.011 | 0.000 (33) | 0.000 (2) | 0.000 (4) | 1.000 | 1.000 |
| Gulf Arabic | 12 | 0.389 | 0.433 | 0.484 (31) | 0.500 (2) | 0.500 (4) | 0.583 | 0.750 |
| MSA | 11 | 0.197 | 0.123 | 0.136 (22) | 0.000 (2) | 0.000 (3) | 1.000 | 1.000 |
| Code-switched | 9 | 0.579 | 0.697 | 1.000 (3) | 1.000 (2) | 0.500 (4) | 0.111 | 0.889 |

The count in brackets is the number of reference tokens the digit, date or time WER is over.

Dialogue, config `rules` (deterministic rule parser):

| language / dialect | turns | intent accuracy | slot F1 | intent accuracy, reference text | slot F1, reference text | turn latency p50 (s) | turn latency p95 (s) | ASR p50 (s) | dialogue p50 (s) | TTS p50 (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| English | 10 | 1.000 | 1.000 | 1.000 | 1.000 | 2.41 | 3.23 | 2.09 | 0.001 | 0.28 |
| Gulf Arabic | 14 | 0.714 | 0.828 | 1.000 | 1.000 | 2.55 | 7.37 | 2.10 | 0.000 | 0.16 |
| MSA | 21 | 0.476 | 0.788 | 1.000 | 1.000 | 2.61 | 3.25 | 2.34 | 0.001 | 0.15 |
| Code-switched | 24 | 0.667 | 0.484 | 1.000 | 1.000 | 2.22 | 4.37 | 2.09 | 0.000 | 0.09 |

Task completion (booking in the database with the caller's phone and a valid audit chain): English 2/2, Gulf Arabic 2/2, MSA 1/2, Code-switched 0/2. All turns: p50 2.29 s, p95 4.18 s (above the 1.5 s target). Replies spoken with an ungrounded fact: 0. Model slots dropped for lack of evidence: 0. NLU fallbacks to rules: 0. Model replies spoken: 0; rejected by the grounding check: 0.

Dialogue, config `llm` (qwen/qwen3.5-9b via LM Studio (GGUF), reasoning off, evidence check on slots):

| language / dialect | turns | intent accuracy | slot F1 | intent accuracy, reference text | slot F1, reference text | turn latency p50 (s) | turn latency p95 (s) | ASR p50 (s) | dialogue p50 (s) | TTS p50 (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| English | 10 | 0.900 | 1.000 | 0.900 | 1.000 | 6.14 | 7.81 | 2.09 | 3.475 | 0.33 |
| Gulf Arabic | 15 | 0.600 | 0.828 | 0.867 | 0.970 | 5.48 | 10.26 | 2.12 | 2.821 | 0.17 |
| MSA | 19 | 0.421 | 0.788 | 0.895 | 1.000 | 4.74 | 7.26 | 2.47 | 2.013 | 0.20 |
| Code-switched | 24 | 0.583 | 0.484 | 0.750 | 1.000 | 4.43 | 6.38 | 2.09 | 1.801 | 0.09 |

Task completion (booking in the database with the caller's phone and a valid audit chain): English 2/2, Gulf Arabic 1/2, MSA 1/2, Code-switched 0/2. All turns: p50 4.70 s, p95 7.99 s (above the 1.5 s target). Replies spoken with an ungrounded fact: 0. Model slots dropped for lack of evidence: 79. NLU fallbacks to rules: 10. Model replies spoken: 0; rejected by the grounding check: 0.

Dialogue, config `llm+phrasing` (llm NLU plus qwen/qwen3.5-9b rewording replies, grounding check):

| language / dialect | turns | intent accuracy | slot F1 | intent accuracy, reference text | slot F1, reference text | turn latency p50 (s) | turn latency p95 (s) | ASR p50 (s) | dialogue p50 (s) | TTS p50 (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| English | 10 | 0.900 | 1.000 | 0.900 | 1.000 | 8.69 | 11.06 | 2.09 | 6.175 | 0.30 |
| Gulf Arabic | 15 | 0.600 | 0.828 | 0.867 | 0.970 | 6.43 | 12.26 | 2.12 | 3.772 | 0.21 |
| MSA | 19 | 0.421 | 0.788 | 0.895 | 1.000 | 5.09 | 7.48 | 2.47 | 2.722 | 0.22 |
| Code-switched | 24 | 0.583 | 0.484 | 0.750 | 1.000 | 3.95 | 6.17 | 2.09 | 1.651 | 0.11 |

Task completion (booking in the database with the caller's phone and a valid audit chain): English 2/2, Gulf Arabic 1/2, MSA 1/2, Code-switched 0/2. All turns: p50 5.31 s, p95 9.94 s (above the 1.5 s target). Replies spoken with an ungrounded fact: 0. Model slots dropped for lack of evidence: 83. NLU fallbacks to rules: 10. Model replies spoken: 60; rejected by the grounding check: 8.

Barge-in (caller line played over the agent's real Piper reply, agent echo in the mic at 0.25 gain, success means stopped within 0.2 s of the caller's first voiced frame):

| interrupting caller | trials | success | success rate | latency p50 (s) | latency max (s) | false stops on echo alone |
|---|---|---|---|---|---|---|
| English | 6 | 6 | 1.000 | 0.100 | 0.200 | 0 |
| Gulf Arabic | 11 | 11 | 1.000 | 0.120 | 0.160 | 0 |
| MSA | 17 | 16 | 0.941 | 0.100 | 0.300 | 0 |
| Code-switched | 22 | 22 | 1.000 | 0.100 | 0.160 | 0 |
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
- Barge-in stopped the agent within the bound in most trials. The MSA row has one miss, printed above.

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
- No model was fine-tuned. Gulf ASR would need consented Gulf recordings for that.
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
