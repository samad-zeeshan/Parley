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
ASR (faster-whisper small, CPU) and dialect ID, per scripted line:

| language / dialect | lines | WER | WER after number normalization | digit WER | date WER | time WER | dialect ID on ASR text | dialect ID on reference text |
|---|---|---|---|---|---|---|---|---|
| English | 12 | 0.367 | 0.011 | 0.000 (33) | 0.000 (2) | 0.000 (4) | 1.000 | 1.000 |
| Gulf Arabic | 12 | 0.389 | 0.433 | 0.484 (31) | 0.500 (2) | 0.500 (4) | 0.583 | 0.750 |
| MSA | 11 | 0.197 | 0.123 | 0.136 (22) | 0.000 (2) | 0.000 (3) | 1.000 | 1.000 |
| Code-switched | 9 | 0.579 | 0.697 | 1.000 (3) | 1.000 (2) | 0.500 (4) | 0.111 | 0.889 |

The count in brackets is the number of reference tokens the digit, date or time WER is over.

Dialogue, config `rules` (deterministic rule parser):

| language / dialect | turns | intent accuracy | slot F1 | intent accuracy, reference text | slot F1, reference text | turn latency p50 (s) | turn latency p95 (s) | ASR p50 (s) | dialogue p50 (s) | TTS p50 (s) | Jev ECE | Jev accepted share |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| English | 10 | 1.000 | 1.000 | 1.000 | 1.000 | 2.48 | 3.32 | 2.09 | 0.001 | 0.34 | n/a | n/a |
| Gulf Arabic | 14 | 0.714 | 0.828 | 1.000 | 1.000 | 2.59 | 7.42 | 2.10 | 0.001 | 0.18 | n/a | n/a |
| MSA | 21 | 0.476 | 0.788 | 1.000 | 1.000 | 2.62 | 3.27 | 2.34 | 0.001 | 0.19 | n/a | n/a |
| Code-switched | 24 | 0.667 | 0.484 | 1.000 | 1.000 | 2.20 | 4.36 | 2.09 | 0.000 | 0.08 | n/a | n/a |

Task completion (booking in the database with the caller's phone and a valid audit chain): English 2/2, Gulf Arabic 2/2, MSA 1/2, Code-switched 0/2. All turns: p50 2.38 s, p95 4.17 s (above the 1.5 s target). Replies spoken with an ungrounded fact: 0. Model slots dropped for lack of evidence: 0. NLU fallbacks to rules: 0. Model replies spoken: 0; rejected by the grounding check: 0.

Dialogue, config `llm` (qwen/qwen3.5-9b via LM Studio (GGUF), reasoning off, evidence check on slots):

| language / dialect | turns | intent accuracy | slot F1 | intent accuracy, reference text | slot F1, reference text | turn latency p50 (s) | turn latency p95 (s) | ASR p50 (s) | dialogue p50 (s) | TTS p50 (s) | Jev ECE | Jev accepted share |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| English | 10 | 0.900 | 1.000 | 0.900 | 1.000 | 4.99 | 5.94 | 2.09 | 2.231 | 0.36 | n/a | n/a |
| Gulf Arabic | 15 | 0.600 | 0.828 | 0.867 | 0.970 | 4.94 | 8.88 | 2.12 | 1.981 | 0.17 | n/a | n/a |
| MSA | 19 | 0.421 | 0.788 | 0.895 | 1.000 | 4.25 | 6.11 | 2.47 | 1.326 | 0.21 | n/a | n/a |
| Code-switched | 24 | 0.583 | 0.484 | 0.750 | 1.000 | 3.63 | 5.60 | 2.09 | 1.272 | 0.10 | n/a | n/a |

Task completion (booking in the database with the caller's phone and a valid audit chain): English 2/2, Gulf Arabic 1/2, MSA 1/2, Code-switched 0/2. All turns: p50 4.16 s, p95 6.05 s (above the 1.5 s target). Replies spoken with an ungrounded fact: 0. Model slots dropped for lack of evidence: 80. NLU fallbacks to rules: 10. Model replies spoken: 0; rejected by the grounding check: 0.

Dialogue, config `llm+phrasing` (llm NLU plus qwen/qwen3.5-9b rewording replies, grounding check):

| language / dialect | turns | intent accuracy | slot F1 | intent accuracy, reference text | slot F1, reference text | turn latency p50 (s) | turn latency p95 (s) | ASR p50 (s) | dialogue p50 (s) | TTS p50 (s) | Jev ECE | Jev accepted share |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| English | 10 | 0.900 | 1.000 | 0.900 | 1.000 | 6.08 | 8.99 | 2.09 | 3.518 | 0.26 | n/a | n/a |
| Gulf Arabic | 15 | 0.600 | 0.828 | 0.867 | 0.970 | 6.03 | 10.73 | 2.12 | 2.708 | 0.26 | n/a | n/a |
| MSA | 19 | 0.421 | 0.788 | 0.895 | 1.000 | 5.25 | 7.94 | 2.47 | 2.633 | 0.21 | n/a | n/a |
| Code-switched | 24 | 0.583 | 0.484 | 0.750 | 1.000 | 4.10 | 6.32 | 2.09 | 1.704 | 0.12 | n/a | n/a |

Task completion (booking in the database with the caller's phone and a valid audit chain): English 2/2, Gulf Arabic 1/2, MSA 1/2, Code-switched 0/2. All turns: p50 5.19 s, p95 8.35 s (above the 1.5 s target). Replies spoken with an ungrounded fact: 0. Model slots dropped for lack of evidence: 85. NLU fallbacks to rules: 10. Model replies spoken: 60; rejected by the grounding check: 8.

Dialogue, config `jev` (Jev intent and yes/no heads with rule escalation, rule slots, templates):

| language / dialect | turns | intent accuracy | slot F1 | intent accuracy, reference text | slot F1, reference text | turn latency p50 (s) | turn latency p95 (s) | ASR p50 (s) | dialogue p50 (s) | TTS p50 (s) | Jev ECE | Jev accepted share |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| English | 10 | 1.000 | 1.000 | 1.000 | 1.000 | 4.05 | 5.01 | 2.09 | 1.592 | 0.35 | 0.050 | 0.900 |
| Gulf Arabic | 17 | 0.824 | 0.812 | 0.588 | 1.000 | 3.89 | 7.44 | 2.07 | 1.552 | 0.21 | 0.339 | 0.529 |
| MSA | 24 | 0.583 | 0.826 | 0.625 | 1.000 | 4.20 | 4.97 | 2.27 | 1.662 | 0.18 | 0.331 | 0.875 |
| Code-switched | 24 | 0.750 | 0.484 | 0.708 | 1.000 | 3.88 | 6.21 | 2.09 | 1.578 | 0.11 | 0.305 | 0.833 |

Task completion (booking in the database with the caller's phone and a valid audit chain): English 2/2, Gulf Arabic 2/2, MSA 1/2, Code-switched 0/2. All turns: p50 4.04 s, p95 5.80 s (above the 1.5 s target). Replies spoken with an ungrounded fact: 0. Model slots dropped for lack of evidence: 0. NLU fallbacks to rules: 0. Model replies spoken: 0; rejected by the grounding check: 0. Jev ECE over all turns: 0.272.

Dialogue, config `jev+llm` (jev NLU plus qwen/qwen3.5-9b rewording replies, Jev judge then grounding check):

| language / dialect | turns | intent accuracy | slot F1 | intent accuracy, reference text | slot F1, reference text | turn latency p50 (s) | turn latency p95 (s) | ASR p50 (s) | dialogue p50 (s) | TTS p50 (s) | Jev ECE | Jev accepted share |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| English | 10 | 1.000 | 1.000 | 1.000 | 1.000 | 6.70 | 11.10 | 2.09 | 4.394 | 0.29 | 0.050 | 0.900 |
| Gulf Arabic | 17 | 0.824 | 0.812 | 0.588 | 1.000 | 6.04 | 12.13 | 2.07 | 3.644 | 0.18 | 0.339 | 0.529 |
| MSA | 24 | 0.583 | 0.826 | 0.625 | 1.000 | 6.04 | 8.99 | 2.27 | 3.637 | 0.18 | 0.331 | 0.875 |
| Code-switched | 24 | 0.750 | 0.484 | 0.708 | 1.000 | 5.34 | 7.45 | 2.09 | 2.910 | 0.11 | 0.305 | 0.833 |

Task completion (booking in the database with the caller's phone and a valid audit chain): English 2/2, Gulf Arabic 2/2, MSA 1/2, Code-switched 0/2. All turns: p50 5.98 s, p95 10.10 s (above the 1.5 s target). Replies spoken with an ungrounded fact: 0. Model slots dropped for lack of evidence: 0. NLU fallbacks to rules: 0. Model replies spoken: 18; rejected by the grounding check: 57. Jev ECE over all turns: 0.272. Judge: 66 replies judged, 51 decided by Jev, 15 escalated to the check, 0 Jev 'grounded' verdicts overruled by the check, 48 rejected.

Jev dialect head (Qwen/Qwen3-1.7B, one forward pass per line):

| language / dialect | lines | accuracy, reference text | ECE, reference text | accuracy, ASR text | ECE, ASR text |
|---|---|---|---|---|---|
| English | 12 | 0.083 | 0.330 | 0.083 | 0.314 |
| Gulf Arabic | 12 | 0.833 | 0.478 | 1.000 | 0.515 |
| MSA | 11 | 0.000 | 0.494 | 0.000 | 0.490 |
| Code-switched | 9 | 0.111 | 0.434 | 0.222 | 0.240 |

Barge-in (caller line played over the agent's real Piper reply, agent echo in the mic at 0.25 gain, success means stopped within 0.2 s of the caller's first voiced frame):

| interrupting caller | trials | success | success rate | latency p50 (s) | latency max (s) | false stops on echo alone |
|---|---|---|---|---|---|---|
| English | 6 | 5 | 0.833 | 0.120 | 0.220 | 0 |
| Gulf Arabic | 11 | 10 | 0.909 | 0.120 | 0.320 | 0 |
| MSA | 18 | 17 | 0.944 | 0.120 | 0.220 | 0 |
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
