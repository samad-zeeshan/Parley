# ADR 0001: Model choices for speech, dialogue and voice

Status: accepted, 2026-09-25

## Context

The agent must run on the reference machine with no cloud call in the request path. The machine is a
Windows 11 laptop with 12 logical CPU cores and no GPU. Callers speak English, Gulf Arabic, Modern
Standard Arabic (MSA), or switch between Arabic and English inside one sentence.

We need three models: speech to text (ASR), a small language model for intent and slot filling, and text
to speech (TTS) in both languages. Before building anything we checked what actually installs and runs
here, and measured ASR on a 20-utterance smoke set.

## The smoke set

`eval/smoke_set.py` holds 20 synthetic utterances: 6 English, 6 Gulf Arabic, 6 MSA, 2 code-switched.
There are no human recordings. We could not get consented Gulf Arabic recordings, so every clip is TTS
audio (see TTS below). This matters for how far the numbers can be trusted:

- The only offline Arabic voice we found is a Jordanian speaker. The "Gulf" clips are Gulf wording
  (أبي، عطني، عقب الظهر، مية) read in a Levantine voice. They test whether the model copes with Gulf
  words, not with a Gulf accent.
- Clean TTS audio is easier than a phone call. Real WER will be higher.
- Two code-switched clips is too few to rank models on. We report them anyway.

Run it with `uv run --extra speech python -m eval.smoke tiny base small`. Results are in
`eval/smoke_results.json`, with every hypothesis next to its reference.

## ASR: what we tried

All runs use faster-whisper (CTranslate2, int8, CPU), greedy decoding, no language hint, because the
caller may use either language. "prompt" means a bilingual vocabulary prompt with area names and booking
words in both scripts (`DOMAIN_PROMPT` in `eval/smoke.py`). It contains no smoke-set sentence.

WER below is after orthographic normalization only (case, punctuation, Arabic diacritics, alef and ta
marbuta variants). Numbers are not normalized here, so "120,000" against "one hundred and twenty
thousand" counts as errors. That is why English WER never drops below 0.27 even though the English
transcripts read correctly.

| model | English | Gulf | MSA | code-switched | median s per utterance |
|---|---|---|---|---|---|
| tiny | 0.339 | 0.562 | 0.562 | 0.643 | 0.48 |
| base | 0.339 | 0.375 | 0.438 | 1.000 | 0.88 |
| small | 0.290 | 0.125 | 0.458 | 0.857 | 2.86 |
| small + prompt | 0.274 | 0.104 | 0.438 | 0.786 | 4.05 |
| medium | 0.274 | 0.083 | 0.188 | 0.786 | 10.22 |
| medium + prompt | 0.274 | 0.062 | 0.167 | 0.643 | 9.36 |
| large-v3-turbo + prompt | 0.274 | 0.333 | 0.104 | 0.429 | 13.01 |

What the transcripts show:

- English is solved at every size above tiny apart from number formatting.
- Gulf wording is handled well from small upward. Common slips are single letters: أبضي for أبغي,
  فلاس for خلاص.
- MSA numbers are the weak point of small: "ميزانيتي مئتا ألف درهم سنويا" came back as
  "ميزانية مئة الفدرها من سنوية", and the MSA phone number collapsed into digits and noise.
- Code switching is bad at every size. Whisper transcribes the Arabic and drops or garbles the English
  segment ("two bedroom" became "تبدرون", or vanished). large-v3-turbo is the least bad and the slowest.
- We did not try Qwen-Audio-3.1-Realtime, the paper's suggested speech model. It needs a GPU runtime we
  do not have on this machine. We also did not try whisper.cpp, because faster-whisper already gave us
  the same model family on CPU.

Decision: faster-whisper small, int8, with the domain prompt, is the default (`MAJLIS_ASR_MODEL=small`).
It is the fastest model with usable Gulf WER. medium is 2.3 times slower for a gain that is mostly on
MSA; it is one environment variable away (`MAJLIS_ASR_MODEL=medium`) for anyone with a faster CPU. No
model here meets a 1.5 second turn budget on this CPU, so we pick for accuracy per second rather than
pretend otherwise.

Consequences:

- Numbers are normalized after ASR (`speech/normalize.py`) and scored separately (digit, date, time
  WER), because the table shows formatting alone costs 0.27 WER on English.
- Code-switched ASR is the largest known gap. The evaluation reports it as its own row and does not
  fold it into Arabic.
- If Gulf WER on real audio turns out much worse, the fix is dialect fine-tuning on consented Gulf
  recordings (the pattern in the Jordanian dialect paper), which we cannot do without data.

## Dialogue model

LM Studio serves GGUF models on this machine through an OpenAI-compatible API at
`http://127.0.0.1:1234/v1`. We sent three NLU requests (English booking request, Gulf booking request,
"الأول زين" as an option choice) to four served models with a strict JSON schema.

- With the models' default reasoning on, a request took 12 to 57 seconds and sometimes returned empty
  content because the whole token budget went to reasoning. `reasoning_effort: "none"` fixes both.
- With reasoning off, calls after the first (which includes loading the model) took: qwen3.5-9b 2.5 to
  3.4 s, gemma-4-12b 4.3 to 7.6 s, gemma-4-26b-a4b 3.2 to 7.0 s, qwen3.6-35b-a3b 3.4 to 7.3 s.
- Every model filled slots nobody said. qwen3.5-9b answered "a two bedroom flat in Dubai Marina" with
  a date, a time window and the phone number 0501234567. That is a fabricated fact, exactly what the
  oracle rule forbids.

Decision: qwen/qwen3.5-9b (GGUF, through LM Studio, reasoning off) for intent and slot filling. It is the
fastest of the four. Because it invents slot values, no slot from the model is accepted unless the
utterance itself supports it (a normalizer entity or the rule parser's reading of the same words).
A deterministic rule parser with the same output schema is the fallback when the model is unreachable
or returns invalid JSON, and it is what the hermetic tests use. The evaluation scores both.

We did not fine-tune a model. The Triage-0.6B LoRA and GGUF pattern would fit, but it needs labelled
dialect data we do not have, and the rule parser plus the evidence check already bound what the model
can do wrong.

## TTS

- Piper (onnx, CPU) installs cleanly and runs offline. `en_US-lessac-medium` for English and
  `ar_JO-kareem-medium` for Arabic. Both synthesize a sentence in under a second here.
- Windows SAPI has English voices only on this machine (David, Zira). No Arabic SAPI voice is installed.
  We use Zira and David as a second English voice in the evaluation so English is not all one speaker.
- No open Gulf Arabic TTS voice was found. The agent speaks Arabic with the Jordanian Piper voice.

Decision: Piper for the agent's voice in both languages; Piper plus SAPI for synthetic callers.
