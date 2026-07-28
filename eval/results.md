# Evaluation results

Generated 2026-09-25T01:55:05+04:00 on AMD Ryzen 5 7600X3D 6-Core Processor (12 logical cores, no GPU). Source: `eval/results.json`. Caller audio: synthetic TTS only; Gulf lines are Gulf wording in a Jordanian Piper voice.

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
