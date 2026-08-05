# Evaluation results

Generated 2026-09-25T14:39:18+04:00 on AMD Ryzen 5 7600X3D 6-Core Processor (12 logical cores, no GPU). Source: `eval/results.json`. Caller audio: synthetic TTS only; Gulf lines are Gulf wording in a Jordanian Piper voice.

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
