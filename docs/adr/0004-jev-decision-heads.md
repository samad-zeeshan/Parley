# ADR 0004: Jev decision heads with a threshold cascade

Status: accepted, 2026-09-25

## Context

The local 9B model (ADR 0001) costs 2 to 4 seconds per NLU call on this CPU and invents slot values.
Most of what the agent needs from a model is a choice among options it already knows: which intent,
yes or no, which language variety, is this reply grounded. Open-Jev and JevLite (arXiv 2609.23959) and
JEV-as-a-Judge (arXiv 2609.26550) describe a readout for exactly that: declare the options, tag each with
a one-token label, run one forward pass, and take the softmax over the label logits at the last
position. No text is generated, so there is nothing to parse and nothing to invent. The papers pair it
with a cascade that accepts confident decisions and escalates the rest.

No code was published with the papers. `dialogue/jev.py` is our own implementation of the description.

## Decision

Four heads, each one forward pass of a small causal LM on CPU through transformers:

| head | options | used for |
|---|---|---|
| intent | the eleven intents in `dialogue/schema.py` | every caller turn, except after "shall I confirm?" |
| yes_no | yes, no | the turn after the agent asks to confirm |
| dialect | English, Gulf Arabic, MSA, mixed | reported next to the lexical ID; the agent does not act on it |
| judge | grounded, not grounded | a model-reworded reply, given the API facts |

Labels are the letters A to P, each one token for the Qwen3 tokenizer (checked at load). The prompt is
rendered with the chat template, thinking off, and ends at the start of the assistant turn.

Probabilities are `softmax(label_logits / T)`, with one temperature T per head fitted by grid search to
minimize negative log likelihood on a held-out fit slice (`eval/jev_calibrate.py`). The fitted values
are in `dialogue/jev_calibration.json`.

Cascade, with one threshold for all heads (`threshold` in the calibration file, overridable with
`PARLEY_JEV_THRESHOLD`):

- intent and yes_no: accept the head's choice when its probability is at or above the threshold,
  otherwise use the rule parser's intent.
- judge: a confident "not grounded" rejects the reply and the template is spoken. Anything else goes to
  the deterministic grounding check. A confident "grounded" is checked too: the oracle rule of ADR 0002
  outranks any model, so the judge can reject replies early but can never let one through.
- Slots never come from Jev. They stay with the rule parser (and the evidence check for the 9B model).

## Model choice, from the smoke test

Items: `eval/jev_calib_set.py` plus the ASR smoke set, none of them in the scripted calls the
evaluation scores. Split by alternation into a fit slice and a test slice. Test-slice numbers:

| model | head | test items | accuracy | ECE before scaling | ECE after scaling | T |
|---|---|---|---|---|---|---|
| Qwen3-0.6B | intent | 20 | 0.55 | 0.227 | 0.187 | 1.88 |
| Qwen3-0.6B | yes_no | 9 | 1.00 | 0.037 | 0.090 | 2.96 |
| Qwen3-0.6B | dialect | 19 | 0.368 | 0.283 | 0.089 | 2.96 |
| Qwen3-0.6B | judge | 15 | 0.333 | 0.429 | 0.173 | 40.07 |
| Qwen3-1.7B | intent | 20 | 0.75 | 0.229 | 0.147 | 2.36 |
| Qwen3-1.7B | yes_no | 9 | 0.778 | 0.219 | 0.172 | 6.54 |
| Qwen3-1.7B | dialect | 19 | 0.421 | 0.597 | 0.162 | 18.13 |
| Qwen3-1.7B | judge | 15 | 0.667 | 0.246 | 0.339 | 8.20 |

Forward pass on the reference machine (12 logical cores, float32): Qwen3-0.6B p50 0.563 s, p95 0.81 s;
Qwen3-1.7B p50 1.468 s, p95 1.995 s. Loading took 7.1 s and 83.9 s (the second includes the first
download).

0.6B is too weak: its judge is below chance on the test slice and its dialect head is at 0.368. We use
Qwen3-1.7B (`PARLEY_JEV_MODEL`). It is still weak on dialect, which the lexical ID in `speech/langid.py`
does better, so the dialect head is reported, not used.

Temperature scaling lowered ECE on six of the eight head and model pairs. It raised it for 0.6B yes_no
and for 1.7B judge. With 9 to 20 test items per head these are noisy estimates, and we print them as
measured.

Threshold: the lowest value on the grid {0.5, 0.6, 0.7, 0.8, 0.9, 0.95} that kept accepted decisions at
or above 0.9 accuracy on the fit slice. For 1.7B that is 0.7: 52.3 percent of fit-slice decisions
accepted, at 0.912 accuracy. 0.8 would accept 41.5 percent at 1.0.

## Consequences

- Each Jev turn adds one forward pass, about 1.5 s on this CPU, on top of ASR. In the evaluation that
  was about the same as the 9B model's NLU call (dialogue p50 1.3 to 2.2 s), so Jev does not bring turn
  latency near 1.5 s. Its gain is a probability per decision and no generated text, not speed.
- The calibration set is small and synthetic. The temperatures and the threshold should be refit on
  real labelled turns before anyone trusts the probabilities.
- LoRA tuning on turns from the scripted caller (the Triage-0.6B pattern) was not done. It would need a
  labelled set separate from the evaluation scripts, and the scripts are the only labelled turns we have.
