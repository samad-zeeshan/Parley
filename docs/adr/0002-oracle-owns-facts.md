# ADR 0002: The booking API owns every fact

Status: accepted, 2026-09-25

## Context

A voice agent that invents a viewing slot is worse than no agent. The caller turns up at a flat nobody
is showing, or at an address that does not exist. The model smoke test (ADR 0001) showed every local
model we tried filling in slots nobody said, including a phone number. So the design cannot rely on the
model behaving.

Touchstone used the rule "a tested deterministic oracle owns every fact, the model only phrases".
Change-Gate added "every tool call is validated against a schema before it runs, and every action goes
into a hash-chained audit log". This project applies both to speech.

## Decision

1. `api/service.py` is the oracle. Slots, prices, addresses and agent names exist only in its answers.
   It enforces the booking rules itself (holds expire, one active hold per slot, confirm idempotent on a
   key) and writes each action, accepted or rejected, to the audit chain.
2. The model never decides what happens next. `dialogue/policy.py` is a pure function from slot state to
   the next action. The model's only jobs are to fill the NLU form and, optionally, to reword a reply.
3. The NLU form is a strict JSON schema (`dialogue/schema.py`): a fixed intent set and six typed slots.
   Output that fails the schema is discarded and the rule parser runs instead. A slot value that passes
   the schema but has no support in the utterance (no matching normalizer entity, or a different reading
   from the rule parser) is dropped.
4. Every tool call goes through `dialogue/tools.py`. Arguments are checked against the tool's JSON
   schema. Then against the session: a hold may only target a slot the API returned in this session, a
   confirm or release only a hold this session made, a cancel only a booking this session made. A
   rejected call is written to the audit log with the reason.
5. Every reply is checked before it is spoken (`dialogue/grounding.py`). The reply is normalized with
   the same code as ASR output. Every date, time, number, phone number, area, tower and slot id in it
   must appear in the facts of this turn: the API results the policy acted on, plus what the caller said.
   A model reply that fails is replaced by the template for the same action. Templates are tested to
   pass the same check in both languages.

## What the test proves

`tests/test_grounding.py::test_injected_hallucinated_slot_is_rejected_and_template_is_spoken` feeds the
phraser a model reply that offers the real Thursday 16:00 slot and an invented Friday 14:30 slot. The
check flags the invented date, the template is spoken, and neither "Friday" nor "14:30" reaches the
caller. Other cases in the same file catch an invented price, tower, area, slot id, and the same
inventions in Arabic ("الجمعة", "مية ألف").

## Consequences

- The agent sounds more like a form than a person when the model is off or rejected. We accept that.
- The grounding check is lexical. It catches a wrong number or name, not a wrong claim built from right
  numbers ("the 16:00 slot is free all week"). Templates avoid such claims; model rewording is the risk,
  and it is optional.
- A reply may repeat what the caller said (their budget, their date), so an echo of a caller's mistake
  is allowed. The booking itself still only uses slots the API returned.
