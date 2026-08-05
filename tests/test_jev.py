"""Jev-style decision heads with a fake model that returns fixed logits.

No transformers, no weights: FixedLogits stands in for the one forward pass.
"""

import math
from datetime import date

import pytest

from dialogue.grounding import Facts
from dialogue.jev import (
    HEADS,
    FixedLogits,
    JevDecider,
    JevNLU,
    expected_calibration_error,
    fit_temperature,
    softmax,
)

TODAY = date(2026, 10, 1)

SLOT = {
    "slot_id": "s-p001-100116", "starts_at": "2026-10-01T16:00", "ends_at": "2026-10-01T16:30",
    "property_ref": "MJ-001", "emirate": "Dubai", "area": "Dubai Marina", "area_ar": "دبي مارينا",
    "address": "Unit 1204, Tower C3, Dubai Marina, Dubai", "address_ar": "شقة 1204، برج ت3، دبي مارينا، دبي",
    "bedrooms": 2, "annual_rent_aed": 155000, "agent_name": "Omar", "agent_name_ar": "عمر",
}


def decider(logits_for, threshold=0.7, temperatures=None):
    return JevDecider(FixedLogits(logits_for), threshold=threshold, temperatures=temperatures or {})


def peaked(index, n, high=6.0):
    return [high if i == index else 0.0 for i in range(n)]


# ---- the readout ----------------------------------------------------------------

def test_probabilities_sum_to_one_for_every_head():
    d = decider(lambda head, prompt, n: [0.3 * i for i in range(n)])
    for name in HEADS:
        probs = d.probabilities(name, **HEADS[name].example)
        assert set(probs) == set(HEADS[name].options)
        assert sum(probs.values()) == pytest.approx(1.0)


def test_one_forward_pass_per_decision():
    model = FixedLogits(lambda head, prompt, n: [0.0] * n)
    JevDecider(model).probabilities("yes_no", text="yes", question="Shall I confirm?")
    assert model.calls == 1


def test_options_are_declared_with_single_letter_labels_in_the_prompt():
    seen = {}

    def capture(head, prompt, n):
        seen["prompt"] = prompt
        return [0.0] * n

    decider(capture).probabilities("dialect", text="hello")
    for label, option in zip("ABCD", HEADS["dialect"].options):
        assert f"{label}) " in seen["prompt"]


def test_temperature_scaling():
    logits = [2.0, 0.0, 0.0]
    sharp, flat = softmax(logits, 1.0), softmax(logits, 4.0)
    assert sharp[0] > flat[0] > 1 / 3
    assert softmax(logits, 1e6)[0] == pytest.approx(1 / 3, abs=1e-3)
    d = decider(lambda h, p, n: peaked(0, n, 2.0), temperatures={"yes_no": 3.0})
    p = d.probabilities("yes_no", text="yes", question="q")
    assert p["yes"] == pytest.approx(math.exp(2 / 3) / (math.exp(2 / 3) + 1))


def test_fit_temperature_cools_an_overconfident_head():
    # Always 6 logits ahead, right only half the time: the fitted T must be well above 1.
    rows = [([6.0, 0.0], 0), ([6.0, 0.0], 1)] * 20
    t = fit_temperature(rows)
    assert t > 3.0
    assert fit_temperature([([3.0, 0.0], 0)] * 30) < 1.0  # always right: sharpen


def test_expected_calibration_error():
    assert expected_calibration_error([(1.0, True)] * 10) == pytest.approx(0.0)
    assert expected_calibration_error([(0.9, False)] * 10) == pytest.approx(0.9)
    assert expected_calibration_error([(0.6, True), (0.6, False)]) == pytest.approx(0.1)


# ---- cascade --------------------------------------------------------------------

def test_confident_intent_is_accepted():
    idx = HEADS["intent"].options.index("cancel_booking")
    nlu = JevNLU(decider(lambda h, p, n: peaked(idx, n)))
    out = nlu.parse("please drop it", TODAY)
    assert out.intent == "cancel_booking" and out.source == "jev"
    assert out.confidence >= 0.7


def test_uncertain_intent_escalates_to_the_rule_parser():
    nlu = JevNLU(decider(lambda h, p, n: [0.0] * n))   # uniform: max p = 1/11
    out = nlu.parse("yes please confirm", TODAY)
    assert out.source == "jev-escalated" and out.intent == "confirm"


def test_threshold_is_configurable():
    idx = HEADS["intent"].options.index("goodbye")
    logits = lambda h, p, n: peaked(idx, n, 2.0)          # noqa: E731 -- about 0.4 on the winner
    assert JevNLU(decider(logits, threshold=0.3)).parse("x", TODAY).source == "jev"
    assert JevNLU(decider(logits, threshold=0.9)).parse("x", TODAY).source == "jev-escalated"


def test_yes_no_head_answers_a_confirmation_question():
    nlu = JevNLU(decider(lambda h, p, n: peaked(1, n) if h == "yes_no" else [0.0] * n))
    out = nlu.parse("mmm la", TODAY, context="ask_confirm")
    assert out.intent == "deny" and out.head == "yes_no" and out.source == "jev"


def test_jev_never_fills_slots():
    idx = HEADS["intent"].options.index("book_viewing")
    out = JevNLU(decider(lambda h, p, n: peaked(idx, n))).parse("a two bedroom in Dubai Marina", TODAY)
    assert out.slots == {"area": "Dubai Marina", "bedrooms": 2}   # the rule parser's reading


# ---- judge ----------------------------------------------------------------------

def test_judge_rejects_an_ungrounded_slot_when_confident():
    d = decider(lambda h, p, n: peaked(1, n))              # B = not grounded
    v = d.judge("I also have Friday at 14:30.", [SLOT], TODAY)
    assert v.grounded is False and v.source == "jev"


def test_uncertain_judge_escalates_to_the_grounding_check():
    d = decider(lambda h, p, n: [0.0, 0.0])
    bad = d.judge("I also have Friday at 14:30.", [SLOT], TODAY)
    good = d.judge("Thursday at 16:00 in Dubai Marina.", [SLOT], TODAY)
    assert bad.source == good.source == "check"
    assert bad.grounded is False and good.grounded is True


def test_a_confident_wrong_judge_cannot_let_an_invented_slot_through():
    # The oracle rule outranks the judge: "grounded" from Jev is still checked.
    d = decider(lambda h, p, n: peaked(0, n))              # A = grounded, confidently wrong
    v = d.judge("I also have Friday at 14:30.", [SLOT], TODAY)
    assert v.grounded is False and v.overruled


def test_judge_prompt_carries_the_api_facts():
    seen = {}

    def capture(head, prompt, n):
        seen["p"] = prompt
        return [0.0] * n

    decider(capture).judge("x", [SLOT], TODAY)
    assert "16:00" in seen["p"] and "155000" in seen["p"] and "Tower C3" in seen["p"]


# ---- dialect head on reference text ---------------------------------------------

@pytest.mark.parametrize("text,label", [
    ("I want to book a viewing", "english"),
    ("أبي أحجز معاينة", "gulf"),
    ("أريد حجز موعد", "msa"),
    ("أبي شقة two bedroom", "switch"),
])
def test_dialect_head_reads_the_label_the_model_prefers(text, label):
    order = HEADS["dialect"].options
    # A fake model that "knows" the answer through the reference text's script and markers.
    from speech.langid import identify

    gold = {"en": "english", "ar-gulf": "gulf", "ar-msa": "msa", "mixed": "switch"}[identify(text).label]
    d = decider(lambda h, p, n: peaked(order.index(gold), n))
    probs = d.probabilities("dialect", text=text)
    assert max(probs, key=probs.get) == label


def test_facts_helper_matches_grounding():
    assert Facts.from_slots([SLOT]).times == {"16:00", "16:30"}
