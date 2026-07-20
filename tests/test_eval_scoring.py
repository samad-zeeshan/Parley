"""Scoring functions of the evaluation harness."""

from datetime import date

import pytest

from eval.scoring import entity_errors, percentile, slot_counts

TODAY = date(2026, 10, 1)


def test_digit_errors_count_single_digits_of_a_phone():
    c = entity_errors("my number is 0501234567", "my number is 0501234568", {"phone", "number"}, TODAY)
    assert (c.ref_words, c.errors) == (10, 1)


def test_digit_errors_ignore_formatting():
    c = entity_errors("one hundred and twenty thousand dirhams", "120,000 dirhams", {"phone", "number"}, TODAY)
    assert (c.ref_words, c.errors) == (1, 0)


def test_missing_number_is_a_deletion():
    c = entity_errors("budget 120000", "budget", {"number"}, TODAY)
    assert c.deletions == 1 and c.wer == 1.0


def test_date_and_time_errors():
    assert entity_errors("Thursday at 4 pm", "Friday at 4 pm", {"date"}, TODAY).errors == 1
    assert entity_errors("Thursday at 4 pm", "Friday at 4 pm", {"time", "time_window"}, TODAY).errors == 0
    c = entity_errors("between four and six pm", "between four and seven pm", {"time", "time_window"}, TODAY)
    assert (c.ref_words, c.errors) == (2, 1)


def test_no_entities_in_reference():
    c = entity_errors("yes please", "yes please", {"date"}, TODAY)
    assert c.ref_words == 0


def test_slot_counts():
    tp, fp, fn = slot_counts({"area": "Dubai Marina", "bedrooms": 2}, {"area": "Dubai Marina", "bedrooms": 3})
    assert (tp, fp, fn) == (1, 1, 1)
    assert slot_counts({}, {}) == (0, 0, 0)
    assert slot_counts({"time_window": ["16:00", "18:00"]}, {"time_window": ["16:00", "18:00"]}) == (1, 0, 0)


@pytest.mark.parametrize("q,expected", [(50, 3.0), (95, 4.8), (0, 1.0), (100, 5.0)])
def test_percentile_linear(q, expected):
    assert percentile([5, 1, 4, 2, 3], q) == pytest.approx(expected)
