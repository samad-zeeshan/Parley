from eval.wer import count_errors, wer
from speech.textnorm import normalize_orthography


def test_identical_is_zero():
    assert wer("a b c", "a b c") == 0.0


def test_counts_each_error_kind():
    c = count_errors("a b c d", "a x c d e")
    assert (c.substitutions, c.deletions, c.insertions, c.ref_words) == (1, 0, 1, 4)
    assert c.wer == 0.5


def test_deletion():
    assert count_errors("a b c", "a c").deletions == 1


def test_arabic_orthography_variants_are_not_errors():
    ref = normalize_orthography("أبي أحجز معاينةً في دبي، الساعة")
    hyp = normalize_orthography("ابي احجز معاينه في دبي الساعه")
    assert wer(ref, hyp) == 0.0


def test_english_case_and_punctuation():
    assert normalize_orthography("Hello, World!") == "hello world"
