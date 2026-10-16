"""
evaluation/fetch_eval_sets.py: the three samples never overlap (offline - only the row choice is tested).

Run: .venv/bin/python -m pytest tests/test_fetch_eval_sets.py -v
"""

from evaluation import fetch_eval_sets as f


def test_extra_questions_are_new_and_reproducible():
    """An extra round never reuses an evaluation or gate-training question, and the same call gives the same rows."""
    eval_rows, gate_rows = f.split_indices(7600, f.EVAL_PER_SET, f.GATE_PER_SET)
    extra = f.extra_indices(7600, 200)
    assert len(extra) == 200 == len(set(extra))
    assert not set(extra) & (set(eval_rows) | set(gate_rows))
    assert extra == f.extra_indices(7600, 200)


def test_an_extra_only_source_is_used_whole():
    """A source the evaluation and gate sets never drew from has nothing to leave out."""
    assert f.extra_indices(120, 200, extra_only=True) == list(range(120))


def test_a_second_extra_round_never_repeats_the_first():
    """Round two of a source draws only from rows no earlier round used."""
    first = f.extra_indices(7600, 200)
    second = f.extra_indices(7600, 200, earlier=frozenset(first))
    assert len(second) == 200 and not set(first) & set(second)


def test_the_plain_arithmetic_variant_asks_as_a_user_would():
    """gsm8k_plain drops "Answer with just the number" and keeps the same correct answer."""
    row = {"question": "What is 2 + 3?", "answer": "2 + 3 = 5\n#### 5"}
    item = f.gsm8k_plain_item(7, row)
    assert item["messages"][0]["content"] == "What is 2 + 3?" and item["expected"] == "5" and item["id"] == "gsm8kplain_7"
