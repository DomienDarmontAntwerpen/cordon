"""
evaluation/benchmark.py: scoring, the FINAL-line protocol, bootstrap verdicts, gate analysis, the routing table.

Run: .venv/bin/python -m pytest tests/test_benchmark.py -v
"""

from evaluation import benchmark


def test_score_extraction_exact_and_numeric_tolerance():
    """score extraction exact and numeric tolerance."""
    expected = {"invoice_number": "INV-1", "amount": 100.0}
    assert benchmark.score_extraction(expected, '{"invoice_number": "INV-1", "amount": 100.0}') == 1.0
    assert benchmark.score_extraction(expected, '{"invoice_number": "INV-1", "amount": 100.4}') == 1.0  # 0.4 is within the 1.0 (1%) numeric tolerance
    assert benchmark.score_extraction(expected, '{"invoice_number": "INV-2", "amount": 100.0}') == 0.5
    assert benchmark.score_extraction(expected, "not json") == 0.0


def test_score_code_pass_and_fail():
    """score code pass and fail."""
    good_answer = "```python\ndef add(a, b):\n    return a + b\n```"
    bad_answer = "```python\ndef add(a, b):\n    return a - b\n```"
    test_code = "assert add(2, 3) == 5"
    assert benchmark.score_code(test_code, good_answer) == 1.0
    assert benchmark.score_code(test_code, bad_answer) == 0.0


def test_rouge_l_identical_and_unrelated_text():
    """rouge l identical and unrelated text."""
    assert benchmark.rouge_l("the cat sat on the mat", "the cat sat on the mat") == 1.0
    assert benchmark.rouge_l("the cat sat on the mat", "completely unrelated content here") < 0.3


def test_heuristic_judge_rewards_word_overlap():
    """heuristic judge rewards word overlap."""
    reference = "Revenue grew 12 percent on strong European demand"
    close = benchmark.heuristic_judge(reference, "Revenue grew 12 percent on strong European demand this quarter")
    far = benchmark.heuristic_judge(reference, "The weather was nice today")
    assert close > far


def test_final_line_protocol_scores_only_the_final_answer():
    """Only the FINAL line counts: negations and worked answers in the text cannot fool or penalise the score."""
    score = benchmark.score_final
    assert score("positive", "The sentiment is not negative; it is positive.\nFINAL: positive", "classification") == 1.0
    assert score("spam", "No, this is not a spam email.\nFINAL: not spam", "classification") == 0.0
    assert score("not_spam", "FINAL: Not spam", "classification") == 1.0
    assert score("positive", "positive, negative, or neutral", "classification") == 0.0          # no FINAL line
    assert score("4", "The change is 4 euros (10 - 6).\nFINAL: 4", "reasoning") == 1.0
    assert score("150", "From 09:15 to 11:45 is 2.5 hours.\nFINAL: 150 minutes", "reasoning") == 1.0
    assert score("4", "FINAL: 14", "reasoning") == 0.0
    assert score("seven", "FINAL: 7", "factual") == 1.0
    assert score("Au", "FINAL: because it is gold", "factual") == 0.0
    assert score("Au", "**FINAL:** Au.", "factual") == 1.0
    assert score("George Orwell", "FINAL: George Orwell", "factual") == 1.0
    # Found in the first live run: models often put FINAL at the end of the sentence, not on a new line.
    assert score("Paris", "The capital of France is Paris. FINAL: Paris", "factual") == 1.0
    assert score("1945", "World War II ended in 1945. FINAL: 1945", "factual") == 1.0
    assert score("4", "First 6, then 4. **Final:** 4", "reasoning") == 1.0
    assert score("4", "The final step gives 4 euros.", "reasoning") == 0.0  # "final" without a colon is not a FINAL line


def test_golden_set_items_carry_the_final_instruction():
    """Classification, factual and reasoning items ask for the FINAL line; other items are unchanged."""
    items = benchmark.load_golden_set(benchmark.GOLDEN_SET_PATH)
    for item in items:
        last = item["messages"][-1]["content"]
        assert (benchmark.FINAL_INSTRUCTION in last) == (item["category"] in benchmark.FINAL_CATEGORIES)


def test_verdict_pass_when_no_regression():
    """verdict pass when no regression."""
    diffs = [0.0] * 40  # D always equals A, on enough items
    lower, upper = benchmark.paired_bootstrap_ci(diffs)
    assert benchmark.verdict_from_ci(lower, upper, len(diffs)) == "pass"


def test_verdict_is_inconclusive_on_too_few_items():
    """Ten identical scores give a (0, 0) interval, but that is no proof of 'no loss': the verdict must be warn."""
    diffs = [0.0] * 10
    lower, upper = benchmark.paired_bootstrap_ci(diffs)
    assert (lower, upper) == (0.0, 0.0)
    assert benchmark.verdict_from_ci(lower, upper, len(diffs)) == "warn"


def test_verdict_fail_when_clear_regression():
    """verdict fail when clear regression."""
    diffs = [-0.5] * 20  # D is always much worse than A
    lower, upper = benchmark.paired_bootstrap_ci(diffs)
    assert benchmark.verdict_from_ci(lower, upper) == "fail"


def test_verdict_warn_when_inconclusive():
    """verdict warn when inconclusive."""
    # A small, noisy sample straddling the -0.03 margin - neither a clear pass nor a clear fail.
    diffs = [0.1, -0.1, 0.0, -0.05, 0.05, -0.2, 0.15, -0.03, 0.02, -0.08]
    lower, upper = benchmark.paired_bootstrap_ci(diffs)
    assert benchmark.verdict_from_ci(lower, upper) == "warn"


def test_bootstrap_ci_is_deterministic_given_seed():
    """bootstrap ci is deterministic given seed."""
    diffs = [0.1, -0.2, 0.05, 0.0, -0.1, 0.2, -0.05, 0.15, 0.0, -0.1]
    first = benchmark.paired_bootstrap_ci(diffs, seed=42)
    second = benchmark.paired_bootstrap_ci(diffs, seed=42)
    assert first == second


def test_simulate_success_b_and_d_never_beat_a_baseline_failure():
    """
    Regression test for the CI-permanently-red bug: B and D must never
    succeed on an item where A (the baseline draw) failed. B/D model "A,
    plus Cordon's processing", so an independent draw could make D randomly
    *luckier* than A on a small sample purely by chance - which previously
    produced a deterministic (not flaky) false ❌ verdict for the
    "classification" category at seed=42. If this ever regresses, that
    permanent failure mode comes back silently.
    """
    aliases = ["mistral-small", "gpt-4o-mini", "mistral-large", "gpt-4o"]
    checked_a_failure = False
    for alias in aliases:
        for i in range(200):
            item_id = f"item-{i}"
            a_ok = benchmark._simulate_success(item_id, "A", alias)
            if not a_ok:
                checked_a_failure = True
                assert benchmark._simulate_success(item_id, "B", alias) is False
                assert benchmark._simulate_success(item_id, "D", alias) is False
    # Guard the test itself: if no A failure ever showed up across 800 draws,
    # the assertions above never ran and this test would pass vacuously.
    assert checked_a_failure


def test_simulate_success_c_is_independent_of_a():
    """
    Condition C (a genuinely different model) is deliberately NOT gated on
    A's draw - comparing an unrelated model's own capability to the
    baseline is the entire point of condition C, so unlike B/D it may
    succeed even when A failed for the same item.
    """
    found_c_succeeding_where_a_failed = False
    for i in range(200):
        item_id = f"item-{i}"
        if not benchmark._simulate_success(item_id, "A", "gpt-4o-mini"):
            if benchmark._simulate_success(item_id, "C", "gpt-4o-mini"):
                found_c_succeeding_where_a_failed = True
                break
    assert found_c_succeeding_where_a_failed


def test_load_golden_set_has_sixty_items_ten_per_category():
    """load golden set has sixty items ten per category."""
    items = benchmark.load_golden_set(benchmark.GOLDEN_SET_PATH)
    assert len(items) == 60
    from collections import Counter

    counts = Counter(item["category"] for item in items)
    assert all(counts[c] == 10 for c in benchmark.CATEGORIES)


def test_load_golden_set_limit_is_per_category():
    """--limit N keeps the first N items of every category, so a smoke run covers all of them."""
    from collections import Counter

    items = benchmark.load_golden_set(benchmark.GOLDEN_SET_PATH, limit=2)
    assert Counter(i["category"] for i in items) == {c: 2 for c in benchmark.CATEGORIES}


def test_a_failed_judge_is_a_failed_call_not_a_neutral_score(monkeypatch):
    """A judge call that fails, or gives no 1-5 rating, raises - it is never scored as a neutral 3."""
    import pytest

    config = {"provider": {"timeout_seconds": 1}}
    monkeypatch.setattr(benchmark.litellm, "completion", lambda **kw: (_ for _ in ()).throw(RuntimeError("down")))
    with pytest.raises(benchmark.JudgeFailed):
        benchmark.llm_judge("openai/gpt-4o", "reference", "answer", config)

    class Reply:
        """A judge reply without a rating."""
        choices = [type("C", (), {"message": type("M", (), {"content": "I cannot rate this."})()})()]
        usage = type("U", (), {"prompt_tokens": 1, "completion_tokens": 1})()
    monkeypatch.setattr(benchmark.litellm, "completion", lambda **kw: Reply())
    with pytest.raises(benchmark.JudgeFailed):
        benchmark.llm_judge("openai/gpt-4o", "reference", "answer", config)


def test_score_code_survives_code_that_loops_or_exits():
    """Model code that never ends, or exits the interpreter, scores 0 and never hangs or stops the benchmark."""
    test = "assert f() == 1"
    assert benchmark.score_code(test, "```python\ndef f():\n    while True:\n        pass\n```", timeout=2) == 0.0
    assert benchmark.score_code(test, "```python\nimport sys\nsys.exit(0)\n```") == 0.0
    assert benchmark.score_code(test, "```python\ndef f():\n    return 1\n```") == 1.0


def test_a_pass_needs_the_exact_loss_bound_too():
    """0 losses in 70 looks like proof to the bootstrap, but up to 4.2% losses could hide: not a pass; 0 in 120 is."""
    seventy = [0.0] * 68 + [1.0, 1.0]
    assert benchmark.paired_bootstrap_ci(seventy)[0] >= 0.0
    assert abs(benchmark.loss_rate_bound(seventy) - 0.0419) < 0.001
    assert benchmark.paired_verdict(seventy).verdict == "warn"
    assert benchmark.paired_verdict([0.0] * 118 + [1.0, 1.0]).verdict == "pass"


def test_a_small_cost_ceiling_still_allows_spending():
    """The safety margin scales with the ceiling: EUR 0.50 allows EUR 0.45, EUR 5 stops at EUR 4.50."""
    small = benchmark._BudgetGuard(0.50)
    assert not small.should_stop()
    small.record(0.44)
    assert not small.should_stop()
    small.record(0.02)
    assert small.should_stop()
    big = benchmark._BudgetGuard(5.0)
    big.record(4.49)
    assert not big.should_stop()
