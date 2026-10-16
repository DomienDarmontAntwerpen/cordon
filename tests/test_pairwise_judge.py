"""
evaluation/pairwise_judge.py: the blind both-orders comparison of a route's replies.

Run: .venv/bin/python -m pytest tests/test_pairwise_judge.py -v
"""
from evaluation import pairwise_judge


def test_a_win_needs_both_orders_and_order_bias_becomes_a_tie():
    """Preferred as A and as B is a win; a judge that always picks A gives a tie, not a win."""
    assert pairwise_judge.outcome("A", "B") == 1.0
    assert pairwise_judge.outcome("B", "A") == 0.0
    assert pairwise_judge.outcome("A", "A") == 0.5 and pairwise_judge.outcome("TIE", "B") == 0.5


def test_only_conversations_both_models_answered_are_judged_and_the_rule_decides():
    """Questions outside the route's kind or without both answers are left out; a clear loss fails."""
    history = [{"role": "user", "content": "I ordered a jacket."}, {"role": "assistant", "content": "It ships today."}]
    questions = [{"id": f"c{i}", "messages": history + [{"role": "user", "content": "Can I change the address?"}]} for i in range(40)]
    questions.append({"id": "single", "messages": [{"role": "user", "content": "Where is my order?"}]})
    cheap = {q["id"]: "cheap reply" for q in questions}
    strong = {q["id"]: "strong reply" for q in questions[:-2]}  # c39 has no strong answer

    def prefers(reply):
        """A judge that always picks the given reply, wherever it stands."""
        return lambda conversation, a, b, config, on_cost: on_cost(0.001) or ("A" if a == reply else "B")

    lost = pairwise_judge.run(questions, cheap, strong, {}, max_eur=5, judge=prefers("strong reply"))
    assert lost["questions"] == 39 and lost["strong_wins"] == 39 and lost["verdict"] == "fail"
    won = pairwise_judge.run(questions, cheap, strong, {}, max_eur=5, judge=prefers("cheap reply"))
    assert won["cheap_share"] == 1.0 and won["verdict"] == "pass" and abs(won["spent_eur"] - 0.078) < 1e-9
