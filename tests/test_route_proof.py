"""
evaluation/route_proof.py: a route is proven together with its quality check, never the cheaper model alone.

Run: .venv/bin/python -m pytest tests/test_route_proof.py -v
"""

import json


from helpers import CONFIG
from evaluation import benchmark, route_proof

QUESTION = [{"role": "user", "content": "Classify the sentiment of this review as positive or negative: ok"}]


def pair(cheap: float, strong: float, answer: str = "positive", graded: bool = False) -> route_proof.Pair:
    """One question both models answered; no confidence, so only the rules can check it."""
    return route_proof.Pair(question="q", answer=answer, confidence=None, response_format=None,
                            cheap_score=cheap, strong_score=strong, graded=graded)


def test_an_answer_the_check_sends_on_counts_as_the_strong_models():
    """The cheaper model's "N/A" is caught by the rules: the strong model's answer stands in, so no loss is counted."""
    pairs = [pair(0.0, 1.0, "N/A")] + [pair(1.0, 1.0)] * 9
    e = route_proof.cascade_evidence(pairs, "gpt-4o-mini", 0.0, 0.06, CONFIG)
    assert (e.escalated, e.cascade_mean, e.verdict.mean_diff) == (0.1, 1.0, 0.0)
    assert abs(e.cost_share - 0.16) < 1e-9


def test_a_check_that_cannot_run_sends_everything_on_so_it_never_saves():
    """Above 0 the check needs the model's confidence; without it every answer goes on, which costs more than strong alone."""
    e = route_proof.cascade_evidence([pair(1.0, 1.0)] * 200, "gpt-4o-mini", 0.5, 0.06, CONFIG)
    assert e.escalated == 1.0 and e.verdict.verdict == "pass" and not e.proven


def test_trying_several_thresholds_raises_the_bar():
    """0 losses in 120 passes once; the route proof tries four thresholds, so it needs more."""
    assert benchmark.paired_verdict([0.0] * 120).verdict == "pass"
    assert not route_proof.cascade_evidence([pair(1.0, 1.0)] * 120, "gpt-4o-mini", 0.0, 0.06, CONFIG).proven
    assert route_proof.cascade_evidence([pair(1.0, 1.0)] * 200, "gpt-4o-mini", 0.0, 0.06, CONFIG).proven


def test_the_cheapest_proven_setting_wins():
    """Among proven settings the lowest cost; without one, the setting closest to proof is reported."""
    proven = route_proof.cascade_evidence([pair(1.0, 1.0)] * 200, "m", 0.0, 0.06, CONFIG)
    dearer = route_proof.cascade_evidence([pair(0.0, 1.0, "N/A")] * 20 + [pair(1.0, 1.0)] * 180, "m", 0.0, 0.06, CONFIG)
    assert route_proof.best_route([dearer, proven]) is proven


def test_questions_the_check_was_trained_on_are_never_evidence():
    """A gate scored on its own training questions would look better than it is: those questions are left out."""
    items = [{"id": f"c{i}", "category": "classification", "messages": QUESTION} for i in range(3)]
    answers = [{"item_id": f"c{i}", "condition": cond, "model": model, "question": "q", "answer": "positive", "score": 1.0}
               for i in range(3) for cond, model in (("A", "gpt-4o"), ("C", "gpt-4o-mini"))]
    pairs = route_proof.pairs_by_bucket(answers, items, CONFIG, excluded=frozenset({"c1"}))
    assert len(pairs[("classification", "easy")]["gpt-4o-mini"]) == 2


def test_the_table_lists_only_proven_cheaper_routes_with_their_evidence(tmp_path):
    """A proven route is written with its threshold and escalation share; an unproven bucket is left out."""
    import yaml

    proven = route_proof.cascade_evidence([pair(1.0, 1.0)] * 200, "gpt-4o-mini", 0.0, 0.06, CONFIG)
    unproven = route_proof.cascade_evidence([pair(1.0, 1.0)] * 20, "gpt-4o-mini", 0.0, 0.06, CONFIG)
    path = tmp_path / "routing_table.yaml"
    route_proof.write_routing_table({("classification", "easy"): [proven], ("chat", "easy"): [unproven]}, path)
    table = yaml.safe_load(path.read_text())["categories"]
    assert table == {"classification": {"easy": {"model": "gpt-4o-mini", "gate_threshold": 0.0, "escalated": 0.0}}}
    assert "on 200 questions" in path.read_text()


def test_rebuild_reads_several_question_files(tmp_path, monkeypatch):
    """A routing table can be rebuilt from answers to questions spread over several files (a set plus extra rounds)."""
    written = {}
    monkeypatch.setattr(route_proof, "write_routing_table", lambda proofs, **kw: written.update(proofs=proofs, **kw))
    answers = []
    for name, ids in (("set.jsonl", ["c0", "c1"]), ("extra.jsonl", ["c2"])):
        (tmp_path / name).write_text("\n".join(json.dumps({"id": i, "category": "classification", "messages": QUESTION})
                                               for i in ids) + "\n")
        for i in ids:
            answers += [{"item_id": i, "condition": cond, "model": model, "question": "q", "answer": "positive", "score": 1.0}
                        for cond, model in (("A", "gpt-4o"), ("C", "gpt-4o-mini"))]
    answers_path = tmp_path / "answers.jsonl"
    answers_path.write_text("\n".join(json.dumps(a) for a in answers) + "\n")
    route_proof.rebuild_routing_table([answers_path], [tmp_path / "set.jsonl", tmp_path / "extra.jsonl"])
    assert written["proofs"][("classification", "easy")][0].verdict.n == 3
    assert written["source"] == "answers.jsonl (set.jsonl, extra.jsonl)"


def test_a_cheaper_answer_the_check_can_read_is_preferred():
    """Two runs of the cheaper model on one question: the one with confidence is the one the check (and the proof) uses."""
    items = [{"id": "c0", "category": "classification", "messages": QUESTION}]
    base = {"item_id": "c0", "question": "q", "score": 1.0}
    answers = [{**base, "condition": "A", "model": "gpt-4o", "answer": "positive"},
               {**base, "condition": "C", "model": "gpt-4o-mini", "answer": "old run"},
               {**base, "condition": "C", "model": "gpt-4o-mini", "answer": "new run", "confidence": {"body_prob": 0.9}}]
    [only] = route_proof.pairs_by_bucket(answers, items, CONFIG, frozenset())[("classification", "easy")]["gpt-4o-mini"]
    assert only.answer == "new run"


def test_every_model_tried_in_a_bucket_raises_the_bar_too():
    """Three models x four thresholds on the same questions: each setting must clear 1 - 0.05/12."""
    assert abs(route_proof.confidence_for(12) - (1 - 0.05 / 12)) < 1e-12
    e = route_proof.cascade_evidence([pair(1.0, 1.0)] * 200, "m", 0.0, 0.06, CONFIG, route_proof.confidence_for(12))
    assert e.confidence == route_proof.confidence_for(12) and e.verdict.loss_bound > route_proof.cascade_evidence(
        [pair(1.0, 1.0)] * 200, "m", 0.0, 0.06, CONFIG).verdict.loss_bound


def test_graded_answers_count_on_average_and_a_failed_one_always_counts_as_a_loss():
    """Summaries differ a little everywhere: the average decides. An empty or failed summary is a loss."""
    graded = [pair(0.45, 0.46, graded=True)] * 100 + [pair(0.50, 0.46, graded=True)] * 100
    assert route_proof.cascade_evidence(graded, "m", 0.0, 0.16, CONFIG).proven
    failures = [pair(0.0, 0.6, answer="A summary.", graded=True)] * 12
    assert not route_proof.cascade_evidence(graded + failures, "m", 0.0, 0.16, CONFIG).proven


def test_in_a_mixed_bucket_wrong_answers_are_bounded_and_graded_ones_move_the_average():
    """Graded answers move the average; wrong answers on right/wrong questions are bounded."""
    graded = [pair(0.45, 0.46, graded=True)] * 150
    wrong = [pair(0.0, 1.0)] * 10 + [pair(1.0, 1.0)] * 40
    assert not route_proof.cascade_evidence(graded + wrong, "m", 0.0, 0.16, CONFIG).proven
    fine = [pair(1.0, 1.0)] * 200
    assert route_proof.cascade_evidence(graded + fine, "m", 0.0, 0.16, CONFIG).proven
