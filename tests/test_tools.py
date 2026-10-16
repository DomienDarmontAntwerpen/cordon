"""
tools/admin.py and evaluation/semantic_threshold.py.

Run: .venv/bin/python -m pytest tests/test_tools.py -v
"""

def test_admin_tool_manages_teams_end_to_end(tmp_path, capsys):
    """tools/admin.py creates a team, switches mode and features, rejects typos, and issues a working key."""
    from src.protection import security
    from src.server.store import Store
    from tools import admin

    db = str(tmp_path / "admin.sqlite")
    Store(db).close()
    run = lambda *args: admin.main(["--db", db, *args])  # noqa: E731

    assert run("new-team", "Support team") == 0
    assert run("mode", "support TEAM", "enforce") == 0          # names are not case-sensitive
    assert run("on", "Support team", "trim", "cache") == 0
    assert run("off", "Support team", "trim") == 0
    assert run("set", "Support team", "budget_daily_eur=50", "pii_action=block") == 0
    assert run("on", "Support team", "cahce") == 1              # typo -> clear error, nothing saved
    assert run("set", "Support team", "mode=sometimes") == 1     # invalid value -> clear error
    assert run("mode", "No such team", "enforce") == 1

    store = Store(db)
    tenant = store.list_tenants()[0]
    assert tenant["settings"]["mode"] == "enforce"
    assert tenant["settings"]["features"] == ["cache"]
    assert tenant["settings"]["budget_daily_eur"] == 50
    assert tenant["settings"]["pii_action"] == "block"

    capsys.readouterr()
    assert run("key", "Support team") == 0
    key = capsys.readouterr().out.strip().splitlines()[-1]
    record = store.find_api_key(security.hash_api_key(key))
    assert record is not None and record.tenant_id == tenant["id"]


def test_semantic_threshold_sweep_and_choice():
    """The threshold evaluation counts hits and wrong hits correctly and picks the lowest safe threshold."""
    import pandas as pd

    from evaluation import semantic_threshold as st

    scored = pd.DataFrame({
        "cosine":       [0.99, 0.98, 0.97, 0.96, 0.95, 0.99],
        "tokens_equal": [True, True, True, True, True, False],
        "label":        [1,    1,    0,    1,    0,    1],
    })
    table = st.sweep(scored, [0.95, 0.97, 0.98])
    row = table.set_index("threshold")
    assert row.loc[0.95, "hits"] == 5 and row.loc[0.95, "wrong_hits"] == 2   # the tokens_equal=False pair never hits
    assert row.loc[0.98, "hits"] == 2 and row.loc[0.98, "wrong_hits"] == 0
    assert row.loc[0.98, "hit_rate"] == 2 / 4                                 # 2 of the 4 true duplicates caught
    assert st.choose_threshold(table, max_wrong_hit_rate=0.0, min_hits=1) == 0.98
    assert st.choose_threshold(table, max_wrong_hit_rate=0.0, min_hits=3) is None
    low, high = st.bootstrap_wrong_hit_ci(scored, 0.95, resamples=200)
    assert 0.0 <= low <= high <= 1.0


def test_admin_tool_never_stores_an_unchecked_value(tmp_path):
    """`set` refuses text or yes/no where a number is expected, instead of storing it and breaking the team."""
    from src.server.store import Store
    from tools import admin

    db = str(tmp_path / "admin.sqlite")
    Store(db).close()
    run = lambda *args: admin.main(["--db", db, *args])  # noqa: E731
    assert run("new-team", "T") == 0
    for bad in ("budget_daily_eur='0'", "max_input_tokens='5'", "rpm=true", "budget_daily_eur=1e3"):
        assert run("set", "T", bad) != 0, bad
    assert run("set", "T", "budget_daily_eur=50") == 0
    tenant = Store(db).list_tenants()[0]
    assert tenant["settings"]["budget_daily_eur"] == 50


# --- evaluation/gate_eval.py: evaluating the quality gate on real answers ---
def test_gate_metrics_count_caught_bad_and_kept_good():
    """At a threshold, bad_caught and good_kept are the shares on each side of it."""
    import numpy as np

    from evaluation import gate_eval

    labels = np.array([1, 1, 1, 0, 0])
    good_probability = np.array([0.9, 0.8, 0.4, 0.3, 0.7])
    m = gate_eval.gate_metrics(labels, good_probability, threshold=0.5)
    assert m["n_bad"] == 2
    assert m["bad_caught"] == 0.5            # one of the two bad answers is below 0.5
    assert m["good_kept"] == round(2 / 3, 3)  # the 0.4 good answer would be escalated needlessly
    assert 0 <= m["auc"] <= 1


def test_load_answers_skips_failed_calls(tmp_path):
    """Failed calls carry no label and are not training data."""
    import json

    from evaluation import gate_eval

    rows = [{"item_id": "a", "question": "q", "answer": "x", "label": 1},
            {"item_id": "b", "question": "q", "answer": "", "label": None}]
    (tmp_path / "answers_1.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    assert [r["item_id"] for r in gate_eval.load_answers(tmp_path)] == ["a"]


def test_candidates_are_compared_on_held_out_questions_and_the_simplest_tie_wins():
    """Grouped CV runs for all three candidates; within the tolerance the simplest one is picked."""
    import numpy as np

    from evaluation import gate_eval

    rng = np.random.default_rng(0)
    rows = [{"item_id": f"q{i // 4}", "label": int(i % 3 != 0)} for i in range(120)]
    features = rng.normal(size=(120, 6)) + np.array([[r["label"] * 2.0] + [0] * 5 for r in rows])
    results = gate_eval.compare_candidates(rows, features, threshold=0.5)
    assert set(results) == set(gate_eval.SIMPLEST_FIRST)
    assert all(r["auc"] > 0.8 for r in results.values())  # the signal is easy here
    tie = {"LogisticRegression": {"auc": 0.90}, "RandomForest": {"auc": 0.905}, "GradientBoosting": {"auc": 0.95}}
    assert gate_eval.pick_simplest(tie) == "GradientBoosting"
    tie["GradientBoosting"]["auc"] = 0.909
    assert gate_eval.pick_simplest(tie) == "LogisticRegression"


# --- evaluation/train.py and the training data ---
def test_benchmark_answers_are_featurized_as_an_app_would_get_them():
    """The benchmark's FINAL marker is removed (a test-setup quirk), its value kept unless already at the end."""
    from evaluation.gate_eval import as_served

    assert as_served("The answer is 14.\n\nFINAL: 14") == "The answer is 14."
    assert as_served("the cost is:\nFINAL: 145") == "the cost is:\n145"
    assert as_served("**FINAL:** 4") == "4"
    assert as_served("no marker here") == "no marker here"


def test_threshold_keeps_the_target_share_of_good_answers():
    """choose_threshold: at most 10% of good answers fall below it (are escalated needlessly)."""
    import numpy as np

    from evaluation import train

    rng = np.random.default_rng(0)
    labels = np.array([1] * 900 + [0] * 100)
    scores = np.concatenate([rng.uniform(0.3, 1.0, 900), rng.uniform(0.0, 0.6, 100)])
    threshold = train.choose_threshold(labels, scores)
    assert abs((scores[labels == 1] < threshold).mean() - 0.10) < 0.01


def test_training_data_leaves_out_the_strong_model_and_route_evidence(tmp_path):
    """
    The classifier only judges cheaper models' answers that carry confidence, so it only trains on
    those - and never on a question the strong model also answered: route_proof.py checks it there.
    """
    import json

    from evaluation import train

    confidence = {"mean_prob": 0.9, "body_prob": 0.9}
    rows = [{"item_id": "a", "model": "gpt-4o", "label": 1, "question": "q", "answer": "x", "confidence": confidence},
            {"item_id": "a", "model": "gpt-4o-mini", "label": 0, "question": "q", "answer": "y", "confidence": confidence},
            {"item_id": "b", "model": "gpt-4o-mini", "label": 1, "question": "q", "answer": "z", "confidence": None},
            {"item_id": "c", "model": "gpt-4o-mini", "label": 1, "question": "q", "answer": "w", "confidence": confidence}]
    (tmp_path / "answers_1.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    config = {"models": {"default_strong": "gpt-4o"}}
    assert [r["item_id"] for r in train.training_rows(config, tmp_path)] == ["c"]


def test_each_kind_of_question_gets_its_own_threshold():
    """A kind the model scores high by nature gets a higher threshold; kinds with few good answers share the default."""
    import numpy as np

    from evaluation import train

    rng = np.random.default_rng(1)
    kinds = np.array(["classification"] * 200 + ["chat"] * 200 + ["extraction"] * 10)
    labels = np.array(([1] * 180 + [0] * 20) * 2 + [1] * 10)
    scores = np.concatenate([rng.uniform(0.6, 1.0, 180), rng.uniform(0.4, 0.8, 20),   # classification: high by nature
                             rng.uniform(0.1, 0.9, 180), rng.uniform(0.0, 0.3, 20), rng.uniform(0.5, 1, 10)])
    own, default = train.choose_thresholds(labels, scores, kinds)
    assert set(own) == {"classification", "chat"} and own["classification"] > own["chat"]
    for kind, threshold in own.items():
        good = scores[(kinds == kind) & (labels == 1)]
        assert abs((good < threshold).mean() - 0.10) < 0.02

