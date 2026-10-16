"""
evaluation/company_day.py: the day's summary compares each request straight vs through Cordon.

Run: .venv/bin/python -m pytest tests/test_company_day.py -v
"""

from evaluation import company_day


def test_the_summary_pairs_each_request_and_totals_the_day():
    """Cost saved and quality per kind and for the whole day, from paired records."""
    records = []
    for i in range(120):  # 0 losses in 120: enough for the exact loss bound to pass
        records += [{"id": f"s{i}", "kind": "support", "route": "direct", "cost_eur": 0.01, "score": 1.0},
                    {"id": f"s{i}", "kind": "support", "route": "cordon", "cost_eur": 0.004, "score": 1.0}]
    summary = company_day.summarize(records)
    assert summary["support"]["saved_pct"] == 60.0 and summary["support"]["requests"] == 120
    assert summary["whole day"]["eur_direct"] == 1.2 and summary["support"]["quality_verdict"] == "pass"
