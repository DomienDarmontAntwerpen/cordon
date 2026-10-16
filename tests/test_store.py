"""
src/server/store.py: no prompt text on disk, erasure, retention.

Run: .venv/bin/python -m pytest tests/test_store.py -v
"""

import time

from src.server.store import Store


def test_raw_database_contains_no_prompt_text(client, tenant_headers, app):
    """Raw database contains no prompt text."""
    secret_prompt_marker = "zzz_unique_prompt_marker_should_never_be_stored_zzz"
    resp = client.post(
        "/v1/chat/completions",
        headers=tenant_headers,
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": secret_prompt_marker}]},
    )
    assert resp.status_code == 200

    db_path = app.state.services.store._db_path
    with open(db_path, "rb") as f:
        raw = f.read()
    assert secret_prompt_marker.encode() not in raw


def test_tenant_erasure_removes_all_rows(tmp_path):
    """Tenant erasure removes all rows."""
    store = Store(str(tmp_path / "erasure.sqlite"))
    store.create_tenant("t1", "Acme", {"mode": "enforce"})
    store.add_api_key("hash1", "cdn_prefix123", "t1", None)
    store.insert_ledger_row(
        {
            "timestamp": time.time(), "tenant_id": "t1", "key_prefix": "cdn_prefix123", "tags_json": "[]",
            "session_hash": None, "mode": "enforce", "holdout": 0, "requested_model": "gpt-4o",
            "served_model": "gpt-4o", "provider": "openai", "tokens_in": 1, "tokens_out": 1,
            "cost_actual_eur": 0.0, "cost_baseline_eur": 0.0, "savings_json": "{}", "cache_type": None,
            "route": None, "fallback": 0, "quality_score": None, "guardrail_actions_json": "{}",
            "overhead_ms": 1.0, "provider_ms": 1.0, "status": "ok", "error_code": None, "error_detail": None,
        }
    )
    assert store.delete_tenant("t1") is True
    assert store.get_tenant("t1") is None
    assert store.query_ledger("t1") == []
    assert store.find_api_key("hash1") is None


def test_retention_purge_deletes_old_ledger_rows(tmp_path):
    """Retention purge deletes old ledger rows."""
    store = Store(str(tmp_path / "purge.sqlite"))
    store.create_tenant("t1", "Acme", {"mode": "enforce"})
    old_row = {
        "timestamp": time.time() - 200 * 86400, "tenant_id": "t1", "key_prefix": "p", "tags_json": "[]",
        "session_hash": None, "mode": "enforce", "holdout": 0, "requested_model": "gpt-4o",
        "served_model": "gpt-4o", "provider": "openai", "tokens_in": 1, "tokens_out": 1,
        "cost_actual_eur": 0.0, "cost_baseline_eur": 0.0, "savings_json": "{}", "cache_type": None,
        "route": None, "fallback": 0, "quality_score": None, "guardrail_actions_json": "{}",
        "overhead_ms": 1.0, "provider_ms": 1.0, "status": "ok", "error_code": None, "error_detail": None,
    }
    store.insert_ledger_row(old_row)
    result = store.purge(retention_days=90)
    assert result["ledger_deleted"] == 1
    assert store.query_ledger("t1") == []
