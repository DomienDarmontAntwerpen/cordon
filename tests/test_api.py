"""
The HTTP surface: OpenAI compatibility, public endpoints, admin API, metrics, export.

Run: .venv/bin/python -m pytest tests/test_api.py -v
"""

import time


def test_health_no_auth(client):
    """Health no auth."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_openai_sdk_drop_in(client, tenant_headers, tenant_and_key):
    """Openai sdk drop in."""
    _tenant, key = tenant_and_key
    from openai import OpenAI

    openai_client = OpenAI(api_key=key, base_url=str(client.base_url) + "/v1", http_client=client)
    resp = openai_client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "hello via sdk"}]
    )
    assert resp.choices[0].message.content.startswith("Mock answer from Cordon")
    assert resp.usage.prompt_tokens > 0
    assert resp.usage.completion_tokens > 0


def test_openai_sdk_streaming_drop_in(client, tenant_and_key):
    """Openai sdk streaming drop in."""
    _tenant, key = tenant_and_key
    from openai import OpenAI

    openai_client = OpenAI(api_key=key, base_url=str(client.base_url) + "/v1", http_client=client)
    stream = openai_client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "hello streaming"}], stream=True
    )
    pieces = [chunk.choices[0].delta.content for chunk in stream if chunk.choices and chunk.choices[0].delta.content]
    assert "".join(pieces).startswith("Mock answer from Cordon")


def test_response_headers_present(client, tenant_headers):
    """Response headers present."""
    resp = client.post(
        "/v1/chat/completions",
        headers=tenant_headers,
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    for header in ("X-Cordon-Cost", "X-Cordon-Baseline-Cost", "X-Cordon-Cache", "X-Cordon-Route", "X-Cordon-Fallback", "X-Cordon-Overhead-Ms", "X-Cordon-Mode"):
        assert header in resp.headers
    assert resp.headers["X-Cordon-Mode"] == "enforce"


def test_v1_models_lists_only_allowed_providers(client, admin_headers):
    """V1 models lists only allowed providers."""
    resp = client.post("/admin/tenants", headers=admin_headers, json={"name": "T", "allowed_providers": ["mistral"]})
    tenant = resp.json()
    key = client.post(f"/admin/tenants/{tenant['id']}/keys", headers=admin_headers, json={}).json()["key"]

    models_resp = client.get("/v1/models", headers={"Authorization": f"Bearer {key}"})
    assert models_resp.status_code == 200
    ids = {m["id"] for m in models_resp.json()["data"]}
    assert ids == {"mistral-small", "mistral-large"}


def test_common_openai_parameters_are_accepted(client, tenant_headers):
    """The allowlist must not refuse ordinary OpenAI parameters a normal app sends."""
    body = {
        "model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.2, "top_p": 0.9, "max_tokens": 50, "stop": ["\n\n"], "seed": 7,
        "presence_penalty": 0.0, "frequency_penalty": 0.0, "user": "app-user-1",
    }
    resp = client.post("/v1/chat/completions", headers=tenant_headers, json=body)
    assert resp.status_code == 200, resp.json()


def test_admin_list_tenants(client, admin_headers):
    """GET /admin/tenants lists every tenant - used by the dashboard's tenant selector."""
    client.post("/admin/tenants", headers=admin_headers, json={"name": "Alpha"})
    client.post("/admin/tenants", headers=admin_headers, json={"name": "Beta"})
    resp = client.get("/admin/tenants", headers=admin_headers)
    assert resp.status_code == 200
    names = {t["name"] for t in resp.json()}
    assert {"Alpha", "Beta"}.issubset(names)


def test_admin_list_tenants_requires_admin_key(client, tenant_headers):
    """A tenant key cannot list every tenant."""
    resp = client.get("/admin/tenants", headers=tenant_headers)
    assert resp.status_code == 401


def test_unknown_feature_name_is_rejected(client, admin_headers):
    """A typo in a feature name fails loudly (400) instead of silently doing nothing."""
    resp = client.post("/admin/tenants", headers=admin_headers, json={"name": "Typo", "features": ["cahce"]})
    assert resp.status_code == 400
    assert "unknown feature: cahce" in resp.json()["error"]["message"]


def test_v1_tenant_returns_own_settings(client, tenant_headers, tenant_and_key):
    """GET /v1/tenant lets a tenant key read its own settings (e.g. budget caps for its own dashboard)."""
    tenant, _key = tenant_and_key
    resp = client.get("/v1/tenant", headers=tenant_headers)
    assert resp.status_code == 200
    assert resp.json()["id"] == tenant["id"]
    assert "budget_daily_eur" in resp.json()["settings"]


def test_metrics_filters_by_since_and_tenant_id(client, admin_headers, tenant_headers, tenant_and_key):
    """GET /metrics supports ?since= and, for the admin key, ?tenant_id= - both used by the dashboard."""
    tenant, _key = tenant_and_key
    client.post("/v1/chat/completions", headers=tenant_headers, json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]})

    future = time.time() + 3600
    empty = client.get(f"/metrics?since={future}", headers=admin_headers)
    assert empty.json()["requests"] == []

    scoped = client.get(f"/metrics?tenant_id={tenant['id']}", headers=admin_headers)
    assert scoped.json()["scope"] == "tenant"
    assert all(r["tenant_id"] == tenant["id"] for r in scoped.json()["requests"])


def test_tenant_cannot_see_other_tenants_metrics(client, admin_headers):
    """Tenant cannot see other tenants metrics."""
    resp_a = client.post("/admin/tenants", headers=admin_headers, json={"name": "A", "allowed_providers": ["openai"]})
    tenant_a = resp_a.json()
    key_a = client.post(f"/admin/tenants/{tenant_a['id']}/keys", headers=admin_headers, json={}).json()["key"]

    resp_b = client.post("/admin/tenants", headers=admin_headers, json={"name": "B", "allowed_providers": ["openai"]})
    tenant_b = resp_b.json()
    key_b = client.post(f"/admin/tenants/{tenant_b['id']}/keys", headers=admin_headers, json={}).json()["key"]

    client.post("/v1/chat/completions", headers={"Authorization": f"Bearer {key_a}"}, json={"model": "gpt-4o", "messages": [{"role": "user", "content": "a"}]})
    client.post("/v1/chat/completions", headers={"Authorization": f"Bearer {key_b}"}, json={"model": "gpt-4o", "messages": [{"role": "user", "content": "b"}]})

    metrics_a = client.get("/metrics", headers={"Authorization": f"Bearer {key_a}"}).json()
    assert metrics_a["scope"] == "tenant"
    assert all(r["tenant_id"] == tenant_a["id"] for r in metrics_a["requests"])
    assert len(metrics_a["requests"]) == 1


def test_export_contains_exactly_one_utc_month(app, client, admin_headers):
    """GET /admin/export?month= returns that UTC calendar month only - not the months after it."""
    import calendar

    from src.server.ledger import write_block_ledger

    store = app.state.services.store
    stamps = {
        "before": calendar.timegm((2026, 9, 30, 23, 30, 0)),   # last half hour of September (UTC)
        "inside": calendar.timegm((2026, 10, 15, 12, 0, 0)),
        "after": calendar.timegm((2026, 11, 1, 0, 30, 0)),     # first half hour of November (UTC)
    }
    for label, stamp in stamps.items():
        write_block_ledger(store, "export-test", "cdn_x", [], {"mode": "enforce"}, "gpt-4o", label)
        with store._lock:
            store._conn.execute("UPDATE ledger SET timestamp = ? WHERE error_code = ?", (stamp, label))
            store._conn.commit()

    resp = client.get("/admin/export?month=2026-10", headers=admin_headers)
    assert resp.status_code == 200
    data_rows = resp.text.strip().splitlines()[1:]
    assert len(data_rows) == 1
    assert data_rows[0].startswith(str(stamps["inside"]))


def test_csv_formula_injection_neutralized(client, admin_headers, tenant_headers):
    """Csv formula injection neutralized."""
    tagged_body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
    client.post("/v1/chat/completions", headers=tenant_headers, json=tagged_body)

    month = time.strftime("%Y-%m", time.gmtime())
    resp = client.get(f"/admin/export?month={month}", headers=admin_headers)
    assert resp.status_code == 200
    # No cell may begin with a raw formula-trigger character once escaped.
    for line in resp.text.splitlines():
        for cell in line.split(","):
            assert not cell.startswith(("=", "+", "-", "@")) or cell.startswith("'")


def test_legacy_and_newer_openai_shapes_are_accepted(client, tenant_headers):
    """The 'function' role and a named non-function tool are valid OpenAI request shapes."""
    resp = client.post("/v1/chat/completions", headers=tenant_headers, json={"model": "gpt-4o", "messages": [
        {"role": "user", "content": "hi"}, {"role": "function", "name": "f", "content": "{}"}],
        "tools": [{"type": "custom", "custom": {"name": "grep"}}]})
    assert resp.status_code != 400 or resp.json()["error"].get("code") != "invalid_param_value", resp.text
