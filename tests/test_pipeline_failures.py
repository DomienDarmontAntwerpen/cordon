"""
When something breaks: fail open for savings, fail closed for security, passthrough, crashes.

Run: .venv/bin/python -m pytest tests/test_pipeline_failures.py -v
"""


from helpers import team


def test_pipeline_unexpected_failure_passthrough_when_open(client, admin_headers, monkeypatch):
    """pipeline unexpected failure passthrough when open."""

    resp = client.post(
        "/admin/tenants", headers=admin_headers,
        json={"name": "T", "mode": "enforce", "allowed_providers": ["openai"], "security_fail_mode": "open"},
    )
    key = client.post(f"/admin/tenants/{resp.json()['id']}/keys", headers=admin_headers, json={}).json()["key"]
    headers = {"Authorization": f"Bearer {key}"}

    from src.server.pipeline import step4_optimize

    def broken(*_args, **_kwargs):
        """Simulate a genuinely unexpected bug in a step after the limits were checked."""
        raise RuntimeError("simulated unexpected bug")

    monkeypatch.setattr(step4_optimize.routing, "classify_request", broken)

    resp2 = client.post("/v1/chat/completions", headers=headers, json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]})
    assert resp2.status_code == 200
    assert resp2.headers["X-Cordon-Error"] == "internal"
    assert resp2.json()["choices"][0]["message"]["content"]


def test_fail_open_never_skips_the_limits(client, admin_headers, monkeypatch):
    """A crash while the size/rate/budget limits are being checked is a 503 even for a fail-open tenant: open skips guardrails, never limits."""
    from src.savings import optimize as optimize_module

    headers = team(client, admin_headers, security_fail_mode="open")

    def broken(*_args, **_kwargs):
        """Simulate a bug inside the limit checks."""
        raise RuntimeError("simulated unexpected bug")

    monkeypatch.setattr(optimize_module, "iter_image_data_urls", broken)
    resp = client.post("/v1/chat/completions", headers=headers, json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 503


def test_pipeline_unexpected_failure_closed_by_default(client, admin_headers, monkeypatch):
    """pipeline unexpected failure closed by default."""
    from src.savings import optimize as optimize_module

    resp = client.post(
        "/admin/tenants", headers=admin_headers,
        json={"name": "T", "mode": "enforce", "allowed_providers": ["openai"]},  # security_fail_mode defaults to "closed"
    )
    key = client.post(f"/admin/tenants/{resp.json()['id']}/keys", headers=admin_headers, json={}).json()["key"]
    headers = {"Authorization": f"Bearer {key}"}

    def broken(*_args, **_kwargs):
        """Simulate a genuinely unexpected bug deep in the optimization step."""
        raise RuntimeError("simulated unexpected bug")

    monkeypatch.setattr(optimize_module, "iter_image_data_urls", broken)

    resp2 = client.post("/v1/chat/completions", headers=headers, json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]})
    assert resp2.status_code == 503
    assert resp2.headers["X-Cordon-Error"] == "security"


def test_mock_fail_step_optimize_fails_open(client, admin_headers):
    """A crashing optimization step (X-Mock-Fail-Step: optimize) still returns 200 - fail open, cost-only."""
    resp = client.post("/admin/tenants", headers=admin_headers, json={"name": "T", "allowed_providers": ["openai"], "features": ["trim"]})
    key = client.post(f"/admin/tenants/{resp.json()['id']}/keys", headers=admin_headers, json={}).json()["key"]
    resp2 = client.post(
        "/v1/chat/completions", headers={"Authorization": f"Bearer {key}", "X-Mock-Fail-Step": "optimize"},
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp2.status_code == 200


def test_mock_fail_step_input_guardrails_fails_closed_by_default(client, admin_headers):
    """A crashing input-guardrail step (X-Mock-Fail-Step: input_guardrails) returns 503 under the default security_fail_mode."""
    resp = client.post("/admin/tenants", headers=admin_headers, json={"name": "T", "allowed_providers": ["openai"]})
    key = client.post(f"/admin/tenants/{resp.json()['id']}/keys", headers=admin_headers, json={}).json()["key"]
    resp2 = client.post(
        "/v1/chat/completions", headers={"Authorization": f"Bearer {key}", "X-Mock-Fail-Step": "input_guardrails"},
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp2.status_code == 503
    assert resp2.headers["X-Cordon-Error"] == "security"


def test_fail_open_passthrough_works_for_streaming_requests(client, admin_headers, monkeypatch):
    """With security_fail_mode=open, an unexpected crash still answers a streaming request as a valid stream."""
    from src.server.pipeline import step4_optimize

    headers = team(client, admin_headers, security_fail_mode="open")
    monkeypatch.setattr(step4_optimize.routing, "classify_request", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}], "stream": True}
    with client.stream("POST", "/v1/chat/completions", headers=headers, json=body) as resp:
        assert resp.status_code == 200
        assert resp.headers["X-Cordon-Error"] == "internal"
        raw = "".join(resp.iter_text())
    assert '"delta": {"content": "This is a mock answer from Cordon."}' in raw
    assert raw.endswith("data: [DONE]\n\n")


def test_crash_after_the_model_call_fails_cleanly_and_books_the_cost(app, client, admin_headers, monkeypatch):
    """A crash in steps 7-9 returns a clean 503 (not a bare 500), books what was spent and releases the rest of the budget."""
    from src.server.pipeline import step9_output

    headers = team(client, admin_headers, budget_daily_eur=5.0)
    monkeypatch.setattr(step9_output.guardrails, "restore_message", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    resp = client.post("/v1/chat/completions", headers=headers, json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 503 and resp.headers["X-Cordon-Error"] == "internal"
    row = app.state.services.store.query_ledger(limit=1)[0]
    assert row["status"] == "error" and row["error_code"] == "internal_error" and row["cost_actual_eur"] > 0
    daily, _monthly = app.state.services.budget_tracker.usage(row["tenant_id"])
    assert daily == row["cost_actual_eur"], "only the real spend stays booked; the reservation is released"


def test_crash_at_stream_start_is_a_clean_503_booked_as_internal(app, client, admin_headers, monkeypatch):
    """A crash inside Cordon before the first chunk: OpenAI-format 503, ledger says internal_error (not 'client disconnected')."""
    from src.server.pipeline import streaming

    headers = team(client, admin_headers)
    monkeypatch.setattr(streaming.guardrails, "StreamingOutputGuardrail", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    resp = client.post("/v1/chat/completions", headers=headers, json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}], "stream": True})
    assert resp.status_code == 503 and resp.json()["error"]["message"] == "an internal error occurred"
    assert app.state.services.store.query_ledger(limit=1)[0]["error_code"] == "internal_error"
    assert app.state.services.capacity.in_flight == 0
