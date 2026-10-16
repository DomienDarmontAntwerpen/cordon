"""
Steps 4, 5, 6 and 8, through the API: trim, cache, routing, the quality gate.

Run: .venv/bin/python -m pytest tests/test_pipeline_savings.py -v
"""

import json

import pytest

from helpers import team, list_message


def test_trim_savings_are_reflected_in_billed_cost(client, admin_headers):
    """
    Regression test: a request trim actually shrinks must bill less than
    its baseline. Previously, mock mode always set usage.prompt_tokens to
    the pre-trim token count, so X-Cordon-Cost equalled X-Cordon-Baseline-Cost
    even when trim/history/tool-output optimization measurably reduced the
    tokens actually sent - savings_json showed a saving, but the billed
    "actual" cost silently never reflected it. Found via demo.py's live
    trim scenario, which this mirrors.
    """
    resp = client.post(
        "/admin/tenants", headers=admin_headers,
        json={"name": "T", "mode": "enforce", "allowed_providers": ["openai"], "features": ["trim"], "holdout_pct": 0.0},
    )
    key = client.post(f"/admin/tenants/{resp.json()['id']}/keys", headers=admin_headers, json={}).json()["key"]
    headers = {"Authorization": f"Bearer {key}"}

    repeated_sentence = "Please remember that the invoice total must always be checked twice. "
    messages = [{"role": "system", "content": "You are a careful finance assistant."}]
    for i in range(40):
        messages.append({"role": "user", "content": f"Earlier turn {i}: " + repeated_sentence * 3})
        messages.append({"role": "assistant", "content": f"Acknowledged, turn {i}."})
    messages.append({"role": "tool", "tool_call_id": "t1", "content": "LINE " * 4000})
    messages.append({"role": "user", "content": repeated_sentence * 5 + " What is today's total?"})

    resp2 = client.post("/v1/chat/completions", headers=headers, json={"model": "gpt-4o", "messages": messages})
    assert resp2.status_code == 200
    actual = float(resp2.headers["X-Cordon-Cost"])
    baseline = float(resp2.headers["X-Cordon-Baseline-Cost"])
    assert actual < baseline


def test_exact_cache_hit_on_identical_request(client, admin_headers):
    """exact cache hit on identical request."""
    resp = client.post("/admin/tenants", headers=admin_headers, json={"name": "T", "mode": "enforce", "allowed_providers": ["openai"], "features": ["cache"], "holdout_pct": 0.0})
    tenant = resp.json()
    key = client.post(f"/admin/tenants/{tenant['id']}/keys", headers=admin_headers, json={}).json()["key"]
    headers = {"Authorization": f"Bearer {key}"}
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "What is 2+2?"}]}

    first = client.post("/v1/chat/completions", headers=headers, json=body)
    assert first.headers["X-Cordon-Cache"] == "none"

    second = client.post("/v1/chat/completions", headers=headers, json=body)
    assert second.headers["X-Cordon-Cache"] == "exact"
    assert float(second.headers["X-Cordon-Cost"]) == 0.0


def test_semantic_cache_hit_on_near_duplicate_question(client, admin_headers):
    """semantic cache hit on near duplicate question."""
    resp = client.post(
        "/admin/tenants", headers=admin_headers,
        json={"name": "T", "mode": "enforce", "allowed_providers": ["openai"], "features": ["cache", "semantic_cache"], "holdout_pct": 0.0},
    )
    tenant = resp.json()
    key = client.post(f"/admin/tenants/{tenant['id']}/keys", headers=admin_headers, json={}).json()["key"]
    headers = {"Authorization": f"Bearer {key}"}

    first = client.post("/v1/chat/completions", headers=headers, json={"model": "gpt-4o", "messages": [{"role": "user", "content": "What is the capital of France?"}]})
    assert first.headers["X-Cordon-Cache"] == "none"

    # A case-only variant: same embedding (cosine ~1.0, clears the strict
    # 0.97 default) but a different exact-cache key (case-sensitive hash),
    # so this can only hit via the semantic path.
    second = client.post("/v1/chat/completions", headers=headers, json={"model": "gpt-4o", "messages": [{"role": "user", "content": "what is the capital of France?"}]})
    assert second.headers["X-Cordon-Cache"] == "semantic"


def test_cache_isolated_between_tenants_via_api(client, admin_headers):
    """cache isolated between tenants via api."""
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "unique cache isolation question"}]}

    resp_a = client.post("/admin/tenants", headers=admin_headers, json={"name": "A", "mode": "enforce", "allowed_providers": ["openai"], "features": ["cache"], "holdout_pct": 0.0})
    key_a = client.post(f"/admin/tenants/{resp_a.json()['id']}/keys", headers=admin_headers, json={}).json()["key"]
    resp_b = client.post("/admin/tenants", headers=admin_headers, json={"name": "B", "mode": "enforce", "allowed_providers": ["openai"], "features": ["cache"], "holdout_pct": 0.0})
    key_b = client.post(f"/admin/tenants/{resp_b.json()['id']}/keys", headers=admin_headers, json={}).json()["key"]

    client.post("/v1/chat/completions", headers={"Authorization": f"Bearer {key_a}"}, json=body)
    second = client.post("/v1/chat/completions", headers={"Authorization": f"Bearer {key_b}"}, json=body)
    assert second.headers["X-Cordon-Cache"] == "none"


def test_semantic_cache_reads_list_format_text(client, admin_headers):
    """Two different questions in list-of-parts format must never share a semantic cache entry."""
    headers = team(client, admin_headers, features=["cache", "semantic_cache"])
    first = client.post("/v1/chat/completions", headers=headers, json={"model": "gpt-4o", "messages": list_message("What is the capital of France?")})
    second = client.post("/v1/chat/completions", headers=headers, json={"model": "gpt-4o", "messages": list_message("Write a poem about dogs.")})
    assert first.status_code == second.status_code == 200
    assert second.headers["X-Cordon-Cache"] == "none"
    assert "poem about dogs" in second.json()["choices"][0]["message"]["content"]  # the mock answer echoes its own question
    again = client.post("/v1/chat/completions", headers=headers, json={"model": "gpt-4o", "messages": list_message("What is the capital of France?")})
    assert again.headers["X-Cordon-Cache"] == "exact"


def test_exact_cache_key_includes_answer_shaping_parameters(client, admin_headers):
    """An answer generated with max_tokens=3 must not be served to a request allowing 2,000 tokens."""
    headers = team(client, admin_headers, features=["cache"])
    base = {"model": "gpt-4o", "messages": [{"role": "user", "content": "Explain photosynthesis."}]}
    assert client.post("/v1/chat/completions", headers=headers, json={**base, "max_tokens": 3}).headers["X-Cordon-Cache"] == "none"
    assert client.post("/v1/chat/completions", headers=headers, json={**base, "max_tokens": 2000}).headers["X-Cordon-Cache"] == "none"
    assert client.post("/v1/chat/completions", headers=headers, json={**base, "max_tokens": 2000}).headers["X-Cordon-Cache"] == "exact"


def test_semantic_cache_never_matches_questions_with_different_files(client, admin_headers):
    """Same text, different attached document: never a semantic hit (one contract's summary for another)."""
    headers = team(client, admin_headers, features=["semantic_cache"])

    def ask(file_data: str):
        """Ask the same question about a different file."""
        content = [{"type": "text", "text": "Summarize this contract in three bullet points."},
                   {"type": "file", "file": {"filename": "contract.pdf", "file_data": file_data}}]
        return client.post("/v1/chat/completions", headers=headers, json={"model": "gpt-4o", "messages": [{"role": "user", "content": content}]})

    assert ask("data:application/pdf;base64,QUFB").headers["X-Cordon-Cache"] == "none"
    assert ask("data:application/pdf;base64,QkJC").headers["X-Cordon-Cache"] == "none"


def test_routing_without_evidence_stays_on_the_strong_model(client, admin_headers, monkeypatch):
    """With no proven cheaper model for this kind of request (empty routing table), routing keeps the strong model."""
    from src.savings import routing

    monkeypatch.setattr(routing, "ROUTING_TABLE", {})
    headers = team(client, admin_headers, features=["routing"], strong="gpt-4o")
    resp = client.post("/v1/chat/completions", headers=headers, json={
        "model": "gpt-4o", "messages": [{"role": "user", "content": "Please extract the following fields: name, date, amount."}],
    })
    assert resp.status_code == 200
    assert resp.headers["X-Cordon-Route"] == "gpt-4o"


def test_routing_selects_cheapest_candidate_for_easy_request(client, admin_headers, proven_extraction_route):
    """routing selects cheapest candidate for easy request."""
    resp = client.post(
        "/admin/tenants", headers=admin_headers,
        json={"name": "T", "mode": "enforce", "allowed_providers": ["openai", "mistral"], "features": ["routing"], "holdout_pct": 0.0, "strong": "gpt-4o"},
    )
    key = client.post(f"/admin/tenants/{resp.json()['id']}/keys", headers=admin_headers, json={}).json()["key"]
    headers = {"Authorization": f"Bearer {key}"}
    resp2 = client.post(
        "/v1/chat/completions", headers=headers,
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Please extract the following fields: name, date, amount."}]},
    )
    assert resp2.status_code == 200
    assert resp2.headers["X-Cordon-Route"] == "mistral-small"
    assert resp2.headers["X-Cordon-Fallback"] == "false"


def test_routing_hard_reasoning_defaults_to_strong(client, admin_headers):
    """routing hard reasoning defaults to strong."""
    resp = client.post(
        "/admin/tenants", headers=admin_headers,
        json={"name": "T", "mode": "enforce", "allowed_providers": ["openai", "mistral"], "features": ["routing"], "holdout_pct": 0.0, "strong": "gpt-4o"},
    )
    key = client.post(f"/admin/tenants/{resp.json()['id']}/keys", headers=admin_headers, json={}).json()["key"]
    headers = {"Authorization": f"Bearer {key}"}
    resp2 = client.post(
        "/v1/chat/completions", headers=headers,
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Solve this step by step: first, find x. then, find y. finally, prove the result."}]},
    )
    assert resp2.status_code == 200
    # gpt-4o is both requested and served - the header always carries the actual served alias.
    assert resp2.headers["X-Cordon-Route"] == "gpt-4o"
    assert resp2.headers["X-Cordon-Fallback"] == "false"


def test_routing_does_not_apply_to_a_directly_requested_model(client, admin_headers):
    """routing does not apply to a directly requested model."""
    resp = client.post(
        "/admin/tenants", headers=admin_headers,
        json={"name": "T", "mode": "enforce", "allowed_providers": ["openai", "mistral"], "features": ["routing"], "holdout_pct": 0.0, "strong": "gpt-4o"},
    )
    key = client.post(f"/admin/tenants/{resp.json()['id']}/keys", headers=admin_headers, json={}).json()["key"]
    headers = {"Authorization": f"Bearer {key}"}
    resp2 = client.post(
        "/v1/chat/completions", headers=headers,
        json={"model": "mistral-small", "messages": [{"role": "user", "content": "Please extract fields"}]},
    )
    assert resp2.status_code == 200
    assert resp2.headers["X-Cordon-Route"] == "mistral-small"
    assert resp2.headers["X-Cordon-Fallback"] == "false"


def test_routing_never_fails_over_to_a_model_pricier_than_strong(client, admin_headers, monkeypatch):
    """With a mid-priced strong model, gpt-4o is never on the ladder: if the strong model fails, Cordon must not fail over to something pricier than the reservation covers."""
    import litellm as litellm_module

    from src.savings import routing

    monkeypatch.setattr(routing, "ROUTING_TABLE", {})  # no evidence: the ladder starts with strong, others behind it
    real_completion = litellm_module.completion
    tried = []

    def only_gpt4o_up(**kwargs):
        """Every model except gpt-4o is down."""
        tried.append(kwargs["model"])
        if kwargs["model"] != "openai/gpt-4o":
            raise RuntimeError("provider down")
        return real_completion(**kwargs)

    monkeypatch.setattr(litellm_module, "completion", only_gpt4o_up)
    headers = team(client, admin_headers, features=["routing"], strong="mistral-large")
    resp = client.post("/v1/chat/completions", headers=headers, json={"model": "mistral-large", "messages": [{"role": "user", "content": "hello"}]})
    assert "openai/gpt-4o" not in tried
    assert resp.status_code == 502


def test_quality_gate_escalates_bad_routed_answer_to_strong(client, admin_headers, proven_extraction_route):
    """quality gate escalates bad routed answer to strong."""
    resp = client.post(
        "/admin/tenants", headers=admin_headers,
        json={"name": "T", "mode": "enforce", "allowed_providers": ["openai", "mistral"], "features": ["routing"], "holdout_pct": 0.0, "strong": "gpt-4o"},
    )
    key = client.post(f"/admin/tenants/{resp.json()['id']}/keys", headers=admin_headers, json={}).json()["key"]
    headers = {"Authorization": f"Bearer {key}", "X-Mock-Bad-Response": "true"}
    resp2 = client.post(
        "/v1/chat/completions", headers=headers,
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Please extract the following fields: name, date, amount."}]},
    )
    assert resp2.status_code == 200
    assert resp2.headers["X-Cordon-Route"] == "gpt-4o"
    assert resp2.headers["X-Cordon-Fallback"] == "true"


def test_quality_gate_does_not_escalate_a_directly_requested_cheap_model(client, admin_headers, proven_extraction_route):
    """quality gate does not escalate a directly requested cheap model."""
    resp = client.post(
        "/admin/tenants", headers=admin_headers,
        json={"name": "T", "mode": "enforce", "allowed_providers": ["openai", "mistral"], "features": ["routing"], "holdout_pct": 0.0, "strong": "gpt-4o"},
    )
    key = client.post(f"/admin/tenants/{resp.json()['id']}/keys", headers=admin_headers, json={}).json()["key"]
    headers = {"Authorization": f"Bearer {key}", "X-Mock-Bad-Response": "true"}
    resp2 = client.post(
        "/v1/chat/completions", headers=headers,
        json={"model": "mistral-small", "messages": [{"role": "user", "content": "anything"}]},
    )
    assert resp2.status_code == 200
    # Never routed (the caller asked for this model), so the check must not override the caller's explicit choice.
    assert resp2.headers["X-Cordon-Fallback"] == "false"


def test_quality_gate_does_not_escalate_tool_call_answers(client, admin_headers, proven_extraction_route, monkeypatch):
    """An answer that is a tool call has no text; the quality gate must not read that as 'empty' and escalate."""
    import litellm as litellm_module

    real_completion = litellm_module.completion

    def tool_call_completion(**kwargs):
        """Answer with a tool call and no text, as a real model does."""
        response = real_completion(**{**kwargs, "mock_response": "x"})
        message = response.choices[0].message
        message.content = None
        message.tool_calls = [litellm_module.ChatCompletionMessageToolCall(
            id="call_1", type="function", function={"name": "lookup_invoice", "arguments": "{\"id\": 7}"})]
        return response

    monkeypatch.setattr(litellm_module, "completion", tool_call_completion)
    headers = team(client, admin_headers, features=["routing"], strong="gpt-4o")
    resp = client.post("/v1/chat/completions", headers=headers, json={
        "model": "gpt-4o", "messages": [{"role": "user", "content": "Please extract the following fields: name, date, amount."}],
    })
    assert resp.status_code == 200
    assert resp.headers["X-Cordon-Route"] == "mistral-small"
    assert resp.headers["X-Cordon-Fallback"] == "false"
    assert resp.json()["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "lookup_invoice"


def test_quality_header_present_when_a_route_was_checked(client, admin_headers, proven_extraction_route):
    """A routed answer carries its check's score; the check only runs on a route."""
    resp = client.post(
        "/admin/tenants", headers=admin_headers,
        json={"name": "T", "mode": "enforce", "allowed_providers": ["openai", "mistral"], "features": ["routing"], "holdout_pct": 0.0, "strong": "gpt-4o"},
    )
    key = client.post(f"/admin/tenants/{resp.json()['id']}/keys", headers=admin_headers, json={}).json()["key"]
    headers = {"Authorization": f"Bearer {key}"}
    resp2 = client.post(
        "/v1/chat/completions", headers=headers,
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Please extract the following fields: name, date."}]},
    )
    assert resp2.status_code == 200
    assert "X-Cordon-Quality" in resp2.headers
    assert 0.0 <= float(resp2.headers["X-Cordon-Quality"]) <= 1.0
    assert resp2.headers["X-Cordon-Check"] in {"rules_only", "classifier", "no_confidence"}


def test_an_escalated_request_claims_no_routing_saving(client, admin_headers, proven_extraction_route):
    """Both models were paid when the check sent the answer on: the request cost more than the strong model alone."""
    headers = {**team(client, admin_headers, features=["routing"], strong="gpt-4o"), "X-Mock-Bad-Response": "true"}
    resp = client.post("/v1/chat/completions", headers=headers, json={
        "model": "gpt-4o", "messages": [{"role": "user", "content": "Please extract the following fields: name, date, amount."}]})
    assert resp.headers["X-Cordon-Fallback"] == "true"
    assert resp.headers["X-Cordon-Check"] == "non_answer"  # the rule that sent it on
    assert "routing" not in _savings_of_last_request(client, headers)
    assert float(resp.headers["X-Cordon-Cost"]) > float(resp.headers["X-Cordon-Baseline-Cost"])


def test_a_stream_never_takes_a_route(client, admin_headers, proven_extraction_route, monkeypatch):
    """A stream cannot be checked or taken back, so it stays on the strong model even where a route is proven."""
    import litellm as litellm_module

    called = []
    real = litellm_module.completion
    monkeypatch.setattr(litellm_module, "completion", lambda **kwargs: called.append(kwargs["model"]) or real(**kwargs))
    headers = team(client, admin_headers, features=["routing"], strong="gpt-4o")
    resp = client.post("/v1/chat/completions", headers=headers, json={
        "model": "gpt-4o", "stream": True,
        "messages": [{"role": "user", "content": "Please extract the following fields: name, date, amount."}]})
    assert resp.status_code == 200 and called == ["openai/gpt-4o"]


def test_observe_counts_what_the_check_would_send_on(client, admin_headers, monkeypatch):
    """A route whose check sends 40% on to the strong model saves 40 points less than the price difference alone."""
    from src.savings import routing

    headers = team(client, admin_headers, mode="observe", features=["routing"], strong="gpt-4o")
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "Please extract the following fields: name, date, amount."}]}
    saved = {}
    for escalated in (0.0, 0.4):
        monkeypatch.setattr(routing, "ROUTING_TABLE", {"extraction": {"easy": routing.Route("mistral-small", 0.0, escalated)}})
        resp = client.post("/v1/chat/completions", headers=headers, json=body)
        saved[escalated] = _savings_of_last_request(client, headers)["routing"]["eur"] / float(resp.headers["X-Cordon-Baseline-Cost"])
    assert abs((saved[0.0] - saved[0.4]) - 0.4) < 0.01


# --- Observe mode: what Cordon would save, without changing anything ---
def _savings_of_last_request(client, headers) -> dict:
    """The savings entries of the tenant's most recent ledger row."""
    import json

    rows = client.get("/metrics", headers=headers).json()["requests"]
    return json.loads(max(rows, key=lambda r: r["timestamp"])["savings_json"])


def test_observe_counts_a_cache_hit_without_serving_it(client, admin_headers):
    """In observe, a repeat question is still answered by the model, and booked as a "would save" cache saving."""
    headers = team(client, admin_headers, mode="observe", features=["cache"])
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "What are your opening hours?"}]}

    client.post("/v1/chat/completions", headers=headers, json=body)
    second = client.post("/v1/chat/completions", headers=headers, json=body)

    assert second.headers["X-Cordon-Cache"] == "none"  # nothing changed for the caller
    assert float(second.headers["X-Cordon-Cost"]) > 0
    entry = _savings_of_last_request(client, headers)["cache"]
    assert entry["applied"] is False
    assert entry["eur"] == pytest.approx(float(second.headers["X-Cordon-Baseline-Cost"]), abs=1e-6)


def test_observe_cache_keeps_no_answer_text(client, admin_headers):
    """Observe never serves from the cache, so what it stores is an empty marker, never the answer."""
    headers = team(client, admin_headers, mode="observe", features=["cache"])
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "What are your opening hours?"}]}
    client.post("/v1/chat/completions", headers=headers, json=body)

    services = client.app.state.services
    tenant_id = client.get("/v1/tenant", headers=headers).json()["id"]
    entries = services.store.cache_entries_for_tenant(tenant_id)
    assert len(entries) == 1
    assert services.cache.get_exact(tenant_id, entries[0]["cache_key"]) == ""


def test_observe_counts_a_cheaper_route_without_taking_it(client, admin_headers, proven_extraction_route):
    """In observe, the strong model still answers; the cheaper route is booked as a "would save" routing saving."""
    headers = team(client, admin_headers, mode="observe", features=["routing"], strong="gpt-4o")
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "Please extract the following fields: name, date, amount."}]}

    resp = client.post("/v1/chat/completions", headers=headers, json=body)

    assert resp.headers["X-Cordon-Route"] == "gpt-4o"
    entry = _savings_of_last_request(client, headers)["routing"]
    assert entry["applied"] is False and entry["eur"] > 0


def test_enforce_savings_are_marked_applied(client, admin_headers):
    """A real cache hit in enforce mode is booked as an applied saving."""
    headers = team(client, admin_headers, features=["cache"])
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "What are your opening hours?"}]}
    client.post("/v1/chat/completions", headers=headers, json=body)
    client.post("/v1/chat/completions", headers=headers, json=body)

    assert _savings_of_last_request(client, headers)["cache"]["applied"] is True


def test_would_save_never_exceeds_the_baseline(client, admin_headers, proven_extraction_route):
    """A would-be cache hit replaces the whole request: it is the only "would save" entry, never stacked on routing."""
    headers = team(client, admin_headers, mode="observe", features=["cache", "routing", "trim"], strong="gpt-4o")
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "Please extract the following fields: name, date, amount."}]}
    client.post("/v1/chat/completions", headers=headers, json=body)
    second = client.post("/v1/chat/completions", headers=headers, json=body)

    savings = _savings_of_last_request(client, headers)
    assert set(savings) == {"cache"}
    assert sum(e["eur"] for e in savings.values()) <= float(second.headers["X-Cordon-Baseline-Cost"]) + 1e-6


def test_would_save_with_trim_and_routing_stays_within_the_baseline(client, admin_headers, proven_extraction_route):
    """Trim and routing together: routing saves a share of what trim leaves, so the sum never passes the baseline."""
    headers = team(client, admin_headers, mode="observe", features=["trim", "routing"], strong="gpt-4o")
    long_history = [{"role": "user", "content": "Here is the contract text. " * 400}, {"role": "assistant", "content": "Noted."}] * 4
    body = {"model": "gpt-4o", "messages": long_history + [{"role": "user", "content": "Please extract the following fields: name, date, amount."}]}
    resp = client.post("/v1/chat/completions", headers=headers, json=body)

    savings = _savings_of_last_request(client, headers)
    assert {"trim", "routing"} <= set(savings)
    assert sum(e["eur"] for e in savings.values()) <= float(resp.headers["X-Cordon-Baseline-Cost"]) + 1e-6


# --- A route and its check: found in review (each test fails without its fix) ---
CLASSIFY = {"model": "gpt-4o", "messages": [{"role": "user", "content": "Classify the sentiment of this review as positive or negative: ok"}]}


@pytest.fixture
def checked_route(monkeypatch):
    """A route proven with the classifier: gpt-4o-mini, check at 0.5, for easy classification."""
    from src.savings import routing

    monkeypatch.setattr(routing, "ROUTING_TABLE", {"classification": {"easy": routing.Route("gpt-4o-mini", 0.5, 0.1)}})


def test_the_reservation_covers_both_models_of_an_escalated_request(client, admin_headers, checked_route, monkeypatch):
    """A routed request may pay the route model and then the strong one: both are reserved, so a cap is never overshot."""
    tracker = client.app.state.services.budget_tracker
    reserved = []
    real_settle = tracker.settle
    monkeypatch.setattr(tracker, "settle", lambda tenant, eur, actual: (reserved.append((eur, actual)), real_settle(tenant, eur, actual)))
    import litellm as litellm_module

    from src.savings import costing

    headers = {**team(client, admin_headers, features=["routing"], strong="gpt-4o"), "X-Mock-Bad-Response": "true"}
    resp = client.post("/v1/chat/completions", headers=headers, json={**CLASSIFY, "max_tokens": 40})
    assert resp.headers["X-Cordon-Fallback"] == "true"
    config = client.app.state.services.config
    tokens = litellm_module.token_counter(model="openai/gpt-4o", messages=CLASSIFY["messages"])
    both = sum(costing.cost_eur(name, tokens, 40, config) for name in ("openai/gpt-4o", "openai/gpt-4o-mini"))
    [(eur, actual)] = reserved
    assert abs(eur - both) < 1e-9 and eur >= actual > 0


def test_a_route_the_budget_cannot_cover_stays_on_the_strong_model(client, admin_headers, checked_route, monkeypatch):
    """Room for the strong model's worst case but not for the route's on top: no route, the request still answers."""
    import litellm as litellm_module

    from src.savings import costing

    config = client.app.state.services.config
    tokens = litellm_module.token_counter(model="openai/gpt-4o", messages=CLASSIFY["messages"])
    strong_only = costing.cost_eur("openai/gpt-4o", tokens, 40, config)
    called = []
    real = litellm_module.completion
    monkeypatch.setattr(litellm_module, "completion", lambda **kwargs: called.append(kwargs["model"]) or real(**kwargs))
    headers = team(client, admin_headers, features=["routing"], strong="gpt-4o", budget_daily_eur=round(strong_only * 1.01, 9))
    resp = client.post("/v1/chat/completions", headers=headers, json={**CLASSIFY, "max_tokens": 40})
    assert resp.status_code == 200 and called == ["openai/gpt-4o"]


def test_a_doubted_answer_is_never_served_or_cached_when_the_strong_model_fails(client, admin_headers, checked_route, monkeypatch):
    """The check rejects the cheaper answer and the strong model is down: a 502, and nothing reaches the cache."""
    import litellm as litellm_module

    real = litellm_module.completion

    def strong_down(**kwargs):
        """gpt-4o is unreachable."""
        if kwargs["model"] == "openai/gpt-4o":
            raise litellm_module.APIConnectionError(message="down", llm_provider="openai", model=kwargs["model"])
        return real(**kwargs)

    monkeypatch.setattr(litellm_module, "completion", strong_down)
    headers = team(client, admin_headers, features=["routing", "cache"], strong="gpt-4o")
    resp = client.post("/v1/chat/completions", headers={**headers, "X-Mock-Bad-Response": "true"}, json=CLASSIFY)
    assert resp.status_code == 502
    monkeypatch.setattr(litellm_module, "completion", real)
    again = client.post("/v1/chat/completions", headers=headers, json=CLASSIFY)
    assert again.status_code == 200 and again.headers["X-Cordon-Cache"] == "none"


def test_a_request_with_tools_never_takes_a_route(client, admin_headers, checked_route, monkeypatch):
    """A tool call has no text to check, so a request that offers tools stays on the strong model."""
    import litellm as litellm_module

    called = []
    real = litellm_module.completion
    monkeypatch.setattr(litellm_module, "completion", lambda **kwargs: called.append(kwargs["model"]) or real(**kwargs))
    headers = team(client, admin_headers, features=["routing"], strong="gpt-4o")
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object", "properties": {}}}}]
    client.post("/v1/chat/completions", headers=headers, json={**CLASSIFY, "tools": tools})
    assert called == ["openai/gpt-4o"]


def test_a_routed_request_never_falls_back_to_an_unproven_model(client, admin_headers, checked_route, monkeypatch):
    """Route model and strong model both down: a 502, not an answer from a cheaper model nobody proved or checked."""
    import litellm as litellm_module

    called = []

    def all_openai_down(**kwargs):
        """Every OpenAI model is unreachable."""
        called.append(kwargs["model"])
        raise litellm_module.APIConnectionError(message="down", llm_provider="openai", model=kwargs["model"])

    monkeypatch.setattr(litellm_module, "completion", all_openai_down)
    headers = team(client, admin_headers, features=["routing"], strong="gpt-4o")
    resp = client.post("/v1/chat/completions", headers=headers, json=CLASSIFY)
    assert resp.status_code == 502 and called == ["openai/gpt-4o-mini", "openai/gpt-4o"]


def test_the_strong_model_refusing_an_escalation_is_a_400_not_an_outage(client, admin_headers, checked_route, monkeypatch):
    """Found in review: step 7 maps a provider's own rejection to 400; the escalation in step 8 must do the same."""
    import litellm as litellm_module

    real = litellm_module.completion

    def strong_rejects(**kwargs):
        """gpt-4o refuses the request as invalid."""
        if kwargs["model"] == "openai/gpt-4o":
            raise litellm_module.BadRequestError("bad", model=kwargs["model"], llm_provider="openai")
        return real(**kwargs)

    monkeypatch.setattr(litellm_module, "completion", strong_rejects)
    headers = {**team(client, admin_headers, features=["routing"], strong="gpt-4o"), "X-Mock-Bad-Response": "true"}
    resp = client.post("/v1/chat/completions", headers=headers, json=CLASSIFY)
    assert resp.status_code == 400 and resp.json()["error"]["code"] == "provider_rejected"


def test_the_response_names_the_kind_of_request(client, admin_headers):
    """X-Cordon-Kind names the routing bucket; a chat sent with earlier replies is a conversation."""
    headers = team(client, admin_headers, features=[])
    question = {"role": "user", "content": "Can I change my delivery address?"}
    single = client.post("/v1/chat/completions", headers=headers, json={"model": "gpt-4o", "messages": [question]})
    history = [{"role": "user", "content": "Where is my order?"}, {"role": "assistant", "content": "It ships today."}]
    conversation = client.post("/v1/chat/completions", headers=headers, json={"model": "gpt-4o", "messages": history + [question]})
    assert single.headers["X-Cordon-Kind"] == "chat/easy"
    assert conversation.headers["X-Cordon-Kind"] == "conversation/easy"


def test_an_answer_the_provider_cut_off_is_sent_on_and_reported(client, admin_headers, proven_extraction_route, monkeypatch):
    """finish_reason "length" fails the check even when the text looks complete, and reaches the caller unchanged."""
    import litellm as litellm_module

    real = litellm_module.completion

    def cut_off_when_cheap(**kwargs):
        """The cheaper model hits its token limit on a sentence that happens to end with a full stop."""
        response = real(**kwargs)
        if kwargs["model"] != "openai/gpt-4o":
            response.choices[0].finish_reason = "length"
        return response

    monkeypatch.setattr(litellm_module, "completion", cut_off_when_cheap)
    headers = team(client, admin_headers, features=["routing"], strong="gpt-4o")
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "Please extract the following fields: name, date, amount."}]}
    resp = client.post("/v1/chat/completions", headers=headers, json=body)
    assert resp.headers["X-Cordon-Check"] == "empty_or_truncated" and resp.headers["X-Cordon-Route"] == "gpt-4o"
    assert resp.json()["choices"][0]["finish_reason"] == "stop"  # the strong model's own finish_reason

    def always_cut_off(**kwargs):
        """Every model hits its token limit: an unrouted request passes that on to the caller."""
        response = real(**kwargs)
        response.choices[0].finish_reason = "length"
        return response

    monkeypatch.setattr(litellm_module, "completion", always_cut_off)
    capped = client.post("/v1/chat/completions", headers=team(client, admin_headers, features=[]), json=body)
    assert capped.json()["choices"][0]["finish_reason"] == "length"


def test_a_cut_off_answer_is_never_cached_and_a_stream_reports_the_providers_finish_reason(client, admin_headers, monkeypatch):
    """finish_reason "length" is passed on as it came - in a stream too - and such an answer is not replayed from the cache."""
    import litellm as litellm_module

    real = litellm_module.completion

    def cut_off(**kwargs):
        """The provider hits the token limit: the last stream chunk, or the one answer, says "length"."""
        response = real(**kwargs)
        if not kwargs.get("stream"):
            response.choices[0].finish_reason = "length"
            return response
        chunks = list(response)
        chunks[-1].choices[0].finish_reason = "length"
        return iter(chunks)

    monkeypatch.setattr(litellm_module, "completion", cut_off)
    headers = team(client, admin_headers, features=["cache"])
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "Tell me about the delivery terms."}]}
    first = client.post("/v1/chat/completions", headers=headers, json=body)
    second = client.post("/v1/chat/completions", headers=headers, json=body)
    assert first.json()["choices"][0]["finish_reason"] == "length" and second.headers["X-Cordon-Cache"] == "none"

    streamed = client.post("/v1/chat/completions", headers=headers, json={**body, "stream": True})
    events = [json.loads(line[6:]) for line in streamed.text.splitlines() if line.startswith("data: {")]
    assert [e["choices"][0]["finish_reason"] for e in events if e["choices"][0]["finish_reason"]] == ["length"]
