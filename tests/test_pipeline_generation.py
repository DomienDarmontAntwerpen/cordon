"""
Step 7 and streaming: failover, timeouts, SSE format, concurrency and capacity.

Run: .venv/bin/python -m pytest tests/test_pipeline_generation.py -v
"""

import json
import time


def test_streaming_raw_sse_has_cordon_comment_and_done(client, tenant_headers):
    """Streaming raw sse has cordon comment and done."""
    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers=tenant_headers,
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    ) as resp:
        assert resp.status_code == 200
        text = "".join(resp.iter_text())
    assert ": cordon " in text
    assert "data: [DONE]" in text
    cordon_line = [line for line in text.splitlines() if line.startswith(": cordon ")][0]
    payload = json.loads(cordon_line[len(": cordon "):])
    assert "cost_eur" in payload
    assert payload["cost_flag"] == "measured"


def test_ladder_failover_to_next_candidate_on_provider_error(client, admin_headers, monkeypatch, proven_extraction_route):
    """ladder failover to next candidate on provider error."""
    import litellm as litellm_module

    resp = client.post(
        "/admin/tenants", headers=admin_headers,
        json={"name": "T", "mode": "enforce", "allowed_providers": ["openai", "mistral"], "features": ["routing"], "holdout_pct": 0.0, "strong": "gpt-4o"},
    )
    key = client.post(f"/admin/tenants/{resp.json()['id']}/keys", headers=admin_headers, json={}).json()["key"]
    headers = {"Authorization": f"Bearer {key}"}

    real_completion = litellm_module.completion
    failed_once = {"done": False}

    def flaky_completion(**kwargs):
        """Fail exactly once (for the first candidate tried), then behave normally for every later call."""
        if not failed_once["done"]:
            failed_once["done"] = True
            raise RuntimeError("simulated transient provider outage")
        return real_completion(**kwargs)

    monkeypatch.setattr(litellm_module, "completion", flaky_completion)

    resp2 = client.post(
        "/v1/chat/completions", headers=headers,
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Please extract the following fields: name, date."}]},
    )
    assert resp2.status_code == 200
    # mistral-small (the route's model) failed; the strong model, next on a route's ladder, served it.
    assert resp2.headers["X-Cordon-Route"] == "gpt-4o"
    assert resp2.headers["X-Cordon-Fallback"] == "true"


def test_every_provider_call_carries_the_configured_timeout(client, tenant_headers, monkeypatch):
    """Every litellm.completion call gets provider.timeout_seconds, so a hung provider cannot hold a request forever."""
    import litellm as litellm_module

    from src.paths import CONFIG_PATH

    import yaml

    expected = yaml.safe_load(open(CONFIG_PATH, encoding="utf-8"))["provider"]["timeout_seconds"]
    seen_timeouts = []
    real_completion = litellm_module.completion

    def recording_completion(**kwargs):
        """Record the timeout each provider call was given."""
        seen_timeouts.append(kwargs.get("timeout"))
        return real_completion(**kwargs)

    monkeypatch.setattr(litellm_module, "completion", recording_completion)
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hello there"}]}
    assert client.post("/v1/chat/completions", headers=tenant_headers, json=body).status_code == 200
    assert client.post("/v1/chat/completions", headers=tenant_headers, json={**body, "stream": True}).status_code == 200
    assert seen_timeouts and all(t == expected for t in seen_timeouts)


def test_streaming_provider_failure_sends_a_valid_sse_error(client, tenant_headers):
    """When every model fails mid-stream setup, the client gets a proper `data: {...error...}` SSE event, then [DONE]."""
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}], "stream": True}
    with client.stream("POST", "/v1/chat/completions", headers={**tenant_headers, "X-Mock-Fail-Step": "llm_call"}, json=body) as resp:
        raw = "".join(resp.iter_text())
    assert 'data: {"error": {"message": "the upstream provider call failed"' in raw
    assert raw.endswith("data: [DONE]\n\n")


def test_slow_provider_calls_run_concurrently(app, client, tenant_headers, monkeypatch):
    """Four requests whose provider call takes 0.5s each finish in about 0.5s together, not 2s one after another."""
    import asyncio

    import httpx
    import litellm as litellm_module

    real_completion = litellm_module.completion

    def slow_completion(**kwargs):
        """Simulate a provider that takes half a second to answer."""
        time.sleep(0.5)
        return real_completion(**kwargs)

    monkeypatch.setattr(litellm_module, "completion", slow_completion)

    async def send_four() -> tuple[list[int], float]:
        """Send four requests at once through the ASGI app; return their status codes and the wall time."""
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://cordon") as async_client:
            start = time.perf_counter()
            responses = await asyncio.gather(*[
                async_client.post(
                    "/v1/chat/completions", headers=tenant_headers,
                    json={"model": "gpt-4o", "messages": [{"role": "user", "content": f"question number {i}"}]},
                )
                for i in range(4)
            ])
            return [r.status_code for r in responses], time.perf_counter() - start

    codes, wall_seconds = asyncio.run(send_four())
    assert codes == [200, 200, 200, 200]
    assert wall_seconds < 1.4, f"requests were serialized: {wall_seconds:.2f}s for four 0.5s calls"


def test_requests_beyond_capacity_get_a_clean_503(app, tenant_headers, monkeypatch):
    """Above server.max_in_flight, Cordon refuses at once with 503 server_busy + Retry-After instead of queueing forever."""
    import asyncio

    import httpx
    import litellm as litellm_module

    from src.server.services import CapacityGate

    app.state.services.capacity = CapacityGate(max_in_flight=2)
    real_completion = litellm_module.completion

    def slow_completion(**kwargs):
        """A provider that takes half a second."""
        time.sleep(0.5)
        return real_completion(**kwargs)

    monkeypatch.setattr(litellm_module, "completion", slow_completion)

    async def send(n: int) -> list:
        """Send n requests at once; return the responses."""
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://cordon") as c:
            return await asyncio.gather(*[
                c.post("/v1/chat/completions", headers=tenant_headers,
                       json={"model": "gpt-4o", "messages": [{"role": "user", "content": f"question {i}"}]})
                for i in range(n)
            ])

    responses = asyncio.run(send(4))
    codes = sorted(r.status_code for r in responses)
    assert codes == [200, 200, 503, 503]
    busy = next(r for r in responses if r.status_code == 503)
    assert busy.json()["error"]["code"] == "server_busy" and busy.headers["Retry-After"] == "1"
    assert app.state.services.capacity.in_flight == 0


def test_streams_hold_their_capacity_slot_and_release_it(app, tenant_headers, monkeypatch):
    """While a (slow) stream is running, it holds its capacity slot: a second request gets 503; afterwards the slot is free."""
    import asyncio

    import httpx
    import litellm as litellm_module

    from src.server.services import CapacityGate

    app.state.services.capacity = CapacityGate(max_in_flight=1)
    real_completion = litellm_module.completion

    def slow_stream_completion(**kwargs):
        """A provider stream that takes 0.1 s per chunk."""
        result = real_completion(**kwargs)
        if not kwargs.get("stream"):
            return result

        def slow():
            """Yield the real chunks slowly."""
            for chunk in result:
                time.sleep(0.1)
                yield chunk
        return slow()

    monkeypatch.setattr(litellm_module, "completion", slow_stream_completion)
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "Tell me a fairly long story about a lighthouse keeper."}]}

    async def scenario() -> tuple[int, int]:
        """Start a stream; half a second later, while it still runs, send a second request."""
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://cordon", timeout=30) as c:
            async def second():
                """The second request, sent while the stream is still running."""
                await asyncio.sleep(0.5)
                return await c.post("/v1/chat/completions", headers=tenant_headers, json=body)

            first, other = await asyncio.gather(
                c.post("/v1/chat/completions", headers=tenant_headers, json={**body, "stream": True}), second()
            )
            return first.status_code, other.status_code

    first_status, second_status = asyncio.run(scenario())
    assert first_status == 200 and second_status == 503
    assert app.state.services.capacity.in_flight == 0


# --- Found by fuzzing: malformed requests and provider rejections ---
def test_malformed_parameters_and_parts_are_refused_at_the_door(client, admin_headers):
    """A wrong-shaped parameter or an unknown message part type is a 400 before any model is called."""
    from helpers import team

    headers = team(client, admin_headers)
    for body in (
        {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}], "tool_choice": 0.7},
        {"model": "gpt-4o", "messages": [{"role": "user", "content": [{"type": "audio", "text": "x"}]}]},
        {"model": "gpt-4o", "messages": [{"role": "hacker", "content": "hi"}]},
        {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}], "tools": [{"type": "function"}]},
        # Found by fuzzing: LiteLLM crashed on these and Cordon answered 502 "provider failed".
        {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}, {"role": "developer", "content": None,
                                          "tool_calls": [{"id": "1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]}]},
        {"model": "gpt-4o", "messages": [{"role": "user"}]},
        {"model": "gpt-4o", "messages": [{"role": "assistant", "content": None, "tool_calls": "x"}]},
    ):
        resp = client.post("/v1/chat/completions", headers=headers, json=body)
        assert resp.status_code == 400 and resp.json()["error"]["code"] == "invalid_param_value", body


def test_a_request_the_provider_rejects_is_not_retried_on_other_models(client, admin_headers, monkeypatch, proven_extraction_route):
    """A provider 400 means the request is wrong: one call, then a 400 to the caller - no failover down the ladder."""
    import litellm as litellm_module

    from helpers import team

    calls = []

    def rejecting(**kwargs):
        """Reject like a provider that finds the request invalid."""
        calls.append(kwargs["model"])
        raise litellm_module.BadRequestError("Invalid value for 'x'", model=kwargs["model"], llm_provider="openai")

    monkeypatch.setattr(litellm_module, "completion", rejecting)
    headers = team(client, admin_headers, features=["routing"], strong="gpt-4o")
    for stream in (False, True):
        calls.clear()
        resp = client.post("/v1/chat/completions", headers=headers, json={
            "model": "gpt-4o", "stream": stream,
            "messages": [{"role": "user", "content": "Please extract the following fields: name, date, amount."}]})
        assert len(calls) == 1, calls
        if stream:
            assert '"code": "provider_rejected"' in resp.text
        else:
            assert resp.status_code == 400 and resp.json()["error"]["code"] == "provider_rejected"


def test_confidence_is_asked_only_of_the_route_model_and_only_when_its_check_needs_it(client, admin_headers, monkeypatch):
    """logprobs go only to the route's model, only for a route checked by the classifier - never to the strong model."""
    import litellm as litellm_module
    from src.savings import routing

    from helpers import team

    sent = []
    real = litellm_module.completion

    def spy(**kwargs):
        """Record whether logprobs were asked."""
        sent.append((kwargs["model"], kwargs.get("logprobs")))
        return real(**kwargs)

    monkeypatch.setattr(litellm_module, "completion", spy)
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "Please extract the following fields: name, date, amount."}]}
    headers = team(client, admin_headers, features=["routing"], strong="gpt-4o")
    monkeypatch.setattr(routing, "ROUTING_TABLE", {"extraction": {"easy": routing.Route("gpt-4o-mini", 0.7, 0.3)}})
    client.post("/v1/chat/completions", headers=headers, json=body)
    assert sent == [("openai/gpt-4o-mini", True)]  # the check passed (mock confidence): no strong-model call

    sent.clear()
    from src.server.pipeline import mock
    monkeypatch.setattr(mock, "mock_confidence", lambda ctx: None)  # an answer without confidence cannot be checked
    client.post("/v1/chat/completions", headers=headers, json=body)
    assert sent == [("openai/gpt-4o-mini", True), ("openai/gpt-4o", None)]  # so the strong model is asked, without logprobs

    sent.clear()
    monkeypatch.setattr(routing, "ROUTING_TABLE", {"extraction": {"easy": routing.Route("gpt-4o-mini", 0.0, 0.0)}})
    client.post("/v1/chat/completions", headers=headers, json=body)
    assert sent == [("openai/gpt-4o-mini", None)]  # rules only: no confidence needed


def test_a_request_too_long_for_the_cheap_model_moves_up_the_ladder(client, admin_headers, monkeypatch, proven_extraction_route):
    """A model-specific 400 (context window) is not 'the request is wrong': the next model gets it."""
    import litellm as litellm_module

    from helpers import team

    calls = []
    real = litellm_module.completion

    def too_small(**kwargs):
        """The cheap model's context is too small; the strong one answers."""
        calls.append(kwargs["model"])
        if kwargs["model"].startswith("mistral/"):
            raise litellm_module.ContextWindowExceededError(message="too long", model=kwargs["model"], llm_provider="mistral")
        return real(**kwargs)

    monkeypatch.setattr(litellm_module, "completion", too_small)
    headers = team(client, admin_headers, features=["routing"], strong="gpt-4o")
    resp = client.post("/v1/chat/completions", headers=headers, json={
        "model": "gpt-4o", "messages": [{"role": "user", "content": "Please extract the following fields: name, date, amount."}]})
    assert resp.status_code == 200 and len(calls) == 2
    assert resp.headers["X-Cordon-Fallback"] == "true"


def test_odd_logprobs_never_break_a_paid_request():
    """answer_confidence returns None on malformed provider data instead of raising."""
    from types import SimpleNamespace as N

    from src.savings import quality

    broken = N(choices=[N(logprobs=N(content=[N(token=None, logprob=None), N(token="x", logprob=None)]))])
    assert quality.answer_confidence(broken) is None
    assert quality.answer_confidence(N(choices=[])) is None


def test_mock_latency_ignores_non_finite_values():
    """X-Mock-Latency-Ms: nan or inf never reaches time.sleep as such."""
    from src.server.pipeline import mock

    assert mock.latency_seconds("nan") == 0.0 and mock.latency_seconds("inf") == 0.0


def test_a_parameter_no_allowed_model_supports_is_a_400_not_a_provider_outage(client, admin_headers):
    """Found by fuzzing: mistral-small rejects presence_penalty (LiteLLM UnsupportedParamsError). That is the request, not a broken provider."""
    from helpers import team

    headers = team(client, admin_headers, features=[], strong="gpt-4o")
    for stream in (False, True):
        resp = client.post("/v1/chat/completions", headers=headers, json={
            "model": "mistral-small", "stream": stream, "presence_penalty": -1, "messages": [{"role": "user", "content": "hi"}]})
        if stream:
            assert '"code": "provider_rejected"' in resp.text
        else:
            assert resp.status_code == 400 and resp.json()["error"]["code"] == "provider_rejected"


def test_message_shapes_the_openai_sdk_writes_are_accepted(client, admin_headers):
    """Found in review: a history built with message.model_dump() carries null fields; a function result may have no content."""
    from helpers import team

    headers = team(client, admin_headers)
    history = [{"role": "user", "content": "hi"},
               {"content": "hello", "refusal": None, "role": "assistant", "annotations": None, "audio": None,
                "function_call": None, "tool_calls": None},
               {"role": "function", "name": "f", "content": None},
               {"role": "user", "content": "and now?"}]
    resp = client.post("/v1/chat/completions", headers=headers, json={"model": "gpt-4o", "messages": history})
    assert resp.status_code == 200, resp.text
