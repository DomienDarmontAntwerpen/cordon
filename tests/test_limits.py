"""
src/protection/limits.py: rate limits, budget reservation, budget downgrade, default max_tokens, the loop detector.

Run: .venv/bin/python -m pytest tests/test_limits.py -v
"""

import pytest


def test_rate_limiter_enforces_rpm():
    """rate limiter enforces rpm."""
    from src.protection.limits import RateLimiter

    limiter = RateLimiter()
    allowed = sum(1 for _ in range(10) if limiter.check_and_consume("cdn_abc", rpm=5, tpm=1_000_000, tokens_needed=1)[0])
    assert allowed == 5


def test_rate_limiter_enforces_tpm():
    """rate limiter enforces tpm."""
    from src.protection.limits import RateLimiter

    limiter = RateLimiter()
    allowed, retry_after = limiter.check_and_consume("cdn_xyz", rpm=1000, tpm=100, tokens_needed=150)
    assert not allowed
    assert retry_after > 0


def test_rate_limiter_keys_are_independent():
    """rate limiter keys are independent."""
    from src.protection.limits import RateLimiter

    limiter = RateLimiter()
    for _ in range(5):
        assert limiter.check_and_consume("cdn_a", rpm=5, tpm=1_000_000, tokens_needed=1)[0]
    # A different key's bucket is untouched by key "cdn_a" being exhausted.
    assert limiter.check_and_consume("cdn_b", rpm=5, tpm=1_000_000, tokens_needed=1)[0]


def test_budget_tracker_daily_cap_blocks_even_with_large_monthly_budget():
    """budget tracker daily cap blocks even with large monthly budget."""
    from src.protection.limits import BudgetTracker

    tracker = BudgetTracker()
    result = tracker.reserve("t1", 5.0, budget_daily_eur=4.0, budget_monthly_eur=1000.0)
    assert not result.allowed
    assert result.reason == "daily_cap"


def test_budget_tracker_monthly_budget_blocks():
    """budget tracker monthly budget blocks."""
    from src.protection.limits import BudgetTracker

    tracker = BudgetTracker()
    tracker.reserve("t1", 5.0, budget_daily_eur=100.0, budget_monthly_eur=10.0)
    result = tracker.reserve("t1", 6.0, budget_daily_eur=100.0, budget_monthly_eur=10.0)
    assert not result.allowed
    assert result.reason == "monthly_budget"


def test_budget_tracker_concurrent_reservations_cannot_overshoot():
    """budget tracker concurrent reservations cannot overshoot."""
    from src.protection.limits import BudgetTracker

    tracker = BudgetTracker()
    first = tracker.reserve("t1", 6.0, budget_daily_eur=10.0, budget_monthly_eur=1000.0)
    second = tracker.reserve("t1", 6.0, budget_daily_eur=10.0, budget_monthly_eur=1000.0)
    assert first.allowed
    assert not second.allowed


def test_budget_tracker_settle_releases_reservation_and_records_actual():
    """budget tracker settle releases reservation and records actual."""
    from src.protection.limits import BudgetTracker

    tracker = BudgetTracker()
    tracker.reserve("t1", 5.0, budget_daily_eur=20.0, budget_monthly_eur=20.0)
    tracker.settle("t1", reserved_eur=5.0, actual_eur=2.0)
    daily, monthly = tracker.usage("t1")
    assert daily == pytest.approx(2.0)
    assert monthly == pytest.approx(2.0)


def test_apply_default_max_tokens_fills_when_missing():
    """apply default max tokens fills when missing."""
    from src.protection.limits import apply_default_max_tokens

    body, defaulted = apply_default_max_tokens({"model": "gpt-4o", "messages": []}, 1024)
    assert body["max_tokens"] == 1024
    assert defaulted


def test_apply_default_max_tokens_noop_when_present():
    """apply default max tokens noop when present."""
    from src.protection.limits import apply_default_max_tokens

    body, defaulted = apply_default_max_tokens({"model": "gpt-4o", "messages": [], "max_tokens": 50}, 1024)
    assert body["max_tokens"] == 50
    assert not defaulted


def test_cheapest_capable_model_skips_incapable_providers():
    """cheapest capable model skips incapable providers."""
    from src.protection.limits import cheapest_capable_model

    pool = ["mistral-small", "gpt-4o-mini", "gpt-4o"]
    aliases = {"mistral-small": "mistral/mistral-small-latest", "gpt-4o-mini": "openai/gpt-4o-mini", "gpt-4o": "openai/gpt-4o"}
    caps = {a: {"vision": True, "json_schema": True, "tools": True} for a in pool}
    caps["mistral-small"]["vision"] = False
    choice = cheapest_capable_model(pool, {"mistral", "openai"}, {"mistral", "openai"}, aliases, {"vision"}, caps)
    assert choice == "gpt-4o-mini"


def test_cheapest_capable_model_none_when_nothing_qualifies():
    """cheapest capable model none when nothing qualifies."""
    from src.protection.limits import cheapest_capable_model

    pool = ["mistral-small"]
    aliases = {"mistral-small": "mistral/mistral-small-latest"}
    caps = {"mistral-small": {"vision": False, "json_schema": True, "tools": True}}
    choice = cheapest_capable_model(pool, {"mistral"}, {"mistral"}, aliases, {"vision"}, caps)
    assert choice is None


def test_circuit_breaker_flags_a_repeating_loop():
    """circuit breaker flags a repeating loop."""
    from src.protection.limits import CircuitBreaker

    breaker = CircuitBreaker(window_seconds=60, max_signatures=5, similarity_threshold=0.95)
    messages = [
        {"role": "user", "content": "do the task"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "1", "function": {"name": "x", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "1", "content": "same result every time"},
    ]
    results = [breaker.check("session-a", messages) for _ in range(8)]
    assert any(results)
    assert results.index(True) == 5  # the 6th call ("more than 5" prior matches)


def test_circuit_breaker_does_not_flag_changing_tool_output():
    """circuit breaker does not flag changing tool output."""
    from src.protection.limits import CircuitBreaker

    breaker = CircuitBreaker(window_seconds=60, max_signatures=5, similarity_threshold=0.95)
    results = []
    for i in range(8):
        messages = [
            {"role": "user", "content": "do the task"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": str(i), "function": {"name": "x", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": str(i), "content": f"result page {i}"},
        ]
        results.append(breaker.check("session-b", messages))
    assert not any(results)


def test_circuit_breaker_sessions_are_independent():
    """circuit breaker sessions are independent."""
    from src.protection.limits import CircuitBreaker

    breaker = CircuitBreaker(window_seconds=60, max_signatures=5, similarity_threshold=0.95)
    messages = [{"role": "user", "content": "hello"}]
    for _ in range(8):
        breaker.check("session-busy", messages)
    assert not breaker.check("session-fresh", messages)


def test_circuit_breaker_forgets_idle_sessions():
    """Session history must not grow without bound: idle sessions are dropped once the limit is passed."""
    import time as time_module

    from src.protection.limits import CircuitBreaker

    breaker = CircuitBreaker(window_seconds=0.05, max_signatures=5, similarity_threshold=0.95)
    breaker.max_tracked_sessions = 3
    for i in range(5):
        breaker.check(f"session-{i}", [{"role": "user", "content": f"message {i}"}])
    time_module.sleep(0.06)
    breaker.check("fresh", [{"role": "user", "content": "new"}])
    assert len(breaker._history) <= 2
