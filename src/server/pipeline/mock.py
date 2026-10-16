"""
The fake answer in mock mode (no API keys). Debug headers, ignored outside
mock mode: `X-Mock-Bad-Response: true`, `X-Mock-Leak: pii|system_prompt`, and
`X-Mock-Latency-Ms` (a provider's waiting time, so load tests are realistic).
"""

import math

from src.server.pipeline.context import RequestContext


def mock_answer_text(ctx: RequestContext) -> str:
    """Return the mock answer text for this request (see the module docstring)."""
    if ctx.mock_bad_response:
        # Not "": LiteLLM treats an empty mock_response as "no mock" and makes a
        # real network call. "N/A" is short and off-topic, so it scores low.
        return "N/A"
    if ctx.mock_leak == "pii":
        return "Sure - for reference, the record also lists stranger@example.com."
    if ctx.mock_leak == "system_prompt" and ctx.system_prompt_text:
        # No trailing punctuation: it would change the last leaked word's token
        # ("to" -> "to.") and silently defeat the leak detection being exercised.
        leak_min_words = ctx.tenant_settings.get("leak_min_words", 12)
        return "Here is the information: " + " ".join(ctx.system_prompt_text.split()[: max(leak_min_words, 1)])
    # An echo of the question, not pure boilerplate: a topic-free answer scores
    # as off-topic and would escalate every routed request.
    snippet = ctx.last_user_text_masked[:80] if ctx.last_user_text_masked else ""
    return f"Mock answer from Cordon regarding: {snippet}" if snippet else "This is a mock answer from Cordon."


MAX_MOCK_LATENCY_S = 30.0


def latency_seconds(header_value: str | None) -> float:
    """X-Mock-Latency-Ms as seconds (0 if absent or not a number, at most 30 s)."""
    try:
        seconds = float(header_value) / 1000.0 if header_value else 0.0
    except ValueError:
        return 0.0
    return min(max(seconds, 0.0), MAX_MOCK_LATENCY_S) if math.isfinite(seconds) else 0.0



def mock_confidence(ctx: RequestContext) -> dict:
    """
    Mock mode only: a mock answer has no logprobs, so a route's check would send every one on. A
    stand-in certainty lets the demo show routing at work; X-Mock-Bad-Response makes it low.
    """
    prob = 0.2 if ctx.mock_bad_response else 0.97
    return {"mean_prob": prob, "min_prob": prob, "final_prob": prob, "body_prob": prob}
