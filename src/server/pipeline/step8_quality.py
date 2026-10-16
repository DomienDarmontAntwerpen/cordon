"""
Step 8: the route's check. When a route's cheaper model answered and the check
(rules, then the classifier at the route's proven threshold) is not satisfied,
ask the strong model once. Both calls are billed. Runs only when a route was
taken (enforce, not holdout); if the check crashes, the strong model is asked:
the route was only proven together with its check. If the strong model cannot
answer either, the doubted answer is never served or cached: a 502.
"""

import logging
import time

from fastapi.responses import JSONResponse

from src.savings import costing, quality
from src.server.http_utils import openai_error
from src.server.pipeline import ladder
from src.server.pipeline.context import RequestContext
from src.server.pipeline.step7_generate import recount_usage
from src.server.services import Services

logger = logging.getLogger("cordon.pipeline")


def run(ctx: RequestContext, services: Services) -> JSONResponse | None:
    """Check a routed answer and escalate once if needed. Returns an error only when the escalation itself fails."""
    if ctx.cache_hit_text is not None:
        return None

    # A routed request never carries tools (routing.choose_route), but a model may still answer with a tool call.
    if ctx.gate_applies() and ctx.served_alias == ctx.route.model and not ctx.answer_tool_calls:
        try:
            verdict = quality.gate_verdict(ctx.last_user_text_masked, ctx.answer_text, services.config,
                                           ctx.route.gate_threshold, ctx.completion_kwargs.get("response_format"),
                                           confidence=ctx.answer_confidence)
        except Exception:
            logger.warning("quality check failed: asking the strong model", exc_info=True)
            verdict = quality.GateVerdict(True, 0.0, "error")
        ctx.quality_score = verdict.score
        if verdict.escalate:
            error = _escalate_to_strong(ctx, services)
            if error is not None:
                return error

    ctx.tokens_in, ctx.tokens_out = ctx.usage.prompt_tokens, ctx.usage.completion_tokens
    return None


def _escalate_to_strong(ctx: RequestContext, services: Services) -> JSONResponse | None:
    """Ask the strong model (through the shared ladder); its answer replaces the doubted one. An error if it cannot."""
    start = time.perf_counter()
    try:
        answered = ladder.call_ladder(ctx, [ctx.strong_alias])
    except ladder.RequestRejected as exc:  # the strong model refuses the request itself: as in step 7, a 400
        _book_unserved(ctx, services, "provider_rejected", exc, start)
        return openai_error(400, "the provider rejected the request: invalid, or not supported by any allowed model",
                            "invalid_request_error", code="provider_rejected")
    except ladder.LadderFailed as exc:
        _book_unserved(ctx, services, "provider_error", exc, start)
        return openai_error(502, "the answer did not pass the quality check and the strong model could not be reached",
                            "api_error", code="provider_error", extra_headers={"X-Cordon-Error": "internal"})
    ctx.provider_ms += (time.perf_counter() - start) * 1000
    response, usage = answered.response, answered.response.usage
    text = response.choices[0].message.content or ""
    if services.settings.mock_mode:
        recount_usage(usage, answered.litellm_name, ctx.completion_kwargs["messages"], text)
    ctx.total_actual_eur += costing.actual_cost_eur(answered.litellm_name, usage.prompt_tokens, usage.completion_tokens,
                                                    services.config)
    ctx.answer_text, ctx.answer_tool_calls, ctx.usage = text, response.choices[0].message.tool_calls, usage
    ctx.served_alias, ctx.served_litellm_name, ctx.fallback_occurred = answered.alias, answered.litellm_name, True
    # Both models were paid for: the request cost more than the strong model alone, so nothing was saved.
    ctx.savings.pop("routing", None)
    return None


def _book_unserved(ctx: RequestContext, services: Services, code: str, exc: Exception, start: float) -> None:
    """The cheaper model was paid but its answer did not pass the check, so it is not served: book what was paid."""
    ctx.provider_ms += (time.perf_counter() - start) * 1000
    ctx.book(services, status="error", error_code=code, error_detail=f"escalation failed: {exc.args[0]}",
             actual_eur=ctx.total_actual_eur, tokens_out=ctx.usage.completion_tokens)
