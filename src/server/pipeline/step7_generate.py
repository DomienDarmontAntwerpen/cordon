"""
Step 7: get the answer - from the cache, or from the models on the ladder,
failing over when a provider fails (502 when every model fails; a request the
provider rejects as invalid is a 400 at once - see ladder.py). Records the
actual cost and the routing saving.
"""

import logging
import time

import litellm
from fastapi.responses import JSONResponse

from src.savings import costing, quality
from src.server.http_utils import openai_error
from src.server.pipeline import ladder, mock
from src.server.pipeline.context import RequestContext
from src.server.services import Services

logger = logging.getLogger("cordon.pipeline")


def run(ctx: RequestContext, services: Services) -> JSONResponse | None:
    """Fill in ctx's answer, usage and cost. Returns a 502 if every model failed, else None."""
    if ctx.cache_hit_text is not None:
        ctx.answer_text = ctx.cache_hit_text
        ctx.tokens_in = ctx.original_prompt_tokens
        ctx.tokens_out = litellm.token_counter(model=ctx.served_litellm_name, text=ctx.answer_text)
        ctx.provider_ms = 0.0
        return None

    config = services.config
    provider_start = time.perf_counter()
    try:
        answered = ladder.call_ladder(ctx)
    except ladder.RequestRejected as exc:
        ctx.provider_ms = (time.perf_counter() - provider_start) * 1000
        ctx.book(services, status="error", error_code="provider_rejected", error_detail=str(exc), served_alias=None)
        return openai_error(400, "the provider rejected the request: invalid, or not supported by any allowed model", "invalid_request_error", code="provider_rejected")
    except ladder.LadderFailed as exc:
        ctx.provider_ms = (time.perf_counter() - provider_start) * 1000
        ctx.book(services, status="error", error_code="provider_error", error_detail=exc.args[0], served_alias=None)
        return openai_error(502, "the upstream provider call failed", "api_error", code="provider_error",
                            extra_headers={"X-Cordon-Error": "internal"})
    ctx.provider_ms = (time.perf_counter() - provider_start) * 1000
    ctx.served_alias, ctx.served_litellm_name, response = answered.alias, answered.litellm_name, answered.response

    ctx.fallback_occurred = ctx.served_alias != ctx.ladder[0]
    ctx.usage = response.usage
    ctx.answer_text = response.choices[0].message.content or ""
    ctx.answer_tool_calls = response.choices[0].message.tool_calls
    ctx.answer_finish_reason = response.choices[0].finish_reason or "stop"
    ctx.answer_confidence = quality.answer_confidence(response)
    if services.settings.mock_mode and ctx.answer_confidence is None and ctx.served_alias in ctx.confidence_models:
        ctx.answer_confidence = mock.mock_confidence(ctx)

    if services.settings.mock_mode:
        # Bill what was actually sent (after trim/vision), as a real provider
        # would - otherwise trim and vision savings never show in actual cost.
        recount_usage(ctx.usage, ctx.served_litellm_name, ctx.completion_kwargs["messages"], ctx.answer_text)

    ctx.total_actual_eur = costing.actual_cost_eur(ctx.served_litellm_name, ctx.usage.prompt_tokens, ctx.usage.completion_tokens, config)

    # Routing saving (always "estimated"): only when the cheaper first rung of the ladder answered.
    if len(ctx.ladder) > 1 and ctx.ladder[0] != ctx.strong_alias and ctx.served_alias == ctx.ladder[0]:
        strong_cost = costing.cost_eur(
            ctx.model_aliases[ctx.strong_alias], ctx.usage.prompt_tokens, ctx.usage.completion_tokens, config
        )
        routing_saving = max(0.0, strong_cost - ctx.total_actual_eur)
        if routing_saving > 0:
            ctx.savings["routing"] = costing.savings_entry("routing", routing_saving)
    return None


def recount_usage(usage, litellm_name: str, messages: list[dict], answer_text: str) -> None:
    """Mock mode only: set usage's token counts from the messages sent and the answer received."""
    prompt_tokens = costing.count_message_tokens(litellm_name, messages)
    completion_tokens = litellm.token_counter(model=litellm_name, text=answer_text) if answer_text else 0
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = prompt_tokens + completion_tokens
