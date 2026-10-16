"""
Step 9: protect the answer, store it in the cache, restore the real values,
book the request and build the response with the `X-Cordon-*` headers.
If output protection crashes, the answer is withheld (503) by default.
"""

import logging
import time
import uuid

from fastapi.responses import JSONResponse

from src.protection import guardrails
from src.savings import costing
from src.server.http_utils import openai_error
from src.server.pipeline import step5_cache
from src.server.pipeline.context import RequestContext
from src.server.pipeline.final_savings import final_savings
from src.server.services import Services

logger = logging.getLogger("cordon.pipeline")


# --- Section: output protection, cache store, restore ---
def protect_output(ctx: RequestContext, services: Services) -> JSONResponse | None:
    """Run the output guardrails, store the answer in the cache, restore placeholders. Returns an error, or None."""
    tenant_settings = ctx.tenant_settings
    try:
        result = guardrails.apply_output_guardrails(
            ctx.answer_text, ctx.own_data, ctx.system_prompt_text,
            tenant_settings.get("output_pii_action", "redact"), tenant_settings.get("output_leak_action", "redact"),
            tenant_settings.get("leak_min_words", 12), observe=ctx.observe,
        )
    except Exception:
        logger.exception("output guardrails failed")
        if tenant_settings.get("security_fail_mode", "closed") == "closed":
            ctx.book(services, status="error", error_code="security_control_failed",
                     actual_eur=ctx.total_actual_eur, tokens_out=ctx.tokens_out)
            return openai_error(503, "a security control failed", "server_error", extra_headers={"X-Cordon-Error": "security"})
        result = guardrails.OutputGuardrailResult(text=ctx.answer_text)

    actions = ctx.guardrail_actions
    for pii_type, count in result.pii_redacted_counts.items():
        actions[f"output_pii_{pii_type}"] = actions.get(f"output_pii_{pii_type}", 0) + count
    if result.leak_redacted:
        actions["system_prompt_leak_redacted"] = actions.get("system_prompt_leak_redacted", 0) + 1

    if result.blocked:
        # The provider was paid for the answer even though the caller never sees it.
        ctx.book(services, status="blocked", error_code=f"output_blocked_{result.blocked_reason}",
                 actual_eur=ctx.total_actual_eur, tokens_out=ctx.tokens_out)
        if result.blocked_reason == "pii":
            return openai_error(422, f"output contains PII: {result.blocked_pii_type}", "invalid_request_error", code="output_blocked")
        return openai_error(422, "output leaked the system prompt", "invalid_request_error", code="output_blocked")

    ctx.processed_text = result.text
    if ctx.answer_finish_reason != "length":  # a cut-off answer is never replayed as a complete one
        step5_cache.store_answer(ctx, services, ctx.processed_text)

    tool_calls = [tc.model_dump() for tc in ctx.answer_tool_calls] if ctx.answer_tool_calls else None
    ctx.restored_message, unrestored = guardrails.restore_message(
        {"role": "assistant", "content": ctx.processed_text, "tool_calls": tool_calls}, ctx.placeholder_mapping,
    )
    if unrestored:
        actions["restore_failed"] = actions.get("restore_failed", 0) + unrestored
    return None


# --- Section: bookkeeping and the response ---
def finish(ctx: RequestContext, services: Services) -> JSONResponse:
    """Book the request (budget, ledger, telemetry) and build the final OpenAI-format response."""
    config = services.config
    tenant_settings = ctx.tenant_settings
    actual_eur = 0.0 if ctx.cache_hit_text is not None else ctx.total_actual_eur
    baseline_eur = costing.baseline_cost_eur(ctx.litellm_name, ctx.original_prompt_tokens, ctx.tokens_out, config)

    savings = final_savings(ctx, config, baseline_eur, ctx.tokens_out)
    overhead_ms = (time.perf_counter() - ctx.start) * 1000 - ctx.provider_ms
    ctx.book(services, status="ok", actual_eur=actual_eur, baseline_eur=baseline_eur, tokens_out=ctx.tokens_out, savings=savings)
    services.telemetry.record_request(
        ctx.tenant_id,
        tenant_settings.get("telemetry_content", "none"),
        metadata={
            "served_model": ctx.served_alias, "tokens_in": ctx.tokens_in, "tokens_out": ctx.tokens_out,
            "cost_eur": actual_eur, "baseline_eur": baseline_eur,
        },
        input_text=ctx.last_user_text_masked,
        output_text=ctx.processed_text,
    )

    headers = {
        "X-Cordon-Cost": f"{actual_eur:.6f}",
        "X-Cordon-Baseline-Cost": f"{baseline_eur:.6f}",
        "X-Cordon-Cache": ctx.cache_type_on_hit or "none",
        "X-Cordon-Route": ctx.served_alias,
        "X-Cordon-Kind": f"{ctx.request_category}/{ctx.request_difficulty}",  # the routing bucket it fell in
        "X-Cordon-Fallback": "true" if ctx.fallback_occurred else "false",
        "X-Cordon-Overhead-Ms": f"{overhead_ms:.2f}",
        "X-Cordon-Mode": tenant_settings.get("mode", "observe"),
    }
    if ctx.quality_score is not None:
        headers["X-Cordon-Quality"] = f"{ctx.quality_score:.4f}"
        headers["X-Cordon-Check"] = ctx.quality_reason or ""  # e.g. rules_only, classifier, refusal
    if ctx.guardrail_actions:  # counts per kind only (e.g. input_pii_email=1), never a value
        headers["X-Cordon-Guardrails"] = ",".join(f"{name}={count}" for name, count in sorted(ctx.guardrail_actions.items()))

    response_body = {
        "id": f"chatcmpl-cordon-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": ctx.served_litellm_name,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": ctx.restored_message["content"],
                    "tool_calls": ctx.restored_message.get("tool_calls"),
                },
                "finish_reason": ctx.answer_finish_reason,
            }
        ],
        "usage": {"prompt_tokens": ctx.tokens_in, "completion_tokens": ctx.tokens_out, "total_tokens": ctx.tokens_in + ctx.tokens_out},
    }
    return JSONResponse(content=response_body, headers=headers)
