"""
The last resort for an unexpected crash in steps 1-6, after the caller was verified.
`security_fail_mode: closed` (default) answers 503; `open` sends the original
request straight to the model and books it as `internal_fail_open`.
"""

import json
import logging

import litellm
from fastapi.responses import JSONResponse, Response, StreamingResponse

from src.protection import security
from src.savings import costing
from src.server.http_utils import openai_error
from src.server.pipeline.context import RequestContext
from src.server.services import Services

logger = logging.getLogger("cordon.pipeline")


def passthrough_or_fail_closed(ctx: RequestContext, services: Services) -> Response:
    """Return a 503 (fail closed), or the requested model's answer to the original request (fail open)."""
    tenant_settings = ctx.tenant_settings
    # Never pass through a request whose size, rate and budget limits were not all checked:
    # "fail open" may skip guardrails by the tenant's choice, never the limits.
    if tenant_settings.get("security_fail_mode", "closed") != "open" or not ctx.limits_passed:
        _book_failure(ctx, services, "internal_fail_closed")
        return openai_error(503, "an internal error occurred", "server_error", extra_headers={"X-Cordon-Error": "security"})

    settings_obj = services.settings
    try:
        kwargs = {k: v for k, v in ctx.original_body.items() if k in security.ALLOWED_REQUEST_PARAMS}
        streamed = bool(kwargs.pop("stream", False))
        kwargs.pop("stream_options", None)
        kwargs["model"] = ctx.litellm_name
        kwargs["timeout"] = services.config["provider"]["timeout_seconds"]
        if settings_obj.mock_mode:
            kwargs["mock_response"] = "This is a mock answer from Cordon."
        response = litellm.completion(**kwargs)
        usage = response.usage
        if settings_obj.mock_mode:
            completion_text = response.choices[0].message.content or ""
            usage.prompt_tokens = costing.count_message_tokens(ctx.litellm_name, kwargs.get("messages", []))
            usage.completion_tokens = litellm.token_counter(model=ctx.litellm_name, text=completion_text) if completion_text else 0
        actual_eur = costing.actual_cost_eur(ctx.litellm_name, usage.prompt_tokens, usage.completion_tokens, services.config)
        ctx.served_alias, ctx.tokens_in = ctx.requested_alias, usage.prompt_tokens
        ctx.book(services, status="passthrough", error_code="internal_fail_open", actual_eur=actual_eur, baseline_eur=actual_eur,
                 tokens_out=usage.completion_tokens, error_detail="pipeline failure, passed through to the requested model")
        headers = {"X-Cordon-Error": "internal", "X-Cordon-Mode": tenant_settings.get("mode", "observe")}
        if streamed:
            # The caller asked for a stream: send the whole answer as one valid SSE chunk.
            return StreamingResponse(_as_single_chunk_stream(response), media_type="text/event-stream", headers=headers)
        return JSONResponse(content=response.model_dump(), headers=headers)
    except Exception:
        logger.exception("passthrough call also failed")
        _book_failure(ctx, services, "internal_fail_open_failed")
        return openai_error(502, "the upstream provider call failed", "api_error", extra_headers={"X-Cordon-Error": "internal"})


def _as_single_chunk_stream(response):
    """Yield a complete (non-streamed) completion as one OpenAI-format SSE chunk, then [DONE]."""
    body = response.model_dump()
    chunk = {
        "id": body.get("id"), "object": "chat.completion.chunk", "created": body.get("created"), "model": body.get("model"),
        "choices": [{"index": 0, "delta": {"content": body["choices"][0]["message"].get("content") or ""}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
    yield b"data: [DONE]\n\n"


def _book_failure(ctx: RequestContext, services: Services, error_code: str) -> None:
    """Book a request that ended in the unexpected-failure handler without an answer. Never raises."""
    try:
        if ctx.tenant_id:  # a request that never got past the API-key check has nothing to book
            ctx.book(services, status="error", error_code=error_code, served_alias=None)
    except Exception:
        logger.exception("could not book a failed request")
