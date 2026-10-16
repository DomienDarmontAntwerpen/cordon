"""
Steps 7-9 for a streamed answer (`"stream": true`).
Failover only before the first chunk; every chunk passes the streaming output
guardrail; the request is booked exactly once however the stream ends.
Streaming never escalates and never writes the cache.
"""

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Iterator

import litellm

from src.protection import guardrails, security
from src.savings import costing
from src.server.pipeline import ladder
from src.server.pipeline.context import RequestContext
from src.server.pipeline.final_savings import final_savings
from src.server.services import Services

logger = logging.getLogger("cordon.pipeline")


@dataclass
class _Stream:
    """What one stream has produced so far (shared by the steps below)."""

    chunk_id: str = field(default_factory=lambda: f"chatcmpl-cordon-{uuid.uuid4().hex[:24]}")
    created: int = field(default_factory=lambda: int(time.time()))
    parts: list[str] = field(default_factory=list)  # every piece of text sent to the client
    started_on: str | None = None  # LiteLLM name of the model that is streaming, once one started
    fallback: bool = False

    def chunk(self, ctx: RequestContext, content: str, finish_reason: str | None = None) -> bytes:
        """One OpenAI-format SSE data chunk carrying `content` as the delta."""
        payload = {
            "id": self.chunk_id, "object": "chat.completion.chunk", "created": self.created, "model": ctx.served_litellm_name,
            "choices": [{"index": 0, "delta": {"content": content} if content else {}, "finish_reason": finish_reason}],
        }
        return f"data: {json.dumps(payload)}\n\n".encode("utf-8")


# --- Section: entry point ---
def stream_response(ctx: RequestContext, services: Services, on_close=None) -> Iterator[bytes]:
    """
    Yield the stream's SSE chunks; book the request and call on_close however the stream ends.

    A plain (not async) generator on purpose: Starlette runs it in a worker
    thread, so waiting on the provider's next chunk never blocks the event loop.
    If the client disconnects, the generator is closed at its current `yield`:
    the `finally` still books the tokens generated so far (the provider bills
    them), so closing a stream early is no way around the budget.
    """
    stream = _Stream()
    try:
        if ctx.cache_hit_text is not None:
            yield from _replay_cache_hit(ctx, stream)
        else:
            yield from _relay_provider_stream(ctx, services, stream)
        if not ctx.booked:
            yield from _finish(ctx, services, stream)
    except Exception:
        # A crash inside Cordon, not the client leaving: book it as such, then let it surface.
        logger.exception("stream failed inside Cordon")
        _book_crash(ctx, services, stream)
        raise
    finally:
        try:
            if not ctx.booked:
                _book_disconnect(ctx, services, stream)
        finally:
            if on_close is not None:
                on_close()


# --- Section: the two sources of text ---
def _replay_cache_hit(ctx: RequestContext, stream: _Stream) -> Iterator[bytes]:
    """Send a cached answer (placeholders restored) as one chunk."""
    text, unrestored = guardrails.restore_placeholders(ctx.cache_hit_text, ctx.placeholder_mapping)
    _count(ctx, "restore_failed", unrestored)
    stream.parts.append(text)
    yield stream.chunk(ctx, text)
    yield stream.chunk(ctx, "", finish_reason="stop")


def _relay_provider_stream(ctx: RequestContext, services: Services, stream: _Stream) -> Iterator[bytes]:
    """Start the provider stream (with failover) and relay it through the output guardrail."""
    provider_start = time.perf_counter()
    try:
        answered = ladder.call_ladder(ctx)
    except ladder.RequestRejected as exc:
        yield from _fail(ctx, services, str(exc), provider_start, code="provider_rejected")
        return
    except ladder.LadderFailed as exc:
        yield from _fail(ctx, services, exc.args[0], provider_start)
        return
    ctx.served_alias, ctx.served_litellm_name = answered.alias, answered.litellm_name
    stream.started_on, stream.fallback = answered.litellm_name, answered.alias != ctx.ladder[0]
    provider_stream = answered.response

    guard = _output_guard(ctx)
    try:
        for chunk in provider_stream:
            delta = _delta_text(chunk)
            safe_text = guard.feed(delta) if delta else ""
            if safe_text:
                stream.parts.append(safe_text)
                yield stream.chunk(ctx, safe_text)
    except Exception as exc:
        yield from _fail(ctx, services, security.sanitize_error(str(exc)), provider_start)
        return

    final_text = guard.finish()
    if final_text:
        stream.parts.append(final_text)
        yield stream.chunk(ctx, final_text)
    yield stream.chunk(ctx, "", finish_reason="stop")
    ctx.provider_ms = (time.perf_counter() - provider_start) * 1000
    _record_guard_findings(ctx, guard)


# --- Section: endings ---
def _fail(ctx: RequestContext, services: Services, error_text: str | None, provider_start: float,
          code: str = "provider_error") -> Iterator[bytes]:
    """Book a provider failure (or rejection) and send the SSE error event plus [DONE]."""
    ctx.provider_ms = (time.perf_counter() - provider_start) * 1000
    ctx.book(services, status="error", error_code=code, error_detail=error_text, served_alias=None)
    if code == "provider_rejected":
        error = {"error": {"message": "the provider rejected the request: invalid, or not supported by any allowed model", "type": "invalid_request_error", "code": code}}
    else:
        error = {"error": {"message": "the upstream provider call failed", "type": "api_error", "code": code}}
    yield f"data: {json.dumps(error)}\n\n".encode("utf-8")
    yield b"data: [DONE]\n\n"


def _finish(ctx: RequestContext, services: Services, stream: _Stream) -> Iterator[bytes]:
    """Book a completed stream, record telemetry, and send the `: cordon` summary line plus [DONE]."""
    config = services.config
    text = "".join(stream.parts)
    tokens_out = litellm.token_counter(model=ctx.served_litellm_name, text=text) if text else 0
    actual_eur = 0.0 if ctx.cache_hit_text is not None else _cost_of(ctx, ctx.served_litellm_name, tokens_out, config)
    baseline_eur = costing.baseline_cost_eur(ctx.litellm_name, ctx.original_prompt_tokens, tokens_out, config)
    ctx.book(services, status="ok", actual_eur=actual_eur, baseline_eur=baseline_eur, tokens_out=tokens_out,
             savings=final_savings(ctx, config, baseline_eur, tokens_out))
    services.telemetry.record_request(
        ctx.tenant_id, ctx.tenant_settings.get("telemetry_content", "none"),
        metadata={"served_model": ctx.served_alias, "tokens_in": ctx.original_prompt_tokens, "tokens_out": tokens_out, "cost_eur": actual_eur},
    )
    summary = {
        "cost_eur": round(actual_eur, 6),
        "cost_flag": "measured",
        "overhead_ms": round((time.perf_counter() - ctx.start) * 1000 - ctx.provider_ms, 2),
        "mode": ctx.tenant_settings.get("mode", "observe"),
        "cache": ctx.cache_type_on_hit or "none",
        "fallback": stream.fallback,
        "quality_score": None,  # a stream takes no route, so there is no check to score
    }
    yield f": cordon {json.dumps(summary)}\n\n".encode("utf-8")
    yield b"data: [DONE]\n\n"


def _book_crash(ctx: RequestContext, services: Services, stream: _Stream) -> None:
    """Book a stream that broke inside Cordon: what the provider already generated is charged. Never raises."""
    try:
        tokens_out, actual_eur = _emitted_cost(ctx, services, stream)
        ctx.book(services, status="error", error_code="internal_error", actual_eur=actual_eur, tokens_out=tokens_out)
    except Exception:
        logger.exception("could not book a failed stream")


def _emitted_cost(ctx: RequestContext, services: Services, stream: _Stream) -> tuple[int, float]:
    """(tokens, EUR) of the text a started provider stream has produced so far; (0, 0.0) if none started."""
    if stream.started_on is None:
        return 0, 0.0
    text = "".join(stream.parts)
    tokens_out = litellm.token_counter(model=stream.started_on, text=text) if text else 0
    return tokens_out, _cost_of(ctx, stream.started_on, tokens_out, services.config)


def _book_disconnect(ctx: RequestContext, services: Services, stream: _Stream) -> None:
    """Book a stream the client closed early: the tokens generated so far are charged. Never raises."""
    try:
        tokens_out, actual_eur = _emitted_cost(ctx, services, stream)
        ctx.book(services, status="error", error_code="client_disconnected", actual_eur=actual_eur, tokens_out=tokens_out,
                 error_detail="stream closed by the client before completion; cost estimated from emitted text")
        logger.info("stream closed before completion (client disconnected); booked EUR %.6f", actual_eur)
    except Exception:
        logger.exception("could not book a disconnected stream")


# --- Section: small helpers ---
def _output_guard(ctx: RequestContext) -> guardrails.StreamingOutputGuardrail:
    """The rolling-buffer output guardrail for this request's settings."""
    settings = ctx.tenant_settings
    return guardrails.StreamingOutputGuardrail(
        ctx.own_data, ctx.system_prompt_text,
        settings.get("output_pii_action", "redact"), settings.get("output_leak_action", "redact"),
        settings.get("leak_min_words", 12), ctx.placeholder_mapping, observe=ctx.observe,
    )


def _delta_text(chunk) -> str:
    """The text content of one provider stream chunk ("" if it carries none)."""
    choices = chunk.model_dump().get("choices") or []
    return (choices[0].get("delta") or {}).get("content") or "" if choices else ""


def _record_guard_findings(ctx: RequestContext, guard: guardrails.StreamingOutputGuardrail) -> None:
    """Add the output guardrail's counts to the request's guardrail actions (counts only, never values)."""
    for pii_type, count in guard.pii_counts.items():
        _count(ctx, f"output_pii_{pii_type}", count)
    _count(ctx, "system_prompt_leak_redacted", 1 if guard.leak_redacted else 0)
    _count(ctx, "restore_failed", guard.unrestored_count)


def _count(ctx: RequestContext, action: str, amount: int) -> None:
    """Add `amount` to one guardrail-action counter (no-op for 0)."""
    if amount:
        ctx.guardrail_actions[action] = ctx.guardrail_actions.get(action, 0) + amount


def _cost_of(ctx: RequestContext, litellm_name: str, tokens_out: int, config: dict) -> float:
    """What the provider bills: the messages actually sent (after trim/vision) plus the generated tokens."""
    prompt_tokens = costing.count_message_tokens(litellm_name, ctx.effective_messages)
    return costing.actual_cost_eur(litellm_name, prompt_tokens, tokens_out, config)

