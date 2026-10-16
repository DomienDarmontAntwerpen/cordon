"""
The conductor: runs the steps for one chat request, in order. Start reading here.

- Step 1's access check runs on the event loop; everything after it runs in the
  request thread pool (above `server.max_in_flight`: 503 server_busy).
- A crash in steps 1-6 goes to passthrough.py; a crash after the model call is
  booked and answered 503.
- Every step returns an error response (stop) or None (go on).
"""

import asyncio
import itertools
import logging
import time

from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from src.server.http_utils import client_ip, openai_error
from src.server.pipeline import (
    mock,
    passthrough,
    step1_admission,
    step2_input_protection,
    step3_loop_detection,
    step4_optimize,
    step5_cache,
    step6_routing,
    step7_generate,
    step8_quality,
    step9_output,
    streaming,
)
from src.server.pipeline.context import RequestContext
from src.server.services import Services

logger = logging.getLogger("cordon.pipeline")

# Steps 1 (limits) to 6: prepare the request. Signature: (request, ctx, services) -> error or None.
PREPARE_STEPS = (
    step1_admission.check_limits,
    step2_input_protection.run,
    step3_loop_detection.run,
    step4_optimize.run,
    step5_cache.lookup,
    step6_routing.run,
)

# Steps 7 to 9 (non-streaming): get, check and protect the answer. Signature: (ctx, services) -> error or None.
ANSWER_STEPS = (
    step7_generate.run,
    step8_quality.run,
    step9_output.protect_output,
)


async def handle_chat_completion(request: Request, services: Services) -> Response:
    """Run one chat-completion request through the pipeline and return the response to send."""
    ctx = RequestContext(start=time.perf_counter(), client_ip=client_ip(request))
    if services.settings.mock_mode:
        ctx.mock_fail_step = request.headers.get("x-mock-fail-step")
        ctx.mock_latency_s = mock.latency_seconds(request.headers.get("x-mock-latency-ms"))

    error = await step1_admission.check_access(request, ctx, services)
    if error is not None:
        return error
    # The rest blocks (token counting, embeddings, the model call): run it on the
    # request thread pool, so one slow request never holds up the others - but
    # only while there is capacity; otherwise refuse at once.
    if not services.capacity.try_enter():
        return openai_error(503, "server is at capacity, retry shortly", "server_error", code="server_busy",
                            extra_headers={"Retry-After": "1"})
    try:
        return await asyncio.get_running_loop().run_in_executor(services.executor, _handle_admitted, request, ctx, services)
    finally:
        if not ctx.stream_owns_capacity_slot:  # a running stream gives its slot back when it ends
            services.capacity.leave()


def _handle_admitted(request: Request, ctx: RequestContext, services: Services) -> Response:
    """Steps 1 (limits) to 9 for a request that passed check_access(). Runs in a worker thread."""
    # Only after check_access(): passthrough skips every control, so the caller must already be verified.
    try:
        error = _run_steps(PREPARE_STEPS, request, ctx, services)
    except Exception:
        logger.exception("unexpected pipeline failure after admission")
        return passthrough.passthrough_or_fail_closed(ctx, services)
    if error is not None:
        # Blocked after step 1 reserved budget for it (e.g. PII or injection block): give it back.
        ctx.settle(services, 0.0)
        return error

    if ctx.completion_kwargs.get("stream"):
        ctx.completion_kwargs.setdefault("stream_options", {"include_usage": True})
        # The stream keeps its capacity slot until it ends, and is "primed" here: its
        # first chunk is produced now, so the generator has started and its cleanup
        # (booking, releasing budget and slot) runs even if the client disconnects
        # before reading anything. An unstarted generator never runs its `finally`.
        ctx.stream_owns_capacity_slot = True
        stream = streaming.stream_response(ctx, services, on_close=services.capacity.leave)
        try:
            first_chunk = next(stream)
        except Exception:  # booked as internal_error (and the slot released) by the stream itself
            return openai_error(503, "an internal error occurred", "server_error", extra_headers={"X-Cordon-Error": "internal"})
        return StreamingResponse(
            itertools.chain([first_chunk], stream),
            media_type="text/event-stream",
            headers={"X-Cordon-Mode": ctx.tenant_settings.get("mode", "observe")},
        )

    try:
        for step in ANSWER_STEPS:
            error = step(ctx, services)
            if error is not None:
                return error
        return step9_output.finish(ctx, services)
    except Exception:
        # The model may already have answered, so no passthrough here: book what was
        # spent, release the rest of the reservation, and fail cleanly.
        logger.exception("unexpected failure after the model call")
        _book_internal_failure(ctx, services)
        return openai_error(503, "an internal error occurred", "server_error", extra_headers={"X-Cordon-Error": "internal"})


def _book_internal_failure(ctx: RequestContext, services: Services) -> None:
    """Book the cost spent so far as an internal error. Never raises."""
    try:
        ctx.book(services, status="error", error_code="internal_error", actual_eur=ctx.total_actual_eur, tokens_out=ctx.tokens_out)
    except Exception:
        logger.exception("could not book an internal failure")


def _run_steps(steps: tuple, request: Request, ctx: RequestContext, services: Services) -> JSONResponse | None:
    """Run the given preparation steps in order; return the first error response, or None."""
    for step in steps:
        error = step(request, ctx, services)
        if error is not None:
            return error
    return None
