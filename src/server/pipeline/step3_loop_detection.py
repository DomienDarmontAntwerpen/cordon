"""
Step 3: stop an agent stuck in a loop (`circuit_breaker` feature, needs X-Session-ID).
A cost control: if it crashes, the request continues.
"""

import logging

from fastapi import Request
from fastapi.responses import JSONResponse

from src.server.http_utils import openai_error
from src.server.pipeline.context import RequestContext
from src.server.services import Services

logger = logging.getLogger("cordon.pipeline")


def run(request: Request, ctx: RequestContext, services: Services) -> JSONResponse | None:
    """Block a detected agent loop. Returns an error response, or None to continue."""
    if "circuit_breaker" not in ctx.features or not ctx.session_id_header:
        return None
    try:
        loop_detected = services.circuit_breaker.check(ctx.session_hash, ctx.masked_messages)
    except Exception:
        logger.warning("circuit breaker failed (fail open)", exc_info=True)
        loop_detected = False
    if not loop_detected:
        return None
    if not ctx.observe:
        ctx.block(services, "agent_loop_detected", session_hash=ctx.session_hash)
        return openai_error(429, "agent loop detected", "rate_limit_error", code="agent_loop_detected")
    ctx.soft_violations.append("agent_loop_detected")
    return None
