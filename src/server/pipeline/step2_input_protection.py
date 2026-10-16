"""
Step 2: mask personal data and score prompt injection before anything leaves.
A security step: if it crashes, the request is stopped (503) by default.
"""

import logging

from fastapi import Request
from fastapi.responses import JSONResponse

from src.protection import guardrails
from src.server.http_utils import openai_error
from src.server.pipeline.context import RequestContext
from src.server.services import Services

logger = logging.getLogger("cordon.pipeline")


def run(request: Request, ctx: RequestContext, services: Services) -> JSONResponse | None:
    """Mask PII and detect injection on the input. Returns an error response, or None to continue."""
    tenant_settings = ctx.tenant_settings
    pii_action = tenant_settings.get("pii_action", "mask")
    injection_action = tenant_settings.get("injection_action", "flag")
    security_fail_mode = tenant_settings.get("security_fail_mode", "closed")

    try:
        if ctx.mock_fail_step == "input_guardrails":
            raise RuntimeError("simulated input guardrail failure (X-Mock-Fail-Step)")
        result = guardrails.apply_input_guardrails(
            ctx.messages, pii_action, injection_action, observe=ctx.observe,
            include_instructions=bool(tenant_settings.get("mask_instructions")),
        )
    except Exception:
        logger.exception("input guardrails failed")
        if security_fail_mode == "closed":
            ctx.book(services, status="error", error_code="security_control_failed", served_alias=None)
            return openai_error(503, "a security control failed", "server_error", extra_headers={"X-Cordon-Error": "security"})
        result = guardrails.InputGuardrailResult(
            masked_messages=ctx.messages, placeholder_mapping={}, own_data=set(), pii_counts={}, injection_score=0.0, injection_detected=False
        )

    for pii_type, count in result.pii_counts.items():
        ctx.guardrail_actions[f"input_pii_{pii_type}"] = count
    if result.injection_detected:
        ctx.guardrail_actions["injection_flagged"] = 1

    if result.blocked:
        block_code = "pii_blocked" if result.blocked_reason == "pii" else "injection_blocked"
        ctx.block(services, block_code, guardrail_actions=ctx.guardrail_actions)
        if result.blocked_reason == "pii":
            return openai_error(400, f"input contains PII: {result.blocked_pii_type}", "invalid_request_error", code="pii_blocked")
        return openai_error(400, "prompt injection detected", "invalid_request_error", code="injection_blocked")

    ctx.masked_messages = result.masked_messages
    ctx.own_data = result.own_data
    ctx.placeholder_mapping = result.placeholder_mapping
    ctx.pii_counts = result.pii_counts
    return None
