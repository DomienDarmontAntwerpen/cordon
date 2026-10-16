"""
Step 4: classify the request once, then trim text and shrink images.
In observe mode or holdout the request is left unchanged and the saving is
recorded as "would save". If it crashes, the request continues unoptimized.
"""

import logging

import litellm
from fastapi import Request
from fastapi.responses import JSONResponse

from src.savings import costing, optimize, routing
from src.server.pipeline.context import RequestContext
from src.server.services import Services

logger = logging.getLogger("cordon.pipeline")


def run(request: Request, ctx: RequestContext, services: Services) -> JSONResponse | None:
    """Classify the request and apply trim and image optimization. Never returns an error."""
    # Classified on the masked, not-yet-trimmed messages: trimming removes
    # duplicates, not the keywords classification looks for.
    ctx.request_category, ctx.request_difficulty = routing.classify_request(ctx.masked_messages, ctx.body.get("response_format"))
    ctx.effective_messages = ctx.masked_messages
    if "trim" in ctx.features:
        _trim(ctx, services)
    if "vision" in ctx.features and ctx.image_count > 0:
        _optimize_images(ctx, services)
    return None


def _trim(ctx: RequestContext, services: Services) -> None:
    """Remove duplicate text, old history and long tool output; record the saving."""
    config = services.config
    try:
        if ctx.mock_fail_step == "optimize":
            raise RuntimeError("simulated optimization failure (X-Mock-Fail-Step)")
        candidate = optimize.trim_messages(ctx.masked_messages)
        candidate = optimize.trim_history(candidate, config["context"]["history_max_tokens"], ctx.litellm_name)
        candidate = optimize.truncate_tool_output(candidate, config["context"]["tool_output_max_tokens"], ctx.litellm_name)
        candidate_tokens = litellm.token_counter(model=ctx.litellm_name, messages=candidate)
        masked_tokens = litellm.token_counter(model=ctx.litellm_name, messages=ctx.masked_messages)
        saved_tokens = max(0, masked_tokens - candidate_tokens)
    except Exception:
        logger.warning("trim/history/tool-output optimization failed (fail open)", exc_info=True)
        candidate, saved_tokens = ctx.masked_messages, 0

    if saved_tokens > 0:
        eur = costing.cost_eur(ctx.litellm_name, saved_tokens, 0, config)
        ctx.savings["trim"] = costing.savings_entry("trim", eur, applied=ctx.cost_opts_apply())
    if ctx.cost_opts_apply():
        ctx.effective_messages = candidate


def _optimize_images(ctx: RequestContext, services: Services) -> None:
    """Downscale images (and strip metadata); record the saving. On any failure the request is left as it was."""
    tenant_settings = ctx.tenant_settings
    image_max_side = tenant_settings.get("image_max_side", 512)
    detail_low_for_photos = bool(tenant_settings.get("image_detail_low_for_photos", True))
    try:
        candidate, saved_tokens = [], 0
        for message in ctx.effective_messages:
            new_message, saved = _shrink_images(message, image_max_side, detail_low_for_photos, ctx.request_category == "extraction")
            candidate.append(new_message)
            saved_tokens += saved
    except Exception:
        logger.warning("image optimization failed (fail open)", exc_info=True)
        return

    if saved_tokens > 0:
        eur = costing.cost_eur(ctx.litellm_name, saved_tokens, 0, services.config)
        ctx.savings["vision"] = costing.savings_entry("vision", eur, applied=ctx.cost_opts_apply())
    if ctx.cost_opts_apply():
        ctx.effective_messages = candidate


def _shrink_images(message: dict, max_side: int, detail_low_for_photos: bool, is_document: bool) -> tuple[dict, int]:
    """A copy of message with every image part downscaled, and the tokens saved. The input is never changed."""
    content = message.get("content")
    if not isinstance(content, list):
        return message, 0
    new_content, saved = [], 0
    for part in content:
        if isinstance(part, dict) and part.get("type") == "image_url":
            result = optimize.optimize_image_part(part, max_side, detail_low_for_photos, is_document=is_document)
            saved += result.savings_tokens
            part = {"type": "image_url", "image_url": {"url": result.data_url, "detail": result.detail}}
        new_content.append(part)
    return {**message, "content": new_content}, saved
