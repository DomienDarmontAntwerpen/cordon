"""
Step 5: answer from the cache when the same question was seen before.

- `lookup()`: the exact cache, then the semantic cache.
- Observe mode notes a hit (`ctx.would_hit_cache`) but never serves it, and
  stores only an empty marker - no answer text.
- `store_answer()`: called from step 9, after the answer passed its checks.
"""

import logging

from fastapi import Request
from fastapi.responses import JSONResponse

from src import messages
from src.savings.cache import check_eligibility
from src.server.pipeline.context import RequestContext
from src.server.services import Services

logger = logging.getLogger("cordon.pipeline")

# Request parameters that change the answer, besides model, messages, temperature,
# tools and response_format: two requests share a cache entry only if these match too.
ANSWER_SHAPING_PARAMS = ("max_tokens", "max_completion_tokens", "top_p", "stop", "seed", "presence_penalty", "frequency_penalty", "tool_choice")


def lookup(request: Request, ctx: RequestContext, services: Services) -> JSONResponse | None:
    """Look the request up in the exact and semantic caches. Never returns an error."""
    ctx.last_user_text_masked = messages.last_user_text(ctx.effective_messages)
    ctx.system_prompt_text = messages.instructions_text(ctx.effective_messages)

    tenant_settings = ctx.tenant_settings
    mode = tenant_settings.get("mode", "observe")
    body = ctx.body
    pii_unmasked = bool(ctx.pii_counts) and tenant_settings.get("pii_action", "mask") != "mask"
    eligibility = check_eligibility(
        ctx.holdout, pii_unmasked, bool(body.get("tools")), ctx.image_count > 0,
        body.get("temperature"), tenant_settings.get("cache_max_temperature", 1.0),
        ctx.last_user_text_masked, request.headers.get("x-cordon-cache", "").strip().lower() == "off",
    )
    # "cache" (exact) and "semantic_cache" are independent feature flags; either enables the shared eligibility rules.
    ctx.cache_eligible = eligibility.eligible and ("cache" in ctx.features or "semantic_cache" in ctx.features)
    if not ctx.cache_eligible:
        return None

    cache = services.cache
    generation_params = {k: v for k, v in body.items() if k in ANSWER_SHAPING_PARAMS}
    key_args = (
        ctx.tenant_id, mode, ctx.requested_alias, ctx.effective_messages,
        body.get("temperature"), body.get("tools"), body.get("response_format"), generation_params,
    )
    hit_text, hit_type = None, None
    if "cache" in ctx.features:
        ctx.cache_key = cache.exact_key(*key_args)
        hit_text = cache.get_exact(ctx.tenant_id, ctx.cache_key)
        hit_type = "exact" if hit_text is not None else None
    # Semantic matching compares text only. A question that also carries a file, audio or an
    # image must never match on its text alone ("Summarize this contract" + a different contract).
    if (hit_text is None and "semantic_cache" in ctx.features and ctx.last_user_text_masked.strip()
            and _question_is_text_only(ctx)):
        ctx.context_hash = cache.context_hash(*key_args)
        hit_text = cache.get_semantic(ctx.tenant_id, ctx.context_hash, ctx.last_user_text_masked)
        hit_type = "semantic" if hit_text is not None else None

    if hit_type is None:
        return None
    if ctx.observe:
        ctx.would_hit_cache = hit_type  # observe: count it, never serve it
    else:
        ctx.cache_hit_text, ctx.cache_type_on_hit = hit_text, hit_type
    return None


def store_answer(ctx: RequestContext, services: Services, text: str) -> None:
    """Write a fresh, guardrail-checked (still masked) answer to the cache. Fails open."""
    if ctx.cache_hit_text is not None or ctx.would_hit_cache or not ctx.cache_eligible:
        return
    if ctx.observe:
        text = ""  # a marker only: observe never serves from the cache, so it keeps no answer text
    try:
        if ctx.cache_key is not None:
            services.cache.put_exact(ctx.tenant_id, ctx.cache_key, text)
        if ctx.context_hash is not None:
            services.cache.put_semantic(ctx.tenant_id, ctx.context_hash, ctx.last_user_text_masked, text)
    except Exception:
        logger.warning("cache write failed (fail open)", exc_info=True)


def _question_is_text_only(ctx: RequestContext) -> bool:
    """True if the user's question carries no file, audio or image (semantic matching compares text only)."""
    index = messages.last_user_index(ctx.effective_messages)
    return index is None or messages.is_text_only(ctx.effective_messages[index])
