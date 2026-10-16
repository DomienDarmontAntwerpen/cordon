"""
Step 1: may this request come in at all? The trust boundary.

- `check_access()`, always on: body size, API key, JSON, parameter allowlist,
  known priced model, allowed provider.
- `check_limits()`: sizes, tags, tokens, rate limit, budget reservation and the
  holdout sample. Blocks in enforce mode; only recorded in observe mode.
"""

import asyncio
import json
import logging
import uuid

from fastapi import Request
from fastapi.responses import JSONResponse

from src.protection import limits, security
from src.savings import costing, optimize
from src.server import settings as settings_module
from src.server.http_utils import BodyTooLargeError, extract_bearer, is_expired, openai_error, parse_tags, read_capped_body
from src.server.pipeline.context import RequestContext
from src.server.services import Services

logger = logging.getLogger("cordon.pipeline")

UNLIMITED = 10**9


def limit(value) -> float:
    """A tenant limit: None (not set) means unlimited; 0 means zero, not unlimited."""
    return UNLIMITED if value is None else value


# --- Section: always-on access checks ---
async def check_access(request: Request, ctx: RequestContext, services: Services) -> JSONResponse | None:
    """
    The always-on checks, in every mode: body size, API key, JSON body,
    parameter allowlist, model and provider. Fills in who is calling and for
    which model. Returns an error response, or None to continue.
    """
    try:
        raw_body = await read_capped_body(request, services.config["request_limits"]["max_body_bytes"])
    except BodyTooLargeError:  # checked before any parsing
        return openai_error(413, "request body exceeds the configured size limit", "invalid_request_error", code="body_too_large")
    return (
        await _authenticate(request, ctx, services)
        or _parse_body(raw_body, ctx)
        or _check_model_and_provider(ctx, services)
    )


async def _authenticate(request: Request, ctx: RequestContext, services: Services) -> JSONResponse | None:
    """
    Accept a valid, unrevoked, unexpired team key; throttle repeated *invalid* keys per IP.

    The key is checked first: behind a proxy every client shares one IP, and an
    attacker's bad keys must not lock out everyone else.
    """
    token = extract_bearer(request)
    tenant_row = None
    if token and token.startswith(security.API_KEY_PREFIX):
        # In a thread: the database lock may be held by a long query (an export, the hourly purge).
        record = await asyncio.to_thread(services.store.find_api_key, security.hash_api_key(token))
        if record and not record.revoked and not is_expired(record.expires_at):
            tenant_row = await asyncio.to_thread(services.store.get_tenant, record.tenant_id)
    if tenant_row is None:
        if services.throttle.is_throttled(ctx.client_ip):
            return openai_error(429, "too many authentication failures, try again later", "rate_limit_error", code="auth_throttled")
        services.throttle.record_failure(ctx.client_ip)
        return openai_error(401, "invalid API key", "authentication_error")

    ctx.tenant_id = tenant_row["id"]
    ctx.tenant_settings = tenant_row["settings"]
    ctx.key_prefix_value = security.key_prefix(token)
    ctx.observe = ctx.tenant_settings.get("mode", "observe") == "observe"
    ctx.features = set(ctx.tenant_settings.get("features") or [])
    return None


def _parse_body(raw_body: bytes, ctx: RequestContext) -> JSONResponse | None:
    """A JSON object whose every parameter is on the allowlist (anything else, e.g. api_base, is refused)."""
    try:
        body = json.loads(raw_body)
    except json.JSONDecodeError:
        return openai_error(400, "request body is not valid JSON", "invalid_request_error")
    if not isinstance(body, dict):
        return openai_error(400, "request body must be a JSON object", "invalid_request_error")
    try:
        security.validate_request_params(body)
    except security.ParameterNotAllowedError as exc:
        return openai_error(400, str(exc), "invalid_request_error", code="param_not_allowed")
    except security.InvalidParameterValueError as exc:
        return openai_error(400, str(exc), "invalid_request_error", code="invalid_param_value")
    ctx.original_body = body
    ctx.body = body
    return None


def _check_model_and_provider(ctx: RequestContext, services: Services) -> JSONResponse | None:
    """A known, priced model from a provider this team may use and Cordon has a key for."""
    config = services.config
    ctx.requested_alias = ctx.body.get("model")
    ctx.model_aliases = config["models"]["model_aliases"]
    if not isinstance(ctx.requested_alias, str) or ctx.requested_alias not in ctx.model_aliases:
        return _refuse(ctx, services, "model_not_allowed", f"unknown model: {ctx.requested_alias!r}")
    ctx.litellm_name = ctx.model_aliases[ctx.requested_alias]
    if not costing.has_known_price(ctx.litellm_name, config):
        return _refuse(ctx, services, "model_not_priced", f"model has no known price: {ctx.requested_alias!r}")

    ctx.provider = settings_module.Settings.model_provider(ctx.litellm_name)
    ctx.allowed_providers = set(ctx.tenant_settings.get("allowed_providers") or [])
    if ctx.provider not in ctx.allowed_providers:
        return _refuse(ctx, services, "provider_not_allowed", f"provider not allowed for this tenant: {ctx.provider!r}")
    if not services.settings.mock_mode and not services.settings.provider_has_key(ctx.provider):
        return _refuse(ctx, services, "provider_unavailable", f"provider not available: {ctx.provider!r}")
    ctx.served_alias = ctx.requested_alias
    ctx.served_litellm_name = ctx.litellm_name
    return None


def _refuse(ctx: RequestContext, services: Services, code: str, message: str, status: int = 400) -> JSONResponse:
    """Write a "blocked" ledger row and return the error - for checks that refuse in every mode."""
    ctx.block(services, code)
    return openai_error(status, message, "invalid_request_error", code=code)


# --- Section: adjustable limits ---
def check_limits(request: Request, ctx: RequestContext, services: Services) -> JSONResponse | None:
    """
    The adjustable limits: sizes, tags, tokens, rate limit, budget; then the holdout sample.

    A violated limit refuses the request in enforce mode; in observe mode it is
    only recorded (ctx.soft_violations). Returns an error response, or None.
    """
    error = _check_messages(ctx, services) or _check_sizes(request, ctx, services)
    if error is not None:
        return error
    worst_case_max_tokens = ctx.worst_case_max_tokens = _apply_max_tokens(ctx)
    # The budget is only reserved for a request that passed the rate limit.
    error = _check_rate_limit(ctx, services, worst_case_max_tokens) or _reserve_budget(ctx, services, worst_case_max_tokens)
    if error is not None:
        return error
    _choose_holdout(request, ctx, services)
    ctx.limits_passed = True  # passthrough.py may only ever skip what comes after this point
    return None


def _check_messages(ctx: RequestContext, services: Services) -> JSONResponse | None:
    """Always on: a non-empty messages array, and no image above the pixel limit (read from its header)."""
    messages = ctx.body.get("messages")
    if not isinstance(messages, list) or not messages:
        return openai_error(400, "messages must be a non-empty array", "invalid_request_error")
    ctx.messages = messages
    try:
        optimize.check_image_pixel_limits(messages, services.config["request_limits"]["max_image_pixels"])
    except optimize.ImageTooLargeError:
        return _refuse(ctx, services, "image_too_large", "image exceeds the pixel limit or could not be verified", status=413)
    return None


def _check_sizes(request: Request, ctx: RequestContext, services: Services) -> JSONResponse | None:
    """Message count, tags, input tokens and image count against the team's limits."""
    request_limits = services.config["request_limits"]
    tenant_settings = ctx.tenant_settings
    if len(ctx.messages) > request_limits["max_messages"]:
        if error := _over_limit(ctx, services, "too_many_messages", 413, "too many messages"):
            return error

    ctx.tags, tag_error = parse_tags(request, request_limits)
    if tag_error:
        if error := _over_limit(ctx, services, "invalid_tag", 400, tag_error):
            return error

    ctx.original_prompt_tokens = costing.count_message_tokens(ctx.litellm_name, ctx.messages)
    if ctx.original_prompt_tokens > limit(tenant_settings.get("max_input_tokens")):
        if error := _over_limit(ctx, services, "input_too_large", 413, "input exceeds max_input_tokens"):
            return error

    ctx.image_count = sum(1 for _ in optimize.iter_image_data_urls(ctx.messages))
    if ctx.image_count > limit(tenant_settings.get("max_images")):
        if error := _over_limit(ctx, services, "too_many_images", 413, "too many images"):
            return error
    return None


def _over_limit(ctx: RequestContext, services: Services, code: str, status: int, message: str) -> JSONResponse | None:
    """A tunable limit was exceeded: refuse in enforce mode, only record it in observe mode."""
    if ctx.observe:
        ctx.soft_violations.append(code)
        return None
    return _refuse(ctx, services, code, message, status=status)


def _apply_max_tokens(ctx: RequestContext) -> int:
    """Fill in the team's default max_tokens (`max_tokens` feature) and return the worst-case answer length."""
    default = ctx.tenant_settings.get("max_tokens_default", 1024)
    if "max_tokens" in ctx.features:
        ctx.body, defaulted = limits.apply_default_max_tokens(ctx.body, default)
        if defaulted:
            ctx.soft_violations.append("max_tokens_defaulted")  # informational; never counted as a saving
    return ctx.body.get("max_tokens") or ctx.body.get("max_completion_tokens") or default


def _choose_holdout(request: Request, ctx: RequestContext, services: Services) -> None:
    """Enforce mode only: put a share of requests (per session when X-Session-ID is sent) in the untouched holdout group."""
    salt = services.settings.hash_salt
    ctx.session_id_header = request.headers.get("x-session-id")
    ctx.session_hash = security.keyed_hash(ctx.tenant_id, ctx.session_id_header, salt) if ctx.session_id_header else None
    source = ctx.session_hash or security.keyed_hash(ctx.tenant_id, str(uuid.uuid4()), salt)
    ctx.holdout = (not ctx.observe) and is_in_holdout(source, ctx.tenant_settings.get("holdout_pct", 0.0) or 0.0)


def _check_rate_limit(ctx: RequestContext, services: Services, worst_case_max_tokens: int) -> JSONResponse | None:
    """Apply the per-key requests/tokens-per-minute limits. Returns an error response, or None to continue."""
    tokens_needed = ctx.original_prompt_tokens + worst_case_max_tokens
    try:
        rate_allowed, retry_after = services.rate_limiter.check_and_consume(
            ctx.key_prefix_value, ctx.tenant_settings.get("rpm", 60), ctx.tenant_settings.get("tpm", 60000), tokens_needed
        )
    except Exception:
        logger.warning("rate limiter failed (fail open)", exc_info=True)
        rate_allowed, retry_after = True, 0.0
    if rate_allowed:
        return None
    if not ctx.observe:
        ctx.block(services, "rate_limit_exceeded")
        return openai_error(
            429, "rate limit exceeded", "rate_limit_error", code="rate_limit_exceeded",
            extra_headers={"Retry-After": str(int(retry_after) + 1)},
        )
    ctx.soft_violations.append("rate_limited")
    return None


def _reserve_budget(ctx: RequestContext, services: Services, worst_case_max_tokens: int) -> JSONResponse | None:
    """
    Reserve the worst-case cost against the daily and monthly budgets.

    Reserved up front, so two requests arriving together can never both pass
    "still under budget?" and overshoot it. Over budget: observe mode only
    records it; enforce mode moves the request to a cheaper model when the
    monthly budget allows a downgrade, and otherwise refuses it (402).
    """
    tenant_settings = ctx.tenant_settings
    ctx.strong_alias = settings_module.resolve_tenant_strong(tenant_settings, services.config)
    caps = (limit(tenant_settings.get("budget_daily_eur")), limit(tenant_settings.get("budget_monthly_eur")))
    worst_case_eur = _worst_case_eur(ctx, services.config, worst_case_max_tokens)

    reason = _reserve(ctx, services, worst_case_eur, caps)
    if reason is None:
        return None
    if ctx.observe:  # observe never blocks and never changes the model
        ctx.soft_violations.append(reason)
        return None
    if reason == "monthly_budget" and tenant_settings.get("budget_action", "block") == "downgrade":
        if _downgrade(ctx, services, worst_case_max_tokens, caps):
            return None
    ctx.block(services, reason)
    message = "daily budget cap reached" if reason == "daily_cap" else "monthly budget reached"
    return openai_error(402, message, "insufficient_quota", code=reason)


def reserve_route(ctx: RequestContext, services: Services, route_model: str) -> bool:
    """
    A routed request may pay the route's model and then the strong model (its check escalated):
    reserve the route model's worst case on top of the strong model's. False when the budget cannot
    cover both - the request then stays on the strong model. A crashing tracker reserves nothing (fail open).
    """
    tenant_settings = ctx.tenant_settings
    caps = (limit(tenant_settings.get("budget_daily_eur")), limit(tenant_settings.get("budget_monthly_eur")))
    try:
        eur = costing.cost_eur(ctx.model_aliases[route_model], ctx.original_prompt_tokens, ctx.worst_case_max_tokens,
                               services.config)
        result = services.budget_tracker.reserve(ctx.tenant_id, eur, *caps)
    except Exception:
        logger.warning("route reservation failed (fail open)", exc_info=True)
        return True
    if result.allowed:
        ctx.reserved_eur += eur
    return result.allowed


def _worst_case_eur(ctx: RequestContext, config: dict, worst_case_max_tokens: int) -> float:
    """The cost of the dearer of the requested model and the strong model (an escalation target); 0 if unpriced."""
    try:
        return max(
            costing.cost_eur(name, ctx.original_prompt_tokens, worst_case_max_tokens, config)
            for name in (ctx.model_aliases[ctx.strong_alias], ctx.litellm_name)
        )
    except Exception:
        return 0.0


def _reserve(ctx: RequestContext, services: Services, eur: float, caps: tuple[float, float]) -> str | None:
    """
    Reserve eur against (daily, monthly) caps: None if allowed, else "daily_cap" or "monthly_budget".

    Sets ctx.reserved_eur to what was really reserved, so settle() releases
    exactly that. Fails open: a crashing tracker reserves nothing (0) - it must
    never release money other requests reserved.
    """
    try:
        result = services.budget_tracker.reserve(ctx.tenant_id, eur, *caps)
    except Exception:
        logger.warning("budget reservation failed (fail open)", exc_info=True)
        ctx.reserved_eur = 0.0
        return None
    if result.allowed:
        ctx.reserved_eur = eur
    return None if result.allowed else result.reason


def _downgrade(ctx: RequestContext, services: Services, worst_case_max_tokens: int, caps: tuple[float, float]) -> bool:
    """Move the request to the cheapest capable model if that still fits the budget. True if it did."""
    config = services.config
    settings_obj = services.settings
    enabled = set(config["models"]["providers"]) if settings_obj.mock_mode else set(settings_obj.provider_keys)
    alias = limits.cheapest_capable_model(
        settings_module.resolve_tenant_pool(ctx.tenant_settings, config), ctx.allowed_providers, enabled,
        ctx.model_aliases, ctx.required_capabilities(), settings_module.all_model_capabilities(config),
    )
    if alias is None:
        return False
    try:
        eur = costing.cost_eur(ctx.model_aliases[alias], ctx.original_prompt_tokens, worst_case_max_tokens, config)
    except Exception:
        return False
    if _reserve(ctx, services, eur, caps) is not None:
        return False
    ctx.served_alias, ctx.served_litellm_name = alias, ctx.model_aliases[alias]
    ctx.soft_violations.append("budget_downgraded")
    return True


def is_in_holdout(keyed_value: str, holdout_pct: float) -> bool:
    """Deterministically map a keyed hash to [0, 1) and compare it against holdout_pct."""
    if holdout_pct <= 0:
        return False
    fraction = (int(keyed_value[:8], 16) % 10_000) / 10_000.0
    return fraction < holdout_pct
