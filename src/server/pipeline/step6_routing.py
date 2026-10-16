"""
Step 6: choose the model ladder and build the model call.
Routing puts a proven cheaper model first (its answer is checked in step 8), and
never a model pricier than the strong one. In observe mode the route is only recorded.
"""

import logging

from fastapi import Request
from fastapi.responses import JSONResponse

from src.protection import security
from src.savings import routing
from src.server.pipeline import mock, step1_admission
from src.server.pipeline.context import RequestContext
from src.server.services import Services

logger = logging.getLogger("cordon.pipeline")


def run(request: Request, ctx: RequestContext, services: Services) -> JSONResponse | None:
    """Choose the model ladder and build the completion call. Never returns an error."""
    settings_obj = services.settings
    ctx.ladder = [ctx.served_alias]
    if _routing_may_run(ctx):
        _choose_route(ctx, services)

    ctx.completion_kwargs = {k: v for k, v in ctx.body.items() if k in security.ALLOWED_REQUEST_PARAMS}
    ctx.completion_kwargs["model"] = ctx.served_litellm_name
    ctx.completion_kwargs["messages"] = ctx.effective_messages
    ctx.completion_kwargs["timeout"] = services.config["provider"]["timeout_seconds"]
    # The route's check reads the cheaper model's certainty: ask that model (only) for it.
    if ctx.route and ctx.route.gate_threshold > 0:
        ctx.confidence_models = frozenset([ctx.route.model])

    if settings_obj.mock_mode:
        ctx.mock_bad_response = request.headers.get("x-mock-bad-response", "").lower() == "true"
        ctx.mock_leak = request.headers.get("x-mock-leak")
        ctx.completion_kwargs["mock_response"] = mock.mock_answer_text(ctx)
    return None


def _choose_route(ctx: RequestContext, services: Services) -> None:
    """Take the proven route for this kind of request (enforce), or only record it (observe)."""
    settings_obj = services.settings
    candidates, _strong = routing.candidate_models(
        ctx.tenant_settings, services.config, ctx.required_capabilities(), settings_obj.provider_keys, settings_obj.mock_mode,
    )
    checkable = frozenset(services.config["models"].get("logprobs_models") or [])
    uncheckable = bool(ctx.body.get("stream") or ctx.body.get("tools"))
    choice = (ctx.request_category, ctx.request_difficulty, candidates, ctx.strong_alias, checkable)
    ladder, route = routing.choose_route(*choice, answer_uncheckable=uncheckable)
    if route and ctx.cost_opts_apply() and not step1_admission.reserve_route(ctx, services, route.model):
        ladder, route = routing.choose_route(*choice, answer_uncheckable=True)  # the budget covers the strong model only
    logger.info("routing decision: category=%s difficulty=%s ladder=%s", ctx.request_category, ctx.request_difficulty, ladder)
    if ctx.cost_opts_apply():
        ctx.ladder, ctx.route = ladder, route
        ctx.served_alias = ladder[0]
        ctx.served_litellm_name = ctx.model_aliases[ctx.served_alias]
    elif ctx.observe:
        ctx.would_route = route


def _routing_may_run(ctx: RequestContext) -> bool:
    """True when routing may pick the model: the caller asked for the strong model, no downgrade, no cache hit, not in holdout."""
    return (
        ctx.cache_hit_text is None
        and "routing" in ctx.features
        and not ctx.holdout
        and ctx.served_alias == ctx.requested_alias
        and ctx.requested_alias == ctx.strong_alias
    )
