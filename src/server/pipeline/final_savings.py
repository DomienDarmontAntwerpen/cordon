"""
A request's savings for its ledger row: what steps 4 and 7 recorded, or a cache
hit (which replaces the whole request), and in observe mode what the cache or
routing *would* have saved (marked not applied, never counted as a real saving).
The entries never add up to more than the baseline cost.
"""

from src.savings import costing
from src.server.pipeline.context import RequestContext

CACHE_TECHNIQUE = {"exact": "cache", "semantic": "semantic_cache"}


def final_savings(ctx: RequestContext, config: dict, baseline_eur: float, tokens_out: int) -> dict:
    """The request's savings entries. A (would-be) cache hit replaces the whole request, so it is the only entry."""
    if ctx.cache_hit_text is not None and ctx.cache_type_on_hit:
        return _cache_entry(ctx.cache_type_on_hit, baseline_eur, applied=True)
    if ctx.would_hit_cache:
        return _cache_entry(ctx.would_hit_cache, baseline_eur, applied=False)
    savings = dict(ctx.savings)
    if ctx.would_route:
        # Routing would save a share of what is left after the other savings, so
        # the entries together can never add up to more than the baseline.
        left = max(0.0, baseline_eur - sum(entry["eur"] for entry in savings.values()))
        saving = left * _routing_share(ctx, config, tokens_out)
        if saving > 0:
            savings["routing"] = costing.savings_entry("routing", saving, applied=False)
    return savings


def _cache_entry(hit_type: str, baseline_eur: float, applied: bool) -> dict:
    """A cache hit saves the whole baseline cost - nothing else ran, so nothing else saved anything."""
    technique = CACHE_TECHNIQUE[hit_type]
    return {technique: costing.savings_entry(technique, baseline_eur, applied=applied)}


def _routing_share(ctx: RequestContext, config: dict, tokens_out: int) -> float:
    """
    The share of a call's cost the route would save against the strong model (0 to 1), for this
    request's tokens: the cheaper model always answers, and the share its check sent to the strong
    model (measured when the route was proven) costs the strong model on top.
    """
    strong = costing.cost_eur(ctx.model_aliases[ctx.strong_alias], ctx.original_prompt_tokens, tokens_out, config)
    cheaper = costing.cost_eur(ctx.model_aliases[ctx.would_route.model], ctx.original_prompt_tokens, tokens_out, config)
    return max(0.0, 1.0 - cheaper / strong - ctx.would_route.escalated) if strong > 0 else 0.0
