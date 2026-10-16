"""
src/savings/costing.py: EUR conversion, known prices, actual/baseline/net cost, measured vs estimated.

Run: .venv/bin/python -m pytest tests/test_costing.py -v
"""

import pytest

from src.savings import costing

from helpers import CONFIG


def test_usd_to_eur_uses_configured_rate():
    """Usd to eur uses configured rate."""
    assert costing.usd_to_eur(1.0, CONFIG) == CONFIG["money"]["usd_to_eur"]
    assert costing.usd_to_eur(0.0, CONFIG) == 0.0


def test_has_known_price_true_for_pool_models():
    """Has known price true for pool models."""
    for alias, litellm_name in CONFIG["models"]["model_aliases"].items():
        assert costing.has_known_price(litellm_name, CONFIG), f"{alias} ({litellm_name}) should have a known price"


def test_has_known_price_false_for_unknown_model():
    """Has known price false for unknown model."""
    assert not costing.has_known_price("openai/this-model-does-not-exist-xyz", CONFIG)


def test_pricing_override_is_used_when_present():
    """Pricing override is used when present."""
    config = dict(CONFIG)
    config = {**CONFIG, "money": {**CONFIG["money"], "pricing_overrides": {"custom/model": {"input_cost_per_token": 1e-6, "output_cost_per_token": 2e-6}}}}
    assert costing.has_known_price("custom/model", config)
    eur = costing.cost_eur("custom/model", 1000, 500, config)
    expected_usd = 1000 * 1e-6 + 500 * 2e-6
    assert eur == pytest.approx(expected_usd * config["money"]["usd_to_eur"])


def test_actual_cost_scales_with_tokens():
    """Actual cost scales with tokens."""
    small = costing.actual_cost_eur("openai/gpt-4o-mini", 100, 50, CONFIG)
    large = costing.actual_cost_eur("openai/gpt-4o-mini", 1000, 500, CONFIG)
    assert large > small
    assert small >= 0


def test_baseline_cost_uses_original_prompt_and_served_output_tokens():
    """Baseline cost uses original prompt and served output tokens."""
    baseline = costing.baseline_cost_eur("openai/gpt-4o", 1000, 200, CONFIG)
    actual = costing.actual_cost_eur("openai/gpt-4o-mini", 1000, 200, CONFIG)
    # Same token counts, but gpt-4o is more expensive than gpt-4o-mini.
    assert baseline > actual


def test_net_savings_is_baseline_minus_actual():
    """Net savings is baseline minus actual."""
    assert costing.net_savings_eur(1.0, 0.4) == pytest.approx(0.6)
    assert costing.net_savings_eur(0.4, 1.0) == pytest.approx(-0.6)


@pytest.mark.parametrize("technique", ["trim", "vision", "cache", "semantic_cache"])
def test_measured_techniques_are_flagged_measured(technique):
    """Measured techniques are flagged measured."""
    assert costing.savings_flag(technique) == "measured"
    entry = costing.savings_entry(technique, 0.01)
    assert entry == {"eur": 0.01, "flag": "measured", "applied": True}


def test_would_save_entries_are_marked_not_applied():
    """An observe-mode "would save" figure is kept apart from real savings."""
    assert costing.savings_entry("cache", 0.01, applied=False)["applied"] is False


def test_routing_is_always_estimated():
    """Routing is always estimated."""
    assert costing.savings_flag("routing") == "estimated"
    entry = costing.savings_entry("routing", 0.02)
    assert entry["flag"] == "estimated"


def test_unknown_technique_raises():
    """Unknown technique raises."""
    with pytest.raises(ValueError):
        costing.savings_flag("not-a-real-technique")
