"""
Cost and savings in EUR.

- Actual cost: what the provider bills for the model that answered.
- Baseline cost: the requested model on the original, untrimmed prompt.
- Every saving is labelled measured or estimated (routing), and applied or
  not (a "would save" figure from observe mode).
- `has_known_price()`: a model without a price is refused, so no cost escapes budgets.
"""

import logging

import litellm

from src import messages as messages_module

logger = logging.getLogger("cordon.costing")

# --- Section: token counting — never crashes on unusual message shapes ---
def count_message_tokens(litellm_name: str, messages: list[dict]) -> int:
    """
    Count the prompt tokens of a message list.

    Parameters: litellm_name - the model (for its tokenizer); messages - the request messages.
    Returns: LiteLLM's count; if LiteLLM cannot count a message shape (it raises on
    valid OpenAI parts such as "refusal" or "input_audio"), the tokens of all text
    in the messages instead. Why: a crash here used to reject valid requests -
    or, for a fail-open tenant, skip every later check.
    """
    messages, image_tokens = _count_images_apart(litellm_name, messages)
    try:
        return litellm.token_counter(model=litellm_name, messages=messages) + image_tokens
    except Exception:
        text = "\n".join(messages_module.message_text(m.get("content")) for m in messages if isinstance(m, dict))
        return (litellm.token_counter(model=litellm_name, text=text) if text else 0) + image_tokens


def _count_images_apart(litellm_name: str, messages: list[dict]) -> tuple[list[dict], int]:
    """
    The messages without their attached (data-URL) images, and those images' tokens as this model is
    billed for them. LiteLLM counts every image with gpt-4o's rule; gpt-4o-mini is billed ~33x that.
    An image Cordon cannot read (a web link) stays in the messages for LiteLLM to count.
    """
    from src.savings import optimize

    kept, tokens = [], 0
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            kept.append(message)
            continue
        parts = []
        for part in content:
            size = _attached_image_size(part)
            if size is None:
                parts.append(part)
            else:
                tokens += optimize.image_tokens(litellm_name, *size, (part.get("image_url") or {}).get("detail", "high"))
        kept.append({**message, "content": parts})
    return kept, tokens


def _attached_image_size(part) -> tuple[int, int] | None:
    """(width, height) of an image part sent as a data URL, or None (not an image, or not readable)."""
    from src.savings import optimize

    if not isinstance(part, dict) or part.get("type") != "image_url":
        return None
    parsed = optimize.parse_data_url((part.get("image_url") or {}).get("url", ""))
    return optimize.read_image_dimensions(parsed[0]) if parsed else None


# --- Section: measured vs estimated — which techniques get which label ---
# Trim, vision downscaling, exact cache and semantic cache all compare a
# concrete "before" token/cost figure to a concrete "after" one, so their
# savings are measured. Routing's counterfactual ("what would the strong
# model have cost for this same request") is never actually called, so it is
# always estimated.
MEASURED_TECHNIQUES: frozenset[str] = frozenset({"trim", "vision", "cache", "semantic_cache"})
ESTIMATED_TECHNIQUES: frozenset[str] = frozenset({"routing"})


def savings_flag(technique: str) -> str:
    """
    Return "measured" or "estimated" for a savings technique name.

    Parameters: technique - one of MEASURED_TECHNIQUES | ESTIMATED_TECHNIQUES.
    Returns: "measured" or "estimated".
    Raises: ValueError for an unrecognized technique name.
    """
    if technique in MEASURED_TECHNIQUES:
        return "measured"
    if technique in ESTIMATED_TECHNIQUES:
        return "estimated"
    raise ValueError(f"unknown savings technique: {technique}")


def savings_entry(technique: str, eur: float, applied: bool = True) -> dict:
    """
    Build one ledger savings_json entry for a technique.

    Parameters: technique - the technique name.
                 eur - the EUR amount saved (or that would be saved) for this request.
                 applied - True when the saving really happened; False for a
                 "would save" figure (observe mode, holdout), where the
                 request was left unchanged.
    Returns: {"eur": eur, "flag": "measured"|"estimated", "applied": bool}.
    """
    return {"eur": eur, "flag": savings_flag(technique), "applied": applied}


# --- Section: currency — LiteLLM prices are USD, Cordon stores and displays EUR ---
def usd_to_eur(amount_usd: float, config: dict) -> float:
    """
    Convert a USD amount to EUR using config.yaml's fixed rate.

    Parameters: amount_usd - the USD amount.
                 config - the parsed config.yaml (reads money.usd_to_eur).
    Returns: the EUR amount.
    """
    return amount_usd * config["money"]["usd_to_eur"]


# --- Section: pricing — LiteLLM's local cost map, with config overrides ---
def has_known_price(litellm_name: str, config: dict) -> bool:
    """
    Check whether a model can be priced, either by LiteLLM or by a config override.

    Parameters: litellm_name - the LiteLLM model name (e.g. "openai/gpt-4o").
                 config - the parsed config.yaml (reads money.pricing_overrides).
    Returns: True if a price is available.
    Why: the gateway's model allowlist must reject any model without a known
    price (400), so a request can never bypass cost reservation and budgets
    simply because Cordon cannot price it. This calls litellm.cost_per_token()
    itself (instead of checking `litellm_name in litellm.model_cost`) because
    LiteLLM's local cost map keys OpenAI models without their provider prefix
    (e.g. "gpt-4o-mini", not "openai/gpt-4o-mini") but still resolves the
    prefixed form correctly at call time - a direct dict-membership check
    would wrongly report every openai/* model as unpriced.
    """
    if litellm_name in (config.get("money", {}).get("pricing_overrides") or {}):
        return True
    try:
        litellm.cost_per_token(model=litellm_name, prompt_tokens=1, completion_tokens=1)
        return True
    except Exception:
        return False


def cost_eur(litellm_name: str, prompt_tokens: int, completion_tokens: int, config: dict) -> float:
    """
    Compute the EUR cost of a call, given token counts.

    Parameters: litellm_name - the LiteLLM model name actually billed.
                 prompt_tokens - input token count.
                 completion_tokens - output token count.
                 config - the parsed config.yaml.
    Returns: the EUR cost (prompt + completion).
    Raises: ValueError if the model has no known price (callers should have
    already enforced has_known_price() at admission; this is a defensive
    fail-closed check, not the primary gate).
    """
    overrides = config.get("money", {}).get("pricing_overrides") or {}
    if litellm_name in overrides:
        rates = overrides[litellm_name]
        usd = prompt_tokens * rates["input_cost_per_token"] + completion_tokens * rates["output_cost_per_token"]
        return usd_to_eur(usd, config)

    try:
        prompt_cost_usd, completion_cost_usd = litellm.cost_per_token(
            model=litellm_name, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
        )
    except Exception as exc:
        raise ValueError(f"no known price for model: {litellm_name}") from exc
    return usd_to_eur(prompt_cost_usd + completion_cost_usd, config)


# --- Section: actual vs baseline vs net — the three numbers every request reports ---
def actual_cost_eur(litellm_name: str, prompt_tokens: int, completion_tokens: int, config: dict) -> float:
    """
    The real cost of the call Cordon actually made (served model, served tokens).

    Parameters: see cost_eur.
    Returns: EUR cost, including any discarded calls on fallback (callers
    sum actual_cost_eur() across every call made for a request, per principle 2).
    """
    return cost_eur(litellm_name, prompt_tokens, completion_tokens, config)


def baseline_cost_eur(
    requested_litellm_name: str, original_prompt_tokens: int, served_completion_tokens: int, config: dict
) -> float:
    """
    The cost Cordon is compared against: the originally requested model, the
    original (untrimmed, unmasked-token-count) prompt, and no cache.

    Parameters: requested_litellm_name - the model the client asked for.
                 original_prompt_tokens - token count of the prompt before
                 any Cordon optimization.
                 served_completion_tokens - the output token count of the
                 answer actually served (baseline output tokens equal served
                 output tokens: the baseline is never
                 re-generated, only re-priced).
                 config - the parsed config.yaml.
    Returns: EUR cost.
    """
    return cost_eur(requested_litellm_name, original_prompt_tokens, served_completion_tokens, config)


def net_savings_eur(baseline_eur: float, actual_eur: float) -> float:
    """
    Net savings for one request: baseline minus actual.

    Parameters: baseline_eur, actual_eur.
    Returns: baseline_eur - actual_eur (negative if Cordon's overhead made the
    request more expensive than the baseline, e.g. a discarded fallback call).
    """
    return baseline_eur - actual_eur
