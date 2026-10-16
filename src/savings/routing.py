"""
Routing: send a request to the cheapest model proven good enough.

- `classify_request()`: category and difficulty, from simple explainable rules.
- `candidate_models()`: enabled, allowed, capable models never pricier than `strong`.
- `choose_route()`: a cheaper model first only where `routing_table.yaml` holds
  benchmark evidence for it *together with its quality check* (the route's gate
  threshold); otherwise the strong model first.
"""

import logging
import re
from dataclasses import dataclass

import yaml

logger = logging.getLogger("cordon.routing")

from src import messages as messages_module
from src.savings import costing
from src.paths import ROUTING_TABLE_PATH as _ROUTING_TABLE_PATH


@dataclass(frozen=True)
class Route:
    """A proven route: the cheaper model, the gate threshold it was proven with, and the share it sent on."""

    model: str
    gate_threshold: float  # 0 = only the rules check its answers
    escalated: float  # measured share of its answers the check sent to the strong model


def _load_routing_table() -> dict:
    """Load routing_table.yaml once at import time: {category: {difficulty: Route}} (possibly empty)."""
    with open(_ROUTING_TABLE_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return {category: {difficulty: Route(**entry) for difficulty, entry in by_difficulty.items()}
            for category, by_difficulty in (data.get("categories") or {}).items()}


ROUTING_TABLE: dict = _load_routing_table()

CATEGORIES: tuple[str, ...] = ("extraction", "classification", "summarization", "code", "reasoning", "chat")

# --- Section: request classification — heuristic, explainable, fast ---
_CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "extraction": ("extract", "pull out", "parse the", "retrieve the", "find the following", "fields:"),
    "classification": ("classify", "categorize", "which category", "label this", "is this spam"),
    "summarization": ("summarize", "summary", "tl;dr", "condense", "shorten this"),
    "code": ("write a function", "debug this", "fix this code", "refactor this"),
    "reasoning": ("solve", "prove that", "step by step", "calculate the", "how many"),
}

_MULTISTEP_WORDS = ("first,", "then,", "next,", "finally,", "step 1", "step 2", "after that")
_CONSTRAINT_WORDS = ("must", "should", "ensure", "require", "need to")
_CODE_BLOCK_REGEX = re.compile(r"```")
# "Write a Python function...", "implement a class...": a request to write code,
# even without a code block (the plain keywords missed "write a Python function").
_CODE_REQUEST_REGEX = re.compile(r"\b(write|implement|create|build)\b[^.?!]{0,40}\b(function|method|script|program)\b")

_HARD_WORD_COUNT_THRESHOLD = 300
_HARD_MULTISTEP_COUNT = 2
_HARD_CONSTRAINT_COUNT = 3
_HARD_CODE_WORD_COUNT = 100


def classify_request(messages: list[dict], response_format: dict | None) -> tuple[str, str]:
    """
    Classify a request's category and difficulty.

    Parameters: messages - the request's messages (masked text is fine and
    preferred - classification never needs the real values).
                 response_format - the request's response_format, if any.
    Returns: (category, difficulty). category is one of CATEGORIES;
    difficulty is "easy" or "hard".
    Why keyword + structural signals: this must run in well under 5ms and
    stay explainable, which rules out a second model call.
    """
    # Only the user's question decides: a system prompt that says "summarize" or
    # "you must..." would otherwise turn every request into summarization/hard.
    text = messages_module.last_user_text(messages).lower()
    has_code_block = bool(_CODE_BLOCK_REGEX.search(text))

    category = "chat"
    if response_format and response_format.get("type") in ("json_schema", "json_object"):
        category = "extraction"
    for candidate_category, keywords in _CATEGORY_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            category = candidate_category
            break
    if has_code_block or _CODE_REQUEST_REGEX.search(text):
        category = "code"

    word_count = len(text.split())
    multistep_count = sum(1 for word in _MULTISTEP_WORDS if word in text)
    constraint_count = sum(1 for word in _CONSTRAINT_WORDS if word in text)

    difficulty = "easy"
    if word_count > _HARD_WORD_COUNT_THRESHOLD:
        difficulty = "hard"
    elif multistep_count >= _HARD_MULTISTEP_COUNT:
        difficulty = "hard"
    elif constraint_count >= _HARD_CONSTRAINT_COUNT:
        difficulty = "hard"
    elif category == "code" and word_count > _HARD_CODE_WORD_COUNT:
        difficulty = "hard"
    elif category == "reasoning" and multistep_count >= 1:
        difficulty = "hard"

    return category, difficulty


# --- Section: candidate selection — enabled, allowed, capable pool models, plus strong ---
def candidate_models(
    tenant_settings: dict,
    config: dict,
    required_capabilities: set[str],
    provider_keys: dict[str, str],
    mock_mode: bool,
) -> tuple[list[str], str]:
    """
    Resolve the candidate model ladder's membership for one request.

    Parameters: tenant_settings - the tenant's settings dict.
                 config - the parsed config.yaml.
                 required_capabilities - capability names the request needs
                 (e.g. {"vision"} if it includes an image).
                 provider_keys - settings.Settings.provider_keys (providers
                 with a configured API key).
                 mock_mode - True in mock mode, where every provider counts as enabled.
    Returns: (candidates, strong_alias). candidates preserves the pool's
    price order and never includes a model more expensive than the strong
    one: routing only ever goes cheaper, so the budget reservation (priced on
    the strong or requested model) always covers whichever candidate answers.
    strong_alias is always appended if not already present, regardless of its
    own capability fit, since it is also the trusted fallback.
    """
    from src.server import settings as settings_module

    pool = settings_module.resolve_tenant_pool(tenant_settings, config)
    strong_alias = settings_module.resolve_tenant_strong(tenant_settings, config)
    allowed_providers = set(tenant_settings.get("allowed_providers") or [])
    model_aliases = config["models"]["model_aliases"]
    capabilities_by_alias = settings_module.all_model_capabilities(config)

    strong_price = _unit_price(model_aliases[strong_alias], config)
    candidates = []
    for alias in pool:
        litellm_name = model_aliases.get(alias)
        if litellm_name is None or _unit_price(litellm_name, config) > strong_price:
            continue
        provider = litellm_name.split("/", 1)[0]
        if provider not in allowed_providers:
            continue
        if not mock_mode and provider not in provider_keys:
            continue
        capabilities = capabilities_by_alias.get(alias, {})
        if not required_capabilities.issubset({name for name, supported in capabilities.items() if supported}):
            continue
        candidates.append(alias)

    if strong_alias not in candidates:
        candidates.append(strong_alias)

    return candidates, strong_alias


def _unit_price(litellm_name: str, config: dict) -> float:
    """A model's price for 1,000 input plus 1,000 output tokens, in EUR - for comparing models."""
    return costing.cost_eur(litellm_name, 1000, 1000, config)


# --- Section: route selection — routing_table.yaml lookup ---
def choose_route(category: str, difficulty: str, candidates: list[str], strong_alias: str,
                 checkable: frozenset, answer_uncheckable: bool = False) -> tuple[list[str], Route | None]:
    """
    Choose the ordered ladder of models to try for one request, and the route it follows.

    Parameters: category, difficulty - from classify_request(). candidates - from
    candidate_models(). strong_alias - the tenant's strong model. checkable - the models whose
    answers the gate can check (they return their confidence); a route proven with a classifier
    threshold is only taken for those. answer_uncheckable - a stream (cannot be taken back once
    sent) or a request with tools (a tool call has no text to check): never routed, because a
    route was proven together with its check.
    Returns: (ladder, route). With a route: its model, then the strong model - nothing unproven
    and unchecked after them. Without one (no evidence, the model not allowed, or its check
    impossible): the strong model first, the other candidates behind it as failover.
    """
    route = ROUTING_TABLE.get(category, {}).get(difficulty)
    if route and (answer_uncheckable or route.model not in candidates or route.model == strong_alias
                  or (route.gate_threshold > 0 and route.model not in checkable)):
        route = None
    if route:
        return [route.model, strong_alias], route
    return [strong_alias] + [alias for alias in candidates if alias != strong_alias], None
