"""
`RequestContext`: everything Cordon knows about one request, filled in step by step.
Fields are grouped by the step that sets them. `book()` is the one way a
request's accounting ends: budget settled and one ledger row, exactly once.
"""

import time
from dataclasses import dataclass, field
from typing import Any

from src.server.ledger import write_block_ledger, write_ledger
from src.server.services import Services


SERVED_MODEL = object()  # book(): "use the model that answered" (None means: no model answered)


@dataclass
class RequestContext:
    """The state of one request as it moves through the pipeline. See the module docstring."""

    # --- Set before step 1 ---
    start: float
    client_ip: str
    mock_fail_step: str | None = None  # X-Mock-Fail-Step (mock mode only): forces a step to crash
    mock_latency_s: float = 0.0  # X-Mock-Latency-Ms (mock mode only): a provider call's waiting time, for load tests

    # --- Step 1, check_access(): who is calling, and for which model ---
    tenant_id: str = ""
    tenant_settings: dict = field(default_factory=dict)
    key_prefix_value: str = ""
    observe: bool = True
    features: set = field(default_factory=set)
    original_body: dict = field(default_factory=dict)  # as sent; used by the passthrough fallback
    body: dict = field(default_factory=dict)  # may get a default max_tokens in check_limits()
    requested_alias: str = ""
    litellm_name: str = ""
    provider: str = ""
    model_aliases: dict = field(default_factory=dict)
    allowed_providers: set = field(default_factory=set)

    # --- Step 1, check_limits(): sizes, rate limit, budget, holdout ---
    messages: list = field(default_factory=list)
    tags: list = field(default_factory=list)
    soft_violations: list = field(default_factory=list)  # observe mode: what would have been blocked
    original_prompt_tokens: int = 0
    image_count: int = 0
    strong_alias: str = ""
    served_alias: str = ""
    served_litellm_name: str = ""
    reserved_eur: float = 0.0
    worst_case_max_tokens: int = 0  # step 1: the answer length the reservation is priced on
    session_id_header: str | None = None
    session_hash: str | None = None
    holdout: bool = False

    # --- Step 2, input protection ---
    masked_messages: list = field(default_factory=list)
    own_data: set = field(default_factory=set)
    placeholder_mapping: dict = field(default_factory=dict)
    pii_counts: dict = field(default_factory=dict)
    guardrail_actions: dict = field(default_factory=dict)

    # --- Step 4, optimize ---
    request_category: str = "chat"
    request_difficulty: str = "easy"
    effective_messages: list = field(default_factory=list)
    savings: dict = field(default_factory=dict)

    # --- Step 5, cache ---
    last_user_text_masked: str = ""
    system_prompt_text: str | None = None
    cache_eligible: bool = False
    cache_key: str | None = None
    context_hash: str | None = None
    cache_hit_text: str | None = None
    cache_type_on_hit: str | None = None
    would_hit_cache: str | None = None  # observe mode: "exact"|"semantic" if the cache would have answered


    # --- Step 6, routing ---
    ladder: list = field(default_factory=list)
    route: Any = None  # routing.Route taken (enforce): its model answers first, its check (step 8) decides
    confidence_models: frozenset = frozenset()  # aliases asked for logprobs: the route's check reads the model's certainty
    would_route: Any = None  # observe mode: the routing.Route that would have been taken
    completion_kwargs: dict = field(default_factory=dict)
    mock_bad_response: bool = False
    mock_leak: str | None = None

    # --- Steps 7-9 (non-streaming only) ---
    usage: Any = None
    answer_text: str = ""
    answer_tool_calls: Any = None
    answer_confidence: dict | None = None  # quality.answer_confidence() of the answer (None without logprobs)
    tokens_in: int | None = None
    tokens_out: int | None = None
    provider_ms: float = 0.0
    total_actual_eur: float = 0.0
    fallback_occurred: bool = False
    quality_score: float | None = None
    quality_reason: str | None = None  # why the check kept or sent on the answer (quality.GateVerdict.reason)
    processed_text: str = ""
    restored_message: dict = field(default_factory=dict)

    # --- Bookkeeping flags ---
    limits_passed: bool = False  # set at the end of step1_admission.check_limits()
    budget_settled: bool = False  # the reservation is settled exactly once (see settle())
    booked: bool = False  # the ledger row is written exactly once (see book())
    stream_owns_capacity_slot: bool = False  # a stream releases its capacity slot when it ends

    def cost_opts_apply(self) -> bool:
        """True when cost optimizations may change the request: enforce mode and not in the holdout sample."""
        return (not self.observe) and (not self.holdout)

    def gate_applies(self) -> bool:
        """True when a route was taken: its cheaper model's answer is checked (step 8) and may go to the strong model."""
        return self.route is not None

    def required_capabilities(self) -> set[str]:
        """The capabilities ({"vision","json_schema","tools"}) this request needs from a model."""
        caps = set()
        if self.image_count > 0:
            caps.add("vision")
        if self.body.get("response_format"):
            caps.add("json_schema")
        if self.body.get("tools"):
            caps.add("tools")
        return caps

    def settle(self, services: Services, actual_eur: float) -> None:
        """Release this request's budget reservation and book its real cost - at most once."""
        if self.budget_settled:
            return
        self.budget_settled = True
        services.budget_tracker.settle(self.tenant_id, self.reserved_eur, actual_eur)

    def book(
        self,
        services: Services,
        *,
        status: str,
        error_code: str | None = None,
        error_detail: str | None = None,
        actual_eur: float = 0.0,
        baseline_eur: float | None = None,
        tokens_out: int | None = None,
        served_alias: str | None | object = SERVED_MODEL,
        savings: dict | None = None,
    ) -> None:
        """
        End this request's accounting: settle the budget and write its one ledger row.

        Every way a request can end after admission - answered, provider failure,
        blocked answer, client disconnect, internal crash, passthrough - goes
        through here, so each request is booked exactly once (a second call does
        nothing). served_alias defaults to the model that answered; pass None
        when no model answered.
        """
        if self.booked:
            return
        self.booked = True
        self.settle(services, actual_eur)
        overhead_ms = (time.perf_counter() - self.start) * 1000 - self.provider_ms
        write_ledger(
            services.store, self.tenant_id, self.key_prefix_value, self.tags, self.tenant_settings, self.requested_alias,
            self.served_alias if served_alias is SERVED_MODEL else served_alias, self.provider,
            self.tokens_in if self.tokens_in is not None else self.original_prompt_tokens, tokens_out,
            actual_eur, baseline_eur, overhead_ms, self.provider_ms, status, error_code, error_detail,
            session_hash=self.session_hash, holdout=self.holdout, savings=savings,
            cache_type=self.cache_type_on_hit, guardrail_actions=self.guardrail_actions,
        )

    def block(
        self, services: Services, error_code: str, session_hash: str | None = None, guardrail_actions: dict | None = None
    ) -> None:
        """Write a zero-cost "blocked" ledger row for this request (see ledger.write_block_ledger)."""
        write_block_ledger(
            services.store, self.tenant_id, self.key_prefix_value, self.tags, self.tenant_settings,
            self.requested_alias, error_code, session_hash=session_hash, guardrail_actions=guardrail_actions,
        )
