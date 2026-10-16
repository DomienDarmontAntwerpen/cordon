"""
Limits: rate limits, budgets and the agent-loop breaker. In memory, one process.

- `RateLimiter`: requests and tokens per minute, per API key.
- `BudgetTracker`: reserves a request's worst-case cost before the model call
  and settles it after, so parallel requests can never overshoot a cap.
- `CircuitBreaker`: flags an agent repeating near-identical messages.
- `cheapest_capable_model()`, `apply_default_max_tokens()`: small helpers.
"""

import hashlib
import json
import logging
import calendar
import time
from dataclasses import dataclass, field
from threading import Lock

from src import messages as messages_module
from src.savings import quality

logger = logging.getLogger("cordon.limits")


# --- Section: rate limiting — two token buckets per API key ---
@dataclass
class _TokenBucket:
    """A continuously-refilling token bucket. capacity/refill_per_second define the rate; tokens is the current level."""

    capacity: float
    refill_per_second: float
    tokens: float
    last_refill: float


class RateLimiter:
    """Per-API-key request and token rate limiting, in-memory, per-process."""

    def __init__(self) -> None:
        """Parameters: none. Buckets are created lazily per key_prefix on first use."""
        self._buckets: dict[str, dict[str, _TokenBucket]] = {}
        self._lock = Lock()

    def check_and_consume(self, key_prefix: str, rpm: int, tpm: int, tokens_needed: int) -> tuple[bool, float]:
        """
        Check and, if allowed, consume one request and tokens_needed tokens.

        Parameters: key_prefix - the caller's API key prefix (the rate-limit
        identity; never the full key).
                     rpm, tpm - the tenant's configured per-key limits.
                     tokens_needed - input tokens + max_tokens for this
                     request, counted before the call.
        Returns: (allowed, retry_after_seconds). If not allowed, retry_after_seconds
        is how long until the limiting bucket would have enough capacity.
        """
        now = time.monotonic()
        with self._lock:
            buckets = self._buckets.setdefault(key_prefix, {})
            request_bucket = buckets.get("requests")
            if request_bucket is None:
                request_bucket = _TokenBucket(capacity=rpm, refill_per_second=rpm / 60.0, tokens=rpm, last_refill=now)
                buckets["requests"] = request_bucket
            token_bucket = buckets.get("tokens")
            if token_bucket is None:
                token_bucket = _TokenBucket(capacity=tpm, refill_per_second=tpm / 60.0, tokens=tpm, last_refill=now)
                buckets["tokens"] = token_bucket

            # Follow the tenant's current limits: a PATCH of rpm/tpm applies on the next request.
            for bucket, limit in ((request_bucket, rpm), (token_bucket, tpm)):
                if bucket.capacity != limit:
                    bucket.capacity = limit
                    bucket.refill_per_second = limit / 60.0
                    bucket.tokens = min(bucket.tokens, limit)

            for bucket in (request_bucket, token_bucket):
                elapsed = now - bucket.last_refill
                bucket.tokens = min(bucket.capacity, bucket.tokens + elapsed * bucket.refill_per_second)
                bucket.last_refill = now

            if request_bucket.tokens < 1:
                retry_after = (1 - request_bucket.tokens) / request_bucket.refill_per_second if request_bucket.refill_per_second > 0 else 60.0
                return False, retry_after
            if token_bucket.tokens < tokens_needed:
                deficit = tokens_needed - token_bucket.tokens
                retry_after = deficit / token_bucket.refill_per_second if token_bucket.refill_per_second > 0 else 60.0
                return False, retry_after

            request_bucket.tokens -= 1
            token_bucket.tokens -= tokens_needed
            return True, 0.0


# --- Section: budget — cost reservation against daily and monthly caps ---
@dataclass
class ReservationResult:
    """Outcome of BudgetTracker.reserve(). allowed - whether the reservation was made. reason - "daily_cap"|"monthly_budget" if not."""

    allowed: bool
    reason: str | None
    daily_used_eur: float
    monthly_used_eur: float


class BudgetTracker:
    """
    Per-tenant daily/monthly EUR budget tracking with up-front cost
    reservation, so concurrent requests cannot jointly overshoot a cap.
    In-memory, per-process (see module docstring).
    """

    def __init__(self) -> None:
        """Parameters: none."""
        self._lock = Lock()
        self._daily_reserved: dict[tuple[str, str], float] = {}
        self._daily_spent: dict[tuple[str, str], float] = {}
        self._monthly_reserved: dict[tuple[str, str], float] = {}
        self._monthly_spent: dict[tuple[str, str], float] = {}

    @staticmethod
    def _day_key() -> str:
        """Current UTC day as YYYY-MM-DD."""
        return time.strftime("%Y-%m-%d", time.gmtime())

    @staticmethod
    def _month_key() -> str:
        """Current UTC month as YYYY-MM."""
        return time.strftime("%Y-%m", time.gmtime())

    def reserve(self, tenant_id: str, worst_case_eur: float, budget_daily_eur: float, budget_monthly_eur: float) -> ReservationResult:
        """
        Reserve a request's worst-case cost against the daily and monthly budgets.

        Parameters: tenant_id - the tenant. worst_case_eur - the reservation
        amount (input tokens + max_tokens, priced on the most expensive
        candidate model - computed by the caller). budget_daily_eur,
        budget_monthly_eur - the tenant's caps.
        Returns: a ReservationResult. The daily cap is always checked first
        and always blocks when it would be exceeded ("the
        daily hard cap always blocks, even when the monthly action is
        downgrade" - the monthly budget_action only matters for the
        monthly_budget reason, never for daily_cap).
        """
        with self._lock:
            day_key = (tenant_id, self._day_key())
            month_key = (tenant_id, self._month_key())
            daily_used = self._daily_spent.get(day_key, 0.0) + self._daily_reserved.get(day_key, 0.0)
            monthly_used = self._monthly_spent.get(month_key, 0.0) + self._monthly_reserved.get(month_key, 0.0)

            if daily_used + worst_case_eur > budget_daily_eur:
                return ReservationResult(False, "daily_cap", daily_used, monthly_used)
            if monthly_used + worst_case_eur > budget_monthly_eur:
                return ReservationResult(False, "monthly_budget", daily_used, monthly_used)

            self._daily_reserved[day_key] = self._daily_reserved.get(day_key, 0.0) + worst_case_eur
            self._monthly_reserved[month_key] = self._monthly_reserved.get(month_key, 0.0) + worst_case_eur
            return ReservationResult(True, None, daily_used, monthly_used)

    def settle(self, tenant_id: str, reserved_eur: float, actual_eur: float) -> None:
        """
        Release a reservation and record the actual spend.

        Parameters: tenant_id - the tenant. reserved_eur - the amount
        originally reserved for this request. actual_eur - what it actually
        cost (may be less, e.g. the answer used fewer tokens than max_tokens;
        or, on a discarded fallback call, the sum of every call made).
        Returns: None.
        """
        with self._lock:
            day_key = (tenant_id, self._day_key())
            month_key = (tenant_id, self._month_key())
            self._daily_reserved[day_key] = max(0.0, self._daily_reserved.get(day_key, 0.0) - reserved_eur)
            self._monthly_reserved[month_key] = max(0.0, self._monthly_reserved.get(month_key, 0.0) - reserved_eur)
            self._daily_spent[day_key] = self._daily_spent.get(day_key, 0.0) + actual_eur
            self._monthly_spent[month_key] = self._monthly_spent.get(month_key, 0.0) + actual_eur

    def restore_spent(self, daily_by_tenant: dict[str, float], monthly_by_tenant: dict[str, float]) -> None:
        """
        Set today's and this month's spend per tenant, e.g. from the ledger at startup.

        Why: the tracker lives in memory, so without this every restart would
        reset each tenant's daily and monthly spend to zero.
        """
        with self._lock:
            day, month = self._day_key(), self._month_key()
            for tenant_id, eur in daily_by_tenant.items():
                self._daily_spent[(tenant_id, day)] = eur
            for tenant_id, eur in monthly_by_tenant.items():
                self._monthly_spent[(tenant_id, month)] = eur

    @staticmethod
    def period_starts() -> tuple[float, float]:
        """Return the Unix timestamps of the start of today and of this month, both UTC."""
        now = time.gmtime()
        day_start = calendar.timegm((now.tm_year, now.tm_mon, now.tm_mday, 0, 0, 0))
        month_start = calendar.timegm((now.tm_year, now.tm_mon, 1, 0, 0, 0))
        return day_start, month_start

    def usage(self, tenant_id: str) -> tuple[float, float]:
        """Return (daily_used_eur, monthly_used_eur), each including in-flight reservations, for alerts/the dashboard."""
        with self._lock:
            day_key = (tenant_id, self._day_key())
            month_key = (tenant_id, self._month_key())
            daily = self._daily_spent.get(day_key, 0.0) + self._daily_reserved.get(day_key, 0.0)
            monthly = self._monthly_spent.get(month_key, 0.0) + self._monthly_reserved.get(month_key, 0.0)
            return daily, monthly


def cheapest_capable_model(
    pool: list[str],
    allowed_providers: set[str],
    enabled_providers: set[str],
    model_aliases: dict[str, str],
    required_capabilities: set[str],
    capabilities_by_alias: dict[str, dict[str, bool]],
) -> str | None:
    """
    Pick the cheapest pool model usable for a budget "downgrade".

    Parameters: pool - the tenant's model pool, cheapest-first (config.yaml's
    ordering convention).
                 allowed_providers - the tenant's allowed_providers.
                 enabled_providers - providers with an API key configured
                 (or, in mock mode, every provider).
                 model_aliases - config.yaml's models.model_aliases.
                 required_capabilities - capability names the request needs
                 (e.g. {"vision"} if it includes an image).
                 capabilities_by_alias - from settings.all_model_capabilities().
    Returns: the first (cheapest) alias in `pool` that is allowed, enabled
    and capable, or None if none qualifies - the caller then blocks (402).
    """
    for alias in pool:
        litellm_name = model_aliases.get(alias)
        if litellm_name is None:
            continue
        provider = litellm_name.split("/", 1)[0]
        if provider not in allowed_providers or provider not in enabled_providers:
            continue
        caps = capabilities_by_alias.get(alias, {})
        if not required_capabilities.issubset({c for c, v in caps.items() if v}):
            continue
        return alias
    return None


# --- Section: circuit breaker — runaway agent loop detection ---
def _hash_last_messages(messages: list[dict], count: int = 3) -> str:
    """SHA-256 hash of the last `count` messages (canonical JSON), used to tell a repeating loop from a changing one."""
    tail = messages[-count:] if len(messages) >= count else messages
    serialized = json.dumps(tail, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def last_message_text(messages: list[dict]) -> str:
    """
    Return the text content of the last message of any role.

    Parameters: messages - the (masked) request messages.
    Returns: the last message's string content, or "" if it has none (e.g.
    an assistant message that only carries tool_calls).
    """
    if not messages:
        return ""
    return messages_module.message_text(messages[-1].get("content"))


@dataclass
class _SessionSignatures:
    """Per-session signature history: (timestamp, embedding_vector, last_3_messages_hash) tuples."""

    entries: list[tuple[float, "object", str]] = field(default_factory=list)


class CircuitBreaker:
    """
    Detects a runaway agent loop per keyed session hash. A "signature" is
    the embedding of the last message (any role) plus a hash of the last 3
    messages; a loop is flagged when more than max_signatures prior
    signatures in the current window are both cosine-similar (> similarity)
    AND share the exact same last-3-messages hash - the hash-equality
    requirement is what lets a normal tool loop (changing tool output, so a
    different hash every turn) pass through unblocked.
    """

    def __init__(self, window_seconds: float, max_signatures: int, similarity_threshold: float) -> None:
        """Parameters: window_seconds, max_signatures, similarity_threshold - config.yaml's thresholds.circuit_breaker."""
        self._window = window_seconds
        self._max_signatures = max_signatures
        self._similarity_threshold = similarity_threshold
        self._history: dict[str, _SessionSignatures] = {}
        self._lock = Lock()
        self.max_tracked_sessions = 10_000  # above this, sessions idle for a full window are forgotten

    def check(self, session_hash: str, messages: list[dict]) -> bool:
        """
        Record this request's signature and check for a runaway loop.

        Parameters: session_hash - the keyed session hash (pipeline/step3_loop_detection.py skips
        calling this entirely when there is no X-Session-ID header - the
        breaker is disabled for that request).
                     messages - the (masked) request messages.
        Returns: True if this request's signature pushed the session over
        max_signatures similar-and-matching signatures within the window
        (the caller returns 429 agent_loop_detected); False otherwise.
        """
        now = time.monotonic()
        text = last_message_text(messages)
        vector = quality.embed_texts([text])[0] if text else None
        last3_hash = _hash_last_messages(messages)

        with self._lock:
            if len(self._history) > self.max_tracked_sessions:
                idle = [h for h, s in self._history.items() if all(now - t >= self._window for t, _v, _h in s.entries)]
                for stale in idle:
                    del self._history[stale]
            session = self._history.setdefault(session_hash, _SessionSignatures())
            session.entries = [(t, v, h) for t, v, h in session.entries if now - t < self._window]

            similar_count = 0
            if vector is not None:
                for _t, prior_vector, prior_hash in session.entries:
                    if prior_hash != last3_hash or prior_vector is None:
                        continue
                    if quality.cosine_similarity(vector, prior_vector) > self._similarity_threshold:
                        similar_count += 1

            session.entries.append((now, vector, last3_hash))
            return similar_count >= self._max_signatures


# --- Section: max_tokens default — recorded, never counted as savings ---
def apply_default_max_tokens(body: dict, max_tokens_default: int) -> tuple[dict, bool]:
    """
    Fill in max_tokens from the tenant default when the request omits it.

    Parameters: body - the request body (already allowlist-validated).
                 max_tokens_default - the tenant's limits.max_tokens_default.
    Returns: (new_body, defaulted) - defaulted is True if max_tokens was
    filled in. This is recorded (defaulted flag, for the
    ledger/logs) but must never be counted as a savings technique - it
    changes the worst-case cost reservation, not the actual cost.
    """
    if body.get("max_tokens") is not None or body.get("max_completion_tokens") is not None:
        return body, False
    new_body = dict(body)
    new_body["max_tokens"] = max_tokens_default
    return new_body, True
