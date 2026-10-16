"""
The long-lived objects every request needs (settings, database, cache,
limiters, thread pool, capacity gate), built once at startup. Budget spend is
restored from the ledger, so a restart does not reset it.
"""

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from src.protection import limits, security
from src.savings.cache import Cache
from src.server import settings as settings_module
from src.server.store import Store
from src.server.telemetry import Telemetry


class CapacityGate:
    """
    Counts requests in flight and refuses new ones above a limit.

    Why: without a limit, a burst of slow model calls queues requests until
    every client times out. Refusing early with 503 + Retry-After lets clients
    back off (or fall back) instead.
    """

    def __init__(self, max_in_flight: int) -> None:
        """Parameters: max_in_flight - requests allowed running or waiting at once."""
        self.max_in_flight = max_in_flight
        self._in_flight = 0
        self._lock = threading.Lock()

    def try_enter(self) -> bool:
        """Take a slot if one is free. Returns False when the server is full."""
        with self._lock:
            if self._in_flight >= self.max_in_flight:
                return False
            self._in_flight += 1
            return True

    def leave(self) -> None:
        """Give a slot back."""
        with self._lock:
            self._in_flight = max(0, self._in_flight - 1)

    @property
    def in_flight(self) -> int:
        """Requests currently in flight."""
        with self._lock:
            return self._in_flight


@dataclass
class Services:
    """Every long-lived object a request may need. One instance per process."""

    settings: settings_module.Settings
    store: Store
    cache: Cache
    telemetry: Telemetry
    throttle: security.AuthFailureThrottle
    rate_limiter: limits.RateLimiter
    budget_tracker: limits.BudgetTracker
    circuit_breaker: limits.CircuitBreaker
    executor: ThreadPoolExecutor = field(default=None)
    capacity: CapacityGate = field(default=None)

    @property
    def config(self) -> dict:
        """The loaded config.yaml, as a dict."""
        return self.settings.config


def build_services(settings_obj: settings_module.Settings, db_path: str) -> Services:
    """
    Create every long-lived object from the loaded settings.

    Parameters: settings_obj - the loaded Settings. db_path - the SQLite file.
    Returns: a Services value, ready to share across requests.
    """
    config = settings_obj.config
    auth_cfg = config["auth"]
    thresholds = config["thresholds"]
    circuit_breaker_cfg = thresholds["circuit_breaker"]
    store = Store(db_path)
    budget_tracker = limits.BudgetTracker()
    day_start, month_start = budget_tracker.period_starts()
    budget_tracker.restore_spent(store.spend_by_tenant(day_start), store.spend_by_tenant(month_start))
    return Services(
        settings=settings_obj,
        store=store,
        cache=Cache(
            store, settings_obj.data_key, settings_obj.hash_salt,
            semantic_threshold=thresholds["semantic_threshold"], cache_ttl_seconds=thresholds["cache_ttl_seconds"],
        ),
        telemetry=Telemetry(settings_obj.langfuse_public_key, settings_obj.langfuse_secret_key, settings_obj.langfuse_host),
        throttle=security.AuthFailureThrottle(
            fail_limit=auth_cfg["fail_limit"],
            fail_window_seconds=auth_cfg["fail_window_seconds"],
            throttle_seconds=auth_cfg["throttle_seconds"],
        ),
        rate_limiter=limits.RateLimiter(),
        budget_tracker=budget_tracker,
        circuit_breaker=limits.CircuitBreaker(
            window_seconds=circuit_breaker_cfg["window_seconds"], max_signatures=circuit_breaker_cfg["max_signatures"],
            similarity_threshold=circuit_breaker_cfg["similarity"],
        ),
        executor=ThreadPoolExecutor(max_workers=config["server"]["worker_threads"], thread_name_prefix="cordon-request"),
        capacity=CapacityGate(config["server"]["max_in_flight"]),
    )
