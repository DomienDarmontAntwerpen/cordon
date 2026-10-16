"""
The ledger: one metadata-only row per request (never any text).
Every way a request ends writes one row - the dashboard, /metrics and the
export read nothing else.
"""

import json
import time
from typing import Any

from src.server.store import Store


def write_block_ledger(
    store: Store,
    tenant_id: str,
    key_prefix_value: str,
    tags: list[str],
    tenant_settings: dict,
    requested_alias: str | None,
    error_code: str,
    session_hash: str | None = None,
    guardrail_actions: dict[str, Any] | None = None,
) -> None:
    """
    Record an admission-time block as a zero-cost "blocked" ledger row.

    Why: these requests are rejected before generation, so `write_ledger`
    never runs for them - without this, the dashboard's "blocked requests by
    reason" panel and its alerts would have no data for exactly the requests
    they exist to summarize.
    """
    store.insert_ledger_row(
        {
            "timestamp": time.time(),
            "tenant_id": tenant_id,
            "key_prefix": key_prefix_value,
            "tags_json": json.dumps(tags),
            "session_hash": session_hash,
            "mode": tenant_settings.get("mode", "observe"),
            "holdout": 0,
            "requested_model": requested_alias,
            "served_model": None,
            "provider": None,
            "tokens_in": None,
            "tokens_out": None,
            "cost_actual_eur": 0.0,
            "cost_baseline_eur": None,
            "savings_json": json.dumps({}),
            "cache_type": None,
            "route": None,
            "fallback": 0,
            "quality_score": None,
            "guardrail_actions_json": json.dumps(guardrail_actions or {}),
            "overhead_ms": 0.0,
            "provider_ms": 0.0,
            "status": "blocked",
            "error_code": error_code,
            "error_detail": None,
        }
    )


def write_ledger(
    store: Store,
    tenant_id: str,
    key_prefix_value: str,
    tags: list[str],
    tenant_settings: dict,
    requested_alias: str,
    served_alias: str | None,
    provider: str | None,
    tokens_in: int | None,
    tokens_out: int | None,
    cost_actual_eur: float | None,
    cost_baseline_eur: float | None,
    overhead_ms: float,
    provider_ms: float,
    status: str,
    error_code: str | None,
    error_detail: str | None,
    session_hash: str | None = None,
    holdout: bool = False,
    savings: dict | None = None,
    cache_type: str | None = None,
    guardrail_actions: dict | None = None,
) -> None:
    """Insert one ledger row. See store.insert_ledger_row for the column meanings."""
    store.insert_ledger_row(
        {
            "timestamp": time.time(),
            "tenant_id": tenant_id,
            "key_prefix": key_prefix_value,
            "tags_json": json.dumps(tags),
            "session_hash": session_hash,
            "mode": tenant_settings.get("mode", "observe"),
            "holdout": 1 if holdout else 0,
            "requested_model": requested_alias,
            "served_model": served_alias,
            "provider": provider,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost_actual_eur": cost_actual_eur,
            "cost_baseline_eur": cost_baseline_eur,
            "savings_json": json.dumps(savings or {}),
            "cache_type": cache_type,
            "route": served_alias if served_alias != requested_alias else None,
            "fallback": 0,
            "quality_score": None,
            "guardrail_actions_json": json.dumps(guardrail_actions or {}),
            "overhead_ms": overhead_ms,
            "provider_ms": provider_ms,
            "status": status,
            "error_code": error_code,
            "error_detail": error_detail,
        }
    )
