"""
SQLite storage: tenants, API keys, the ledger, the encrypted cache and the admin log.
Metadata only - never prompt or answer text. One connection under one lock;
`purge()` removes data older than the retention period.
"""

import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

logger = logging.getLogger("cordon.store")

# --- Section: schema — metadata and encrypted blobs only, never prompt/answer text ---
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS tenants (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    settings_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS api_keys (
    hash TEXT PRIMARY KEY,
    prefix TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT,
    revoked INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_api_keys_tenant ON api_keys (tenant_id);

CREATE TABLE IF NOT EXISTS ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    tenant_id TEXT NOT NULL,
    key_prefix TEXT,
    tags_json TEXT,
    session_hash TEXT,
    mode TEXT NOT NULL,
    holdout INTEGER NOT NULL DEFAULT 0,
    requested_model TEXT,
    served_model TEXT,
    provider TEXT,
    tokens_in INTEGER,
    tokens_out INTEGER,
    cost_actual_eur REAL,
    cost_baseline_eur REAL,
    savings_json TEXT,
    cache_type TEXT,
    route TEXT,
    fallback INTEGER NOT NULL DEFAULT 0,
    quality_score REAL,
    guardrail_actions_json TEXT,
    overhead_ms REAL,
    provider_ms REAL,
    status TEXT,
    error_code TEXT,
    error_detail TEXT
);
CREATE INDEX IF NOT EXISTS idx_ledger_tenant_time ON ledger (tenant_id, timestamp);

CREATE TABLE IF NOT EXISTS cache (
    cache_key TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    cache_type TEXT NOT NULL,
    context_hash TEXT,
    value_encrypted BLOB NOT NULL,
    vector BLOB,
    negation_set_json TEXT,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cache_tenant ON cache (tenant_id);
CREATE INDEX IF NOT EXISTS idx_cache_context ON cache (tenant_id, cache_type, context_hash);
CREATE INDEX IF NOT EXISTS idx_cache_expires ON cache (expires_at);

CREATE TABLE IF NOT EXISTS admin_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    action TEXT NOT NULL,
    target TEXT,
    detail TEXT
);
"""


@dataclass
class ApiKeyRecord:
    """A row from api_keys, without the plaintext key (which is never stored)."""

    hash: str
    prefix: str
    tenant_id: str
    created_at: str
    expires_at: str | None
    revoked: bool


class Store:
    """
    SQLite-backed persistence for tenants, API keys, the ledger, the cache and
    the admin log. One connection, created with check_same_thread=False, used
    under a single lock for every call - correct for Cordon's single-worker
    process and simple enough to reason about for a security
    review.
    """

    def __init__(self, db_path: str) -> None:
        """
        Parameters: db_path - filesystem path to the SQLite file (tests use a
        tmp_path database so runs never share state).
        """
        self._db_path = db_path
        self._lock = Lock()
        self._conn = self._connect()
        with self._lock:
            self._conn.executescript(SCHEMA_SQL)
            self._conn.commit()

    def _connect(self) -> sqlite3.Connection:
        """
        Open the WAL-mode connection and set the database file's permissions to 0600.

        Parameters: none.
        Returns: the sqlite3.Connection.
        Why 0600: the database holds tenant settings and (encrypted) cache
        values; it should not be world- or group-readable on disk.
        """
        path = Path(self._db_path)
        if path.parent and not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        if self._db_path != ":memory:":
            try:
                os.chmod(self._db_path, 0o600)
            except OSError:
                logger.warning("could not chmod database file to 0600")
        return conn

    def close(self) -> None:
        """Close the underlying connection. Parameters/Returns: none."""
        with self._lock:
            self._conn.close()

    # --- Section: tenants — settings stored as one JSON blob per tenant ---
    def create_tenant(self, tenant_id: str, name: str, settings: dict[str, Any]) -> None:
        """
        Insert a new tenant.

        Parameters: tenant_id - caller-chosen unique ID.
                     name - a human-readable label (not treated as secret).
                     settings - the tenant's resolved settings dict (mode,
                     strong, pool, features, limits, ... - see config.yaml's
                     tenant_defaults for the shape).
        Returns: None.
        """
        with self._lock:
            self._conn.execute(
                "INSERT INTO tenants (id, name, settings_json, created_at) VALUES (?, ?, ?, ?)",
                (tenant_id, name, json.dumps(settings), _now_iso()),
            )
            self._conn.commit()

    def get_tenant(self, tenant_id: str) -> dict[str, Any] | None:
        """Return one tenant's {id, name, settings, created_at} or None if it does not exist."""
        with self._lock:
            row = self._conn.execute("SELECT * FROM tenants WHERE id = ?", (tenant_id,)).fetchone()
        return _tenant_row_to_dict(row) if row else None

    def list_tenants(self) -> list[dict[str, Any]]:
        """Return every tenant as a list of {id, name, settings, created_at} dicts."""
        with self._lock:
            rows = self._conn.execute("SELECT * FROM tenants ORDER BY created_at").fetchall()
        return [_tenant_row_to_dict(r) for r in rows]

    def patch_tenant(self, tenant_id: str, settings: dict[str, Any]) -> bool:
        """
        Replace a tenant's settings dict.

        Parameters: tenant_id - the tenant to update.
                     settings - the full, already-merged settings dict to store.
        Returns: True if a row was updated, False if the tenant does not exist.
        """
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE tenants SET settings_json = ? WHERE id = ?",
                (json.dumps(settings), tenant_id),
            )
            self._conn.commit()
            return cursor.rowcount > 0

    def delete_tenant(self, tenant_id: str) -> bool:
        """
        Right-to-erasure delete: remove a tenant and every row that references it
        (API keys, ledger, cache).

        Parameters: tenant_id - the tenant to erase.
        Returns: True if the tenant existed and was deleted.
        Why: GDPR erasure requests must remove all of a tenant's data, not
        just its settings row - this is the one place that happens atomically.
        """
        with self._lock:
            cursor = self._conn.execute("DELETE FROM tenants WHERE id = ?", (tenant_id,))
            existed = cursor.rowcount > 0
            self._conn.execute("DELETE FROM api_keys WHERE tenant_id = ?", (tenant_id,))
            self._conn.execute("DELETE FROM ledger WHERE tenant_id = ?", (tenant_id,))
            self._conn.execute("DELETE FROM cache WHERE tenant_id = ?", (tenant_id,))
            self._conn.commit()
        return existed

    # --- Section: api_keys — only the hash is ever stored ---
    def add_api_key(self, key_hash: str, prefix: str, tenant_id: str, expires_at: str | None) -> None:
        """
        Store a newly issued API key's hash (never the plaintext).

        Parameters: key_hash - SHA-256 hex hash (security.hash_api_key).
                     prefix - the key's display prefix (security.key_prefix).
                     tenant_id - owning tenant.
                     expires_at - optional ISO-8601 expiry.
        Returns: None.
        """
        with self._lock:
            self._conn.execute(
                "INSERT INTO api_keys (hash, prefix, tenant_id, created_at, expires_at, revoked) "
                "VALUES (?, ?, ?, ?, ?, 0)",
                (key_hash, prefix, tenant_id, _now_iso(), expires_at),
            )
            self._conn.commit()

    def find_api_key(self, key_hash: str) -> ApiKeyRecord | None:
        """Return the api_keys row for a given hash, or None. Does not check revocation/expiry."""
        with self._lock:
            row = self._conn.execute("SELECT * FROM api_keys WHERE hash = ?", (key_hash,)).fetchone()
        if not row:
            return None
        return ApiKeyRecord(
            hash=row["hash"],
            prefix=row["prefix"],
            tenant_id=row["tenant_id"],
            created_at=row["created_at"],
            expires_at=row["expires_at"],
            revoked=bool(row["revoked"]),
        )

    def revoke_api_key(self, prefix: str) -> bool:
        """
        Revoke every (normally one) API key matching a given display prefix.

        Parameters: prefix - the key's display prefix.
        Returns: True if at least one key was revoked.
        """
        with self._lock:
            cursor = self._conn.execute("UPDATE api_keys SET revoked = 1 WHERE prefix = ?", (prefix,))
            self._conn.commit()
            return cursor.rowcount > 0

    # --- Section: ledger — one metadata-only row per request ---
    def insert_ledger_row(self, row: dict[str, Any]) -> None:
        """
        Insert one ledger row.

        Parameters: row - a dict matching the `ledger` table's columns
        (missing columns default to NULL). `tags_json`, `savings_json` and
        `guardrail_actions_json` are expected to already be JSON strings.
        Returns: None.
        Why metadata only: see the module docstring - this is the one write
        path for request outcomes, and it must never be handed prompt or
        answer content.
        """
        columns = [
            "timestamp", "tenant_id", "key_prefix", "tags_json", "session_hash", "mode", "holdout",
            "requested_model", "served_model", "provider", "tokens_in", "tokens_out",
            "cost_actual_eur", "cost_baseline_eur", "savings_json", "cache_type", "route",
            "fallback", "quality_score", "guardrail_actions_json", "overhead_ms", "provider_ms",
            "status", "error_code", "error_detail",
        ]
        values = [row.get(c) for c in columns]
        placeholders = ", ".join("?" for _ in columns)
        with self._lock:
            self._conn.execute(
                f"INSERT INTO ledger ({', '.join(columns)}) VALUES ({placeholders})", values
            )
            self._conn.commit()

    def query_ledger(
        self,
        tenant_id: str | None = None,
        since: float | None = None,
        limit: int = 1000,
        until: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Query ledger rows, newest first.

        Parameters: tenant_id - restrict to one tenant (None = all tenants, admin only).
                     since - only rows with timestamp >= this value.
                     limit - maximum rows returned.
                     until - only rows with timestamp < this value.
        Returns: a list of row dicts.
        """
        clauses, params = [], []
        if tenant_id is not None:
            clauses.append("tenant_id = ?")
            params.append(tenant_id)
        if since is not None:
            clauses.append("timestamp >= ?")
            params.append(since)
        if until is not None:
            clauses.append("timestamp < ?")
            params.append(until)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM ledger {where} ORDER BY timestamp DESC LIMIT ?",
                (*params, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def spend_by_tenant(self, since: float) -> dict[str, float]:
        """
        Sum the actual cost of every ledger row since a timestamp, per tenant.

        Parameters: since - a Unix timestamp (e.g. the start of today, UTC).
        Returns: {tenant_id: total cost_actual_eur}. Used at startup to restore
        the budget tracker's in-memory spend after a restart.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT tenant_id, SUM(cost_actual_eur) FROM ledger "
                "WHERE timestamp >= ? AND cost_actual_eur IS NOT NULL GROUP BY tenant_id",
                (since,),
            ).fetchall()
        return {tenant_id: total or 0.0 for tenant_id, total in rows}

    # --- Section: cache — encrypted values, tenant-scoped ---
    def cache_get(self, cache_key: str, tenant_id: str) -> dict[str, Any] | None:
        """Return a non-expired cache row for (cache_key, tenant_id), or None."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM cache WHERE cache_key = ? AND tenant_id = ? AND expires_at > ?",
                (cache_key, tenant_id, time.time()),
            ).fetchone()
        return dict(row) if row else None

    def cache_put(
        self,
        cache_key: str,
        tenant_id: str,
        cache_type: str,
        value_encrypted: bytes,
        vector: bytes | None,
        negation_set: list[str] | None,
        ttl_seconds: int,
        context_hash: str | None = None,
    ) -> None:
        """
        Insert or replace a cache row.

        Parameters: cache_key - the HMAC cache key (cache.py).
                     tenant_id - owning tenant (cache isolation).
                     cache_type - "exact" or "semantic".
                     value_encrypted - AES-GCM ciphertext (security.CacheEncryption).
                     vector - the embedding bytes for a semantic entry, else None.
                     negation_set - the token/number set for a semantic entry, else None.
                     ttl_seconds - seconds until this row expires.
                     context_hash - for a semantic entry, the hash of every
                     part of the request except the last user message (used
                     to find candidate rows at lookup time without storing
                     any message text); None for an exact entry.
        Returns: None.
        """
        now = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO cache "
                "(cache_key, tenant_id, cache_type, context_hash, value_encrypted, vector, negation_set_json, created_at, expires_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    cache_key, tenant_id, cache_type, context_hash, value_encrypted, vector,
                    json.dumps(negation_set) if negation_set is not None else None,
                    now, now + ttl_seconds,
                ),
            )
            self._conn.commit()

    def cache_entries_for_tenant(
        self, tenant_id: str, cache_type: str | None = None, context_hash: str | None = None
    ) -> list[dict[str, Any]]:
        """Return all non-expired cache rows for a tenant, optionally filtered by cache_type and/or context_hash (the semantic-cache lookup scan)."""
        clauses = ["tenant_id = ?", "expires_at > ?"]
        params: list[Any] = [tenant_id, time.time()]
        if cache_type is not None:
            clauses.append("cache_type = ?")
            params.append(cache_type)
        if context_hash is not None:
            clauses.append("context_hash = ?")
            params.append(context_hash)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM cache WHERE {' AND '.join(clauses)}", params
            ).fetchall()
        return [dict(r) for r in rows]

    # --- Section: admin_log — every admin action, no secrets ---
    def write_admin_log(self, action: str, target: str | None, detail: str | None) -> None:
        """
        Record one admin action.

        Parameters: action - e.g. "create_tenant", "revoke_key", "delete_tenant".
                     target - the affected tenant ID or key prefix.
                     detail - a short, non-secret description.
        Returns: None.
        """
        with self._lock:
            self._conn.execute(
                "INSERT INTO admin_log (timestamp, action, target, detail) VALUES (?, ?, ?, ?)",
                (time.time(), action, target, detail),
            )
            self._conn.commit()

    # --- Section: retention — purge old ledger rows and expired cache rows ---
    def purge(self, retention_days: int) -> dict[str, int]:
        """
        Delete ledger rows older than retention_days and expired cache rows.

        Parameters: retention_days - the configured retention window.
        Returns: {"ledger_deleted": n, "cache_deleted": n}.
        Why: data minimization and enforced retention. Runs at startup and every hour (gateway.py).
        """
        cutoff = time.time() - retention_days * 86400
        with self._lock:
            ledger_cursor = self._conn.execute("DELETE FROM ledger WHERE timestamp < ?", (cutoff,))
            cache_cursor = self._conn.execute("DELETE FROM cache WHERE expires_at < ?", (time.time(),))
            self._conn.commit()
        result = {"ledger_deleted": ledger_cursor.rowcount, "cache_deleted": cache_cursor.rowcount}
        logger.info("retention purge: %s", result)
        return result


# --- Section: helpers ---
def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string (no microseconds)."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _tenant_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    """Convert a `tenants` table row into {id, name, settings, created_at}, parsing settings_json."""
    return {
        "id": row["id"],
        "name": row["name"],
        "settings": json.loads(row["settings_json"]),
        "created_at": row["created_at"],
    }
