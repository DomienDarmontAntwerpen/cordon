"""
The admin API (admin key only): tenants, API keys and the monthly CSV export.
DELETE of a tenant is the GDPR right to erasure. Every action goes to the admin log.
"""

import calendar
import csv
import io
import json
import secrets
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from src.protection import security
from src.server import settings as settings_module
from src.server.http_utils import client_ip, csv_safe, extract_bearer, openai_error
from src.server.schemas import (
    ApiKeyCreateRequest,
    ApiKeyCreateResponse,
    TenantCreateRequest,
    TenantResponse,
    TenantSettingsPayload,
)
from src.server.services import Services

EXPORT_COLUMNS = [
    "timestamp", "tenant_id", "tags", "requested_model", "served_model", "provider",
    "tokens_in", "tokens_out", "cost_actual_eur", "cost_baseline_eur", "cache_type", "fallback",
    "quality_score", "overhead_ms", "provider_ms", "status", "mode",
]


def register_admin_routes(app: FastAPI, services: Services) -> None:
    """Add the admin endpoints to the app."""
    store = services.store
    settings_obj = services.settings
    throttle = services.throttle

    def _require_admin(request: Request) -> JSONResponse | None:
        """Check the admin bearer token and the throttle. Returns an error response, or None if authorized."""
        ip = client_ip(request)
        if security.verify_admin_key(extract_bearer(request), settings_obj.admin_key):
            return None
        if throttle.is_throttled(ip):
            return openai_error(429, "too many authentication failures, try again later", "rate_limit_error")
        throttle.record_failure(ip)
        return openai_error(401, "admin authentication required", "authentication_error")

    # --- Section: tenants ---
    @app.get("/admin/tenants", response_model=list[TenantResponse])
    def list_tenants(request: Request):
        """List every tenant (id, name, settings, created_at) - used by the dashboard's tenant selector."""
        if (err := _require_admin(request)) is not None:
            return err
        return [TenantResponse(**row) for row in store.list_tenants()]

    @app.post("/admin/tenants", response_model=TenantResponse)
    def create_tenant(request: Request, payload: TenantCreateRequest):
        """Create a tenant, starting from config.yaml's secure tenant_defaults."""
        if (err := _require_admin(request)) is not None:
            return err
        tenant_id = payload.tenant_id or secrets.token_urlsafe(9)
        if store.get_tenant(tenant_id) is not None:
            return openai_error(400, f"tenant already exists: {tenant_id}", "invalid_request_error")
        overrides = payload.model_dump(exclude={"name", "tenant_id"}, exclude_none=True)
        defaults = settings_module.default_tenant_settings(settings_obj.config)
        try:
            resolved = settings_module.merge_tenant_settings(defaults, overrides, settings_obj.config)
        except ValueError as exc:
            return openai_error(400, str(exc), "invalid_request_error")
        store.create_tenant(tenant_id, payload.name, resolved)
        store.write_admin_log("create_tenant", tenant_id, f"name={payload.name}")
        return TenantResponse(**store.get_tenant(tenant_id))

    @app.patch("/admin/tenants/{tenant_id}", response_model=TenantResponse)
    def patch_tenant(request: Request, tenant_id: str, payload: TenantSettingsPayload):
        """Merge new setting values onto an existing tenant."""
        if (err := _require_admin(request)) is not None:
            return err
        tenant_row = store.get_tenant(tenant_id)
        if tenant_row is None:
            return openai_error(404, f"tenant not found: {tenant_id}", "not_found_error")
        overrides = payload.model_dump(exclude_none=True)
        try:
            resolved = settings_module.merge_tenant_settings(tenant_row["settings"], overrides, settings_obj.config)
        except ValueError as exc:
            return openai_error(400, str(exc), "invalid_request_error")
        store.patch_tenant(tenant_id, resolved)
        store.write_admin_log("patch_tenant", tenant_id, f"fields={sorted(overrides.keys())}")
        return TenantResponse(**store.get_tenant(tenant_id))

    @app.delete("/admin/tenants/{tenant_id}")
    def delete_tenant(request: Request, tenant_id: str):
        """Right-to-erasure delete: removes the tenant and every row referencing it."""
        if (err := _require_admin(request)) is not None:
            return err
        existed = store.delete_tenant(tenant_id)
        store.write_admin_log("delete_tenant", tenant_id, None)
        if not existed:
            return openai_error(404, f"tenant not found: {tenant_id}", "not_found_error")
        return {"deleted": tenant_id}

    # --- Section: API keys ---
    @app.post("/admin/tenants/{tenant_id}/keys", response_model=ApiKeyCreateResponse)
    def create_api_key(request: Request, tenant_id: str, payload: ApiKeyCreateRequest):
        """Issue a new API key for a tenant. The plaintext key is shown exactly once."""
        if (err := _require_admin(request)) is not None:
            return err
        if store.get_tenant(tenant_id) is None:
            return openai_error(404, f"tenant not found: {tenant_id}", "not_found_error")
        plaintext, key_hash = security.generate_api_key()
        prefix = security.key_prefix(plaintext)
        store.add_api_key(key_hash, prefix, tenant_id, payload.expires_at)
        store.write_admin_log("create_api_key", tenant_id, f"prefix={prefix}")
        return ApiKeyCreateResponse(key=plaintext, prefix=prefix, tenant_id=tenant_id, expires_at=payload.expires_at)

    @app.delete("/admin/keys/{prefix}")
    def revoke_api_key(request: Request, prefix: str):
        """Revoke an API key by its display prefix."""
        if (err := _require_admin(request)) is not None:
            return err
        revoked = store.revoke_api_key(prefix)
        store.write_admin_log("revoke_api_key", prefix, None)
        if not revoked:
            return openai_error(404, f"key not found: {prefix}", "not_found_error")
        return {"revoked": prefix}

    # --- Section: export ---
    @app.get("/admin/export")
    def export_csv(request: Request, month: str):
        """Export one UTC calendar month ("YYYY-MM") of the ledger as CSV, metadata only."""
        if (err := _require_admin(request)) is not None:
            return err
        try:
            since, until = utc_month_bounds(month)
        except ValueError:
            return openai_error(400, "month must be in YYYY-MM format", "invalid_request_error")

        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(EXPORT_COLUMNS)
        rows = store.query_ledger(tenant_id=None, since=since, until=until, limit=100000)
        for row in rows:
            tags = ",".join(json.loads(row.get("tags_json") or "[]"))
            values = [
                row["timestamp"], row["tenant_id"], tags, row["requested_model"], row["served_model"],
                row["provider"], row["tokens_in"], row["tokens_out"], row["cost_actual_eur"],
                row["cost_baseline_eur"], row["cache_type"], row["fallback"], row["quality_score"],
                row["overhead_ms"], row["provider_ms"], row["status"], row["mode"],
            ]
            writer.writerow([csv_safe(v) for v in values])
        return StreamingResponse(
            iter([buffer.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="cordon_export_{month}.csv"'},
        )


def utc_month_bounds(month: str) -> tuple[float, float]:
    """
    Return the Unix timestamps of the start of a UTC month and of the next month.

    Parameters: month - "YYYY-MM".
    Returns: (start, end) - a ledger row belongs to the month if start <= timestamp < end.
    Raises: ValueError if month is not in YYYY-MM format.
    Why UTC: ledger timestamps and budget periods are UTC; the server's local
    time zone must not move the month boundary.
    """
    parsed = time.strptime(month, "%Y-%m")
    year, mon = parsed.tm_year, parsed.tm_mon
    next_year, next_mon = (year + 1, 1) if mon == 12 else (year, mon + 1)
    return calendar.timegm((year, mon, 1, 0, 0, 0)), calendar.timegm((next_year, next_mon, 1, 0, 0, 0))
