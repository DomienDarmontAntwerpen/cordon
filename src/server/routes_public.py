"""
Non-admin endpoints: /health, /v1/models, /v1/tenant, /metrics and the dashboard.
A tenant key only ever sees its own tenant's rows.
"""

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.paths import STATIC_DIR
from src.protection import security
from src.server import settings as settings_module
from src.server.http_utils import extract_bearer, is_expired, openai_error
from src.server.schemas import TenantResponse
from src.server.services import Services


def register_public_routes(app: FastAPI, services: Services) -> None:
    """Add the public endpoints to the app."""
    store = services.store
    settings_obj = services.settings

    def _tenant_for_key(request: Request) -> dict | None:
        """Return the tenant row for a valid, unrevoked, unexpired tenant key, or None."""
        token = extract_bearer(request)
        if not token or not token.startswith(security.API_KEY_PREFIX):
            return None
        record = store.find_api_key(security.hash_api_key(token))
        if not record or record.revoked or is_expired(record.expires_at):
            return None
        return store.get_tenant(record.tenant_id)

    @app.get("/health")
    def health() -> dict:
        """Liveness probe. No authentication, no internal details."""
        return {"status": "ok"}

    @app.get("/v1/tenant")
    def get_own_tenant(request: Request):
        """Return the authenticated tenant's own settings (the dashboard's budget card needs its caps)."""
        tenant_row = _tenant_for_key(request)
        if tenant_row is None:
            return openai_error(401, "invalid API key", "authentication_error")
        return TenantResponse(**tenant_row)

    @app.get("/v1/models")
    def list_models(request: Request):
        """Return the model aliases the authenticated tenant is allowed to request."""
        tenant_row = _tenant_for_key(request)
        if tenant_row is None:
            return openai_error(401, "invalid API key", "authentication_error")
        allowed_providers = set(tenant_row["settings"].get("allowed_providers") or [])
        data = [
            {"id": alias, "object": "model", "owned_by": "cordon"}
            for alias, litellm_name in settings_obj.config["models"]["model_aliases"].items()
            if settings_module.Settings.model_provider(litellm_name) in allowed_providers
        ]
        return {"object": "list", "data": data}

    @app.get("/metrics")
    def metrics(request: Request, tenant_id: str | None = None, since: float | None = None):
        """Return raw ledger rows for the dashboard (tenant-scoped for a tenant key, any tenant for the admin key)."""
        token = extract_bearer(request)
        if security.verify_admin_key(token, settings_obj.admin_key):
            rows = store.query_ledger(tenant_id=tenant_id, since=since, limit=5000)
            return {"scope": "tenant" if tenant_id else "all", "tenant_id": tenant_id, "requests": rows}

        if token and token.startswith(security.API_KEY_PREFIX):
            record = store.find_api_key(security.hash_api_key(token))
            if record and not record.revoked and not is_expired(record.expires_at):
                rows = store.query_ledger(tenant_id=record.tenant_id, since=since, limit=5000)
                return {"scope": "tenant", "tenant_id": record.tenant_id, "requests": rows}

        return openai_error(401, "invalid API key", "authentication_error")

    @app.get("/dashboard")
    def dashboard():
        """Serve the dashboard's static HTML page."""
        index_path = STATIC_DIR / "dashboard.html"
        if not index_path.exists():
            return openai_error(404, "dashboard not yet built", "not_found_error")
        return FileResponse(index_path)

    if STATIC_DIR.exists():
        app.mount("/dashboard-assets", StaticFiles(directory=STATIC_DIR), name="dashboard-assets")
