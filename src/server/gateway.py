"""
The app factory: builds the FastAPI app and wires its parts together.

Startup loads the embedding model and starts the hourly data purge; every
response gets the security headers. Request logic lives in pipeline/run.py.

    uvicorn --factory src.server.gateway:create_app --workers 1
"""

import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError

from src.protection import security
from src.savings import quality
from src.server import settings as settings_module
from src.server.pipeline.run import handle_chat_completion
from src.server.http_utils import openai_error
from src.server.routes_admin import register_admin_routes
from src.server.routes_public import register_public_routes
from src.server.services import build_services
from src.server.store import Store

DEFAULT_DB_PATH = "data/cordon.sqlite"
PURGE_INTERVAL_SECONDS = 3600


def create_app(settings_obj: settings_module.Settings | None = None, db_path: str | None = None) -> FastAPI:
    """
    Build the Cordon FastAPI application.

    Parameters: settings_obj - pre-loaded Settings (tests pass one); None
    loads them from .env and config.yaml. db_path - the SQLite file; None
    uses the CORDON_DB_PATH environment variable, then "data/cordon.sqlite"
    (the variable lets a plain `uvicorn --factory` process, such as the
    demo's server, use its own database).
    Returns: a configured FastAPI app, ready for uvicorn or a TestClient.
    """
    if settings_obj is None:
        settings_obj = settings_module.load_settings()
    settings_module.configure_logging(settings_obj)
    services = build_services(settings_obj, db_path or os.environ.get("CORDON_DB_PATH", DEFAULT_DB_PATH))
    retention_days = settings_obj.config["retention"]["retention_days"]
    stop_event = asyncio.Event()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Load the embedding model, purge expired data, run the hourly purge, and clean up on shutdown."""
        # Loaded eagerly so the first request never waits seconds for it while blocking the event loop.
        await asyncio.to_thread(quality.get_embedding_model)
        services.store.purge(retention_days)
        purge_task = asyncio.create_task(_purge_loop(services.store, retention_days, stop_event))
        try:
            yield
        finally:
            stop_event.set()
            purge_task.cancel()
            services.telemetry.shutdown()
            services.executor.shutdown(wait=False, cancel_futures=True)
            services.store.close()

    app = FastAPI(title="Cordon", lifespan=lifespan)
    app.state.services = services

    cors_origins = settings_obj.config.get("cors", {}).get("origins") or []
    if cors_origins:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(CORSMiddleware, allow_origins=cors_origins, allow_methods=["*"], allow_headers=["*"])

    @app.middleware("http")
    async def _security_headers_mw(request: Request, call_next):
        """Add security headers (and the dashboard's CSP) to every response."""
        response = await call_next(request)
        for name, value in security.security_headers(dashboard=request.url.path.startswith("/dashboard")).items():
            response.headers[name] = value
        return response

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        """OpenAI-compatible chat completions. Runs the nine-step pipeline - see pipeline/run.py."""
        return await handle_chat_completion(request, services)

    register_public_routes(app, services)
    register_admin_routes(app, services)
    app.add_exception_handler(RequestValidationError, _invalid_body)
    return app


async def _purge_loop(store: Store, retention_days: int, stop_event: asyncio.Event) -> None:
    """Run store.purge() once an hour until stop_event is set."""
    try:
        while not stop_event.is_set():
            await asyncio.sleep(PURGE_INTERVAL_SECONDS)
            if stop_event.is_set():
                break
            await asyncio.to_thread(store.purge, retention_days)
    except asyncio.CancelledError:
        pass


async def _invalid_body(request: Request, exc: RequestValidationError):
    """A body that fails validation: a 422 in OpenAI's error format, naming the field but never echoing the value."""
    first = exc.errors()[0] if exc.errors() else {}
    field = ".".join(str(part) for part in first.get("loc", ())[1:]) or "body"
    return openai_error(422, f"invalid value for {field}: {first.get('msg', 'invalid request body')}", "invalid_request_error")
