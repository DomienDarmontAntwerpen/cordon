"""
Shared fixtures: mock mode for the whole session, one embedding-model load,
and per test a fresh app on its own temporary database, a TestClient, and a
ready-made team with an API key. Every test runs offline and deterministic.
"""

import os

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
os.environ.setdefault("MOCK_MODE", "true")

import pytest
from fastapi.testclient import TestClient

from src.server import settings as settings_module
from src.server.gateway import create_app

TEST_ADMIN_KEY = "test-admin-key-" + "a" * 24  # 40 chars, well above the 32-char minimum


@pytest.fixture(scope="session", autouse=True)
def _load_embedding_model_once():
    """Load the shared embedding model once per test session rather than per test."""
    from sentence_transformers import SentenceTransformer

    SentenceTransformer("all-MiniLM-L6-v2")
    yield


@pytest.fixture
def cordon_settings(tmp_path, monkeypatch):
    """A Settings instance in mock mode with a fixed admin key, isolated from any real .env."""
    monkeypatch.setenv("MOCK_MODE", "true")
    monkeypatch.setenv("CORDON_ADMIN_KEY", TEST_ADMIN_KEY)
    monkeypatch.delenv("CORDON_DATA_KEY", raising=False)
    monkeypatch.delenv("CORDON_HASH_SALT", raising=False)
    return settings_module.load_settings(env_file=str(tmp_path / "does-not-exist.env"))


@pytest.fixture
def app(tmp_path, cordon_settings):
    """A Cordon FastAPI app backed by a fresh tmp_path SQLite database."""
    db_path = str(tmp_path / "cordon_test.sqlite")
    return create_app(settings_obj=cordon_settings, db_path=db_path)


@pytest.fixture
def client(app):
    """A TestClient for `app`, with lifespan (startup/shutdown) triggered."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def admin_headers():
    """Authorization header for the admin API."""
    return {"Authorization": f"Bearer {TEST_ADMIN_KEY}"}


@pytest.fixture
def tenant_and_key(client, admin_headers):
    """Create one enforce-mode tenant (allowing openai and mistral) and one API key for it."""
    tenant_resp = client.post(
        "/admin/tenants",
        headers=admin_headers,
        json={"name": "Test Tenant", "mode": "enforce", "allowed_providers": ["openai", "mistral"]},
    )
    tenant = tenant_resp.json()
    key_resp = client.post(f"/admin/tenants/{tenant['id']}/keys", headers=admin_headers, json={})
    return tenant, key_resp.json()["key"]


@pytest.fixture
def tenant_headers(tenant_and_key):
    """Authorization header for the tenant API key created by tenant_and_key."""
    _tenant, key = tenant_and_key
    return {"Authorization": f"Bearer {key}"}


@pytest.fixture
def proven_extraction_route(monkeypatch):
    """A routing table in which the benchmark proved the cheapest model, with only the rules checking it, for easy extraction."""
    from src.savings import routing

    monkeypatch.setattr(routing, "ROUTING_TABLE", {"extraction": {"easy": routing.Route("mistral-small", 0.0, 0.0)}})
