"""
src/protection/security.py and settings startup checks: keys, admin auth, throttling, headers, log redaction, model integrity.

Run: .venv/bin/python -m pytest tests/test_security.py -v
"""

import base64
import os
import pytest
import time

from src.protection import security
from src.server import settings as settings_module


def test_api_key_hashed_not_plaintext():
    """Api key hashed not plaintext."""
    plaintext, key_hash = security.generate_api_key()
    assert plaintext.startswith("cdn_")
    assert key_hash != plaintext
    assert key_hash not in plaintext
    assert security.verify_api_key(plaintext, key_hash)
    assert not security.verify_api_key("cdn_" + "x" * 40, key_hash)


def test_revoked_key_rejected(client, admin_headers, tenant_and_key):
    """Revoked key rejected."""
    tenant, key = tenant_and_key
    prefix = key[:12]
    resp = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200

    revoke_resp = client.delete(f"/admin/keys/{prefix}", headers=admin_headers)
    assert revoke_resp.status_code == 200

    resp2 = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp2.status_code == 401


def test_expired_key_rejected(client, admin_headers, tenant_and_key):
    """Expired key rejected."""
    tenant, _key = tenant_and_key
    past = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 3600))
    resp = client.post(f"/admin/tenants/{tenant['id']}/keys", headers=admin_headers, json={"expires_at": past})
    expired_key = resp.json()["key"]

    resp2 = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {expired_key}"},
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp2.status_code == 401


def test_admin_requires_header_not_query_string(client, admin_headers):
    """Admin requires header not query string."""
    resp = client.post("/admin/tenants", headers=admin_headers, json={"name": "T"})
    assert resp.status_code == 200

    resp2 = client.post(f"/admin/tenants?admin_key={admin_headers['Authorization'][7:]}", json={"name": "T2"})
    assert resp2.status_code == 401


def test_admin_disabled_when_no_admin_key_set(tmp_path, monkeypatch):
    """Admin disabled when no admin key set."""
    monkeypatch.setenv("MOCK_MODE", "true")
    monkeypatch.delenv("CORDON_ADMIN_KEY", raising=False)
    monkeypatch.delenv("CORDON_DATA_KEY", raising=False)
    monkeypatch.delenv("CORDON_HASH_SALT", raising=False)
    settings_obj = settings_module.load_settings(env_file=str(tmp_path / "none.env"))
    assert settings_obj.admin_key is None

    from fastapi.testclient import TestClient

    from src.server.gateway import create_app

    app = create_app(settings_obj=settings_obj, db_path=str(tmp_path / "db.sqlite"))
    with TestClient(app) as client:
        resp = client.post("/admin/tenants", headers={"Authorization": "Bearer anything"}, json={"name": "X"})
        assert resp.status_code == 401


def test_auth_failure_throttling(client):
    """Auth failure throttling."""
    bad_headers = {"Authorization": "Bearer cdn_" + "x" * 40}
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
    statuses = []
    for _ in range(15):
        resp = client.post("/v1/chat/completions", headers=bad_headers, json=body)
        statuses.append(resp.status_code)
    assert 401 in statuses
    assert 429 in statuses
    # once throttled, stays throttled for subsequent attempts in this burst
    assert statuses[-1] == 429


def test_valid_key_is_never_locked_out_by_someone_elses_bad_keys(client, tenant_headers):
    """Behind a proxy all clients share one IP: after 11 bad keys, a valid key must still work."""
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
    for _ in range(11):
        client.post("/v1/chat/completions", headers={"Authorization": "Bearer cdn_wrong"}, json=body)
    assert client.post("/v1/chat/completions", headers={"Authorization": "Bearer cdn_wrong"}, json=body).status_code == 429
    assert client.post("/v1/chat/completions", headers=tenant_headers, json=body).status_code == 200


def test_security_headers_present(client):
    """Security headers present."""
    resp = client.get("/health")
    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    assert resp.headers["X-Frame-Options"] == "DENY"
    assert resp.headers["Referrer-Policy"] == "no-referrer"
    assert resp.headers["Cache-Control"] == "no-store"
    assert "Content-Security-Policy" not in resp.headers


def test_dashboard_csp_present(client):
    """Dashboard csp present."""
    resp = client.get("/dashboard")
    assert "Content-Security-Policy" in resp.headers
    assert "default-src 'self'" in resp.headers["Content-Security-Policy"]


def test_cors_off_by_default(app):
    """Cors off by default."""
    assert app.state.services.settings.config["cors"]["origins"] == []


def test_log_filter_redacts_pii_and_secrets(caplog):
    """Log filter redacts pii and secrets."""
    import logging

    secrets_in_use = ["super-secret-admin-key-value-1234567890"]
    log_filter = security.install_log_redaction(secrets_in_use)
    try:
        test_logger = logging.getLogger("cordon.test.redaction")
        with caplog.at_level(logging.INFO, logger="cordon.test.redaction"):
            test_logger.info(
                "provider error for %s: your key %s and admin secret %s leaked",
                "person@example.com",
                "sk-abcdefghijklmnop",
                "super-secret-admin-key-value-1234567890",
            )
        message = caplog.records[-1].getMessage()
        assert "person@example.com" not in message
        assert "sk-abcdefghijklmnop" not in message
        assert "super-secret-admin-key-value-1234567890" not in message
        assert "[REDACTED_EMAIL]" in message
        assert "[REDACTED_SECRET]" in message
    finally:
        for name in (None, "uvicorn", "uvicorn.access", "uvicorn.error", "litellm", "LiteLLM"):
            logging.getLogger(name).removeFilter(log_filter)


def test_log_filter_does_not_break_percent_style_logging_with_tuple_args(caplog):
    """Guards against the regression where the filter clobbered record.args and crashed uvicorn's access formatter."""
    import logging

    log_filter = security.install_log_redaction([])
    try:
        test_logger = logging.getLogger("cordon.test.argshape")
        with caplog.at_level(logging.INFO, logger="cordon.test.argshape"):
            test_logger.info("%s - %s %s HTTP/%s", "127.0.0.1:12345", "GET", "/health", "1.1")
        record = caplog.records[-1]
        assert isinstance(record.args, tuple)
        assert len(record.args) == 4
        assert record.getMessage()  # must not raise
    finally:
        for name in (None, "uvicorn", "uvicorn.access", "uvicorn.error", "litellm", "LiteLLM"):
            logging.getLogger(name).removeFilter(log_filter)


def test_model_hash_mismatch_refuses_to_load(tmp_path):
    """Model hash mismatch refuses to load."""
    model_path = tmp_path / "model.pkl"
    model_path.write_bytes(b"not a real model, just bytes")
    real_hash = security.file_sha256(str(model_path))
    assert security.verify_model_integrity(str(model_path), real_hash)
    assert not security.verify_model_integrity(str(model_path), "0" * 64)


def test_startup_refuses_outside_mock_mode_without_secrets(tmp_path, monkeypatch):
    """Startup refuses outside mock mode without secrets."""
    monkeypatch.setenv("MOCK_MODE", "false")
    monkeypatch.delenv("CORDON_ADMIN_KEY", raising=False)
    monkeypatch.delenv("CORDON_DATA_KEY", raising=False)
    monkeypatch.delenv("CORDON_HASH_SALT", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(settings_module.StartupError):
        settings_module.load_settings(env_file=str(tmp_path / "none.env"))


def test_startup_refuses_example_admin_key(tmp_path, monkeypatch):
    """Startup refuses example admin key."""
    monkeypatch.setenv("MOCK_MODE", "false")
    monkeypatch.setenv("CORDON_ADMIN_KEY", settings_module.EXAMPLE_ADMIN_KEY_VALUE)
    monkeypatch.setenv("CORDON_DATA_KEY", base64.b64encode(os.urandom(32)).decode())
    monkeypatch.setenv("CORDON_HASH_SALT", "a" * 20)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-for-test")
    with pytest.raises(settings_module.StartupError):
        settings_module.load_settings(env_file=str(tmp_path / "none.env"))


def test_startup_succeeds_with_valid_secrets_outside_mock_mode(tmp_path, monkeypatch):
    """Startup succeeds with valid secrets outside mock mode."""
    monkeypatch.setenv("MOCK_MODE", "false")
    monkeypatch.setenv("CORDON_ADMIN_KEY", "a" * 40)
    monkeypatch.setenv("CORDON_DATA_KEY", base64.b64encode(os.urandom(32)).decode())
    monkeypatch.setenv("CORDON_HASH_SALT", "a" * 20)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-for-test")
    settings_obj = settings_module.load_settings(env_file=str(tmp_path / "none.env"))
    assert settings_obj.mock_mode is False
    assert not settings_obj.ephemeral_secrets


def test_mock_headers_ignored_outside_mock_mode(tmp_path, monkeypatch):
    """Mock headers ignored outside mock mode."""
    import litellm as litellm_module

    monkeypatch.setenv("MOCK_MODE", "false")
    monkeypatch.setenv("CORDON_ADMIN_KEY", "a" * 40)
    monkeypatch.setenv("CORDON_DATA_KEY", base64.b64encode(os.urandom(32)).decode())
    monkeypatch.setenv("CORDON_HASH_SALT", "a" * 20)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-for-test")
    settings_obj = settings_module.load_settings(env_file=str(tmp_path / "none.env"))

    from fastapi.testclient import TestClient

    from src.server.gateway import create_app

    app = create_app(settings_obj=settings_obj, db_path=str(tmp_path / "db.sqlite"))

    captured_kwargs = {}

    def fake_completion(**kwargs):
        """Fake completion."""
        captured_kwargs.update(kwargs)
        raise RuntimeError("stub: no real network call in this test")

    monkeypatch.setattr(litellm_module, "completion", fake_completion)

    with TestClient(app) as client:
        admin_headers = {"Authorization": f"Bearer {settings_obj.admin_key}"}
        tenant_resp = client.post("/admin/tenants", headers=admin_headers, json={"name": "T", "mode": "enforce", "allowed_providers": ["openai"]})
        tenant = tenant_resp.json()
        key_resp = client.post(f"/admin/tenants/{tenant['id']}/keys", headers=admin_headers, json={})
        key = key_resp.json()["key"]

        client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "X-Mock-Bad-Response": "true", "X-Mock-Fail-Step": "llm_call",
                     "X-Mock-Latency-Ms": "30000"},
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert "mock_response" not in captured_kwargs
    assert captured_kwargs, "X-Mock-Fail-Step must not stop a real call outside mock mode"


def test_mock_latency_header_is_bounded():
    """X-Mock-Latency-Ms: milliseconds to seconds, never negative, at most 30 s, junk is 0."""
    from src.server.pipeline import mock

    assert mock.latency_seconds("250") == 0.25
    assert mock.latency_seconds("-5") == 0.0
    assert mock.latency_seconds("999999999") == mock.MAX_MOCK_LATENCY_S
    assert mock.latency_seconds("abc") == 0.0 and mock.latency_seconds(None) == 0.0
