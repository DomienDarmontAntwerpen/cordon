"""
Live checks against the real OpenAI API: a plain and a streamed call, the
openai SDK as a drop-in, an exact cache hit at zero cost, and billed cost
within 5% of the provider's own usage. Skipped without OPENAI_API_KEY and
never run by a plain `pytest`; the whole file costs well under one cent.

Run: .venv/bin/python -m pytest -m live -v
"""

import base64
import os
import secrets
import time

import litellm
import pytest
from dotenv import dotenv_values
from fastapi.testclient import TestClient

from src.paths import REPO_ROOT
from src.server import settings as settings_module
from src.server.gateway import create_app

pytestmark = pytest.mark.live

MODEL = "gpt-4o-mini"
SHORT = {"max_tokens": 20, "temperature": 0}


def _openai_key() -> str | None:
    """Return OPENAI_API_KEY from the environment, else from .env, else None."""
    return os.environ.get("OPENAI_API_KEY") or dotenv_values(REPO_ROOT / ".env").get("OPENAI_API_KEY") or None


@pytest.fixture
def live(tmp_path, monkeypatch):
    """A live-mode Cordon TestClient plus the Authorization header of one enforce tenant with the cache on."""
    key = _openai_key()
    if not key:
        pytest.skip("OPENAI_API_KEY is not set (environment or .env)")
    admin_key = secrets.token_urlsafe(32)
    monkeypatch.setenv("MOCK_MODE", "false")
    monkeypatch.setenv("OPENAI_API_KEY", key)
    monkeypatch.setenv("CORDON_ADMIN_KEY", admin_key)
    monkeypatch.setenv("CORDON_DATA_KEY", base64.b64encode(os.urandom(32)).decode())
    monkeypatch.setenv("CORDON_HASH_SALT", secrets.token_urlsafe(16))
    settings_obj = settings_module.load_settings(env_file=str(tmp_path / "no.env"))
    assert not settings_obj.mock_mode

    with TestClient(create_app(settings_obj=settings_obj, db_path=str(tmp_path / "live.sqlite"))) as client:
        admin = {"Authorization": f"Bearer {admin_key}"}
        tenant = client.post("/admin/tenants", headers=admin, json={
            "name": "live", "mode": "enforce", "allowed_providers": ["openai"], "features": ["cache"], "holdout_pct": 0.0,
        }).json()
        tenant_key = client.post(f"/admin/tenants/{tenant['id']}/keys", headers=admin, json={}).json()["key"]
        yield client, {"Authorization": f"Bearer {tenant_key}"}, tenant_key, settings_obj


def _ask(question: str) -> dict:
    """A short, deterministic request body for MODEL."""
    return {"model": MODEL, "messages": [{"role": "user", "content": question}], **SHORT}


def test_live_plain_call(live):
    """A plain request returns a real answer, real usage and a positive measured cost."""
    client, headers, _key, _settings = live
    resp = client.post("/v1/chat/completions", headers=headers, json=_ask("Reply with the single word: pong"))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["choices"][0]["message"]["content"].strip()
    assert body["usage"]["prompt_tokens"] > 0 and body["usage"]["completion_tokens"] > 0
    assert float(resp.headers["X-Cordon-Cost"]) > 0
    assert resp.headers["X-Cordon-Mode"] == "enforce"


def test_live_streaming(live):
    """A streamed request yields content chunks, the final `: cordon` cost line, and [DONE]."""
    client, headers, _key, _settings = live
    with client.stream("POST", "/v1/chat/completions", headers=headers, json={**_ask("Count from 1 to 5."), "stream": True}) as resp:
        assert resp.status_code == 200
        raw = "".join(resp.iter_text())
    assert '"content"' in raw
    assert ": cordon {" in raw
    assert raw.rstrip().endswith("data: [DONE]")


def test_live_openai_sdk_drop_in(live):
    """The official openai SDK works through Cordon unchanged - plain and streamed."""
    from openai import OpenAI

    client, _headers, tenant_key, _settings = live
    sdk = OpenAI(api_key=tenant_key, base_url=str(client.base_url) + "/v1", http_client=client)
    resp = sdk.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": "Say hello."}], **SHORT)
    assert resp.choices[0].message.content.strip()
    stream = sdk.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": "Say hello."}], stream=True, **SHORT)
    text = "".join(c.choices[0].delta.content or "" for c in stream if c.choices)
    assert text.strip()


def test_live_exact_cache_hit(live):
    """The same request twice: the second is an exact cache hit at zero cost and much faster."""
    client, headers, _key, _settings = live
    body = _ask(f"What is the capital of Belgium? (run {time.time_ns()})")
    first = client.post("/v1/chat/completions", headers=headers, json=body)
    assert first.status_code == 200 and first.headers["X-Cordon-Cache"] == "none"
    second = client.post("/v1/chat/completions", headers=headers, json=body)
    assert second.status_code == 200
    assert second.headers["X-Cordon-Cache"] == "exact"
    assert float(second.headers["X-Cordon-Cost"]) == 0.0
    assert second.json()["choices"][0]["message"]["content"] == first.json()["choices"][0]["message"]["content"]


def test_live_cost_matches_provider_usage(live):
    """Cordon's billed cost is within 5% of the provider-reported usage priced at the provider's rates."""
    client, headers, _key, settings_obj = live
    resp = client.post("/v1/chat/completions", headers=headers, json=_ask("Name three primary colours."))
    assert resp.status_code == 200
    usage = resp.json()["usage"]
    prompt_usd, completion_usd = litellm.cost_per_token(
        model=f"openai/{MODEL}", prompt_tokens=usage["prompt_tokens"], completion_tokens=usage["completion_tokens"]
    )
    provider_eur = (prompt_usd + completion_usd) * settings_obj.config["money"]["usd_to_eur"]
    cordon_eur = float(resp.headers["X-Cordon-Cost"])
    assert provider_eur > 0
    assert abs(cordon_eur - provider_eur) <= 0.05 * provider_eur, (cordon_eur, provider_eur)
