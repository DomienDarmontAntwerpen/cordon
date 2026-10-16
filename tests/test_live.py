"""
Live checks against the real OpenAI API: a plain and a streamed call, the
openai SDK as a drop-in, an exact cache hit at zero cost, billed cost within
5% of the provider's own usage - and each component that changes what reaches
the provider: masking, images, trimming, the answer-length cap and a route
with its check. Skipped without OPENAI_API_KEY and never run by a plain
`pytest`; the whole file costs about one cent.

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
        client.admin_headers = admin  # for tests that need a team with other features (_team)
        yield client, {"Authorization": f"Bearer {tenant_key}"}, tenant_key, settings_obj


def _team(client, **settings) -> dict:
    """The Authorization header of a new enforce team (OpenAI only, no holdout) with the given settings."""
    tenant = client.post("/admin/tenants", headers=client.admin_headers, json={
        "name": f"live-{secrets.token_hex(3)}", "mode": "enforce", "allowed_providers": ["openai"], "holdout_pct": 0.0, **settings,
    }).json()
    key = client.post(f"/admin/tenants/{tenant['id']}/keys", headers=client.admin_headers, json={}).json()["key"]
    return {"Authorization": f"Bearer {key}"}


def _record_provider_calls(monkeypatch) -> list[dict]:
    """Record every call's arguments on its way to the real provider (the call itself still happens)."""
    calls, real = [], litellm.completion
    monkeypatch.setattr(litellm, "completion", lambda **kwargs: calls.append(kwargs) or real(**kwargs))
    return calls


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


def test_live_personal_data_never_reaches_the_provider_and_comes_back(live, monkeypatch):
    """OpenAI sees placeholders only; the caller gets the real email back in the answer."""
    client, _headers, _key, _settings = live
    calls = _record_provider_calls(monkeypatch)
    headers = _team(client, features=[], pii_action="mask")
    resp = client.post("/v1/chat/completions", headers=headers, json={**SHORT, "model": MODEL, "max_tokens": 40, "messages": [
        {"role": "user", "content": "Copy this line exactly, nothing else: Contact jan.peeters@example.com, account BE71096123456769"}]})
    assert resp.status_code == 200, resp.text
    sent = str(calls[-1]["messages"])
    assert "jan.peeters@example.com" not in sent and "BE71096123456769" not in sent
    assert "jan.peeters@example.com" in resp.json()["choices"][0]["message"]["content"]


def _photo_with_number(number: str) -> str:
    """A 2000x1500 JPEG data URL with a large number on it and GPS EXIF data, as a phone photo has."""
    import io

    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (2000, 1500), "white")
    ImageDraw.Draw(image).text((500, 300), number, fill="black", font=ImageFont.load_default(size=700))
    exif = Image.Exif()
    exif[0x8825] = {1: "N", 2: (51.0, 13.0, 0.0)}  # GPSInfo: where the photo was taken
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", exif=exif)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def test_live_image_is_shrunk_and_cleaned_and_still_read(live, monkeypatch):
    """A phone-size photo: OpenAI gets a small, low-detail copy without EXIF, still reads it, and it costs less."""
    from src.savings import optimize

    client, _headers, _key, _settings = live
    body = {**SHORT, "model": MODEL, "messages": [{"role": "user", "content": [
        {"type": "text", "text": "What number is written in this image? Reply with the digits only."},
        {"type": "image_url", "image_url": {"url": _photo_with_number("47"), "detail": "high"}}]}]}
    direct = litellm.completion(**{**body, "model": f"openai/{MODEL}"})
    calls = _record_provider_calls(monkeypatch)
    # A high-detail photo is 25,501 prompt tokens for gpt-4o-mini, above the default 16,000 input limit.
    resp = client.post("/v1/chat/completions", headers=_team(client, features=["vision"], max_input_tokens=60000), json=body)
    assert resp.status_code == 200, resp.text
    sent = calls[-1]["messages"][0]["content"][1]["image_url"]
    raw, _mime = optimize.parse_data_url(sent["url"])
    assert sent["detail"] == "low" and max(optimize.read_image_dimensions(raw)) <= 512 and b"Exif" not in raw
    assert "47" in resp.json()["choices"][0]["message"]["content"] and "47" in direct.choices[0].message.content
    assert resp.json()["usage"]["prompt_tokens"] < direct.usage.prompt_tokens / 5  # billed by OpenAI itself
    assert float(resp.headers["X-Cordon-Cost"]) < float(resp.headers["X-Cordon-Baseline-Cost"])


def test_live_trimmed_prompt_is_billed_less(live):
    """Text pasted five times: OpenAI bills Cordon's request for far fewer prompt tokens than the original."""
    client, _headers, _key, _settings = live
    pasted = "The delivery terms say that goods ship within two working days of payment. " * 5
    body = _ask(pasted + "\n\nHow fast do goods ship?")
    direct = litellm.completion(**{**body, "model": f"openai/{MODEL}"})
    resp = client.post("/v1/chat/completions", headers=_team(client, features=["trim"]), json=body)
    assert resp.status_code == 200, resp.text
    assert resp.json()["usage"]["prompt_tokens"] < direct.usage.prompt_tokens * 0.6


def test_live_answer_length_cap_and_a_cut_off_answer_are_reported(live, monkeypatch):
    """The team's length cap reaches OpenAI, and an answer OpenAI cut off arrives with finish_reason "length"."""
    client, _headers, _key, _settings = live
    calls = _record_provider_calls(monkeypatch)
    headers = _team(client, features=["max_tokens"], max_tokens_default=5)
    resp = client.post("/v1/chat/completions", headers=headers, json={"model": MODEL, "messages": [
        {"role": "user", "content": "Tell me a long story about a lighthouse keeper."}]})
    assert resp.status_code == 200, resp.text
    assert calls[-1]["max_tokens"] == 5
    assert resp.json()["choices"][0]["finish_reason"] == "length"


def test_live_a_proven_route_answers_with_the_cheaper_model_and_its_check(live):
    """A classification goes to gpt-4.1-mini (the committed route); the check's verdict is reported and the answer is right."""
    client, _headers, _key, _settings = live
    headers = _team(client, features=["routing"], strong="gpt-4o")
    resp = client.post("/v1/chat/completions", headers=headers, json={"model": "gpt-4o", "max_tokens": 10, "temperature": 0,
        "messages": [{"role": "user", "content": 'Classify the sentiment of this review as positive or negative: "A warm, funny film."'}]})
    assert resp.status_code == 200, resp.text
    assert resp.headers["X-Cordon-Kind"] == "classification/easy"
    assert resp.headers["X-Cordon-Check"] in {"classifier", "rules_only"} or resp.headers["X-Cordon-Fallback"] == "true"
    assert "positive" in resp.json()["choices"][0]["message"]["content"].lower()
    if resp.headers["X-Cordon-Fallback"] == "false":
        assert resp.headers["X-Cordon-Route"] == "gpt-4.1-mini"
        assert float(resp.headers["X-Cordon-Cost"]) < float(resp.headers["X-Cordon-Baseline-Cost"])

