"""
Plain helper functions and constants shared by the test files (fixtures live in conftest.py).

Usage (pytest.ini puts tests/ on the import path):
    from helpers import CONFIG, team
"""

import json

import yaml

from src.paths import CONFIG_PATH
from tools import audit

CONFIG = yaml.safe_load(open(CONFIG_PATH, encoding="utf-8"))
AUDIT_CONFIG = yaml.safe_load(open(audit.CONFIG_PATH, encoding="utf-8"))
AUDIT_MODEL_ALIASES = AUDIT_CONFIG["models"]["model_aliases"]
PII_VALUES = {"email": "alice@example.com", "iban": "BE68539007547034", "phone": "+32 470 12 34 56"}


def team(client, admin_headers, **settings) -> dict:
    """Create an enforce-mode tenant (holdout off) with the given settings; return its Authorization header."""
    body = {"name": "Review", "mode": "enforce", "allowed_providers": ["openai", "mistral"], "holdout_pct": 0.0, **settings}
    tenant = client.post("/admin/tenants", headers=admin_headers, json=body).json()
    key = client.post(f"/admin/tenants/{tenant['id']}/keys", headers=admin_headers, json={}).json()["key"]
    return {"Authorization": f"Bearer {key}"}


def list_message(text: str) -> list[dict]:
    """A user message in the OpenAI list-of-parts format."""
    return [{"role": "user", "content": [{"type": "text", "text": text}]}]


def capture_provider_messages(monkeypatch) -> list:
    """Intercept litellm.completion; return the list every call's `messages` are appended to."""
    import litellm as litellm_module

    captured = []
    real_completion = litellm_module.completion

    def capturing_completion(**kwargs):
        """Record exactly what Cordon sends to the provider, then answer normally."""
        captured.append(json.dumps(kwargs.get("messages")))
        return real_completion(**kwargs)

    monkeypatch.setattr(litellm_module, "completion", capturing_completion)
    return captured


def png_data_url(width: int, height: int) -> str:
    """Build a base64 PNG data URL for a solid-color test image of the given size."""
    import base64

    import cv2
    import numpy as np

    image = np.zeros((height, width, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode()


def make_cache(tmp_path):
    """Build a Cache instance backed by a tmp_path database, for test isolation."""
    import os

    from src.savings.cache import Cache
    from src.server.store import Store

    store = Store(str(tmp_path / "cache_test.sqlite"))
    return Cache(store, os.urandom(32), hash_salt="a" * 20, semantic_threshold=0.85, cache_ttl_seconds=3600), store


def audit_record(model="gpt-4o", text="hello world", tokens_in=50, tokens_out=20, session_id=None, tags=None, max_tokens=None, timestamp=0):
    """Build one minimal audit-log record for a test fixture."""
    return {
        "model": model, "messages": [{"role": "user", "content": text}], "tokens_in": tokens_in, "tokens_out": tokens_out,
        "tags": tags or [], "session_id": session_id, "max_tokens": max_tokens, "timestamp": timestamp, "images": [],
    }
