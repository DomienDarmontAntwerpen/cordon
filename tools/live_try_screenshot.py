"""
The dashboard's "Try a request" tab, live: one real support-conversation turn through Cordon (a real
answer, under 0.1 cent), saved as a screenshot. Needs OPENAI_API_KEY and the Cordon secrets in .env.

    .venv/bin/python -m tools.live_try_screenshot docs/images/dashboard_try.png
"""

import os
import sys
import tempfile
import threading
import time

os.environ.update(MOCK_MODE="false", LITELLM_LOCAL_MODEL_COST_MAP="True")

import httpx  # noqa: E402 - after the environment is set
import uvicorn  # noqa: E402

from src.server import settings as settings_module  # noqa: E402
from src.server.gateway import create_app  # noqa: E402

PORT = 8796
TRY_PROMPT = "If it is late, can I get a refund? You can reach me at jan.peeters@example.com"
URL = f"http://127.0.0.1:{PORT}"


def start_live_server() -> tuple[uvicorn.Server, str]:
    """A live Cordon on a temporary database; returns the server and its admin key."""
    settings_obj = settings_module.load_settings()
    app = create_app(settings_obj=settings_obj, db_path=os.path.join(tempfile.mkdtemp(), "try.sqlite"))
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.1)
    return server, settings_obj.admin_key


def try_team_key(admin_key: str) -> str:
    """A team with every saving and protection on (except the semantic cache), capped at EUR 0.50 a day."""
    admin = {"Authorization": f"Bearer {admin_key}"}
    team = httpx.post(f"{URL}/admin/tenants", headers=admin, json={
        "name": "Try it yourself", "mode": "enforce", "holdout_pct": 0.0, "allowed_providers": ["openai"],
        "features": ["trim", "cache", "routing", "max_tokens", "circuit_breaker"], "pii_action": "mask",
        "budget_daily_eur": 0.5}).json()
    key = httpx.post(f"{URL}/admin/tenants/{team['id']}/keys", headers=admin, json={}).json()["key"]
    # One warm-up request (another question, so the cache plays no part): the first request after a
    # start loads the quality check's model, which is not Cordon's usual time per request.
    httpx.post(f"{URL}/v1/chat/completions", headers={"Authorization": f"Bearer {key}"}, timeout=60, json={
        "model": "gpt-4o", "messages": [{"role": "user", "content": "Classify the sentiment as positive or negative: \"fine\""}]})
    return key


def take(path: str, admin_key: str, team_key: str) -> str:
    """Log in, send the panel's example prompt, save the screenshot; returns the real answer."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=2)
        page.goto(f"{URL}/dashboard")
        page.fill("#connect-key-input", admin_key)
        page.click("#connect-form button[type=submit]")
        page.wait_for_selector("#app:not([hidden])")
        page.click(".tab[data-tab-target=try]")
        page.fill("#try-key", team_key)
        page.fill("#try-prompt", TRY_PROMPT)
        page.check("#try-history")  # sent as the next turn of a support conversation: the conversation route
        page.click("#try-send")
        page.wait_for_selector("#try-result:not([hidden])", timeout=60000)
        page.wait_for_timeout(1000)
        page.screenshot(path=path, full_page=True)
        answer = page.inner_text("#try-answer")
        browser.close()
    return answer


def main(argv: list[str] | None = None) -> int:
    """Start a live Cordon, take the screenshot, stop."""
    path = (argv if argv is not None else sys.argv[1:] or ["docs/images/dashboard_try.png"])[0]
    server, admin_key = start_live_server()
    try:
        print("answer:", take(path, admin_key, try_team_key(admin_key)))
    finally:
        server.should_exit = True
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
