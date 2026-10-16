"""
Load test: many requests at the same moment against a real Cordon server (mock mode, no costs).

Each request waits `--latency-ms` at the simulated provider, like a real model
call. Reports what clients get (answered / "busy, retry" / anything else),
how long they wait, the throughput, and whether the bookkeeping stays exact
under load: one ledger row per request, and no budget left reserved.

    .venv/bin/python -m evaluation.load_test --requests 1000 --latency-ms 1000
    .venv/bin/python -m evaluation.load_test --requests 1000 --worker-threads 256 --max-in-flight 1024
"""

import argparse
import asyncio
import os
import secrets
import tempfile
import threading
import time
from collections import Counter

os.environ.update(MOCK_MODE="true", LITELLM_LOCAL_MODEL_COST_MAP="True")
os.environ.setdefault("CORDON_ADMIN_KEY", secrets.token_urlsafe(32))

import httpx  # noqa: E402 - after the environment is set
import uvicorn  # noqa: E402

from src.server import settings as settings_module  # noqa: E402
from src.server.gateway import create_app  # noqa: E402

PORT = 8798
URL = f"http://127.0.0.1:{PORT}"


def start_server(worker_threads: int | None, max_in_flight: int | None):
    """A real Cordon on a temporary database, with optional capacity overrides; returns (server, app)."""
    settings_obj = settings_module.load_settings()
    if worker_threads:
        settings_obj.config["server"]["worker_threads"] = worker_threads
    if max_in_flight:
        settings_obj.config["server"]["max_in_flight"] = max_in_flight
    app = create_app(settings_obj=settings_obj, db_path=os.path.join(tempfile.mkdtemp(), "load.sqlite"))
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="error", backlog=4096))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.05)
    return server, app


def create_team() -> dict:
    """An enforce team with limits far above the test, so only Cordon's own capacity is measured."""
    admin = {"Authorization": f"Bearer {os.environ['CORDON_ADMIN_KEY']}"}
    with httpx.Client(base_url=URL) as client:
        team = client.post("/admin/tenants", headers=admin, json={
            "name": "Load", "mode": "enforce", "holdout_pct": 0.0, "allowed_providers": ["openai"],
            "rpm": 10**6, "tpm": 10**9, "budget_daily_eur": 10**6, "budget_monthly_eur": 10**6}).json()
        key = client.post(f"/admin/tenants/{team['id']}/keys", headers=admin, json={}).json()["key"]
    return {"Authorization": f"Bearer {key}"}


async def fire(n: int, latency_ms: int, headers: dict) -> list[tuple[int, float]]:
    """Send n requests at once (unique questions, so nothing comes from the cache); return (status, seconds) each."""
    limits = httpx.Limits(max_connections=n, max_keepalive_connections=n)
    async with httpx.AsyncClient(base_url=URL, limits=limits, timeout=120) as client:
        async def one(i: int) -> tuple[int, float]:
            """One request; a connection error counts as status 0."""
            start = time.perf_counter()
            try:
                resp = await client.post("/v1/chat/completions", headers={**headers, "X-Mock-Latency-Ms": str(latency_ms)},
                                         json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": f"Question {i}?"}]})
                return resp.status_code, time.perf_counter() - start
            except httpx.HTTPError:
                return 0, time.perf_counter() - start
        return await asyncio.gather(*(one(i) for i in range(n)))


def percentile(values: list[float], p: float) -> float:
    """The p-th percentile (0-1) of values; 0 for none."""
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(p * len(ordered)))] if ordered else 0.0


def report(results: list[tuple[int, float]], wall_s: float, app) -> dict:
    """Summarize what clients got and check the bookkeeping."""
    statuses = Counter(status for status, _ in results)
    answered = [seconds for status, seconds in results if status == 200]
    services = app.state.services
    ledger_rows = len(services.store.query_ledger(limit=10**6))
    tracker = services.budget_tracker
    left_reserved = sum(tracker._daily_reserved.values()) + sum(tracker._monthly_reserved.values())
    return {
        "statuses": dict(statuses), "wall_seconds": round(wall_s, 2),
        "answered_per_second": round(len(answered) / wall_s, 1) if wall_s else 0,
        "answered_p50_s": round(percentile(answered, 0.5), 2), "answered_p95_s": round(percentile(answered, 0.95), 2),
        "ledger_rows_for_answers": ledger_rows, "budget_left_reserved_eur": round(left_reserved, 9),
    }


def main(argv: list[str] | None = None) -> int:
    """Run one load test and print the report."""
    parser = argparse.ArgumentParser(description="Load-test a mock-mode Cordon.")
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--latency-ms", type=int, default=1000, help="simulated provider time per call")
    parser.add_argument("--worker-threads", type=int, default=None, help="override server.worker_threads")
    parser.add_argument("--max-in-flight", type=int, default=None, help="override server.max_in_flight")
    args = parser.parse_args(argv)

    server, app = start_server(args.worker_threads, args.max_in_flight)
    headers = create_team()
    start = time.perf_counter()
    results = asyncio.run(fire(args.requests, args.latency_ms, headers))
    summary = report(results, time.perf_counter() - start, app)
    config = app.state.services.config["server"]
    print(f"{args.requests} requests at once, {args.latency_ms} ms per provider call, "
          f"worker_threads={config['worker_threads']} max_in_flight={config['max_in_flight']}")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    server.should_exit = True
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
