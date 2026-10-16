"""
Measures the time Cordon itself adds per request (p50/p95), all features on,
excluding the model call. The README's latency figure comes from here.

    .venv/bin/python -m evaluation.measure_overhead --requests 500
"""

import argparse
import os
import secrets
import statistics
import sys
import tempfile
import threading
import time
from pathlib import Path

ALL_FEATURES = ["trim", "vision", "cache", "semantic_cache", "circuit_breaker", "max_tokens", "routing"]
PORT = 8798


def measure(requests: int) -> tuple[float, float]:
    """
    Run the measurement and return (p50_ms, p95_ms).

    Parameters: requests - how many measured (non-cached) requests to send.
    """
    os.environ["MOCK_MODE"] = "true"
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    os.environ["CORDON_ADMIN_KEY"] = secrets.token_urlsafe(32)

    import httpx
    import uvicorn

    from src.server.gateway import create_app

    with tempfile.TemporaryDirectory() as tmp_dir:
        app = create_app(db_path=str(Path(tmp_dir) / "overhead.sqlite"))
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="warning"))
        threading.Thread(target=server.run, daemon=True).start()
        while not server.started:
            time.sleep(0.1)
        try:
            client = httpx.Client(base_url=f"http://127.0.0.1:{PORT}", timeout=60)
            admin = {"Authorization": f"Bearer {os.environ['CORDON_ADMIN_KEY']}"}
            tenant = client.post("/admin/tenants", headers=admin, json={
                "name": "overhead", "mode": "enforce", "holdout_pct": 0.0, "rpm": 100000, "tpm": 10**9,
                "allowed_providers": ["openai", "mistral"], "features": ALL_FEATURES,
            }).json()
            key = client.post(f"/admin/tenants/{tenant['id']}/keys", headers=admin, json={}).json()["key"]
            headers = {"Authorization": f"Bearer {key}"}

            for i in range(10):  # warm-up: first calls load models and fill caches of the libraries
                client.post("/v1/chat/completions", headers=headers,
                            json={"model": "gpt-4o", "messages": [{"role": "user", "content": f"warm up {i}"}]})

            overheads = []
            for i in range(requests):
                question = f"Summarize item {i}: the quarterly report shows revenue growth of {i} percent."
                resp = client.post(
                    "/v1/chat/completions", headers={**headers, "X-Session-ID": f"session-{i}"},
                    json={"model": "gpt-4o", "messages": [{"role": "user", "content": question}]},
                )
                if resp.status_code != 200 or resp.headers.get("X-Cordon-Cache") != "none":
                    raise RuntimeError(f"unexpected response {resp.status_code}: {resp.text[:200]}")
                overheads.append(float(resp.headers["X-Cordon-Overhead-Ms"]))
        finally:
            server.should_exit = True

    overheads.sort()
    return statistics.median(overheads), overheads[max(0, int(0.95 * len(overheads)) - 1)]


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code."""
    parser = argparse.ArgumentParser(description="Measure Cordon's own per-request overhead (mock mode).")
    parser.add_argument("--requests", type=int, default=200)
    args = parser.parse_args(argv)
    p50, p95 = measure(args.requests)
    print(f"Cordon overhead over {args.requests} requests, all features on: p50 {p50:.1f} ms, p95 {p95:.1f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
