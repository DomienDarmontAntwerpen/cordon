"""
Checks the nginx failsafe (deploy/failsafe.conf.template) end to end in Docker,
between a stand-in Cordon and a stand-in provider - no API key needed:
Cordon up; a refusal comes back unchanged; a request Cordon drops goes direct
but the next one goes to Cordon again; Cordon down -> provider with the
client's key; failsafe restart while Cordon is gone; Cordon back.

    .venv/bin/python -m evaluation.failsafe_check     # needs a running Docker daemon
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import httpx

NETWORK = "cordon-failsafe-check"
NGINX_IMAGE = "nginx:1.27.3-alpine"
STUB_IMAGE = "python:3.12-slim"
PORT = 18080
PROVIDER_KEY = "sk-failsafe-check"
TEMPLATE = Path(__file__).resolve().parent.parent / "deploy" / "failsafe.conf.template"

# A stand-in HTTP service: answers every request with who it is, the Authorization header and the body it got.
STUB = '''
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
NAME = sys.argv[1]
class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0))).decode()
        if NAME == "cordon" and self.headers.get("X-Check-Drop"):  # Cordon hangs up mid-request, like a crash
            self.close_connection = True
            return
        status = 403 if self.path.endswith("/refuse") else 200
        reply = json.dumps({"answered_by": NAME, "authorization": self.headers.get("Authorization"), "body": body}).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)
    def log_message(self, *args):
        pass
HTTPServer(("0.0.0.0", 8787), Handler).serve_forever()
'''


# --- Section: Docker helpers ---
def docker(*args: str, check: bool = True) -> str:
    """Run one docker command and return its output."""
    result = subprocess.run(["docker", *args], capture_output=True, text=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"docker {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def start_stub(name: str) -> None:
    """Start a stand-in service named `name` on the check network; return once it accepts connections."""
    docker("run", "-d", "--name", name, "--network", NETWORK, STUB_IMAGE, "python", "-c", STUB, name)
    probe = "import socket; socket.create_connection(('127.0.0.1', 8787), 1)"
    for _ in range(60):
        if subprocess.run(["docker", "exec", name, "python", "-c", probe], capture_output=True).returncode == 0:
            return
        time.sleep(0.5)
    raise RuntimeError(f"stand-in {name} did not start")


def start_failsafe() -> None:
    """Start the real failsafe, its direct path pointed at the stand-in provider."""
    docker(
        "run", "-d", "--name", "failsafe", "--network", NETWORK, "-p", f"127.0.0.1:{PORT}:8080",
        "-e", f"OPENAI_API_KEY={PROVIDER_KEY}", "-e", "FAILSAFE_DIRECT_URL=http://provider:8787",
        "-e", "FAILSAFE_DIRECT_HOST=provider",
        "-v", f"{TEMPLATE}:/etc/nginx/templates/default.conf.template:ro", NGINX_IMAGE,
    )


def cleanup() -> None:
    """Remove every container and the network this check created."""
    for name in ("failsafe", "cordon", "provider"):
        docker("rm", "-f", name, check=False)
    docker("network", "rm", NETWORK, check=False)


# --- Section: requests ---
def ask(path: str = "/v1/chat/completions", wait_s: float = 20.0, key: str = "cdn_team_key", drop: bool = False) -> httpx.Response:
    """POST a chat request with a team key through the failsafe; retry until it answers (startup)."""
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
    deadline = time.time() + wait_s
    while True:
        try:
            return httpx.post(f"http://127.0.0.1:{PORT}{path}", json=body,
                              headers={"Authorization": f"Bearer {key}", **({"X-Check-Drop": "1"} if drop else {})}, timeout=10)
        except httpx.HTTPError:
            if time.time() > deadline:
                raise
            time.sleep(0.5)


def answered_by(response: httpx.Response) -> str:
    """Which stand-in answered: "cordon" or "provider" ("nobody" for an nginx error page)."""
    try:
        return response.json()["answered_by"]
    except (ValueError, KeyError):
        return "nobody"


def check(name: str, condition: bool) -> bool:
    """Print one PASS/FAIL line and return the condition."""
    print(f"{'PASS' if condition else 'FAIL'}  {name}")
    return condition


# --- Section: the checks ---
def run_checks() -> bool:
    """Run the five scenarios in order; True if every check passed."""
    results = []
    first = ask()
    results.append(check("Cordon up: Cordon answers", answered_by(first) == "cordon"))
    results.append(check("Cordon up: the team key reaches Cordon", first.json()["authorization"] == "Bearer cdn_team_key"))

    refused = ask("/v1/refuse")
    results.append(check("A refusal by Cordon comes back unchanged", refused.status_code == 403 and answered_by(refused) == "cordon"))

    dropped = ask(drop=True)  # Cordon hangs up mid-request (a crash): only that request goes direct...
    results.append(check("A request Cordon drops is answered direct", answered_by(dropped) == "provider"))
    after_drop = ask()  # ...and the next one goes to Cordon again
    results.append(check("One dropped request does not send the next ones around Cordon", answered_by(after_drop) == "cordon"))

    docker("stop", "-t", "0", "cordon")  # at once, like a crash
    direct = ask()
    results.append(check("Cordon down: the provider answers", direct.status_code == 200 and answered_by(direct) == "provider"))
    results.append(check("Cordon down: the client's provider key replaces the team key",
                         direct.json()["authorization"] == f"Bearer {PROVIDER_KEY}"))
    results.append(check("Cordon down: the request body arrives intact", json.loads(direct.json()["body"])["model"] == "gpt-4o"))
    results.append(check("Cordon down: marked X-Cordon-Failsafe: direct", direct.headers.get("x-cordon-failsafe") == "direct"))
    results.append(check("Cordon down: a request without a Cordon team key is refused", ask(key="sk-anything").status_code == 401))
    results.append(check("Cordon down: only chat requests go direct", ask("/v1/other").status_code == 503))

    docker("rm", "-f", "cordon")
    docker("restart", "failsafe")
    restarted = ask()
    results.append(check("Failsafe restarted while Cordon is gone: it still answers", answered_by(restarted) == "provider"))

    start_stub("cordon")
    back = None
    for _ in range(60):  # found again after a DNS refresh (at most 10 s)
        back = ask()
        if answered_by(back) == "cordon":
            break
        time.sleep(1)
    results.append(check("Cordon back: Cordon answers again", answered_by(back) == "cordon"))
    return all(results)


def main() -> int:
    """Set up the network and containers, run the checks, always clean up."""
    cleanup()
    try:
        docker("network", "create", NETWORK)
        start_stub("cordon")
        start_stub("provider")
        start_failsafe()
        ok = run_checks()
    finally:
        cleanup()
    print("\nfailsafe check:", "all passed" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
