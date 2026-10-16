#!/usr/bin/env bash
# tools/ci.sh - the single source of truth for "CI green". .github/workflows/ci.yml runs this
# exact script; running it locally and getting exit code 0 is what "CI green"
# means - there is no separate CI-only logic anywhere else.
#
# Steps: install (CPU torch index) -> setup_models -> train.py (+ verify the
# model hash) -> pytest -> benchmark.py (mock, a pipeline regression gate,
# not a model-quality gate) -> fail on any benchmark ❌ -> pip-audit if the
# network is reachable, skipped with a notice otherwise.
set -euo pipefail
cd "$(dirname "$0")/.."

# --- Section: python selection — same priority order as the initial setup ---
PYTHON_BIN=""
for candidate in python3.12 python3.11 python3.13; do
  if command -v "$candidate" >/dev/null 2>&1; then
    PYTHON_BIN="$candidate"
    break
  fi
done
if [ -z "$PYTHON_BIN" ]; then
  echo "ci.sh: no python3.12/3.11/3.13 found on PATH - install one first (see README.md)." >&2
  exit 1
fi
echo "ci.sh: using $PYTHON_BIN ($(command -v "$PYTHON_BIN"))"

# --- Section: venv — created once, every command below goes through it ---
if [ ! -d .venv ]; then
  echo "ci.sh: creating .venv"
  "$PYTHON_BIN" -m venv .venv
fi

# --- Section: install — CPU-only torch first, so pip never pulls a CUDA build ---
echo "ci.sh: installing dependencies"
.venv/bin/pip install --upgrade pip --quiet
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet
.venv/bin/pip install -r requirements.txt --quiet

# --- Section: model prefetch — embedding model + spaCy en_core_web_sm, offline afterwards ---
echo "ci.sh: prefetching models"
.venv/bin/python -m tools.setup_models

# --- Section: train.py — (re)trains the quality classifier and verifies its own hash ---
echo "ci.sh: training the quality classifier"
.venv/bin/python -m evaluation.train
.venv/bin/python -c "
import sys
sys.path.insert(0, '.')
import yaml
from src.protection import security
config = yaml.safe_load(open('src/config/config.yaml'))
expected = config['retention']['model_sha256']
if not security.verify_model_integrity('src/models/model.pkl', expected):
    print('ci.sh: src/models/model.pkl does not match the hash just written to config.yaml', file=sys.stderr)
    sys.exit(1)
print('ci.sh: model hash verified')
"

# --- Section: pytest — offline, deterministic, seed 42 (live tests are skipped by default) ---
echo "ci.sh: running tests"
LITELLM_LOCAL_MODEL_COST_MAP=True MOCK_MODE=true .venv/bin/python -m pytest tests/ -v

# --- Section: benchmark.py (mock) — a pipeline regression gate, not a model-quality gate ---
echo "ci.sh: running the benchmark (mock mode)"
set +e
LITELLM_LOCAL_MODEL_COST_MAP=True MOCK_MODE=true CORDON_ADMIN_KEY="$(.venv/bin/python -c 'import secrets; print(secrets.token_urlsafe(32))')" \
  .venv/bin/python -m evaluation.benchmark
BENCHMARK_EXIT=$?
set -e
if [ "$BENCHMARK_EXIT" -ne 0 ]; then
  echo "ci.sh: benchmark reported a clear ❌ regression - failing the build." >&2
  exit 1
fi

# --- Section: pip-audit — only if the network is actually reachable, skipped with a notice otherwise ---
echo "ci.sh: checking dependencies for known vulnerabilities (pip-audit)"
if .venv/bin/python -c "import socket; socket.create_connection(('pypi.org', 443), timeout=3)" >/dev/null 2>&1; then
  .venv/bin/pip install pip-audit --quiet
  .venv/bin/pip-audit || echo "ci.sh: pip-audit reported findings - review them, but this does not fail the build on its own."
else
  echo "ci.sh: network unreachable - skipping pip-audit (notice only, not a failure)."
fi

echo "ci.sh: CI green."
