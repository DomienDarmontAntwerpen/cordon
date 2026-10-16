# The Cordon image: CPU-only, non-root, read-only at runtime, models baked in (offline).

FROM python:3.12-slim

# --- Section: system deps — only what torch/opencv/spaCy need to import, nothing else ---
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Section: install — CPU-only torch first, so pip never pulls a CUDA build ---
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# --- Section: app code + model prefetch — baked into the image, offline at runtime ---
# The embedding model is downloaded into /app/models (not root's home), so the
# non-root runtime user can read it; at runtime the Hugging Face libraries are
# switched to offline mode, so the read-only container never tries to download.
ENV HF_HOME=/app/models
COPY src/ src/
COPY tools/ tools/
RUN python -m tools.setup_models
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# --- Section: non-root user — the image never runs as root ---
RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin cordon \
    && mkdir -p data reports \
    && chown -R cordon:cordon /app
USER 10001

EXPOSE 8787

# --- Section: healthcheck — same /health endpoint the compose file and demo.py poll ---
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8787/health', timeout=2)" || exit 1

# --- Section: run — single uvicorn worker, matching the in-memory per-process state design ---
CMD ["python", "-c", "import os, uvicorn; from src.server.gateway import create_app; uvicorn.run(create_app(), host='0.0.0.0', port=int(os.environ.get('CORDON_PORT', '8787')), workers=1)"]
