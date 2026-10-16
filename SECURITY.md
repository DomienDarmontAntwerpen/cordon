# Security

This document is Cordon's threat model: what it defends against, where,
and what residual risk remains after each control. It is written to match
the code exactly, file by file - if a claim here is wrong, the code (or
this document) has drifted and should be fixed.

## Threat model

| Asset / threat | Control | File | Residual risk |
|---|---|---|---|
| Stolen tenant key | Keys are random tokens (`cdn_` prefix), stored only as a SHA-256 hash, compared via `hmac.compare_digest`; revocable by prefix; expirable. | `security.py`, `store.py` | A stolen, unrevoked, unexpired key can use that tenant's quota until revoked. No key rotation reminders exist yet. |
| Stolen admin key | Same hashing/comparison; disabled entirely (not just "unauthenticated") if `CORDON_ADMIN_KEY` is unset; bearer header only, never a query string. | `security.py`, `routes_admin.py` | A stolen admin key controls every tenant on this one instance - this is why one Cordon instance per client org matters (blast radius is one org, never cross-client). |
| Brute force (keys, admin) | Per-client-IP auth-failure throttle (`fail_limit` per `fail_window_seconds`, then blocked for `throttle_seconds`); shared between tenant-key and admin auth paths. | `security.py` (`AuthFailureThrottle`), `pipeline/step1_admission.py`, `routes_admin.py` | A distributed attack from many IPs is not slowed by this (no global rate limit on auth failures) - a reverse proxy/WAF in front is expected to carry that. |
| Parameter injection / key exfiltration via `api_base` | A strict allowlist of request parameters forwarded to LiteLLM (`ALLOWED_REQUEST_PARAMS`); `api_base`, `custom_llm_provider`, `mock_response`, `extra_headers`, `metadata` and anything else outside the allowlist is rejected with 400, never forwarded. | `security.py` (`validate_request_params`) | None known - this is a strict allowlist, not a denylist, so a new LiteLLM parameter is rejected by default until explicitly added. |
| Prompt injection / system-prompt leakage | Input: phrase/regex detection (EN/NL/FR) + suspicious-base64 pattern, `flag` or `block`. Output: word n-gram overlap against the real system prompt, `redact` or `block`, including across streamed chunk boundaries (a lookback buffer, not just the current chunk). | `guardrails.py`, `patterns.yaml` | Regex/phrase detection misses novel phrasings a trained classifier would catch - documented in REFERENCE's limitations, not silently assumed solved. |
| PII leakage - input | Linear-time regex (email, phone, IBAN, BE national number, Luhn-checked card numbers), `off`/`mask`/`block`, reversible masking restored only in the final response. | `guardrails.py`, `patterns.yaml` | Names and addresses are not detected (need multilingual NER) - documented, not covered. |
| PII leakage - output | Foreign-PII (not the tenant's own data) detection and `redact`/`block` on the model's answer, including inside streamed output and tool-call arguments. | `guardrails.py` | Same structured-identifiers-only limitation as input. |
| PII leakage - tool calls | Masking is applied to message content before the call and restored in both `content` and `tool_calls[].function.arguments` in the response. | `guardrails.py` (`mask_messages`, `restore_message`) | A tool-call argument containing PII the masker does not recognize (e.g. a name) is not masked. |
| PII leakage - logs | A redacting log filter scrubs PII/secret patterns from every record and its args, installed on the root, uvicorn and LiteLLM loggers, before anything is written. | `security.py` (`RedactingLogFilter`) | A new log call that bypasses the standard `logging` module (e.g. raw `print`) would not be scrubbed - none currently do. |
| PII leakage - cache | The ledger and cache store metadata and encrypted values only - no plaintext prompt or answer text is ever written to either table. The semantic cache's guard tokens (negations, numbers, names) are stored only as tenant-scoped keyed hashes. | `store.py`, `cache.py` | The semantic cache stores each question's embedding vector unencrypted; research on embedding inversion shows text can be partly reconstructed from such vectors - keep `semantic_cache` off for tenants with sensitive prompts (it is off by default). The encryption key (`CORDON_DATA_KEY`) in `.env` is the remaining secret to protect. |
| PII leakage - telemetry | `telemetry_content: none` (the tenant default) sends Langfuse no prompt/answer content at all, only metadata (model, tokens, cost). | `telemetry.py` | A tenant that opts into richer `telemetry_content` is deliberately choosing to send more - that is their explicit setting, not a default. |
| PII leakage - error messages | Provider error text is sanitized and truncated (`sanitize_error`) before it is logged, stored or returned - providers sometimes echo request content back in validation errors. | `security.py` | A provider error format not yet seen by `sanitize_error`'s patterns could leak a fragment before the pattern set is updated. |
| Cross-tenant access | Every ledger/cache/key query is scoped by `tenant_id`; cache values are AES-GCM encrypted with the tenant ID as associated data, so decryption fails even with the right key under the wrong tenant; `GET /metrics` with a tenant key always resolves to that key's own `tenant_id`, ignoring any `?tenant_id=` it is sent. | `store.py`, `cache.py`, `routes_public.py` (`/metrics`) | A single process, single SQLite file holds every tenant - a SQL-injection-class bug would be a cross-tenant incident; all queries are parameterized today. |
| DoS - large bodies | `max_body_bytes` enforced on the raw body **before** JSON parsing. | `http_utils.py` (`read_capped_body`), called first in `pipeline/step1_admission.py` | None known at this layer; a reverse proxy should still cap connection-level abuse. |
| DoS - decompression bombs (images) | Image pixel count is checked from the header **before** decoding, rejecting anything over `max_image_pixels`; an unparseable header fails closed (rejected), not silently allowed. | `optimize.py` (`check_image_pixel_limits`) | None known for base64 images; remote image URLs are never fetched by Cordon at all (see SSRF below), so they cannot decompression-bomb Cordon itself. |
| DoS - token floods | `max_messages`, `max_input_tokens`, `max_images`, per-tenant `rpm`/`tpm` token-bucket limits, daily/monthly budget caps with upfront worst-case reservation (not after-the-fact). | `pipeline/step1_admission.py`, `limits.py` | Limits are per-process, in-memory (single worker) - a multi-instance deployment without shared state could let a client exceed limits across instances (see docs/REFERENCE.md, limitations). |
| DoS - regex backtracking | Detection patterns (PII, injection) are written to be linear-time (no nested quantifiers over attacker-controlled spans); the same patterns run in `guardrails.py` and `patterns.yaml`. | `guardrails.py`, `patterns.yaml` | A future pattern added without this discipline could reintroduce catastrophic backtracking - worth a regex-complexity review on any new pattern. |
| Runaway cost / concurrent overshoot | Cost is **reserved** (worst-case pricing) at admission time, before the call, and settled afterward - two concurrent requests cannot both pass a check against the same remaining budget and then both spend it. | `limits.py` (`BudgetTracker.reserve`/`settle`) | Reservation is per-process; see the token-flood row above for the multi-instance caveat. |
| Malicious model file | `src/models/model.pkl`'s SHA-256 is checked against `config.yaml`'s recorded hash before it is ever unpickled, at startup and in `quality.score_answer()`; a mismatch refuses to load. | `security.py` (`verify_model_integrity`) | Pickle is inherently code-executing on load; the hash check stops a *tampered* file, not a vulnerability in `pickle` itself if the committed file were ever compromised at the source. |
| SSRF | Cordon never fetches a remote `image_url` itself - only base64 `data:` URLs are decoded locally; a remote URL is passed through to the provider unchanged, so Cordon never makes an outbound request to a URL the client controls. | `optimize.py` (`parse_data_url`) | The provider itself fetches the remote URL - that is explicitly the provider's own SSRF surface, outside Cordon's control. |
| Dashboard XSS | Strict CSP (`script-src 'self'; style-src 'self'`, no inline anything); every dynamic value is set via `textContent`, never `innerHTML`, in `dashboard.js`. | `security.py` (`DASHBOARD_CSP`), `src/server/static/dashboard.js` | None known - no user-controlled HTML is ever parsed by the page. |
| CSV formula injection | Every exported CSV cell starting with `=`, `+`, `-` or `@` is prefixed with `'` before being written, in both `GET /admin/export` and `audit.py`'s reports. | `http_utils.py` (`csv_safe`) | None known. |
| Supply chain | `requirements.txt` pins exact versions; `tools/ci.sh` runs `pip-audit` on every CI run (network permitting); vendored `chart.umd.min.js` is Chart.js 4.4.9 (UMD build), SHA-256 `bce154080959c574be0bb6b1a924ff32f08ebc6ff460c159171f51c53802c844`. | `requirements.txt`, `tools/ci.sh` | Versions are pinned but not hash-locked yet (`pip-compile --generate-hashes` is a reasonable next step - see docs/REFERENCE.md, limitations). |
| Cordon outage | The failsafe (nginx, in front of Cordon) sends a request straight to the provider only when Cordon gives no answer at all (unreachable, connection dropped, or no reply within 5 minutes); any answer from Cordon - including a refusal - is passed back unchanged. | `deploy/failsafe.conf.template`, checked by `evaluation/failsafe_check.py` | A request sent direct gets **no** masking, limits or ledger row; it is marked `X-Cordon-Failsafe: direct` and in nginx's access log. During an outage the failsafe forwards any request with the client's key, so it must be reachable only from the client's own network. |

## Data handling

- **What is stored:** the ledger (`store.py`) holds request *metadata* only
  - timestamps, tenant/key/tag identifiers, model/provider names, token
  counts, costs, cache/route/quality/guardrail-action labels, status and
  sanitized error text. **No prompt or answer text is ever written to it.**
  The cache stores AES-GCM encrypted answer text, keyed by a hash of the
  (masked) prompt, never in plaintext, and only when caching is enabled.
- **Where:** a single local SQLite file (WAL mode), one file per Cordon
  deployment - never shared between clients, never synced anywhere by
  Cordon itself.
- **Encryption:** cache values are AES-GCM encrypted with `CORDON_DATA_KEY`
  (32 bytes, from `.env`), with the tenant ID as associated data - a value
  cannot be decrypted under the wrong tenant even with the correct key.
- **Retention:** `store.purge(retention_days)` deletes ledger rows and
  expired cache rows older than `retention.retention_days` (default 90
  days) - run it on a schedule (cron, or at startup) for an actual
  retention policy, not just a configured number.
- **Erasure:** `DELETE /admin/tenants/{id}` is a real right-to-erasure
  delete - it removes the tenant row and every row referencing it (API
  keys, ledger, cache) atomically, not just the tenant's settings.

## Data residency

The per-provider `allowed_providers` allowlist is a data-residency control,
not just a feature flag. `config.yaml`'s default pool and defaults:

- **Mistral** - EU-hosted (France). No US CLOUD Act exposure. Allowed by
  default and supported, but not yet verified with live calls (all live
  evidence so far is OpenAI), and no route uses it.
- **OpenAI** - US-hosted, with an EU data-processing addendum available;
  the underlying infrastructure is still US-based. Allowed by default
  (it is also the default `strong` baseline model).
- **DeepSeek** - China-hosted. **Off by default** - a tenant must
  explicitly add `"deepseek"` to `allowed_providers` to use it. A tenant
  handling EU personal data should generally not add it.

## Observe mode

**Observe mode provides no privacy protection beyond Cordon's always-on
controls** (body-size/image-pixel limits, the parameter allowlist). It is
designed to measure and log what *would* happen - including what PII or
injection activity *would* be masked or blocked - without actually
changing, masking, blocking or redirecting anything. A tenant handling
sensitive data should not treat Observe mode as protective; it is a
measurement step on the way to Enforce, not a safe default to stay on
indefinitely.

## Known limitations

- Names and addresses are not detected as PII - only structured
  identifiers (email, phone, IBAN, national ID numbers, card numbers).
- Personal data inside the system (or developer) message is not masked by
  default: the app's own instructions are sent unchanged, which keeps the
  provider's prompt-cache discount working. RAG apps that paste retrieved
  documents into the system message therefore send any personal data in them
  unmasked, unless the team turns on `mask_instructions` (dashboard: "Mask
  personal data in instructions"), which masks the system message too at the
  cost of the provider's prompt-cache discount.
- Prompt-injection detection is phrase/regex-based, not a trained
  classifier - it will miss novel phrasings.
- No harmful-content moderation on model outputs - Cordon guards against
  PII/leak/injection, not generally unsafe content.
- Single process - rate limits, the circuit breaker and cost reservation
  are correct in-memory state for exactly one uvicorn worker; a
  multi-instance deployment needs to move that state to Redis.
- TLS is expected to be terminated by a reverse proxy in front of Cordon;
  Cordon itself binds loopback by default and does not terminate TLS.
- No hash-locked installs yet (`requirements.txt` pins versions, not
  hashes).
- Failsafe limits: a Cordon that hangs is bypassed only after nginx's 5-minute
  read timeout, and Cordon may by then still be calling the provider - that
  request can be billed twice, and the direct one has no ledger row. A
  streamed answer that breaks off after its first chunk cannot fail over
  (part of it was already sent). The direct path serves OpenAI models only.

## Reporting a vulnerability

If you find a security issue in this codebase, please open a private
report rather than a public issue - via GitHub's private vulnerability
reporting on this repository, or by contacting the maintainer directly.
Include enough detail to reproduce it (affected file/endpoint, request
shape, expected vs actual behavior). This is a portfolio/consulting
project, not a monitored production service with an SLA, but reports are
read and acted on.

---

This document and the code are kept in sync as a matter of
process: a control listed here that no longer matches the code, or a
control in the code not listed here, is a bug in this file and should be
fixed the same way any other bug would be.
