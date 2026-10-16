"""
Security primitives.

- API keys (`cdn_...`, stored hashed, compared in constant time) and the admin key.
- Throttling of repeated failed logins, per IP.
- The request-parameter allowlist (stops e.g. `api_base` from redirecting our keys).
- Keyed hashes, security headers, the log-redaction filter, error sanitizing.
- AES-GCM encryption of cache values, and the model file's SHA-256 check.
"""

import base64
import hashlib
import hmac
import logging
import os
import secrets
import time
from dataclasses import dataclass, field
from threading import Lock

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from src.protection.guardrails import redact

logger = logging.getLogger("cordon.security")

# --- Section: API keys — format, hashing, constant-time comparison ---
API_KEY_PREFIX = "cdn_"
API_KEY_RANDOM_BYTES = 32  # urlsafe-encoded, shown once at creation


def generate_api_key() -> tuple[str, str]:
    """
    Generate a new tenant API key.

    Parameters: none.
    Returns: (plaintext_key, sha256_hex_hash). The plaintext is shown once at
    creation and never stored; only the hash is persisted (store.py).
    Why: if the database is ever read, a stolen hash cannot be used as a key
    (SHA-256 is not reversible), and comparison at auth time never needs the
    plaintext from storage.
    """
    plaintext = API_KEY_PREFIX + secrets.token_urlsafe(API_KEY_RANDOM_BYTES)
    return plaintext, hash_api_key(plaintext)


def hash_api_key(plaintext: str) -> str:
    """
    Hash an API key for storage/comparison.

    Parameters: plaintext - the full "cdn_..." key string.
    Returns: the SHA-256 hex digest.
    Why: plain API keys are never stored; only their hash is, so a database
    leak does not hand out usable keys.
    """
    return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()


def verify_api_key(plaintext: str, stored_hash: str) -> bool:
    """
    Check a presented API key against a stored hash in constant time.

    Parameters: plaintext - the key presented by the caller.
                 stored_hash - the SHA-256 hex hash on file for that tenant.
    Returns: True if they match.
    Why: `==` on strings short-circuits on the first differing byte, which
    leaks timing information an attacker can use to guess a key character by
    character; `hmac.compare_digest` does not.
    """
    return hmac.compare_digest(hash_api_key(plaintext), stored_hash)


def key_prefix(plaintext: str) -> str:
    """
    Return a short, non-secret prefix of an API key for display/lookup.

    Parameters: plaintext - the full key.
    Returns: the first 12 characters (e.g. "cdn_AbCdEfGh"), used in admin UIs
    and the ledger so a key can be identified without ever storing or logging
    the full value.
    """
    return plaintext[:12]


# --- Section: admin auth — bearer header only, never query string ---
def verify_admin_key(provided: str | None, configured: str | None) -> bool:
    """
    Verify an admin bearer token in constant time.

    Parameters: provided - the value from the Authorization header (or None).
                 configured - the deployment's CORDON_ADMIN_KEY (or None).
    Returns: True only if both are set and match.
    Why: if CORDON_ADMIN_KEY is unset, /admin must always stay disabled -
    returning False here (never True) when `configured` is falsy enforces
    that regardless of what the caller sends.
    """
    if not configured or not provided:
        return False
    return hmac.compare_digest(provided, configured)


# --- Section: auth-failure throttling — per client IP, in-memory (single-process) ---
@dataclass
class _ThrottleState:
    """Per-IP failure timestamps, pruned to the configured window on each check."""

    failures: list[float] = field(default_factory=list)
    blocked_until: float = 0.0


MAX_TRACKED_IPS = 10_000  # above this, stale per-IP throttle entries are pruned


class AuthFailureThrottle:
    """
    Tracks authentication failures per client IP and throttles repeat offenders.

    Why in-memory: Cordon runs a single uvicorn worker, so
    per-process state is correct here; a multi-instance deployment needs
    Redis (documented in README.md).
    """

    def __init__(self, fail_limit: int, fail_window_seconds: float, throttle_seconds: float) -> None:
        """
        Parameters: fail_limit - failures allowed within the window before throttling.
                     fail_window_seconds - the sliding window length.
                     throttle_seconds - how long a throttled IP stays blocked.
        """
        self._fail_limit = fail_limit
        self._fail_window = fail_window_seconds
        self._throttle_seconds = throttle_seconds
        self._state: dict[str, _ThrottleState] = {}
        self._lock = Lock()

    def is_throttled(self, client_ip: str) -> bool:
        """
        Check whether a client IP is currently throttled.

        Parameters: client_ip - the caller's IP address.
        Returns: True if the IP is within its throttle period.
        """
        with self._lock:
            state = self._state.get(client_ip)
            if state is None:
                return False
            return time.monotonic() < state.blocked_until

    def record_failure(self, client_ip: str) -> bool:
        """
        Record an auth failure for a client IP and decide whether to throttle it.

        Parameters: client_ip - the caller's IP address.
        Returns: True if this failure pushed the IP over the limit (now throttled).
        Why: more than auth_fail_limit failures per fail_window_seconds per IP
        triggers a 429 for throttle_seconds.
        """
        now = time.monotonic()
        with self._lock:
            if len(self._state) > MAX_TRACKED_IPS:
                self._prune(now)
            state = self._state.setdefault(client_ip, _ThrottleState())
            state.failures = [t for t in state.failures if now - t < self._fail_window]
            state.failures.append(now)
            if len(state.failures) > self._fail_limit:
                state.blocked_until = now + self._throttle_seconds
                return True
            return False

    def _prune(self, now: float) -> None:
        """Forget IPs that are neither blocked nor have a failure inside the window (caller holds the lock)."""
        for ip in [ip for ip, s in self._state.items()
                   if now >= s.blocked_until and not any(now - t < self._fail_window for t in s.failures)]:
            del self._state[ip]

    def record_success(self, client_ip: str) -> None:
        """
        Clear failure history for a client IP after a successful auth.

        Parameters: client_ip - the caller's IP address.
        Returns: None.
        """
        with self._lock:
            self._state.pop(client_ip, None)


# --- Section: request parameter allowlist — what is ever forwarded to LiteLLM ---
# Any field not in this set is rejected with 400. This is what stops a client
# from exfiltrating Cordon's own provider keys by setting api_base/base_url to
# point at a host they control, or from faking a response via mock_response.
ALLOWED_REQUEST_PARAMS: frozenset[str] = frozenset(
    {
        "model",
        "messages",
        "temperature",
        "top_p",
        "max_tokens",
        "max_completion_tokens",
        "stream",
        "stream_options",
        "tools",
        "tool_choice",
        "response_format",
        "stop",
        "seed",
        "n",
        "presence_penalty",
        "frequency_penalty",
        "user",
    }
)


class ParameterNotAllowedError(ValueError):
    """Raised when a request body contains a field outside ALLOWED_REQUEST_PARAMS."""


class InvalidParameterValueError(ValueError):
    """Raised when an allowed field has a value Cordon refuses (e.g. n != 1)."""


def validate_request_params(body: dict) -> dict:
    """
    Enforce the request-parameter allowlist and the shape of every allowed parameter.

    Parameters: body - the parsed JSON request body.
    Returns: the same dict, if every field is allowed and well-formed.
    Raises: ParameterNotAllowedError naming the first disallowed field;
    InvalidParameterValueError for the first value of the wrong shape (see PARAM_RULES).
    Why: this is the single choke point that prevents parameter injection
    (api_base, custom_llm_provider, mock_response, ...) from ever reaching
    LiteLLM - and a malformed request is refused here with a clear 400,
    instead of being sent to every provider on the ladder and failing there.
    """
    disallowed = sorted(set(body.keys()) - ALLOWED_REQUEST_PARAMS)
    if disallowed:
        raise ParameterNotAllowedError(f"parameter not allowed: {disallowed[0]}")
    for name, value in body.items():
        rule = PARAM_RULES.get(name)
        if value is not None and rule is not None:
            problem = rule(value)
            if problem:
                raise InvalidParameterValueError(f"parameter '{name}' {problem}")
    return body


MESSAGE_ROLES = frozenset({"system", "developer", "user", "assistant", "tool", "function"})
TOOL_CHOICES = frozenset({"none", "auto", "required"})


def _number(low: float, high: float, integer: bool = False):
    """A rule: a number (an integer if required) within [low, high]."""
    kind = "an integer" if integer else "a number"

    def rule(value) -> str | None:
        """None if value fits, else what it must be."""
        ok = isinstance(value, (int, float)) and not isinstance(value, bool) and (not integer or isinstance(value, int))
        return None if ok and low <= value <= high else f"must be {kind} between {low:g} and {high:g}"
    return rule


# OpenAI's message part types, each with the field it must carry and that field's type.
PART_FIELDS = {"text": ("text", str), "image_url": ("image_url", dict), "input_audio": ("input_audio", dict),
               "file": ("file", dict), "refusal": ("refusal", str)}


def _part_ok(part) -> bool:
    """True for a known message part type that carries its field with the right type."""
    if not isinstance(part, dict) or part.get("type") not in PART_FIELDS:
        return False
    field, kind = PART_FIELDS[part["type"]]
    return isinstance(part.get(field), kind)


def _message_problem(message) -> str | None:
    """What is wrong with one chat message, or None."""
    if not isinstance(message, dict) or message.get("role") not in MESSAGE_ROLES:
        return "must hold messages with a role (system, developer, user, assistant, tool or function)"
    content, role = message.get("content"), message["role"]
    if content is not None and not isinstance(content, str) and not (isinstance(content, list) and all(map(_part_ok, content))):
        return f"content must be text or a list of parts of a known type ({', '.join(PART_FIELDS)})"
    # As in OpenAI's API: only an assistant message (one that calls tools) or a legacy function result
    # may leave out its content, and only an assistant message calls tools. Anything else would fail at
    # the provider instead. A null field is an absent one: the SDK's message.model_dump() writes them.
    if content is None and role not in ("assistant", "function"):
        return f"a {role} message needs content"
    tool_calls = message.get("tool_calls")
    if tool_calls is not None and (role != "assistant" or not isinstance(tool_calls, list)
                                   or not all(isinstance(call, dict) for call in tool_calls)):
        return "tool_calls must be a list, on an assistant message"
    return None


def _messages(value) -> str | None:
    """A rule: a list of well-formed chat messages."""
    if not isinstance(value, list):
        return "must be a list of messages"
    return next((problem for problem in map(_message_problem, value) if problem), None)


def _tool_ok(tool) -> bool:
    """A function tool with a named function, or another typed tool (e.g. "custom") with a name."""
    if not isinstance(tool, dict) or not isinstance(tool.get("type"), str):
        return False
    if tool["type"] == "function":
        return isinstance(tool.get("function"), dict) and isinstance(tool["function"].get("name"), str)
    return isinstance(tool.get("name") or (tool.get(tool["type"]) or {}).get("name"), str)


def _tools(value) -> str | None:
    """A rule: a list of typed tools, each with a name."""
    return None if isinstance(value, list) and all(map(_tool_ok, value)) else "must be a list of typed tools, each with a name"


def _stop(value) -> str | None:
    """A rule: a string, or a list of at most 4 strings."""
    ok = isinstance(value, str) or (isinstance(value, list) and len(value) <= 4 and all(isinstance(s, str) for s in value))
    return None if ok else "must be a string or a list of at most 4 strings"


# One rule per allowed parameter: returns None when the value is fine, else what it must be.
PARAM_RULES = {
    "model": lambda v: None if isinstance(v, str) else "must be a model name",
    "messages": _messages,
    "max_tokens": _number(1, 1_000_000, integer=True),
    "max_completion_tokens": _number(1, 1_000_000, integer=True),
    "temperature": _number(0, 2),
    "top_p": _number(0, 1),
    "presence_penalty": _number(-2, 2),
    "frequency_penalty": _number(-2, 2),
    "seed": _number(-(2**63), 2**63 - 1, integer=True),
    "n": lambda v: None if v == 1 and not isinstance(v, bool) else "must be 1",
    "stream": lambda v: None if isinstance(v, bool) else "must be true or false",
    "stream_options": lambda v: None if isinstance(v, dict) else "must be an object",
    "tools": _tools,
    "tool_choice": lambda v: None if (isinstance(v, str) and v in TOOL_CHOICES) or isinstance(v, dict)
    else "must be none, auto, required or an object",
    "response_format": lambda v: None if isinstance(v, dict) and isinstance(v.get("type"), str) else "must be an object with a type",
    "stop": _stop,
    "user": lambda v: None if isinstance(v, str) and len(v) <= 256 else "must be text of at most 256 characters",
}


# --- Section: keyed hashes — HMAC-SHA256 over tenant_id + value, salted ---
def keyed_hash(tenant_id: str, value: str, salt: str) -> str:
    """
    Compute a tenant-scoped, salted HMAC-SHA256 hash.

    Parameters: tenant_id - the tenant this value belongs to.
                 value - the raw value to hash (e.g. a session ID).
                 salt - CORDON_HASH_SALT.
    Returns: the hex digest.
    Why: session IDs and prompt fingerprints must not be guessable or
    collide across tenants; keying the HMAC on tenant_id achieves both -
    the same raw session ID from two different tenants hashes differently.
    """
    message = f"{tenant_id}:{value}".encode("utf-8")
    return hmac.new(salt.encode("utf-8"), message, hashlib.sha256).hexdigest()


# --- Section: security headers — applied to every response ---
SECURITY_HEADERS: dict[str, str] = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
}

# API and admin responses must never be cached by an intermediary.
NO_STORE_HEADER: dict[str, str] = {"Cache-Control": "no-store"}

# The dashboard is the only HTML page Cordon serves; it gets a strict CSP on
# top of the headers above. No inline scripts/styles, no framing, no
# cross-origin fetches beyond self.
DASHBOARD_CSP: str = (
    "default-src 'self'; script-src 'self'; style-src 'self'; "
    "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'"
)


def security_headers(dashboard: bool = False) -> dict[str, str]:
    """
    Build the security headers for a response.

    Parameters: dashboard - True for the /dashboard HTML response, which also
    gets the Content-Security-Policy header.
    Returns: a dict of header name -> value, merging SECURITY_HEADERS,
    NO_STORE_HEADER, and (if dashboard) DASHBOARD_CSP.
    """
    headers = dict(SECURITY_HEADERS)
    headers.update(NO_STORE_HEADER)
    if dashboard:
        headers["Content-Security-Policy"] = DASHBOARD_CSP
    return headers


# --- Section: error sanitizing — redacted and truncated before log/store/return ---
ERROR_DETAIL_MAX_CHARS = 200


def sanitize_error(text: str) -> str:
    """
    Redact and truncate error text before it is logged, stored or returned.

    Parameters: text - the raw provider or internal error message.
    Returns: the redacted text, truncated to ERROR_DETAIL_MAX_CHARS.
    Why: provider errors sometimes echo request content back (e.g. "invalid
    value for 'messages[0].content': <the prompt>"); this is the choke point
    that stops that content from reaching the ledger, the logs, or the caller.
    """
    if not text:
        return text
    return redact(text)[:ERROR_DETAIL_MAX_CHARS]


# --- Section: log redaction filter — wraps the root logger for the process lifetime ---
class RedactingLogFilter(logging.Filter):
    """
    A logging.Filter that redacts PII, secret patterns, and deployment secrets
    from every log record's formatted message.
    """

    def __init__(self, deployment_secrets: list[str]) -> None:
        """
        Parameters: deployment_secrets - the actual secret values in use
        (CORDON_ADMIN_KEY, CORDON_DATA_KEY, CORDON_HASH_SALT) to scrub
        literally, in addition to guardrails.redact()'s pattern-based
        scrubbing - a deployment secret need not match a generic "sk-"/"cdn_"
        shape.
        """
        super().__init__()
        self._deployment_secrets = [s for s in deployment_secrets if s]

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Redact a log record in place before it reaches any handler.

        Parameters: record - the log record being emitted.
        Returns: True always (never drops a record, only rewrites it).
        Why scrub msg and args separately instead of pre-rendering the
        message: some loggers (uvicorn's access logger) attach a `%`-style
        format string to `record.msg` and rely on `record.args` staying a
        specific-shaped tuple (client_addr, method, path, http_version,
        status_code) for their own custom formatMessage(). Collapsing
        everything into a fully rendered `record.msg` with `record.args = ()`
        breaks that formatter. Scrubbing each piece in place keeps the
        original structure intact for every formatter while still redacting
        every string that will end up in the final output.
        """
        try:
            if isinstance(record.msg, str):
                record.msg = self._scrub(record.msg)
            if record.args:
                if isinstance(record.args, dict):
                    record.args = {k: self._scrub(v) if isinstance(v, str) else v for k, v in record.args.items()}
                else:
                    args_tuple = record.args if isinstance(record.args, tuple) else (record.args,)
                    record.args = tuple(self._scrub(a) if isinstance(a, str) else a for a in args_tuple)
        except Exception:
            # Redaction must never crash logging itself; worst case, a record
            # goes out unscrubbed rather than taking the process down.
            logger.warning("log redaction filter failed on a record", exc_info=True)
        return True

    def _scrub(self, text: str) -> str:
        """Run one string through guardrails.redact() and then literal deployment-secret scrubbing."""
        scrubbed = redact(text)
        for secret in self._deployment_secrets:
            if secret and secret in scrubbed:
                scrubbed = scrubbed.replace(secret, "[REDACTED_SECRET]")
        return scrubbed


def install_log_redaction(deployment_secrets: list[str]) -> RedactingLogFilter:
    """
    Attach a RedactingLogFilter to the root logger and the uvicorn/LiteLLM loggers.

    Parameters: deployment_secrets - see RedactingLogFilter.__init__.
    Returns: the installed filter (kept so settings.py can hold a reference).
    Why: a logging.Filter attached only to a Logger object is checked just
    once, for the logger that originally created the record - a record from
    "uvicorn.access" that merely propagates up to the root logger's handler
    never re-runs the root logger's own filters. The redaction must therefore
    sit on every HANDLER in the hierarchy as well as on the loggers
    themselves, so it catches records regardless of where they were created
    or which handler finally emits them.
    """
    log_filter = RedactingLogFilter(deployment_secrets)
    # LiteLLM's own logger is named "LiteLLM" (not lowercase "litellm") in the
    # installed version; both spellings are targeted defensively in case that
    # changes across versions.
    for name in (None, "uvicorn", "uvicorn.access", "uvicorn.error", "litellm", "LiteLLM"):
        target_logger = logging.getLogger(name)
        target_logger.addFilter(log_filter)
        for handler in target_logger.handlers:
            handler.addFilter(log_filter)
    return log_filter


# --- Section: encryption at rest — AES-GCM, tenant-scoped via associated data ---
class CacheEncryption:
    """
    AES-GCM encryption for cache values, with a random nonce per record and
    the tenant ID as associated data so ciphertexts cannot be moved between
    tenants.
    """

    NONCE_BYTES = 12

    def __init__(self, data_key: bytes) -> None:
        """
        Parameters: data_key - exactly 32 bytes (CORDON_DATA_KEY, base64-decoded).
        Raises: ValueError if data_key is not 32 bytes.
        """
        if len(data_key) != 32:
            raise ValueError("CORDON_DATA_KEY must decode to exactly 32 bytes")
        self._aesgcm = AESGCM(data_key)

    def encrypt(self, tenant_id: str, plaintext: bytes) -> bytes:
        """
        Encrypt a value for one tenant.

        Parameters: tenant_id - used as associated data (not secret, but authenticated).
                     plaintext - the bytes to encrypt.
        Returns: nonce || ciphertext, suitable for storing as one blob.
        Why: binding tenant_id as AAD means decrypting with a different
        tenant_id fails - an attacker who copies a ciphertext row between
        tenants cannot read it even with the correct data key.
        """
        nonce = os.urandom(self.NONCE_BYTES)
        ciphertext = self._aesgcm.encrypt(nonce, plaintext, tenant_id.encode("utf-8"))
        return nonce + ciphertext

    def decrypt(self, tenant_id: str, blob: bytes) -> bytes:
        """
        Decrypt a value for one tenant.

        Parameters: tenant_id - must match the tenant_id used at encryption time.
                     blob - the nonce || ciphertext produced by encrypt().
        Returns: the original plaintext bytes.
        Raises: cryptography.exceptions.InvalidTag if tenant_id or the blob
        was tampered with or does not belong to this tenant.
        """
        nonce, ciphertext = blob[: self.NONCE_BYTES], blob[self.NONCE_BYTES :]
        return self._aesgcm.decrypt(nonce, ciphertext, tenant_id.encode("utf-8"))


def decode_data_key(value: str) -> bytes:
    """
    Decode CORDON_DATA_KEY from its base64 env-var representation.

    Parameters: value - the base64 string from the environment.
    Returns: the decoded bytes.
    Raises: ValueError if decoding fails or the result is not 32 bytes.
    """
    try:
        decoded = base64.b64decode(value, validate=True)
    except Exception as exc:
        raise ValueError("CORDON_DATA_KEY is not valid base64") from exc
    if len(decoded) != 32:
        raise ValueError("CORDON_DATA_KEY must decode to exactly 32 bytes")
    return decoded


# --- Section: model integrity — verify before joblib.load, pickles can execute code ---
def file_sha256(path: str) -> str:
    """
    Compute the SHA-256 hex digest of a file.

    Parameters: path - filesystem path to the file.
    Returns: the hex digest.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model_integrity(path: str, expected_sha256: str) -> bool:
    """
    Verify a model file's SHA-256 before it is ever unpickled.

    Parameters: path - filesystem path to src/models/model.pkl.
                 expected_sha256 - the hash recorded in config.yaml (retention.model_sha256).
    Returns: True if the file's hash matches.
    Why: joblib.load() on a pickle file can execute arbitrary code; a hash
    mismatch (a swapped or corrupted file) must be caught before that call,
    not after.
    """
    return hmac.compare_digest(file_sha256(path), expected_sha256)
