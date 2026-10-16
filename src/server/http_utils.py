"""
Small stateless helpers shared by the routes: OpenAI-format errors, a
size-capped body reader, bearer/IP/expiry helpers, tag parsing and CSV-safe cells.
"""

import re
import time
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

from src.server.schemas import ErrorDetail, ErrorResponse


# --- Section: error responses ---
def openai_error(
    status_code: int,
    message: str,
    error_type: str,
    code: str | None = None,
    extra_headers: dict[str, str] | None = None,
) -> JSONResponse:
    """
    Build an OpenAI-format error JSONResponse.

    Parameters: status_code - HTTP status. message - safe, never echoes
    request content. error_type - OpenAI error type string. code - optional
    machine-readable code. extra_headers - e.g. X-Cordon-Error, Retry-After.
    Returns: a JSONResponse ready to return from a route.
    """
    body = ErrorResponse(error=ErrorDetail(message=message, type=error_type, code=code)).model_dump(exclude_none=True)
    return JSONResponse(status_code=status_code, content=body, headers=extra_headers or {})


# --- Section: request body ---
class BodyTooLargeError(Exception):
    """Raised by read_capped_body when the stream exceeds the configured limit."""


async def read_capped_body(request: Request, max_bytes: int) -> bytes:
    """
    Read a request body incrementally, aborting as soon as it exceeds max_bytes.

    Parameters: request - the incoming request. max_bytes - the configured
    request_limits.max_body_bytes.
    Returns: the full body bytes, if within the limit.
    Raises: BodyTooLargeError as soon as the cumulative size exceeds max_bytes
    - memory use is bounded to max_bytes plus at most one chunk, never the
    full oversized body ("size limits before any expensive work").
    """
    size = 0
    chunks: list[bytes] = []
    async for chunk in request.stream():
        size += len(chunk)
        if size > max_bytes:
            raise BodyTooLargeError()
        chunks.append(chunk)
    return b"".join(chunks)


# --- Section: authentication helpers ---
def client_ip(request: Request) -> str:
    """Return the caller's IP for throttling purposes. No proxy headers are trusted (see SECURITY.md)."""
    return request.client.host if request.client else "unknown"


def now_iso() -> str:
    """Current UTC time as ISO-8601 (matches store._now_iso's format for string comparison)."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def is_expired(expires_at: str | None) -> bool:
    """True if an ISO-8601 expiry timestamp is in the past. None never expires."""
    return expires_at is not None and expires_at < now_iso()


def extract_bearer(request: Request) -> str | None:
    """Extract a bearer token from the Authorization header only - never from a query string."""
    header = request.headers.get("authorization")
    if not header or not header.lower().startswith("bearer "):
        return None
    return header[len("bearer "):].strip()


# --- Section: tags ---
TAG_SPLIT_REGEX = re.compile(r"\s*,\s*")


def parse_tags(request: Request, request_limits: dict) -> tuple[list[str], str | None]:
    """
    Parse and validate the X-Cordon-Tags header.

    Parameters: request - the incoming request. request_limits - config.yaml's
    request_limits (max_tags, tag_regex).
    Returns: (tags, error_message). error_message is None if every tag is
    valid and within max_tags; otherwise tags is [] and error_message names
    the problem.
    """
    raw = request.headers.get("x-cordon-tags", "").strip()
    if not raw:
        return [], None
    tags = [t for t in TAG_SPLIT_REGEX.split(raw) if t]
    if len(tags) > request_limits["max_tags"]:
        return [], f"too many tags (max {request_limits['max_tags']})"
    tag_regex = re.compile(request_limits["tag_regex"])
    for tag in tags:
        if not tag_regex.fullmatch(tag):
            return [], f"invalid tag: {tag!r}"
    return tags, None


# --- Section: CSV export ---
def csv_safe(value: Any) -> Any:
    """
    Prefix a CSV cell with a leading apostrophe if it starts with =, +, -, or @.

    Parameters: value - the raw cell value.
    Returns: the value unchanged, unless it is a string starting with a
    formula-trigger character, in which case a leading "'" is prepended.
    Why: opening an untrusted CSV in a spreadsheet application can execute a
    formula starting with one of these characters (CSV/formula injection).
    """
    if isinstance(value, str) and value[:1] in ("=", "+", "-", "@"):
        return "'" + value
    return value
