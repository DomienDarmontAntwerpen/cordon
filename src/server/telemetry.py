"""
Optional Langfuse tracing, in a background thread so it never slows a request.
Metadata only by default; content only when a tenant opts in, and redacted again.
A telemetry failure never affects the request.
"""

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from src.protection.guardrails import redact

logger = logging.getLogger("cordon.telemetry")


class Telemetry:
    """
    Background-only Langfuse telemetry sender. Safe to construct even without
    any Langfuse keys (it then does nothing on every call).
    """

    def __init__(self, langfuse_public_key: str | None, langfuse_secret_key: str | None, langfuse_host: str) -> None:
        """
        Parameters: langfuse_public_key, langfuse_secret_key - from Settings;
        both must be set for telemetry to be enabled.
                     langfuse_host - the Langfuse Cloud host (EU region by default).
        """
        self._enabled = bool(langfuse_public_key and langfuse_secret_key)
        self._client = None
        if self._enabled:
            try:
                from langfuse import Langfuse

                self._client = Langfuse(
                    public_key=langfuse_public_key, secret_key=langfuse_secret_key, host=langfuse_host
                )
            except Exception:
                logger.exception("failed to initialize Langfuse client; telemetry disabled for this process")
                self._enabled = False
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="cordon-telemetry")

    @property
    def enabled(self) -> bool:
        """True if a Langfuse client was constructed (both keys were present and init succeeded)."""
        return self._enabled

    def record_request(
        self,
        tenant_id: str,
        telemetry_content_mode: str,
        metadata: dict[str, Any],
        input_text: str | None = None,
        output_text: str | None = None,
    ) -> Future | None:
        """
        Submit one request's telemetry in the background.

        Parameters: tenant_id - the owning tenant.
                     telemetry_content_mode - "none" or "redacted" (tenant setting).
                     metadata - model, tokens, cost, route, cache, fallback,
                     quality score, guardrail counts - never prompt/answer text.
                     input_text, output_text - the masked prompt/answer text,
                     only used when telemetry_content_mode == "redacted" (and
                     redacted again here before being sent).
        Returns: the submitted Future (callers ignore it; tests may wait on
        it), or None if telemetry is disabled.
        """
        if not self._enabled:
            return None
        return self._executor.submit(
            self._send, tenant_id, telemetry_content_mode, dict(metadata), input_text, output_text
        )

    def _send(
        self,
        tenant_id: str,
        mode: str,
        metadata: dict[str, Any],
        input_text: str | None,
        output_text: str | None,
    ) -> None:
        """
        Build and send one Langfuse generation. Runs on a background thread;
        never raises (any exception is logged and swallowed).
        """
        try:
            payload_input = redact(input_text) if (mode == "redacted" and input_text) else None
            payload_output = redact(output_text) if (mode == "redacted" and output_text) else None
            observation = self._client.start_observation(
                name="chat_completion",
                as_type="generation",
                input=payload_input,
                output=payload_output,
                metadata={**metadata, "tenant_id": tenant_id},
                model=metadata.get("served_model"),
            )
            observation.end()
        except Exception:
            logger.warning("telemetry send failed (ignored, request already completed)", exc_info=True)

    def flush(self) -> None:
        """Force-flush any buffered Langfuse data. Used at shutdown and by tests."""
        if self._enabled and self._client is not None:
            try:
                self._client.flush()
            except Exception:
                logger.warning("telemetry flush failed (ignored)", exc_info=True)

    def shutdown(self) -> None:
        """Flush and stop the background executor. Parameters/Returns: none."""
        self.flush()
        self._executor.shutdown(wait=False)
