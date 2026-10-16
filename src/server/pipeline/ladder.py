"""
The model ladder: try the models in order until one answers. Used by step 7 and by streaming.

- A provider that fails (down, timeout, rate limit, server error): try the next model.
- A provider that rejects the request itself (400/422 - a malformed request):
  stop at once with `RequestRejected` - every other model would reject it too.
- A rejection that only concerns this model (too long for its context, an
  unsupported parameter, its content policy): try the next model. If every model
  rejects it that way, no allowed model can take the request: `RequestRejected`.
"""

import logging
import time
from dataclasses import dataclass

import litellm

from src.protection import security
from src.server.pipeline.context import RequestContext

logger = logging.getLogger("cordon.pipeline")

# The provider says the request itself is wrong (not that the provider is unwell)...
REJECTIONS = (litellm.BadRequestError, litellm.UnprocessableEntityError)
# ...except when it is only wrong *for this model*: too long for its context, a parameter it does
# not support, or its own content policy. Another model on the ladder may well accept it.
MODEL_SPECIFIC = (litellm.ContextWindowExceededError, litellm.UnsupportedParamsError, litellm.ContentPolicyViolationError)


class RequestRejected(Exception):
    """The provider rejected the request itself; carries the sanitized reason for the caller."""


class LadderFailed(Exception):
    """Every model on the ladder failed; carries the last sanitized error."""


@dataclass
class Answered:
    """The first model that answered: its alias, LiteLLM name and the raw response (or stream)."""

    alias: str
    litellm_name: str
    response: object


def call_ladder(ctx: RequestContext, aliases: list[str] | None = None, **extra) -> Answered:
    """Call each model on `aliases` (default ctx.ladder) with ctx.completion_kwargs (plus `extra`) until one answers."""
    last_error, only_model_rejections = None, True
    for alias in ctx.ladder if aliases is None else aliases:
        kwargs = {**ctx.completion_kwargs, **extra, "model": ctx.model_aliases[alias]}
        if alias in ctx.confidence_models:
            kwargs["logprobs"] = True  # only models known to support it (config.yaml models.logprobs_models)
        try:
            if ctx.mock_fail_step == "llm_call":
                raise RuntimeError("simulated provider failure (X-Mock-Fail-Step)")
            if ctx.mock_latency_s:
                time.sleep(ctx.mock_latency_s)  # mock mode only: a real provider's waiting time
            return Answered(alias, kwargs["model"], litellm.completion(**kwargs))
        except MODEL_SPECIFIC as exc:
            last_error = security.sanitize_error(str(exc))
            logger.warning("%s cannot take this request, trying the next model: %s", alias, last_error)
        except REJECTIONS as exc:
            raise RequestRejected(security.sanitize_error(str(exc))) from None
        except Exception as exc:
            last_error, only_model_rejections = security.sanitize_error(str(exc)), False
            logger.warning("provider call failed for %s, failing over if a candidate remains: %s", alias, last_error)
    if only_model_rejections and last_error is not None:
        raise RequestRejected(last_error)  # e.g. a parameter none of the allowed models supports: the request, not a provider
    raise LadderFailed(last_error)
