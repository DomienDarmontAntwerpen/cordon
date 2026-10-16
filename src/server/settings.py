"""
Settings: .env plus config.yaml, and the startup checks.

Outside mock mode Cordon refuses to start with a missing or weak secret, a
model file that fails its hash check, or no key for the strong model's
provider. Mock mode generates throwaway secrets instead.
"""

import base64
import logging
import os
import secrets as secrets_module
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.protection import security

logger = logging.getLogger("cordon.settings")

# --- Section: constants — paths, thresholds and env-var names used by the checks below ---
from src.paths import CONFIG_PATH, MODEL_PATH

# A deployment that pastes this literal placeholder instead of generating its
# own admin key is refused exactly like a missing one would be.
EXAMPLE_ADMIN_KEY_VALUE = "changeme-please-generate-a-real-32-character-admin-key"
MIN_ADMIN_KEY_LENGTH = 32
MIN_HASH_SALT_LENGTH = 16

PROVIDER_ENV_VARS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}


class StartupError(RuntimeError):
    """Raised to refuse startup outside mock mode; the message lists every failing check."""


# --- Section: Settings — the one object every other module reads configuration from ---
@dataclass
class Settings:
    """
    Resolved configuration for one Cordon process.

    Parameters/fields: see inline comments. `config` is the raw parsed
    config.yaml; callers read tenant_defaults/thresholds/etc. from it rather
    than Settings re-exposing every nested key.
    Why: one object, constructed once at startup, avoids every module reading
    os.environ or the YAML file independently (and risking a different view
    of "is mock mode on" between modules).
    """

    mock_mode: bool
    bind: str
    port: int
    admin_key: str | None  # None means /admin is disabled, in any mode
    data_key: bytes  # decoded, exactly 32 bytes
    hash_salt: str
    config: dict
    provider_keys: dict[str, str]  # provider name -> API key, only for providers that have one
    ephemeral_secrets: bool  # True if any secret was generated for mock mode
    langfuse_public_key: str | None
    langfuse_secret_key: str | None
    langfuse_host: str

    def provider_has_key(self, provider: str) -> bool:
        """Return True if `provider` has an API key configured (always True in mock mode's eyes is NOT implied - callers that only care about mock behaviour should check mock_mode separately)."""
        return provider in self.provider_keys

    @staticmethod
    def model_provider(litellm_name: str) -> str:
        """Return the provider segment of a LiteLLM model name, e.g. 'openai/gpt-4o' -> 'openai'."""
        return litellm_name.split("/", 1)[0]


# --- Section: config.yaml loading ---
def _load_config(path: Path) -> dict:
    """Load and parse config.yaml. Parameters: path. Returns: the parsed dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _env_bool(name: str, default: bool) -> bool:
    """Parse a boolean-ish environment variable (1/true/yes/on, case-insensitive)."""
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _configure_litellm() -> None:
    """
    Force LiteLLM's pricing to the local map and silence its debug output.

    Why: LITELLM_LOCAL_MODEL_COST_MAP=True means model_cost lookups never hit
    the network, which is required for offline/deterministic tests and demo
    runs (principle 7). suppress_debug_info=True stops LiteLLM from printing
    raw prompts to stdout on certain error paths.
    """
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    import litellm

    litellm.suppress_debug_info = True


# --- Section: load_settings — the single startup entry point ---
def load_settings(env_file: str | Path | None = None, config_path: Path = CONFIG_PATH) -> Settings:
    """
    Load environment and config, apply fail-closed startup checks, and return Settings.

    Parameters: env_file - path to a .env file (None lets python-dotenv find
    the default `.env` in the current directory; tests pass an explicit path
    or none at all to stay isolated).
                 config_path - path to config.yaml (overridable in tests).
    Returns: a populated Settings object.
    Raises: StartupError if, outside mock mode, any required secret is
    missing/weak, the model file does not match its hash, or the default
    strong model's provider has no key.
    """
    _configure_litellm()
    load_dotenv(dotenv_path=env_file, override=False)
    mock_mode = _env_bool("MOCK_MODE", True)
    config = _load_config(config_path)
    provider_keys = {provider: os.environ[var].strip() for provider, var in PROVIDER_ENV_VARS.items()
                     if os.environ.get(var, "").strip()}
    admin_key = os.environ.get("CORDON_ADMIN_KEY", "").strip() or None
    data_key_b64 = os.environ.get("CORDON_DATA_KEY", "").strip() or None
    hash_salt = os.environ.get("CORDON_HASH_SALT", "").strip() or None

    ephemeral_secrets = False
    if mock_mode:
        data_key_b64, hash_salt, ephemeral_secrets = _mock_secrets(data_key_b64, hash_salt, admin_key)
    else:
        errors = _startup_errors(admin_key, data_key_b64, hash_salt, config, provider_keys)
        if errors:
            raise StartupError("Cordon refuses to start outside mock mode:\n- " + "\n- ".join(errors))

    return Settings(
        mock_mode=mock_mode,
        bind=os.environ.get("CORDON_BIND", "127.0.0.1"),
        port=int(os.environ.get("CORDON_PORT", "8787")),
        admin_key=admin_key,
        data_key=security.decode_data_key(data_key_b64),
        hash_salt=hash_salt,
        config=config,
        provider_keys=provider_keys,
        ephemeral_secrets=ephemeral_secrets,
        langfuse_public_key=os.environ.get("LANGFUSE_PUBLIC_KEY") or None,
        langfuse_secret_key=os.environ.get("LANGFUSE_SECRET_KEY") or None,
        langfuse_host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
    )


def _mock_secrets(data_key_b64: str | None, hash_salt: str | None, admin_key: str | None) -> tuple[str, str, bool]:
    """
    Mock mode makes no real provider call: missing data key and salt get throwaway values. The
    admin key never does - /admin stays disabled unless it is configured, in every mode.
    Returns (data_key_b64, hash_salt, whether any value is throwaway).
    """
    ephemeral = False
    if not data_key_b64:
        data_key_b64, ephemeral = base64.b64encode(os.urandom(32)).decode("ascii"), True
        logger.warning("MOCK_MODE: generated an ephemeral CORDON_DATA_KEY; cache contents will not survive a restart")
    if not hash_salt:
        hash_salt, ephemeral = secrets_module.token_urlsafe(24), True
        logger.warning("MOCK_MODE: generated an ephemeral CORDON_HASH_SALT")
    if not admin_key:
        logger.warning("MOCK_MODE: CORDON_ADMIN_KEY is unset; /admin stays disabled")
    return data_key_b64, hash_salt, ephemeral


def _startup_errors(admin_key, data_key_b64, hash_salt, config: dict, provider_keys: dict) -> list[str]:
    """Everything that makes a live start unsafe: weak or missing secrets, a tampered model file, no key for the strong model."""
    errors: list[str] = []
    if not admin_key or len(admin_key) < MIN_ADMIN_KEY_LENGTH or admin_key == EXAMPLE_ADMIN_KEY_VALUE:
        errors.append("CORDON_ADMIN_KEY is missing, shorter than 32 characters, or equals the example value")
    if not data_key_b64:
        errors.append("CORDON_DATA_KEY is missing")
    else:
        try:
            security.decode_data_key(data_key_b64)
        except ValueError as exc:
            errors.append(f"CORDON_DATA_KEY is invalid: {exc}")
    if not hash_salt or len(hash_salt) < MIN_HASH_SALT_LENGTH:
        errors.append("CORDON_HASH_SALT is missing or shorter than 16 characters")
    model_sha256 = (config.get("retention") or {}).get("model_sha256")
    if model_sha256 and not MODEL_PATH.exists():
        errors.append("src/models/model.pkl is missing but retention.model_sha256 is configured")
    elif model_sha256 and not security.verify_model_integrity(str(MODEL_PATH), model_sha256):
        errors.append("src/models/model.pkl does not match the configured model_sha256")
    strong_alias = config["models"]["default_strong"]
    strong_provider = Settings.model_provider(config["models"]["model_aliases"][strong_alias])
    if strong_provider not in provider_keys:
        errors.append(f"default strong model '{strong_alias}' uses provider '{strong_provider}', which has no API key configured")
    return errors


# --- Section: logging setup — attaches the redaction filter built in security.py ---
def configure_logging(settings: Settings, level: int = logging.INFO) -> security.RedactingLogFilter:
    """
    Configure root logging and install the secret/PII redaction filter.

    Parameters: settings - the loaded Settings (its secrets are what gets
    scrubbed literally, in addition to pattern-based redaction).
                 level - the root logger level.
    Returns: the installed RedactingLogFilter.
    Why: this must run before any other module logs anything, so no log line
    in the process's lifetime ever carries a secret or PII value.
    """
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    deployment_secrets = [settings.admin_key, settings.hash_salt, *settings.provider_keys.values()]
    return security.install_log_redaction(deployment_secrets)


# --- Section: model capabilities — LiteLLM detection plus config overrides ---
CAPABILITY_KEYS: tuple[str, ...] = ("vision", "json_schema", "tools")


def model_capabilities(alias: str, litellm_name: str, config: dict) -> dict[str, bool]:
    """
    Resolve the vision/json_schema/tools capabilities for one model alias.

    Parameters: alias - the client-facing alias (e.g. "deepseek-chat").
                 litellm_name - the LiteLLM model name (e.g. "deepseek/deepseek-chat").
                 config - the parsed config.yaml.
    Returns: {"vision": bool, "json_schema": bool, "tools": bool}.
    Why: LiteLLM's supports_vision/supports_response_schema/supports_function_calling
    are the source of truth, but config.yaml's models.capability_overrides can
    correct a model LiteLLM reports wrong (see config.yaml's deepseek-chat comment).
    """
    import litellm

    detected = {
        "vision": bool(litellm.supports_vision(model=litellm_name)),
        "json_schema": bool(litellm.supports_response_schema(model=litellm_name)),
        "tools": bool(litellm.supports_function_calling(model=litellm_name)),
    }
    overrides = ((config.get("models") or {}).get("capability_overrides") or {}).get(alias, {})
    detected.update({k: v for k, v in overrides.items() if k in CAPABILITY_KEYS})
    return detected


def all_model_capabilities(config: dict) -> dict[str, dict[str, bool]]:
    """
    Resolve capabilities for every configured model alias.

    Parameters: config - the parsed config.yaml.
    Returns: {alias: {"vision": bool, "json_schema": bool, "tools": bool}, ...}
    for every alias in models.model_aliases.
    """
    aliases = config["models"]["model_aliases"]
    return {alias: model_capabilities(alias, litellm_name, config) for alias, litellm_name in aliases.items()}


# --- Section: tenant settings — the secure-default starting point for every tenant ---
KNOWN_FEATURES: tuple[str, ...] = (
    "trim", "vision", "cache", "semantic_cache", "routing", "max_tokens", "circuit_breaker",
)  # "routing" includes its quality check: a route is only proven together with it


def default_tenant_settings(config: dict) -> dict:
    """
    Return a fresh copy of config.yaml's tenant_defaults.

    Parameters: config - the parsed config.yaml.
    Returns: a new dict (safe to mutate) with the secure defaults every new
    tenant starts from: the defaults are the secure choice.
    """
    import copy

    return copy.deepcopy(config["tenant_defaults"])


def merge_tenant_settings(current: dict, overrides: dict, config: dict) -> dict:
    """
    Merge caller-supplied overrides into a tenant's settings.

    Parameters: current - the tenant's existing settings dict.
                 overrides - new values for a subset of the same keys; keys
                 not present in `current` are rejected (unknown setting name).
                 config - config.yaml, for the known models and providers.
    Returns: a new merged dict (current is not mutated).
    Raises: ValueError naming the first unknown key, feature, model or provider.
    Every value is validated and typed by schemas.TenantSettingsPayload (ranges,
    strict numbers), so no caller - API or tools/admin.py - can store an unchecked one.
    Why: this is what POST/PATCH /admin/tenants and tools/admin.py go through,
    so a typo in a setting or feature name fails loudly instead of being
    silently ignored.
    """
    overrides = _checked_values(overrides)
    unknown_features = [f for f in overrides.get("features") or [] if f not in KNOWN_FEATURES]
    if unknown_features:
        raise ValueError(f"unknown feature: {unknown_features[0]} (known: {', '.join(KNOWN_FEATURES)})")
    merged = dict(current)
    for key, value in overrides.items():
        if value is not None:
            merged[key] = value
    # A team stored before a feature was retired (the separate "quality_gate") keeps working.
    if "features" in merged:
        merged["features"] = [f for f in merged["features"] or [] if f in KNOWN_FEATURES]
    _check_models_and_providers(merged, config)
    return merged


def _checked_values(overrides: dict) -> dict:
    """
    Overrides validated and typed by the API schema (ranges, strict numbers).

    Known names come from the schema, not from the team's stored settings, so
    a setting added later can be switched on for a team created before it.
    """
    from pydantic import ValidationError

    from src.server.schemas import TenantSettingsPayload

    unknown = [k for k, v in overrides.items() if v is not None and k not in TenantSettingsPayload.model_fields]
    if unknown:
        raise ValueError(f"unknown tenant setting: {unknown[0]}")
    try:
        return TenantSettingsPayload(**overrides).model_dump(exclude_none=True)
    except ValidationError as exc:
        first = exc.errors()[0]
        raise ValueError(f"invalid value for {first['loc'][0]}: {first['msg']}") from None


def _check_models_and_providers(tenant_settings: dict, config: dict) -> None:
    """Raise ValueError if `strong`, `pool` or `allowed_providers` name something config.yaml does not know."""
    aliases = config["models"]["model_aliases"]
    for alias in [tenant_settings.get("strong")] + list(tenant_settings.get("pool") or []):
        if alias is not None and alias not in aliases:
            raise ValueError(f"unknown model: {alias} (known: {', '.join(aliases)})")
    providers = config["models"]["providers"]
    for provider in tenant_settings.get("allowed_providers") or []:
        if provider not in providers:
            raise ValueError(f"unknown provider: {provider} (known: {', '.join(providers)})")


def resolve_tenant_pool(tenant_settings: dict, config: dict) -> list[str]:
    """
    Resolve a tenant's model pool to a concrete list of aliases.

    Parameters: tenant_settings - the tenant's settings dict (pool may be None).
                 config - the parsed config.yaml.
    Returns: tenant_settings["pool"] if set, else config.models.default_pool.
    """
    return tenant_settings.get("pool") or config["models"]["default_pool"]


def resolve_tenant_strong(tenant_settings: dict, config: dict) -> str:
    """
    Resolve a tenant's strong-model alias.

    Parameters: tenant_settings - the tenant's settings dict (strong may be None).
                 config - the parsed config.yaml.
    Returns: tenant_settings["strong"] if set, else config.models.default_strong.
    """
    return tenant_settings.get("strong") or config["models"]["default_strong"]
