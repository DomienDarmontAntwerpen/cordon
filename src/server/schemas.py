"""
Every JSON body the API accepts or returns, as Pydantic models: errors in
OpenAI's format, tenants, and API keys.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


# --- Section: errors ---
class ErrorDetail(BaseModel):
    """OpenAI-format error detail. message - human-readable, never echoes request content."""

    message: str
    type: str
    param: str | None = None
    code: str | None = None


class ErrorResponse(BaseModel):
    """OpenAI-format error envelope, used for every non-2xx response from Cordon."""

    error: ErrorDetail


# --- Section: tenants ---
class TenantSettingsPayload(BaseModel):
    """
    All tenant settings, every field optional. Used for both tenant creation
    (merged onto config.yaml's tenant_defaults) and PATCH (merged onto the
    tenant's current settings). See config.yaml's tenant_defaults for the
    meaning of each field.
    """

    mode: Literal["observe", "enforce"] | None = None
    pool: list[str] | None = None
    strong: str | None = None
    features: list[str] | None = None
    allowed_providers: list[str] | None = None
    pii_action: Literal["off", "mask", "block"] | None = None
    injection_action: Literal["flag", "block"] | None = None
    output_pii_action: Literal["off", "redact", "block"] | None = None
    output_leak_action: Literal["off", "redact", "block"] | None = None
    leak_min_words: int | None = Field(default=None, strict=True, ge=1)
    security_fail_mode: Literal["open", "closed"] | None = None
    telemetry_content: Literal["none", "redacted"] | None = None
    rpm: int | None = Field(default=None, strict=True, ge=0)
    tpm: int | None = Field(default=None, strict=True, ge=0)
    budget_daily_eur: float | None = Field(default=None, strict=True, allow_inf_nan=False, ge=0)
    budget_monthly_eur: float | None = Field(default=None, strict=True, allow_inf_nan=False, ge=0)
    budget_action: Literal["block", "downgrade"] | None = None
    max_tokens_default: int | None = Field(default=None, strict=True, ge=1)
    max_input_tokens: int | None = Field(default=None, strict=True, ge=0)
    max_images: int | None = Field(default=None, strict=True, ge=0)
    image_max_side: int | None = Field(default=None, strict=True, ge=1)
    image_detail_low_for_photos: bool | None = None
    cache_max_temperature: float | None = Field(default=None, strict=True, allow_inf_nan=False, ge=0, le=2)
    mask_instructions: bool | None = None
    holdout_pct: float | None = Field(default=None, strict=True, allow_inf_nan=False, ge=0, le=1)


class TenantCreateRequest(TenantSettingsPayload):
    """A new tenant's name, optional explicit ID, and any setting overrides."""

    name: str
    tenant_id: str | None = None


class TenantResponse(BaseModel):
    """A tenant as returned by the admin API - settings are never secret."""

    id: str
    name: str
    settings: dict[str, Any]
    created_at: str


# --- Section: API keys ---
class ApiKeyCreateRequest(BaseModel):
    """Optional expiry for a newly issued API key."""

    expires_at: str | None = None


class ApiKeyCreateResponse(BaseModel):
    """A newly issued API key. `key` is shown exactly once and never stored."""

    key: str
    prefix: str
    tenant_id: str
    expires_at: str | None
