from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from soc_llm_policy.io import read_yaml_list, require_exists

_SHA256_HEX_LENGTH = 64


class ModelProfile(BaseModel):
    name: str = Field(..., min_length=1)
    deployment: str = Field(..., min_length=1)
    provider: str = "azure_openai"
    billing_mode: Literal["metered", "local"] = "metered"
    model: str = ""
    model_digest: str = ""
    api_version: str = ""
    azure_token_parameter: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"
    thinking: bool | None = None
    response_format: Literal["text", "json"] = "text"
    enabled: bool = True
    prompt_price_per_1k_usd: float = Field(default=0.0, ge=0.0)
    completion_price_per_1k_usd: float = Field(default=0.0, ge=0.0)
    pricing_source_url: str = ""
    pricing_verified_on: date | None = None
    pricing_valid_through: date | None = None
    planning_prompt_tokens: int | None = Field(default=None, ge=1)
    planning_completion_tokens: int | None = Field(default=None, ge=1)


def normalize_provider(provider: str) -> str:
    normalized = provider.strip().lower() or "azure_openai"
    allowed = {"azure_openai", "openai", "anthropic", "ollama", "fixture"}
    if normalized not in allowed:
        raise ValueError(
            f"Invalid provider {provider!r}. Expected one of: "
            + ", ".join(sorted(allowed))
        )
    return normalized


def validate_pricing_metadata(
    profile: ModelProfile,
    *,
    as_of: date | None = None,
) -> None:
    if profile.billing_mode == "local":
        if profile.provider != "ollama":
            raise ValueError(
                f"Local billing mode requires the Ollama provider for {profile.name}."
            )
        if profile.prompt_price_per_1k_usd != 0:
            raise ValueError(f"Local prompt price must be zero for {profile.name}.")
        if profile.completion_price_per_1k_usd != 0:
            raise ValueError(f"Local completion price must be zero for {profile.name}.")
        if len(profile.model_digest) != _SHA256_HEX_LENGTH or any(
            character not in "0123456789abcdef" for character in profile.model_digest
        ):
            raise ValueError(f"Local model digest is invalid for {profile.name}.")
        return
    effective_date = as_of or date.today()
    if profile.prompt_price_per_1k_usd <= 0:
        raise ValueError(f"Prompt price is missing or zero for {profile.name}.")
    if profile.completion_price_per_1k_usd <= 0:
        raise ValueError(f"Completion price is missing or zero for {profile.name}.")
    if not profile.pricing_source_url.strip():
        raise ValueError(f"Pricing source URL is missing for {profile.name}.")
    if profile.pricing_verified_on is None:
        raise ValueError(f"Pricing verification date is missing for {profile.name}.")
    if profile.pricing_verified_on > effective_date:
        raise ValueError(
            f"Pricing verification date is in the future for {profile.name}."
        )
    if (
        profile.pricing_valid_through is not None
        and profile.pricing_valid_through < effective_date
    ):
        raise ValueError(f"Frozen pricing has expired for {profile.name}.")


def model_environment(profile: ModelProfile) -> dict[str, str]:
    provider = normalize_provider(profile.provider)
    deployment = profile.deployment.strip()
    if not deployment:
        raise ValueError(f"Model profile {profile.name!r} has empty deployment.")
    values = {
        "SOC_LLM_PROVIDER": provider,
        "SOC_LLM_BILLING_MODE": profile.billing_mode,
        "SOC_LLM_PROMPT_PRICE_PER_1K_USD": str(profile.prompt_price_per_1k_usd),
        "SOC_LLM_COMPLETION_PRICE_PER_1K_USD": str(
            profile.completion_price_per_1k_usd
        ),
    }
    if profile.planning_prompt_tokens is not None:
        values["SOC_LLM_PLANNING_PROMPT_TOKENS"] = str(
            profile.planning_prompt_tokens
        )
    if provider == "azure_openai":
        values["AZURE_OPENAI_DEPLOYMENT"] = deployment
        values["AZURE_OPENAI_TOKEN_PARAMETER"] = profile.azure_token_parameter
        if profile.api_version:
            values["AZURE_OPENAI_API_VERSION"] = profile.api_version
    elif provider == "openai":
        values["OPENAI_MODEL"] = deployment
    elif provider == "anthropic":
        values["ANTHROPIC_MODEL"] = deployment
        if profile.api_version:
            values["ANTHROPIC_API_VERSION"] = profile.api_version
    elif provider == "ollama":
        values["OLLAMA_MODEL"] = deployment
        values["OLLAMA_MODEL_DIGEST"] = profile.model_digest
        if profile.thinking is not None:
            values["OLLAMA_THINK"] = "true" if profile.thinking else "false"
        values["OLLAMA_RESPONSE_FORMAT"] = profile.response_format
    else:
        values["SOC_LLM_FIXTURE_DEPLOYMENT"] = deployment
    return values


def load_model_registry(path: Path) -> list[ModelProfile]:
    require_exists(path)
    raw = read_yaml_list(path)
    return [ModelProfile.model_validate(item) for item in raw]


def select_model_profiles(
    profiles: list[ModelProfile],
    selectors: list[str] | None,
) -> list[ModelProfile]:
    if not selectors:
        enabled = [profile for profile in profiles if profile.enabled]
        if not enabled:
            raise ValueError("No enabled model in the configured model registry.")
        return enabled

    selected: list[ModelProfile] = []
    for selector in selectors:
        matched = next(
            (
                profile
                for profile in profiles
                if selector in (profile.name, profile.deployment)
            ),
            None,
        )
        if matched is not None:
            selected.append(matched)
            continue
        raise ValueError(
            f"Model selector {selector!r} is not declared in the registry. "
            "Add an explicit model profile instead of using an implicit provider."
        )
    return selected
