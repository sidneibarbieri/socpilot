"""Deterministic LLM-port replay for zero-cost pipeline validation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from soc_llm_policy.interfaces import LLMRecommendation
from soc_llm_policy.io import (
    ActionCatalogItem,
    IncidentMeta,
    PolicyRule,
    TelemetryEvent,
)
from soc_llm_policy.result_models import LLMUsage


class FixtureResponse(BaseModel):
    incident_id: str = Field(..., min_length=1)
    policy_prompt_mode: Literal["none", "inline_constraints"]
    recommended_actions: list[str]
    reasoning: str = "Deterministic fixture response."
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    latency_ms: int = Field(default=0, ge=0)
    output_truncated: bool = False

    @model_validator(mode="after")
    def validate_unique_actions(self) -> FixtureResponse:
        if len(self.recommended_actions) != len(set(self.recommended_actions)):
            raise ValueError("Fixture actions must be unique")
        return self


class FixtureManifest(BaseModel):
    schema_version: Literal[1] = 1
    deployment: str = Field(default="fixture-replay", min_length=1)
    fixture_profile: str | None = None
    constitutes_model_evidence: Literal[False] = False
    responses: list[FixtureResponse] = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_unique_response_keys(self) -> FixtureManifest:
        keys = [
            (response.incident_id, response.policy_prompt_mode)
            for response in self.responses
        ]
        if len(keys) != len(set(keys)):
            raise ValueError(
                "Fixture responses must be unique by incident_id and "
                "policy_prompt_mode"
            )
        return self


class FixtureLLM:
    """Serve declared recommendations through the production LLM interface."""

    def __init__(self, manifest: FixtureManifest) -> None:
        self._manifest = manifest
        self._responses = {
            (response.incident_id, response.policy_prompt_mode): response
            for response in manifest.responses
        }

    @classmethod
    def from_path(cls, path: Path) -> FixtureLLM:
        payload = path.read_text(encoding="utf-8")
        return cls(FixtureManifest.model_validate_json(payload))

    @property
    def deployment(self) -> str:
        return self._manifest.deployment

    def recommend(
        self,
        meta: IncidentMeta,
        telemetry: list[TelemetryEvent],
        catalog: list[ActionCatalogItem],
        *,
        policy_rules: list[PolicyRule] | None = None,
        policy_prompt_mode: Literal["none", "inline_constraints"] = "none",
    ) -> LLMRecommendation:
        del telemetry, policy_rules
        key = (meta.incident_id, policy_prompt_mode)
        if key not in self._responses:
            raise KeyError(
                "No fixture response for "
                f"incident_id={meta.incident_id!r}, "
                f"policy_prompt_mode={policy_prompt_mode!r}"
            )
        response = self._responses[key]
        valid_actions = {item.action_id for item in catalog}
        unknown_actions = [
            action
            for action in response.recommended_actions
            if action not in valid_actions
        ]
        actions = [
            action
            for action in response.recommended_actions
            if action in valid_actions
        ]
        usage = LLMUsage(
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.prompt_tokens + response.completion_tokens,
        )
        return LLMRecommendation(
            actions=actions,
            reasoning=response.reasoning,
            hallucinated_actions=unknown_actions,
            usage=usage,
            latency_ms=response.latency_ms,
            estimated_cost_usd=0.0,
            prompt_messages=[],
            prompt_sha256=None,
            returned_model=self.deployment,
            truncated=response.output_truncated,
        )
