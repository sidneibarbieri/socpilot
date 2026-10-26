# src/soc_llm_policy/interfaces.py
"""
Formal system interfaces using typing.Protocol.

Protocol defines structural contracts (static duck typing):
any class implementing the required methods satisfies
the interface without inheritance or tight coupling.

Direct benefit: pipeline.py depends on LLMPort, not LLMAdapter.
Swapping Azure for OpenAI, Anthropic, or a test mock
requires zero changes in pipeline.py.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from soc_llm_policy.io import (
    ActionCatalogItem,
    IncidentMeta,
    PolicyRule,
    TelemetryEvent,
)
from soc_llm_policy.result_models import LLMUsage


@dataclass(frozen=True)
class LLMRecommendation:
    actions: list[str]
    reasoning: str
    hallucinated_actions: list[str] | None = None
    usage: LLMUsage | None = None
    latency_ms: int | None = None
    estimated_cost_usd: float | None = None
    prompt_messages: list[dict[str, str]] | None = None
    prompt_sha256: str | None = None
    returned_model: str | None = None
    # True when the provider cut the output at the token ceiling; such a run is
    # truncation, not a model decision, and must be excluded before trusting it.
    truncated: bool = False


@runtime_checkable
class LLMPort(Protocol):
    """Contract for any LLM adapter implementation.

    pipeline.py depends on this interface, not concrete implementations.
    This keeps coupling low: adding a new model backend
    means implementing this contract without modifying the pipeline.
    """

    @property
    def deployment(self) -> str:
        """Model/deployment identifier for traceability."""
        ...

    def recommend(
        self,
        meta: IncidentMeta,
        telemetry: list[TelemetryEvent],
        catalog: list[ActionCatalogItem],
        *,
        policy_rules: list[PolicyRule] | None = None,
        policy_prompt_mode: Literal["none", "inline_constraints"] = "none",
    ) -> LLMRecommendation:
        """Recommend response actions given incident context.

        Args:
            meta: incident metadata (incident_meta.json).
            telemetry: normalized events (incident_telemetry.jsonl).
            catalog: available actions (action_catalog.yaml).

        Returns:
            LLMRecommendation with actions, reasoning, and usage/cost metadata.
        """
        ...


type LLMFactory = Callable[[], LLMPort]
