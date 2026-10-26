"""Proportionality policy (Oracle B).

Authored experimental policy: an action's impact must not exceed the incident's
demand. ``phase_rank`` ranks catalog phases by disruptiveness; ``demand_ceiling``
maps a demand level (the higher of severity and asset_criticality) to the maximum
phase_rank an action may reach. Both matrices are authored and pending domain
review; activation is not reported until reviewed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from soc_llm_policy.io import ActionCatalogItem, IncidentMeta

_LEVEL_RANK = {"low": 1, "medium": 2, "high": 3, "critical": 4}
_RANK_LEVEL = {rank: level for level, rank in _LEVEL_RANK.items()}


class ProportionalityPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    policy_id: str = Field(..., min_length=1)
    review_status: Literal["pending_review", "approved_for_experiment"]
    phase_rank: dict[str, int] = Field(..., min_length=1)
    demand_ceiling: dict[str, int]

    @model_validator(mode="after")
    def validate_policy_surface(self) -> ProportionalityPolicy:
        required_levels = set(_LEVEL_RANK)
        if set(self.demand_ceiling) != required_levels:
            raise ValueError(
                "demand_ceiling must define exactly low, medium, high, and critical"
            )
        if any(rank < 1 for rank in self.phase_rank.values()):
            raise ValueError("phase_rank values must be positive")
        if any(ceiling < 1 for ceiling in self.demand_ceiling.values()):
            raise ValueError("demand_ceiling values must be positive")
        ordered_ceilings = [
            self.demand_ceiling[level]
            for level in ("low", "medium", "high", "critical")
        ]
        if ordered_ceilings != sorted(ordered_ceilings):
            raise ValueError("demand_ceiling must be monotonic")
        if max(ordered_ceilings) > max(self.phase_rank.values()):
            raise ValueError("demand_ceiling exceeds the maximum phase rank")
        return self

    def require_approved(self) -> None:
        if self.review_status != "approved_for_experiment":
            raise ValueError(
                f"Proportionality policy {self.policy_id!r} is not approved "
                "for experiment execution"
            )

    def demand_level(self, meta: IncidentMeta) -> str:
        demand = max(_LEVEL_RANK[meta.severity], _LEVEL_RANK[meta.asset_criticality])
        return _RANK_LEVEL[demand]

    def ceiling_for(self, meta: IncidentMeta) -> int:
        return self.demand_ceiling[self.demand_level(meta)]

    def impact_rank(
        self, action_id: str, catalog: list[ActionCatalogItem]
    ) -> int:
        phase_by_action = {item.action_id: item.phase for item in catalog}
        if action_id not in phase_by_action:
            raise ValueError(f"Action {action_id!r} is absent from the action catalog")
        phase = phase_by_action[action_id]
        if not phase:
            raise ValueError(f"Action {action_id!r} has no declared phase")
        if phase not in self.phase_rank:
            raise ValueError(
                f"Action {action_id!r} uses unranked phase {phase!r} "
                f"under policy {self.policy_id!r}"
            )
        return self.phase_rank[phase]


def load_proportionality_policy(path: Path) -> ProportionalityPolicy:
    """Load and validate the authored proportionality policy from YAML."""
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return ProportionalityPolicy.model_validate(data)
