from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from soc_llm_policy.io import PolicyRule


def operational_rules(rules: Iterable[PolicyRule]) -> list[PolicyRule]:
    """Return rules supported for measurement on the incident corpus."""
    return [rule for rule in rules if rule.evidence_tier == "operational"]


def operational_rule_records(
    records: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Filter unparsed policy records for prompts and model-judge inputs."""
    return [
        dict(record)
        for record in records
        if record.get("evidence_tier", "operational") == "operational"
    ]
