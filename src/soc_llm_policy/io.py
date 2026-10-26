from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class IncidentMeta(BaseModel):
    """Metadata for one security incident.

    Pipeline input contract: every incident must provide these fields
    before reaching the LLM or the policy engine.
    """

    incident_id: str = Field(..., min_length=1)
    incident_type: str = Field(..., min_length=1)
    severity: Literal["low", "medium", "high", "critical"]
    asset_criticality: Literal["low", "medium", "high", "critical"]
    asset_role: str = Field(..., min_length=1)
    time_window_start: str = Field(..., min_length=1)
    time_window_end: str = Field(..., min_length=1)
    final_status: str = "unknown"


class TelemetryDetails(BaseModel):
    """Details associated with a normalized telemetry event."""

    command: str | None = None
    severity: str | None = None
    log_source: str | None = None
    source_summary: dict[str, Any] | None = None
    raw: dict[str, Any] | None = None


class TelemetryEvent(BaseModel):
    """Canonical event consumed by the engine and the LLM adapter."""

    event_type: str = Field(..., min_length=1)
    category: str | None = None
    timestamp: str | None = None
    source_type: str | None = None
    source_ip: str | None = None
    dest_ip: str | None = None
    username: str | None = None
    details: TelemetryDetails = Field(default_factory=TelemetryDetails)


class ActionCatalogItem(BaseModel):
    """An executable action in the SOC catalog."""

    action_id: str = Field(..., min_length=1)
    requires_approval: bool = False
    reversible: bool = False
    phase: str | None = None
    operational_role: str | None = None
    baseline_support: str | None = None


class PolicyRule(BaseModel):
    """Operational SOC policy rule.

    `Literal` replaces regex pattern checks: more readable, clearer errors,
    and static validation by the type checker.
    """

    rule_id: str = Field(..., min_length=1)
    type: Literal["mandatory", "prohibit_before", "require_approval", "prohibit_when"]
    evidence_tier: Literal["operational", "specification"] = "operational"
    severity: Literal["hard", "soft", "warning"] = "hard"
    action: str = Field(..., min_length=1)
    # The operational object the governed action acts on. An annotation on the
    # rule, never a container for it: rules stay independent and are evaluated
    # over the incident context; governance_scope names what they govern.
    governance_scope: Literal["endpoint", "network", "identity"] | None = None
    scope: dict[str, Any] | None = None
    condition_action: str | None = None
    rationale: str | None = None
    evidence_basis: str | None = None
    repair_operator: str | None = None
    alternative_repair_operator: str | None = None
    official_eval_activated: bool | None = None
    approval_evidence: dict[str, Any] | None = None


class HumanAction(BaseModel):
    """Action executed or recommended by a human analyst in the incident."""

    action_id: str = Field(..., min_length=1)
    timestamp: str | None = None
    approval: bool | None = None
    order: int | None = None


def require_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")


def read_json(path: Path) -> dict[str, Any]:
    require_exists(path)
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"JSON top-level value must be an object: {path}")
    return obj


def read_jsonl(path: Path, *, strict: bool = False) -> list[dict[str, Any]]:
    """Read telemetry JSONL with optional tolerance for corrupted lines.

    strict=False keeps current behavior: invalid lines are skipped with warnings.
    strict=True enables fail-fast: stop at the first invalid line.
    """
    require_exists(path)
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped_line = line.strip()
            if not stripped_line:
                continue
            try:
                record = json.loads(stripped_line)
            except json.JSONDecodeError as exc:
                message = f"Invalid line {line_no} in {path}: {exc}"
                if strict:
                    raise ValueError(message) from exc
                print(f"Warning: {message}")
                continue
            if isinstance(record, dict):
                records.append(record)
            else:
                message = f"Line {line_no} ignored (not an object) in {path}"
                if strict:
                    raise ValueError(message)
                print(f"Warning: {message}")
    return records


def read_yaml_list(path: Path) -> list[dict[str, Any]]:
    require_exists(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if data is None:
        return []
    if not isinstance(data, list):
        raise ValueError(f"YAML top-level value must be a list: {path}")

    out: list[dict[str, Any]] = []
    for item_index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(
                f"YAML item must be an object. path={path} index={item_index}"
            )
        out.append(item)
    return out


# Parsers fail fast and propagate Pydantic ValidationError directly.
# No try/except wrapper is used here because the native validation error already
# points to the exact field, received value, and failure reason.


def parse_incident_meta(raw: dict[str, Any]) -> IncidentMeta:
    """Validate and return incident metadata.

    Raises:
        pydantic.ValidationError: if a required field is missing or invalid.
    """
    return IncidentMeta.model_validate(raw)


def parse_action_catalog(raw: Iterable[dict[str, Any]]) -> list[ActionCatalogItem]:
    """Validate and return action catalog.

    Raises:
        pydantic.ValidationError: if any catalog entry is invalid.
    """
    return [ActionCatalogItem.model_validate(item) for item in raw]


def parse_telemetry(raw: Iterable[dict[str, Any]]) -> list[TelemetryEvent]:
    """Validate and return normalized telemetry."""
    return [TelemetryEvent.model_validate(item) for item in raw]


def parse_human_actions(raw: Iterable[dict[str, Any]]) -> list[HumanAction]:
    """Validate and return human actions for the incident."""
    return [HumanAction.model_validate(item) for item in raw]


def validate_human_action_order(actions: list[HumanAction]) -> None:
    """Validate `order` sequence for strict experiments.

    Rules:
    - all entries must define `order`;
    - order must start at 1;
    - order values must be unique and contiguous up to N.
    """
    if not actions:
        return
    missing = [
        idx
        for idx, action in enumerate(actions, start=1)
        if action.order is None
    ]
    if missing:
        raise ValueError(f"Human actions missing 'order' at rows: {missing}")
    orders = [int(action.order) for action in actions if action.order is not None]
    if any(order < 1 for order in orders):
        raise ValueError(f"Invalid 'order' value (<1): {orders}")
    unique = sorted(set(orders))
    expected = list(range(1, len(actions) + 1))
    if unique != expected:
        raise ValueError(
            "'order' must be contiguous and unique. "
            f"Expected={expected} received={orders}"
        )


def parse_human_actions_strict_order(
    raw: Iterable[dict[str, Any]],
) -> list[HumanAction]:
    """Validate human actions and require explicit consistent ordering."""
    actions = parse_human_actions(raw)
    validate_human_action_order(actions)
    return actions


def parse_rules(raw: Iterable[dict[str, Any]]) -> list[PolicyRule]:
    """Validate and return policy rules.

    Raises:
        pydantic.ValidationError: if any rule is invalid.
    """
    return [PolicyRule.model_validate(item) for item in raw]
