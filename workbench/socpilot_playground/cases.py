"""Typed policy-case boundary reusing SOCpilot's input models and checker."""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import version
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from soc_llm_policy.engine import (
    EnforcementRequest,
    rule_scope_matches,
    validate_rule_consistency,
    validate_rule_scope,
)
from soc_llm_policy.io import (
    ActionCatalogItem,
    IncidentMeta,
    PolicyRule,
    TelemetryDetails,
    TelemetryEvent,
)

from .service import evaluate_request

ActionId = Annotated[str, Field(min_length=1, max_length=256, pattern=r"^\S+$")]


def reject_nonfinite_number(value: str) -> None:
    raise ValueError("Non-finite JSON numbers are not supported")


def decode_json(payload: str | bytes) -> object:
    return json.loads(payload, parse_constant=reject_nonfinite_number)


class CaseCatalogItem(ActionCatalogItem):
    model_config = ConfigDict(extra="forbid", strict=True)
    action_id: ActionId


class CaseRule(PolicyRule):
    model_config = ConfigDict(extra="forbid", strict=True)
    action: ActionId
    condition_action: ActionId | None = None

    @model_validator(mode="after")
    def validate_scope(self) -> Self:
        validate_rule_scope(self.scope)
        for key, values in (self.scope or {}).items():
            if key == "global":
                if values is not True:
                    raise ValueError(
                        "scope.global must be true; omit it for content predicates"
                    )
            elif not values or any(
                not isinstance(value, str) or not value for value in values
            ):
                raise ValueError(f"Scope {key} must contain non-empty strings")
        return self


class CaseTelemetryDetails(TelemetryDetails):
    model_config = ConfigDict(extra="forbid", strict=True)


class CaseTelemetryEvent(TelemetryEvent):
    model_config = ConfigDict(extra="forbid", strict=True)
    details: CaseTelemetryDetails = Field(default_factory=CaseTelemetryDetails)


class CaseIncidentMeta(IncidentMeta):
    model_config = ConfigDict(extra="forbid", strict=True)


class PolicyCase(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal["1"] = "1"
    case_id: str = Field(min_length=1, max_length=128)
    actions: list[ActionId] = Field(max_length=256)
    catalog: list[CaseCatalogItem] = Field(min_length=1, max_length=128)
    rules: list[CaseRule] = Field(min_length=1, max_length=128)
    telemetry: list[CaseTelemetryEvent] = Field(default_factory=list, max_length=256)
    incident_meta: CaseIncidentMeta | None = None
    incident_approved_actions: list[ActionId] = Field(
        default_factory=list, max_length=128
    )
    approval_policy_mode: Literal["remove", "defer_to_human_approval"] = "remove"

    @model_validator(mode="after")
    def validate_references(self) -> Self:
        catalog_ids = [item.action_id for item in self.catalog]
        if len(catalog_ids) != len(set(catalog_ids)):
            raise ValueError("Catalog action identifiers must be unique")
        rule_ids = [rule.rule_id for rule in self.rules]
        if len(rule_ids) != len(set(rule_ids)):
            raise ValueError("Rule identifiers must be unique")
        referenced = set(self.actions) | set(self.incident_approved_actions)
        for rule in self.rules:
            referenced.add(rule.action)
            if rule.type == "prohibit_before" and not rule.condition_action:
                raise ValueError(
                    f"Ordering rule {rule.rule_id} requires condition_action"
                )
            if rule.condition_action:
                referenced.add(rule.condition_action)
            rule_scope_matches(self.incident_meta, self.telemetry, rule.scope)
        if referenced - set(catalog_ids):
            raise ValueError(
                "All actions, rule references and approvals must belong to the catalog"
            )
        validate_rule_consistency(self.rules)
        return self

    def to_request(self) -> EnforcementRequest:
        return EnforcementRequest(
            llm_actions=self.actions,
            telemetry=self.telemetry,
            rules=self.rules,
            catalog=self.catalog,
            incident_meta=self.incident_meta,
            incident_approved_actions=set(self.incident_approved_actions),
            approval_policy_mode=self.approval_policy_mode,
        )


def evaluate_case(case: PolicyCase) -> dict[str, object]:
    result = evaluate_request(
        case.to_request(), evidence_kind="user_supplied_policy_case"
    )
    input_case = case.model_dump(mode="json")
    serialized = json.dumps(
        input_case, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    result.update(
        {
            "case_id": case.case_id,
            "input_case": input_case,
            "input_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
            "engine_version": version("soc-llm-policy"),
        }
    )
    return result
