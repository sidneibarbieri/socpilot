from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from soc_llm_policy.io import (
    ActionCatalogItem,
    IncidentMeta,
    PolicyRule,
    TelemetryEvent,
)
from soc_llm_policy.proportionality import ProportionalityPolicy


@dataclass(frozen=True)
class Violation:
    """A policy-rule violation detected in the action plan."""

    rule_id: str
    severity: str
    type: str
    action: str
    repair: str
    missing_before: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "rule_id": self.rule_id,
            "severity": self.severity,
            "type": self.type,
            "action": self.action,
            "repair": self.repair,
        }
        if self.missing_before is not None:
            d["missing_before"] = self.missing_before
        return d


@dataclass(frozen=True)
class EnforcementRequest:
    llm_actions: list[str]
    telemetry: list[TelemetryEvent | dict[str, Any]]
    rules: list[PolicyRule]
    catalog: list[ActionCatalogItem]
    incident_approved_actions: set[str] | None = None
    approval_policy_mode: Literal["remove", "defer_to_human_approval"] = "remove"
    incident_meta: IncidentMeta | None = None
    proportionality_policy: ProportionalityPolicy | None = None


def _find_order_cycle(edges: dict[str, set[str]]) -> list[str] | None:
    """Return one cycle path if found, otherwise None."""
    visiting: set[str] = set()
    visited: set[str] = set()
    stack: list[str] = []

    def dfs(node: str) -> list[str] | None:
        if node in visiting:
            if node in stack:
                start = stack.index(node)
                return [*stack[start:], node]
            return [node, node]
        if node in visited:
            return None

        visiting.add(node)
        stack.append(node)
        for nxt in sorted(edges.get(node, set())):
            found = dfs(nxt)
            if found is not None:
                return found
        stack.pop()
        visiting.remove(node)
        visited.add(node)
        return None

    for node in sorted(edges):
        found = dfs(node)
        if found is not None:
            return found
    return None


def validate_rule_consistency(rules: list[PolicyRule]) -> None:
    """Fail fast on policy-rule conflicts that make deterministic repair unsound.

    Current checks:
      1) Self-dependency in order constraints.
      2) Cycles in the partial-order graph from prohibit_before rules.
    """
    errors: list[str] = []
    edges: dict[str, set[str]] = {}

    for rule in rules:
        if rule.type == "mandatory":
            continue

        if rule.type == "require_approval":
            continue

        if rule.type != "prohibit_before" or not rule.condition_action:
            continue

        if rule.action == rule.condition_action:
            errors.append(
                "invalid_order_rule:"
                f" rule_id={rule.rule_id} has self-dependency ({rule.action})"
            )
            continue

        edges.setdefault(rule.condition_action, set()).add(rule.action)

    cycle = _find_order_cycle(edges)
    if cycle is not None:
        cycle_path = " -> ".join(cycle)
        errors.append(f"order_cycle_detected: {cycle_path}")

    if errors:
        raise ValueError("Inconsistent policy rules: " + " | ".join(errors))


def _normalize_telemetry(
    telemetry: list[TelemetryEvent | dict[str, Any]],
) -> list[TelemetryEvent]:
    return [
        event
        if isinstance(event, TelemetryEvent)
        else TelemetryEvent.model_validate(event)
        for event in telemetry
    ]


def _normalized_needles(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(value).lower() for value in values if value]


def _contains_any(text: str | None, needles: list[str]) -> bool:
    if not needles:
        return True
    haystack = (text or "").lower()
    return any(needle in haystack for needle in needles)


def _contains_all(text: str | None, needles: list[str]) -> bool:
    if not needles:
        return True
    haystack = (text or "").lower()
    return all(needle in haystack for needle in needles)


def _details_text(event: TelemetryEvent) -> str:
    details = event.details.model_dump(mode="json")
    return json.dumps(details, sort_keys=True, ensure_ascii=True).lower()


def _event_matches_scope(event: TelemetryEvent, scope: dict[str, Any]) -> bool:
    event_type_needles = _normalized_needles(scope.get("event_type_contains"))
    category_needles = _normalized_needles(scope.get("category_contains"))
    command_needles = _normalized_needles(scope.get("command_contains"))
    details_needles = _normalized_needles(scope.get("details_text_contains"))
    severity_needles = _normalized_needles(scope.get("severity_contains"))

    has_predicate = any(
        [
            event_type_needles,
            category_needles,
            command_needles,
            details_needles,
            severity_needles,
        ]
    )
    if not has_predicate:
        return False

    return (
        _contains_any(event.event_type, event_type_needles)
        and _contains_any(event.category, category_needles)
        and _contains_all(event.details.command, command_needles)
        and _contains_any(_details_text(event), details_needles)
        and _contains_any(event.details.severity, severity_needles)
    )


def telemetry_matches_scope(
    telemetry: list[TelemetryEvent | dict[str, Any]],
    scope: dict[str, Any] | None,
) -> bool:
    """Check whether any telemetry event satisfies a typed rule scope."""
    if not scope:
        return False

    for event in _normalize_telemetry(telemetry):
        if _event_matches_scope(event, scope):
            return True
    return False


_INCIDENT_SCOPE_FIELDS = {
    "incident_asset_criticality_in": "asset_criticality",
    "incident_severity_in": "severity",
    "incident_type_in": "incident_type",
}
_TELEMETRY_SCOPE_KEYS = {
    "event_type_contains",
    "category_contains",
    "command_contains",
    "details_text_contains",
    "severity_contains",
}
_ALLOWED_SCOPE_KEYS = {"global", *_INCIDENT_SCOPE_FIELDS, *_TELEMETRY_SCOPE_KEYS}


def validate_rule_scope(scope: dict[str, Any] | None) -> None:
    """Validate the supported policy-scope language."""
    if not scope:
        return
    unknown_keys = sorted(set(scope) - _ALLOWED_SCOPE_KEYS)
    if unknown_keys:
        raise ValueError(f"Unsupported policy scope keys: {unknown_keys}")
    predicate_keys = set(scope) - {"global"}
    if scope.get("global") is True and predicate_keys:
        raise ValueError("A global scope cannot declare content predicates")
    for key in predicate_keys:
        values = scope[key]
        if not isinstance(values, list):
            raise ValueError(f"Policy scope {key!r} must be a list")
        if key in _INCIDENT_SCOPE_FIELDS and not values:
            raise ValueError(
                f"Incident policy scope {key!r} must be a non-empty list"
            )


def _has_scope_predicate(
    scope: dict[str, Any] | None,
    keys: set[str],
) -> bool:
    return bool(scope and any(key in scope for key in keys))


def incident_matches_scope(
    meta: IncidentMeta | None,
    scope: dict[str, Any] | None,
) -> bool:
    """Check the incident-level predicates in a rule scope.

    Predicates are ANDed. A scope with no incident-level predicate does not
    restrict by incident and returns True. Missing metadata for a declared
    predicate is an experiment configuration error and fails loudly.
    """
    validate_rule_scope(scope)
    if not scope:
        return True

    present = [
        (key, field)
        for key, field in _INCIDENT_SCOPE_FIELDS.items()
        if key in scope
    ]
    if not present:
        return True
    if meta is None:
        raise ValueError("Incident metadata is required by the declared policy scope")

    for key, field in present:
        allowed = {str(value).lower() for value in scope[key]}
        if str(getattr(meta, field)).lower() not in allowed:
            return False
    return True


def rule_scope_matches(
    incident_meta: IncidentMeta | None,
    telemetry: list[TelemetryEvent | dict[str, Any]],
    scope: dict[str, Any] | None,
) -> bool:
    """Evaluate all declared incident and telemetry predicates as a conjunction."""
    validate_rule_scope(scope)
    if not scope or scope.get("global") is True:
        return True
    if not incident_matches_scope(incident_meta, scope):
        return False
    if not _has_scope_predicate(scope, _TELEMETRY_SCOPE_KEYS):
        return True
    return telemetry_matches_scope(telemetry, scope)


def telemetry_has_indicator(
    telemetry: list[TelemetryEvent | dict[str, Any]],
    event_type_contains: list[str],
    command_contains: list[str],
) -> bool:
    """Check whether telemetry events satisfy the legacy indicator set."""
    event_type_needles = [
        event_type.lower() for event_type in event_type_contains if event_type
    ]
    command_needles = [
        command.lower() for command in command_contains if command
    ]

    for event in _normalize_telemetry(telemetry):
        normalized_event_type = event.event_type.lower()
        normalized_command = (event.details.command or "").lower()

        if any(
            needle in normalized_event_type for needle in event_type_needles
        ) and all(needle in normalized_command for needle in command_needles):
            return True
    return False


def validate_catalog(
    action_ids: list[str],
    catalog: list[ActionCatalogItem],
) -> list[str]:
    """Filter action_ids that are not present in the catalog."""
    allowed_actions = {catalog_item.action_id for catalog_item in catalog}
    return [action_id for action_id in action_ids if action_id in allowed_actions]


def approvals_map(catalog: list[ActionCatalogItem]) -> Mapping[str, bool]:
    """Return action_id -> requires_approval mapping."""
    return {
        catalog_item.action_id: catalog_item.requires_approval
        for catalog_item in catalog
    }


def _apply_mandatory_rules(
    filtered: list[str],
    telemetry: list[TelemetryEvent | dict[str, Any]],
    rules: list[PolicyRule],
    incident_meta: IncidentMeta | None,
) -> tuple[list[str], list[Violation]]:
    """Step 1: missing mandatory actions when indicator is present."""
    violations: list[Violation] = []

    for rule in rules:
        if rule.type != "mandatory":
            continue

        if not rule_scope_matches(incident_meta, telemetry, rule.scope):
            continue

        if rule.action not in filtered:
            violations.append(
                Violation(
                    rule_id=rule.rule_id,
                    severity=rule.severity,
                    type="missing_mandatory",
                    action=rule.action,
                    repair="insert",
                )
            )
            filtered = [rule.action, *filtered]

    return filtered, violations


def _apply_prohibit_when_rules(
    filtered: list[str],
    telemetry: list[TelemetryEvent | dict[str, Any]],
    rules: list[PolicyRule],
    incident_meta: IncidentMeta | None,
) -> tuple[list[str], list[Violation]]:
    """Remove an action that is prohibited while the rule scope holds.

    The mirror of a mandatory rule: a mandatory rule inserts an action when an
    indicator is present; a prohibit_when rule removes one. Rule R5 (restraint
    on a benign verdict) is expressed this way, one rule per prohibited action.
    """
    violations: list[Violation] = []

    for rule in rules:
        if rule.type != "prohibit_when":
            continue

        if not rule_scope_matches(incident_meta, telemetry, rule.scope):
            continue

        if rule.action in filtered:
            violations.append(
                Violation(
                    rule_id=rule.rule_id,
                    severity=rule.severity,
                    type="prohibited_when",
                    action=rule.action,
                    repair="remove",
                )
            )
            filtered = [action for action in filtered if action != rule.action]

    return filtered, violations


def _apply_order_rules(
    filtered: list[str],
    rules: list[PolicyRule],
    telemetry: list[TelemetryEvent | dict[str, Any]],
    incident_meta: IncidentMeta | None,
) -> tuple[list[str], list[Violation]]:
    """Step 2: action-order violations (prohibit_before).

    Order constraints can form acyclic chains. Apply them to a fixed point so
    a missing prerequisite introduced by one rule can itself trigger another
    prerequisite in the same enforcement pass.
    """
    violations: list[Violation] = []
    emitted: set[tuple[str, str, str]] = set()
    order_rules = [
        rule
        for rule in rules
        if rule.type == "prohibit_before"
        and rule.condition_action
        and rule_scope_matches(incident_meta, telemetry, rule.scope)
    ]
    if not order_rules:
        return filtered, violations

    repair_limit = len(order_rules) * max(1, len(filtered) + len(order_rules) + 1)
    repair_count = 0
    changed = True
    while changed:
        changed = False
        for rule in order_rules:
            if rule.action not in filtered:
                continue

            idx = filtered.index(rule.action)
            condition_action = str(rule.condition_action)
            if condition_action in filtered[:idx]:
                continue

            signature = (rule.rule_id, rule.action, condition_action)
            if signature not in emitted:
                violations.append(
                    Violation(
                        rule_id=rule.rule_id,
                        severity=rule.severity,
                        type="order_violation",
                        action=rule.action,
                        repair="insert_before",
                        missing_before=condition_action,
                    )
                )
                emitted.add(signature)

            if condition_action in filtered:
                filtered = [action for action in filtered if action != condition_action]
                idx = filtered.index(rule.action)
            filtered.insert(idx, condition_action)
            changed = True
            repair_count += 1
            if repair_count > repair_limit:
                raise ValueError("Order repair did not converge")

    return filtered, violations


def _apply_approval_rules(
    filtered: list[str],
    request: EnforcementRequest,
    telemetry: list[TelemetryEvent | dict[str, Any]],
) -> tuple[list[str], list[Violation]]:
    """Step 3: enforce approval-gated actions.

    Modes:
      - remove: drop approval-gated actions from enforced plan.
      - defer_to_human_approval: keep action, mark as deferred for human approval.
    """
    if request.approval_policy_mode not in {
        "remove",
        "defer_to_human_approval",
    }:
        raise ValueError(
            f"Invalid approval_policy_mode: {request.approval_policy_mode}"
        )

    violations: list[Violation] = []
    approvals = approvals_map(request.catalog)
    approved_actions = request.incident_approved_actions or set()

    for rule in request.rules:
        if rule.type != "require_approval":
            continue
        if not rule_scope_matches(
            request.incident_meta,
            telemetry,
            rule.scope,
        ):
            continue
        if (
            rule.action in filtered
            and approvals.get(rule.action, False)
            and rule.action not in approved_actions
        ):
            if request.approval_policy_mode == "remove":
                violations.append(
                    Violation(
                        rule_id=rule.rule_id,
                        severity=rule.severity,
                        type="approval_required",
                        action=rule.action,
                        repair="remove",
                    )
                )
                filtered = [
                    action_id for action_id in filtered if action_id != rule.action
                ]
            else:
                violations.append(
                    Violation(
                        rule_id=rule.rule_id,
                        severity=rule.severity,
                        type="approval_deferred",
                        action=rule.action,
                        repair="defer_to_human_approval",
                    )
                )

    return filtered, violations


def _apply_proportionality_rules(
    filtered: list[str],
    incident_meta: IncidentMeta | None,
    catalog: list[ActionCatalogItem],
    policy: ProportionalityPolicy | None,
) -> tuple[list[str], list[Violation]]:
    """Remove actions whose impact exceeds the incident's demand ceiling."""
    if policy is None:
        return filtered, []
    policy.require_approved()
    if incident_meta is None:
        raise ValueError("Incident metadata is required by proportionality policy")

    ceiling = policy.ceiling_for(incident_meta)
    violations: list[Violation] = []
    kept: list[str] = []
    for action in filtered:
        rank = policy.impact_rank(action, catalog)
        if rank > ceiling:
            violations.append(
                Violation(
                    rule_id="PROPORTIONALITY",
                    severity="hard",
                    type="disproportionate_action",
                    action=action,
                    repair="remove",
                )
            )
            continue
        kept.append(action)
    return kept, violations


def _deduplicate(actions: list[str]) -> list[str]:
    """Remove duplicates while preserving insertion order."""
    seen: set[str] = set()
    result: list[str] = []
    for action in actions:
        if action not in seen:
            result.append(action)
            seen.add(action)
    return result


# Public entrypoint


def _coerce_enforcement_request(
    request: EnforcementRequest | None,
    kwargs: dict[str, Any],
) -> EnforcementRequest:
    if request is not None:
        if kwargs:
            raise TypeError("Do not mix EnforcementRequest with legacy keyword args.")
        return request
    if not kwargs:
        raise TypeError("enforce_policy requires an EnforcementRequest.")
    coerced = EnforcementRequest(
        llm_actions=kwargs.pop("llm_actions"),
        telemetry=kwargs.pop("telemetry"),
        rules=kwargs.pop("rules"),
        catalog=kwargs.pop("catalog"),
        incident_approved_actions=kwargs.pop("incident_approved_actions", None),
        approval_policy_mode=kwargs.pop("approval_policy_mode", "remove"),
        incident_meta=kwargs.pop("incident_meta", None),
        proportionality_policy=kwargs.pop("proportionality_policy", None),
    )
    if kwargs:
        raise TypeError(f"Unexpected keyword arguments: {sorted(kwargs)}")
    return coerced


def enforce_policy(
    request: EnforcementRequest | None = None,
    **kwargs: Any,
) -> tuple[list[Violation], list[str]]:
    """Verify and correct an LLM action plan against policy rules.

    Applies mandatory, restraint, ordering, proportionality, and approval steps.
    Each step can add violations and modify the plan.

    Returns:
        Tuple (violations, enforced_actions).
    """
    request = _coerce_enforcement_request(request, kwargs)
    validate_rule_consistency(request.rules)

    filtered = list(validate_catalog(request.llm_actions, request.catalog))
    normalized_telemetry = _normalize_telemetry(request.telemetry)

    filtered, mandatory_violations = _apply_mandatory_rules(
        filtered,
        normalized_telemetry,
        request.rules,
        request.incident_meta,
    )
    filtered, restraint_violations = _apply_prohibit_when_rules(
        filtered,
        normalized_telemetry,
        request.rules,
        request.incident_meta,
    )
    filtered, order_violations = _apply_order_rules(
        filtered,
        request.rules,
        normalized_telemetry,
        request.incident_meta,
    )
    filtered, proportionality_violations = _apply_proportionality_rules(
        filtered,
        request.incident_meta,
        request.catalog,
        request.proportionality_policy,
    )
    filtered, approval_violations = _apply_approval_rules(
        filtered,
        request,
        normalized_telemetry,
    )

    violations = (
        mandatory_violations
        + restraint_violations
        + order_violations
        + proportionality_violations
        + approval_violations
    )
    enforced = _deduplicate(validate_catalog(filtered, request.catalog))

    return violations, enforced
