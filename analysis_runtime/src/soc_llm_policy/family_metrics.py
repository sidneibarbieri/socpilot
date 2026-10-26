"""Independent, denominatored measurements for policy families."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from soc_llm_policy.engine import (
    EnforcementRequest,
    approvals_map,
    rule_scope_matches,
    validate_rule_consistency,
)
from soc_llm_policy.io import PolicyRule
from soc_llm_policy.result_models import (
    PolicyFamilyAggregate,
    PolicyFamilyMetricsReport,
    PolicyFamilyOutcome,
    PolicyRuleOutcome,
    PolicyScopeKind,
)

_CONTENT_SCOPE_KEYS = {
    "incident_asset_criticality_in",
    "incident_severity_in",
    "incident_type_in",
    "event_type_contains",
    "category_contains",
    "command_contains",
    "details_text_contains",
    "severity_contains",
}
_RULE_FAMILY = {
    "mandatory": "mandatory",
    "prohibit_before": "ordering",
    "require_approval": "approval",
    "prohibit_when": "restraint",
}
_VIOLATION_TYPE = {
    "mandatory": "missing_mandatory",
    "prohibit_before": "order_violation",
    "require_approval": "approval_required",
    "prohibit_when": "prohibited_when",
}


def scope_kind_of(scope: dict[str, Any] | None) -> PolicyScopeKind:
    if scope and any(scope.get(key) for key in _CONTENT_SCOPE_KEYS):
        return "content_triggered"
    return "global"


def _rule_is_violated(
    rule: PolicyRule,
    proposed_actions: list[str],
    approved_actions: set[str],
) -> bool:
    if rule.type == "mandatory":
        return rule.action not in proposed_actions
    if rule.type == "prohibit_when":
        return rule.action in proposed_actions
    if rule.type == "require_approval":
        return rule.action not in approved_actions
    if rule.type == "prohibit_before":
        if rule.condition_action is None:
            raise ValueError(f"Ordering rule {rule.rule_id!r} has no condition_action")
        action_index = proposed_actions.index(rule.action)
        return rule.condition_action not in proposed_actions[:action_index]
    raise ValueError(f"Unsupported policy rule type: {rule.type}")


def _rule_is_applicable(
    rule: PolicyRule,
    request: EnforcementRequest,
    approval_required: dict[str, bool],
) -> bool:
    scope_matches = rule_scope_matches(
        request.incident_meta,
        request.telemetry,
        rule.scope,
    )
    if not scope_matches:
        return False
    if rule.type in {"prohibit_before", "require_approval"}:
        if rule.action not in request.llm_actions:
            return False
    if rule.type == "require_approval":
        return approval_required[rule.action]
    return True


def _rule_outcomes(
    request: EnforcementRequest,
) -> list[tuple[str, PolicyScopeKind, PolicyRuleOutcome]]:
    validate_rule_consistency(request.rules)
    catalog_actions = {item.action_id for item in request.catalog}
    approval_required = dict(approvals_map(request.catalog))
    approved_actions = request.incident_approved_actions or set()
    outcomes: list[tuple[str, PolicyScopeKind, PolicyRuleOutcome]] = []

    for rule in request.rules:
        if rule.action not in catalog_actions:
            raise ValueError(
                f"Policy rule {rule.rule_id!r} references unknown action "
                f"{rule.action!r}"
            )
        applicable = _rule_is_applicable(rule, request, approval_required)
        violated = (
            applicable
            and _rule_is_violated(rule, request.llm_actions, approved_actions)
        )
        outcomes.append(
            (
                _RULE_FAMILY[rule.type],
                scope_kind_of(rule.scope),
                PolicyRuleOutcome(
                    rule_id=rule.rule_id,
                    applicable=applicable,
                    violated=violated,
                    violation_type=_VIOLATION_TYPE[rule.type] if violated else None,
                ),
            )
        )
    return outcomes


def _proportionality_outcome(
    request: EnforcementRequest,
) -> tuple[str, PolicyScopeKind, PolicyRuleOutcome] | None:
    policy = request.proportionality_policy
    if policy is None:
        return None
    policy.require_approved()
    if request.incident_meta is None:
        raise ValueError("Incident metadata is required by proportionality policy")

    catalog_actions = {item.action_id for item in request.catalog}
    proposed_actions = [
        action for action in request.llm_actions if action in catalog_actions
    ]
    if not proposed_actions:
        applicable = False
        violated = False
    else:
        applicable = True
        ceiling = policy.ceiling_for(request.incident_meta)
        violated = any(
            policy.impact_rank(action, request.catalog) > ceiling
            for action in proposed_actions
        )
    return (
        "proportionality",
        "content_triggered",
        PolicyRuleOutcome(
            rule_id=policy.policy_id,
            applicable=applicable,
            violated=violated,
            violation_type="disproportionate_action" if violated else None,
        ),
    )


def evaluate_cell_families(
    request: EnforcementRequest,
) -> list[PolicyFamilyOutcome]:
    """Evaluate every family independently on the unrepaired proposed plan."""
    grouped: dict[
        tuple[str, PolicyScopeKind],
        list[PolicyRuleOutcome],
    ] = defaultdict(list)
    for family_id, scope_kind, outcome in _rule_outcomes(request):
        grouped[(family_id, scope_kind)].append(outcome)
    proportionality = _proportionality_outcome(request)
    if proportionality is not None:
        family_id, scope_kind, outcome = proportionality
        grouped[(family_id, scope_kind)].append(outcome)

    return [
        PolicyFamilyOutcome(
            family_id=family_id,
            scope_kind=scope_kind,
            applicable=any(item.applicable for item in rule_outcomes),
            violated=any(item.violated for item in rule_outcomes),
            rule_outcomes=sorted(
                rule_outcomes,
                key=lambda item: item.rule_id,
            ),
        )
        for (family_id, scope_kind), rule_outcomes in sorted(grouped.items())
    ]


def aggregate_families(
    cells: list[list[PolicyFamilyOutcome]],
) -> PolicyFamilyMetricsReport:
    """Aggregate cells only when they share the same policy-family surface."""
    if not cells:
        return PolicyFamilyMetricsReport(total_cells=0)

    expected_keys = {
        (outcome.family_id, outcome.scope_kind)
        for outcome in cells[0]
    }
    counters: dict[tuple[str, PolicyScopeKind], dict[str, int]] = {
        key: {"applicable": 0, "violated": 0}
        for key in expected_keys
    }
    for cell_index, outcomes in enumerate(cells):
        outcomes_by_key = {
            (outcome.family_id, outcome.scope_kind): outcome
            for outcome in outcomes
        }
        if set(outcomes_by_key) != expected_keys:
            raise ValueError(
                "Cannot pool cells with different policy-family surfaces: "
                f"cell_index={cell_index}"
            )
        for key, outcome in outcomes_by_key.items():
            counters[key]["applicable"] += int(outcome.applicable)
            counters[key]["violated"] += int(outcome.violated)

    total_cells = len(cells)
    aggregates = [
        PolicyFamilyAggregate(
            family_id=family_id,
            scope_kind=scope_kind,
            cell_count=total_cells,
            applicable_count=counters[(family_id, scope_kind)]["applicable"],
            violated_count=counters[(family_id, scope_kind)]["violated"],
            applicability_rate=round(
                counters[(family_id, scope_kind)]["applicable"] / total_cells,
                4,
            ),
            violation_rate=(
                round(
                    counters[(family_id, scope_kind)]["violated"]
                    / counters[(family_id, scope_kind)]["applicable"],
                    4,
                )
                if counters[(family_id, scope_kind)]["applicable"]
                else None
            ),
        )
        for family_id, scope_kind in sorted(expected_keys)
    ]
    return PolicyFamilyMetricsReport(
        total_cells=total_cells,
        families=aggregates,
    )
