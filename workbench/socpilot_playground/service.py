"""Shared checking service for guided proposals and imported policy cases."""

from dataclasses import replace

from soc_llm_policy.engine import EnforcementRequest, enforce_policy
from soc_llm_policy.family_metrics import evaluate_cell_families


def rule_outcomes(request: EnforcementRequest) -> list[dict[str, object]]:
    return [
        family.model_dump(mode="json") for family in evaluate_cell_families(request)
    ]


def violated_rules(outcomes: list[dict]) -> list[str]:
    return [
        rule["rule_id"]
        for family in outcomes
        for rule in family["rule_outcomes"]
        if rule["violated"]
    ]


def evaluate_request(
    request: EnforcementRequest, *, evidence_kind: str
) -> dict[str, object]:
    before = rule_outcomes(request)
    edits, repaired_actions = enforce_policy(request)
    repaired_request = replace(request, llm_actions=repaired_actions)
    after = rule_outcomes(repaired_request)
    repeated_edits, repeated_actions = enforce_policy(repaired_request)
    return {
        "evidence_kind": evidence_kind,
        "host_actions_executed": False,
        "constitutes_model_evidence": False,
        "proposed_actions": list(request.llm_actions),
        "plan_changed": repaired_actions != request.llm_actions,
        "policy": [rule.model_dump(exclude_none=True) for rule in request.rules],
        "before": before,
        "violated_rules_before": violated_rules(before),
        "edits": [edit.to_dict() for edit in edits],
        "repaired_actions": repaired_actions,
        "after": after,
        "violated_rules_after": violated_rules(after),
        "second_pass_unchanged": repeated_actions == repaired_actions,
        "second_pass_edits": [edit.to_dict() for edit in repeated_edits],
    }
