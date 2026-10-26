"""Run a synthetic policy-checking example without files, credentials, or models."""

from __future__ import annotations

import json
from dataclasses import replace

from soc_llm_policy.engine import EnforcementRequest, enforce_policy
from soc_llm_policy.family_metrics import evaluate_cell_families
from soc_llm_policy.io import ActionCatalogItem, PolicyRule


def example_request() -> EnforcementRequest:
    return EnforcementRequest(
        llm_actions=["reimage_host", "isolate_host"],
        telemetry=[],
        catalog=[ActionCatalogItem(action_id=action) for action in (
            "collect_forensics", "reimage_host", "isolate_host",
        )],
        rules=[
            PolicyRule(
                rule_id="evidence_before_reimage", type="prohibit_before",
                severity="hard", action="reimage_host",
                condition_action="collect_forensics",
            ),
            PolicyRule(
                rule_id="no_isolation_in_example", type="prohibit_when",
                severity="hard", action="isolate_host", scope={},
                repair_operator="remove",
            ),
        ],
    )


def run_example() -> dict[str, object]:
    request = example_request()
    before = evaluate_cell_families(request)
    edits, repaired = enforce_policy(request)
    repaired_request = replace(request, llm_actions=repaired)
    after = evaluate_cell_families(repaired_request)
    repeated_edits, repeated_plan = enforce_policy(repaired_request)
    if any(outcome.violated for outcome in after) or repeated_edits or repeated_plan != repaired:
        raise ValueError("The example repair failed its postcondition or second-pass check")
    return {
        "evidence_kind": "synthetic_demonstration",
        "constitutes_model_evidence": False,
        "proposed_actions": request.llm_actions,
        "policy": [rule.model_dump(exclude_none=True) for rule in request.rules],
        "before": [outcome.model_dump() for outcome in before],
        "edits": [edit.to_dict() for edit in edits],
        "repaired_actions": repaired,
        "after": [outcome.model_dump() for outcome in after],
        "second_pass_unchanged": True,
    }


def main() -> None:
    print(json.dumps(run_example(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
