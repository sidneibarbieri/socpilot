from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from soc_llm_policy.io import parse_rules, read_yaml_list


class RuleDesignDecision(BaseModel):
    """Substantive provenance: why the rule exists and why it was chosen over
    the alternatives that were weighed. A rule without this cannot be shipped."""

    normative_basis: str = Field(..., min_length=1)
    alternatives_considered: list[str] = Field(..., min_length=2)
    chosen_because: str = Field(..., min_length=1)
    activation_tier: Literal["operational", "coverage"]


class ReviewedRule(BaseModel):
    rule_id: str = Field(..., min_length=1)
    decision: Literal[
        "transfer_supporting",
        "operational_local",
        "structural",
        "specification_only",
    ]
    # Both values describe research provenance, not institutional approval.
    # Corpus-derived rules use the first export batch for trigger selection and
    # the second batch for an activation check.
    provenance: Literal["authored_experimental_policy", "corpus_derived_policy"]
    design: RuleDesignDecision
    oracle_tests: list[str] = Field(..., min_length=1)
    limitation: str = Field(..., min_length=1)


class NormalizationSensitivityReview(BaseModel):
    leave_one_out_status: Literal["robust", "acknowledged_sensitive"]
    reviewed_by_role: str = Field(..., min_length=1)
    reviewed_at_utc: str = Field(..., min_length=1)
    decision: str = Field(..., min_length=1)
    limitation: str = Field(..., min_length=1)


class PolicyReviewRecord(BaseModel):
    schema_version: Literal[2]
    policy_id: str = Field(..., min_length=1)
    status: Literal["approved_for_controlled_experiment"]
    approved_by_role: str = Field(..., min_length=1)
    approved_at_utc: str = Field(..., min_length=1)
    constraints_sha256: str = Field(..., min_length=64, max_length=64)
    action_catalog_sha256: str = Field(..., min_length=64, max_length=64)
    scope_limitation: str = Field(..., min_length=1)
    normalization_sensitivity: NormalizationSensitivityReview | None = None
    reviewed_rules: list[ReviewedRule] = Field(..., min_length=1)
    excluded_candidate_surfaces: list[str] = Field(default_factory=list)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_policy_review(path: Path) -> PolicyReviewRecord:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Policy review record must be a mapping: {path}")
    return PolicyReviewRecord.model_validate(payload)


def validate_policy_review(
    review_path: Path,
    constraints_path: Path,
    action_catalog_path: Path,
) -> PolicyReviewRecord:
    record = load_policy_review(review_path)
    if sha256_file(constraints_path) != record.constraints_sha256:
        raise ValueError("Policy review constraints hash does not match.")
    if sha256_file(action_catalog_path) != record.action_catalog_sha256:
        raise ValueError("Policy review action-catalog hash does not match.")

    policy_rule_ids = {
        rule.rule_id for rule in parse_rules(read_yaml_list(constraints_path))
    }
    reviewed_rule_ids = [rule.rule_id for rule in record.reviewed_rules]
    if len(reviewed_rule_ids) != len(set(reviewed_rule_ids)):
        raise ValueError("Policy review contains duplicate rule IDs.")
    if set(reviewed_rule_ids) != policy_rule_ids:
        raise ValueError(
            "Policy review must cover exactly the selected policy rule IDs."
        )
    reviewed_by_id = {rule.rule_id: rule for rule in record.reviewed_rules}
    for policy_rule in parse_rules(read_yaml_list(constraints_path)):
        reviewed_rule = reviewed_by_id[policy_rule.rule_id]
        if (
            policy_rule.evidence_tier == "specification"
            and reviewed_rule.design.activation_tier != "coverage"
        ):
            raise ValueError(
                "Specification-only rules must use the coverage activation tier: "
                f"{policy_rule.rule_id}"
            )
        if (
            policy_rule.evidence_tier == "specification"
            and reviewed_rule.decision != "specification_only"
        ):
            raise ValueError(
                "Specification-only rules must use the specification decision: "
                f"{policy_rule.rule_id}"
            )
        if (
            policy_rule.evidence_tier == "operational"
            and reviewed_rule.decision == "specification_only"
        ):
            raise ValueError(
                "Operational rules cannot use the specification decision: "
                f"{policy_rule.rule_id}"
            )
    return record
