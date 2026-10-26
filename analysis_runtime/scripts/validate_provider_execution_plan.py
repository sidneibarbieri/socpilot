#!/usr/bin/env python3
"""Validate the provider-execution plan before any provider call."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

from pydantic import BaseModel, Field

from soc_llm_policy.corpus_campaign import validate_campaign
from soc_llm_policy.durable_io import atomic_write_json
from soc_llm_policy.execution_source import validate_execution_source_manifest
from soc_llm_policy.frozen_inputs import (
    PROVIDER_INPUT_ROLES,
    FrozenInputBinding,
    resolve_frozen_input_bindings,
)
from soc_llm_policy.io import parse_action_catalog, parse_rules, read_yaml_list
from soc_llm_policy.llm_adapter import DEFAULT_MAX_COMPLETION_TOKENS
from soc_llm_policy.models_registry import (
    load_model_registry,
    validate_pricing_metadata,
)
from soc_llm_policy.paths import (
    RepoPaths,
    repo_relative_path,
    resolve_repo_relative_path,
    resolve_repo_root,
)
from soc_llm_policy.pipeline import list_inbox_incidents
from soc_llm_policy.policy_review import validate_policy_review
from soc_llm_policy.protocol_freeze import (
    PROTOCOL_FREEZE_SCHEMA,
    compute_incident_hashes,
)
from soc_llm_policy.provider_plan import (
    FrozenInput,
    ModelCostEstimate,
    ProviderExecutionPlan,
    load_provider_plan,
)
from soc_llm_policy.provider_readiness import READINESS_MAX_COMPLETION_TOKENS

_COST_TOTAL_TOLERANCE = 1e-6


class ExpectedExecutionRequest(BaseModel):
    dataset_release_id: str = ""
    incident_ids: list[str] = Field(default_factory=list)
    models: list[str] = Field(default_factory=list)
    arms: list[str] = Field(default_factory=list)
    repeats: int = 0


ExpectedExecutionRequest.model_rebuild()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument(
        "--plan-json",
        type=Path,
        default=Path("results/analysis/provider_execution_plan.json"),
    )
    parser.add_argument("--require-approval", action="store_true")
    parser.add_argument("--expected-dataset-release-id", default="")
    parser.add_argument("--expected-incident-ids", default="")
    parser.add_argument("--expected-models", default="")
    parser.add_argument("--expected-arms", default="")
    parser.add_argument("--expected-repeats", type=int, default=0)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/analysis/provider_execution_plan_validation.json"),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(path: Path, repo_root: Path) -> Path:
    if path.is_absolute():
        return path
    return repo_root / path


def load_plan(path: Path) -> ProviderExecutionPlan:
    return load_provider_plan(path)


def parse_csv_list(value: str) -> list[str]:
    items: list[str] = []
    for raw_item in value.split(","):
        item = raw_item.strip()
        if item and item not in items:
            items.append(item)
    return items


def compare_expected_value(
    issues: list[str],
    *,
    field_name: str,
    expected: object,
    actual: object,
) -> None:
    if expected and expected != actual:
        issues.append(f"{field_name} mismatch: expected {expected}, received {actual}")


def compare_expected_incidents(
    issues: list[str],
    metadata: dict[str, object],
    *,
    expected: list[str],
    actual: list[str],
) -> None:
    if not expected:
        metadata["incident_order_differs_from_command"] = False
        return
    if sorted(expected) != sorted(actual):
        issues.append(
            "incident_ids mismatch: expected the same incident set, received "
            "different values"
        )
        metadata["incident_order_differs_from_command"] = expected != actual
        return
    metadata["incident_order_differs_from_command"] = expected != actual


def validate_hashes(
    repo_root: Path,
    frozen_inputs: dict[str, FrozenInput],
) -> list[str]:
    issues: list[str] = []
    required_inputs = {
        "action_catalog",
        "constraints",
        "models_registry",
        "policy_review_record",
        "protocol_freeze",
    }
    missing_input_names = sorted(required_inputs - set(frozen_inputs))
    for input_name in missing_input_names:
        issues.append(f"missing frozen input: {input_name}")

    for input_name, frozen_input in sorted(frozen_inputs.items()):
        relative_path = Path(frozen_input.path)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            issues.append(f"frozen input path escapes repository: {input_name}")
            continue
        path = resolve_repo_relative_path(frozen_input.path, repo_root)
        resolved_root = repo_root.expanduser().resolve()
        if path != resolved_root and resolved_root not in path.parents:
            issues.append(f"frozen input path escapes repository: {input_name}")
            continue
        if not path.exists():
            issues.append(f"frozen input not found: {input_name} -> {path}")
            continue
        actual_hash = sha256_file(path)
        if actual_hash != frozen_input.sha256:
            issues.append(
                f"hash mismatch for {input_name}: expected "
                f"{frozen_input.sha256}, received {actual_hash}"
            )
    return issues


def validate_execution_source_binding(
    *,
    repo_root: Path,
    freeze: dict[str, object],
    bindings: dict[str, FrozenInputBinding],
) -> list[str]:
    source_identity = freeze.get("source_identity")
    source_binding = bindings["execution_source_manifest"]
    if not isinstance(source_identity, dict) or (
        source_identity.get("kind") != "execution_source_manifest"
        or source_identity.get("path") != source_binding.path
        or source_identity.get("sha256") != source_binding.sha256
    ):
        return ["protocol freeze execution source binding is inconsistent"]

    try:
        source_manifest = validate_execution_source_manifest(
            repo_root=repo_root,
            manifest_path=source_binding.resolved_path,
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        return [f"invalid execution source manifest: {exc}"]
    if source_identity.get("file_count") != source_manifest.file_count:
        return ["protocol freeze execution source file count is inconsistent"]
    return []


def validate_protocol_freeze(
    repo_root: Path,
    plan: ProviderExecutionPlan,
) -> list[str]:
    frozen_input = plan.frozen_inputs.get("protocol_freeze")
    if frozen_input is None:
        return ["missing frozen input: protocol_freeze"]
    path = resolve_repo_relative_path(frozen_input.path, repo_root)
    if not path.exists():
        return [f"frozen input not found: protocol_freeze -> {path}"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    issues: list[str] = []
    schema_version = payload.get("schema_version")
    if schema_version != PROTOCOL_FREEZE_SCHEMA:
        issues.append(
            f"protocol freeze schema_version must be {PROTOCOL_FREEZE_SCHEMA}"
        )
    if payload.get("state") != "pre_execution_protocol_freeze":
        issues.append("protocol freeze has an invalid pre-execution state")
    if payload.get("constitutes_model_evidence") is not False:
        issues.append("protocol freeze must not claim model evidence")
    if payload.get("execution_phase") != "primary":
        issues.append("protocol freeze execution_phase must be primary")
    expected = {
        "dataset_release_id": plan.dataset_release_id,
        "eval_protocol_version": plan.eval_protocol_version,
        "incident_ids": plan.incident_ids,
        "official_models": plan.models,
        "arms": plan.arms,
        "repeats": plan.repeats,
        "repeat_reduction": plan.repeat_reduction,
        "primary_cell_count": plan.primary_cell_count,
        "diagnostic_task_ceiling": plan.diagnostic_task_ceiling,
        "scientific_task_ceiling": plan.scientific_task_ceiling,
    }
    issues.extend(
        f"protocol freeze mismatch: {field_name}"
        for field_name, expected_value in expected.items()
        if payload.get(field_name) != expected_value
    )
    retry_policy = payload.get("retry_policy")
    if not isinstance(retry_policy, dict):
        issues.append("protocol freeze retry_policy must be an object")
    elif retry_policy.get("llm_attempt_limit") != (
        plan.provider_transport_attempt_limit
    ):
        issues.append("protocol freeze provider-attempt limit differs from the plan")
    input_hashes = payload.get("input_hashes")
    if not isinstance(input_hashes, dict):
        issues.append("protocol freeze input_hashes must be an object")
        return issues
    bindings = None
    try:
        bindings = resolve_frozen_input_bindings(
            input_hashes,
            repo_root,
            required_roles=(
                *PROVIDER_INPUT_ROLES,
                "policy_preflight_manifest",
                "execution_source_manifest",
            ),
        )
    except (FileNotFoundError, ValueError) as exc:
        issues.append(f"invalid protocol freeze input binding: {exc}")

    if bindings is not None:
        issues.extend(
            validate_execution_source_binding(
                repo_root=repo_root,
                freeze=payload,
                bindings=bindings,
            )
        )

        for role in PROVIDER_INPUT_ROLES:
            plan_binding = plan.frozen_inputs.get(role)
            freeze_binding = bindings[role]
            if plan_binding is None:
                issues.append(f"plan/freeze input mismatch: {role}")
                continue
            if (
                plan_binding.path != freeze_binding.path
                or plan_binding.sha256 != freeze_binding.sha256
            ):
                issues.append(f"plan/freeze input mismatch: {role}")

        campaign_plan_binding = plan.frozen_inputs.get("corpus_campaign_manifest")
        campaign_freeze_binding = input_hashes.get("corpus_campaign_manifest")
        if campaign_plan_binding is not None:
            if not isinstance(campaign_freeze_binding, dict) or (
                campaign_plan_binding.path != campaign_freeze_binding.get("path")
                or campaign_plan_binding.sha256 != campaign_freeze_binding.get("sha256")
            ):
                issues.append("plan/freeze input mismatch: corpus_campaign_manifest")

        preflight_payload = json.loads(
            bindings["policy_preflight_manifest"].resolved_path.read_text(
                encoding="utf-8"
            )
        )
        checks = preflight_payload.get("checks")
        required_checks = {
            "policy_review_valid",
            "preflight_reports_share_policy_and_corpora",
            "raw_exports_map_bijectively",
        }
        if (
            preflight_payload.get("constitutes_model_evidence") is not False
            or not isinstance(checks, dict)
            or any(checks.get(name) is not True for name in required_checks)
        ):
            issues.append("policy-preflight manifest is not release-ready")
        sensitivity = preflight_payload.get("normalization_sensitivity")
        if not isinstance(sensitivity, dict) or sensitivity.get(
            "review_status"
        ) not in {"robust", "acknowledged_sensitive"}:
            issues.append("policy-preflight sensitivity review is incomplete")
    expected_incident_hashes = payload.get("incident_hashes")
    if not isinstance(expected_incident_hashes, dict):
        issues.append("protocol freeze incident_hashes must be an object")
        return issues
    paths = RepoPaths(repo_root=repo_root)
    actual_incident_hashes = compute_incident_hashes(paths, plan.incident_ids)
    if expected_incident_hashes != actual_incident_hashes:
        issues.append("protocol freeze incident content hash mismatch")
    return issues


def validate_incidents(paths: RepoPaths, incident_ids: list[str]) -> list[str]:
    available_incidents = set(list_inbox_incidents(paths))
    missing_incidents = [
        incident_id
        for incident_id in incident_ids
        if incident_id not in available_incidents
    ]
    duplicate_incidents = sorted(
        incident_id
        for incident_id in set(incident_ids)
        if incident_ids.count(incident_id) > 1
    )
    issues: list[str] = []
    if missing_incidents:
        issues.append(f"incident IDs not found in dataset: {missing_incidents}")
    if duplicate_incidents:
        issues.append(f"duplicate incident IDs in plan: {duplicate_incidents}")
    return issues


def validate_campaign_binding(
    repo_root: Path,
    plan: ProviderExecutionPlan,
) -> list[str]:
    binding = plan.frozen_inputs.get("corpus_campaign_manifest")
    if binding is None:
        return []
    try:
        manifest = validate_campaign(
            repo_root=repo_root,
            manifest_path=resolve_repo_relative_path(binding.path, repo_root),
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        return [f"invalid corpus campaign: {exc}"]
    if sorted(manifest.incident_partitions) != sorted(plan.incident_ids):
        return ["corpus campaign incident set differs from the provider plan"]
    return []


def validate_models(
    registry_path: Path,
    plan: ProviderExecutionPlan,
) -> list[str]:
    profiles = load_model_registry(registry_path)
    by_selector = {
        selector: profile
        for profile in profiles
        for selector in (profile.name, profile.deployment)
    }
    issues: list[str] = []
    for model_name in plan.provider_readiness.model_names:
        profile = by_selector.get(model_name)
        if profile is None:
            issues.append(f"model not found in registry: {model_name}")
            continue
        if model_name in plan.models and not profile.enabled:
            issues.append(f"model is disabled in the registry: {model_name}")
        try:
            validate_pricing_metadata(profile, as_of=date.today())
        except ValueError as exc:
            issues.append(str(exc))
    for estimate in plan.cost_estimate.models:
        profile = by_selector.get(estimate.model_name)
        if profile is None:
            continue
        expected_metadata = {
            "provider": profile.provider,
            "billing mode": profile.billing_mode,
            "prompt price": profile.prompt_price_per_1k_usd,
            "completion price": profile.completion_price_per_1k_usd,
            "pricing source": profile.pricing_source_url,
            "pricing verification date": profile.pricing_verified_on,
            "pricing validity date": profile.pricing_valid_through,
        }
        observed_metadata = {
            "provider": estimate.provider,
            "billing mode": estimate.billing_mode,
            "prompt price": estimate.prompt_price_per_1k_usd,
            "completion price": estimate.completion_price_per_1k_usd,
            "pricing source": estimate.pricing_source_url,
            "pricing verification date": estimate.pricing_verified_on,
            "pricing validity date": estimate.pricing_valid_through,
        }
        for label, expected_value in expected_metadata.items():
            if observed_metadata[label] != expected_value:
                issues.append(
                    f"cost estimate {label} differs from the registry for "
                    f"{estimate.study_id}/{estimate.model_name}"
                )
    return issues


def validate_arms(arm_names: list[str]) -> list[str]:
    allowed_arms = {"llm_zero", "llm_policy_prompt"}
    invalid_arms = [arm_name for arm_name in arm_names if arm_name not in allowed_arms]
    return [f"invalid arms: {invalid_arms}"] if invalid_arms else []


def validate_policy_surface(
    action_catalog_path: Path,
    constraints_path: Path,
) -> list[str]:
    catalog = parse_action_catalog(read_yaml_list(action_catalog_path))
    rules = parse_rules(read_yaml_list(constraints_path))
    action_ids = {item.action_id for item in catalog}
    issues: list[str] = []
    if len(action_ids) != len(catalog):
        issues.append("duplicate action IDs in action catalog")
    rule_ids = [rule.rule_id for rule in rules]
    if len(set(rule_ids)) != len(rule_ids):
        issues.append("duplicate rule IDs in constraints")
    for rule in rules:
        if rule.action not in action_ids:
            issues.append(
                f"rule {rule.rule_id} references unknown action {rule.action}"
            )
        if rule.condition_action and rule.condition_action not in action_ids:
            issues.append(
                f"rule {rule.rule_id} references unknown condition action "
                f"{rule.condition_action}"
            )
        if rule.type == "prohibit_before" and not rule.condition_action:
            issues.append(f"rule {rule.rule_id} requires condition_action")
    return issues


def validate_policy_review_record(
    review_record_path: Path,
    constraints_path: Path,
    action_catalog_path: Path,
) -> list[str]:
    try:
        validate_policy_review(
            review_record_path,
            constraints_path,
            action_catalog_path,
        )
    except (FileNotFoundError, ValueError) as exc:
        return [f"invalid policy review record: {exc}"]
    return []


def resolved_plan_inputs(
    repo_root: Path,
    plan: ProviderExecutionPlan,
) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    for role in PROVIDER_INPUT_ROLES:
        entry = plan.frozen_inputs.get(role)
        if entry is None:
            continue
        candidate = resolve_repo_relative_path(entry.path, repo_root)
        if candidate.is_file():
            resolved[role] = candidate
    return resolved


def validate_approval(plan: ProviderExecutionPlan, require_approval: bool) -> list[str]:
    issues: list[str] = []
    if plan.approval.status == "draft":
        if require_approval:
            issues.append("execution plan is not approved")
        return issues
    if plan.approval.status != "approved":
        issues.append("execution plan is not approved")
    if not plan.approval.approved_by.strip():
        issues.append("approved_by is required for approved execution")
    if not plan.approval.approved_at_utc.strip():
        issues.append("approved_at_utc is required for approved execution")
    if plan.approval.approved_budget_usd is None:
        issues.append("approved_budget_usd is required for approved execution")
    elif (
        plan.approval.approved_budget_usd
        < plan.cost_estimate.approval_budget_with_headroom_usd
    ):
        issues.append("approved provider budget is below the required headroom")
    return issues


def _validate_model_cost_estimate(
    estimate: ModelCostEstimate,
    expected_request_count: int | None,
) -> list[str]:
    issues: list[str] = []
    identity = f"{estimate.study_id}/{estimate.model_name}"
    if (
        expected_request_count is not None
        and estimate.request_count != expected_request_count
    ):
        issues.append(
            f"cost estimate request count mismatch for {identity}: expected "
            f"{expected_request_count}, received {estimate.request_count}"
        )
    expected_max_completion = (
        READINESS_MAX_COMPLETION_TOKENS
        if estimate.study_id == "provider_readiness"
        else DEFAULT_MAX_COMPLETION_TOKENS
    )
    if estimate.max_completion_tokens_per_request != expected_max_completion:
        issues.append(
            f"cost estimate completion-token limit mismatch for {identity}: "
            f"expected {expected_max_completion}, received "
            f"{estimate.max_completion_tokens_per_request}"
        )
    recomputed_row_cost = round(
        estimate.request_count
        * (
            estimate.assumed_prompt_tokens_per_request
            * estimate.prompt_price_per_1k_usd
            / 1000
            + estimate.assumed_completion_tokens_per_request
            * estimate.completion_price_per_1k_usd
            / 1000
        ),
        6,
    )
    if abs(recomputed_row_cost - estimate.estimated_cost_usd) > _COST_TOTAL_TOLERANCE:
        issues.append(
            f"cost estimate arithmetic mismatch for {identity}: expected "
            f"{recomputed_row_cost:.6f}, received {estimate.estimated_cost_usd:.6f}"
        )
    recomputed_reserved_cost = round(
        estimate.request_count
        * (
            estimate.assumed_prompt_tokens_per_request
            * estimate.prompt_price_per_1k_usd
            / 1000
            + estimate.max_completion_tokens_per_request
            * estimate.completion_price_per_1k_usd
            / 1000
        ),
        6,
    )
    if (
        abs(recomputed_reserved_cost - estimate.reserved_cost_usd)
        > _COST_TOTAL_TOLERANCE
    ):
        issues.append(
            f"cost reservation arithmetic mismatch for {identity}: expected "
            f"{recomputed_reserved_cost:.6f}, received "
            f"{estimate.reserved_cost_usd:.6f}"
        )
    return issues


def validate_cost_estimate(plan: ProviderExecutionPlan) -> list[str]:
    issues: list[str] = []
    model_estimates = plan.cost_estimate.models
    observed_rows = [
        (estimate.study_id, estimate.model_name) for estimate in model_estimates
    ]
    if len(set(observed_rows)) != len(observed_rows):
        issues.append("cost estimate contains duplicate study/model rows")
    primary_requests_per_model = (
        len(plan.incident_ids) * len(plan.arms) * plan.repeats
    )
    expected_counts = {
        ("provider_readiness", model_name): 1
        for model_name in plan.provider_readiness.model_names
    }
    expected_counts.update(
        {
            ("primary", model_name): primary_requests_per_model
            for model_name in plan.models
        }
    )
    for study in plan.diagnostic_studies:
        task_counts = study.settings.get("task_count_by_model")
        if not isinstance(task_counts, dict):
            issues.append(f"diagnostic study {study.study_id} has no model counts")
            continue
        for model_name in study.models:
            request_count = task_counts.get(model_name)
            if not isinstance(request_count, int) or request_count < 1:
                issues.append(
                    f"diagnostic study {study.study_id} has invalid count for "
                    f"{model_name}"
                )
                continue
            expected_counts[(study.study_id, model_name)] = request_count
    if set(observed_rows) != set(expected_counts):
        issues.append("cost estimate study/model set differs from execution plan")
    for estimate in model_estimates:
        issues.extend(
            _validate_model_cost_estimate(
                estimate,
                expected_counts.get((estimate.study_id, estimate.model_name)),
            )
        )
    recomputed_total = round(
        sum(estimate.estimated_cost_usd for estimate in model_estimates),
        6,
    )
    if (
        abs(recomputed_total - plan.cost_estimate.estimated_total_usd)
        > _COST_TOTAL_TOLERANCE
    ):
        issues.append("estimated total cost does not equal the per-model sum")
    if (
        plan.cost_estimate.approval_budget_with_headroom_usd
        < plan.cost_estimate.estimated_total_usd
    ):
        issues.append("approval cost budget is lower than the estimated total")
    observed_study_totals: dict[str, float] = {}
    for estimate in model_estimates:
        observed_study_totals[estimate.study_id] = round(
            observed_study_totals.get(estimate.study_id, 0.0)
            + estimate.estimated_cost_usd,
            6,
        )
    if observed_study_totals != plan.cost_estimate.study_estimated_cost_usd:
        issues.append("per-study cost totals differ from the per-model estimates")
    observed_reserved_totals: dict[str, float] = {}
    for estimate in model_estimates:
        observed_reserved_totals[estimate.study_id] = round(
            observed_reserved_totals.get(estimate.study_id, 0.0)
            + estimate.reserved_cost_usd,
            6,
        )
    if observed_reserved_totals != plan.cost_estimate.study_reserved_cost_usd:
        issues.append("per-study reserved costs differ from the per-model estimates")
    expected_studies = {study_id for study_id, _model_name in expected_counts}
    if set(plan.cost_estimate.study_budget_ceiling_usd) != expected_studies:
        issues.append("per-study budget ceilings differ from execution plan")
    applied_budget = round(
        sum(plan.cost_estimate.study_budget_ceiling_usd.values()), 2
    )
    if applied_budget != plan.cost_estimate.approval_budget_with_headroom_usd:
        issues.append("approved budget differs from the applied study ceilings")
    for study_id, estimated_cost in observed_study_totals.items():
        if plan.cost_estimate.study_budget_ceiling_usd.get(study_id, -1) < estimated_cost:
            issues.append(f"provider budget is below estimated cost for {study_id}")
        if (
            plan.cost_estimate.study_budget_ceiling_usd.get(study_id, -1)
            < observed_reserved_totals[study_id]
        ):
            issues.append(f"provider budget is below reserved cost for {study_id}")
    return issues


def build_validation_report(
    *,
    repo_root: Path,
    plan_path: Path,
    require_approval: bool,
    expected_execution: ExpectedExecutionRequest,
) -> dict[str, object]:
    paths = RepoPaths(repo_root=repo_root)
    plan = load_plan(plan_path)
    plan_inputs = resolved_plan_inputs(repo_root, plan)
    expected_primary_cell_count = (
        len(plan.incident_ids) * len(plan.models) * len(plan.arms) * plan.repeats
    )
    issues: list[str] = []
    metadata: dict[str, object] = {}
    if expected_primary_cell_count != plan.primary_cell_count:
        issues.append(
            "primary_cell_count mismatch: "
            f"plan={plan.primary_cell_count}, computed={expected_primary_cell_count}"
        )
    compare_expected_value(
        issues,
        field_name="dataset_release_id",
        expected=expected_execution.dataset_release_id,
        actual=plan.dataset_release_id,
    )
    compare_expected_incidents(
        issues,
        metadata,
        expected=expected_execution.incident_ids,
        actual=plan.incident_ids,
    )
    compare_expected_value(
        issues,
        field_name="models",
        expected=expected_execution.models,
        actual=plan.models,
    )
    compare_expected_value(
        issues,
        field_name="arms",
        expected=expected_execution.arms,
        actual=plan.arms,
    )
    compare_expected_value(
        issues,
        field_name="repeats",
        expected=expected_execution.repeats,
        actual=plan.repeats,
    )
    issues.extend(validate_approval(plan, require_approval))
    issues.extend(validate_cost_estimate(plan))
    issues.extend(validate_hashes(repo_root, plan.frozen_inputs))
    issues.extend(validate_protocol_freeze(repo_root, plan))
    issues.extend(validate_campaign_binding(repo_root, plan))
    issues.extend(validate_incidents(paths, plan.incident_ids))
    issues.extend(validate_arms(plan.arms))
    if "models_registry" in plan_inputs:
        issues.extend(
            validate_models(
                plan_inputs["models_registry"],
                plan,
            )
        )
    if {"action_catalog", "constraints"}.issubset(plan_inputs):
        issues.extend(
            validate_policy_surface(
                plan_inputs["action_catalog"],
                plan_inputs["constraints"],
            )
        )
    if {"policy_review_record", "constraints", "action_catalog"}.issubset(plan_inputs):
        issues.extend(
            validate_policy_review_record(
                plan_inputs["policy_review_record"],
                plan_inputs["constraints"],
                plan_inputs["action_catalog"],
            )
        )

    return {
        "status": "ok" if not issues else "failed",
        "plan_id": plan.plan_id,
        "plan_path": repo_relative_path(plan_path, repo_root),
        "require_approval": require_approval,
        "approval_status": plan.approval.status,
        "dataset_release_id": plan.dataset_release_id,
        "eval_protocol_version": plan.eval_protocol_version,
        "incident_count": len(plan.incident_ids),
        "model_count": len(plan.models),
        "arm_count": len(plan.arms),
        "repeats": plan.repeats,
        "primary_cell_count": expected_primary_cell_count,
        "cost_estimate": plan.cost_estimate.model_dump(mode="json"),
        "metadata": metadata,
        "issues": issues,
    }


def main() -> None:
    args = parse_args()
    repo_root = resolve_repo_root(args.repo_root)
    plan_path = resolve_path(args.plan_json, repo_root)
    output_path = resolve_path(args.output_json, repo_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = build_validation_report(
        repo_root=repo_root,
        plan_path=plan_path,
        require_approval=bool(args.require_approval),
        expected_execution=ExpectedExecutionRequest(
            dataset_release_id=str(args.expected_dataset_release_id).strip(),
            incident_ids=parse_csv_list(str(args.expected_incident_ids)),
            models=parse_csv_list(str(args.expected_models)),
            arms=parse_csv_list(str(args.expected_arms)),
            repeats=int(args.expected_repeats),
        ),
    )
    atomic_write_json(output_path, report)
    print(f"Wrote {output_path}")
    if report["status"] != "ok":
        raise SystemExit(f"Provider execution plan validation failed: {output_path}")
    print("Provider execution plan validation OK.")


if __name__ == "__main__":
    main()
