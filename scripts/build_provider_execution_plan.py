#!/usr/bin/env python3
"""Build a draft provider-execution plan from zero-cost evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from decimal import ROUND_CEILING, Decimal
from pathlib import Path

from pydantic import BaseModel, Field

from soc_llm_policy.corpus_campaign import (
    CorpusCampaignManifest,
    validate_campaign,
)
from soc_llm_policy.durable_io import atomic_write_json
from soc_llm_policy.execution_source import validate_execution_source_manifest
from soc_llm_policy.frozen_inputs import (
    PROVIDER_INPUT_ROLES,
    FrozenInputBinding,
    resolve_frozen_input_bindings,
)
from soc_llm_policy.llm_adapter import DEFAULT_MAX_COMPLETION_TOKENS
from soc_llm_policy.models_registry import (
    ModelProfile,
    load_model_registry,
    validate_pricing_metadata,
)
from soc_llm_policy.paths import RepoPaths, repo_relative_path, resolve_repo_root
from soc_llm_policy.pipeline import list_inbox_incidents
from soc_llm_policy.policy_review import validate_policy_review
from soc_llm_policy.protocol_freeze import PROTOCOL_FREEZE_SCHEMA
from soc_llm_policy.provider_plan import (
    CostEstimate,
    DiagnosticStudyPlan,
    ExecutionApproval,
    FrozenInput,
    ModelCostEstimate,
    ProviderExecutionPlan,
    ProviderReadinessSpec,
)
from soc_llm_policy.provider_readiness import (
    READINESS_MAX_COMPLETION_TOKENS,
    READINESS_PROMPT_TOKEN_BUDGET,
    READINESS_STUDY_ID,
    readiness_prompt_sha256,
)

DEFAULT_MODELS = ["openai_gpt55", "ollama_qwen36"]
DEFAULT_ARMS = ["llm_zero", "llm_policy_prompt"]
COST_HEADROOM_MULTIPLIER = 1.5


def _ceil_usd(value: float) -> float:
    return float(
        Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_CEILING)
    )


def _request_cost(
    *,
    request_count: int,
    prompt_tokens: int,
    completion_tokens: int,
    profile: ModelProfile,
) -> float:
    return request_count * (
        prompt_tokens * profile.prompt_price_per_1k_usd / 1000
        + completion_tokens * profile.completion_price_per_1k_usd / 1000
    )


class FreezeExpectation(BaseModel):
    dataset_release_id: str
    protocol_version: str
    incident_ids: list[str]
    model_names: list[str]
    arm_names: list[str]
    repeats: int
    policy_evidence_tier: str
    repeat_reduction: str


class IncidentSignalRecord(BaseModel):
    incident_id: str = Field(..., min_length=1)
    score: int = Field(..., ge=0)
    family_count: int = Field(..., ge=0)
    signals: dict[str, int] = Field(default_factory=dict)


class CandidateReport(BaseModel):
    ranked_incidents: list[IncidentSignalRecord]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument(
        "--candidates-json",
        type=Path,
        default=Path("results/analysis/provider_execution_candidates.json"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/analysis/provider_execution_plan.json"),
    )
    parser.add_argument(
        "--protocol-freeze-json",
        type=Path,
        default=Path("results/analysis/provider_protocol_freeze.json"),
    )
    parser.add_argument("--dataset-release-id", required=True)
    parser.add_argument("--eval-protocol-version", default="canonical_campaign")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Use every incident in the active corpus source.",
    )
    parser.add_argument("--corpus-campaign-manifest", type=Path, default=None)
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    parser.add_argument("--repeats", type=int, default=1)
    return parser.parse_args()


def parse_csv_list(value: str) -> list[str]:
    items: list[str] = []
    for raw_item in value.split(","):
        item = raw_item.strip()
        if item and item not in items:
            items.append(item)
    return items


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_candidate_report(path: Path) -> CandidateReport:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return CandidateReport.model_validate(payload)


def load_json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def resolve_output_path(path: Path, repo_root: Path) -> Path:
    if path.is_absolute():
        return path
    return repo_root / path


def load_candidate_incidents(path: Path, count: int) -> list[str]:
    report = load_candidate_report(path)
    selected: list[str] = []
    for record in report.ranked_incidents:
        if record.incident_id not in selected:
            selected.append(record.incident_id)
        if len(selected) >= count:
            break
    if len(selected) < count:
        raise ValueError(
            f"Candidate report contains {len(selected)} unique incidents; "
            f"{count} requested."
        )
    return selected


def validate_incident_ids(paths: RepoPaths, incident_ids: list[str]) -> None:
    available = set(list_inbox_incidents(paths))
    missing = [
        incident_id for incident_id in incident_ids if incident_id not in available
    ]
    if missing:
        raise FileNotFoundError(f"Incident IDs not found in dataset: {missing}")


def resolve_model_profiles(
    registry_path: Path,
    model_names: list[str],
) -> list[ModelProfile]:
    profiles = load_model_registry(registry_path)
    by_selector = {
        selector: profile
        for profile in profiles
        for selector in (profile.name, profile.deployment)
    }
    unresolved = [name for name in model_names if name not in by_selector]
    if unresolved:
        raise ValueError(f"Model names not found in registry: {unresolved}")
    return [by_selector[name] for name in model_names]


def estimate_cost(
    *,
    primary_profiles: list[ModelProfile],
    primary_requests_per_model: int,
    readiness_profiles: list[ModelProfile],
    diagnostic_studies: list[DiagnosticStudyPlan],
    all_profiles: list[ModelProfile],
) -> CostEstimate:
    profiles_by_name = {profile.name: profile for profile in all_profiles}
    estimates: list[ModelCostEstimate] = []
    for profile in readiness_profiles:
        validate_pricing_metadata(profile)
        estimated_cost = _request_cost(
            request_count=1,
            prompt_tokens=READINESS_PROMPT_TOKEN_BUDGET,
            completion_tokens=READINESS_MAX_COMPLETION_TOKENS,
            profile=profile,
        )
        estimates.append(
            ModelCostEstimate(
                study_id=READINESS_STUDY_ID,
                model_name=profile.name,
                provider=profile.provider,
                billing_mode=profile.billing_mode,
                request_count=1,
                assumed_prompt_tokens_per_request=READINESS_PROMPT_TOKEN_BUDGET,
                assumed_completion_tokens_per_request=(
                    READINESS_MAX_COMPLETION_TOKENS
                ),
                max_completion_tokens_per_request=(
                    READINESS_MAX_COMPLETION_TOKENS
                ),
                prompt_price_per_1k_usd=profile.prompt_price_per_1k_usd,
                completion_price_per_1k_usd=profile.completion_price_per_1k_usd,
                pricing_source_url=profile.pricing_source_url,
                pricing_verified_on=profile.pricing_verified_on,
                pricing_valid_through=profile.pricing_valid_through,
                estimated_cost_usd=round(estimated_cost, 6),
                reserved_cost_usd=round(estimated_cost, 6),
            )
        )
    for profile in primary_profiles:
        validate_pricing_metadata(profile)
        if profile.planning_prompt_tokens is None:
            raise ValueError(
                f"Model {profile.name} has no planning_prompt_tokens assumption."
            )
        if profile.planning_completion_tokens is None:
            raise ValueError(
                f"Model {profile.name} has no planning_completion_tokens assumption."
            )
        estimated_cost = _request_cost(
            request_count=primary_requests_per_model,
            prompt_tokens=profile.planning_prompt_tokens,
            completion_tokens=profile.planning_completion_tokens,
            profile=profile,
        )
        reserved_cost = _request_cost(
            request_count=primary_requests_per_model,
            prompt_tokens=profile.planning_prompt_tokens,
            completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
            profile=profile,
        )
        estimates.append(
            ModelCostEstimate(
                study_id="primary",
                model_name=profile.name,
                provider=profile.provider,
                billing_mode=profile.billing_mode,
                request_count=primary_requests_per_model,
                assumed_prompt_tokens_per_request=profile.planning_prompt_tokens,
                assumed_completion_tokens_per_request=(
                    profile.planning_completion_tokens
                ),
                max_completion_tokens_per_request=DEFAULT_MAX_COMPLETION_TOKENS,
                prompt_price_per_1k_usd=profile.prompt_price_per_1k_usd,
                completion_price_per_1k_usd=profile.completion_price_per_1k_usd,
                pricing_source_url=profile.pricing_source_url,
                pricing_verified_on=profile.pricing_verified_on,
                pricing_valid_through=profile.pricing_valid_through,
                estimated_cost_usd=round(estimated_cost, 6),
                reserved_cost_usd=round(reserved_cost, 6),
            )
        )
    for study in diagnostic_studies:
        assumptions = study.settings.get("token_assumptions")
        if not isinstance(assumptions, dict):
            raise ValueError(f"Diagnostic study {study.study_id} has no token assumptions.")
        per_model_counts = study.settings.get("task_count_by_model")
        if not isinstance(per_model_counts, dict):
            raise ValueError(f"Diagnostic study {study.study_id} has no model counts.")
        for model_name in study.models:
            profile = profiles_by_name.get(model_name)
            if profile is None:
                raise ValueError(f"Diagnostic model is missing from registry: {model_name}")
            validate_pricing_metadata(profile)
            token_values = assumptions.get(model_name)
            request_count = per_model_counts.get(model_name)
            if not isinstance(token_values, dict) or not isinstance(
                request_count, int
            ):
                raise ValueError(
                    f"Diagnostic study {study.study_id} has incomplete cost inputs "
                    f"for {model_name}."
                )
            prompt_tokens = int(token_values["prompt_tokens"])
            completion_tokens = int(token_values["completion_tokens"])
            estimated_cost = _request_cost(
                request_count=request_count,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                profile=profile,
            )
            reserved_cost = _request_cost(
                request_count=request_count,
                prompt_tokens=prompt_tokens,
                completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
                profile=profile,
            )
            estimates.append(
                ModelCostEstimate(
                    study_id=study.study_id,
                    model_name=profile.name,
                    provider=profile.provider,
                    billing_mode=profile.billing_mode,
                    request_count=request_count,
                    assumed_prompt_tokens_per_request=prompt_tokens,
                    assumed_completion_tokens_per_request=completion_tokens,
                    max_completion_tokens_per_request=(
                        DEFAULT_MAX_COMPLETION_TOKENS
                    ),
                    prompt_price_per_1k_usd=profile.prompt_price_per_1k_usd,
                    completion_price_per_1k_usd=profile.completion_price_per_1k_usd,
                    pricing_source_url=profile.pricing_source_url,
                    pricing_verified_on=profile.pricing_verified_on,
                    pricing_valid_through=profile.pricing_valid_through,
                    estimated_cost_usd=round(estimated_cost, 6),
                    reserved_cost_usd=round(reserved_cost, 6),
                )
            )
    total = sum(item.estimated_cost_usd for item in estimates)
    study_totals: dict[str, float] = {}
    for item in estimates:
        study_totals[item.study_id] = (
            study_totals.get(item.study_id, 0.0) + item.estimated_cost_usd
        )
    study_totals = {name: round(value, 6) for name, value in study_totals.items()}
    study_reserved_totals: dict[str, float] = {}
    for item in estimates:
        study_reserved_totals[item.study_id] = (
            study_reserved_totals.get(item.study_id, 0.0) + item.reserved_cost_usd
        )
    study_reserved_totals = {
        name: round(value, 6) for name, value in study_reserved_totals.items()
    }
    study_budget_ceilings = {
        name: _ceil_usd(
            max(
                value * COST_HEADROOM_MULTIPLIER,
                study_reserved_totals[name],
            )
        )
        for name, value in study_totals.items()
    }
    return CostEstimate(
        basis=(
            "Per-model token counts are planning estimates, not scientific "
            "evidence. Expected cost uses the planning completion length. The "
            "runtime guard instead reserves the fixed maximum completion-token "
            "allowance before each request. Each study ceiling covers the larger "
            "of that reserved exposure and 1.5 times expected cost. Provider prices, "
            "verification dates, and token assumptions are fixed in the model "
            "registry. The append-only provider log records realized usage and "
            "cost."
        ),
        models=estimates,
        estimated_total_usd=round(total, 6),
        approval_budget_with_headroom_usd=_ceil_usd(
            sum(study_budget_ceilings.values())
        ),
        study_estimated_cost_usd=study_totals,
        study_reserved_cost_usd=study_reserved_totals,
        study_budget_ceiling_usd=study_budget_ceilings,
    )


def _freeze_input(payload: object, role: str) -> FrozenInput:
    if not isinstance(payload, dict):
        raise ValueError(f"Protocol freeze is missing diagnostic input: {role}")
    path = payload.get("path")
    sha256 = payload.get("sha256")
    return FrozenInput(path=str(path), sha256=str(sha256))


def diagnostic_plans(protocol_freeze: dict[str, object]) -> list[DiagnosticStudyPlan]:
    raw_studies = protocol_freeze.get("diagnostic_studies")
    if not isinstance(raw_studies, dict):
        raise ValueError("Protocol freeze is missing diagnostic studies.")
    input_hashes = protocol_freeze.get("input_hashes")
    if not isinstance(input_hashes, dict):
        raise ValueError("Protocol freeze is missing input hashes.")
    studies: list[DiagnosticStudyPlan] = []
    for study_id in (
        "independent_judge",
        "judge_ablation",
        "adversarial",
        "projection_fidelity",
    ):
        raw = raw_studies.get(study_id)
        if not isinstance(raw, dict):
            raise ValueError(f"Protocol freeze is missing diagnostic study: {study_id}")
        if study_id in {"independent_judge", "judge_ablation"}:
            models = [str(raw["model"])]
            assumptions = {models[0]: raw["token_assumption"]}
            counts = {models[0]: int(raw["task_count_ceiling"])}
            roles = ["llm_judge_protocol", "campaign_diagnostics"]
        elif study_id == "adversarial":
            models = [str(value) for value in raw["models"]]
            assumptions = raw["token_assumptions"]
            per_model = (
                int(raw["scenario_count"])
                * len(raw["arms"])
                * int(raw["repeats"])
            )
            counts = {model: per_model for model in models}
            roles = [
                "adversarial_pair_manifest",
                "adversarial_scenario_manifest",
                "campaign_diagnostics",
            ]
        else:
            models = [str(value) for value in raw["models"]]
            assumptions = raw["token_assumptions"]
            counts = {model: int(raw["plan_count"]) for model in models}
            roles = ["projection_fidelity_probe", "campaign_diagnostics"]
        settings = dict(raw)
        settings["token_assumptions"] = assumptions
        settings["task_count_by_model"] = counts
        studies.append(
            DiagnosticStudyPlan(
                study_id=study_id,
                kind=str(raw["kind"]),
                task_count_ceiling=int(raw["task_count_ceiling"]),
                models=models,
                depends_on=str(raw["depends_on"]),
                settings=settings,
                frozen_inputs={
                    role: _freeze_input(input_hashes.get(role), role) for role in roles
                },
            )
        )
    expected = raw_studies.get("diagnostic_task_ceiling")
    observed = sum(study.task_count_ceiling for study in studies)
    if expected != observed:
        raise ValueError("Protocol freeze diagnostic task ceiling is inconsistent.")
    return studies


def validate_protocol_freeze(
    freeze: dict[str, object],
    expectation: FreezeExpectation,
) -> None:
    if freeze.get("schema_version") != PROTOCOL_FREEZE_SCHEMA:
        raise ValueError(
            "Provider plan requires protocol-freeze schema version "
            f"{PROTOCOL_FREEZE_SCHEMA}."
        )
    if freeze.get("state") != "pre_execution_protocol_freeze":
        raise ValueError("Provider plan requires a pre-execution protocol freeze.")
    if freeze.get("constitutes_model_evidence") is not False:
        raise ValueError("Provider plan requires a non-evidentiary protocol freeze.")
    expected = {
        "dataset_release_id": expectation.dataset_release_id,
        "eval_protocol_version": expectation.protocol_version,
        "incident_ids": expectation.incident_ids,
        "official_models": expectation.model_names,
        "arms": expectation.arm_names,
        "repeats": expectation.repeats,
        "policy_evidence_tier": expectation.policy_evidence_tier,
        "repeat_reduction": expectation.repeat_reduction,
    }
    mismatches = [
        field_name
        for field_name, expected_value in expected.items()
        if freeze.get(field_name) != expected_value
    ]
    if mismatches:
        raise ValueError(
            "Protocol freeze does not match the requested provider plan: "
            + ", ".join(mismatches)
        )
    if freeze.get("execution_phase") != "primary":
        raise ValueError("Provider plan requires the canonical primary protocol freeze.")
    input_hashes = freeze.get("input_hashes")
    if not isinstance(input_hashes, dict):
        raise ValueError("Provider plan requires protocol-freeze input hashes.")
    preflight = input_hashes.get("policy_preflight_manifest")
    if not isinstance(preflight, dict):
        raise ValueError("Provider plan requires a policy-preflight manifest.")
    if not isinstance(preflight.get("path"), str) or not isinstance(
        preflight.get("sha256"), str
    ):
        raise ValueError("Policy-preflight input hash is incomplete.")
    missing_roles = sorted(set(PROVIDER_INPUT_ROLES) - set(input_hashes))
    if missing_roles:
        raise ValueError(
            "Protocol freeze is missing provider inputs: " + ", ".join(missing_roles)
        )
    source_entry = input_hashes.get("execution_source_manifest")
    source_identity = freeze.get("source_identity")
    if not isinstance(source_entry, dict) or not isinstance(source_identity, dict):
        raise ValueError("Provider plan requires a sealed execution source.")
    if (
        source_identity.get("kind") != "execution_source_manifest"
        or source_identity.get("path") != source_entry.get("path")
        or source_identity.get("sha256") != source_entry.get("sha256")
    ):
        raise ValueError("Protocol freeze execution source binding is inconsistent.")


def frozen_inputs(
    paths: RepoPaths,
    candidate_path: Path | None,
    protocol_freeze_path: Path,
    freeze_bindings: dict[str, FrozenInputBinding],
    campaign_path: Path | None,
) -> dict[str, FrozenInput]:
    generated_inputs = {"protocol_freeze": protocol_freeze_path}
    if candidate_path is not None:
        generated_inputs["candidate_report"] = candidate_path
    if campaign_path is not None:
        generated_inputs["corpus_campaign_manifest"] = campaign_path
    bindings = {
        role: FrozenInput(
            path=freeze_bindings[role].path,
            sha256=freeze_bindings[role].sha256,
        )
        for role in PROVIDER_INPUT_ROLES
    }
    bindings.update(
        {
            name: FrozenInput(
                path=_required_repo_relative_path(path, paths.repo_root),
                sha256=sha256_file(path),
            )
            for name, path in generated_inputs.items()
        }
    )
    return bindings


def _campaign_for_plan(
    *,
    args: argparse.Namespace,
    repo_root: Path,
    protocol_freeze: dict[str, object],
    incident_ids: list[str],
) -> tuple[CorpusCampaignManifest | None, Path | None]:
    if args.corpus_campaign_manifest is None:
        return None, None
    campaign_path = resolve_output_path(args.corpus_campaign_manifest, repo_root)
    campaign = validate_campaign(
        repo_root=repo_root,
        manifest_path=campaign_path,
    )
    if sorted(campaign.incident_partitions) != sorted(incident_ids):
        raise ValueError("Corpus campaign differs from the provider incident set.")
    input_hashes = protocol_freeze.get("input_hashes")
    entry = (
        input_hashes.get("corpus_campaign_manifest")
        if isinstance(input_hashes, dict)
        else None
    )
    expected_path = repo_relative_path(campaign_path, repo_root)
    expected_hash = sha256_file(campaign_path)
    if not isinstance(entry, dict) or entry.get("path") != expected_path:
        raise ValueError("Protocol freeze does not bind the corpus campaign path.")
    if entry.get("sha256") != expected_hash:
        raise ValueError("Protocol freeze does not bind the corpus campaign hash.")
    return campaign, campaign_path


def _required_repo_relative_path(path: Path, repo_root: Path) -> str:
    relative_path = repo_relative_path(path, repo_root)
    if Path(relative_path).is_absolute():
        raise ValueError(f"Provider-plan input must stay inside the repository: {path}")
    return relative_path


def build_plan(args: argparse.Namespace) -> ProviderExecutionPlan:
    repo_root = resolve_repo_root(args.repo_root)
    paths = RepoPaths(repo_root=repo_root)
    candidate_path = (
        None if args.all else resolve_output_path(args.candidates_json, repo_root)
    )
    protocol_freeze_path = resolve_output_path(args.protocol_freeze_json, repo_root)
    protocol_freeze = load_json_object(protocol_freeze_path)
    incident_ids = (
        list_inbox_incidents(paths)
        if args.all
        else load_candidate_incidents(candidate_path, int(args.count))
    )
    model_names = parse_csv_list(str(args.models))
    arm_names = parse_csv_list(str(args.arms))
    campaign, campaign_path = _campaign_for_plan(
        args=args,
        repo_root=repo_root,
        protocol_freeze=protocol_freeze,
        incident_ids=incident_ids,
    )

    validate_incident_ids(paths, incident_ids)
    if not arm_names:
        raise ValueError("At least one arm is required.")
    if int(args.repeats) < 1:
        raise ValueError("--repeats must be >= 1.")
    validate_protocol_freeze(
        protocol_freeze,
        FreezeExpectation(
            dataset_release_id=str(args.dataset_release_id).strip(),
            protocol_version=str(args.eval_protocol_version).strip(),
            incident_ids=incident_ids,
            model_names=model_names,
            arm_names=arm_names,
            repeats=int(args.repeats),
            policy_evidence_tier="operational",
            repeat_reduction=str(protocol_freeze.get("repeat_reduction") or ""),
        ),
    )
    freeze_bindings = resolve_frozen_input_bindings(
        protocol_freeze.get("input_hashes"),
        repo_root,
        required_roles=(*PROVIDER_INPUT_ROLES, "execution_source_manifest"),
    )
    validate_execution_source_manifest(
        repo_root=repo_root,
        manifest_path=freeze_bindings["execution_source_manifest"].resolved_path,
    )
    validate_policy_review(
        freeze_bindings["policy_review_record"].resolved_path,
        freeze_bindings["constraints"].resolved_path,
        freeze_bindings["action_catalog"].resolved_path,
    )
    model_profiles = resolve_model_profiles(
        freeze_bindings["models_registry"].resolved_path,
        model_names,
    )
    all_model_profiles = load_model_registry(
        freeze_bindings["models_registry"].resolved_path
    )
    diagnostics = diagnostic_plans(protocol_freeze)
    profiles_by_name = {profile.name: profile for profile in all_model_profiles}
    readiness_model_names = list(
        dict.fromkeys(
            [
                *model_names,
                *(
                    model_name
                    for study in diagnostics
                    for model_name in study.models
                ),
            ]
        )
    )
    readiness_profiles = [profiles_by_name[name] for name in readiness_model_names]

    primary_cell_count = (
        len(incident_ids) * len(model_names) * len(arm_names) * int(args.repeats)
    )
    if protocol_freeze.get("primary_cell_count") != primary_cell_count:
        raise ValueError(
            "Protocol freeze primary_cell_count does not match the provider plan."
        )
    requests_per_model = len(incident_ids) * len(arm_names) * int(args.repeats)
    freeze_sha256 = sha256_file(protocol_freeze_path)
    generated_at_utc = protocol_freeze.get("generated_at_utc")
    if not isinstance(generated_at_utc, str) or not generated_at_utc:
        raise ValueError("Protocol freeze has no generation timestamp.")
    provider_attempt_limit = int(
        protocol_freeze["retry_policy"]["llm_attempt_limit"]
    )
    diagnostic_task_ceiling = sum(
        study.task_count_ceiling for study in diagnostics
    )
    scientific_task_ceiling = primary_cell_count + diagnostic_task_ceiling
    readiness = ProviderReadinessSpec(
        model_names=readiness_model_names,
        request_count=len(readiness_model_names),
        provider_attempt_limit=provider_attempt_limit,
        provider_attempt_ceiling=len(readiness_model_names) * provider_attempt_limit,
        max_completion_tokens=READINESS_MAX_COMPLETION_TOKENS,
        prompt_sha256=readiness_prompt_sha256(),
    )
    return ProviderExecutionPlan(
        generated_at_utc=generated_at_utc,
        plan_id=f"measurement_{freeze_sha256[:16]}",
        purpose=(
            "Provider execution plan derived from the single published protocol "
            "freeze. Explicit approval is required before any paid provider call."
        ),
        approval=ExecutionApproval(),
        dataset_release_id=str(args.dataset_release_id).strip(),
        eval_protocol_version=str(args.eval_protocol_version).strip(),
        incident_ids=incident_ids,
        models=model_names,
        arms=arm_names,
        repeats=int(args.repeats),
        repeat_reduction=str(protocol_freeze["repeat_reduction"]),
        policy_evidence_tier=str(protocol_freeze["policy_evidence_tier"]),
        primary_cell_count=primary_cell_count,
        diagnostic_task_ceiling=diagnostic_task_ceiling,
        scientific_task_ceiling=scientific_task_ceiling,
        provider_transport_attempt_limit=provider_attempt_limit,
        provider_transport_attempt_ceiling=(
            scientific_task_ceiling + readiness.request_count
        )
        * provider_attempt_limit,
        provider_readiness=readiness,
        execution_sequence=[
            str(value)
            for value in protocol_freeze["diagnostic_studies"]["execution_sequence"]
        ],
        diagnostic_studies=diagnostics,
        cost_estimate=estimate_cost(
            primary_profiles=model_profiles,
            primary_requests_per_model=requests_per_model,
            readiness_profiles=readiness_profiles,
            diagnostic_studies=diagnostics,
            all_profiles=all_model_profiles,
        ),
        frozen_inputs=frozen_inputs(
            paths,
            candidate_path,
            protocol_freeze_path,
            freeze_bindings,
            campaign_path,
        ),
        selection_basis=(
            {
                "source": repo_relative_path(paths.inbox_incidents_dir, repo_root),
                "method": (
                    "all campaign incidents"
                    if campaign is not None
                    else "all corpus incidents"
                ),
                "incident_count": len(incident_ids),
                "partitions": (
                    {
                        label: partition.incident_count
                        for label, partition in campaign.partitions.items()
                    }
                    if campaign is not None
                    else {}
                ),
            }
            if candidate_path is None
            else {
                "source": repo_relative_path(candidate_path, repo_root),
                "method": "top-ranked zero-cost signal coverage",
                "incident_count": len(incident_ids),
            }
        ),
        execution_rules=[
            "Do not execute provider calls while approval.status is draft.",
            "Do not change corpus, policy, model registry, arms, repeats, or "
            "repeat reduction after approval.",
            "Rerun zero-cost validation immediately before execution.",
            "Treat this run as a separate measurement object if policy or "
            "corpus changes.",
        ],
    )


def main() -> None:
    args = parse_args()
    repo_root = resolve_repo_root(args.repo_root)
    output_path = resolve_output_path(args.output_json, repo_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plan = build_plan(args)
    atomic_write_json(output_path, plan.model_dump(mode="json"))
    print(f"Wrote {output_path}")
    print("Plan status: draft. Manual approval is required before provider execution.")


if __name__ == "__main__":
    main()
