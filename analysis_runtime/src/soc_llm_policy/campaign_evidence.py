"""Bind every terminal study in one provider campaign to one evidence manifest."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from soc_llm_policy.campaign_diagnostics import (
    CompletedDiagnosticStudy,
    validate_campaign_diagnostics,
    validate_completed_diagnostic_study,
)
from soc_llm_policy.durable_io import atomic_write_json
from soc_llm_policy.execution_evidence import (
    ExecutionEvidence,
    load_execution_evidence,
)
from soc_llm_policy.preflight_evidence import sha256_file
from soc_llm_policy.provider_cost import ProviderCostTracker
from soc_llm_policy.provider_plan import ProviderExecutionPlan, load_provider_plan
from soc_llm_policy.provider_readiness import load_provider_readiness
from soc_llm_policy.publication_redaction import chain_digest, expected_digest

CAMPAIGN_EVIDENCE_SCHEMA = 1
CAMPAIGN_EVIDENCE_STATE = "terminal_campaign_evidence"
_DERIVED_REPORT_SCHEMA = 2

_DIAGNOSTIC_STUDIES = (
    "independent_judge",
    "judge_ablation",
    "projection_fidelity",
)
_STATIC_REPORT_NAMES = (
    "llm_judge_verdicts.json",
    "llm_judge_run_report.json",
    "judge_ablation_report.json",
    "separation_report.json",
    "separation_input.schema.json",
    "separation_report.schema.json",
    "llm_judge_input.schema.json",
    "adversarial_report.json",
    "projection_fidelity_report.json",
    "projection_diagnostics.json",
    "stability_diagnostics.json",
    "repeat_reduction_sensitivity.json",
)


@dataclass(frozen=True)
class CampaignEvidence:
    manifest_path: Path
    campaign_id: str
    plan_path: Path
    protocol_freeze_path: Path
    provider_readiness_path: Path
    provider_readiness_attempt_log_path: Path
    primary_evidence: ExecutionEvidence
    adversarial_evidence: ExecutionEvidence
    diagnostic_run_dirs: dict[str, Path]
    report_files: dict[str, Path]
    accounting: dict[str, Any]


@dataclass(frozen=True)
class ProviderAttemptAccounting:
    request_count: int
    observation_count: int
    realized_cost_usd: float
    ambiguous_reserved_cost_usd: float

    @property
    def budget_exposure_usd(self) -> float:
        return self.realized_cost_usd + self.ambiguous_reserved_cost_usd


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def _inside_repo(path: Path, repo_root: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    root = repo_root.expanduser().resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"{label} must remain inside the repository.")
    return resolved


def _binding(path: Path, repo_root: Path) -> dict[str, str]:
    path = _inside_repo(path, repo_root, "Campaign evidence input")
    if not path.is_file():
        raise FileNotFoundError(f"Campaign evidence input is missing: {path}")
    return {
        "path": path.relative_to(repo_root).as_posix(),
        "sha256": chain_digest(repo_root, path, sha256_file(path)),
    }


def _resolve_binding(value: object, repo_root: Path, label: str) -> Path:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a file binding.")
    relative_path = value.get("path")
    expected_hash = value.get("sha256")
    if not isinstance(relative_path, str) or not isinstance(expected_hash, str):
        raise ValueError(f"{label} is incomplete.")
    candidate = Path(relative_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"{label} escapes the repository.")
    path = repo_root / candidate
    if not path.is_file() or sha256_file(path) != expected_digest(
        repo_root, path, expected_hash
    ):
        raise ValueError(f"{label} changed after the campaign was sealed.")
    return path


def _plan_freeze(plan: ProviderExecutionPlan, repo_root: Path) -> tuple[Path, str]:
    binding = plan.frozen_inputs.get("protocol_freeze")
    if binding is None:
        raise ValueError("Provider plan does not bind a protocol freeze.")
    freeze_path = _inside_repo(
        repo_root / binding.path,
        repo_root,
        "Protocol freeze",
    )
    if not freeze_path.is_file() or sha256_file(freeze_path) != expected_digest(
        repo_root, freeze_path, binding.sha256
    ):
        raise ValueError("Provider plan protocol-freeze binding is invalid.")
    return freeze_path, binding.sha256


def _attempt_accounting(path: Path) -> ProviderAttemptAccounting:
    if not path.is_file():
        return ProviderAttemptAccounting(0, 0, 0.0, 0.0)
    tracker = ProviderCostTracker(path)
    tracker.refresh()
    if tracker.unknown_cost_count:
        raise ValueError(f"Provider cost accounting is incomplete: {path}")
    return ProviderAttemptAccounting(
        request_count=tracker.request_count,
        observation_count=tracker.response_count,
        realized_cost_usd=float(tracker.accrued_usd),
        ambiguous_reserved_cost_usd=float(tracker.reserved_exposure_usd),
    )


def _status_counts(ledger_path: Path, row_key: str) -> dict[str, int]:
    ledger = _read_json(ledger_path)
    rows = ledger.get(row_key)
    if not isinstance(rows, list):
        raise ValueError(f"{ledger_path} has no {row_key} list.")
    observed: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("status"), str):
            raise ValueError(f"{ledger_path} contains an invalid status row.")
        status = str(row["status"])
        observed[status] = observed.get(status, 0) + 1
    declared = ledger.get("status_counts")
    known_statuses = {
        "planned",
        "running",
        "completed",
        "refused",
        "parse_failed",
        "failed",
    }
    if (
        not isinstance(declared, dict)
        or any(
            key not in known_statuses or type(count) is not int or count < 0
            for key, count in declared.items()
        )
        or {key: count for key, count in declared.items() if count != 0} != observed
    ):
        raise ValueError(f"{ledger_path} status counts are inconsistent.")
    return dict(sorted(observed.items()))


def _execution_study_accounting(
    evidence: ExecutionEvidence,
    budget_ceiling_usd: float,
) -> dict[str, Any]:
    matrix = _read_json(evidence.files["run_matrix"].path)
    cells = matrix.get("cells")
    if not isinstance(cells, list) or matrix.get("planned_cell_count") != len(cells):
        raise ValueError("Execution evidence has inconsistent matrix accounting.")
    statuses = _status_counts(evidence.files["ledger"].path, "cells")
    attempts = _attempt_accounting(evidence.files["provider_attempts"].path)
    if attempts.observation_count != len(cells):
        raise ValueError("Every execution cell must bind one provider observation.")
    if attempts.budget_exposure_usd > budget_ceiling_usd:
        raise ValueError("Execution-study cost exposure exceeds its approved ceiling.")
    return {
        "scientific_task_count": len(cells),
        "status_counts": statuses,
        "provider_request_count": attempts.request_count,
        "provider_observation_count": attempts.observation_count,
        "realized_cost_usd": round(attempts.realized_cost_usd, 6),
        "ambiguous_reserved_cost_usd": round(attempts.ambiguous_reserved_cost_usd, 6),
        "budget_exposure_usd": round(attempts.budget_exposure_usd, 6),
        "budget_ceiling_usd": budget_ceiling_usd,
    }


def _diagnostic_snapshot(
    summary: CompletedDiagnosticStudy,
    repo_root: Path,
) -> dict[str, object]:
    if (summary.run_dir / "run_lease.json").exists():
        raise ValueError(
            f"Diagnostic study still has an active lease: {summary.study_id}"
        )
    files = sorted(path for path in summary.run_dir.rglob("*") if path.is_file())
    required = {"task_matrix.json", "diagnostic_ledger.json"}
    if not required.issubset(path.name for path in files):
        raise ValueError(
            f"Diagnostic study is missing its durable state: {summary.study_id}"
        )
    if (
        summary.provider_task_count
        and not (summary.run_dir / "provider_attempts.jsonl").is_file()
    ):
        raise FileNotFoundError(
            f"Diagnostic provider-attempt log is missing: {summary.study_id}"
        )
    return {
        "path": summary.run_dir.relative_to(repo_root).as_posix(),
        "files": {
            path.relative_to(summary.run_dir).as_posix(): sha256_file(path)
            for path in files
        },
    }


def _snapshot_matches_release(
    source_snapshot: object,
    release_snapshot: object,
    repo_root: Path,
) -> bool:
    if not isinstance(source_snapshot, dict) or not isinstance(release_snapshot, dict):
        return False
    if source_snapshot.keys() != release_snapshot.keys():
        return False
    release_root = release_snapshot.get("path")
    source_files = source_snapshot.get("files")
    release_files = release_snapshot.get("files")
    if not isinstance(release_root, str) or source_snapshot.get("path") != release_root:
        return False
    if not isinstance(source_files, dict) or not isinstance(release_files, dict):
        return False
    if source_files.keys() != release_files.keys():
        return False
    for relative_path, source_hash in source_files.items():
        if not isinstance(relative_path, str) or not isinstance(source_hash, str):
            return False
        release_path = repo_root / release_root / relative_path
        if (
            expected_digest(repo_root, release_path, source_hash)
            != release_files[relative_path]
        ):
            return False
    return True


def _diagnostic_study_matches_release(
    observed_study: object,
    expected_study: object,
    repo_root: Path,
) -> bool:
    if not isinstance(observed_study, dict) or not isinstance(expected_study, dict):
        return False
    if observed_study.keys() != expected_study.keys():
        return False
    return _snapshot_matches_release(
        observed_study.get("snapshot"),
        expected_study.get("snapshot"),
        repo_root,
    )


def _studies_match_release(observed: object, expected: object, repo_root: Path) -> bool:
    if not isinstance(observed, dict) or not isinstance(expected, dict):
        return False
    if observed.keys() != expected.keys():
        return False
    for study_id, expected_study in expected.items():
        observed_study = observed[study_id]
        if study_id in _DIAGNOSTIC_STUDIES:
            if not _diagnostic_study_matches_release(
                observed_study, expected_study, repo_root
            ):
                return False
        elif observed_study != expected_study:
            return False
    return True


def _diagnostic_accounting(
    summary: CompletedDiagnosticStudy,
    budget_ceiling_usd: float,
) -> dict[str, Any]:
    attempts = _attempt_accounting(summary.run_dir / "provider_attempts.jsonl")
    if attempts.observation_count != summary.provider_task_count:
        raise ValueError(
            f"Diagnostic observations do not cover {summary.study_id} tasks."
        )
    if attempts.budget_exposure_usd > budget_ceiling_usd:
        raise ValueError(
            f"Diagnostic cost exposure exceeds its ceiling: {summary.study_id}"
        )
    return {
        "population_source_count": summary.population_source_count,
        "selected_source_count": summary.selected_source_count,
        "provider_task_count": summary.provider_task_count,
        "unavailable_source_count": summary.unavailable_source_count,
        "status_counts": summary.status_counts,
        "output_count": summary.output_count,
        "provider_request_count": attempts.request_count,
        "provider_observation_count": attempts.observation_count,
        "realized_cost_usd": round(attempts.realized_cost_usd, 6),
        "ambiguous_reserved_cost_usd": round(attempts.ambiguous_reserved_cost_usd, 6),
        "budget_exposure_usd": round(attempts.budget_exposure_usd, 6),
        "budget_ceiling_usd": budget_ceiling_usd,
    }


def _report_paths(plan: ProviderExecutionPlan, repo_root: Path) -> dict[str, Path]:
    analysis_dir = repo_root / "artifact_outputs" / "analysis"
    if not analysis_dir.is_dir():
        analysis_dir = repo_root / "results" / "analysis"
    names = list(_STATIC_REPORT_NAMES)
    fidelity = next(
        study
        for study in plan.diagnostic_studies
        if study.study_id == "projection_fidelity"
    )
    names.extend(
        f"projection_fidelity_extractions_{model_name}.json"
        for model_name in fidelity.models
    )
    paths = {name: analysis_dir / name for name in sorted(names)}
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Campaign reports are missing: {missing}")
    return paths


def _validate_reports(
    *,
    plan: ProviderExecutionPlan,
    freeze_sha256: str,
    reports: dict[str, Path],
    primary: dict[str, Any],
    adversarial: dict[str, Any],
    diagnostics: dict[str, dict[str, Any]],
    primary_evidence: ExecutionEvidence,
) -> None:
    judge = _read_json(reports["llm_judge_run_report.json"])
    if (
        judge.get("run_id") != plan.plan_id
        or judge.get("protocol_freeze_sha256") != freeze_sha256
        or judge.get("provider_task_count")
        != diagnostics["independent_judge"]["provider_task_count"]
        or judge.get("status_counts")
        != diagnostics["independent_judge"]["status_counts"]
    ):
        raise ValueError("Independent-judge report differs from terminal evidence.")
    verdicts = _read_json(reports["llm_judge_verdicts.json"])
    verdict_rows = verdicts.get("verdicts")
    if (
        verdicts.get("run_id") != plan.plan_id
        or not isinstance(verdict_rows, list)
        or len(verdict_rows) != diagnostics["independent_judge"]["output_count"]
    ):
        raise ValueError("Independent-judge verdicts differ from terminal evidence.")
    ablation = _read_json(reports["judge_ablation_report.json"])
    if (
        ablation.get("run_id") != plan.plan_id
        or ablation.get("protocol_freeze_sha256") != freeze_sha256
        or ablation.get("provider_task_count")
        != diagnostics["judge_ablation"]["provider_task_count"]
        or ablation.get("status_counts")
        != diagnostics["judge_ablation"]["status_counts"]
    ):
        raise ValueError("Judge-ablation report differs from terminal evidence.")
    separation = _read_json(reports["separation_report.json"])
    separation_body = separation.get("report")
    if (
        separation.get("run_id") != plan.plan_id
        or separation.get("constitutes_model_evidence") is not True
        or not isinstance(separation_body, dict)
        or separation_body.get("total_cells")
        != diagnostics["independent_judge"]["provider_task_count"]
    ):
        raise ValueError("Separation report differs from primary evidence.")
    adversarial_report = _read_json(reports["adversarial_report.json"])
    if adversarial_report.get("planned_pair_count") * 2 != adversarial[
        "scientific_task_count"
    ] or adversarial_report.get("pair_count", 0) + adversarial_report.get(
        "incomplete_pair_count", 0
    ) != adversarial_report.get("planned_pair_count"):
        raise ValueError("Adversarial report loses planned-pair accounting.")
    fidelity = _read_json(reports["projection_fidelity_report.json"])
    fidelity_diagnostic = diagnostics["projection_fidelity"]
    if (
        fidelity.get("model_count")
        != len(
            next(
                study.models
                for study in plan.diagnostic_studies
                if study.study_id == "projection_fidelity"
            )
        )
        or fidelity.get("overall_scored_plans", 0)
        + fidelity.get("overall_refused_plans", 0)
        + fidelity.get("overall_parse_failed_plans", 0)
        + fidelity.get("overall_output_unavailable_plans", 0)
        != fidelity_diagnostic["provider_task_count"]
        or fidelity.get("overall_output_unavailable_plans", 0)
        != fidelity_diagnostic["status_counts"].get("failed", 0)
    ):
        raise ValueError("Projection-fidelity report differs from terminal evidence.")

    bundle = _read_json(primary_evidence.files["analysis_bundle"].path)
    bundle_rows = bundle.get("incidents")
    bundle_summary = bundle.get("summary")
    if not isinstance(bundle_rows, list) or not isinstance(bundle_summary, dict):
        raise ValueError("Primary analysis bundle is incomplete.")
    plan_cell_count = len(bundle_rows)

    projection = _read_json(reports["projection_diagnostics.json"])
    if (
        projection.get("schema_version") != _DERIVED_REPORT_SCHEMA
        or projection.get("run_ids") != [plan.plan_id]
        or projection.get("cell_count") != plan_cell_count
    ):
        raise ValueError("Projection diagnostics differ from primary evidence.")

    repeated_groups: dict[tuple[str, str], int] = {}
    for row in bundle_rows:
        if not isinstance(row, dict):
            raise ValueError("Primary analysis bundle contains an invalid cell row.")
        incident = str(row.get("incident_dir_id") or row.get("incident_id"))
        label = str(row.get("model_label"))
        key = (incident, label)
        repeated_groups[key] = repeated_groups.get(key, 0) + 1
    if any(count > plan.repeats for count in repeated_groups.values()):
        raise ValueError("Primary analysis contains too many repeats in a group.")
    complete_repeat_groups = sum(
        count == plan.repeats for count in repeated_groups.values()
    )
    incomplete_repeat_groups = len(repeated_groups) - complete_repeat_groups
    if bundle_summary.get("expected_repeats") != plan.repeats:
        raise ValueError("Primary analysis uses the wrong repeat schedule.")
    if (
        bundle_summary.get("complete_repeat_group_count") != complete_repeat_groups
        or bundle_summary.get("incomplete_repeat_group_count")
        != incomplete_repeat_groups
    ):
        raise ValueError("Primary repeat-group accounting is inconsistent.")
    stability = _read_json(reports["stability_diagnostics.json"])
    if (
        stability.get("schema_version") != _DERIVED_REPORT_SCHEMA
        or stability.get("run_ids") != [plan.plan_id]
        or stability.get("source_cell_count") != plan_cell_count
        or stability.get("expected_repeats") != plan.repeats
        or stability.get("observed_repeat_group_count") != len(repeated_groups)
        or stability.get("repeat_group_count") != complete_repeat_groups
        or stability.get("incomplete_repeat_group_count") != incomplete_repeat_groups
    ):
        raise ValueError("Stability diagnostics differ from primary evidence.")

    sensitivity = _read_json(reports["repeat_reduction_sensitivity.json"])
    contrasts_per_stratum = (
        len(plan.arms) * len(plan.models) * (len(plan.models) - 1) // 2
        + len(plan.models) * len(plan.arms) * (len(plan.arms) - 1) // 2
    )
    source_strata = sorted(
        {
            str(row.get("corpus_stratum"))
            for row in bundle_rows
            if row.get("corpus_stratum") is not None
        }
    )
    analysis_strata = (
        sorted([*source_strata, "all_incidents"]) if source_strata else [None]
    )
    expected_contrasts = contrasts_per_stratum * len(analysis_strata)
    analysis_group_multiplier = 2 if source_strata else 1
    if (
        sensitivity.get("schema_version") != _DERIVED_REPORT_SCHEMA
        or sensitivity.get("reported_reduction") != plan.repeat_reduction
        or sensitivity.get("expected_repeats") != plan.repeats
        or sensitivity.get("source_observed_repeat_group_count") != len(repeated_groups)
        or sensitivity.get("source_repeat_group_count") != complete_repeat_groups
        or sensitivity.get("source_incomplete_repeat_group_count")
        != incomplete_repeat_groups
        or sensitivity.get("observed_repeat_group_count")
        != len(repeated_groups) * analysis_group_multiplier
        or sensitivity.get("repeat_group_count")
        != complete_repeat_groups * analysis_group_multiplier
        or sensitivity.get("incomplete_repeat_group_count")
        != incomplete_repeat_groups * analysis_group_multiplier
        or sensitivity.get("comparison_family") != "single_factor"
        or sensitivity.get("comparison_count") != expected_contrasts
        or sensitivity.get("corpus_strata") != analysis_strata
        or sensitivity.get("same_comparison_family_under_every_reduction") is not True
    ):
        raise ValueError("Repeat-reduction sensitivity differs from primary evidence.")
    if primary["scientific_task_count"] != plan.primary_cell_count:
        raise ValueError("Primary evidence differs from the approved plan.")


def _campaign_state(
    *,
    repo_root: Path,
    plan_path: Path,
    primary_evidence_path: Path,
    adversarial_evidence_path: Path,
    provider_readiness_path: Path,
) -> tuple[dict[str, Any], CampaignEvidence]:
    repo_root = repo_root.expanduser().resolve()
    plan_path = _inside_repo(plan_path, repo_root, "Provider plan")
    plan = load_provider_plan(plan_path)
    if plan.approval.status != "approved":
        raise ValueError("Terminal campaign evidence requires an approved plan.")
    provider_readiness = load_provider_readiness(
        repo_root=repo_root,
        plan_path=plan_path,
        report_path=provider_readiness_path,
    )
    provider_readiness_attempt_log_path = _inside_repo(
        repo_root / provider_readiness.provider_attempt_log.path,
        repo_root,
        "Provider-readiness attempt log",
    )
    validate_campaign_diagnostics(plan_path, repo_root)
    freeze_path, freeze_sha256 = _plan_freeze(plan, repo_root)
    primary_evidence = load_execution_evidence(
        repo_root=repo_root,
        manifest_path=primary_evidence_path,
    )
    adversarial_evidence = load_execution_evidence(
        repo_root=repo_root,
        manifest_path=adversarial_evidence_path,
    )
    if primary_evidence.run_id != plan.plan_id:
        raise ValueError("Primary execution evidence has the wrong run id.")
    if adversarial_evidence.run_id != f"{plan.plan_id}__adversarial":
        raise ValueError("Adversarial execution evidence has the wrong run id.")
    for evidence in (primary_evidence, adversarial_evidence):
        if evidence.protocol_freeze.sha256 != freeze_sha256:
            raise ValueError("Campaign studies do not share one protocol freeze.")

    completed = {
        study_id: validate_completed_diagnostic_study(
            repo_root=repo_root,
            plan_path=plan_path,
            study_id=study_id,
        )
        for study_id in _DIAGNOSTIC_STUDIES
    }
    diagnostic_accounting = {
        study_id: _diagnostic_accounting(
            summary,
            plan.cost_estimate.study_budget_ceiling_usd[study_id],
        )
        for study_id, summary in completed.items()
    }
    primary_accounting = _execution_study_accounting(
        primary_evidence,
        plan.cost_estimate.study_budget_ceiling_usd["primary"],
    )
    adversarial_accounting = _execution_study_accounting(
        adversarial_evidence,
        plan.cost_estimate.study_budget_ceiling_usd["adversarial"],
    )
    readiness_attempts = _attempt_accounting(provider_readiness_attempt_log_path)
    readiness_budget = plan.cost_estimate.study_budget_ceiling_usd["provider_readiness"]
    if readiness_attempts.budget_exposure_usd > readiness_budget:
        raise ValueError("Provider-readiness cost exposure exceeds its ceiling.")
    observed_tasks = (
        primary_accounting["scientific_task_count"]
        + adversarial_accounting["scientific_task_count"]
        + sum(row["provider_task_count"] for row in diagnostic_accounting.values())
    )
    observed_requests = (
        provider_readiness.provider_request_count
        + primary_accounting["provider_request_count"]
        + adversarial_accounting["provider_request_count"]
        + sum(row["provider_request_count"] for row in diagnostic_accounting.values())
    )
    realized_cost = (
        provider_readiness.realized_cost_usd
        + primary_accounting["realized_cost_usd"]
        + adversarial_accounting["realized_cost_usd"]
        + sum(row["realized_cost_usd"] for row in diagnostic_accounting.values())
    )
    ambiguous_reserved_cost = (
        readiness_attempts.ambiguous_reserved_cost_usd
        + primary_accounting["ambiguous_reserved_cost_usd"]
        + adversarial_accounting["ambiguous_reserved_cost_usd"]
        + sum(
            row["ambiguous_reserved_cost_usd"] for row in diagnostic_accounting.values()
        )
    )
    budget_exposure = realized_cost + ambiguous_reserved_cost
    if observed_tasks > plan.scientific_task_ceiling:
        raise ValueError("Observed scientific tasks exceed the approved ceiling.")
    if observed_requests > plan.provider_transport_attempt_ceiling:
        raise ValueError("Provider requests exceed the approved transport ceiling.")
    if plan.approval.approved_budget_usd is None:
        raise ValueError("Approved campaign has no numeric provider budget.")
    if budget_exposure > plan.approval.approved_budget_usd:
        raise ValueError("Campaign cost exposure exceeds the approved budget.")

    reports = _report_paths(plan, repo_root)
    _validate_reports(
        plan=plan,
        freeze_sha256=freeze_sha256,
        reports=reports,
        primary=primary_accounting,
        adversarial=adversarial_accounting,
        diagnostics=diagnostic_accounting,
        primary_evidence=primary_evidence,
    )
    accounting = {
        "provider_readiness": {
            "model_count": len(provider_readiness.results),
            "provider_request_count": provider_readiness.provider_request_count,
            "provider_response_count": provider_readiness.provider_response_count,
            "realized_cost_usd": provider_readiness.realized_cost_usd,
            "ambiguous_reserved_cost_usd": round(
                readiness_attempts.ambiguous_reserved_cost_usd, 6
            ),
            "budget_exposure_usd": round(readiness_attempts.budget_exposure_usd, 6),
            "budget_ceiling_usd": readiness_budget,
        },
        "primary": primary_accounting,
        "independent_judge": diagnostic_accounting["independent_judge"],
        "judge_ablation": diagnostic_accounting["judge_ablation"],
        "adversarial": adversarial_accounting,
        "projection_fidelity": diagnostic_accounting["projection_fidelity"],
        "scientific_task_count_observed": observed_tasks,
        "scientific_task_ceiling": plan.scientific_task_ceiling,
        "provider_request_count_observed": observed_requests,
        "provider_transport_attempt_limit": plan.provider_transport_attempt_limit,
        "provider_transport_attempt_ceiling": plan.provider_transport_attempt_ceiling,
        "realized_cost_usd": round(realized_cost, 6),
        "ambiguous_reserved_cost_usd": round(ambiguous_reserved_cost, 6),
        "budget_exposure_usd": round(budget_exposure, 6),
        "approved_budget_usd": plan.approval.approved_budget_usd,
    }
    state = {
        "campaign_id": plan.plan_id,
        "provider_plan": _binding(plan_path, repo_root),
        "protocol_freeze": _binding(freeze_path, repo_root),
        "provider_readiness": _binding(provider_readiness_path, repo_root),
        "studies": {
            "primary": {
                "execution_evidence": _binding(
                    primary_evidence.manifest_path, repo_root
                )
            },
            "adversarial": {
                "execution_evidence": _binding(
                    adversarial_evidence.manifest_path, repo_root
                )
            },
            **{
                study_id: {"snapshot": _diagnostic_snapshot(summary, repo_root)}
                for study_id, summary in completed.items()
            },
        },
        "reports": {name: _binding(path, repo_root) for name, path in reports.items()},
        "accounting": accounting,
        "checks": {
            "all_studies_terminal": True,
            "provider_readiness_passed": True,
            "one_protocol_freeze": True,
            "provider_attempts_accounted": True,
            "planned_denominators_conserved": True,
            "reports_bound_to_terminal_evidence": True,
        },
    }
    evidence = CampaignEvidence(
        manifest_path=Path(),
        campaign_id=plan.plan_id,
        plan_path=plan_path,
        protocol_freeze_path=freeze_path,
        provider_readiness_path=provider_readiness_path,
        provider_readiness_attempt_log_path=provider_readiness_attempt_log_path,
        primary_evidence=primary_evidence,
        adversarial_evidence=adversarial_evidence,
        diagnostic_run_dirs={
            study_id: summary.run_dir for study_id, summary in completed.items()
        },
        report_files=reports,
        accounting=accounting,
    )
    return state, evidence


def build_campaign_evidence_manifest(
    *,
    repo_root: Path,
    plan_path: Path,
    primary_evidence_path: Path,
    adversarial_evidence_path: Path,
    provider_readiness_path: Path,
    output_json: Path,
) -> Path:
    repo_root = repo_root.expanduser().resolve()
    output_json = _inside_repo(output_json, repo_root, "Campaign evidence manifest")
    if output_json.exists():
        raise FileExistsError(f"Campaign evidence already exists: {output_json}")
    state, _ = _campaign_state(
        repo_root=repo_root,
        plan_path=plan_path,
        primary_evidence_path=primary_evidence_path,
        adversarial_evidence_path=adversarial_evidence_path,
        provider_readiness_path=provider_readiness_path,
    )
    payload = {
        "schema_version": CAMPAIGN_EVIDENCE_SCHEMA,
        "state": CAMPAIGN_EVIDENCE_STATE,
        "constitutes_model_evidence": True,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        **state,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_json, payload)
    return output_json


def load_campaign_evidence(*, repo_root: Path, manifest_path: Path) -> CampaignEvidence:
    repo_root = repo_root.expanduser().resolve()
    manifest_path = _inside_repo(manifest_path, repo_root, "Campaign evidence manifest")
    payload = _read_json(manifest_path)
    if (
        payload.get("schema_version") != CAMPAIGN_EVIDENCE_SCHEMA
        or payload.get("state") != CAMPAIGN_EVIDENCE_STATE
        or payload.get("constitutes_model_evidence") is not True
    ):
        raise ValueError("Campaign evidence manifest has an invalid state.")
    studies = payload.get("studies")
    if not isinstance(studies, dict):
        raise ValueError("Campaign evidence manifest has no studies.")
    plan_path = _resolve_binding(
        payload.get("provider_plan"), repo_root, "Provider plan"
    )
    provider_readiness_path = _resolve_binding(
        payload.get("provider_readiness"),
        repo_root,
        "Provider readiness",
    )
    primary = studies.get("primary")
    adversarial = studies.get("adversarial")
    if not isinstance(primary, dict) or not isinstance(adversarial, dict):
        raise ValueError("Campaign evidence lacks execution studies.")
    primary_path = _resolve_binding(
        primary.get("execution_evidence"), repo_root, "Primary execution evidence"
    )
    adversarial_path = _resolve_binding(
        adversarial.get("execution_evidence"),
        repo_root,
        "Adversarial execution evidence",
    )
    expected, evidence = _campaign_state(
        repo_root=repo_root,
        plan_path=plan_path,
        primary_evidence_path=primary_path,
        adversarial_evidence_path=adversarial_path,
        provider_readiness_path=provider_readiness_path,
    )
    observed = {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "schema_version",
            "state",
            "constitutes_model_evidence",
            "generated_at_utc",
        }
    }
    observed_studies = observed.pop("studies", None)
    expected_studies = expected.pop("studies", None)
    if observed != expected or not _studies_match_release(
        observed_studies, expected_studies, repo_root
    ):
        raise ValueError("Campaign evidence differs from its terminal sources.")
    return CampaignEvidence(
        manifest_path=manifest_path,
        campaign_id=evidence.campaign_id,
        plan_path=evidence.plan_path,
        protocol_freeze_path=evidence.protocol_freeze_path,
        provider_readiness_path=evidence.provider_readiness_path,
        provider_readiness_attempt_log_path=(
            evidence.provider_readiness_attempt_log_path
        ),
        primary_evidence=evidence.primary_evidence,
        adversarial_evidence=evidence.adversarial_evidence,
        diagnostic_run_dirs=evidence.diagnostic_run_dirs,
        report_files=evidence.report_files,
        accounting=evidence.accounting,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--plan", type=Path, required=True)
    build.add_argument("--primary-evidence", type=Path, required=True)
    build.add_argument("--adversarial-evidence", type=Path, required=True)
    build.add_argument("--provider-readiness-report", type=Path, required=True)
    build.add_argument("--output-json", type=Path, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repo_root = args.repo_root.expanduser().resolve()
    if args.command == "build":
        path = build_campaign_evidence_manifest(
            repo_root=repo_root,
            plan_path=args.plan,
            primary_evidence_path=args.primary_evidence,
            adversarial_evidence_path=args.adversarial_evidence,
            provider_readiness_path=args.provider_readiness_report,
            output_json=args.output_json,
        )
        print(f"Campaign evidence written: {path}")
        return
    evidence = load_campaign_evidence(
        repo_root=repo_root,
        manifest_path=args.manifest,
    )
    print(
        json.dumps(
            {
                "campaign_id": evidence.campaign_id,
                "accounting": evidence.accounting,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
