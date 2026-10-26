from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from soc_llm_policy.paths import repo_relative_path
from soc_llm_policy.reference_metrics import (
    reference_values,
    summarize_reference_metrics,
)
from soc_llm_policy.result_models import VerifierOutputModel

_Z_95 = 1.959963984540054
_ALL_INCIDENTS_STRATUM = "all_incidents"
type RepeatReduction = Literal["first", "last", "majority", "any"]


def _analysis_strata(corpus_stratum: str | None) -> tuple[str | None, ...]:
    if corpus_stratum is None or corpus_stratum == _ALL_INCIDENTS_STRATUM:
        return (corpus_stratum,)
    return (corpus_stratum, _ALL_INCIDENTS_STRATUM)


def _with_all_incident_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not any(row.get("corpus_stratum") is not None for row in rows):
        return rows
    aggregate_rows = [{**row, "corpus_stratum": _ALL_INCIDENTS_STRATUM} for row in rows]
    return [*rows, *aggregate_rows]


class CorpusStratumSummaryModel(BaseModel):
    run_count: int
    incident_count: int
    runs_with_violations: int
    run_violation_rate: float
    incidents_with_violations: int
    incident_violation_rate: float


class SummaryModel(BaseModel):
    run_count: int
    incident_count: int
    incidents_with_violations: int
    incident_violation_rate: float
    mode_counts: dict[str, int]
    violations_by_rule: dict[str, int]
    violations_by_type: dict[str, int]
    violations_by_severity: dict[str, int]
    llm_only_actions: dict[str, int]
    human_only_actions: dict[str, int]
    attack_candidate_count_total: int
    attack_curated_count_total: int
    attack_ab_gap_count_total: int
    attack_curated_to_candidate_ratio: float
    attack_technique_counts: dict[str, int]
    attack_tactic_counts: dict[str, int]
    llm_prompt_tokens_total: int
    llm_completion_tokens_total: int
    llm_total_tokens_total: int
    llm_latency_ms_total: int
    llm_latency_ms_avg_per_run: float
    llm_cost_estimated_usd_total: float
    llm_cost_estimated_usd_avg_per_run: float
    llm_hallucinated_action_count_total: int
    llm_hallucination_rate_avg_per_run: float
    enforcement_actions_inserted_count_total: int
    enforcement_actions_removed_count_total: int
    enforcement_actions_deferred_count_total: int
    enforcement_actions_reordered_count_total: int
    enforcement_action_modification_count_total: int
    enforcement_modified_run_count: int
    enforcement_modification_rate: float
    task_coverage_raw_avg: float | None
    task_coverage_enforced_avg: float | None
    task_coverage_delta_avg: float | None
    task_coverage_drop_run_count: int | None
    task_coverage_drop_rate: float | None
    reference_run_count: int = 0
    pairwise_comparison_count: int
    violation_rate_by_tactic: dict[str, float]
    violation_rate_by_technique: dict[str, float]
    support_count_by_tactic: dict[str, int]
    support_count_by_technique: dict[str, int]
    violation_rate_by_incident_type: dict[str, float]
    violation_rate_by_severity: dict[str, float]
    violation_rate_by_asset_criticality: dict[str, float]
    support_count_by_incident_type: dict[str, int]
    support_count_by_severity: dict[str, int]
    support_count_by_asset_criticality: dict[str, int]
    corpus_strata: dict[str, CorpusStratumSummaryModel] = Field(default_factory=dict)
    repeat_reduction: RepeatReduction = "last"
    expected_repeats: int | None = None
    complete_repeat_group_count: int = 0
    incomplete_repeat_group_count: int = 0


class IncidentRowModel(BaseModel):
    incident_id: str
    incident_dir_id: str
    incident_type: str
    severity: str
    asset_criticality: str
    mode: str
    run_tag: str
    approval_policy_mode: str
    llm_deployment: str
    llm_arm: str
    model_label: str
    corpus_stratum: str | None = None
    experiment_repeat_index: int | None = None
    experiment_cell_id: str | None = None
    violation_count: int
    hard_violation_count: int
    llm_action_count: int
    human_action_count: int
    enforced_action_count: int
    llm_only_action_count: int
    human_only_action_count: int
    attack_match_count: int
    attack_candidate_count: int
    attack_ab_gap_count: int
    attack_curated_to_candidate_ratio: float
    attack_technique_count: int
    attack_tactic_count: int
    llm_prompt_tokens: int
    llm_completion_tokens: int
    llm_total_tokens: int
    llm_latency_ms: int
    llm_cost_estimated_usd: float
    llm_hallucinated_action_count: int
    llm_hallucination_rate: float
    enforcement_actions_inserted_count: int
    enforcement_actions_removed_count: int
    enforcement_actions_deferred_count: int
    enforcement_actions_reordered_count: int
    enforcement_action_modification_count: int
    enforcement_modified: bool
    precision_raw: float | None
    recall_raw: float | None
    task_coverage_raw: float | None
    f1_raw: float | None
    jaccard_raw: float | None
    precision_enforced: float | None
    recall_enforced: float | None
    task_coverage_enforced: float | None
    f1_enforced: float | None
    jaccard_enforced: float | None
    delta_jaccard: float | None
    task_coverage_delta: float | None


class ByModelRowModel(BaseModel):
    model_label: str
    corpus_stratum: str | None = None
    run_count: int
    runs_with_violations: int
    run_violation_rate: float
    run_violation_rate_ci_low: float
    run_violation_rate_ci_high: float
    run_violation_rate_ci_low_clustered: float | None = None
    run_violation_rate_ci_high_clustered: float | None = None
    violation_count: int
    hard_violation_count: int
    llm_only_action_count: int
    human_only_action_count: int
    llm_prompt_tokens_total: int
    llm_completion_tokens_total: int
    llm_total_tokens_total: int
    llm_latency_ms_total: int
    llm_latency_ms_avg: float
    llm_cost_estimated_usd_total: float
    llm_hallucinated_action_count_total: int
    llm_hallucination_rate_avg: float
    enforcement_actions_inserted_count_total: int
    enforcement_actions_removed_count_total: int
    enforcement_actions_deferred_count_total: int
    enforcement_actions_reordered_count_total: int
    enforcement_action_modification_count_total: int
    enforcement_modified_run_count: int
    enforcement_modification_rate: float
    precision_raw_avg: float | None
    recall_raw_avg: float | None
    task_coverage_raw_avg: float | None
    f1_raw_avg: float | None
    jaccard_raw_avg: float | None
    precision_enforced_avg: float | None
    recall_enforced_avg: float | None
    task_coverage_enforced_avg: float | None
    f1_enforced_avg: float | None
    jaccard_enforced_avg: float | None
    delta_jaccard_avg: float | None
    task_coverage_delta_avg: float | None
    task_coverage_drop_run_count: int | None
    task_coverage_drop_rate: float | None
    reference_run_count: int = 0


class ByRuleRowModel(BaseModel):
    model_label: str
    corpus_stratum: str | None = None
    rule_id: str
    violation_count: int
    hard_violation_count: int


class ByRuleTreatmentRowModel(BaseModel):
    rule_id: str
    model: str
    corpus_stratum: str | None = None
    llm_zero_run_count: int
    llm_policy_prompt_run_count: int
    llm_zero_violation_count: int
    llm_policy_prompt_violation_count: int
    llm_zero_violation_rate: float
    llm_policy_prompt_violation_rate: float
    violation_rate_diff_zero_minus_policy: float


class PairwiseRowModel(BaseModel):
    model_a: str
    model_b: str
    contrast_type: Literal["model", "prompt_condition", "generic"]
    corpus_stratum: str | None = None
    paired_incident_count: int
    run_count_a: int
    run_count_b: int
    violation_rate_a: float
    violation_rate_b: float
    rate_diff_a_minus_b: float
    z_score: float
    p_value_two_sided: float
    p_value_holm: float
    cohens_h: float
    cliffs_delta: float
    mcnemar_b_count: int
    mcnemar_c_count: int
    mcnemar_chi2: float
    mcnemar_p_value_two_sided: float
    mcnemar_p_value_holm: float


class AnalysisBundleModel(BaseModel):
    schema_version: Literal[1] = 1
    generated_at_utc: str = Field(..., min_length=1)
    eval_protocol_version: str = Field(..., min_length=1)
    outputs_incidents_dir: str = Field(..., min_length=1)
    incident_filter: list[str] = Field(default_factory=list)
    summary: SummaryModel
    incidents: list[IncidentRowModel]
    by_model: list[ByModelRowModel]
    by_rule: list[ByRuleRowModel]
    by_rule_treatment: list[ByRuleTreatmentRowModel]
    pairwise: list[PairwiseRowModel]


@dataclass(frozen=True)
class AnalysisBundleInput:
    eval_protocol_version: str
    repo_root: Path
    outputs_incidents_dir: Path
    incident_filter: set[str] | None
    summary: dict[str, Any]
    rows: list[dict[str, Any]]
    by_model_rows: list[dict[str, Any]]
    by_rule_rows: list[dict[str, Any]]
    by_rule_treatment_rows: list[dict[str, Any]]
    pairwise_rows: list[dict[str, Any]]


@dataclass
class _AnalyzeAccumulator:
    rows: list[dict[str, Any]]
    violations_by_rule: Counter[str]
    violations_by_type: Counter[str]
    violations_by_severity: Counter[str]
    mode_counter: Counter[str]
    llm_only_actions: Counter[str]
    human_only_actions: Counter[str]
    technique_counter: Counter[str]
    tactic_counter: Counter[str]
    violations_by_rule_and_model: Counter[tuple[str | None, str, str]]
    hard_violations_by_rule_and_model: Counter[tuple[str | None, str, str]]
    incidents_with_violations_set: set[str]
    unique_incident_ids: set[str]
    attack_summary_by_incident: dict[str, VerifierOutputModel]
    tactic_run_counter: Counter[str]
    tactic_violation_run_counter: Counter[str]
    technique_run_counter: Counter[str]
    technique_violation_run_counter: Counter[str]
    incident_type_run_counter: Counter[str]
    incident_type_violation_run_counter: Counter[str]
    severity_run_counter: Counter[str]
    severity_violation_run_counter: Counter[str]
    criticality_run_counter: Counter[str]
    criticality_violation_run_counter: Counter[str]
    run_count_by_model_arm: Counter[tuple[str | None, str, str]]
    violation_run_count_by_model_arm_rule: Counter[tuple[str | None, str, str, str]]


def _create_accumulator() -> _AnalyzeAccumulator:
    return _AnalyzeAccumulator(
        rows=[],
        violations_by_rule=Counter(),
        violations_by_type=Counter(),
        violations_by_severity=Counter(),
        mode_counter=Counter(),
        llm_only_actions=Counter(),
        human_only_actions=Counter(),
        technique_counter=Counter(),
        tactic_counter=Counter(),
        violations_by_rule_and_model=Counter(),
        hard_violations_by_rule_and_model=Counter(),
        incidents_with_violations_set=set(),
        unique_incident_ids=set(),
        attack_summary_by_incident={},
        tactic_run_counter=Counter(),
        tactic_violation_run_counter=Counter(),
        technique_run_counter=Counter(),
        technique_violation_run_counter=Counter(),
        incident_type_run_counter=Counter(),
        incident_type_violation_run_counter=Counter(),
        severity_run_counter=Counter(),
        severity_violation_run_counter=Counter(),
        criticality_run_counter=Counter(),
        criticality_violation_run_counter=Counter(),
        run_count_by_model_arm=Counter(),
        violation_run_count_by_model_arm_rule=Counter(),
    )


def _accumulate_run_verifier(
    state: _AnalyzeAccumulator,
    verifier: VerifierOutputModel,
    row: dict[str, Any],
) -> None:
    incident_key = str(
        row.get("incident_dir_id") or verifier.incident_dir_id or verifier.incident_id
    )
    state.rows.append(row)
    state.unique_incident_ids.add(incident_key)
    if incident_key not in state.attack_summary_by_incident:
        state.attack_summary_by_incident[incident_key] = verifier

    state.mode_counter[verifier.mode] += 1
    model_label = str(row.get("model_label", "unknown")) or "unknown"
    corpus_stratum = verifier.experiment_corpus_stratum
    incident_type = str(row.get("incident_type", "unknown")) or "unknown"
    severity = str(row.get("severity", "unknown")) or "unknown"
    criticality = str(row.get("asset_criticality", "unknown")) or "unknown"
    state.incident_type_run_counter[incident_type] += 1
    state.severity_run_counter[severity] += 1
    state.criticality_run_counter[criticality] += 1
    if verifier.violations:
        state.incidents_with_violations_set.add(incident_key)
        state.incident_type_violation_run_counter[incident_type] += 1
        state.severity_violation_run_counter[severity] += 1
        state.criticality_violation_run_counter[criticality] += 1
    analysis_strata = _analysis_strata(corpus_stratum)
    for violation in verifier.violations:
        state.violations_by_rule[violation.rule_id] += 1
        state.violations_by_type[violation.type] += 1
        state.violations_by_severity[violation.severity] += 1
        for analysis_stratum in analysis_strata:
            state.violations_by_rule_and_model[
                (analysis_stratum, model_label, violation.rule_id)
            ] += 1
            if violation.severity == "hard":
                state.hard_violations_by_rule_and_model[
                    (analysis_stratum, model_label, violation.rule_id)
                ] += 1
    llm_deployment = str(row.get("llm_deployment", "")).strip()
    llm_arm = str(row.get("llm_arm", "")).strip()
    if llm_deployment and llm_arm in {"llm_zero", "llm_policy_prompt"}:
        violated_rules = {violation.rule_id for violation in verifier.violations}
        for analysis_stratum in analysis_strata:
            state.run_count_by_model_arm[
                (analysis_stratum, llm_deployment, llm_arm)
            ] += 1
            for violated_rule in violated_rules:
                state.violation_run_count_by_model_arm_rule[
                    (analysis_stratum, llm_deployment, llm_arm, violated_rule)
                ] += 1

    human_actions = set(verifier.human_actions)
    llm_actions = set(verifier.llm_actions)
    for action in sorted(llm_actions - human_actions):
        state.llm_only_actions[action] += 1
    for action in sorted(human_actions - llm_actions):
        state.human_only_actions[action] += 1

    tactic_ids = sorted({item.tactic_id for item in verifier.attack_summary.tactics})
    technique_ids = sorted(
        {item.technique_id for item in verifier.attack_summary.techniques}
    )
    has_violation = len(verifier.violations) > 0
    for tactic_id in tactic_ids:
        state.tactic_run_counter[tactic_id] += 1
        if has_violation:
            state.tactic_violation_run_counter[tactic_id] += 1
    for technique_id in technique_ids:
        state.technique_run_counter[technique_id] += 1
        if has_violation:
            state.technique_violation_run_counter[technique_id] += 1


def _accumulate_attack_unique(
    state: _AnalyzeAccumulator,
) -> tuple[int, int, int]:
    total_candidate_count = 0
    total_curated_count = 0
    total_ab_gap_count = 0
    for verifier in state.attack_summary_by_incident.values():
        total_candidate_count += verifier.metrics.attack_candidate_count
        total_curated_count += verifier.metrics.attack_match_count
        total_ab_gap_count += max(
            verifier.metrics.attack_candidate_count
            - verifier.metrics.attack_match_count,
            0,
        )
        for item in verifier.attack_summary.technique_counts:
            state.technique_counter[item.technique_id] += item.count
        for item in verifier.attack_summary.tactic_counts:
            state.tactic_counter[item.tactic_id] += item.count
    return total_candidate_count, total_curated_count, total_ab_gap_count


def _violation_rate_by_key(
    run_counter: Counter[str],
    violation_counter: Counter[str],
) -> tuple[dict[str, float], dict[str, int]]:
    rates: dict[str, float] = {}
    support: dict[str, int] = {}
    for key in sorted(run_counter):
        total = int(run_counter[key])
        support[key] = total
        violations = int(violation_counter[key])
        rates[key] = round((violations / total) if total else 0.0, 4)
    return rates, support


def _corpus_stratum_summaries(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, int | float]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        corpus_stratum = row.get("corpus_stratum")
        if corpus_stratum is None:
            continue
        grouped.setdefault(str(corpus_stratum), []).append(row)

    summaries: dict[str, dict[str, int | float]] = {}
    if grouped:
        grouped[_ALL_INCIDENTS_STRATUM] = list(rows)
    for corpus_stratum, stratum_rows in sorted(grouped.items()):
        incident_ids = {
            str(row.get("incident_dir_id") or row["incident_id"])
            for row in stratum_rows
        }
        violating_rows = [
            row for row in stratum_rows if int(row.get("violation_count", 0)) > 0
        ]
        violating_incidents = {
            str(row.get("incident_dir_id") or row["incident_id"])
            for row in violating_rows
        }
        summaries[corpus_stratum] = {
            "run_count": len(stratum_rows),
            "incident_count": len(incident_ids),
            "runs_with_violations": len(violating_rows),
            "run_violation_rate": round(len(violating_rows) / len(stratum_rows), 4),
            "incidents_with_violations": len(violating_incidents),
            "incident_violation_rate": round(
                len(violating_incidents) / len(incident_ids), 4
            ),
        }
    return summaries


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid JSON (expected object): {path}")
    return data


def _load_verifier_output(path: Path) -> VerifierOutputModel:
    return VerifierOutputModel.model_validate(_load_json(path))


def _incident_row(verifier: VerifierOutputModel) -> dict[str, Any]:
    metrics = verifier.metrics

    curated_count = metrics.attack_match_count
    candidate_count = metrics.attack_candidate_count
    ab_gap = max(candidate_count - curated_count, 0)
    ab_curated_ratio = (
        round(curated_count / candidate_count, 4) if candidate_count else 0.0
    )
    mode = verifier.mode
    approval_policy_mode = verifier.approval_policy_mode or "remove"
    llm_deployment = verifier.llm_deployment or ""
    llm_arm = verifier.llm_arm or ""
    model_label = (
        f"{llm_deployment}|{llm_arm}"
        if llm_deployment and llm_arm
        else (llm_deployment or mode or "unknown")
    )
    if approval_policy_mode != "remove":
        model_label = f"{model_label}|{approval_policy_mode}"
    llm_prompt_tokens = verifier.llm_usage.prompt_tokens if verifier.llm_usage else 0
    llm_completion_tokens = (
        verifier.llm_usage.completion_tokens if verifier.llm_usage else 0
    )
    llm_total_tokens = verifier.llm_usage.total_tokens if verifier.llm_usage else 0
    llm_latency_ms = verifier.llm_latency_ms or 0
    llm_cost_estimated = verifier.llm_cost_estimated_usd or 0.0

    return {
        "incident_id": verifier.incident_id,
        "incident_dir_id": verifier.incident_dir_id or verifier.incident_id,
        "incident_type": verifier.incident_type or "unknown",
        "severity": verifier.severity or "unknown",
        "asset_criticality": verifier.asset_criticality or "unknown",
        "mode": mode,
        "run_tag": verifier.run_tag,
        "approval_policy_mode": approval_policy_mode,
        "llm_deployment": llm_deployment,
        "llm_arm": llm_arm,
        "model_label": model_label,
        "corpus_stratum": verifier.experiment_corpus_stratum,
        **(
            {"experiment_repeat_index": verifier.experiment_repeat_index}
            if verifier.experiment_repeat_index is not None
            else {}
        ),
        **(
            {"experiment_cell_id": verifier.experiment_cell_id}
            if verifier.experiment_cell_id is not None
            else {}
        ),
        "violation_count": metrics.violation_count,
        "hard_violation_count": metrics.hard_violation_count,
        "llm_action_count": metrics.llm_action_count,
        "human_action_count": metrics.human_action_count,
        "enforced_action_count": metrics.enforced_action_count,
        "llm_only_action_count": metrics.llm_only_action_count,
        "human_only_action_count": metrics.human_only_action_count,
        "attack_match_count": metrics.attack_match_count,
        "attack_candidate_count": candidate_count,
        "attack_ab_gap_count": ab_gap,
        "attack_curated_to_candidate_ratio": ab_curated_ratio,
        "attack_technique_count": metrics.attack_technique_count,
        "attack_tactic_count": metrics.attack_tactic_count,
        "llm_prompt_tokens": llm_prompt_tokens,
        "llm_completion_tokens": llm_completion_tokens,
        "llm_total_tokens": llm_total_tokens,
        "llm_latency_ms": llm_latency_ms,
        "llm_cost_estimated_usd": round(llm_cost_estimated, 6),
        "llm_hallucinated_action_count": metrics.llm_hallucinated_action_count,
        "llm_hallucination_rate": metrics.llm_hallucination_rate,
        "enforcement_actions_inserted_count": (
            metrics.enforcement_actions_inserted_count
        ),
        "enforcement_actions_removed_count": (
            metrics.enforcement_actions_removed_count
        ),
        "enforcement_actions_deferred_count": (
            metrics.enforcement_actions_deferred_count
        ),
        "enforcement_actions_reordered_count": (
            metrics.enforcement_actions_reordered_count
        ),
        "enforcement_action_modification_count": (
            metrics.enforcement_action_modification_count
        ),
        "enforcement_modified": metrics.enforcement_modified,
        **reference_values(metrics),
    }


def _collect_verifier_files(
    outputs_incidents_dir: Path,
    canonical_only: bool = False,
) -> list[Path]:
    incident_dirs = sorted(
        path for path in outputs_incidents_dir.iterdir() if path.is_dir()
    )
    selected: list[Path] = []
    for incident_dir in incident_dirs:
        canonical = incident_dir / "verifier_output.json"
        if canonical_only:
            if canonical.exists():
                selected.append(canonical)
            continue
        versioned = sorted(incident_dir.glob("verifier_output_*.json"))
        if versioned:
            selected.extend(versioned)
            continue
        if canonical.exists():
            selected.append(canonical)
    return selected


def _select_latest_runs_per_incident_model(
    verifier_files: list[Path],
    run_id: str | list[str] | None = None,
) -> list[tuple[VerifierOutputModel, dict[str, Any]]]:
    """Select the outputs that constitute one evaluation.

    With ``run_id`` set, only outputs from the named run are kept. A list remains
    accepted for deterministic replay of archived campaigns and is deduplicated
    by experimental coordinate rather than by cell ID. The first named run wins
    if archived records overlap. New campaigns use one run ID and resume in
    place. Without ``run_id``, legacy behavior keeps the latest output per
    (incident_dir_id, model_label), which mixes runs and must not be used for
    official numbers.
    """
    if run_id is not None:
        run_ids = [run_id] if isinstance(run_id, str) else list(run_id)
        run_rank = {rid: index for index, rid in enumerate(run_ids)}
        by_coord: dict[
            tuple[str, str, str, int | None],
            tuple[int, VerifierOutputModel, dict[str, Any]],
        ] = {}
        for path in verifier_files:
            verifier = _load_verifier_output(path)
            if verifier.experiment_run_id not in run_rank:
                continue
            row = _incident_row(verifier)
            coord = (
                str(row["incident_dir_id"]),
                str(verifier.llm_deployment or ""),
                str(verifier.llm_arm or ""),
                verifier.experiment_repeat_index,
            )
            rank = run_rank[str(verifier.experiment_run_id)]
            current = by_coord.get(coord)
            if current is None or rank < current[0]:
                by_coord[coord] = (rank, verifier, row)
        selected = [(verifier, row) for _rank, verifier, row in by_coord.values()]
        selected.sort(
            key=lambda item: (
                str(item[1]["incident_dir_id"]),
                str(item[1]["model_label"]),
            )
        )
        return selected

    latest: dict[tuple[str, str], tuple[str, VerifierOutputModel, dict[str, Any]]] = {}
    for path in verifier_files:
        verifier = _load_verifier_output(path)
        row = _incident_row(verifier)
        key = (str(row["incident_dir_id"]), str(row["model_label"]))
        run_tag = str(row.get("run_tag", "") or "")
        current = latest.get(key)
        if current is None or run_tag > current[0]:
            latest[key] = (run_tag, verifier, row)
    selected = [value[1:] for value in latest.values()]
    selected.sort(
        key=lambda item: (str(item[1]["incident_dir_id"]), str(item[1]["model_label"]))
    )
    return selected


def _parse_incident_filter(raw: str | None) -> set[str] | None:
    if raw is None:
        return None
    values = {item.strip() for item in raw.split(",") if item.strip()}
    return values or None


def _build_by_model_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str | None, str], dict[str, Any]] = {}
    for row in _with_all_incident_rows(rows):
        model = str(row.get("model_label", "unknown")) or "unknown"
        corpus_stratum = row.get("corpus_stratum")
        if corpus_stratum is not None:
            corpus_stratum = str(corpus_stratum)
        bucket = grouped.setdefault(
            (corpus_stratum, model),
            {
                "run_count": 0,
                "runs_with_violations": 0,
                "cluster_cells": {},
                "violation_count": 0,
                "hard_violation_count": 0,
                "llm_only_action_count": 0,
                "human_only_action_count": 0,
                "llm_prompt_tokens_total": 0,
                "llm_completion_tokens_total": 0,
                "llm_total_tokens_total": 0,
                "llm_latency_ms_total": 0,
                "llm_cost_estimated_usd_total": 0.0,
                "llm_hallucinated_action_count_total": 0,
                "llm_hallucination_rate_sum": 0.0,
                "enforcement_actions_inserted_count_total": 0,
                "enforcement_actions_removed_count_total": 0,
                "enforcement_actions_deferred_count_total": 0,
                "enforcement_actions_reordered_count_total": 0,
                "enforcement_action_modification_count_total": 0,
                "enforcement_modified_run_count": 0,
                "reference_rows": [],
            },
        )
        bucket["run_count"] += 1
        bucket["violation_count"] += int(row.get("violation_count", 0))
        bucket["hard_violation_count"] += int(row.get("hard_violation_count", 0))
        bucket["llm_only_action_count"] += int(row.get("llm_only_action_count", 0))
        bucket["human_only_action_count"] += int(row.get("human_only_action_count", 0))
        bucket["llm_prompt_tokens_total"] += int(row.get("llm_prompt_tokens", 0))
        bucket["llm_completion_tokens_total"] += int(
            row.get("llm_completion_tokens", 0)
        )
        bucket["llm_total_tokens_total"] += int(row.get("llm_total_tokens", 0))
        bucket["llm_latency_ms_total"] += int(row.get("llm_latency_ms", 0))
        bucket["llm_cost_estimated_usd_total"] += float(
            row.get("llm_cost_estimated_usd", 0.0)
        )
        bucket["llm_hallucinated_action_count_total"] += int(
            row.get("llm_hallucinated_action_count", 0)
        )
        bucket["llm_hallucination_rate_sum"] += float(
            row.get("llm_hallucination_rate", 0.0)
        )
        bucket["enforcement_actions_inserted_count_total"] += int(
            row.get("enforcement_actions_inserted_count", 0)
        )
        bucket["enforcement_actions_removed_count_total"] += int(
            row.get("enforcement_actions_removed_count", 0)
        )
        bucket["enforcement_actions_deferred_count_total"] += int(
            row.get("enforcement_actions_deferred_count", 0)
        )
        bucket["enforcement_actions_reordered_count_total"] += int(
            row.get("enforcement_actions_reordered_count", 0)
        )
        bucket["enforcement_action_modification_count_total"] += int(
            row.get("enforcement_action_modification_count", 0)
        )
        if bool(row.get("enforcement_modified", False)):
            bucket["enforcement_modified_run_count"] += 1
        bucket["reference_rows"].append(row)
        if int(row.get("violation_count", 0)) > 0:
            bucket["runs_with_violations"] += 1
        incident_key = str(row.get("incident_dir_id", row.get("incident_id", "")))
        violating_cells, total_cells = bucket["cluster_cells"].get(incident_key, (0, 0))
        bucket["cluster_cells"][incident_key] = (
            violating_cells + (1 if int(row.get("violation_count", 0)) > 0 else 0),
            total_cells + 1,
        )

    out: list[dict[str, Any]] = []
    for corpus_stratum, model in sorted(
        grouped, key=lambda item: (item[0] or "", item[1])
    ):
        bucket = grouped[(corpus_stratum, model)]
        run_count = bucket["run_count"]
        runs_with_violations = bucket["runs_with_violations"]
        run_violation_rate = (
            round(runs_with_violations / run_count, 4) if run_count else 0.0
        )
        ci_low, ci_high = _wilson_interval(runs_with_violations, run_count)
        clustered_low, clustered_high = _clustered_bootstrap_rate_ci(
            bucket["cluster_cells"]
        )
        out.append(
            {
                "model_label": model,
                **(
                    {"corpus_stratum": corpus_stratum}
                    if corpus_stratum is not None
                    else {}
                ),
                "run_count": run_count,
                "runs_with_violations": runs_with_violations,
                "run_violation_rate": run_violation_rate,
                "run_violation_rate_ci_low": ci_low,
                "run_violation_rate_ci_high": ci_high,
                "run_violation_rate_ci_low_clustered": clustered_low,
                "run_violation_rate_ci_high_clustered": clustered_high,
                "violation_count": bucket["violation_count"],
                "hard_violation_count": bucket["hard_violation_count"],
                "llm_only_action_count": bucket["llm_only_action_count"],
                "human_only_action_count": bucket["human_only_action_count"],
                "llm_prompt_tokens_total": bucket["llm_prompt_tokens_total"],
                "llm_completion_tokens_total": bucket["llm_completion_tokens_total"],
                "llm_total_tokens_total": bucket["llm_total_tokens_total"],
                "llm_latency_ms_total": bucket["llm_latency_ms_total"],
                "llm_latency_ms_avg": round(
                    bucket["llm_latency_ms_total"] / run_count,
                    2,
                ),
                "llm_cost_estimated_usd_total": round(
                    bucket["llm_cost_estimated_usd_total"], 6
                ),
                "llm_hallucinated_action_count_total": bucket[
                    "llm_hallucinated_action_count_total"
                ],
                "llm_hallucination_rate_avg": round(
                    bucket["llm_hallucination_rate_sum"] / run_count,
                    4,
                ),
                "enforcement_actions_inserted_count_total": bucket[
                    "enforcement_actions_inserted_count_total"
                ],
                "enforcement_actions_removed_count_total": bucket[
                    "enforcement_actions_removed_count_total"
                ],
                "enforcement_actions_deferred_count_total": bucket[
                    "enforcement_actions_deferred_count_total"
                ],
                "enforcement_actions_reordered_count_total": bucket[
                    "enforcement_actions_reordered_count_total"
                ],
                "enforcement_action_modification_count_total": bucket[
                    "enforcement_action_modification_count_total"
                ],
                "enforcement_modified_run_count": bucket[
                    "enforcement_modified_run_count"
                ],
                "enforcement_modification_rate": round(
                    bucket["enforcement_modified_run_count"] / run_count,
                    4,
                ),
                **summarize_reference_metrics(bucket["reference_rows"]),
            }
        )
    return out


def _build_by_rule_rows(
    rule_model_counter: Counter[tuple[str | None, str, str]],
    rule_model_hard_counter: Counter[tuple[str | None, str, str]],
) -> list[dict[str, Any]]:
    pairs = sorted(rule_model_counter, key=lambda item: (item[0] or "", *item[1:]))
    out: list[dict[str, Any]] = []
    for corpus_stratum, model_label, rule_id in pairs:
        counter_key = (corpus_stratum, model_label, rule_id)
        total = int(rule_model_counter[counter_key])
        hard = int(rule_model_hard_counter[counter_key])
        out.append(
            {
                "model_label": model_label,
                **(
                    {"corpus_stratum": corpus_stratum}
                    if corpus_stratum is not None
                    else {}
                ),
                "rule_id": rule_id,
                "violation_count": total,
                "hard_violation_count": hard,
            }
        )
    return out


def _build_by_rule_treatment_rows(
    run_count_by_model_arm: Counter[tuple[str | None, str, str]],
    violation_run_count_by_model_arm_rule: Counter[tuple[str | None, str, str, str]],
) -> list[dict[str, Any]]:
    """Compare per-rule violation rates between llm_zero and llm_policy_prompt."""
    output: list[dict[str, Any]] = []
    strata_and_models = sorted(
        {(stratum, model) for stratum, model, _arm in run_count_by_model_arm},
        key=lambda item: (item[0] or "", item[1]),
    )
    for corpus_stratum, model in strata_and_models:
        zero_runs = int(
            run_count_by_model_arm.get((corpus_stratum, model, "llm_zero"), 0)
        )
        policy_runs = int(
            run_count_by_model_arm.get((corpus_stratum, model, "llm_policy_prompt"), 0)
        )
        if zero_runs == 0 or policy_runs == 0:
            continue
        rule_ids = sorted(
            {
                rule
                for stratum, model_key, _arm, rule in violation_run_count_by_model_arm_rule
                if stratum == corpus_stratum and model_key == model
            }
        )
        for rule_id in rule_ids:
            zero_count = int(
                violation_run_count_by_model_arm_rule.get(
                    (corpus_stratum, model, "llm_zero", rule_id),
                    0,
                )
            )
            policy_count = int(
                violation_run_count_by_model_arm_rule.get(
                    (corpus_stratum, model, "llm_policy_prompt", rule_id),
                    0,
                )
            )
            zero_rate = round(zero_count / zero_runs, 4) if zero_runs else 0.0
            policy_rate = round(policy_count / policy_runs, 4) if policy_runs else 0.0
            output.append(
                {
                    "rule_id": rule_id,
                    "model": model,
                    **(
                        {"corpus_stratum": corpus_stratum}
                        if corpus_stratum is not None
                        else {}
                    ),
                    "llm_zero_run_count": zero_runs,
                    "llm_policy_prompt_run_count": policy_runs,
                    "llm_zero_violation_count": zero_count,
                    "llm_policy_prompt_violation_count": policy_count,
                    "llm_zero_violation_rate": zero_rate,
                    "llm_policy_prompt_violation_rate": policy_rate,
                    "violation_rate_diff_zero_minus_policy": round(
                        zero_rate - policy_rate,
                        4,
                    ),
                }
            )
    return output


def _clustered_bootstrap_rate_ci(
    cluster_cells: dict[str, tuple[int, int]],
    resamples: int = 2000,
    seed: int = 20260707,
) -> tuple[float, float]:
    """Percentile CI for a cell rate, resampling incident clusters.

    With repeated cells per incident, treating cells as independent
    (Wilson over cells) understates uncertainty. Resampling whole incidents
    with replacement keeps within-incident correlation intact. Seeded, so the
    interval replays.
    """
    clusters = sorted(cluster_cells)
    if not clusters:
        return 0.0, 0.0
    rng = random.Random(seed)
    rates: list[float] = []
    for _ in range(resamples):
        successes = 0
        total = 0
        for _ in clusters:
            violating, cells = cluster_cells[clusters[rng.randrange(len(clusters))]]
            successes += violating
            total += cells
        rates.append(successes / total if total else 0.0)
    rates.sort()
    low = rates[int(0.025 * (len(rates) - 1))]
    high = rates[int(0.975 * (len(rates) - 1))]
    return round(low, 4), round(high, 4)


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    success_rate = successes / total
    denom = 1.0 + (_Z_95**2) / total
    center = (success_rate + (_Z_95**2) / (2 * total)) / denom
    margin = (
        _Z_95
        * math.sqrt(
            (success_rate * (1 - success_rate) / total)
            + (_Z_95**2) / (4 * (total**2))
        )
        / denom
    )
    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    return round(low, 4), round(high, 4)


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _two_proportion_ztest(
    successes_a: int,
    total_a: int,
    successes_b: int,
    total_b: int,
) -> tuple[float, float]:
    if total_a <= 0 or total_b <= 0:
        return 0.0, 1.0
    rate_a = successes_a / total_a
    rate_b = successes_b / total_b
    pooled = (successes_a + successes_b) / (total_a + total_b)
    denom = math.sqrt(pooled * (1 - pooled) * ((1 / total_a) + (1 / total_b)))
    if denom == 0:
        return 0.0, 1.0
    z_score = (rate_a - rate_b) / denom
    p_value = 2 * (1 - _normal_cdf(abs(z_score)))
    return z_score, max(0.0, min(1.0, p_value))


def _cohens_h(rate_a: float, rate_b: float) -> float:
    return round(2 * (math.asin(math.sqrt(rate_a)) - math.asin(math.sqrt(rate_b))), 6)


def _cliffs_delta_binary(values_a: list[int], values_b: list[int]) -> float:
    if not values_a or not values_b:
        return 0.0
    if len(values_a) != len(values_b):
        return 0.0
    signed_sum = 0
    for a_value, b_value in zip(values_a, values_b, strict=True):
        if a_value > b_value:
            signed_sum += 1
        elif a_value < b_value:
            signed_sum -= 1
    return round(signed_sum / len(values_a), 6)


def _holm_adjust(p_values: list[float]) -> list[float]:
    comparison_count = len(p_values)
    if comparison_count == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * comparison_count
    running_max = 0.0
    for rank, (original_index, p_value) in enumerate(indexed, start=1):
        corrected = min(1.0, (comparison_count - rank + 1) * p_value)
        running_max = max(running_max, corrected)
        adjusted[original_index] = running_max
    return [round(value, 6) for value in adjusted]


def _mcnemar_test(
    b_count: int,
    c_count: int,
) -> tuple[float, float]:
    """Return the statistic and an unrounded two-sided p-value.

    Holm multiplies this p by the number of remaining comparisons. Rounding it
    to six places first turns 1.97e-5 into 2e-5, and the adjusted value into
    1.2e-4 instead of 1.18e-4. Round when reporting, never before arithmetic.
    """
    discordant = b_count + c_count
    if discordant == 0:
        return 0.0, 1.0
    chi2_stat = ((abs(b_count - c_count) - 1) ** 2) / discordant
    # chi-square(1) CDF: erf(sqrt(x/2))
    p_value = 1 - math.erf(math.sqrt(chi2_stat / 2))
    return chi2_stat, max(0.0, min(1.0, p_value))


def _repeat_verdict(rows: list[dict[str, Any]], reduction: RepeatReduction) -> int:
    ordered = sorted(
        rows,
        key=lambda row: (
            int(row.get("experiment_repeat_index") or 0),
            str(row.get("run_tag") or ""),
        ),
    )
    verdicts = [int(int(row.get("violation_count", 0)) > 0) for row in ordered]
    if reduction == "first":
        return verdicts[0]
    if reduction == "last":
        return verdicts[-1]
    if reduction == "majority":
        if len(verdicts) % 2 == 0:
            raise ValueError("Majority repeat reduction requires an odd repeat count.")
        return int(sum(verdicts) > len(verdicts) / 2)
    if reduction == "any":
        return int(any(verdicts))
    raise ValueError(f"Unsupported repeat reduction: {reduction}")


def _is_complete_repeat_group(
    rows: list[dict[str, Any]], expected_repeats: int | None
) -> bool:
    if expected_repeats is None:
        return True
    indices = [row.get("experiment_repeat_index") for row in rows]
    if any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
        raise ValueError("Official paired analysis requires integer repeat indices.")
    if len(set(indices)) != len(indices):
        raise ValueError("A paired-analysis group contains duplicate repeat indices.")
    expected_indices = set(range(1, expected_repeats + 1))
    unexpected = set(indices) - expected_indices
    if unexpected:
        raise ValueError(f"Repeat indices exceed the declared schedule: {unexpected}")
    return set(indices) == expected_indices


def _repeat_group_accounting(
    rows: list[dict[str, Any]], expected_repeats: int | None
) -> tuple[int, int]:
    if expected_repeats is None:
        return 0, 0
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["model_label"]),
            str(row.get("incident_dir_id") or row["incident_id"]),
        )
        grouped.setdefault(key, []).append(row)
    complete = sum(
        _is_complete_repeat_group(group, expected_repeats)
        for group in grouped.values()
    )
    return complete, len(grouped) - complete


def _build_pairwise_rows_from_runs(
    rows: list[dict[str, Any]],
    repeat_reduction: RepeatReduction = "last",
    expected_repeats: int | None = None,
) -> list[dict[str, Any]]:
    grouped_by_model_incident: dict[
        tuple[str | None, str, str], list[dict[str, Any]]
    ] = {}
    label_factors: dict[tuple[str | None, str], tuple[str, str] | None] = {}
    for row in _with_all_incident_rows(rows):
        model_label = str(row["model_label"])
        incident_key = str(row.get("incident_dir_id") or row["incident_id"])
        corpus_stratum = row.get("corpus_stratum")
        if corpus_stratum is not None:
            corpus_stratum = str(corpus_stratum)
        key = (corpus_stratum, model_label, incident_key)
        grouped_by_model_incident.setdefault(key, []).append(row)
        deployment = str(row.get("llm_deployment") or "")
        prompt_condition = str(row.get("llm_arm") or "")
        factors = (deployment, prompt_condition) if deployment and prompt_condition else None
        factor_key = (corpus_stratum, model_label)
        previous = label_factors.setdefault(factor_key, factors)
        if previous != factors:
            raise ValueError(f"Model label {model_label!r} maps to conflicting factors.")

    stratum_models: dict[str | None, dict[str, dict[str, dict[str, Any]]]] = {}
    for (
        corpus_stratum,
        model_label,
        incident_id,
    ), repeated_rows in grouped_by_model_incident.items():
        if not _is_complete_repeat_group(repeated_rows, expected_repeats):
            continue
        stratum_models.setdefault(corpus_stratum, {}).setdefault(model_label, {})[
            incident_id
        ] = {"violation_count": _repeat_verdict(repeated_rows, repeat_reduction)}

    out: list[dict[str, Any]] = []
    for corpus_stratum in sorted(stratum_models, key=lambda value: value or ""):
        model_to_incidents = stratum_models[corpus_stratum]
        models = sorted(model_to_incidents)
        for model_index in range(len(models)):
            for comparison_index in range(model_index + 1, len(models)):
                model_a = models[model_index]
                model_b = models[comparison_index]
                factors_a = label_factors[(corpus_stratum, model_a)]
                factors_b = label_factors[(corpus_stratum, model_b)]
                if factors_a is not None and factors_b is not None:
                    same_model = factors_a[0] == factors_b[0]
                    same_condition = factors_a[1] == factors_b[1]
                    if same_model == same_condition:
                        # A planned contrast changes exactly one experimental
                        # factor. Cross-model, cross-condition pairs confound both.
                        continue
                    contrast_type = "prompt_condition" if same_model else "model"
                else:
                    # Preserve generic analysis for legacy labels that do not
                    # encode the campaign's model|arm factorial design.
                    contrast_type = "generic"
                incidents_a = model_to_incidents[model_a]
                incidents_b = model_to_incidents[model_b]
                common_incidents = sorted(set(incidents_a) & set(incidents_b))
                if not common_incidents:
                    continue

                row = _pairwise_row(
                    corpus_stratum=corpus_stratum,
                    model_a=model_a,
                    model_b=model_b,
                    incidents_a=incidents_a,
                    incidents_b=incidents_b,
                    common_incidents=common_incidents,
                )
                row["contrast_type"] = contrast_type
                out.append(row)

    if out:
        adjusted_z = _holm_adjust([float(item["p_value_two_sided"]) for item in out])
        adjusted_m = _holm_adjust(
            [float(item["mcnemar_p_value_two_sided"]) for item in out]
        )
        for row, holm_z, holm_m in zip(out, adjusted_z, adjusted_m, strict=True):
            row["p_value_holm"] = holm_z
            row["mcnemar_p_value_holm"] = holm_m
        # Rounded once, after every adjustment, so the report reads cleanly and
        # no downstream arithmetic ever sees a truncated p-value.
        for row in out:
            for field in (
                "p_value_two_sided",
                "p_value_holm",
                "mcnemar_chi2",
                "mcnemar_p_value_two_sided",
                "mcnemar_p_value_holm",
            ):
                row[field] = round(float(row[field]), 6)
    return out


def _pairwise_row(
    *,
    corpus_stratum: str | None,
    model_a: str,
    model_b: str,
    incidents_a: dict[str, dict[str, Any]],
    incidents_b: dict[str, dict[str, Any]],
    common_incidents: list[str],
) -> dict[str, Any]:
    runs_with_violations_a = sum(
        int(incidents_a[incident_id]["violation_count"]) > 0
        for incident_id in common_incidents
    )
    runs_with_violations_b = sum(
        int(incidents_b[incident_id]["violation_count"]) > 0
        for incident_id in common_incidents
    )
    b_count = sum(
        (
            int(incidents_a[incident_id]["violation_count"]) > 0
            and int(incidents_b[incident_id]["violation_count"]) == 0
        )
        for incident_id in common_incidents
    )
    c_count = sum(
        (
            int(incidents_a[incident_id]["violation_count"]) == 0
            and int(incidents_b[incident_id]["violation_count"]) > 0
        )
        for incident_id in common_incidents
    )
    paired_count = len(common_incidents)
    binary_a = [
        int(int(incidents_a[incident_id]["violation_count"]) > 0)
        for incident_id in common_incidents
    ]
    binary_b = [
        int(int(incidents_b[incident_id]["violation_count"]) > 0)
        for incident_id in common_incidents
    ]
    z_score, p_value = _two_proportion_ztest(
        runs_with_violations_a,
        paired_count,
        runs_with_violations_b,
        paired_count,
    )
    mcnemar_chi2, mcnemar_p = _mcnemar_test(b_count, c_count)
    rate_a = runs_with_violations_a / paired_count
    rate_b = runs_with_violations_b / paired_count
    return {
        "model_a": model_a,
        "model_b": model_b,
        **({"corpus_stratum": corpus_stratum} if corpus_stratum is not None else {}),
        "paired_incident_count": paired_count,
        "run_count_a": paired_count,
        "run_count_b": paired_count,
        "violation_rate_a": round(rate_a, 4),
        "violation_rate_b": round(rate_b, 4),
        "rate_diff_a_minus_b": round(rate_a - rate_b, 4),
        "z_score": round(z_score, 6),
        "p_value_two_sided": round(p_value, 6),
        "p_value_holm": 1.0,
        "cohens_h": _cohens_h(rate_a, rate_b),
        "cliffs_delta": _cliffs_delta_binary(binary_a, binary_b),
        "mcnemar_b_count": b_count,
        "mcnemar_c_count": c_count,
        "mcnemar_chi2": mcnemar_chi2,
        "mcnemar_p_value_two_sided": mcnemar_p,
        "mcnemar_p_value_holm": 1.0,
    }


def analyze_incident_outputs_full(
    outputs_incidents_dir: Path,
    incident_filter: set[str] | None = None,
    canonical_only: bool = False,
    run_id: str | None = None,
    repeat_reduction: RepeatReduction = "last",
    expected_repeats: int | None = None,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    verifier_files = _collect_verifier_files(
        outputs_incidents_dir,
        canonical_only=canonical_only,
    )
    if incident_filter:
        verifier_files = [
            path for path in verifier_files if path.parent.name in incident_filter
        ]
    if not verifier_files:
        raise FileNotFoundError(
            "No verifier output found under "
            f"{outputs_incidents_dir}/*/verifier_output*.json"
        )

    selected_runs = _select_latest_runs_per_incident_model(
        verifier_files,
        run_id=(
            [rid.strip() for rid in run_id.split(",") if rid.strip()]
            if isinstance(run_id, str)
            else run_id
        ),
    )
    state = _create_accumulator()

    for verifier, row in selected_runs:
        _accumulate_run_verifier(state, verifier, row)

    total_candidate_count, total_curated_count, total_ab_gap_count = (
        _accumulate_attack_unique(state)
    )
    unique_incident_count = len(state.unique_incident_ids)
    total_prompt_tokens = sum(int(row["llm_prompt_tokens"]) for row in state.rows)
    total_completion_tokens = sum(
        int(row["llm_completion_tokens"]) for row in state.rows
    )
    total_tokens = sum(int(row["llm_total_tokens"]) for row in state.rows)
    total_latency_ms = sum(int(row.get("llm_latency_ms", 0)) for row in state.rows)
    total_estimated_cost = sum(
        float(row["llm_cost_estimated_usd"]) for row in state.rows
    )
    total_hallucinated_actions = sum(
        int(row["llm_hallucinated_action_count"]) for row in state.rows
    )
    avg_hallucination_rate = (
        round(
            sum(float(row["llm_hallucination_rate"]) for row in state.rows)
            / len(state.rows),
            4,
        )
        if state.rows
        else 0.0
    )
    total_enforcement_inserted = sum(
        int(row["enforcement_actions_inserted_count"]) for row in state.rows
    )
    total_enforcement_removed = sum(
        int(row["enforcement_actions_removed_count"]) for row in state.rows
    )
    total_enforcement_deferred = sum(
        int(row.get("enforcement_actions_deferred_count", 0)) for row in state.rows
    )
    total_enforcement_reordered = sum(
        int(row["enforcement_actions_reordered_count"]) for row in state.rows
    )
    total_enforcement_modification_count = sum(
        int(row["enforcement_action_modification_count"]) for row in state.rows
    )
    enforcement_modified_run_count = sum(
        int(bool(row["enforcement_modified"])) for row in state.rows
    )
    average_estimated_cost = (
        round(total_estimated_cost / len(state.rows), 6) if state.rows else 0.0
    )
    average_latency_ms = (
        round(total_latency_ms / len(state.rows), 2) if state.rows else 0.0
    )
    summary = {
        "run_count": len(state.rows),
        "incident_count": unique_incident_count,
        "incidents_with_violations": len(state.incidents_with_violations_set),
        "incident_violation_rate": (
            round(len(state.incidents_with_violations_set) / unique_incident_count, 4)
            if unique_incident_count
            else 0.0
        ),
        "mode_counts": dict(state.mode_counter),
        "violations_by_rule": dict(state.violations_by_rule),
        "violations_by_type": dict(state.violations_by_type),
        "violations_by_severity": dict(state.violations_by_severity),
        "llm_only_actions": dict(state.llm_only_actions),
        "human_only_actions": dict(state.human_only_actions),
        "attack_candidate_count_total": total_candidate_count,
        "attack_curated_count_total": total_curated_count,
        "attack_ab_gap_count_total": total_ab_gap_count,
        "attack_curated_to_candidate_ratio": (
            round(total_curated_count / total_candidate_count, 4)
            if total_candidate_count
            else 0.0
        ),
        "attack_technique_counts": dict(state.technique_counter),
        "attack_tactic_counts": dict(state.tactic_counter),
        "llm_prompt_tokens_total": total_prompt_tokens,
        "llm_completion_tokens_total": total_completion_tokens,
        "llm_total_tokens_total": total_tokens,
        "llm_latency_ms_total": total_latency_ms,
        "llm_latency_ms_avg_per_run": average_latency_ms,
        "llm_cost_estimated_usd_total": round(total_estimated_cost, 6),
        "llm_cost_estimated_usd_avg_per_run": average_estimated_cost,
        "llm_hallucinated_action_count_total": total_hallucinated_actions,
        "llm_hallucination_rate_avg_per_run": avg_hallucination_rate,
        "enforcement_actions_inserted_count_total": total_enforcement_inserted,
        "enforcement_actions_removed_count_total": total_enforcement_removed,
        "enforcement_actions_deferred_count_total": total_enforcement_deferred,
        "enforcement_actions_reordered_count_total": total_enforcement_reordered,
        "enforcement_action_modification_count_total": (
            total_enforcement_modification_count
        ),
        "enforcement_modified_run_count": enforcement_modified_run_count,
        "enforcement_modification_rate": (
            round(enforcement_modified_run_count / len(state.rows), 4)
            if state.rows
            else 0.0
        ),
        **summarize_reference_metrics(state.rows),
    }
    tactic_rates, tactic_support = _violation_rate_by_key(
        state.tactic_run_counter,
        state.tactic_violation_run_counter,
    )
    technique_rates, technique_support = _violation_rate_by_key(
        state.technique_run_counter,
        state.technique_violation_run_counter,
    )
    summary["violation_rate_by_tactic"] = tactic_rates
    summary["violation_rate_by_technique"] = technique_rates
    summary["support_count_by_tactic"] = tactic_support
    summary["support_count_by_technique"] = technique_support
    incident_type_rates, incident_type_support = _violation_rate_by_key(
        state.incident_type_run_counter,
        state.incident_type_violation_run_counter,
    )
    severity_rates, severity_support = _violation_rate_by_key(
        state.severity_run_counter,
        state.severity_violation_run_counter,
    )
    criticality_rates, criticality_support = _violation_rate_by_key(
        state.criticality_run_counter,
        state.criticality_violation_run_counter,
    )
    summary["violation_rate_by_incident_type"] = incident_type_rates
    summary["violation_rate_by_severity"] = severity_rates
    summary["violation_rate_by_asset_criticality"] = criticality_rates
    summary["support_count_by_incident_type"] = incident_type_support
    summary["support_count_by_severity"] = severity_support
    summary["support_count_by_asset_criticality"] = criticality_support
    summary["corpus_strata"] = _corpus_stratum_summaries(state.rows)
    by_model_rows = _build_by_model_rows(state.rows)
    by_rule_rows = _build_by_rule_rows(
        state.violations_by_rule_and_model,
        state.hard_violations_by_rule_and_model,
    )
    by_rule_treatment_rows = _build_by_rule_treatment_rows(
        state.run_count_by_model_arm,
        state.violation_run_count_by_model_arm_rule,
    )
    pairwise_rows = _build_pairwise_rows_from_runs(
        state.rows,
        repeat_reduction=repeat_reduction,
        expected_repeats=expected_repeats,
    )
    complete_groups, incomplete_groups = _repeat_group_accounting(
        state.rows, expected_repeats
    )
    summary["pairwise_comparison_count"] = len(pairwise_rows)
    summary["repeat_reduction"] = repeat_reduction
    summary["expected_repeats"] = expected_repeats
    summary["complete_repeat_group_count"] = complete_groups
    summary["incomplete_repeat_group_count"] = incomplete_groups
    return (
        summary,
        state.rows,
        by_model_rows,
        by_rule_rows,
        by_rule_treatment_rows,
        pairwise_rows,
    )


def analyze_incident_outputs(
    outputs_incidents_dir: Path,
    incident_filter: set[str] | None = None,
    canonical_only: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    (
        summary,
        rows,
        _by_model_rows,
        _by_rule_rows,
        _by_rule_treatment_rows,
        _pairwise_rows,
    ) = analyze_incident_outputs_full(
        outputs_incidents_dir,
        incident_filter,
        canonical_only=canonical_only,
    )
    return summary, rows


def _write_summary_json(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_analysis_bundle(data: AnalysisBundleInput) -> dict[str, Any]:
    bundle = AnalysisBundleModel(
        generated_at_utc=datetime.now(UTC).isoformat(),
        eval_protocol_version=data.eval_protocol_version,
        outputs_incidents_dir=repo_relative_path(
            data.outputs_incidents_dir,
            data.repo_root,
        ),
        incident_filter=sorted(data.incident_filter) if data.incident_filter else [],
        summary=SummaryModel.model_validate(data.summary),
        incidents=[IncidentRowModel.model_validate(item) for item in data.rows],
        by_model=[ByModelRowModel.model_validate(item) for item in data.by_model_rows],
        by_rule=[ByRuleRowModel.model_validate(item) for item in data.by_rule_rows],
        by_rule_treatment=[
            ByRuleTreatmentRowModel.model_validate(item)
            for item in data.by_rule_treatment_rows
        ],
        pairwise=[PairwiseRowModel.model_validate(item) for item in data.pairwise_rows],
    )
    return bundle.model_dump(mode="json")


def _write_bundle_json(path: Path, bundle: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_rows_csv(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    fieldnames: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and fieldnames is None:
        raise ValueError("No rows available for CSV export.")
    csv_fields = fieldnames or list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="soc_llm_policy.analyze")
    parser.add_argument(
        "--outputs-incidents-dir",
        default="results/incidents",
        help="Directory containing */verifier_output*.json subfolders.",
    )
    parser.add_argument(
        "--summary-json",
        default="results/analysis/summary.json",
        help="Path to aggregated JSON summary output.",
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Also export derived CSVs (incidents/by_model/by_rule/pairwise).",
    )
    parser.add_argument(
        "--incidents-csv",
        default="results/analysis/incidents.csv",
        help="Path to incident-level metrics CSV.",
    )
    parser.add_argument(
        "--analysis-bundle-json",
        default="results/analysis/analysis_bundle.json",
        help="Path to canonical typed JSON for analytics consumption.",
    )
    parser.add_argument(
        "--eval-protocol-version",
        default="official",
        help="Evaluation state label used to generate this bundle.",
    )
    parser.add_argument(
        "--by-model-csv",
        default="results/analysis/by_model.csv",
        help="Path to model-level aggregated CSV.",
    )
    parser.add_argument(
        "--by-rule-csv",
        default="results/analysis/by_rule.csv",
        help="Path to rule-by-model aggregated CSV.",
    )
    parser.add_argument(
        "--pairwise-csv",
        default="results/analysis/model_pairwise.csv",
        help="Path to pairwise model statistical comparison CSV.",
    )
    parser.add_argument(
        "--by-rule-treatment-csv",
        default="results/analysis/by_rule_treatment.csv",
        help="Path to per-rule treatment-effect CSV for llm_zero vs llm_policy_prompt.",
    )
    parser.add_argument(
        "--incidents",
        default=None,
        help="Optional CSV incident filter. Example: INC_001,INC_002",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "Restrict the analysis to one experiment run and keep every cell "
            "(repeats included). Required for official numbers."
        ),
    )
    parser.add_argument(
        "--repeat-reduction",
        default="last",
        choices=["first", "last", "majority", "any"],
        help="Reduce repeated cells to one verdict for paired incident tests.",
    )
    parser.add_argument(
        "--expected-repeats",
        type=int,
        default=None,
        help=(
            "Require this complete repeat schedule for paired tests. Groups with "
            "terminal non-plan observations remain in accounting but not in a "
            "paired contrast."
        ),
    )
    parser.add_argument(
        "--canonical-only",
        action="store_true",
        help=(
            "Analyze only canonical verifier_output.json per incident, "
            "ignoring versioned histories."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.expected_repeats is not None and args.expected_repeats < 1:
        raise ValueError("--expected-repeats must be positive.")
    outputs_incidents_dir = Path(args.outputs_incidents_dir).resolve()
    summary_json = Path(args.summary_json).resolve()
    incidents_csv = Path(args.incidents_csv).resolve()
    analysis_bundle_json = Path(args.analysis_bundle_json).resolve()
    by_model_csv = Path(args.by_model_csv).resolve()
    by_rule_csv = Path(args.by_rule_csv).resolve()
    by_rule_treatment_csv = Path(args.by_rule_treatment_csv).resolve()
    pairwise_csv = Path(args.pairwise_csv).resolve()
    incident_filter = _parse_incident_filter(args.incidents)

    (
        summary,
        rows,
        by_model_rows,
        by_rule_rows,
        by_rule_treatment_rows,
        pairwise_rows,
    ) = analyze_incident_outputs_full(
        outputs_incidents_dir,
        incident_filter,
        canonical_only=bool(args.canonical_only),
        run_id=args.run_id,
        repeat_reduction=args.repeat_reduction,
        expected_repeats=args.expected_repeats,
    )
    _write_summary_json(summary_json, summary)
    bundle = _build_analysis_bundle(
        AnalysisBundleInput(
            eval_protocol_version=str(args.eval_protocol_version),
            repo_root=Path.cwd().resolve(),
            outputs_incidents_dir=outputs_incidents_dir,
            incident_filter=incident_filter,
            summary=summary,
            rows=rows,
            by_model_rows=by_model_rows,
            by_rule_rows=by_rule_rows,
            by_rule_treatment_rows=by_rule_treatment_rows,
            pairwise_rows=pairwise_rows,
        )
    )
    _write_bundle_json(analysis_bundle_json, bundle)
    if args.export_csv:
        _write_rows_csv(incidents_csv, rows)
        _write_rows_csv(by_model_csv, by_model_rows)
        _write_rows_csv(
            by_rule_csv,
            by_rule_rows,
            fieldnames=[
                "model_label",
                "rule_id",
                "violation_count",
                "hard_violation_count",
            ],
        )
        _write_rows_csv(
            by_rule_treatment_csv,
            by_rule_treatment_rows,
            fieldnames=[
                "rule_id",
                "model",
                "llm_zero_run_count",
                "llm_policy_prompt_run_count",
                "llm_zero_violation_count",
                "llm_policy_prompt_violation_count",
                "llm_zero_violation_rate",
                "llm_policy_prompt_violation_rate",
                "violation_rate_diff_zero_minus_policy",
            ],
        )
        _write_rows_csv(
            pairwise_csv,
            pairwise_rows,
            fieldnames=[
                "model_a",
                "model_b",
                "paired_incident_count",
                "run_count_a",
                "run_count_b",
                "violation_rate_a",
                "violation_rate_b",
                "rate_diff_a_minus_b",
                "z_score",
                "p_value_two_sided",
                "p_value_holm",
                "cohens_h",
                "cliffs_delta",
                "mcnemar_b_count",
                "mcnemar_c_count",
                "mcnemar_chi2",
                "mcnemar_p_value_two_sided",
                "mcnemar_p_value_holm",
            ],
        )

    print(f"Summary saved at: {summary_json}")
    print(f"Typed analysis bundle saved at: {analysis_bundle_json}")
    if args.export_csv:
        print(f"Incident CSV saved at: {incidents_csv}")
        print(f"Model CSV saved at: {by_model_csv}")
        print(f"Rule CSV saved at: {by_rule_csv}")
        print(f"Rule treatment CSV saved at: {by_rule_treatment_csv}")
        print(f"Pairwise model CSV saved at: {pairwise_csv}")
    print(
        "Incidents analyzed: "
        f"{summary['incident_count']} | "
        "incidents with at least one violating run: "
        f"{summary['incident_violation_rate']:.2%}"
    )


if __name__ == "__main__":
    main()
