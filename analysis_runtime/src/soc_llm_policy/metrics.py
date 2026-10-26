from __future__ import annotations

from dataclasses import dataclass

from soc_llm_policy.engine import Violation
from soc_llm_policy.result_models import (
    AttackCandidatesSummary,
    AttackContextSummary,
    IncidentMetrics,
)


@dataclass(frozen=True)
class IncidentMetricInput:
    human_actions: list[str]
    llm_actions: list[str]
    enforced_actions: list[str]
    violations: list[Violation]
    attack_candidates: AttackCandidatesSummary
    attack_summary: AttackContextSummary
    llm_hallucinated_actions: int = 0


def _safe_div(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


def _prf_jaccard(predicted: set[str], expected: set[str]) -> tuple[float | None, ...]:
    if not expected:
        return None, None, None, None
    intersection = len(predicted & expected)
    precision = _safe_div(intersection, len(predicted))
    recall = _safe_div(intersection, len(expected))
    f1_denominator = precision + recall
    f1 = round((2 * precision * recall / f1_denominator), 4) if f1_denominator else 0.0
    union = len(predicted | expected)
    jaccard = _safe_div(intersection, union)
    return precision, recall, f1, jaccard


def build_incident_metrics(data: IncidentMetricInput) -> IncidentMetrics:
    """Build per-incident metrics for downstream analysis."""
    severities = [violation.severity for violation in data.violations]
    llm_set = set(data.llm_actions)
    human_set = set(data.human_actions)
    enforced_set = set(data.enforced_actions)
    precision_raw, recall_raw, f1_raw, jaccard_raw = _prf_jaccard(llm_set, human_set)
    precision_enf, recall_enf, f1_enf, jaccard_enf = _prf_jaccard(
        enforced_set,
        human_set,
    )
    inserted = sum(
        violation.type == "missing_mandatory" for violation in data.violations
    )
    removed = sum(
        violation.repair == "remove" for violation in data.violations
    )
    deferred = sum(
        violation.type == "approval_deferred" for violation in data.violations
    )
    reordered = sum(
        violation.type == "order_violation" for violation in data.violations
    )
    modifications = inserted + removed + deferred + reordered
    llm_count = len(data.llm_actions)
    hallucination_rate = (
        round(data.llm_hallucinated_actions / llm_count, 4)
        if llm_count
        else 0.0
    )
    return IncidentMetrics(
        human_action_count=len(data.human_actions),
        llm_action_count=len(data.llm_actions),
        enforced_action_count=len(data.enforced_actions),
        violation_count=len(data.violations),
        hard_violation_count=sum(severity == "hard" for severity in severities),
        soft_violation_count=sum(severity == "soft" for severity in severities),
        warning_violation_count=sum(severity == "warning" for severity in severities),
        llm_only_action_count=len(
            sorted(set(data.llm_actions) - set(data.human_actions))
        ),
        human_only_action_count=len(
            sorted(set(data.human_actions) - set(data.llm_actions))
        ),
        attack_match_count=data.attack_summary.match_count,
        attack_high_confidence_match_count=(
            data.attack_summary.high_confidence_match_count
        ),
        attack_candidate_count=data.attack_candidates.candidate_count,
        attack_technique_count=len(data.attack_summary.techniques),
        attack_tactic_count=len(data.attack_summary.tactics),
        enforcement_actions_inserted_count=inserted,
        enforcement_actions_removed_count=removed,
        enforcement_actions_deferred_count=deferred,
        enforcement_actions_reordered_count=reordered,
        enforcement_action_modification_count=modifications,
        enforcement_modified=modifications > 0,
        llm_hallucinated_action_count=data.llm_hallucinated_actions,
        llm_hallucination_rate=hallucination_rate,
        precision_raw=precision_raw,
        recall_raw=recall_raw,
        f1_raw=f1_raw,
        jaccard_raw=jaccard_raw,
        precision_enforced=precision_enf,
        recall_enforced=recall_enf,
        f1_enforced=f1_enf,
        jaccard_enforced=jaccard_enf,
        delta_jaccard=(round(jaccard_enf - jaccard_raw, 4)
                       if jaccard_enf is not None and jaccard_raw is not None else None),
    )
