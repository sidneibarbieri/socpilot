"""False-accept / false-reject of text-level compliance checks vs the typed verifier.

The typed verifier is the reference oracle: a plan is non-compliant iff it incurs a
hard violation. Each baseline produces its own accept/reject verdict on the same
plan; this module reports, per baseline, false-accept (baseline accepts a plan the
verifier rejects) and false-reject (baseline rejects a plan the verifier accepts),
each with an explicit denominator.

No model is called here. The verifier and prompt-only baseline are computed;
the LLM-judge verdict is supplied as input
because it is model-produced. This keeps the analysis pure and testable.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from soc_llm_policy.engine import EnforcementRequest, enforce_policy

SEPARATION_ANALYSIS_VERSION = "1.2.0"
SEPARATION_INPUT_SCHEMA_VERSION = 1
SEPARATION_REPORT_SCHEMA_VERSION = 3

BASELINE_DESCRIPTIONS = {
    "llm_judge": (
        "A separately prompted model classifies the persisted plan as compliant "
        "or non-compliant without repairing or enforcing it."
    ),
    "prompt_guardrail": (
        "Accepts every proposed plan; a prompt-only guardrail has no post-hoc "
        "detector, so its false-accept rate equals the verifier non-compliant rate."
    ),
}


class SeparationInputViolation(BaseModel):
    model_config = ConfigDict(extra="allow")

    rule_id: str = Field(..., min_length=1)
    severity: Literal["hard", "soft", "warning"]
    type: str = Field(..., min_length=1)
    action: str = Field(..., min_length=1)


class SeparationVerifierOutput(BaseModel):
    """Schema-constrained projection of a persisted verifier output."""

    model_config = ConfigDict(extra="allow")

    incident_id: str = Field(..., min_length=1)
    experiment_cell_id: str | None = None
    llm_arm: str | None = None
    llm_actions: list[str]
    violations: list[SeparationInputViolation]


class SeparationCell(BaseModel):
    cell_id: str = Field(..., min_length=1)
    incident_id: str = Field(..., min_length=1)
    llm_arm: str | None = None
    plan_actions: list[str]
    verifier_compliant: bool
    hard_violation_rule_ids: list[str]
    baseline_compliant: dict[str, bool]
    baseline_reasons: dict[str, str] = Field(default_factory=dict)


class LLMJudgeVerdict(BaseModel):
    cell_id: str = Field(..., min_length=1)
    compliant: bool
    reason: str = Field(..., min_length=1)


class LLMJudgeVerdictSet(BaseModel):
    schema_version: Literal[1] = 1
    run_id: str = Field(..., min_length=1)
    judge_model: str = Field(..., min_length=1)
    judge_prompt_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    verdicts: list[LLMJudgeVerdict]

    @model_validator(mode="after")
    def validate_cell_ids(self) -> LLMJudgeVerdictSet:
        cell_ids = [verdict.cell_id for verdict in self.verdicts]
        if len(cell_ids) != len(set(cell_ids)):
            raise ValueError("LLM-judge verdict cell ids must be unique")
        return self


class BaselineSeparation(BaseModel):
    baseline_id: str = Field(..., min_length=1)
    cell_count: int = Field(..., ge=0)
    missing_count: int = Field(..., ge=0)
    verifier_noncompliant_count: int = Field(..., ge=0)
    verifier_compliant_count: int = Field(..., ge=0)
    false_accept_count: int = Field(..., ge=0)
    false_accept_denominator: int = Field(..., ge=0)
    false_reject_count: int = Field(..., ge=0)
    false_reject_denominator: int = Field(..., ge=0)
    false_accept_rate: float | None = None
    false_reject_rate: float | None = None
    agreement_count: int = Field(..., ge=0)
    disagreement_count: int = Field(..., ge=0)

    @model_validator(mode="after")
    def validate_accounting(self) -> BaselineSeparation:
        if self.cell_count != (
            self.verifier_noncompliant_count + self.verifier_compliant_count
        ):
            raise ValueError("Verifier denominators must sum to cell_count")
        if self.false_accept_denominator != self.verifier_noncompliant_count:
            raise ValueError(
                "false_accept_denominator must equal verifier_noncompliant_count"
            )
        if self.false_reject_denominator != self.verifier_compliant_count:
            raise ValueError(
                "false_reject_denominator must equal verifier_compliant_count"
            )
        if self.false_accept_count > self.false_accept_denominator:
            raise ValueError("false_accept_count exceeds its denominator")
        if self.false_reject_count > self.false_reject_denominator:
            raise ValueError("false_reject_count exceeds its denominator")
        if self.agreement_count + self.disagreement_count != self.cell_count:
            raise ValueError("Agreement accounting must conserve cell_count")
        _validate_rate(
            self.false_accept_rate,
            self.false_accept_count,
            self.false_accept_denominator,
            "false_accept_rate",
        )
        _validate_rate(
            self.false_reject_rate,
            self.false_reject_count,
            self.false_reject_denominator,
            "false_reject_rate",
        )
        return self


class CellBaselineTrace(BaseModel):
    baseline_id: str = Field(..., min_length=1)
    baseline_compliant: bool
    outcome: Literal["agreement", "false_accept", "false_reject"]
    reason: str = Field(..., min_length=1)


class SeparationCellTrace(BaseModel):
    cell_id: str = Field(..., min_length=1)
    incident_id: str = Field(..., min_length=1)
    llm_arm: str | None = None
    plan_actions: list[str]
    verifier_compliant: bool
    hard_violation_rule_ids: list[str]
    baselines: list[CellBaselineTrace]


class SeparationReport(BaseModel):
    total_cells: int = Field(..., ge=0)
    verifier_noncompliant_count: int = Field(..., ge=0)
    verifier_compliant_count: int = Field(..., ge=0)
    baselines: list[BaselineSeparation]

    @model_validator(mode="after")
    def validate_accounting(self) -> SeparationReport:
        if self.total_cells != (
            self.verifier_noncompliant_count + self.verifier_compliant_count
        ):
            raise ValueError("Verifier counts must sum to total_cells")
        for baseline in self.baselines:
            if baseline.cell_count + baseline.missing_count != self.total_cells:
                raise ValueError(
                    f"Baseline {baseline.baseline_id!r} has inconsistent coverage"
                )
        return self


class SeparationReportEnvelope(BaseModel):
    schema_version: Literal[3] = SEPARATION_REPORT_SCHEMA_VERSION
    analysis_version: str = SEPARATION_ANALYSIS_VERSION
    input_schema_version: Literal[1] = SEPARATION_INPUT_SCHEMA_VERSION
    input_schema_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    run_id: str = Field(..., min_length=1)
    dataset_release_id: str = Field(..., min_length=1)
    generated_at: str = Field(..., min_length=1)
    git_commit: str | None = None
    constitutes_model_evidence: bool
    recommendation_source_kind: str = Field(..., min_length=1)
    limitations: list[str]
    baseline_descriptions: dict[str, str]
    excluded_baselines: list[str]
    report: SeparationReport
    cells: list[SeparationCellTrace]

    @model_validator(mode="after")
    def validate_cell_trace_accounting(self) -> SeparationReportEnvelope:
        if len(self.cells) != self.report.total_cells:
            raise ValueError("Cell traces must cover every reported cell")
        cell_ids = [cell.cell_id for cell in self.cells]
        if len(cell_ids) != len(set(cell_ids)):
            raise ValueError("Cell trace ids must be unique")
        expected_baselines = {item.baseline_id for item in self.report.baselines}
        observed_counts = {baseline_id: 0 for baseline_id in expected_baselines}
        for cell in self.cells:
            actual_baselines = {item.baseline_id for item in cell.baselines}
            if not actual_baselines.issubset(expected_baselines):
                raise ValueError(
                    f"Cell {cell.cell_id!r} has an undeclared baseline trace"
                )
            for baseline_id in actual_baselines:
                observed_counts[baseline_id] += 1
        expected_counts = {
            baseline.baseline_id: baseline.cell_count
            for baseline in self.report.baselines
        }
        if observed_counts != expected_counts:
            raise ValueError("Cell traces do not reproduce baseline coverage")
        if not self.constitutes_model_evidence and not any(
            "not constitute model evidence" in limitation.lower()
            for limitation in self.limitations
        ):
            raise ValueError(
                "Non-model reports must state that limit in the limitations"
            )
        return self


def verifier_compliant(request: EnforcementRequest) -> bool:
    """Reference verdict: compliant iff the plan incurs no hard violation."""
    violations, _ = enforce_policy(request)
    return not any(violation.severity == "hard" for violation in violations)


def prompt_guardrail_compliant() -> bool:
    """Prompt-only guardrail verdict: it has no post-hoc detector, so it accepts."""
    return True


def compliant_from_violations(violations: Iterable[Mapping]) -> bool:
    """Reference verdict read from a persisted verifier output: no hard violation."""
    return not any(violation.get("severity") == "hard" for violation in violations)


def build_separation_cells(
    verifier_outputs: list[Mapping],
) -> list[SeparationCell]:
    """Build cells from persisted verifier outputs and the deterministic baselines.

    LLM-judge is intentionally excluded here: it is model-produced and joins only
    from the provider campaign. This path stays zero-cost and deterministic.
    """
    cells: list[SeparationCell] = []
    for raw_output in verifier_outputs:
        output = SeparationVerifierOutput.model_validate(raw_output)
        plan_actions = list(output.llm_actions)
        cell_id = output.experiment_cell_id or (
            f"{output.incident_id}::{output.llm_arm or 'na'}"
        )
        hard_violation_rule_ids = sorted(
            violation.rule_id
            for violation in output.violations
            if violation.severity == "hard"
        )
        cells.append(
            SeparationCell(
                cell_id=str(cell_id),
                incident_id=output.incident_id,
                llm_arm=output.llm_arm,
                plan_actions=plan_actions,
                verifier_compliant=not hard_violation_rule_ids,
                hard_violation_rule_ids=hard_violation_rule_ids,
                baseline_compliant={
                    "prompt_guardrail": prompt_guardrail_compliant(),
                },
            )
        )
    cell_ids = [cell.cell_id for cell in cells]
    if len(cell_ids) != len(set(cell_ids)):
        raise ValueError("Verifier outputs contain duplicate experiment cell ids")
    return cells


def attach_llm_judge_verdicts(
    cells: list[SeparationCell],
    verdict_set: LLMJudgeVerdictSet,
) -> list[SeparationCell]:
    """Attach available model-produced judge verdicts without imputing failures."""
    verdicts = {verdict.cell_id: verdict for verdict in verdict_set.verdicts}
    expected_ids = {cell.cell_id for cell in cells}
    observed_ids = set(verdicts)
    if not observed_ids.issubset(expected_ids):
        unexpected = sorted(observed_ids - expected_ids)
        raise ValueError(
            "LLM-judge verdicts do not match separation cells: "
            f"unexpected={unexpected}"
        )
    return [
        cell.model_copy(
            update={
                "baseline_compliant": {
                    **cell.baseline_compliant,
                    **(
                        {"llm_judge": verdicts[cell.cell_id].compliant}
                        if cell.cell_id in verdicts
                        else {}
                    ),
                },
                "baseline_reasons": {
                    **cell.baseline_reasons,
                    **(
                        {"llm_judge": verdicts[cell.cell_id].reason}
                        if cell.cell_id in verdicts
                        else {}
                    ),
                },
            },
            deep=True,
        )
        for cell in cells
    ]


def _baseline_ids(cells: list[SeparationCell]) -> list[str]:
    return sorted(
        {
            baseline_id
            for cell in cells
            for baseline_id in cell.baseline_compliant
        }
    )


def separation_metrics(cells: list[SeparationCell]) -> SeparationReport:
    """Report false-accept and false-reject per baseline against the verifier."""
    if not cells:
        return SeparationReport(
            total_cells=0,
            verifier_noncompliant_count=0,
            verifier_compliant_count=0,
            baselines=[],
        )

    verifier_noncompliant = sum(not cell.verifier_compliant for cell in cells)
    verifier_compliant_count = sum(cell.verifier_compliant for cell in cells)

    baselines = []
    for baseline_id in _baseline_ids(cells):
        covered_cells = [
            cell for cell in cells if baseline_id in cell.baseline_compliant
        ]
        covered_noncompliant = sum(
            not cell.verifier_compliant for cell in covered_cells
        )
        covered_compliant = sum(cell.verifier_compliant for cell in covered_cells)
        false_accept = sum(
            cell.baseline_compliant[baseline_id] and not cell.verifier_compliant
            for cell in covered_cells
        )
        false_reject = sum(
            not cell.baseline_compliant[baseline_id] and cell.verifier_compliant
            for cell in covered_cells
        )
        agreement = sum(
            cell.baseline_compliant[baseline_id] == cell.verifier_compliant
            for cell in covered_cells
        )
        baselines.append(
            BaselineSeparation(
                baseline_id=baseline_id,
                cell_count=len(covered_cells),
                missing_count=len(cells) - len(covered_cells),
                verifier_noncompliant_count=covered_noncompliant,
                verifier_compliant_count=covered_compliant,
                false_accept_count=false_accept,
                false_accept_denominator=covered_noncompliant,
                false_reject_count=false_reject,
                false_reject_denominator=covered_compliant,
                false_accept_rate=(
                    round(false_accept / covered_noncompliant, 4)
                    if covered_noncompliant
                    else None
                ),
                false_reject_rate=(
                    round(false_reject / covered_compliant, 4)
                    if covered_compliant
                    else None
                ),
                agreement_count=agreement,
                disagreement_count=len(covered_cells) - agreement,
            )
        )
    return SeparationReport(
        total_cells=len(cells),
        verifier_noncompliant_count=verifier_noncompliant,
        verifier_compliant_count=verifier_compliant_count,
        baselines=baselines,
    )


def separation_cell_traces(
    cells: list[SeparationCell],
) -> list[SeparationCellTrace]:
    """Explain each aggregate outcome using deterministic cell-level evidence."""
    traces: list[SeparationCellTrace] = []
    for cell in cells:
        baseline_traces = [
            _baseline_trace(cell, baseline_id)
            for baseline_id in sorted(cell.baseline_compliant)
        ]
        traces.append(
            SeparationCellTrace(
                cell_id=cell.cell_id,
                incident_id=cell.incident_id,
                llm_arm=cell.llm_arm,
                plan_actions=cell.plan_actions,
                verifier_compliant=cell.verifier_compliant,
                hard_violation_rule_ids=cell.hard_violation_rule_ids,
                baselines=baseline_traces,
            )
        )
    return traces


def _baseline_trace(
    cell: SeparationCell,
    baseline_id: str,
) -> CellBaselineTrace:
    baseline_compliant = cell.baseline_compliant[baseline_id]
    if baseline_compliant == cell.verifier_compliant:
        outcome: Literal["agreement", "false_accept", "false_reject"] = "agreement"
    elif baseline_compliant:
        outcome = "false_accept"
    else:
        outcome = "false_reject"
    return CellBaselineTrace(
        baseline_id=baseline_id,
        baseline_compliant=baseline_compliant,
        outcome=outcome,
        reason=_baseline_reason(cell, baseline_id, outcome),
    )


def _baseline_reason(
    cell: SeparationCell,
    baseline_id: str,
    outcome: Literal["agreement", "false_accept", "false_reject"],
) -> str:
    reason_builder = {
        "prompt_guardrail": _prompt_guardrail_reason,
        "llm_judge": _llm_judge_reason,
    }.get(baseline_id)
    if reason_builder is None:
        raise ValueError(f"Unsupported baseline: {baseline_id}")
    return reason_builder(cell, outcome)


def _prompt_guardrail_reason(
    cell: SeparationCell,
    outcome: Literal["agreement", "false_accept", "false_reject"],
) -> str:
    hard_rules = ", ".join(cell.hard_violation_rule_ids)
    if outcome == "false_accept":
        return f"accept-all baseline missed hard violation(s): {hard_rules}"
    return "accept-all baseline agreed with a verifier-compliant plan"


def _llm_judge_reason(
    cell: SeparationCell,
    outcome: Literal["agreement", "false_accept", "false_reject"],
) -> str:
    hard_rules = ", ".join(cell.hard_violation_rule_ids)
    judge_reason = cell.baseline_reasons["llm_judge"]
    if outcome == "false_accept":
        return (
            f"LLM judge missed hard violation(s) {hard_rules}; "
            f"judge rationale: {judge_reason}"
        )
    if outcome == "false_reject":
        return (
            "LLM judge rejected a verifier-compliant plan; "
            f"judge rationale: {judge_reason}"
        )
    return f"LLM judge agreed with the typed verifier: {judge_reason}"


def _validate_rate(
    rate: float | None,
    numerator: int,
    denominator: int,
    field_name: str,
) -> None:
    expected = round(numerator / denominator, 4) if denominator else None
    if rate != expected:
        raise ValueError(f"{field_name} must equal count / denominator")
