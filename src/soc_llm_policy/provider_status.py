"""Read operational progress without running models or interpreting their outputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from soc_llm_policy.provider_cost import ProviderCostTracker
from soc_llm_policy.provider_plan import ProviderExecutionPlan, load_provider_plan
from soc_llm_policy.provider_readiness import load_provider_readiness

type CellStatus = Literal[
    "planned", "running", "completed", "refused", "parse_failed", "failed"
]
type Count = Annotated[int, Field(strict=True, ge=0)]
_TERMINAL_OBSERVATIONS = {"completed", "refused", "parse_failed"}


class LedgerRow(BaseModel):
    status: CellStatus
    cell_id: str | None = None
    task_id: str | None = None
    run_id: str | None = None
    protocol_freeze_sha256: str | None = None


class ProgressLedger(BaseModel):
    status_counts: dict[CellStatus, Count]
    cells: list[LedgerRow] | None = None
    tasks: list[LedgerRow] | None = None
    cell_count: Count | None = None
    provider_task_count: Count | None = None
    unavailable_source_count: Count | None = None
    task_count_ceiling: Count | None = None
    study_id: str | None = None
    protocol_freeze_sha256: str | None = None

    @property
    def rows(self) -> list[LedgerRow]:
        return self.cells if self.cells is not None else self.tasks or []

    @model_validator(mode="after")
    def validate_accounting(self) -> ProgressLedger:
        if (self.cells is None) == (self.tasks is None):
            raise ValueError("Progress ledger must contain exactly one row collection.")
        count = self.cell_count if self.cells is not None else self.provider_task_count
        if count != len(self.rows):
            raise ValueError("Progress ledger count differs from its rows.")
        identities = [
            row.cell_id if self.cells is not None else row.task_id for row in self.rows
        ]
        if any(not identity for identity in identities) or len(set(identities)) != count:
            raise ValueError("Progress ledger contains missing or duplicate identities.")
        observed = Counter(row.status for row in self.rows)
        if observed != Counter(self.status_counts):
            raise ValueError("Progress ledger status counts differ from its rows.")
        return self


class CostStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    record_state: Literal["absent", "present"]
    accrued_usd: float | None
    ambiguous_reserved_usd: float | None
    budget_exposure_usd: float | None
    response_count: Count | None
    unknown_cost_count: Count | None


class StudyStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    state: Literal["no_ledger", "recorded", "failure_recorded", "terminal_pending_audit"]
    expected_task_ceiling: Count
    materialized_tasks: Count | None = None
    unavailable_sources: Count | None = None
    status_counts: dict[CellStatus, Count] | None = None
    ledger_updated_at_utc: str | None = None
    provider_cost: CostStatus


class ExecutionStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    generated_at_utc: str
    plan_id: str
    protocol_sha256: str
    approval_status: Literal["draft", "approved"]
    approved_budget_usd: float | None
    scientific_task_ceiling: Count
    provider_transport_attempt_limit: Count
    provider_transport_attempt_ceiling: Count
    readiness_state: Literal["no_report", "passed"]
    readiness_cost: CostStatus
    studies: dict[str, StudyStatus]


def _cost_status(path: Path) -> CostStatus:
    if not path.exists():
        return CostStatus(
            record_state="absent", accrued_usd=None, ambiguous_reserved_usd=None,
            budget_exposure_usd=None, response_count=None, unknown_cost_count=None,
        )
    tracker = ProviderCostTracker(path)
    tracker.refresh()
    return CostStatus(
        record_state="present", accrued_usd=float(tracker.accrued_usd),
        ambiguous_reserved_usd=float(tracker.reserved_exposure_usd),
        budget_exposure_usd=float(tracker.budget_exposure_usd),
        response_count=tracker.response_count, unknown_cost_count=tracker.unknown_cost_count,
    )


def _study_status(
    root: Path, plan: ProviderExecutionPlan, study_id: str,
) -> StudyStatus:
    run_id = plan.plan_id if study_id == "primary" else f"{plan.plan_id}__{study_id}"
    if Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("Run identity must be a directory name, not a path.")
    directory = root / "artifact_outputs" / "experiments" / run_id
    diagnostic = study_id in {"independent_judge", "judge_ablation", "projection_fidelity"}
    path = directory / ("diagnostic_ledger.json" if diagnostic else "run_ledger.json")
    ceiling = plan.primary_cell_count if study_id == "primary" else next(
        study.task_count_ceiling for study in plan.diagnostic_studies
        if study.study_id == study_id
    )
    result = StudyStatus(
        run_id=run_id, state="no_ledger", expected_task_ceiling=ceiling,
        provider_cost=_cost_status(directory / "provider_attempts.jsonl"),
    )
    if not path.exists():
        return result
    ledger = ProgressLedger.model_validate_json(path.read_text(encoding="utf-8"))
    digest = plan.frozen_inputs["protocol_freeze"].sha256
    if diagnostic:
        if ledger.tasks is None or ledger.unavailable_source_count is None:
            raise ValueError("Diagnostic ledger must declare tasks and unavailable sources.")
        if (ledger.study_id, ledger.protocol_freeze_sha256, ledger.task_count_ceiling) != (
            study_id, digest, ceiling,
        ):
            raise ValueError("Diagnostic ledger does not match the selected plan.")
    elif ledger.cells is None or any(
        (row.run_id, row.protocol_freeze_sha256) != (run_id, digest) for row in ledger.rows
    ):
        raise ValueError("Cell ledger does not match the selected plan.")
    if len(ledger.rows) > ceiling or (study_id == "primary" and len(ledger.rows) != ceiling):
        raise ValueError("Materialized task count differs from the plan's bounds.")
    counts = ledger.status_counts
    state = "failure_recorded" if counts.get("failed", 0) else "recorded"
    if all(row.status in _TERMINAL_OBSERVATIONS for row in ledger.rows):
        state = "terminal_pending_audit"
    return StudyStatus.model_validate({**result.model_dump(),
        "state": state, "materialized_tasks": len(ledger.rows),
        "unavailable_sources": ledger.unavailable_source_count,
        "status_counts": counts,
        "ledger_updated_at_utc": datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(),
    })


def read_execution_status(root: Path, plan_path: Path) -> ExecutionStatus:
    plan = load_provider_plan(plan_path)
    readiness_path = plan_path.parent / "derived" / "provider_readiness.json"
    readiness_state = "no_report"
    if readiness_path.exists():
        load_provider_readiness(
            repo_root=root, plan_path=plan_path, report_path=readiness_path,
        )
        readiness_state = "passed"
    return ExecutionStatus(
        generated_at_utc=datetime.now(UTC).isoformat(), plan_id=plan.plan_id,
        protocol_sha256=plan.frozen_inputs["protocol_freeze"].sha256,
        approval_status=plan.approval.status,
        approved_budget_usd=plan.approval.approved_budget_usd,
        scientific_task_ceiling=plan.scientific_task_ceiling,
        provider_transport_attempt_limit=plan.provider_transport_attempt_limit,
        provider_transport_attempt_ceiling=plan.provider_transport_attempt_ceiling,
        readiness_state=readiness_state,
        readiness_cost=_cost_status(readiness_path.with_name("provider_readiness_attempts.jsonl")),
        studies={study: _study_status(root, plan, study) for study in plan.execution_sequence},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    report = read_execution_status(args.repo_root, args.plan)
    print(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
