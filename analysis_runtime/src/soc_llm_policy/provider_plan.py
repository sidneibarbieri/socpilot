from __future__ import annotations

import argparse
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from soc_llm_policy.durable_io import atomic_write_json


class FrozenInput(BaseModel):
    path: str = Field(..., min_length=1)
    sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")


class ExecutionApproval(BaseModel):
    status: Literal["draft", "approved"] = "draft"
    approved_by: str = ""
    approved_at_utc: str = ""
    approved_budget_usd: float | None = Field(default=None, gt=0)
    approval_note: str = ""


class ModelCostEstimate(BaseModel):
    study_id: str = Field(..., min_length=1)
    model_name: str = Field(..., min_length=1)
    provider: str = Field(..., min_length=1)
    billing_mode: Literal["metered", "local"] = "metered"
    request_count: int = Field(..., ge=1)
    assumed_prompt_tokens_per_request: int = Field(..., ge=1)
    assumed_completion_tokens_per_request: int = Field(..., ge=1)
    max_completion_tokens_per_request: int = Field(..., ge=1)
    prompt_price_per_1k_usd: float = Field(..., ge=0)
    completion_price_per_1k_usd: float = Field(..., ge=0)
    pricing_source_url: str = Field(..., min_length=1)
    pricing_verified_on: date
    pricing_valid_through: date | None = None
    estimated_cost_usd: float = Field(..., ge=0)
    reserved_cost_usd: float = Field(..., ge=0)

    @model_validator(mode="after")
    def validate_token_bounds(self) -> ModelCostEstimate:
        if (
            self.assumed_completion_tokens_per_request
            > self.max_completion_tokens_per_request
        ):
            raise ValueError(
                "Assumed completion tokens exceed the fixed request limit."
            )
        return self


class CostEstimate(BaseModel):
    basis: str = Field(..., min_length=1)
    models: list[ModelCostEstimate] = Field(..., min_length=1)
    estimated_total_usd: float = Field(..., ge=0)
    approval_budget_with_headroom_usd: float = Field(..., ge=0)
    study_estimated_cost_usd: dict[str, float] = Field(default_factory=dict)
    study_reserved_cost_usd: dict[str, float] = Field(default_factory=dict)
    study_budget_ceiling_usd: dict[str, float] = Field(default_factory=dict)


class DiagnosticStudyPlan(BaseModel):
    study_id: str = Field(..., min_length=1)
    kind: Literal["dependent_diagnostic", "fixed_sample_diagnostic", "fixed_matrix_diagnostic"]
    task_count_ceiling: int = Field(..., ge=1)
    models: list[str] = Field(..., min_length=1)
    depends_on: str = Field(..., min_length=1)
    settings: dict[str, object] = Field(default_factory=dict)
    frozen_inputs: dict[str, FrozenInput] = Field(default_factory=dict)


class ProviderReadinessSpec(BaseModel):
    model_names: list[str] = Field(..., min_length=1)
    request_count: int = Field(..., ge=1)
    provider_attempt_limit: int = Field(..., ge=1)
    provider_attempt_ceiling: int = Field(..., ge=1)
    max_completion_tokens: int = Field(..., ge=1)
    prompt_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_accounting(self) -> ProviderReadinessSpec:
        if len(self.model_names) != len(set(self.model_names)):
            raise ValueError("Provider-readiness model names must be unique.")
        if self.request_count != len(self.model_names):
            raise ValueError("Provider-readiness request count is inconsistent.")
        if self.provider_attempt_ceiling != (
            self.request_count * self.provider_attempt_limit
        ):
            raise ValueError("Provider-readiness attempt ceiling is inconsistent.")
        return self


class ProviderExecutionPlan(BaseModel):
    generated_at_utc: str = Field(..., min_length=1)
    plan_id: str = Field(..., min_length=1)
    purpose: str = Field(..., min_length=1)
    approval: ExecutionApproval
    dataset_release_id: str = Field(..., min_length=1)
    eval_protocol_version: str = Field(..., min_length=1)
    policy_evidence_tier: Literal["operational"] = "operational"
    incident_ids: list[str] = Field(..., min_length=1)
    models: list[str] = Field(..., min_length=1)
    arms: list[str] = Field(..., min_length=1)
    repeats: int = Field(..., ge=1)
    repeat_reduction: Literal["first", "last", "majority", "any"]
    primary_cell_count: int = Field(..., ge=1)
    diagnostic_task_ceiling: int = Field(..., ge=0)
    scientific_task_ceiling: int = Field(..., ge=1)
    provider_transport_attempt_limit: int = Field(..., ge=1)
    provider_transport_attempt_ceiling: int = Field(..., ge=1)
    provider_readiness: ProviderReadinessSpec
    execution_sequence: list[str] = Field(..., min_length=1)
    diagnostic_studies: list[DiagnosticStudyPlan] = Field(default_factory=list)
    cost_estimate: CostEstimate
    frozen_inputs: dict[str, FrozenInput]
    selection_basis: dict[str, object] = Field(default_factory=dict)
    execution_rules: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_campaign_accounting(self) -> ProviderExecutionPlan:
        for label, values in (
            ("incident IDs", self.incident_ids),
            ("models", self.models),
            ("arms", self.arms),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"Provider plan {label} must be unique.")
        expected_primary = (
            len(self.incident_ids)
            * len(self.models)
            * len(self.arms)
            * self.repeats
        )
        if self.primary_cell_count != expected_primary:
            raise ValueError("Primary cell count does not match plan dimensions.")
        study_ids = [study.study_id for study in self.diagnostic_studies]
        if len(study_ids) != len(set(study_ids)):
            raise ValueError("Diagnostic study ids must be unique.")
        diagnostic_count = sum(
            study.task_count_ceiling for study in self.diagnostic_studies
        )
        if self.diagnostic_task_ceiling != diagnostic_count:
            raise ValueError("Diagnostic task ceiling does not match its studies.")
        if self.scientific_task_ceiling != self.primary_cell_count + diagnostic_count:
            raise ValueError("Scientific task ceiling is inconsistent.")
        if self.execution_sequence != ["primary", *study_ids]:
            raise ValueError("Execution sequence must list the primary and studies once.")
        expected_readiness_models = list(
            dict.fromkeys(
                [
                    *self.models,
                    *(
                        model_name
                        for study in self.diagnostic_studies
                        for model_name in study.models
                    ),
                ]
            )
        )
        if self.provider_readiness.model_names != expected_readiness_models:
            raise ValueError(
                "Provider readiness must cover every campaign model exactly once."
            )
        expected_transport_ceiling = (
            self.scientific_task_ceiling
            * self.provider_transport_attempt_limit
            + self.provider_readiness.provider_attempt_ceiling
        )
        if self.provider_transport_attempt_ceiling != expected_transport_ceiling:
            raise ValueError("Provider transport-attempt ceiling is inconsistent.")
        return self


def load_provider_plan(path: Path) -> ProviderExecutionPlan:
    return ProviderExecutionPlan.model_validate_json(path.read_text(encoding="utf-8"))


def approve_provider_plan(
    path: Path,
    *,
    approved_budget_usd: float,
    approved_by: str,
) -> ProviderExecutionPlan:
    plan = load_provider_plan(path)
    if plan.approval.status != "draft":
        raise ValueError("Only a draft provider plan can be approved.")
    required_budget = plan.cost_estimate.approval_budget_with_headroom_usd
    if approved_budget_usd < required_budget:
        raise ValueError(
            f"Approved budget USD {approved_budget_usd:.2f} is below the "
            f"required headroom USD {required_budget:.2f}."
        )
    operator = approved_by.strip()
    if not operator:
        raise ValueError("approved_by must be non-empty.")
    approved_plan = plan.model_copy(
        update={
            "approval": ExecutionApproval(
                status="approved",
                approved_by=operator,
                approved_at_utc=datetime.now(UTC).isoformat(),
                approved_budget_usd=round(approved_budget_usd, 2),
                approval_note="Provider cost ceiling approved before execution.",
            )
        }
    )
    atomic_write_json(path, approved_plan.model_dump(mode="json"))
    return approved_plan


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage a frozen provider plan.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    approve_parser = subparsers.add_parser(
        "approve",
        help="Approve a draft plan with an immutable provider-cost ceiling.",
    )
    approve_parser.add_argument("--plan", type=Path, required=True)
    approve_parser.add_argument("--budget-usd", type=float, required=True)
    approve_parser.add_argument(
        "--approved-by",
        default="artifact_operator",
        help="Role or pseudonymous operator identifier recorded in the plan.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "approve":
        plan = approve_provider_plan(
            args.plan,
            approved_budget_usd=args.budget_usd,
            approved_by=args.approved_by,
        )
        print(
            f"Approved {plan.plan_id} with provider-cost ceiling "
            f"USD {plan.approval.approved_budget_usd:.2f}."
        )


if __name__ == "__main__":
    main()
