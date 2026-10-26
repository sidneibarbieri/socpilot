from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from soc_llm_policy.provider_plan import (
    ModelCostEstimate,
    approve_provider_plan,
    load_provider_plan,
)


def _write_draft_plan(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "generated_at_utc": "2026-07-14T00:00:00+00:00",
                "plan_id": "official_campaign",
                "purpose": "Controlled provider campaign",
                "approval": {"status": "draft"},
                "dataset_release_id": "corpus_400",
                "eval_protocol_version": "official",
                "incident_ids": ["INC_001"],
                "models": ["model_a"],
                "arms": ["llm_zero"],
                "repeats": 1,
                "repeat_reduction": "majority",
                "primary_cell_count": 1,
                "diagnostic_task_ceiling": 0,
                "scientific_task_ceiling": 1,
                "provider_transport_attempt_limit": 3,
                "provider_transport_attempt_ceiling": 6,
                "provider_readiness": {
                    "model_names": ["model_a"],
                    "request_count": 1,
                    "provider_attempt_limit": 3,
                    "provider_attempt_ceiling": 3,
                    "max_completion_tokens": 8,
                    "prompt_sha256": "b" * 64,
                },
                "execution_sequence": ["primary"],
                "diagnostic_studies": [],
                "cost_estimate": {
                    "basis": "test estimate",
                    "models": [
                        {
                            "study_id": "primary",
                            "model_name": "model_a",
                            "provider": "openai",
                            "request_count": 1,
                            "assumed_prompt_tokens_per_request": 100,
                            "assumed_completion_tokens_per_request": 20,
                            "max_completion_tokens_per_request": 4000,
                            "prompt_price_per_1k_usd": 0.01,
                            "completion_price_per_1k_usd": 0.02,
                            "pricing_source_url": "https://provider.example/pricing",
                            "pricing_verified_on": "2026-07-14",
                            "estimated_cost_usd": 0.0014,
                            "reserved_cost_usd": 0.081,
                        }
                    ],
                    "estimated_total_usd": 0.0014,
                    "approval_budget_with_headroom_usd": 0.01,
                    "study_estimated_cost_usd": {"primary": 0.0014},
                    "study_reserved_cost_usd": {"primary": 0.081},
                    "study_budget_ceiling_usd": {"primary": 0.01},
                },
                "frozen_inputs": {
                    "protocol_freeze": {
                        "path": "state/protocol_freeze.json",
                        "sha256": "a" * 64,
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def test_approval_persists_numeric_budget_and_is_single_use(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    _write_draft_plan(plan_path)

    approved = approve_provider_plan(
        plan_path,
        approved_budget_usd=0.02,
        approved_by="artifact_operator",
    )

    assert approved.approval.status == "approved"
    assert approved.approval.approved_budget_usd == 0.02
    assert load_provider_plan(plan_path) == approved
    with pytest.raises(ValueError, match="Only a draft"):
        approve_provider_plan(
            plan_path,
            approved_budget_usd=0.03,
            approved_by="artifact_operator",
        )


def test_approval_rejects_budget_below_required_headroom(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    _write_draft_plan(plan_path)

    with pytest.raises(ValueError, match="below the required headroom"):
        approve_provider_plan(
            plan_path,
            approved_budget_usd=0.009,
            approved_by="artifact_operator",
        )

    assert load_provider_plan(plan_path).approval.status == "draft"


def test_transport_ceiling_accounts_for_scientific_and_readiness_attempts(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.json"
    _write_draft_plan(plan_path)
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    payload["provider_transport_attempt_ceiling"] = 3
    plan_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="transport-attempt ceiling"):
        load_provider_plan(plan_path)


def test_readiness_and_scientific_attempt_limits_are_independent(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.json"
    _write_draft_plan(plan_path)
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    payload["provider_readiness"]["provider_attempt_limit"] = 2
    payload["provider_readiness"]["provider_attempt_ceiling"] = 2
    payload["provider_transport_attempt_ceiling"] = 5
    plan_path.write_text(json.dumps(payload), encoding="utf-8")

    plan = load_provider_plan(plan_path)

    assert plan.provider_transport_attempt_limit == 3
    assert plan.provider_readiness.provider_attempt_limit == 2
    assert plan.provider_transport_attempt_ceiling == 5


def test_cost_estimate_rejects_assumption_above_request_limit() -> None:
    with pytest.raises(ValidationError, match="exceed the fixed request limit"):
        ModelCostEstimate(
            study_id="primary",
            model_name="model_a",
            provider="openai",
            request_count=1,
            assumed_prompt_tokens_per_request=100,
            assumed_completion_tokens_per_request=101,
            max_completion_tokens_per_request=100,
            prompt_price_per_1k_usd=0.01,
            completion_price_per_1k_usd=0.02,
            pricing_source_url="https://provider.example/pricing",
            pricing_verified_on="2026-07-14",
            estimated_cost_usd=0.00302,
            reserved_cost_usd=0.003,
        )
