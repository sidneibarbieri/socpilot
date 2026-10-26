from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from test_provider_plan import _write_draft_plan

from scripts.render_execution_dashboard import render_html
from soc_llm_policy.provider_status import ProgressLedger, read_execution_status


def _ledger(status: str = "completed") -> dict[str, object]:
    return {
        "cell_count": 1, "status_counts": {status: 1},
        "cells": [{
            "cell_id": "cell-1", "status": status, "run_id": "official_campaign",
            "protocol_freeze_sha256": "a" * 64,
        }],
    }


def test_missing_records_are_unknown_and_status_is_read_only(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    _write_draft_plan(plan_path)
    before = plan_path.read_bytes()
    report = read_execution_status(tmp_path, plan_path)
    primary = report.studies["primary"]
    assert primary.state == "no_ledger"
    assert primary.status_counts is None
    assert primary.materialized_tasks is None
    assert primary.provider_cost.accrued_usd is None
    assert report.readiness_state == "no_report"
    assert report.approved_budget_usd is None
    assert plan_path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [plan_path]
    page = render_html(report)
    assert "Not recorded" in page
    assert "No report; generation not verified" in page
    assert "No failures recorded" not in page
    assert "run_violation_rate" not in page


@pytest.mark.parametrize("status", ["completed", "refused", "parse_failed", "failed", "running", "planned"])
def test_status_recounts_rows_without_declaring_scientific_success(
    tmp_path: Path, status: str,
) -> None:
    plan_path = tmp_path / "plan.json"
    _write_draft_plan(plan_path)
    ledger_path = tmp_path / "artifact_outputs/experiments/official_campaign/run_ledger.json"
    ledger_path.parent.mkdir(parents=True)
    ledger_path.write_text(json.dumps(_ledger(status)), encoding="utf-8")
    before = ledger_path.read_bytes()
    report = read_execution_status(tmp_path, plan_path)
    primary = report.studies["primary"]
    assert primary.status_counts == {status: 1}
    expected = "recorded"
    if status in {"completed", "refused", "parse_failed"}:
        expected = "terminal_pending_audit"
    elif status == "failed":
        expected = "failure_recorded"
    assert primary.state == expected
    assert ledger_path.read_bytes() == before


@pytest.mark.parametrize("counts", [{}, {"completed": 0}, {"completed": True}, {"completed": "1"}, {"unknown": 1}])
def test_malformed_counts_are_not_coerced_to_success(counts: dict[str, object]) -> None:
    payload = _ledger()
    payload["status_counts"] = counts
    with pytest.raises(ValidationError):
        ProgressLedger.model_validate(payload)


def test_duplicate_rows_fail_instead_of_inflating_progress() -> None:
    payload = _ledger()
    payload.update(cell_count=2, status_counts={"completed": 2})
    payload["cells"] = payload["cells"] * 2
    with pytest.raises(ValidationError, match="duplicate identities"):
        ProgressLedger.model_validate(payload)


def test_empty_diagnostic_records_retain_unavailable_sources() -> None:
    ledger = ProgressLedger.model_validate({
        "tasks": [], "provider_task_count": 0, "status_counts": {},
        "unavailable_source_count": 3,
    })
    assert ledger.rows == []
    assert ledger.unavailable_source_count == 3


def test_foreign_protocol_is_rejected(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    _write_draft_plan(plan_path)
    ledger_path = tmp_path / "artifact_outputs/experiments/official_campaign/run_ledger.json"
    ledger_path.parent.mkdir(parents=True)
    payload = _ledger()
    payload["cells"][0]["protocol_freeze_sha256"] = "b" * 64
    ledger_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="selected plan"):
        read_execution_status(tmp_path, plan_path)


def test_html_escapes_plan_labels_and_keeps_snapshot_caveat(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    _write_draft_plan(plan_path)
    report = read_execution_status(tmp_path, plan_path)
    report.plan_id = '<script>alert("unsafe label")</script>'
    page = render_html(report)
    assert report.plan_id not in page
    assert "&lt;script&gt;" in page
    assert "does not\nprove that its worker is alive" in page
    assert 'aria-label="Study status table"' in page
    assert 'id="study-filter"' in page
