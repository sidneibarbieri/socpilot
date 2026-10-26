"""Validate explicit analysis amendments without rewriting failed observations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class LostOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    task_id: str = Field(min_length=1)
    ledger_row_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reason: str = Field(min_length=20)


class OutputLossAmendment(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: int
    study_id: str
    ledger_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_matrix_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider_attempts_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    losses: list[LostOutput] = Field(min_length=1)


def row_sha256(row: dict[str, Any]) -> str:
    encoded = json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_output_loss_amendment(
    run_dir: Path, study_id: str, rows: dict[str, dict[str, Any]]
) -> dict[str, str]:
    path = run_dir / "output_loss_amendment.json"
    if not path.exists():
        return {}
    amendment = OutputLossAmendment.model_validate_json(path.read_text())
    if (
        amendment.schema_version != 1
        or study_id != "projection_fidelity"
        or amendment.study_id != study_id
    ):
        raise ValueError("Output-loss amendments apply only to projection fidelity v1.")
    bindings = {
        "diagnostic_ledger.json": amendment.ledger_sha256,
        "task_matrix.json": amendment.task_matrix_sha256,
        "provider_attempts.jsonl": amendment.provider_attempts_sha256,
    }
    for filename, expected in bindings.items():
        if hashlib.sha256((run_dir / filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Output-loss amendment no longer binds {filename}.")
    reasons = {}
    for loss in amendment.losses:
        row = rows.get(loss.task_id)
        if loss.task_id in reasons or row is None:
            raise ValueError("Output-loss amendment repeats or invents a task.")
        if (
            row.get("status") != "failed"
            or row.get("output") is not None
            or row.get("execution_count") != 1
        ):
            raise ValueError(
                "Output loss must preserve one failed execution without output."
            )
        if row_sha256(row) != loss.ledger_row_sha256:
            raise ValueError(
                "Output-loss amendment does not bind the original failed row."
            )
        reasons[loss.task_id] = loss.reason
    return reasons
