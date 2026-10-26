"""Validate recovery probes against the provider-attempt ledger."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ConnectivityProbeRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal[1]
    run_id: str | None
    cell_id: str
    execution_id: str
    provider: str
    provider_attempt_index: int = Field(ge=1)
    probe_index: int = Field(ge=1)
    started_at_utc: datetime
    finished_at_utc: datetime
    endpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    outcome: Literal["reachable", "unreachable"]
    exception_type: str | None
    consecutive_successes: int = Field(ge=0)
    required_consecutive_successes: int = Field(ge=1)

    @property
    def attempt_identity(self) -> tuple[str, str, int]:
        return self.cell_id, self.execution_id, self.provider_attempt_index


def _timestamp(value: str) -> datetime:
    stamp = datetime.fromisoformat(value)
    if stamp.utcoffset() is None:
        raise ValueError("Recovery timestamps must include a timezone.")
    return stamp


def _validate_probe(
    record: ConnectivityProbeRecord,
    previous: ConnectivityProbeRecord | None,
    attempt: dict[str, Any],
    required_successes: int,
) -> None:
    if attempt["outcome"] not in {"transport_error", "connection_error"}:
        raise ValueError("Recovery probe has no matching transport failure.")
    if record.provider != attempt["provider"]:
        raise ValueError("Recovery probe changed provider.")
    if record.required_consecutive_successes != required_successes:
        raise ValueError("Recovery probe changed the protocol success threshold.")
    expected_index = previous.probe_index + 1 if previous else 1
    if record.probe_index != expected_index:
        raise ValueError("Recovery probe indexes must be contiguous and unique.")
    prior_successes = previous.consecutive_successes if previous else 0
    expected_successes = prior_successes + 1 if record.outcome == "reachable" else 0
    if record.consecutive_successes != expected_successes:
        raise ValueError("Recovery probe has inconsistent consecutive successes.")
    if record.consecutive_successes > required_successes or (
        previous and prior_successes == required_successes
    ):
        raise ValueError("Recovery probes continued after reaching the threshold.")
    if (record.outcome == "reachable") != (record.exception_type is None):
        raise ValueError("Recovery probe has inconsistent error metadata.")
    if previous and record.endpoint_sha256 != previous.endpoint_sha256:
        raise ValueError("Recovery probes changed endpoint.")
    start = _timestamp(record.started_at_utc.isoformat())
    finish = _timestamp(record.finished_at_utc.isoformat())
    prior_finish = (
        previous.finished_at_utc if previous else _timestamp(attempt["finished_at_utc"])
    )
    if not prior_finish <= start <= finish:
        raise ValueError("Recovery probe timestamps are out of order.")


def validate_connectivity_evidence(
    *, log_path: Path, attempt_path: Path, required_successes: int
) -> None:
    """Check every in-process network retry and every recorded recovery probe."""
    attempts = (
        [
            json.loads(line)
            for line in attempt_path.read_text(encoding="utf-8").splitlines()
        ]
        if attempt_path.exists()
        else []
    )
    starts = {}
    outcomes = {}
    for attempt in attempts:
        identity = (
            attempt["cell_id"],
            attempt["execution_id"],
            attempt["provider_attempt_index"],
        )
        target = starts if attempt["outcome"] == "request_started" else outcomes
        target[identity] = attempt

    last_probes: dict[tuple[str, str, int], ConnectivityProbeRecord] = {}
    for line in log_path.read_text(encoding="utf-8").splitlines():
        record = ConnectivityProbeRecord.model_validate_json(line)
        identity = record.attempt_identity
        if identity not in outcomes:
            raise ValueError("Recovery probe references an unknown provider attempt.")
        _validate_probe(
            record, last_probes.get(identity), outcomes[identity], required_successes
        )
        last_probes[identity] = record

    for identity, attempt in outcomes.items():
        if attempt["outcome"] not in {"transport_error", "connection_error"}:
            continue
        next_start = starts.get((identity[0], identity[1], identity[2] + 1))
        if next_start is None:
            continue
        probe = last_probes.get(identity)
        if probe is None or probe.consecutive_successes != required_successes:
            raise ValueError("Network retry has no completed connectivity recovery.")
        if probe.finished_at_utc > _timestamp(next_start["started_at_utc"]):
            raise ValueError("Network retry started before connectivity recovery.")
