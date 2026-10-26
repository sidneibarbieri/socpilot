from __future__ import annotations

import fcntl
import hashlib
import json
import os
import random
import socket
import time
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from soc_llm_policy.durable_io import atomic_write_json
from soc_llm_policy.scenario_bank import ScenarioRecord

type ExecutionOrder = Literal["sequential", "global_randomized", "paired_blocked"]
type TerminalObservationStatus = Literal["refused", "parse_failed"]

RUN_LEDGER_SCHEMA = 4
_TERMINAL_CELL_STATUSES = frozenset({"completed", "refused", "parse_failed"})

# A lease shorter than this cannot survive normal provider latency, so a worker
# that set one would keep evicting itself between calls.
_MIN_LEASE_SECONDS = 60


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class RunMatrixDesign:
    run_id: str
    dataset_release_id: str
    eval_protocol_version: str
    incidents: tuple[str, ...]
    models: tuple[tuple[str, str, str], ...]
    arms: tuple[str, ...]
    repeats: int
    execution_order: ExecutionOrder
    seed: int
    policy_bundle_sha256: str | None = None
    protocol_freeze_sha256: str | None = None
    scenario_bank_sha256: str | None = None
    incident_partitions: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class EvaluationCell:
    cell_id: str
    order_index: int
    run_id: str
    dataset_release_id: str
    eval_protocol_version: str
    incident_id: str
    model_name: str
    deployment: str
    provider: str
    arm: str
    repeat_index: int
    policy_bundle_sha256: str | None = None
    protocol_freeze_sha256: str | None = None
    scenario_bank_sha256: str | None = None
    scenario_stratum: str | None = None
    corpus_stratum: str | None = None
    base_incident_id: str | None = None
    flipped_field: str | None = None
    flip_from: str | None = None
    flip_to: str | None = None


def _cell_id(identity: dict[str, object]) -> str:
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_run_matrix(
    design: RunMatrixDesign,
    scenario_records: Mapping[str, ScenarioRecord] | None = None,
) -> list[EvaluationCell]:
    if design.repeats < 1:
        raise ValueError("repeats must be >= 1")
    incident_partitions = dict(design.incident_partitions)
    if len(incident_partitions) != len(design.incident_partitions):
        raise ValueError("Incident partition assignments must be unique.")
    if incident_partitions and set(incident_partitions) != set(design.incidents):
        raise ValueError(
            "Incident partition assignments must match the run matrix incidents."
        )

    cell_specs = [
        (incident_id, model_name, deployment, provider, arm, repeat_index)
        for repeat_index in range(1, design.repeats + 1)
        for incident_id in design.incidents
        for model_name, deployment, provider in design.models
        for arm in design.arms
    ]
    rng = random.Random(design.seed)
    if design.execution_order == "global_randomized":
        rng.shuffle(cell_specs)
    elif design.execution_order == "paired_blocked":
        blocks = [
            [
                (incident_id, model_name, deployment, provider, arm, repeat_index)
                for model_name, deployment, provider in design.models
                for arm in design.arms
            ]
            for repeat_index in range(1, design.repeats + 1)
            for incident_id in design.incidents
        ]
        for block in blocks:
            rng.shuffle(block)
        rng.shuffle(blocks)
        cell_specs = [spec for block in blocks for spec in block]
    elif design.execution_order != "sequential":
        raise ValueError(f"Unsupported execution order: {design.execution_order}")

    cells: list[EvaluationCell] = []
    for order_index, (
        incident_id,
        model_name,
        deployment,
        provider,
        arm,
        repeat_index,
    ) in enumerate(cell_specs, start=1):
        scenario = (
            scenario_records.get(incident_id) if scenario_records is not None else None
        )
        corpus_stratum = incident_partitions.get(incident_id)
        if scenario_records is not None and scenario is None:
            raise ValueError(
                f"Missing scenario provenance for incident_id={incident_id!r}"
            )
        scenario_identity = (
            {
                "scenario_bank_sha256": design.scenario_bank_sha256,
                "scenario_stratum": scenario.stratum,
                "base_incident_id": scenario.base_incident_id,
                "flipped_field": scenario.flipped_field,
                "flip_from": scenario.flip_from,
                "flip_to": scenario.flip_to,
            }
            if scenario is not None
            else {}
        )
        cells.append(
            EvaluationCell(
                cell_id=_cell_id(
                    {
                        "arm": arm,
                        "dataset_release_id": design.dataset_release_id,
                        "deployment": deployment,
                        "eval_protocol_version": design.eval_protocol_version,
                        "incident_id": incident_id,
                        "model_name": model_name,
                        "provider": provider,
                        "repeat_index": repeat_index,
                        "run_id": design.run_id,
                        **(
                            {"policy_bundle_sha256": design.policy_bundle_sha256}
                            if design.policy_bundle_sha256 is not None
                            else {}
                        ),
                        **(
                            {"protocol_freeze_sha256": design.protocol_freeze_sha256}
                            if design.protocol_freeze_sha256 is not None
                            else {}
                        ),
                        **scenario_identity,
                        **(
                            {"corpus_stratum": corpus_stratum}
                            if corpus_stratum is not None
                            else {}
                        ),
                    }
                ),
                order_index=order_index,
                run_id=design.run_id,
                dataset_release_id=design.dataset_release_id,
                eval_protocol_version=design.eval_protocol_version,
                incident_id=incident_id,
                model_name=model_name,
                deployment=deployment,
                provider=provider,
                arm=arm,
                repeat_index=repeat_index,
                policy_bundle_sha256=design.policy_bundle_sha256,
                protocol_freeze_sha256=design.protocol_freeze_sha256,
                scenario_bank_sha256=design.scenario_bank_sha256,
                scenario_stratum=scenario.stratum if scenario is not None else None,
                corpus_stratum=corpus_stratum,
                base_incident_id=(
                    scenario.base_incident_id if scenario is not None else None
                ),
                flipped_field=scenario.flipped_field if scenario is not None else None,
                flip_from=scenario.flip_from if scenario is not None else None,
                flip_to=scenario.flip_to if scenario is not None else None,
            )
        )
    return cells


def run_matrix_manifest(
    cells: list[EvaluationCell],
    *,
    execution_order: ExecutionOrder,
    seed: int,
) -> dict[str, object]:
    serialized_cells = [asdict(cell) for cell in cells]
    canonical = json.dumps(
        serialized_cells,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    corpus_stratum_counts: dict[str, int] = {}
    for cell in cells:
        if cell.corpus_stratum is None:
            continue
        corpus_stratum_counts[cell.corpus_stratum] = (
            corpus_stratum_counts.get(cell.corpus_stratum, 0) + 1
        )
    return {
        "schema_version": 1,
        "execution_order": execution_order,
        "seed": seed,
        "planned_cell_count": len(cells),
        "plan_sha256": hashlib.sha256(canonical).hexdigest(),
        "corpus_stratum_cell_counts": dict(sorted(corpus_stratum_counts.items())),
        "cells": serialized_cells,
    }


class RunLease:
    """Single-host ownership, with a kernel lock held across long provider waits."""

    def __init__(
        self,
        path: Path,
        run_id: str,
        *,
        lease_seconds: int,
        allow_stale_takeover: bool,
    ) -> None:
        if lease_seconds < _MIN_LEASE_SECONDS:
            raise ValueError(
                f"Run lease must be at least {_MIN_LEASE_SECONDS} seconds."
            )
        self.path = path
        self.run_id = run_id
        self.lease_seconds = lease_seconds
        self.allow_stale_takeover = allow_stale_takeover
        self.worker_id = uuid.uuid4().hex
        self._lock_descriptor: int | None = None

    def __enter__(self) -> RunLease:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_descriptor = os.open(self.path.parent, os.O_RDONLY)
        acquired = False
        try:
            try:
                fcntl.flock(self._lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise FileExistsError("Run directory is owned by worker holding its lock.") from exc
            self._acquire()
            acquired = True
        finally:
            if not acquired:
                self._unlock()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "run_id": self.run_id,
            "worker_id": self.worker_id,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "updated_at_utc": _utc_now(),
            "lease_expires_at_unix": time.time() + self.lease_seconds,
        }

    def _write_exclusive(self) -> None:
        payload = json.dumps(self._payload(), ensure_ascii=False, indent=2).encode()
        descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _acquire(self) -> None:
        try:
            self._write_exclusive()
            return
        except FileExistsError:
            existing = json.loads(self.path.read_text(encoding="utf-8"))
        if float(existing.get("lease_expires_at_unix", 0.0)) > time.time():
            raise FileExistsError(
                f"Run {self.run_id!r} is owned by worker "
                f"{existing.get('worker_id')!r} until lease expiry."
            )
        if not self.allow_stale_takeover:
            raise FileExistsError(
                f"Run {self.run_id!r} has a stale lease; resume explicitly."
            )
        self._require_dead_local_owner(existing)
        self.path.unlink()
        self._write_exclusive()

    @staticmethod
    def _require_dead_local_owner(existing: dict[str, object]) -> None:
        if existing.get("hostname") != socket.gethostname():
            raise FileExistsError("Cannot automatically resume a lease from another host.")
        owner_pid = existing.get("pid")
        if isinstance(owner_pid, bool) or not isinstance(owner_pid, int) or owner_pid <= 0:
            raise ValueError("Stale lease has no valid owner process identity.")
        try:
            os.kill(owner_pid, 0)
        except ProcessLookupError:
            return
        except PermissionError as exc:
            raise FileExistsError("Cannot verify that the previous lease owner stopped.") from exc
        raise FileExistsError("Previous lease owner is still alive; expiry is not permission to replace it.")

    def _unlock(self) -> None:
        if self._lock_descriptor is not None:
            os.close(self._lock_descriptor)
            self._lock_descriptor = None

    def refresh(self) -> None:
        current = json.loads(self.path.read_text(encoding="utf-8"))
        if current.get("worker_id") != self.worker_id:
            raise RuntimeError("Run lease ownership changed during execution.")
        atomic_write_json(self.path, self._payload())

    def release(self) -> None:
        try:
            if self.path.exists():
                current = json.loads(self.path.read_text(encoding="utf-8"))
                if current.get("worker_id") == self.worker_id:
                    self.path.unlink()
        finally:
            self._unlock()


class RunLedger:
    """The durable record of every cell's state and worker executions.

    Scientific repeats are separate cells. Each cell holds an append-only list
    of worker executions, while provider transport attempts live only in the
    provider-attempt log. A resume marks an unfinished worker execution as
    interrupted and never re-executes a terminal cell.
    """

    def __init__(
        self,
        path: Path,
        cells: list[EvaluationCell],
        *,
        resume: bool = False,
        worker_id: str = "local",
    ) -> None:
        self.path = path
        self.worker_id = worker_id
        fresh_entries: dict[str, dict[str, object]] = {
            cell.cell_id: {
                **asdict(cell),
                "status": "planned",
                "execution_count": 0,
                "executions": [],
                "exception_type": None,
                "error": None,
                "output_evidence": None,
            }
            for cell in cells
        }
        if resume:
            self._entries = self._load(fresh_entries)
            self._mark_interrupted_executions()
        else:
            self._entries = fresh_entries
        self._write()

    def _load(
        self, expected: dict[str, dict[str, object]]
    ) -> dict[str, dict[str, object]]:
        if not self.path.is_file():
            raise FileNotFoundError(f"Run ledger does not exist: {self.path}")
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != RUN_LEDGER_SCHEMA
        ):
            raise ValueError("Run ledger has an unsupported schema version.")
        rows = payload.get("cells")
        if not isinstance(rows, list):
            raise ValueError(f"Run ledger has no cell list: {self.path}")
        observed = {
            str(row.get("cell_id")): row for row in rows if isinstance(row, dict)
        }
        if len(observed) != len(rows):
            raise ValueError("Run ledger contains duplicate or invalid cell entries.")
        if set(observed) != set(expected):
            raise ValueError("Run ledger cells differ from the fixed run matrix.")
        immutable_fields = tuple(EvaluationCell.__dataclass_fields__)
        for cell_id, expected_entry in expected.items():
            entry = observed[cell_id]
            if entry.get("status") not in {
                "planned",
                "running",
                "completed",
                "refused",
                "parse_failed",
                "failed",
            }:
                raise ValueError(f"Run ledger has invalid status for {cell_id}.")
            for field in immutable_fields:
                if entry.get(field) != expected_entry[field]:
                    raise ValueError(
                        f"Run ledger changed immutable field {field!r} for {cell_id}."
                    )
            executions = entry.get("executions")
            execution_count = entry.get("execution_count")
            if (
                not isinstance(executions, list)
                or isinstance(execution_count, bool)
                or not isinstance(execution_count, int)
                or execution_count != len(executions)
            ):
                raise ValueError(f"Run ledger has invalid execution history for {cell_id}.")
            entry.setdefault("output_evidence", None)
        return observed

    def _mark_interrupted_executions(self) -> None:
        for entry in self._entries.values():
            if entry.get("status") != "running":
                continue
            executions = entry.get("executions")
            if isinstance(executions, list) and executions:
                execution = executions[-1]
                if (
                    isinstance(execution, dict)
                    and execution.get("status") == "running"
                ):
                    execution["status"] = "interrupted"
                    execution["finished_at_utc"] = _utc_now()
            entry["status"] = "planned"
            entry["exception_type"] = "InterruptedExecution"
            entry["error"] = (
                "Previous worker stopped before recording a terminal state."
            )

    def mark_running(self, cell: EvaluationCell) -> str:
        entry = self._entries[cell.cell_id]
        if entry["status"] in _TERMINAL_CELL_STATUSES:
            raise ValueError(f"Terminal cell cannot be re-executed: {cell.cell_id}")
        entry["status"] = "running"
        entry["execution_count"] = int(entry["execution_count"]) + 1
        entry["exception_type"] = None
        entry["error"] = None
        execution_id = f"{cell.cell_id}:{entry['execution_count']}"
        executions = entry["executions"]
        if not isinstance(executions, list):
            raise ValueError(f"Invalid execution history for cell {cell.cell_id}")
        executions.append(
            {
                "execution_id": execution_id,
                "execution_index": entry["execution_count"],
                "worker_id": self.worker_id,
                "started_at_utc": _utc_now(),
                "finished_at_utc": None,
                "status": "running",
                "exception_type": None,
                "error": None,
            }
        )
        self._write()
        return execution_id

    def mark_completed(
        self,
        cell: EvaluationCell,
        execution_id: str,
        output_evidence: dict[str, object] | None = None,
    ) -> None:
        entry = self._entries[cell.cell_id]
        self._finish_execution(entry, execution_id, "completed", None)
        entry["status"] = "completed"
        entry["output_evidence"] = output_evidence
        self._write()

    def mark_failed(
        self, cell: EvaluationCell, execution_id: str, exc: Exception
    ) -> None:
        entry = self._entries[cell.cell_id]
        self._finish_execution(entry, execution_id, "failed", exc)
        entry["status"] = "failed"
        entry["exception_type"] = exc.__class__.__name__
        entry["error"] = str(exc).replace("\n", " ").strip()
        self._write()

    def mark_terminal_observation(
        self,
        cell: EvaluationCell,
        execution_id: str,
        status: TerminalObservationStatus,
        exc: Exception,
    ) -> None:
        entry = self._entries[cell.cell_id]
        self._finish_execution(entry, execution_id, status, exc)
        entry["status"] = status
        entry["exception_type"] = exc.__class__.__name__
        entry["error"] = str(exc).replace("\n", " ").strip()
        entry["output_evidence"] = None
        self._write()

    def bind_persisted_completed(
        self, cell: EvaluationCell, output_evidence: dict[str, object]
    ) -> None:
        entry = self._entries[cell.cell_id]
        executions = entry.get("executions")
        if not isinstance(executions, list) or not executions:
            raise ValueError("Persisted output has no worker execution to reconcile.")
        execution = executions[-1]
        if not isinstance(execution, dict) or execution.get("status") != "interrupted":
            raise ValueError("Persisted output does not follow an interrupted execution.")
        execution["status"] = "completed"
        execution["exception_type"] = None
        execution["error"] = None
        execution["reconciled_from_persisted_output"] = True
        entry["status"] = "completed"
        entry["exception_type"] = None
        entry["error"] = None
        entry["output_evidence"] = output_evidence
        self._write()

    def reopen_failed(self, cell: EvaluationCell) -> None:
        entry = self._entries[cell.cell_id]
        if entry.get("status") != "failed":
            raise ValueError(f"Cell is not failed: {cell.cell_id}")
        entry["status"] = "planned"
        self._write()

    @staticmethod
    def _finish_execution(
        entry: dict[str, object],
        execution_id: str,
        status: str,
        exc: Exception | None,
    ) -> None:
        executions = entry.get("executions")
        if not isinstance(executions, list) or not executions:
            raise ValueError("Cell has no active execution to finish.")
        execution = executions[-1]
        if (
            not isinstance(execution, dict)
            or execution.get("execution_id") != execution_id
        ):
            raise ValueError(
                "Execution identity does not match the active cell execution."
            )
        if execution.get("status") != "running":
            raise ValueError("Cell execution is already terminal.")
        execution["status"] = status
        execution["finished_at_utc"] = _utc_now()
        execution["exception_type"] = exc.__class__.__name__ if exc else None
        execution["error"] = (
            str(exc).replace("\n", " ").strip() if exc else None
        )

    def status(self, cell: EvaluationCell) -> str:
        return str(self._entries[cell.cell_id]["status"])

    def exception_type(self, cell: EvaluationCell) -> str | None:
        value = self._entries[cell.cell_id].get("exception_type")
        return str(value) if value else None

    def output_evidence(self, cell: EvaluationCell) -> dict[str, object] | None:
        value = self._entries[cell.cell_id].get("output_evidence")
        return value if isinstance(value, dict) else None

    def error(self, cell: EvaluationCell) -> str | None:
        value = self._entries[cell.cell_id].get("error")
        return str(value) if value else None

    def completed_cells(self, cells: list[EvaluationCell]) -> list[EvaluationCell]:
        return [cell for cell in cells if self.status(cell) == "completed"]

    def terminal_observation_cells(
        self, cells: list[EvaluationCell]
    ) -> list[EvaluationCell]:
        return [
            cell
            for cell in cells
            if self.status(cell) in {"refused", "parse_failed"}
        ]

    def failed_cells(self, cells: list[EvaluationCell]) -> list[EvaluationCell]:
        return [cell for cell in cells if self.status(cell) == "failed"]

    def _write(self) -> None:
        entries = sorted(
            self._entries.values(), key=lambda item: int(item["order_index"])
        )
        counts = {
            status: sum(entry["status"] == status for entry in entries)
            for status in (
                "planned",
                "running",
                "completed",
                "refused",
                "parse_failed",
                "failed",
            )
        }
        atomic_write_json(
            self.path,
            {
                "schema_version": RUN_LEDGER_SCHEMA,
                "cell_count": len(entries),
                "status_counts": counts,
                "cells": entries,
            },
        )
