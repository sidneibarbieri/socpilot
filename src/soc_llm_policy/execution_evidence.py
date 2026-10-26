"""Bind a terminally accounted provider snapshot to post-execution evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from soc_llm_policy.connectivity_evidence import validate_connectivity_evidence
from soc_llm_policy.corpus_campaign import validate_campaign
from soc_llm_policy.evaluation_plan import RUN_LEDGER_SCHEMA
from soc_llm_policy.frozen_inputs import (
    PROVIDER_INPUT_ROLES,
    resolve_frozen_input_bindings,
)
from soc_llm_policy.paths import RepoPaths
from soc_llm_policy.preflight_evidence import sha256_file
from soc_llm_policy.protocol_freeze import (
    PROTOCOL_FREEZE_SCHEMA,
    compute_incident_hashes,
)

_SCHEMA_VERSION = 1
_STATE = "post_execution_evidence"
_SHA256_LENGTH = 64
_TERMINAL_OBSERVATION_EXCEPTIONS = {
    "refused": "LLMRefusalError",
    "parse_failed": "LLMResponseParseError",
}
_REQUIRED_SNAPSHOT_FILES = {
    "analysis_bundle": "analysis_bundle.json",
    "connectivity_recovery": "connectivity_recovery.jsonl",
    "data_preflight": "data_preflight.json",
    "data_quality": "data_quality_report.json",
    "ledger": "run_ledger.json",
    "manifest": "manifest.json",
    "provider_attempts": "provider_attempts.jsonl",
    "protocol_freeze": "protocol_freeze.json",
    "run_matrix": "run_matrix.json",
    "summary": "summary.json",
}
_PROVIDER_ATTEMPT_OUTCOMES = {
    "connection_error",
    "http_error",
    "response_received",
    "request_started",
    "terminal_model_response",
    "transport_error",
}
_MODEL_OBSERVATION_OUTCOMES = {
    "response_received",
    "terminal_model_response",
}


@dataclass(frozen=True)
class EvidenceFile:
    path: Path
    relative_path: str
    sha256: str


@dataclass(frozen=True)
class ExecutionEvidence:
    manifest_path: Path
    run_id: str
    run_dir: Path
    protocol_freeze: EvidenceFile
    files: dict[str, EvidenceFile]


@dataclass(frozen=True)
class _RunContext:
    run_id: str
    protocol_study: str
    incident_ids: list[str]
    models: list[str]
    arms: list[str]
    repeats: int
    execution_order: str
    matrix_seed: int
    repeat_reduction: str
    approval_policy_mode: str
    scenario_bank_sha256: str | None
    manifest: dict[str, Any]
    freeze_sha256: str


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string.")
    return value


def _string_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ValueError(f"{label} must be a list of non-empty strings.")
    return list(value)


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer.")
    return value


def _sha256(value: Any, label: str) -> str:
    value = _string(value, label)
    if len(value) != _SHA256_LENGTH:
        raise ValueError(f"{label} must be a SHA-256 digest.")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{label} must be a SHA-256 digest.") from exc
    return value


def _inside_repo(path: Path, repo_root: Path, label: str) -> Path:
    resolved_path = path.expanduser().resolve()
    resolved_root = repo_root.expanduser().resolve()
    if resolved_path == resolved_root or resolved_root not in resolved_path.parents:
        raise ValueError(f"{label} must be inside the repository root.")
    return resolved_path


def _repo_file(repo_root: Path, value: Any, label: str) -> Path:
    relative = Path(_string(value, label))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} escapes the repository root.")
    path = _inside_repo(repo_root / relative, repo_root, label)
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    return path


def _repo_directory(repo_root: Path, value: Any, label: str) -> Path:
    relative = Path(_string(value, label))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} escapes the repository root.")
    path = _inside_repo(repo_root / relative, repo_root, label)
    if not path.is_dir():
        raise FileNotFoundError(f"{label} is missing: {path}")
    return path


def _evidence_file(repo_root: Path, path: Path) -> EvidenceFile:
    path = _inside_repo(path, repo_root, "Evidence file")
    if not path.is_file():
        raise FileNotFoundError(f"Evidence file is missing: {path}")
    return EvidenceFile(
        path=path,
        relative_path=path.relative_to(repo_root.expanduser().resolve()).as_posix(),
        sha256=sha256_file(path),
    )


def _validate_protocol_freeze(
    payload: dict[str, Any],
    repo_root: Path,
) -> tuple[list[str], str]:
    if payload.get("schema_version") != PROTOCOL_FREEZE_SCHEMA:
        raise ValueError(
            f"Protocol freeze must use schema version {PROTOCOL_FREEZE_SCHEMA}."
        )
    if payload.get("state") != "pre_execution_protocol_freeze":
        raise ValueError("Protocol freeze has an invalid state.")
    if payload.get("constitutes_model_evidence") is not False:
        raise ValueError("Protocol freeze must not claim model evidence.")
    incident_ids = _string_list(
        payload.get("incident_ids"),
        "Protocol freeze incident_ids",
    )
    if len(set(incident_ids)) != len(incident_ids):
        raise ValueError("Protocol freeze incident_ids must be unique.")
    if payload.get("incident_count") != len(incident_ids):
        raise ValueError("Protocol freeze incident_count does not match incident_ids.")
    expected_cell_count = (
        len(incident_ids)
        * len(_string_list(payload.get("official_models"), "Protocol freeze models"))
        * len(_string_list(payload.get("arms"), "Protocol freeze arms"))
        * _positive_int(payload.get("repeats"), "Protocol freeze repeats")
    )
    if payload.get("primary_cell_count") != expected_cell_count:
        raise ValueError("Protocol freeze primary_cell_count is inconsistent.")
    input_hashes = _mapping(payload.get("input_hashes"), "Protocol freeze input_hashes")
    bindings = resolve_frozen_input_bindings(
        input_hashes,
        repo_root,
        required_roles=(*PROVIDER_INPUT_ROLES, "policy_preflight_manifest"),
    )
    expected_incident_hashes = _mapping(
        payload.get("incident_hashes"), "Protocol freeze incident_hashes"
    )
    incident_paths = RepoPaths(repo_root=repo_root)
    campaign_binding = bindings.get("corpus_campaign_manifest")
    if campaign_binding is not None:
        campaign = validate_campaign(
            repo_root=repo_root,
            manifest_path=campaign_binding.resolved_path,
        )
        incident_paths = RepoPaths(
            repo_root=repo_root,
            inbox_incidents_override=repo_root / campaign.view_path,
        )
    if expected_incident_hashes != compute_incident_hashes(
        incident_paths,
        incident_ids,
    ):
        raise ValueError("Protocol freeze incident content hashes changed.")
    return incident_ids, bindings["constraints"].sha256


def _snapshot_files(repo_root: Path, run_dir: Path) -> dict[str, EvidenceFile]:
    files = {
        key: _evidence_file(repo_root, run_dir / filename)
        for key, filename in _REQUIRED_SNAPSHOT_FILES.items()
    }
    failure_path = run_dir / "experiment_failures.json"
    if failure_path.is_file():
        files["experiment_failures"] = _evidence_file(repo_root, failure_path)
    return files


def _validate_run_manifest(
    *,
    manifest: dict[str, Any],
    freeze: dict[str, Any],
    freeze_sha256: str,
    run_dir: Path,
    incident_ids: list[str],
    repo_root: Path,
) -> _RunContext:
    run_id = _string(manifest.get("run_id"), "Run manifest run_id")
    if run_id != run_dir.name:
        raise ValueError("Run manifest run_id does not match its snapshot directory.")
    if manifest.get("llm_fixture_sha256") is not None:
        raise ValueError("Fixture replay cannot be promoted to model evidence.")

    freeze_binding = _mapping(manifest.get("protocol_freeze"), "Run manifest freeze")
    if freeze_binding.get("sha256") != freeze_sha256:
        raise ValueError("Run manifest does not bind the copied protocol freeze.")
    if freeze_binding.get("snapshot_path") != "protocol_freeze.json":
        raise ValueError("Run manifest has an invalid protocol-freeze snapshot path.")

    protocol_study = str(manifest.get("protocol_study") or "primary")
    if protocol_study == "primary":
        study_incident_ids = incident_ids
        models = _string_list(
            freeze.get("official_models"), "Protocol freeze models"
        )
        arms = _string_list(freeze.get("arms"), "Protocol freeze arms")
        repeats = _positive_int(freeze.get("repeats"), "Protocol freeze repeats")
        execution_order = _string(
            freeze.get("execution_order"), "Protocol freeze execution_order"
        )
        matrix_seed = freeze.get("matrix_seed")
        repeat_reduction = _string(
            freeze.get("repeat_reduction"), "Protocol freeze repeat_reduction"
        )
        scenario_bank_sha256 = None
    elif protocol_study == "adversarial":
        diagnostics = _mapping(
            freeze.get("diagnostic_studies"),
            "Protocol freeze diagnostic studies",
        )
        study = _mapping(
            diagnostics.get("adversarial"),
            "Protocol freeze adversarial study",
        )
        study_incident_ids = _string_list(
            study.get("scenario_ids"), "Adversarial scenario ids"
        )
        models = _string_list(study.get("models"), "Adversarial models")
        arms = _string_list(study.get("arms"), "Adversarial arms")
        repeats = _positive_int(study.get("repeats"), "Adversarial repeats")
        execution_order = _string(
            study.get("execution_order"), "Adversarial execution_order"
        )
        matrix_seed = study.get("matrix_seed")
        repeat_reduction = _string(
            study.get("repeat_reduction"), "Adversarial repeat_reduction"
        )
        input_hashes = _mapping(
            freeze.get("input_hashes"), "Protocol freeze input_hashes"
        )
        scenario_binding = _mapping(
            input_hashes.get("adversarial_scenario_manifest"),
            "Adversarial scenario-manifest binding",
        )
        scenario_bank_sha256 = _sha256(
            scenario_binding.get("sha256"), "Adversarial scenario-manifest hash"
        )
        for role in (
            "adversarial_pair_manifest",
            "adversarial_scenario_manifest",
        ):
            binding = _mapping(input_hashes.get(role), f"Frozen input {role}")
            path = _repo_file(repo_root, binding.get("path"), f"Frozen input {role}")
            if sha256_file(path) != _sha256(binding.get("sha256"), role):
                raise ValueError(f"Frozen adversarial input changed: {role}.")
    else:
        raise ValueError(f"Unsupported protocol study: {protocol_study}.")
    if isinstance(matrix_seed, bool) or not isinstance(matrix_seed, int):
        raise ValueError("Protocol study matrix_seed must be an integer.")
    approval_policy_mode = _string(
        manifest.get("approval_policy_mode"), "Run manifest approval_policy_mode"
    )

    expected_values = {
        "dataset_release_id": freeze.get("dataset_release_id"),
        "eval_protocol_version": freeze.get("eval_protocol_version"),
        "incidents": study_incident_ids,
        "models": models,
        "arms": arms,
        "repeats": repeats,
        "execution_order": execution_order,
        "matrix_seed": matrix_seed,
        "repeat_reduction": repeat_reduction,
        "scenario_bank_sha256": scenario_bank_sha256,
    }
    mismatches = [
        field
        for field, expected in expected_values.items()
        if manifest.get(field) != expected
    ]
    if mismatches:
        raise ValueError(
            "Run manifest differs from its protocol freeze: " + ", ".join(mismatches)
        )
    return _RunContext(
        run_id=run_id,
        protocol_study=protocol_study,
        incident_ids=study_incident_ids,
        models=models,
        arms=arms,
        repeats=repeats,
        execution_order=execution_order,
        matrix_seed=matrix_seed,
        repeat_reduction=repeat_reduction,
        approval_policy_mode=approval_policy_mode,
        scenario_bank_sha256=scenario_bank_sha256,
        manifest=manifest,
        freeze_sha256=freeze_sha256,
    )


def _validate_run_matrix(
    *,
    run_matrix: dict[str, Any],
    freeze: dict[str, Any],
    freeze_sha256: str,
    incident_ids: list[str],
    context: _RunContext,
) -> dict[str, dict[str, Any]]:
    cells = run_matrix.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("Run matrix must contain at least one cell.")
    if run_matrix.get("planned_cell_count") != len(cells):
        raise ValueError("Run matrix planned_cell_count does not match its cells.")
    if run_matrix.get("execution_order") != context.execution_order:
        raise ValueError("Run matrix execution order differs from the freeze.")
    if run_matrix.get("seed") != context.matrix_seed:
        raise ValueError("Run matrix seed differs from the freeze.")
    canonical_cells = json.dumps(
        cells,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    if run_matrix.get("plan_sha256") != hashlib.sha256(canonical_cells).hexdigest():
        raise ValueError("Run matrix plan hash differs from its cells.")
    expected_cell_count = (
        len(incident_ids)
        * len(context.models)
        * len(context.arms)
        * context.repeats
    )
    if context.protocol_study == "primary" and (
        freeze.get("primary_cell_count") != expected_cell_count
    ):
        raise ValueError("Protocol freeze primary_cell_count is inconsistent.")
    if len(cells) != expected_cell_count:
        raise ValueError("Run matrix cell count differs from the frozen design.")
    matrix_cells: dict[str, dict[str, Any]] = {}
    observed_coordinates: set[tuple[str, str, str, int]] = set()
    for cell in cells:
        cell_data = _mapping(cell, "Run matrix cell")
        cell_id = _string(cell_data.get("cell_id"), "Run matrix cell_id")
        if cell_id in matrix_cells:
            raise ValueError("Run matrix contains duplicate cell IDs.")
        matrix_cells[cell_id] = cell_data
        if cell_data.get("protocol_freeze_sha256") != freeze_sha256:
            raise ValueError("Run matrix cell does not bind the protocol freeze.")
        if cell_data.get("scenario_bank_sha256") != context.scenario_bank_sha256:
            raise ValueError("Run matrix cell has the wrong scenario-bank binding.")
        coordinate = (
            _string(cell_data.get("incident_id"), "Run matrix incident_id"),
            _string(cell_data.get("model_name"), "Run matrix model_name"),
            _string(cell_data.get("arm"), "Run matrix arm"),
            _positive_int(cell_data.get("repeat_index"), "Run matrix repeat_index"),
        )
        if coordinate in observed_coordinates:
            raise ValueError("Run matrix contains duplicate scientific coordinates.")
        observed_coordinates.add(coordinate)
        _string(cell_data.get("provider"), "Run matrix provider")
        _string(cell_data.get("deployment"), "Run matrix deployment")

    expected_coordinates = {
        (incident_id, model, arm, repeat_index)
        for incident_id in incident_ids
        for model in context.models
        for arm in context.arms
        for repeat_index in range(1, context.repeats + 1)
    }
    if observed_coordinates != expected_coordinates:
        raise ValueError("Run matrix coordinates differ from the frozen design.")
    return matrix_cells


def _validate_cell_output_evidence(
    *,
    repo_root: Path,
    context: _RunContext,
    matrix_cell: dict[str, Any],
    ledger_cell: dict[str, Any],
) -> None:
    cell_id = _string(matrix_cell.get("cell_id"), "Run matrix cell_id")
    evidence = _mapping(
        ledger_cell.get("output_evidence"),
        f"Run ledger output evidence for {cell_id}",
    )
    returned_model = _string(
        evidence.get("returned_model"), f"Run ledger returned model for {cell_id}"
    )
    if (
        matrix_cell.get("provider") != "azure_openai"
        and returned_model != matrix_cell.get("deployment")
    ):
        raise ValueError("Run ledger output differs from the frozen deployment.")
    file_rows = evidence.get("files")
    if not isinstance(file_rows, list) or not file_rows:
        raise ValueError("Run ledger output evidence has no files.")
    verifier_outputs: list[dict[str, Any]] = []
    observed_paths: set[str] = set()
    for row in file_rows:
        entry = _mapping(row, "Run ledger output evidence file")
        relative_path = _string(entry.get("path"), "Output evidence path")
        if relative_path in observed_paths:
            raise ValueError("Run ledger output evidence contains duplicate paths.")
        observed_paths.add(relative_path)
        path = _repo_file(repo_root, relative_path, "Output evidence file")
        if sha256_file(path) != _sha256(entry.get("sha256"), "Output evidence hash"):
            raise ValueError(f"Run ledger output evidence changed: {relative_path}")
        if path.name.startswith("verifier_output_") and path.suffix == ".json":
            verifier_outputs.append(_read_json(path))
    if len(verifier_outputs) != 1:
        raise ValueError("Each completed cell must bind one verifier output.")
    verifier = verifier_outputs[0]
    if (
        verifier.get("experiment_run_id") != context.run_id
        or verifier.get("experiment_cell_id") != cell_id
        or verifier.get("incident_id") != matrix_cell.get("incident_id")
    ):
        raise ValueError("Verifier output identity differs from the run matrix.")


def _validate_run_ledger(
    *,
    ledger: dict[str, Any],
    matrix_cells: dict[str, dict[str, Any]],
    freeze_sha256: str,
    repo_root: Path,
    context: _RunContext,
) -> tuple[int, dict[str, str], dict[str, str], dict[str, set[str]]]:
    if ledger.get("schema_version") != RUN_LEDGER_SCHEMA:
        raise ValueError("Run ledger has an unsupported schema version.")
    matrix_ids = set(matrix_cells)
    ledger_cells = ledger.get("cells")
    if ledger.get("cell_count") != len(matrix_ids) or not isinstance(
        ledger_cells, list
    ):
        raise ValueError("Run ledger does not match the run matrix cell count.")
    ledger_ids: set[str] = set()
    completed_count = 0
    terminal_observations: dict[str, str] = {}
    final_statuses: dict[str, str] = {}
    observed_status_counts: dict[str, int] = {}
    execution_ids_by_cell: dict[str, set[str]] = {}
    for cell in ledger_cells:
        cell_data = _mapping(cell, "Run ledger cell")
        cell_id = _string(cell_data.get("cell_id"), "Run ledger cell_id")
        if cell_id not in matrix_ids:
            raise ValueError("Run ledger references a cell outside the run matrix.")
        if cell_id in ledger_ids:
            raise ValueError("Run ledger contains duplicate cell IDs.")
        ledger_ids.add(cell_id)
        status = _string(cell_data.get("status"), "Run ledger cell status")
        execution_ids_by_cell[cell_id] = _validate_cell_execution_history(
            cell_data, status=status
        )
        final_statuses[cell_id] = status
        observed_status_counts[status] = observed_status_counts.get(status, 0) + 1
        if status == "completed":
            completed_count += 1
            _validate_cell_output_evidence(
                repo_root=repo_root,
                context=context,
                matrix_cell=matrix_cells[cell_id],
                ledger_cell=cell_data,
            )
        elif status in _TERMINAL_OBSERVATION_EXCEPTIONS:
            expected_exception = _TERMINAL_OBSERVATION_EXCEPTIONS[status]
            if cell_data.get("exception_type") != expected_exception:
                raise ValueError(
                    f"Run ledger {status} cell has the wrong exception type."
                )
            terminal_observations[cell_id] = expected_exception
            if cell_data.get("output_evidence") is not None:
                raise ValueError("Terminal non-plan observation has output evidence.")
        else:
            raise ValueError(
                "Post-execution evidence contains a nonterminal or operationally "
                f"failed cell: {status}."
            )
        if cell_data.get("protocol_freeze_sha256") != freeze_sha256:
            raise ValueError("Run ledger cell does not bind the protocol freeze.")
    if ledger_ids != matrix_ids:
        raise ValueError("Run ledger cell IDs differ from the run matrix.")
    status_counts = _mapping(ledger.get("status_counts"), "Run ledger status_counts")
    for status in (
        "planned",
        "running",
        "completed",
        "refused",
        "parse_failed",
        "failed",
    ):
        if status_counts.get(status, 0) != observed_status_counts.get(status, 0):
            raise ValueError("Run ledger status_counts do not match its cells.")
    if sum(observed_status_counts.values()) != len(matrix_ids):
        raise ValueError("Run ledger status counts do not cover the run matrix.")
    return (
        completed_count,
        terminal_observations,
        final_statuses,
        execution_ids_by_cell,
    )


def _validate_cell_execution_history(
    cell: dict[str, Any], *, status: str
) -> set[str]:
    cell_id = _string(cell.get("cell_id"), "Run ledger cell_id")
    execution_count = cell.get("execution_count")
    executions = cell.get("executions")
    if (
        isinstance(execution_count, bool)
        or not isinstance(execution_count, int)
        or execution_count < 1
        or not isinstance(executions, list)
        or len(executions) != execution_count
    ):
        raise ValueError("Terminal cell has invalid worker-execution accounting.")
    execution_ids: set[str] = set()
    for execution_index, raw_execution in enumerate(executions, start=1):
        execution = _mapping(raw_execution, "Run ledger execution")
        execution_id = execution.get("execution_id")
        if (
            execution_id != f"{cell_id}:{execution_index}"
            or execution.get("execution_index") != execution_index
            or not isinstance(execution.get("worker_id"), str)
            or not isinstance(execution.get("started_at_utc"), str)
            or not isinstance(execution.get("finished_at_utc"), str)
        ):
            raise ValueError("Run ledger worker-execution identity is invalid.")
        execution_ids.add(str(execution_id))
        execution_status = _string(
            execution.get("status"), "Run ledger execution status"
        )
        if execution_index < execution_count and execution_status != "interrupted":
            raise ValueError("Only interrupted worker executions may precede a resume.")
    if _mapping(executions[-1], "Run ledger final execution").get("status") != status:
        raise ValueError("Run ledger cell status differs from its final execution.")
    return execution_ids


def _read_provider_attempt_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.endswith("\n"):
                raise ValueError("Provider-attempt log has an incomplete final record.")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Provider-attempt log has invalid JSON at line {line_number}."
                ) from exc
            records.append(_mapping(record, "Provider-attempt record"))
    if not records:
        raise ValueError("Provider-attempt log is empty.")
    return records


def _provider_attempt_identity(
    record: dict[str, Any],
    matrix_cells: dict[str, dict[str, Any]],
    frozen_attempt_limit: int,
) -> tuple[tuple[str, str, int], str, int]:
    if record.get("schema_version") != 1:
        raise ValueError("Provider-attempt record has an invalid schema.")
    cell_id = _string(record.get("cell_id"), "Provider-attempt cell_id")
    if cell_id not in matrix_cells:
        raise ValueError("Provider-attempt log references an unknown cell.")
    execution_id = _string(
        record.get("execution_id"), "Provider-attempt execution_id"
    )
    provider_attempt_index = _positive_int(
        record.get("provider_attempt_index"), "Provider-attempt index"
    )
    configured_limit = _positive_int(
        record.get("configured_attempt_limit"),
        "Provider-attempt configured limit",
    )
    if configured_limit > frozen_attempt_limit:
        raise ValueError("Provider-attempt limit exceeds the protocol freeze.")
    if provider_attempt_index > configured_limit:
        raise ValueError("Provider-attempt index exceeds its configured limit.")
    provider = _string(record.get("provider"), "Provider-attempt provider")
    if provider != matrix_cells[cell_id].get("provider"):
        raise ValueError("Provider-attempt provider differs from the run matrix.")
    outcome = _string(record.get("outcome"), "Provider-attempt outcome")
    if outcome not in _PROVIDER_ATTEMPT_OUTCOMES:
        raise ValueError("Provider-attempt log contains an unknown outcome.")
    _sha256(record.get("request_sha256"), "Provider-attempt request_sha256")
    _string(record.get("started_at_utc"), "Provider-attempt start time")
    return (cell_id, execution_id, provider_attempt_index), outcome, configured_limit


def _validate_terminal_attempt_record(
    record: dict[str, Any],
    outcome: str,
    matrix_cell: dict[str, Any],
) -> None:
    _string(record.get("finished_at_utc"), "Provider-attempt finish time")
    if not isinstance(record.get("retryable"), bool):
        raise ValueError("Provider-attempt retryable must be boolean.")
    if outcome not in _MODEL_OBSERVATION_OUTCOMES:
        return
    if not isinstance(record.get("usage"), dict):
        raise ValueError("Model observation has no token-usage record.")
    returned_model = _string(
        record.get("returned_model"), "Model observation returned_model"
    )
    if (
        matrix_cell.get("provider") != "azure_openai"
        and returned_model != matrix_cell.get("deployment")
    ):
        raise ValueError("Model observation differs from the frozen deployment.")
    try:
        cost = Decimal(str(record.get("estimated_cost_usd")))
    except InvalidOperation as exc:
        raise ValueError("Model observation has invalid cost accounting.") from exc
    if not cost.is_finite() or cost < 0:
        raise ValueError("Model observation has invalid cost accounting.")


def _validate_attempt_reservation(
    record: dict[str, Any],
    matrix_cell: dict[str, Any],
) -> None:
    _positive_int(
        record.get("prompt_tokens_reserved"),
        "Provider-attempt prompt-token reservation",
    )
    _positive_int(
        record.get("max_completion_tokens_reserved"),
        "Provider-attempt completion-token reservation",
    )
    try:
        reserved_cost = Decimal(str(record.get("reserved_cost_usd")))
    except InvalidOperation as exc:
        raise ValueError("Provider attempt has invalid cost reservation.") from exc
    local_provider = matrix_cell.get("provider") == "ollama"
    if (record.get("billing_mode") == "local") != local_provider:
        raise ValueError("Provider attempt has inconsistent billing semantics.")
    if not reserved_cost.is_finite() or reserved_cost < 0:
        raise ValueError("Provider attempt has invalid cost reservation.")
    if not local_provider and reserved_cost == 0:
        raise ValueError("Provider attempt has invalid cost reservation.")


def _validate_provider_attempt_log(
    *,
    path: Path,
    matrix_cells: dict[str, dict[str, Any]],
    final_statuses: dict[str, str],
    execution_ids_by_cell: dict[str, set[str]],
    freeze: dict[str, Any],
) -> None:
    retry_policy = _mapping(
        freeze.get("retry_policy"), "Protocol freeze retry_policy"
    )
    frozen_attempt_limit = _positive_int(
        retry_policy.get("llm_attempt_limit"),
        "Protocol freeze llm_attempt_limit",
    )
    attempt_states: dict[tuple[str, str, int], str] = {}
    attempt_indexes: dict[tuple[str, str], set[int]] = {}
    operation_limits: dict[tuple[str, str], int] = {}
    cell_request_counts: dict[str, int] = {}
    request_hashes: dict[str, str] = {}
    observations: dict[str, list[str]] = {}

    for record in _read_provider_attempt_records(path):
        identity, outcome, configured_limit = _provider_attempt_identity(
            record, matrix_cells, frozen_attempt_limit
        )
        cell_id, execution_id, provider_attempt_index = identity
        if execution_id not in execution_ids_by_cell[cell_id]:
            raise ValueError(
                "Provider-attempt log references an unknown worker execution."
            )
        request_sha256 = _sha256(
            record.get("request_sha256"), "Provider-attempt request_sha256"
        )
        previous_hash = request_hashes.setdefault(cell_id, request_sha256)
        if previous_hash != request_sha256:
            raise ValueError("Provider retries changed the frozen cell request.")
        if outcome == "request_started":
            _validate_attempt_reservation(record, matrix_cells[cell_id])
            if identity in attempt_states:
                raise ValueError(
                    "Provider-attempt log contains a duplicate request start."
                )
            attempt_states[identity] = "started"
            operation_id = (cell_id, execution_id)
            prior_request_count = cell_request_counts.get(cell_id, 0)
            if operation_id not in operation_limits:
                expected_limit = frozen_attempt_limit - prior_request_count
                if configured_limit != expected_limit:
                    raise ValueError(
                        "Provider operation does not use the remaining frozen "
                        "attempt allowance."
                    )
                operation_limits[operation_id] = configured_limit
            elif operation_limits[operation_id] != configured_limit:
                raise ValueError("Provider operation changed its attempt limit.")
            cell_request_counts[cell_id] = prior_request_count + 1
            if cell_request_counts[cell_id] > frozen_attempt_limit:
                raise ValueError("Provider requests exceed the frozen attempt limit.")
            attempt_indexes.setdefault(operation_id, set()).add(
                provider_attempt_index
            )
            continue
        if attempt_states.get(identity) != "started":
            raise ValueError("Provider-attempt outcome has no matching request start.")
        attempt_states[identity] = outcome
        _validate_terminal_attempt_record(record, outcome, matrix_cells[cell_id])
        if outcome in _MODEL_OBSERVATION_OUTCOMES:
            observations.setdefault(cell_id, []).append(outcome)

    if any(state == "started" for state in attempt_states.values()):
        raise ValueError("Provider-attempt log contains an unresolved request.")
    for indexes in attempt_indexes.values():
        if indexes != set(range(1, max(indexes) + 1)):
            raise ValueError("Provider-attempt indexes are not contiguous.")
    for cell_id, status in final_statuses.items():
        observed = observations.get(cell_id, [])
        valid_observation = (
            observed == ["response_received"]
            if status == "completed"
            else observed == ["terminal_model_response"]
            if status == "refused"
            else observed in (["response_received"], ["terminal_model_response"])
        )
        if not valid_observation:
            raise ValueError(
                "Each terminal cell must have exactly one matching model observation."
            )


def _validate_terminal_observation_log(
    *, run_dir: Path, terminal_observations: dict[str, str]
) -> None:
    path = run_dir / "experiment_failures.json"
    if not terminal_observations:
        if path.exists():
            raise ValueError("Run snapshot contains an unexpected failure log.")
        return
    if not path.is_file():
        raise FileNotFoundError("Terminal observations require a failure log.")
    payload = _read_json(path)
    rows = payload.get("failures")
    if not isinstance(rows, list) or payload.get("failure_count") != len(rows):
        raise ValueError("Experiment failure log has an invalid shape.")
    observed: dict[str, str] = {}
    for row in rows:
        row_data = _mapping(row, "Experiment failure row")
        cell_id = _string(row_data.get("cell_id"), "Experiment failure cell_id")
        exception_type = _string(
            row_data.get("exception_type"), "Experiment failure exception_type"
        )
        if cell_id in observed:
            raise ValueError("Experiment failure log contains duplicate cells.")
        observed[cell_id] = exception_type
    if observed != terminal_observations:
        raise ValueError(
            "Experiment failure log does not match terminal observations."
        )


def _validate_analysis_bundle(
    *,
    bundle: dict[str, Any],
    summary: dict[str, Any],
    context: _RunContext,
    expected_cell_count: int,
) -> None:
    experiment = _mapping(bundle.get("experiment"), "Analysis bundle experiment")
    recommendation = _mapping(
        experiment.get("recommendation_source"),
        "Analysis bundle recommendation source",
    )
    if (
        recommendation.get("kind") != "provider"
        or recommendation.get("constitutes_model_evidence") is not True
    ):
        raise ValueError("Analysis bundle is not provider model evidence.")
    bundle_freeze = _mapping(
        experiment.get("protocol_freeze"),
        "Analysis bundle protocol freeze",
    )
    if bundle_freeze.get("sha256") != context.freeze_sha256:
        raise ValueError("Analysis bundle does not bind the protocol freeze.")
    if experiment.get("run_id") != context.run_id:
        raise ValueError("Analysis bundle run_id differs from the run manifest.")
    if experiment.get("protocol_study") != context.protocol_study:
        raise ValueError("Analysis bundle study differs from the run manifest.")
    if experiment.get("dataset_release_id") != context.manifest.get(
        "dataset_release_id"
    ):
        raise ValueError(
            "Analysis bundle dataset release differs from the run manifest."
        )
    if experiment.get("eval_protocol_version") != context.manifest.get(
        "eval_protocol_version"
    ):
        raise ValueError(
            "Analysis bundle protocol version differs from the run manifest."
        )
    if experiment.get("arms") != context.arms:
        raise ValueError("Analysis bundle arms differ from the protocol freeze.")
    if experiment.get("repeats") != context.repeats:
        raise ValueError("Analysis bundle repeats differ from the protocol freeze.")
    if experiment.get("repeat_reduction") != context.repeat_reduction:
        raise ValueError(
            "Analysis bundle repeat reduction differs from the protocol freeze."
        )
    bundle_matrix = _mapping(
        experiment.get("run_matrix"), "Analysis bundle run matrix"
    )
    frozen_cell_count = (
        len(context.incident_ids)
        * len(context.models)
        * len(context.arms)
        * context.repeats
    )
    if (
        bundle_matrix.get("execution_order") != context.execution_order
        or bundle_matrix.get("seed") != context.matrix_seed
        or bundle_matrix.get("planned_cell_count") != frozen_cell_count
        or bundle_matrix.get("protocol_freeze_sha256") != context.freeze_sha256
    ):
        raise ValueError("Analysis bundle run matrix differs from the frozen design.")
    scenario = experiment.get("scenario_bank")
    if context.scenario_bank_sha256 is None:
        if scenario is not None:
            raise ValueError("Primary analysis bundle unexpectedly binds a scenario bank.")
    else:
        scenario_data = _mapping(scenario, "Analysis bundle scenario bank")
        if scenario_data.get("manifest_sha256") != context.scenario_bank_sha256:
            raise ValueError("Analysis bundle has the wrong scenario-bank binding.")
    summary_incident_count = summary.get("incident_count")
    if not isinstance(summary_incident_count, int) or not (
        0 <= summary_incident_count <= len(context.incident_ids)
    ):
        raise ValueError("Summary incident_count is outside the frozen incident set.")
    bundle_summary = _mapping(bundle.get("summary"), "Analysis bundle summary")
    if bundle_summary != summary:
        raise ValueError("Analysis bundle summary differs from summary.json.")
    rows = bundle.get("incidents")
    if not isinstance(rows, list) or len(rows) != expected_cell_count:
        raise ValueError("Analysis bundle row count differs from the run matrix.")
    if bundle.get("row_count") != len(rows):
        raise ValueError("Analysis bundle row_count differs from its rows.")
    if summary.get("run_count") != expected_cell_count:
        raise ValueError("Summary run_count differs from the run matrix.")


def _validate_snapshot(
    *,
    repo_root: Path,
    run_dir: Path,
) -> tuple[str, dict[str, EvidenceFile]]:
    run_dir = _inside_repo(run_dir, repo_root, "Run snapshot")
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run snapshot is missing: {run_dir}")

    files = _snapshot_files(repo_root, run_dir)
    manifest = _read_json(files["manifest"].path)
    freeze = _read_json(files["protocol_freeze"].path)
    run_matrix = _read_json(files["run_matrix"].path)
    ledger = _read_json(files["ledger"].path)
    bundle = _read_json(files["analysis_bundle"].path)
    summary = _read_json(files["summary"].path)

    freeze_sha256 = files["protocol_freeze"].sha256
    incident_ids, _ = _validate_protocol_freeze(freeze, repo_root)
    context = _validate_run_manifest(
        manifest=manifest,
        freeze=freeze,
        freeze_sha256=freeze_sha256,
        run_dir=run_dir,
        incident_ids=incident_ids,
        repo_root=repo_root,
    )
    matrix_cells = _validate_run_matrix(
        run_matrix=run_matrix,
        freeze=freeze,
        freeze_sha256=freeze_sha256,
        incident_ids=context.incident_ids,
        context=context,
    )
    (
        completed_count,
        terminal_observations,
        final_statuses,
        execution_ids_by_cell,
    ) = _validate_run_ledger(
        ledger=ledger,
        matrix_cells=matrix_cells,
        freeze_sha256=freeze_sha256,
        repo_root=repo_root,
        context=context,
    )
    _validate_provider_attempt_log(
        path=files["provider_attempts"].path,
        matrix_cells=matrix_cells,
        final_statuses=final_statuses,
        execution_ids_by_cell=execution_ids_by_cell,
        freeze=freeze,
    )
    validate_connectivity_evidence(
        log_path=files["connectivity_recovery"].path,
        attempt_path=files["provider_attempts"].path,
        required_successes=freeze["retry_policy"]["connectivity_required_successes"],
    )
    _validate_terminal_observation_log(
        run_dir=run_dir,
        terminal_observations=terminal_observations,
    )
    _validate_analysis_bundle(
        bundle=bundle,
        summary=summary,
        context=context,
        expected_cell_count=completed_count,
    )

    return context.run_id, files


def build_execution_evidence_manifest(
    *,
    repo_root: Path,
    run_id: str,
    output_json: Path,
) -> Path:
    """Write a one-time manifest for a terminally accounted campaign."""
    repo_root = repo_root.expanduser().resolve()
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be a single directory name.")
    run_dir = RepoPaths(repo_root=repo_root).outputs_experiments_dir / run_id
    validated_run_id, files = _validate_snapshot(
        repo_root=repo_root,
        run_dir=run_dir,
    )
    if validated_run_id != run_id:
        raise ValueError("Validated run_id differs from the requested run_id.")

    output_json = _inside_repo(output_json, repo_root, "Evidence manifest")
    if output_json.exists():
        raise FileExistsError(f"Evidence manifest already exists: {output_json}")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "state": _STATE,
        "constitutes_model_evidence": True,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "snapshot": {
            "path": run_dir.relative_to(repo_root).as_posix(),
            "files": {
                key: {
                    "path": evidence_file.relative_path,
                    "sha256": evidence_file.sha256,
                }
                for key, evidence_file in sorted(files.items())
            },
        },
        "protocol_freeze": {
            "path": files["protocol_freeze"].relative_path,
            "sha256": files["protocol_freeze"].sha256,
            "schema_version": PROTOCOL_FREEZE_SCHEMA,
        },
        "checks": {
            "all_cells_terminally_accounted": True,
            "analysis_bundle_is_provider_evidence": True,
            "manifest_matches_protocol_freeze": True,
            "provider_attempts_match_terminal_cells": True,
            "snapshot_hashes_recorded": True,
        },
    }
    output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_json


def load_execution_evidence(
    *,
    repo_root: Path,
    manifest_path: Path,
) -> ExecutionEvidence:
    """Verify an evidence manifest and return its immutable source files."""
    repo_root = repo_root.expanduser().resolve()
    manifest_path = _inside_repo(manifest_path, repo_root, "Evidence manifest")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Evidence manifest is missing: {manifest_path}")
    payload = _read_json(manifest_path)
    if payload.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError("Execution evidence has an unsupported schema version.")
    if payload.get("state") != _STATE:
        raise ValueError("Execution evidence has an invalid state.")
    if payload.get("constitutes_model_evidence") is not True:
        raise ValueError("Execution evidence must explicitly claim model evidence.")
    run_id = _string(payload.get("run_id"), "Execution evidence run_id")
    snapshot = _mapping(payload.get("snapshot"), "Execution evidence snapshot")
    run_dir = _repo_directory(
        repo_root,
        snapshot.get("path"),
        "Execution evidence snapshot path",
    )
    if run_dir.parent.name != "experiments":
        raise ValueError("Execution evidence snapshot path is not an experiment run.")
    if run_dir.name != run_id:
        raise ValueError("Execution evidence run_id does not match the snapshot path.")
    snapshot_files = _mapping(snapshot.get("files"), "Execution evidence files")

    validated_run_id, files = _validate_snapshot(repo_root=repo_root, run_dir=run_dir)
    if validated_run_id != run_id:
        raise ValueError("Execution evidence snapshot run_id changed.")
    if not set(_REQUIRED_SNAPSHOT_FILES).issubset(snapshot_files) or set(
        snapshot_files
    ) != set(files):
        raise ValueError(
            "Execution evidence does not record the required snapshot files."
        )
    for key, evidence_file in files.items():
        entry = _mapping(snapshot_files.get(key), f"Execution evidence file {key}")
        if entry.get("path") != evidence_file.relative_path:
            raise ValueError(f"Execution evidence path mismatch: {key}")
        if entry.get("sha256") != evidence_file.sha256:
            raise ValueError(f"Execution evidence hash mismatch: {key}")

    protocol_freeze = _mapping(
        payload.get("protocol_freeze"), "Execution evidence protocol freeze"
    )
    if (
        protocol_freeze.get("path") != files["protocol_freeze"].relative_path
        or protocol_freeze.get("sha256") != files["protocol_freeze"].sha256
        or protocol_freeze.get("schema_version") != PROTOCOL_FREEZE_SCHEMA
    ):
        raise ValueError("Execution evidence protocol-freeze binding is invalid.")
    checks = _mapping(payload.get("checks"), "Execution evidence checks")
    if not checks or any(value is not True for value in checks.values()):
        raise ValueError("Execution evidence contains failed or missing checks.")
    return ExecutionEvidence(
        manifest_path=manifest_path,
        run_id=run_id,
        run_dir=run_dir,
        protocol_freeze=files["protocol_freeze"],
        files=files,
    )
