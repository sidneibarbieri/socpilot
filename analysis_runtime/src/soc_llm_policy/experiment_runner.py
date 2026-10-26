from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import traceback
import urllib.parse
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from soc_llm_policy.analyze import main as analyze_main
from soc_llm_policy.dotenv_utils import load_project_dotenv
from soc_llm_policy.durable_io import atomic_write_json
from soc_llm_policy.evaluation_plan import (
    EvaluationCell,
    ExecutionOrder,
    RunLease,
    RunLedger,
    RunMatrixDesign,
    build_run_matrix,
    run_matrix_manifest,
)
from soc_llm_policy.experiment_family_metrics import (
    persist_experiment_family_metrics,
)
from soc_llm_policy.fixture_llm import FixtureManifest
from soc_llm_policy.frozen_inputs import (
    PROVIDER_INPUT_ROLES,
    require_active_input_paths,
    resolve_frozen_input_bindings,
)
from soc_llm_policy.io import (
    parse_human_actions,
    parse_human_actions_strict_order,
    parse_incident_meta,
    read_json,
    read_jsonl,
)
from soc_llm_policy.llm_adapter import (
    DEFAULT_MAX_COMPLETION_TOKENS,
    LLMRefusalError,
    LLMResponseParseError,
    LLMTransportError,
)
from soc_llm_policy.models_registry import (
    ModelProfile,
    load_model_registry,
    model_environment,
    normalize_provider,
    select_model_profiles,
)
from soc_llm_policy.paths import (
    RepoPaths,
    repo_relative_path,
    resolve_repo_relative_path,
    resolve_repo_root,
)
from soc_llm_policy.pipeline import list_inbox_incidents
from soc_llm_policy.pipeline import main as pipeline_main
from soc_llm_policy.policy_bundle import build_policy_bundle
from soc_llm_policy.proportionality import load_proportionality_policy
from soc_llm_policy.protocol_freeze import (
    PROTOCOL_FREEZE_SCHEMA,
    compute_incident_hashes,
)
from soc_llm_policy.provider_cost import ProviderCostTracker
from soc_llm_policy.scenario_bank import (
    ScenarioBankManifest,
    ScenarioRecord,
    load_manifest,
    materialize_evaluation_dataset,
)

type MainFn = Callable[[list[str] | None], None]
type ProviderConfigCheckFn = Callable[[ModelProfile], tuple[bool, str]]
type ProtocolStudy = Literal["primary", "adversarial"]


@dataclass(frozen=True)
class ProtocolFreezeBinding:
    path: Path
    relative_path: str
    sha256: str
    schema_version: int
    incident_partitions: tuple[tuple[str, str], ...]
    repeat_reduction: str
    provider_attempt_limit: int
    connectivity_probe_timeout_seconds: float
    connectivity_probe_interval_seconds: float
    connectivity_recovery_timeout_seconds: float
    connectivity_required_successes: int
    study_id: ProtocolStudy = "primary"


@dataclass(frozen=True)
class ProtocolFreezeExpectation:
    incident_ids: tuple[str, ...]
    official_models: tuple[str, ...]
    arms: tuple[str, ...]
    repeats: int
    dataset_release_id: str
    eval_protocol_version: str
    execution_order: ExecutionOrder
    matrix_seed: int
    repeat_reduction: str = "last"


@dataclass(frozen=True)
class ExperimentRunConfig:
    paths: RepoPaths
    incidents: list[str]
    model_profiles: list[ModelProfile]
    arms: list[str]
    repeats: int
    run_id: str
    dataset_release_id: str
    min_incidents: int
    clean_target: bool
    fail_fast: bool
    preflight: bool
    preflight_data: bool
    strict_data: bool
    archive_run: bool
    eval_protocol_version: str
    data_preflight_json: Path
    data_quality_json: Path
    summary_json: Path
    analysis_bundle_json: Path
    failures_csv: Path
    failures_json: Path
    export_failures_csv: bool
    approved_budget_usd: float | None = None
    approval_policy_mode: str = "remove"
    execution_order: ExecutionOrder = "paired_blocked"
    matrix_seed: int = 0
    proportionality_policy_path: Path | None = None
    policy_bundle_sha256: str | None = None
    persist_policy_family_metrics: bool = False
    llm_fixture_path: Path | None = None
    llm_fixture_sha256: str | None = None
    scenario_manifest_path: Path | None = None
    scenario_bank_dir: Path | None = None
    scenario_source_incidents_dir: Path | None = None
    scenario_bank_sha256: str | None = None
    protocol_freeze: ProtocolFreezeBinding | None = None
    # Resume an interrupted run in place: reload its ledger, skip the cells it
    # already completed, and re-execute only the pending ones under a lease that
    # keeps a second worker out. The frozen matrix must match the original run.
    resume: bool = False
    lease_seconds: int = 900
    require_output_evidence: bool = False
    repeat_reduction: str = "last"
    protocol_study: ProtocolStudy = "primary"
    provider_attempt_limit: int = 3
    connectivity_probe_timeout_seconds: float = 5.0
    connectivity_probe_interval_seconds: float = 10.0
    connectivity_recovery_timeout_seconds: float = 900.0
    connectivity_required_successes: int = 3


@dataclass(frozen=True)
class ExperimentMetadataInput:
    summary_path: Path
    bundle_path: Path
    data_preflight_path: Path
    data_quality_path: Path
    failures_path: Path
    profiles: list[ModelProfile]
    selected_profiles: list[ModelProfile]
    arms: list[str]
    repeats: int
    repeat_reduction: str
    run_id: str
    dataset_release_id: str
    preflight: bool
    preflight_data: bool
    strict_data: bool
    fail_fast: bool
    archive_run: bool
    eval_protocol_version: str
    git_commit: str
    incident_count: int
    successful_run_count: int
    terminal_observation_count: int
    preflight_failure_count: int
    execution_failure_count: int
    execution_order: ExecutionOrder
    matrix_seed: int
    run_matrix_path: Path
    policy_bundle_sha256: str | None
    policy_family_metrics_path: Path | None
    llm_fixture_sha256: str | None = None
    scenario_bank_sha256: str | None = None
    scenario_counts: dict[str, int] | None = None
    protocol_freeze: ProtocolFreezeBinding | None = None
    approved_budget_usd: float | None = None
    accrued_provider_cost_usd: float | None = None
    protocol_study: ProtocolStudy = "primary"


def _build_experiment_coverage(data: ExperimentMetadataInput) -> dict[str, object]:
    planned_model_count = len(data.profiles)
    selected_model_count = len(data.selected_profiles)
    planned_arm_count = len(data.arms)
    planned_run_count = (
        data.incident_count * selected_model_count * planned_arm_count * data.repeats
    )
    attempted_run_count = (
        data.successful_run_count
        + data.terminal_observation_count
        + data.execution_failure_count
    )
    run_success_rate = (
        round(data.successful_run_count / attempted_run_count, 4)
        if attempted_run_count
        else 0.0
    )
    return {
        "planned_model_count": planned_model_count,
        "selected_model_count": selected_model_count,
        "planned_arm_count": planned_arm_count,
        "planned_repeat_count": data.repeats,
        "repeat_reduction": data.repeat_reduction,
        "preflight_failure_count": data.preflight_failure_count,
        "preflight_data_enabled": data.preflight_data,
        "incident_count_input": data.incident_count,
        "planned_run_count": planned_run_count,
        "attempted_run_count": attempted_run_count,
        "successful_run_count": data.successful_run_count,
        "terminal_observation_count": data.terminal_observation_count,
        "execution_failure_count": data.execution_failure_count,
        "run_success_rate": run_success_rate,
    }


def _read_json_if_exists(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    return payload


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_frozen_input_path(
    *,
    freeze_path: Path,
    repo_root: Path,
    role: str,
    active_path: Path,
) -> None:
    freeze = _read_json_if_exists(freeze_path)
    inputs = freeze.get("input_hashes")
    entry = inputs.get(role) if isinstance(inputs, dict) else None
    if not isinstance(entry, dict):
        raise ValueError(f"Protocol freeze is missing input role {role}.")
    expected = {
        "path": repo_relative_path(active_path, repo_root),
        "sha256": _sha256_file(active_path),
    }
    if entry != expected:
        raise ValueError(f"Active {role} differs from the protocol freeze.")


def _load_protocol_freeze_binding(
    *,
    paths: RepoPaths,
    freeze_path: Path,
    expectation: ProtocolFreezeExpectation,
    study_id: ProtocolStudy = "primary",
) -> ProtocolFreezeBinding:
    payload = _read_json_if_exists(freeze_path)
    if not payload:
        raise FileNotFoundError(f"Protocol freeze manifest not found: {freeze_path}")
    if payload.get("schema_version") != PROTOCOL_FREEZE_SCHEMA:
        raise ValueError(
            "Controlled execution requires protocol-freeze schema version "
            f"{PROTOCOL_FREEZE_SCHEMA}, found {payload.get('schema_version')!r}."
        )
    if payload.get("state") != "pre_execution_protocol_freeze":
        raise ValueError("Protocol freeze is not a pre-execution state.")
    if payload.get("constitutes_model_evidence") is not False:
        raise ValueError("Protocol freeze must not claim model evidence.")

    expected = {
        "dataset_release_id": expectation.dataset_release_id,
        "eval_protocol_version": expectation.eval_protocol_version,
    }
    if study_id == "primary":
        expected.update(
            {
                "incident_ids": list(expectation.incident_ids),
                "official_models": list(expectation.official_models),
                "arms": list(expectation.arms),
                "repeats": expectation.repeats,
                "execution_order": expectation.execution_order,
                "matrix_seed": expectation.matrix_seed,
                "repeat_reduction": expectation.repeat_reduction,
            }
        )
        observed = payload
    else:
        studies = payload.get("diagnostic_studies")
        adversarial = studies.get("adversarial") if isinstance(studies, dict) else None
        if not isinstance(adversarial, dict):
            raise ValueError("Protocol freeze has no adversarial study.")
        expected.update(
            {
                "scenario_ids": list(expectation.incident_ids),
                "models": list(expectation.official_models),
                "arms": list(expectation.arms),
                "repeats": expectation.repeats,
                "execution_order": expectation.execution_order,
                "matrix_seed": expectation.matrix_seed,
                "repeat_reduction": expectation.repeat_reduction,
            }
        )
        observed = {
            "dataset_release_id": payload.get("dataset_release_id"),
            "eval_protocol_version": payload.get("eval_protocol_version"),
            **adversarial,
        }
    mismatches = [
        name
        for name, expected_value in expected.items()
        if observed.get(name) != expected_value
    ]
    if mismatches:
        raise ValueError(
            "Protocol freeze does not match this execution: " + ", ".join(mismatches)
        )

    input_hashes = payload.get("input_hashes")
    if (
        not isinstance(input_hashes, dict)
        or "policy_preflight_manifest" not in input_hashes
    ):
        raise ValueError("Protocol freeze is missing the policy-preflight binding.")
    required_roles = list(PROVIDER_INPUT_ROLES)
    if study_id == "adversarial":
        required_roles.extend(
            ["adversarial_pair_manifest", "adversarial_scenario_manifest"]
        )
    bindings = resolve_frozen_input_bindings(
        input_hashes,
        paths.repo_root,
        required_roles=required_roles,
    )
    require_active_input_paths(
        bindings,
        {
            "action_catalog": paths.inbox_action_catalog_path,
            "constraints": paths.inbox_constraints_path,
            "policy_review_record": paths.policy_review_record_path,
            "models_registry": paths.models_registry_path,
        },
    )

    if study_id == "primary":
        incident_hashes = payload.get("incident_hashes")
        if incident_hashes != compute_incident_hashes(
            paths, list(expectation.incident_ids)
        ):
            raise ValueError("Protocol freeze incident content hash changed.")

        incident_partitions = payload.get("incident_partitions")
        if not isinstance(incident_partitions, dict):
            raise ValueError("Protocol freeze is missing incident partitions.")
        if set(incident_partitions) != set(expectation.incident_ids):
            raise ValueError(
                "Protocol freeze incident partitions do not match the matrix."
            )
        if not all(
            isinstance(incident_id, str)
            and isinstance(stratum, str)
            and stratum.strip()
            for incident_id, stratum in incident_partitions.items()
        ):
            raise ValueError("Protocol freeze contains an invalid incident partition.")
    else:
        incident_partitions = {
            incident_id: "adversarial_diagnostic"
            for incident_id in expectation.incident_ids
        }

    retry_policy = payload.get("retry_policy")
    if not isinstance(retry_policy, dict):
        raise ValueError("Protocol freeze is missing its provider-attempt policy.")
    provider_attempt_limit = retry_policy.get("llm_attempt_limit")
    if (
        isinstance(provider_attempt_limit, bool)
        or not isinstance(provider_attempt_limit, int)
        or provider_attempt_limit < 1
    ):
        raise ValueError("Protocol freeze has an invalid provider-attempt limit.")
    connectivity_probe_timeout_seconds = retry_policy.get(
        "connectivity_probe_timeout_seconds"
    )
    connectivity_probe_interval_seconds = retry_policy.get(
        "connectivity_probe_interval_seconds"
    )
    connectivity_recovery_timeout_seconds = retry_policy.get(
        "connectivity_recovery_timeout_seconds"
    )
    connectivity_required_successes = retry_policy.get(
        "connectivity_required_successes"
    )
    connectivity_times = (
        connectivity_probe_timeout_seconds,
        connectivity_probe_interval_seconds,
        connectivity_recovery_timeout_seconds,
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
        for value in connectivity_times
    ):
        raise ValueError("Protocol freeze has invalid connectivity time settings.")
    if (
        isinstance(connectivity_required_successes, bool)
        or not isinstance(connectivity_required_successes, int)
        or connectivity_required_successes < 1
    ):
        raise ValueError(
            "Protocol freeze has an invalid connectivity success threshold."
        )

    return ProtocolFreezeBinding(
        path=freeze_path,
        relative_path=repo_relative_path(freeze_path, paths.repo_root),
        sha256=_sha256_file(freeze_path),
        schema_version=PROTOCOL_FREEZE_SCHEMA,
        incident_partitions=tuple(sorted(incident_partitions.items())),
        repeat_reduction=expectation.repeat_reduction,
        provider_attempt_limit=provider_attempt_limit,
        connectivity_probe_timeout_seconds=float(
            connectivity_probe_timeout_seconds
        ),
        connectivity_probe_interval_seconds=float(
            connectivity_probe_interval_seconds
        ),
        connectivity_recovery_timeout_seconds=float(
            connectivity_recovery_timeout_seconds
        ),
        connectivity_required_successes=connectivity_required_successes,
        study_id=study_id,
    )


def _validate_protocol_freeze_binding(config: ExperimentRunConfig) -> None:
    binding = config.protocol_freeze
    if binding is None:
        return
    observed = _load_protocol_freeze_binding(
        paths=config.paths,
        freeze_path=binding.path,
        study_id=config.protocol_study,
        expectation=ProtocolFreezeExpectation(
            incident_ids=tuple(config.incidents),
            official_models=tuple(profile.name for profile in config.model_profiles),
            arms=tuple(config.arms),
            repeats=config.repeats,
            dataset_release_id=config.dataset_release_id,
            eval_protocol_version=config.eval_protocol_version,
            execution_order=config.execution_order,
            matrix_seed=config.matrix_seed,
            repeat_reduction=config.repeat_reduction,
        ),
    )
    if observed.sha256 != binding.sha256:
        raise ValueError("Protocol freeze changed after it was bound to the run.")
    if observed.study_id != binding.study_id:
        raise ValueError("Protocol freeze study binding changed after planning.")
    if observed.provider_attempt_limit != config.provider_attempt_limit:
        raise ValueError("Runtime provider-attempt limit differs from the freeze.")
    runtime_connectivity = (
        config.connectivity_probe_timeout_seconds,
        config.connectivity_probe_interval_seconds,
        config.connectivity_recovery_timeout_seconds,
        config.connectivity_required_successes,
    )
    observed_connectivity = (
        observed.connectivity_probe_timeout_seconds,
        observed.connectivity_probe_interval_seconds,
        observed.connectivity_recovery_timeout_seconds,
        observed.connectivity_required_successes,
    )
    if runtime_connectivity != observed_connectivity:
        raise ValueError("Runtime connectivity policy differs from the freeze.")


def _build_failure_summary(path: Path) -> dict[str, object]:
    payload = _read_json_if_exists(path)
    failures = payload.get("failures", [])
    if not isinstance(failures, list):
        failures = []

    by_exception: dict[str, int] = {}
    by_model: dict[str, int] = {}
    preflight_failure_count = 0
    parse_failure_count = 0

    for item in failures:
        if not isinstance(item, dict):
            continue
        exception_type = str(item.get("exception_type", "UnknownError"))
        model = str(item.get("model", "unknown"))
        by_exception[exception_type] = by_exception.get(exception_type, 0) + 1
        by_model[model] = by_model.get(model, 0) + 1
        if exception_type == "PreflightError":
            preflight_failure_count += 1
        if exception_type == "LLMResponseParseError":
            parse_failure_count += 1

    return {
        "failure_count": int(payload.get("failure_count", len(failures))),
        "failure_count_by_exception_type": by_exception,
        "failure_count_by_model": by_model,
        "preflight_failure_count": preflight_failure_count,
        "parse_failure_count": parse_failure_count,
        "failure_log_present": path.exists(),
    }


def _sanitize_run_component(value: str) -> str:
    safe = "".join(ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in value)
    return safe.strip("_") or "unknown"


def _default_run_id() -> str:
    return datetime.now(UTC).strftime("exp_%Y%m%dT%H%M%S%fZ")


def _parse_csv_list(raw: str | None) -> list[str]:
    if raw is None:
        return []
    values: list[str] = []
    for part in raw.split(","):
        item = part.strip()
        if item and item not in values:
            values.append(item)
    return values


def _resolve_incidents(
    paths: RepoPaths,
    incidents_arg: str | None,
    all_incidents: bool,
) -> list[str]:
    if all_incidents:
        return list_inbox_incidents(paths)
    if not incidents_arg:
        raise ValueError("Provide --incidents INC_001,INC_002 or use --all.")
    incidents = _parse_csv_list(incidents_arg)
    if not incidents:
        raise ValueError("Empty incident list after parsing --incidents.")
    return incidents


def _resolve_analysis_output(
    *,
    explicit_value: str | None,
    paths: RepoPaths,
    default_filename: str,
) -> Path:
    if explicit_value is None:
        return paths.outputs_analysis_dir / default_filename
    return resolve_repo_relative_path(explicit_value, paths.repo_root)


def _scenario_records_by_id(
    manifest: ScenarioBankManifest,
) -> dict[str, ScenarioRecord]:
    return {record.scenario_id: record for record in manifest.records}


def _bind_scenario_dataset(
    config: ExperimentRunConfig,
    run_dir: Path,
) -> tuple[ExperimentRunConfig, dict[str, ScenarioRecord]]:
    if config.scenario_manifest_path is None:
        return config, {}
    if config.scenario_bank_dir is None or config.scenario_source_incidents_dir is None:
        raise ValueError(
            "Scenario execution requires scenario_bank_dir and "
            "scenario_source_incidents_dir"
        )

    manifest_payload = config.scenario_manifest_path.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
    if (
        config.scenario_bank_sha256 is not None
        and manifest_sha256 != config.scenario_bank_sha256
    ):
        raise ValueError("Scenario-bank manifest hash changed before execution")
    manifest = load_manifest(config.scenario_manifest_path)
    scenario_records = _scenario_records_by_id(manifest)
    if set(config.incidents) != set(scenario_records):
        raise ValueError("Configured incidents do not match the scenario-bank manifest")

    destination = run_dir / "scenario_input"
    dataset_dir = materialize_evaluation_dataset(
        manifest=manifest,
        source_incidents_dir=config.scenario_source_incidents_dir,
        bank_dir=config.scenario_bank_dir,
        destination=destination,
        reuse_existing=config.resume,
    )
    bound_paths = RepoPaths(
        repo_root=config.paths.repo_root,
        inbox_incidents_override=dataset_dir,
    )
    return (
        replace(
            config,
            paths=bound_paths,
            scenario_bank_sha256=manifest_sha256,
        ),
        scenario_records,
    )


@contextmanager
def _temporary_env(var_name: str, value: str) -> Iterator[None]:
    previous = os.environ.get(var_name)
    os.environ[var_name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(var_name, None)
        else:
            os.environ[var_name] = previous


@contextmanager
def _temporary_env_map(values: dict[str, str]) -> Iterator[None]:
    previous: dict[str, str | None] = {key: os.environ.get(key) for key in values}
    for key, value in values.items():
        os.environ[key] = value
    try:
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def _clean_selected_outputs(paths: RepoPaths, incidents: list[str]) -> None:
    for incident_id in incidents:
        target = paths.outputs_incident_dir(incident_id)
        if target.exists():
            shutil.rmtree(target)


def _validate_incident_inputs(
    paths: RepoPaths,
    incidents: list[str],
) -> dict[str, object]:
    details: list[dict[str, object]] = []
    required_runtime_inputs = (
        paths.inbox_action_catalog_path,
        paths.inbox_constraints_path,
        paths.attack_mapping_path,
        paths.mitre_manifest_path,
    )
    global_missing_files = [
        repo_relative_path(path, paths.repo_root)
        for path in required_runtime_inputs
        if not path.is_file()
    ]
    missing_total = len(global_missing_files)
    parse_error_total = 0
    approval_missing_total = 0
    order_missing_total = 0
    for incident_id in incidents:
        incident_dir = paths.inbox_incident_dir(incident_id)
        missing_files: list[str] = []
        parse_errors: list[str] = []
        approval_missing_count = 0
        order_missing_count = 0
        if not incident_dir.exists():
            missing_files.append(str(incident_dir))
        else:
            for req in ("incident_meta.json", "incident_human_actions.jsonl"):
                candidate = incident_dir / req
                if not candidate.exists():
                    missing_files.append(str(candidate))
            if not missing_files:
                try:
                    parse_incident_meta(read_json(incident_dir / "incident_meta.json"))
                except Exception as exc:
                    parse_errors.append(f"invalid incident_meta: {exc}")
                try:
                    actions_path = incident_dir / "incident_human_actions.jsonl"
                    actions = parse_human_actions(read_jsonl(actions_path, strict=True))
                    approval_missing_count = sum(
                        action.approval is None for action in actions
                    )
                    order_missing_count = sum(
                        action.order is None for action in actions
                    )
                    parse_human_actions_strict_order(
                        read_jsonl(actions_path, strict=True)
                    )
                except Exception as exc:
                    parse_errors.append(f"invalid incident_human_actions: {exc}")
        details.append(
            {
                "incident_id": incident_id,
                "ok": len(missing_files) == 0 and len(parse_errors) == 0,
                "missing_files": missing_files,
                "parse_errors": parse_errors,
                "approval_missing_count": approval_missing_count,
                "order_missing_count": order_missing_count,
            }
        )
        missing_total += len(missing_files)
        parse_error_total += len(parse_errors)
        approval_missing_total += approval_missing_count
        order_missing_total += order_missing_count
    return {
        "incident_count": len(incidents),
        "missing_file_count": missing_total,
        "parse_error_count": parse_error_total,
        "approval_missing_total": approval_missing_total,
        "order_missing_total": order_missing_total,
        "global_missing_files": global_missing_files,
        "incidents": details,
    }


def _write_data_preflight_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_data_quality_report(preflight: dict[str, object]) -> dict[str, object]:
    incident_count = int(preflight.get("incident_count", 0))
    approval_missing_total = int(preflight.get("approval_missing_total", 0))
    order_missing_total = int(preflight.get("order_missing_total", 0))
    parse_error_count = int(preflight.get("parse_error_count", 0))
    return {
        "incident_count": incident_count,
        "parse_error_count": parse_error_count,
        "approval_missing_total": approval_missing_total,
        "order_missing_total": order_missing_total,
        "data_quality_ok": (
            int(preflight.get("missing_file_count", 0)) == 0 and parse_error_count == 0
        ),
    }


def _write_data_quality_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def _resolve_git_commit(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _load_dotenv_if_available() -> None:
    load_project_dotenv()


def _normalize_provider(provider: str) -> str:
    return normalize_provider(provider)


def _build_llm_env_values(profile: ModelProfile) -> dict[str, str]:
    return model_environment(profile)


def _valid_http_endpoint(value: str) -> bool:
    parsed = urllib.parse.urlsplit(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _check_provider_configuration(profile: ModelProfile) -> tuple[bool, str]:
    """Validate provider configuration without making an external request."""
    provider = _normalize_provider(profile.provider)
    if not profile.deployment.strip():
        return False, "model or deployment identifier is empty"
    if provider == "fixture":
        return True, "local_configuration_ok"

    if provider == "azure_openai":
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()
        api_key = os.environ.get("AZURE_OPENAI_API_KEY", "").strip()
        if not endpoint or not api_key:
            return False, "AZURE_OPENAI_ENDPOINT/API_KEY missing"
    elif provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        endpoint = os.environ.get(
            "OPENAI_API_BASE", "https://api.openai.com/v1"
        ).strip()
        if not api_key:
            return False, "OPENAI_API_KEY missing"
    elif provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        endpoint = os.environ.get(
            "ANTHROPIC_API_BASE",
            "https://api.anthropic.com/v1",
        ).strip()
        if not api_key:
            return False, "ANTHROPIC_API_KEY missing"
    elif provider == "ollama":
        endpoint = os.environ.get(
            "OLLAMA_API_BASE", "http://127.0.0.1:11434"
        ).strip()
    else:
        return False, f"unsupported provider: {provider}"

    if not _valid_http_endpoint(endpoint):
        return False, f"invalid {provider} endpoint URL"
    return True, "local_configuration_ok"


def _cell_output_evidence(
    paths: RepoPaths, cell: EvaluationCell
) -> dict[str, object] | None:
    """Locate and hash the files that constitute one plan observation."""
    matches: list[tuple[Path, dict[str, object]]] = []
    for path in sorted(
        paths.outputs_incident_dir(cell.incident_id).glob("verifier_output_*.json")
    ):
        payload = _read_json_if_exists(path)
        if (
            payload.get("experiment_run_id") == cell.run_id
            and payload.get("experiment_cell_id") == cell.cell_id
        ):
            matches.append((path, payload))
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError(
            f"Cell {cell.cell_id} has {len(matches)} persisted verifier outputs."
        )

    verifier_path, verifier = matches[0]
    run_tag = str(verifier.get("run_tag") or "")
    if not run_tag:
        raise ValueError(f"Cell {cell.cell_id} verifier output has no run_tag.")
    files = [
        verifier_path,
        paths.outputs_llm_actions_versioned_path(cell.incident_id, run_tag),
        paths.outputs_llm_input_snapshot_versioned_path(cell.incident_id, run_tag),
    ]
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Cell {cell.cell_id} is missing persisted evidence: {missing}"
        )
    raw_actions = _read_json_if_exists(files[1])
    return {
        "run_tag": run_tag,
        "returned_model": raw_actions.get("returned_model"),
        "files": [
            {
                "path": repo_relative_path(path, paths.repo_root),
                "sha256": _sha256_file(path),
            }
            for path in files
        ],
    }


def _validate_output_evidence(
    paths: RepoPaths, cell: EvaluationCell, evidence: dict[str, object]
) -> None:
    files = evidence.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError(f"Cell {cell.cell_id} has no persisted output file list.")
    for item in files:
        if not isinstance(item, dict):
            raise ValueError(f"Cell {cell.cell_id} has invalid output evidence.")
        relative_path = item.get("path")
        expected_hash = item.get("sha256")
        if not isinstance(relative_path, str) or not isinstance(expected_hash, str):
            raise ValueError(f"Cell {cell.cell_id} has incomplete output evidence.")
        path = resolve_repo_relative_path(relative_path, paths.repo_root)
        if not path.is_file() or _sha256_file(path) != expected_hash:
            raise ValueError(
                f"Cell {cell.cell_id} output evidence changed: {relative_path}"
            )
    observed = _cell_output_evidence(paths, cell)
    if observed != evidence:
        raise ValueError(
            f"Cell {cell.cell_id} output evidence no longer identifies its observation."
        )


def _validate_returned_model(cell: EvaluationCell, evidence: dict[str, object]) -> None:
    # An Azure deployment name is an opaque alias for the model behind it, so the
    # returned model id legitimately differs from the requested deployment; the
    # id is still recorded for provenance. A direct provider echoes the requested
    # model id, where a mismatch is the silent fallback this check must catch.
    if cell.provider == "azure_openai":
        return
    returned_model = evidence.get("returned_model")
    if returned_model != cell.deployment:
        raise RuntimeError(
            f"Cell {cell.cell_id} requested {cell.deployment!r} but the provider "
            f"returned {returned_model!r}."
        )


def _provider_attempt_state_by_cell(
    attempt_log: Path,
) -> dict[str, dict[str, bool | int]]:
    if not attempt_log.is_file():
        return {}
    started_attempts: dict[tuple[str, str, int], str] = {}
    finished_attempts: set[tuple[str, str, int]] = set()
    response_cells: set[str] = set()
    request_counts: dict[str, int] = {}
    for line_number, line in enumerate(
        attempt_log.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid provider-attempt record at {attempt_log}:{line_number}"
            ) from error
        if not isinstance(record, dict):
            continue
        cell_id = record.get("cell_id")
        execution_id = record.get("execution_id")
        provider_attempt_index = record.get("provider_attempt_index")
        if (
            not isinstance(cell_id, str)
            or not isinstance(execution_id, str)
            or not isinstance(provider_attempt_index, int)
        ):
            raise ValueError(
                f"Invalid provider-attempt identity at {attempt_log}:{line_number}"
            )
        identity = (cell_id, execution_id, provider_attempt_index)
        outcome = record.get("outcome")
        if outcome == "request_started":
            if identity in started_attempts:
                raise ValueError(
                    f"Duplicate provider-attempt start at {attempt_log}:{line_number}"
                )
            started_attempts[identity] = cell_id
            request_counts[cell_id] = request_counts.get(cell_id, 0) + 1
        else:
            if identity not in started_attempts:
                raise ValueError(
                    f"Provider-attempt outcome has no start at "
                    f"{attempt_log}:{line_number}"
                )
            if identity in finished_attempts:
                raise ValueError(
                    f"Duplicate provider-attempt outcome at "
                    f"{attempt_log}:{line_number}"
                )
            finished_attempts.add(identity)
        if outcome in {
            "response_received",
            "terminal_model_response",
        }:
            response_cells.add(cell_id)
    unresolved_cells = {
        cell_id
        for identity, cell_id in started_attempts.items()
        if identity not in finished_attempts
    }
    return {
        cell_id: {
            "response_received": cell_id in response_cells,
            "unresolved_request": cell_id in unresolved_cells,
            "request_count": request_counts.get(cell_id, 0),
        }
        for cell_id in response_cells | unresolved_cells | set(request_counts)
    }


def _reconcile_resume_outputs(
    config: ExperimentRunConfig,
    ledger: RunLedger,
    cells: list[EvaluationCell],
) -> None:
    """Bind resumable cells to persisted evidence or reject ambiguity."""
    attempt_log = (
        config.paths.outputs_experiment_dir(config.run_id) / "provider_attempts.jsonl"
    )
    attempt_states = _provider_attempt_state_by_cell(attempt_log)
    attempt_limit = config.provider_attempt_limit
    for cell in cells:
        status = ledger.status(cell)
        recorded = ledger.output_evidence(cell)
        if status == "completed" and recorded is not None:
            _validate_output_evidence(config.paths, cell, recorded)
            _validate_returned_model(cell, recorded)
            continue
        if status in {"refused", "parse_failed"}:
            continue
        observed = _cell_output_evidence(config.paths, cell)
        if observed is not None:
            _validate_returned_model(cell, observed)
            ledger.bind_persisted_completed(cell, observed)
            continue
        attempt_state = attempt_states.get(cell.cell_id, {})
        if attempt_state.get("unresolved_request"):
            raise RuntimeError(
                f"Cell {cell.cell_id} has a provider request without a recorded "
                "outcome. Its observation status is ambiguous, so this run "
                "cannot resume automatically."
            )
        if attempt_state.get("response_received"):
            raise RuntimeError(
                f"Cell {cell.cell_id} received a provider response but has no "
                "complete persisted observation. Automatic retry would create "
                "an ambiguous duplicate, so this run cannot resume."
            )
        if int(attempt_state.get("request_count", 0)) >= attempt_limit:
            raise RuntimeError(
                f"Cell {cell.cell_id} consumed its frozen provider-attempt limit."
            )
        if status == "failed":
            if ledger.exception_type(cell) != "LLMTransportError":
                raise RuntimeError(
                    f"Cell {cell.cell_id} failed with "
                    f"{ledger.exception_type(cell) or 'an unknown error'}; only a "
                    "transport failure without a provider response can resume."
                )
            ledger.reopen_failed(cell)


def _cell_is_executable(ledger: RunLedger, cell: EvaluationCell) -> bool:
    return ledger.status(cell) == "planned"


def _execute_run(
    config: ExperimentRunConfig,
    cells: list[EvaluationCell],
    profiles_by_identity: dict[tuple[str, str, str], ModelProfile],
    ledger: RunLedger,
    run_pipeline: MainFn,
    lease: RunLease,
    run_dir: Path,
) -> tuple[int, int, list[dict[str, str]], list[EvaluationCell]]:
    failures: list[dict[str, str]] = []
    completed_cells: list[EvaluationCell] = list(ledger.completed_cells(cells))
    success_count = len(completed_cells)
    execution_failure_count = 0
    cost_tracker = ProviderCostTracker(run_dir / "provider_attempts.jsonl")
    connectivity_log = run_dir / "connectivity_recovery.jsonl"
    connectivity_log.touch(mode=0o600, exist_ok=True)
    attempt_states = _provider_attempt_state_by_cell(
        run_dir / "provider_attempts.jsonl"
    )

    for cell in cells:
        if not _cell_is_executable(ledger, cell):
            continue
        lease.refresh()
        profile = profiles_by_identity[
            (cell.model_name, cell.deployment, cell.provider)
        ]
        provider = _normalize_provider(profile.provider)
        remaining_attempts: int | None = None
        if provider != "fixture":
            consumed_attempts = int(
                attempt_states.get(cell.cell_id, {}).get("request_count", 0)
            )
            remaining_attempts = config.provider_attempt_limit - consumed_attempts
            if remaining_attempts < 1:
                raise RuntimeError(
                    f"Cell {cell.cell_id} consumed its frozen provider-attempt limit."
                )
        if config.approved_budget_usd is not None and provider != "fixture":
            cost_tracker.require_budget_for_next_request(
                profile,
                config.approved_budget_usd,
                max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
            )
        execution_id = ledger.mark_running(cell)
        env_values = {
            **_build_llm_env_values(profile),
            "SOC_EXPERIMENT_RUN_ID": cell.run_id,
            "SOC_EXPERIMENT_CELL_ID": cell.cell_id,
            "SOC_EXPERIMENT_REPEAT_INDEX": str(cell.repeat_index),
            "SOC_EXPERIMENT_ORDER_INDEX": str(cell.order_index),
            "SOC_EXPERIMENT_EXECUTION_ID": execution_id,
            "SOC_EXPERIMENT_ATTEMPT_LOG": str(run_dir / "provider_attempts.jsonl"),
            "SOC_EXPERIMENT_CONNECTIVITY_LOG": str(connectivity_log),
            "SOC_LLM_CONNECTIVITY_RECOVERY_ENABLED": "1",
            "SOC_LLM_CONNECTIVITY_PROBE_TIMEOUT_SECONDS": str(
                config.connectivity_probe_timeout_seconds
            ),
            "SOC_LLM_CONNECTIVITY_PROBE_INTERVAL_SECONDS": str(
                config.connectivity_probe_interval_seconds
            ),
            "SOC_LLM_CONNECTIVITY_RECOVERY_TIMEOUT_SECONDS": str(
                config.connectivity_recovery_timeout_seconds
            ),
            "SOC_LLM_CONNECTIVITY_REQUIRED_SUCCESSES": str(
                config.connectivity_required_successes
            ),
            **(
                {
                    "SOC_EXPERIMENT_APPROVED_BUDGET_USD": str(
                        config.approved_budget_usd
                    )
                }
                if config.approved_budget_usd is not None
                else {}
            ),
            **(
                {"SOC_LLM_ATTEMPT_LIMIT": str(remaining_attempts)}
                if remaining_attempts is not None
                else {}
            ),
            "SOC_LLM_POLICY_INBOX_INCIDENTS_DIR": str(config.paths.inbox_incidents_dir),
            **(
                {"SOC_EXPERIMENT_CORPUS_STRATUM": cell.corpus_stratum}
                if cell.corpus_stratum is not None
                else {}
            ),
        }
        print(
            "\n=== Cell "
            f"{cell.order_index}/{len(cells)}: model={cell.model_name} "
            f"provider={provider} arm={cell.arm} repeat={cell.repeat_index} "
            f"incident={cell.incident_id} ==="
        )
        try:
            with _temporary_env_map(env_values):
                run_pipeline(
                    [
                        "--repo-root",
                        str(config.paths.repo_root),
                        "--incident",
                        cell.incident_id,
                        "--llm-mode",
                        "--llm-arm",
                        cell.arm,
                        "--approval-policy-mode",
                        config.approval_policy_mode,
                        *(
                            [
                                "--proportionality-policy",
                                str(config.proportionality_policy_path),
                            ]
                            if config.proportionality_policy_path is not None
                            else []
                        ),
                        *(
                            ["--llm-fixture", str(config.llm_fixture_path)]
                            if config.llm_fixture_path is not None
                            else []
                        ),
                        *(["--strict-data"] if config.strict_data else []),
                    ]
                )
            output_evidence = _cell_output_evidence(config.paths, cell)
            if output_evidence is None and config.require_output_evidence:
                raise RuntimeError(
                    f"Cell {cell.cell_id} completed without persisted verifier evidence."
                )
            if output_evidence is not None and config.require_output_evidence:
                _validate_returned_model(cell, output_evidence)
            ledger.mark_completed(cell, execution_id, output_evidence)
            success_count += 1
            completed_cells.append(cell)
        except Exception as exc:
            terminal_observation = isinstance(
                exc, (LLMRefusalError, LLMResponseParseError)
            )
            recoverable_transport_failure = isinstance(exc, LLMTransportError)
            if isinstance(exc, LLMRefusalError):
                ledger.mark_terminal_observation(cell, execution_id, "refused", exc)
            elif isinstance(exc, LLMResponseParseError):
                ledger.mark_terminal_observation(
                    cell, execution_id, "parse_failed", exc
                )
            else:
                ledger.mark_failed(cell, execution_id, exc)
            message = str(exc).replace("\n", " ").strip()
            failures.append(
                {
                    "cell_id": cell.cell_id,
                    "order_index": str(cell.order_index),
                    "model": cell.model_name,
                    "deployment": cell.deployment,
                    "arm": cell.arm,
                    "repeat_index": str(cell.repeat_index),
                    "run_id": config.run_id,
                    "incident_id": cell.incident_id,
                    "error": message,
                    "exception_type": exc.__class__.__name__,
                    "traceback": traceback.format_exc(),
                }
            )
            if not terminal_observation:
                execution_failure_count += 1
            observation_label = (
                "Terminal observation in " if terminal_observation else "Failure in "
            )
            print(
                observation_label
                + f"cell={cell.cell_id} model={cell.model_name} arm={cell.arm} "
                f"repeat={cell.repeat_index} incident={cell.incident_id}: {message}"
            )
            if not terminal_observation and (
                config.fail_fast or not recoverable_transport_failure
            ):
                raise
    return (
        success_count,
        execution_failure_count,
        failures,
        completed_cells,
    )


def _run_data_preflight(config: ExperimentRunConfig) -> None:
    data_preflight = _validate_incident_inputs(config.paths, config.incidents)
    _write_data_preflight_report(config.data_preflight_json, data_preflight)
    data_quality = _build_data_quality_report(data_preflight)
    _write_data_quality_report(config.data_quality_json, data_quality)

    missing_total = int(data_preflight.get("missing_file_count", 0))
    parse_error_total = int(data_preflight.get("parse_error_count", 0))
    if missing_total > 0 or parse_error_total > 0:
        raise FileNotFoundError(
            f"Data preflight failed. See: {config.data_preflight_json}"
        )
    print(f"Data preflight OK: {config.data_preflight_json}")


def _require_new_run(run_dir: Path, run_id: str) -> None:
    existing_run_files = sorted(run_dir.iterdir()) if run_dir.exists() else []
    if not existing_run_files:
        return
    existing = ", ".join(str(path) for path in existing_run_files)
    raise FileExistsError(
        f"Evaluation run_id {run_id!r} already exists: {existing}. "
        "Use a new run_id, or pass --resume to continue this run."
    )


def _require_resumable_run(run_dir: Path, run_id: str) -> None:
    ledger_path = run_dir / "run_ledger.json"
    if not ledger_path.is_file():
        raise FileNotFoundError(
            f"Cannot resume {run_id!r}: no run ledger at {ledger_path}. "
            "A resume continues an interrupted run; start a new run instead."
        )


def _select_models_for_execution(
    config: ExperimentRunConfig,
    provider_config_check: ProviderConfigCheckFn,
) -> tuple[list[ModelProfile], list[dict[str, str]], int]:
    models_to_run: list[ModelProfile] = []
    failure_rows: list[dict[str, str]] = []
    preflight_failure_count = 0

    for profile in config.model_profiles:
        deployment = profile.deployment
        provider = _normalize_provider(profile.provider)
        env_values = _build_llm_env_values(profile)
        if provider == "fixture":
            models_to_run.append(profile)
            continue
        if not config.preflight:
            models_to_run.append(profile)
            continue

        with _temporary_env_map(env_values):
            ok, reason = provider_config_check(profile)

        if ok:
            print(
                "Provider configuration OK for model="
                f"{profile.name} (provider={provider}, "
                f"deployment={deployment}, {reason})"
            )
            models_to_run.append(profile)
            continue

        message = f"provider_configuration_failed: {reason}"
        preflight_failure_count += 1
        failure_rows.append(
            {
                "model": profile.name,
                "deployment": deployment,
                "arm": "__all__",
                "repeat_index": "0",
                "run_id": config.run_id,
                "incident_id": "__preflight__",
                "error": message,
                "exception_type": "ProviderConfigurationError",
                "traceback": "",
            }
        )
        print(
            "Provider configuration failed for model="
            f"{profile.name} (provider={provider}, deployment={deployment}): {reason}"
        )

    return models_to_run, failure_rows, preflight_failure_count


def _prepare_run_ledger(
    config: ExperimentRunConfig,
    lease: RunLease,
    cells: list[EvaluationCell],
    run_matrix_path: Path,
    ledger_path: Path,
) -> RunLedger:
    matrix_manifest = run_matrix_manifest(
        cells,
        execution_order=config.execution_order,
        seed=config.matrix_seed,
    )
    if config.resume:
        if _read_json_if_exists(run_matrix_path) != matrix_manifest:
            raise ValueError(
                "Resumed run matrix differs from the immutable original matrix."
            )
    else:
        atomic_write_json(run_matrix_path, matrix_manifest)
    ledger = RunLedger(
        ledger_path,
        cells,
        resume=config.resume,
        worker_id=lease.worker_id,
    )
    if config.resume:
        _reconcile_resume_outputs(config, ledger, cells)
    return ledger


def _ledger_failure_rows(
    config: ExperimentRunConfig,
    ledger: RunLedger,
    cells: list[EvaluationCell],
    new_failures: list[dict[str, str]],
) -> list[dict[str, str]]:
    rows = {row["cell_id"]: row for row in new_failures}
    for cell in ledger.failed_cells(cells):
        rows.setdefault(
            cell.cell_id,
            {
                "cell_id": cell.cell_id,
                "order_index": str(cell.order_index),
                "model": cell.model_name,
                "deployment": cell.deployment,
                "arm": cell.arm,
                "repeat_index": str(cell.repeat_index),
                "run_id": config.run_id,
                "incident_id": cell.incident_id,
                "error": ledger.error(cell) or "",
                "exception_type": ledger.exception_type(cell) or "UnknownError",
                "traceback": "",
            },
        )
    for cell in ledger.terminal_observation_cells(cells):
        rows.setdefault(
            cell.cell_id,
            {
                "cell_id": cell.cell_id,
                "order_index": str(cell.order_index),
                "model": cell.model_name,
                "deployment": cell.deployment,
                "arm": cell.arm,
                "repeat_index": str(cell.repeat_index),
                "run_id": config.run_id,
                "incident_id": cell.incident_id,
                "error": ledger.error(cell) or "",
                "exception_type": ledger.exception_type(cell) or "UnknownError",
                "traceback": "",
            },
        )
    return list(rows.values())


def _require_provider_budget(config: ExperimentRunConfig) -> None:
    if config.protocol_freeze is None or config.llm_fixture_path is not None:
        return
    if config.approved_budget_usd is None:
        raise ValueError(
            "Frozen provider execution requires an approved provider-cost budget."
        )


def run_experiments(
    config: ExperimentRunConfig,
    run_pipeline: MainFn = pipeline_main,
    run_analyze: MainFn = analyze_main,
    provider_config_check: ProviderConfigCheckFn = _check_provider_configuration,
) -> None:
    run_dir = config.paths.outputs_experiment_dir(config.run_id)
    run_matrix_path = run_dir / "run_matrix.json"
    ledger_path = run_dir / "run_ledger.json"
    if config.resume:
        if config.clean_target:
            raise ValueError("A resumed run cannot clean its existing outputs.")
        _require_resumable_run(run_dir, config.run_id)
    else:
        _require_new_run(run_dir, config.run_id)
    _validate_protocol_freeze_binding(config)
    _require_provider_budget(config)
    config, scenario_records = _bind_scenario_dataset(config, run_dir)
    if config.persist_policy_family_metrics and not config.policy_bundle_sha256:
        raise ValueError(
            "policy_bundle_sha256 is required when policy-family metrics are persisted"
        )

    if config.preflight_data:
        _run_data_preflight(config)
    if config.proportionality_policy_path is not None:
        load_proportionality_policy(
            config.proportionality_policy_path
        ).require_approved()

    models_to_run, failure_rows, preflight_failure_count = _select_models_for_execution(
        config,
        provider_config_check,
    )
    if config.protocol_freeze is not None and preflight_failure_count:
        raise RuntimeError(
            "A frozen provider run cannot change its matrix after a model "
            "provider-configuration failure. No experimental cell was executed."
        )
    success_count = 0
    execution_failure_count = 0

    cells = build_run_matrix(
        RunMatrixDesign(
            run_id=config.run_id,
            dataset_release_id=config.dataset_release_id,
            eval_protocol_version=config.eval_protocol_version,
            incidents=tuple(config.incidents),
            models=tuple(
                (
                    profile.name,
                    profile.deployment,
                    _normalize_provider(profile.provider),
                )
                for profile in models_to_run
            ),
            arms=tuple(config.arms),
            repeats=config.repeats,
            execution_order=config.execution_order,
            seed=config.matrix_seed,
            policy_bundle_sha256=config.policy_bundle_sha256,
            protocol_freeze_sha256=(
                config.protocol_freeze.sha256
                if config.protocol_freeze is not None
                else None
            ),
            scenario_bank_sha256=config.scenario_bank_sha256,
            incident_partitions=(
                config.protocol_freeze.incident_partitions
                if config.protocol_freeze is not None
                else ()
            ),
        ),
        scenario_records=scenario_records or None,
    )
    lease = RunLease(
        run_dir / "run_lease.json",
        config.run_id,
        lease_seconds=config.lease_seconds,
        allow_stale_takeover=config.resume,
    )
    with lease:
        ledger = _prepare_run_ledger(
            config,
            lease,
            cells,
            run_matrix_path,
            ledger_path,
        )
        if config.clean_target:
            _clean_selected_outputs(config.paths, config.incidents)
        (
            success_count,
            execution_failure_count,
            execution_failures,
            completed_cells,
        ) = _execute_run(
            config,
            cells,
            {
                (
                    profile.name,
                    profile.deployment,
                    _normalize_provider(profile.provider),
                ): profile
                for profile in models_to_run
            },
            ledger,
            run_pipeline,
            lease,
            run_dir,
        )
    failure_rows.extend(_ledger_failure_rows(config, ledger, cells, execution_failures))
    completed_cells = ledger.completed_cells(cells)
    success_count = len(completed_cells)
    terminal_observation_count = len(ledger.terminal_observation_cells(cells))
    execution_failure_count = len(ledger.failed_cells(cells))
    provider_cost_tracker = ProviderCostTracker(run_dir / "provider_attempts.jsonl")
    provider_cost_tracker.refresh()

    if failure_rows:
        _write_failures_json(config.failures_json, failure_rows)
        if config.export_failures_csv:
            _write_failures_csv(config.failures_csv, failure_rows)
            print(f"Failures saved at: {config.failures_csv}")
        else:
            print(f"Failures saved at: {config.failures_json}")
    else:
        # Prevent stale failure artifacts from previous runs from being misread
        # as failures of the current run.
        if config.failures_json.exists():
            config.failures_json.unlink()
        if config.failures_csv.exists():
            config.failures_csv.unlink()

    if success_count == 0:
        raise RuntimeError("No successful execution in experiment_runner.")

    policy_family_metrics_path = None
    if config.persist_policy_family_metrics:
        policy_family_metrics_path = run_dir / "policy_family_metrics.json"
        persist_experiment_family_metrics(
            outputs_incidents_dir=config.paths.outputs_incidents_dir,
            run_dir=run_dir,
            experiment_run_id=config.run_id,
            completed_cells=completed_cells,
        )

    run_analyze(
        [
            "--outputs-incidents-dir",
            str(config.paths.outputs_incidents_dir),
            "--summary-json",
            str(config.summary_json),
            "--analysis-bundle-json",
            str(config.analysis_bundle_json),
            "--eval-protocol-version",
            config.eval_protocol_version,
            "--incidents",
            ",".join(config.incidents),
            "--run-id",
            config.run_id,
            "--repeat-reduction",
            config.repeat_reduction,
            "--expected-repeats",
            str(config.repeats),
        ]
    )
    _attach_experiment_metadata(
        ExperimentMetadataInput(
            summary_path=config.summary_json,
            bundle_path=config.analysis_bundle_json,
            data_preflight_path=config.data_preflight_json,
            data_quality_path=config.data_quality_json,
            failures_path=config.failures_json,
            profiles=config.model_profiles,
            selected_profiles=models_to_run,
            arms=config.arms,
            repeats=config.repeats,
            repeat_reduction=config.repeat_reduction,
            run_id=config.run_id,
            dataset_release_id=config.dataset_release_id,
            preflight=config.preflight,
            preflight_data=config.preflight_data,
            strict_data=config.strict_data,
            fail_fast=config.fail_fast,
            archive_run=config.archive_run,
            eval_protocol_version=config.eval_protocol_version,
            git_commit=_resolve_git_commit(config.paths.repo_root),
            incident_count=len(config.incidents),
            successful_run_count=success_count,
            terminal_observation_count=terminal_observation_count,
            preflight_failure_count=preflight_failure_count,
            execution_failure_count=execution_failure_count,
            execution_order=config.execution_order,
            matrix_seed=config.matrix_seed,
            run_matrix_path=run_matrix_path,
            policy_bundle_sha256=config.policy_bundle_sha256,
            policy_family_metrics_path=policy_family_metrics_path,
            llm_fixture_sha256=config.llm_fixture_sha256,
            scenario_bank_sha256=config.scenario_bank_sha256,
            scenario_counts={
                stratum: sum(
                    record.stratum == stratum for record in scenario_records.values()
                )
                for stratum in sorted(
                    {record.stratum for record in scenario_records.values()}
                )
            },
            protocol_freeze=config.protocol_freeze,
            approved_budget_usd=config.approved_budget_usd,
            accrued_provider_cost_usd=float(provider_cost_tracker.accrued_usd),
            protocol_study=config.protocol_study,
        )
    )
    incident_count = _resolve_analyzed_incident_count(config)
    if incident_count < config.min_incidents:
        raise RuntimeError(
            "Analyzed incident count below minimum threshold: "
            f"{incident_count} < {config.min_incidents}"
        )
    if config.archive_run:
        _archive_experiment_outputs(config)


def _write_failures_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "cell_id",
                "order_index",
                "model",
                "deployment",
                "arm",
                "repeat_index",
                "run_id",
                "incident_id",
                "error",
                "exception_type",
                "traceback",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_failures_json(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"failure_count": len(rows), "failures": rows}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _attach_experiment_metadata(data: ExperimentMetadataInput) -> None:
    coverage = _build_experiment_coverage(data)
    data_preflight = _read_json_if_exists(data.data_preflight_path)
    data_quality = _read_json_if_exists(data.data_quality_path)
    failure_summary = _build_failure_summary(data.failures_path)
    run_matrix = _read_json_if_exists(data.run_matrix_path)
    policy_family_metrics = (
        _read_json_if_exists(data.policy_family_metrics_path)
        if data.policy_family_metrics_path is not None
        else {}
    )
    if data.summary_path.exists():
        summary_data = json.loads(data.summary_path.read_text(encoding="utf-8"))
        summary_data["experiment_coverage"] = coverage
        data.summary_path.write_text(
            json.dumps(summary_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if not data.bundle_path.exists():
        return
    bundle_data = json.loads(data.bundle_path.read_text(encoding="utf-8"))
    bundle_data["experiment"] = {
        "run_id": data.run_id,
        "dataset_release_id": data.dataset_release_id,
        "eval_protocol_version": data.eval_protocol_version,
        "protocol_study": data.protocol_study,
        "git_commit": data.git_commit,
        "preflight_enabled": data.preflight,
        "preflight_data_enabled": data.preflight_data,
        "strict_data_enabled": data.strict_data,
        "fail_fast_enabled": data.fail_fast,
        "archive_run_enabled": data.archive_run,
        "arms": data.arms,
        "repeats": data.repeats,
        "repeat_reduction": data.repeat_reduction,
        "coverage": coverage,
        "data_preflight": data_preflight,
        "data_quality": data_quality,
        "failure_summary": failure_summary,
        "run_matrix": {
            "execution_order": data.execution_order,
            "seed": data.matrix_seed,
            "plan_sha256": run_matrix.get("plan_sha256"),
            "planned_cell_count": run_matrix.get("planned_cell_count"),
            "corpus_stratum_cell_counts": run_matrix.get(
                "corpus_stratum_cell_counts", {}
            ),
            "policy_bundle_sha256": data.policy_bundle_sha256,
            "protocol_freeze_sha256": (
                data.protocol_freeze.sha256
                if data.protocol_freeze is not None
                else None
            ),
        },
        "protocol_freeze": (
            {
                "path": data.protocol_freeze.relative_path,
                "sha256": data.protocol_freeze.sha256,
                "schema_version": data.protocol_freeze.schema_version,
                "study_id": data.protocol_freeze.study_id,
            }
            if data.protocol_freeze is not None
            else None
        ),
        "provider_cost_control": (
            {
                "approved_budget_usd": data.approved_budget_usd,
                "accrued_provider_cost_usd": data.accrued_provider_cost_usd,
                "enforcement": "pre_call_projected_cost_ceiling",
            }
            if data.approved_budget_usd is not None
            else None
        ),
        "policy_family_metrics": policy_family_metrics,
        "recommendation_source": (
            {
                "kind": "fixture_replay",
                "fixture_sha256": data.llm_fixture_sha256,
                "constitutes_model_evidence": False,
            }
            if data.llm_fixture_sha256 is not None
            else {
                "kind": "provider",
                "constitutes_model_evidence": True,
            }
        ),
        "scenario_bank": (
            {
                "manifest_sha256": data.scenario_bank_sha256,
                "count_by_stratum": data.scenario_counts or {},
            }
            if data.scenario_bank_sha256 is not None
            else None
        ),
        "registry_models": [
            profile.model_dump(mode="json") for profile in data.profiles
        ],
        "selected_models": [
            profile.model_dump(mode="json") for profile in data.selected_profiles
        ],
    }
    data.bundle_path.write_text(
        json.dumps(bundle_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _archive_experiment_outputs(config: ExperimentRunConfig) -> None:
    run_dir = config.paths.outputs_experiment_dir(config.run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    if config.summary_json.exists():
        shutil.copy2(config.summary_json, run_dir / "summary.json")
    if config.analysis_bundle_json.exists():
        shutil.copy2(config.analysis_bundle_json, run_dir / "analysis_bundle.json")
    if config.failures_json.exists():
        shutil.copy2(config.failures_json, run_dir / "experiment_failures.json")
    if config.data_preflight_json.exists():
        shutil.copy2(config.data_preflight_json, run_dir / "data_preflight.json")
    if config.data_quality_json.exists():
        shutil.copy2(config.data_quality_json, run_dir / "data_quality_report.json")
    if config.export_failures_csv and config.failures_csv.exists():
        shutil.copy2(config.failures_csv, run_dir / "experiment_failures.csv")
    if config.protocol_freeze is not None:
        shutil.copy2(config.protocol_freeze.path, run_dir / "protocol_freeze.json")

    manifest = {
        "run_id": config.run_id,
        "dataset_release_id": config.dataset_release_id,
        "eval_protocol_version": config.eval_protocol_version,
        "protocol_study": config.protocol_study,
        "incidents": config.incidents,
        "models": [profile.name for profile in config.model_profiles],
        "deployments": [profile.deployment for profile in config.model_profiles],
        "arms": config.arms,
        "repeats": config.repeats,
        "repeat_reduction": config.repeat_reduction,
        "strict_data": config.strict_data,
        "approval_policy_mode": config.approval_policy_mode,
        "preflight": config.preflight,
        "preflight_data": config.preflight_data,
        "execution_order": config.execution_order,
        "matrix_seed": config.matrix_seed,
        "policy_bundle_sha256": config.policy_bundle_sha256,
        "llm_fixture_sha256": config.llm_fixture_sha256,
        "scenario_bank_sha256": config.scenario_bank_sha256,
        "protocol_freeze": (
            {
                "source_path": config.protocol_freeze.relative_path,
                "sha256": config.protocol_freeze.sha256,
                "schema_version": config.protocol_freeze.schema_version,
                "study_id": config.protocol_freeze.study_id,
                "snapshot_path": "protocol_freeze.json",
            }
            if config.protocol_freeze is not None
            else None
        ),
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Reproducibility snapshot saved at: {run_dir}")


def _resolve_analyzed_incident_count(config: ExperimentRunConfig) -> int:
    if not config.summary_json.exists():
        return len(config.incidents)
    summary_data = json.loads(config.summary_json.read_text(encoding="utf-8"))
    count = int(summary_data.get("incident_count", 0))
    if count > 0:
        return count
    return len(config.incidents)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="soc_llm_policy.experiment_runner")
    parser.add_argument("--repo-root", default=None)
    parser.add_argument(
        "--models",
        default=None,
        help="CSV list of models/deployments. Optional override over registry.",
    )
    parser.add_argument(
        "--llm-fixture",
        default=None,
        help=(
            "Zero-cost recommendation fixture for end-to-end pipeline "
            "validation. Fixture runs are labeled as non-model evidence."
        ),
    )
    parser.add_argument(
        "--arms",
        default="llm_zero,llm_policy_prompt",
        help="CSV list of LLM arms. Example: llm_zero,llm_policy_prompt",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of repeats per combination (model x arm x incident).",
    )
    parser.add_argument(
        "--execution-order",
        choices=["sequential", "global_randomized", "paired_blocked"],
        default="paired_blocked",
        help=(
            "Cell execution schedule. paired_blocked preserves incident/repeat "
            "comparisons while interleaving model and prompt-arm cells."
        ),
    )
    parser.add_argument(
        "--matrix-seed",
        type=int,
        default=0,
        help="Seed used to construct the immutable run matrix.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Unique experiment ID. If omitted, UTC timestamp is generated.",
    )
    parser.add_argument(
        "--dataset-release-id",
        default="unknown",
        help="Dataset release identifier used for the experiment.",
    )
    parser.add_argument(
        "--min-incidents",
        type=int,
        default=1,
        help="Minimum analyzed incidents required to consider the run valid.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--incidents",
        default=None,
        help="CSV incident list. Example: INC_001,INC_002",
    )
    group.add_argument("--all", action="store_true")
    group.add_argument(
        "--scenario-manifest",
        default=None,
        help=(
            "Portable scenario-bank manifest. Its declared scenarios replace "
            "implicit directory enumeration."
        ),
    )
    parser.add_argument(
        "--scenario-bank-dir",
        default=None,
        help="Directory containing scenario_bank_manifest.json and packages/.",
    )
    parser.add_argument(
        "--scenario-source-incidents",
        default=None,
        help="Canonical source corpus referenced by the scenario-bank manifest.",
    )
    parser.add_argument("--clean-target", action="store_true")
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first execution error (default: continue and record).",
    )
    parser.add_argument(
        "--no-preflight",
        action="store_true",
        help="Disable local provider-configuration checks before execution.",
    )
    parser.add_argument(
        "--no-preflight-data",
        action="store_true",
        help="Disable input file preflight validation for incidents.",
    )
    parser.add_argument(
        "--no-strict-data",
        action="store_true",
        help=(
            "Disable pipeline fail-fast data mode "
            "(not recommended for frozen evaluations)."
        ),
    )
    parser.add_argument(
        "--approval-policy-mode",
        default="remove",
        choices=["remove", "defer_to_human_approval"],
        help=("Approval handling mode forwarded to pipeline runs."),
    )
    parser.add_argument(
        "--proportionality-policy",
        default=None,
        help=(
            "Optional reviewed proportionality-policy YAML. Pending policies "
            "fail before evaluation execution."
        ),
    )
    parser.add_argument(
        "--summary-json",
        default=None,
    )
    parser.add_argument(
        "--data-preflight-json",
        default=None,
    )
    parser.add_argument(
        "--data-quality-json",
        default=None,
    )
    parser.add_argument(
        "--analysis-bundle-json",
        default=None,
    )
    parser.add_argument(
        "--eval-protocol-version",
        default="official",
        help="Evaluation state label recorded in the final bundle.",
    )
    parser.add_argument(
        "--protocol-freeze-manifest",
        default=None,
        help="Current-schema pre-execution freeze required for provider execution.",
    )
    parser.add_argument(
        "--protocol-study",
        choices=["primary", "adversarial"],
        default="primary",
        help="Study bound inside the single campaign protocol freeze.",
    )
    parser.add_argument(
        "--validate-protocol-freeze-only",
        action="store_true",
        help="Validate the exact runtime binding, then exit before provider access.",
    )
    parser.add_argument(
        "--no-archive-run",
        action="store_true",
        help="Disable run snapshot in results/experiments/<run_id>/.",
    )
    parser.add_argument(
        "--failures-csv",
        default=None,
    )
    parser.add_argument(
        "--failures-json",
        default=None,
    )
    parser.add_argument(
        "--export-failures-csv",
        action="store_true",
        help="Export failures to CSV as well (default: JSON only).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Continue an interrupted run: reuse its frozen matrix and ledger, "
            "skip cells already completed, and re-run only the outstanding cells."
        ),
    )
    parser.add_argument(
        "--lease-seconds",
        type=int,
        default=900,
        help=(
            "Ownership lease horizon for the run directory. A second worker may "
            "take over only after the lease expires (with --resume)."
        ),
    )
    parser.add_argument(
        "--repeat-reduction",
        default="last",
        choices=["first", "last", "majority", "any"],
        help="Frozen reduction from repeated cells to one paired verdict.",
    )
    parser.add_argument(
        "--approved-budget-usd",
        type=float,
        default=None,
        help=(
            "Approved provider-cost ceiling. Required for a frozen provider run; "
            "checked before every scientific cell."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    _load_dotenv_if_available()
    args = build_parser().parse_args(argv)
    repo_root = resolve_repo_root(args.repo_root)
    paths = RepoPaths(repo_root=repo_root)

    fixture_path = (
        resolve_repo_relative_path(str(args.llm_fixture), repo_root)
        if args.llm_fixture
        else None
    )
    fixture_sha256 = None
    if fixture_path is not None:
        if args.models:
            raise ValueError("--models cannot be combined with --llm-fixture")
        fixture_payload = fixture_path.read_bytes()
        fixture_manifest = FixtureManifest.model_validate_json(
            fixture_payload.decode("utf-8")
        )
        fixture_sha256 = hashlib.sha256(fixture_payload).hexdigest()
        model_profiles = [
            ModelProfile(
                name=fixture_manifest.deployment,
                deployment=fixture_manifest.deployment,
                provider="fixture",
                enabled=True,
            )
        ]
    else:
        profiles = load_model_registry(paths.models_registry_path)
        model_selectors = _parse_csv_list(args.models)
        model_profiles = select_model_profiles(
            profiles,
            model_selectors if model_selectors else None,
        )
    arms = _parse_csv_list(args.arms)
    allowed_arms = {"llm_zero", "llm_policy_prompt"}
    invalid_arms = [arm for arm in arms if arm not in allowed_arms]
    if invalid_arms:
        raise ValueError(f"Invalid arms in --arms: {invalid_arms}")
    if not arms:
        raise ValueError("No arm specified in --arms.")
    if int(args.repeats) <= 0:
        raise ValueError("--repeats must be >= 1.")
    if int(args.min_incidents) <= 0:
        raise ValueError("--min-incidents must be >= 1.")
    run_id = (
        _sanitize_run_component(str(args.run_id)) if args.run_id else _default_run_id()
    )
    dataset_release_id = _sanitize_run_component(str(args.dataset_release_id))
    scenario_manifest_path = (
        resolve_repo_relative_path(str(args.scenario_manifest), repo_root)
        if args.scenario_manifest
        else None
    )
    if scenario_manifest_path is not None:
        if not args.scenario_bank_dir or not args.scenario_source_incidents:
            raise ValueError(
                "--scenario-manifest requires --scenario-bank-dir and "
                "--scenario-source-incidents"
            )
        scenario_manifest = load_manifest(scenario_manifest_path)
        incidents = [record.scenario_id for record in scenario_manifest.records]
        scenario_bank_sha256 = hashlib.sha256(
            scenario_manifest_path.read_bytes()
        ).hexdigest()
        scenario_bank_dir = resolve_repo_relative_path(
            str(args.scenario_bank_dir),
            repo_root,
        )
        scenario_source_incidents_dir = resolve_repo_relative_path(
            str(args.scenario_source_incidents),
            repo_root,
        )
    else:
        incidents = _resolve_incidents(paths, args.incidents, bool(args.all))
        scenario_bank_sha256 = None
        scenario_bank_dir = None
        scenario_source_incidents_dir = None

    proportionality_policy_path = (
        resolve_repo_relative_path(str(args.proportionality_policy), repo_root)
        if args.proportionality_policy
        else None
    )
    policy_bundle = build_policy_bundle(
        repo_root=repo_root,
        action_catalog_path=paths.inbox_action_catalog_path,
        constraints_path=paths.inbox_constraints_path,
        proportionality_policy_path=proportionality_policy_path,
    )
    protocol_freeze_path = (
        resolve_repo_relative_path(str(args.protocol_freeze_manifest), repo_root)
        if args.protocol_freeze_manifest
        else None
    )
    if fixture_path is None and protocol_freeze_path is None:
        raise ValueError("Provider execution requires --protocol-freeze-manifest.")
    protocol_study = str(args.protocol_study)
    if scenario_manifest_path is not None and protocol_study != "adversarial":
        raise ValueError(
            "Scenario-bank execution must use --protocol-study adversarial."
        )
    if protocol_study == "adversarial" and scenario_manifest_path is None:
        raise ValueError(
            "--protocol-study adversarial requires --scenario-manifest."
        )
    if protocol_freeze_path is not None and scenario_manifest_path is not None:
        _require_frozen_input_path(
            freeze_path=protocol_freeze_path,
            repo_root=repo_root,
            role="adversarial_scenario_manifest",
            active_path=scenario_manifest_path,
        )
    approval_policy_mode = str(args.approval_policy_mode)
    protocol_freeze = (
        _load_protocol_freeze_binding(
            paths=paths,
            freeze_path=protocol_freeze_path,
            study_id=protocol_study,
            expectation=ProtocolFreezeExpectation(
                incident_ids=tuple(incidents),
                official_models=tuple(profile.name for profile in model_profiles),
                arms=tuple(arms),
                repeats=int(args.repeats),
                dataset_release_id=dataset_release_id,
                eval_protocol_version=str(args.eval_protocol_version),
                execution_order=str(args.execution_order),
                matrix_seed=int(args.matrix_seed),
                repeat_reduction=str(args.repeat_reduction),
            ),
        )
        if protocol_freeze_path is not None
        else None
    )
    if args.validate_protocol_freeze_only:
        if protocol_freeze is None:
            raise ValueError(
                "--validate-protocol-freeze-only requires a protocol freeze."
            )
        print(
            "Protocol freeze runtime binding OK: "
            f"{protocol_freeze.relative_path} ({protocol_freeze.sha256})"
        )
        return

    run_experiments(
        ExperimentRunConfig(
            paths=paths,
            incidents=incidents,
            model_profiles=model_profiles,
            arms=arms,
            repeats=int(args.repeats),
            run_id=run_id,
            dataset_release_id=dataset_release_id,
            min_incidents=int(args.min_incidents),
            clean_target=bool(args.clean_target),
            fail_fast=bool(args.fail_fast),
            preflight=not bool(args.no_preflight),
            preflight_data=not bool(args.no_preflight_data),
            strict_data=not bool(args.no_strict_data),
            approval_policy_mode=approval_policy_mode,
            archive_run=not bool(args.no_archive_run),
            eval_protocol_version=str(args.eval_protocol_version),
            data_preflight_json=_resolve_analysis_output(
                explicit_value=args.data_preflight_json,
                paths=paths,
                default_filename="data_preflight.json",
            ),
            data_quality_json=_resolve_analysis_output(
                explicit_value=args.data_quality_json,
                paths=paths,
                default_filename="data_quality_report.json",
            ),
            summary_json=_resolve_analysis_output(
                explicit_value=args.summary_json,
                paths=paths,
                default_filename="summary.json",
            ),
            analysis_bundle_json=_resolve_analysis_output(
                explicit_value=args.analysis_bundle_json,
                paths=paths,
                default_filename="analysis_bundle.json",
            ),
            failures_csv=_resolve_analysis_output(
                explicit_value=args.failures_csv,
                paths=paths,
                default_filename="experiment_failures.csv",
            ),
            failures_json=_resolve_analysis_output(
                explicit_value=args.failures_json,
                paths=paths,
                default_filename="experiment_failures.json",
            ),
            export_failures_csv=bool(args.export_failures_csv),
            approved_budget_usd=args.approved_budget_usd,
            execution_order=str(args.execution_order),
            matrix_seed=int(args.matrix_seed),
            proportionality_policy_path=proportionality_policy_path,
            policy_bundle_sha256=policy_bundle.bundle_sha256,
            persist_policy_family_metrics=True,
            llm_fixture_path=fixture_path,
            llm_fixture_sha256=fixture_sha256,
            scenario_manifest_path=scenario_manifest_path,
            scenario_bank_dir=scenario_bank_dir,
            scenario_source_incidents_dir=scenario_source_incidents_dir,
            scenario_bank_sha256=scenario_bank_sha256,
            protocol_freeze=protocol_freeze,
            resume=bool(args.resume),
            lease_seconds=int(args.lease_seconds),
            require_output_evidence=True,
            repeat_reduction=str(args.repeat_reduction),
            protocol_study=protocol_study,
            provider_attempt_limit=(
                protocol_freeze.provider_attempt_limit
                if protocol_freeze is not None
                else 3
            ),
            connectivity_probe_timeout_seconds=(
                protocol_freeze.connectivity_probe_timeout_seconds
                if protocol_freeze is not None
                else 5.0
            ),
            connectivity_probe_interval_seconds=(
                protocol_freeze.connectivity_probe_interval_seconds
                if protocol_freeze is not None
                else 10.0
            ),
            connectivity_recovery_timeout_seconds=(
                protocol_freeze.connectivity_recovery_timeout_seconds
                if protocol_freeze is not None
                else 900.0
            ),
            connectivity_required_successes=(
                protocol_freeze.connectivity_required_successes
                if protocol_freeze is not None
                else 3
            ),
        )
    )


if __name__ == "__main__":
    main()
