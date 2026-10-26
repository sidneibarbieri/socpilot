from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from soc_llm_policy.campaign_protocol import (
    ProviderTransportProtocol,
    build_diagnostic_freeze,
    load_campaign_protocol,
)
from soc_llm_policy.corpus_campaign import validate_campaign
from soc_llm_policy.durable_io import atomic_write_json
from soc_llm_policy.execution_source import validate_execution_source_manifest
from soc_llm_policy.frozen_inputs import PROVIDER_INPUT_ROLES
from soc_llm_policy.paths import RepoPaths, repo_relative_path, resolve_repo_root
from soc_llm_policy.pipeline import list_inbox_incidents
from soc_llm_policy.policy_predicates import predicate_digest
from soc_llm_policy.preflight_evidence import (
    sha256_file,
    validate_policy_preflight_reports,
)

# Read contracts for the two serialized objects this module owns. A consumer
# checks the number it finds against the constant it imports from here; nobody
# restates the literal. The state a freeze describes is identified by the hashes
# it carries, never by this number.
PROTOCOL_FREEZE_SCHEMA = 8
POLICY_PREFLIGHT_MANIFEST_SCHEMA = 1


@dataclass(frozen=True)
class ProtocolFreezeOptions:
    dataset_release_id: str
    protocol_version: str
    execution_phase: str
    model_names: list[str]
    smoke_model_names: list[str]
    arm_names: list[str]
    policy_evidence_tier: str
    repeats: int
    matrix_seed: int
    execution_order: str
    llm_attempt_limit: int
    llm_retry_wait_seconds: float
    connectivity_probe_timeout_seconds: float
    connectivity_probe_interval_seconds: float
    connectivity_recovery_timeout_seconds: float
    connectivity_required_successes: int
    output_json: Path
    include_all_incidents: bool
    incident_ids: list[str]
    policy_preflight_manifest: Path | None = None
    corpus_campaign_manifest: Path | None = None
    execution_source_manifest: Path | None = None
    campaign_protocol_path: Path | None = None
    repeat_reduction: str = "last"


def sha256_incident_package(path: Path) -> str:
    if not path.is_dir():
        raise FileNotFoundError(f"Incident package not found: {path}")
    digest = hashlib.sha256()
    files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    for file_path in files:
        relative_path = file_path.relative_to(path).as_posix()
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(65536), b""):
                digest.update(chunk)
    return digest.hexdigest()


def compute_incident_hashes(
    paths: RepoPaths,
    incident_ids: list[str],
) -> dict[str, str]:
    return {
        incident_id: sha256_incident_package(paths.inbox_incident_dir(incident_id))
        for incident_id in incident_ids
    }


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def _git_commit(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _incident_ids(
    paths: RepoPaths,
    include_all: bool,
    selected_incident_ids: list[str],
) -> list[str]:
    available = list_inbox_incidents(paths)
    if include_all:
        return available
    if not selected_incident_ids:
        raise ValueError("No incidents selected for the protocol freeze manifest.")
    missing = sorted(set(selected_incident_ids) - set(available))
    if missing:
        raise FileNotFoundError(f"Incident IDs not found in dataset: {missing}")
    return selected_incident_ids


def _validate_policy_preflight_manifest(
    path: Path,
    constraints_path: Path,
) -> None:
    payload = _read_json(path)
    if payload.get("schema_version") != POLICY_PREFLIGHT_MANIFEST_SCHEMA:
        raise ValueError("Policy preflight manifest has an unsupported schema.")
    if payload.get("constitutes_model_evidence") is not False:
        raise ValueError("Policy preflight manifest must not claim model evidence.")
    if payload.get("state") != "policy_and_corpus_preflight":
        raise ValueError("Policy preflight manifest has an unexpected state.")
    if payload.get("constraints_sha256") != sha256_file(constraints_path):
        raise ValueError("Policy preflight manifest does not match the active policy.")
    if payload.get("policy_evidence_tier") != "operational":
        raise ValueError("Policy preflight must select the operational evidence tier.")
    if payload.get("policy_rule_counts") != {
        "operational": 15,
        "specification": 4,
    }:
        raise ValueError("Policy preflight contains unexpected policy-tier counts.")
    checks = payload.get("checks")
    required_checks = {
        "policy_review_valid",
        "policy_tiers_valid",
        "raw_exports_map_bijectively",
        "preflight_reports_share_policy_and_corpora",
    }
    if (
        not isinstance(checks, dict)
        or not required_checks.issubset(checks)
        or any(checks[name] is not True for name in required_checks)
    ):
        raise ValueError("Policy preflight manifest contains failed or missing checks.")

    sensitivity = payload.get("normalization_sensitivity")
    if not isinstance(sensitivity, dict):
        raise ValueError(
            "Policy preflight manifest is missing normalization sensitivity."
        )
    all_survive = sensitivity.get("leave_one_out_all_survive")
    review_status = sensitivity.get("review_status")
    if not isinstance(all_survive, bool) or review_status not in {
        "robust",
        "acknowledged_sensitive",
    }:
        raise ValueError("Policy preflight has an invalid normalization sensitivity.")
    expected_status = "robust" if all_survive else "acknowledged_sensitive"
    if review_status != expected_status:
        raise ValueError(
            "Policy preflight normalization sensitivity conflicts with review status."
        )

    counts = (
        payload.get("raw_export_count"),
        payload.get("derivation_incident_count"),
        payload.get("heldout_incident_count"),
    )
    if (
        any(isinstance(value, bool) or not isinstance(value, int) for value in counts)
        or counts[0] != counts[1] + counts[2]
    ):
        raise ValueError("Policy preflight manifest has inconsistent corpus counts.")

    reports = payload.get("reports")
    if not isinstance(reports, dict):
        raise ValueError("Policy preflight manifest is missing report hashes.")

    def read_report(name: str) -> dict[str, Any]:
        entry = reports.get(name)
        if not isinstance(entry, dict):
            raise ValueError(f"Policy preflight manifest is missing {name}.")
        relative_path = entry.get("path")
        expected_hash = entry.get("sha256")
        if not isinstance(relative_path, str) or not isinstance(expected_hash, str):
            raise ValueError(f"Policy preflight manifest has an invalid {name} entry.")
        report_path = Path(relative_path)
        if report_path.is_absolute() or ".." in report_path.parts:
            raise ValueError(f"Policy preflight report path escapes its state: {name}.")
        candidate = path.parent / report_path
        if not candidate.is_file():
            raise FileNotFoundError(f"Policy preflight report is missing: {candidate}")
        if sha256_file(candidate) != expected_hash:
            raise ValueError(f"Policy preflight report hash mismatch: {name}.")
        return _read_json(candidate)

    activation = read_report("heldout_activation")
    partition = read_report("corpus_partition_audit")
    normalization = read_report("normalization_ablation")
    read_report("policy_bundle")
    validate_policy_preflight_reports(
        heldout_activation=activation,
        partition_audit=partition,
        normalization_ablation=normalization,
        expected_predicate_sha256=predicate_digest(constraints_path),
    )


def build_protocol_freeze_manifest(
    *,
    paths: RepoPaths,
    options: ProtocolFreezeOptions,
) -> dict[str, Any]:
    transport = ProviderTransportProtocol(
        attempt_limit=options.llm_attempt_limit,
        retry_wait_seconds=options.llm_retry_wait_seconds,
        connectivity_probe_timeout_seconds=options.connectivity_probe_timeout_seconds,
        connectivity_probe_interval_seconds=options.connectivity_probe_interval_seconds,
        connectivity_recovery_timeout_seconds=options.connectivity_recovery_timeout_seconds,
        connectivity_required_successes=options.connectivity_required_successes,
    )
    if options.execution_phase != "primary":
        raise ValueError("The canonical provider campaign uses execution phase primary.")
    if options.repeat_reduction not in {"first", "last", "majority", "any"}:
        raise ValueError(f"Invalid repeat reduction: {options.repeat_reduction}")
    if options.repeat_reduction == "majority" and options.repeats % 2 == 0:
        raise ValueError("Majority repeat reduction requires an odd repeat count.")
    incident_ids = _incident_ids(
        paths,
        options.include_all_incidents,
        options.incident_ids,
    )
    dataset_audit_path = paths.outputs_analysis_dir / "dataset_audit.json"
    corpus_readiness_path = paths.outputs_analysis_dir / "corpus_readiness.json"
    release_readiness_path = paths.outputs_analysis_dir / "release_readiness.json"
    corpus_readiness = (
        _read_json(corpus_readiness_path) if corpus_readiness_path.exists() else {}
    )
    global_artifact_scope = corpus_readiness.get("global_artifact_scope", {})
    if not isinstance(global_artifact_scope, dict):
        global_artifact_scope = {}
    release_readiness = (
        _read_json(release_readiness_path) if release_readiness_path.exists() else {}
    )

    if options.policy_preflight_manifest is None:
        raise ValueError(
            "A controlled protocol freeze requires a policy-preflight manifest."
        )

    _validate_policy_preflight_manifest(
        options.policy_preflight_manifest,
        paths.inbox_constraints_path,
    )

    campaign = None
    if options.corpus_campaign_manifest is not None:
        campaign = validate_campaign(
            repo_root=paths.repo_root,
            manifest_path=options.corpus_campaign_manifest,
        )
        if sorted(campaign.incident_partitions) != sorted(incident_ids):
            raise ValueError(
                "Corpus campaign assignments differ from the frozen incident set."
            )

    source_identity: dict[str, object] | None = None
    if options.execution_source_manifest is not None:
        execution_source = validate_execution_source_manifest(
            repo_root=paths.repo_root,
            manifest_path=options.execution_source_manifest,
        )
        source_identity = {
            "kind": "execution_source_manifest",
            "path": repo_relative_path(execution_source.path, paths.repo_root),
            "sha256": execution_source.sha256,
            "file_count": execution_source.file_count,
        }

    hashed_inputs = [
        ("action_catalog", paths.inbox_action_catalog_path),
        ("constraints", paths.inbox_constraints_path),
        ("policy_review_record", paths.policy_review_record_path),
        ("models_registry", paths.models_freeze_path),
        ("attack_mapping", paths.attack_mapping_path),
        ("anonymization_policy", paths.anonymization_policy_path),
        ("llm_adapter", paths.repo_root / "src/soc_llm_policy/llm_adapter.py"),
        (
            "experiment_runner",
            paths.repo_root / "src/soc_llm_policy/experiment_runner.py",
        ),
        ("evaluation_plan", paths.repo_root / "src/soc_llm_policy/evaluation_plan.py"),
        ("policy_preflight_manifest", options.policy_preflight_manifest),
    ]
    if options.corpus_campaign_manifest is not None:
        hashed_inputs.append(
            ("corpus_campaign_manifest", options.corpus_campaign_manifest)
        )
    if options.execution_source_manifest is not None:
        hashed_inputs.append(
            ("execution_source_manifest", options.execution_source_manifest)
        )
    input_hashes: dict[str, dict[str, str]] = {}
    for name, path in hashed_inputs:
        if not path.exists():
            continue
        input_hashes[name] = {
            "path": repo_relative_path(path, paths.repo_root),
            "sha256": sha256_file(path),
        }
    missing_provider_inputs = sorted(set(PROVIDER_INPUT_ROLES) - set(input_hashes))
    if missing_provider_inputs:
        raise FileNotFoundError(
            "Controlled protocol freeze is missing inputs: "
            + ", ".join(missing_provider_inputs)
        )

    primary_cell_count = (
        len(incident_ids)
        * len(options.model_names)
        * len(options.arm_names)
        * options.repeats
    )
    diagnostic_studies: dict[str, object] = {}
    if options.campaign_protocol_path is not None:
        campaign_protocol = load_campaign_protocol(options.campaign_protocol_path)
        if campaign_protocol.provider_transport != transport:
            raise ValueError("Campaign transport policy differs from freeze options.")
        primary = campaign_protocol.primary
        expected_primary = {
            "incident_count": len(incident_ids),
            "models": options.model_names,
            "arms": options.arm_names,
            "repeats": options.repeats,
            "policy_evidence_tier": options.policy_evidence_tier,
            "repeat_reduction": options.repeat_reduction,
            "execution_order": options.execution_order,
            "matrix_seed": options.matrix_seed,
        }
        observed_primary = {
            "incident_count": primary.incident_count,
            "models": primary.models,
            "arms": primary.arms,
            "repeats": primary.repeats,
            "policy_evidence_tier": primary.policy_evidence_tier,
            "repeat_reduction": primary.repeat_reduction,
            "execution_order": primary.execution_order,
            "matrix_seed": primary.matrix_seed,
        }
        if expected_primary != observed_primary:
            raise ValueError(
                "Campaign protocol primary settings differ from freeze options."
            )
        diagnostic_studies, diagnostic_inputs = build_diagnostic_freeze(
            repo_root=paths.repo_root,
            campaign_protocol_path=options.campaign_protocol_path,
            protocol=campaign_protocol,
            primary_cell_count=primary_cell_count,
        )
        input_hashes.update(diagnostic_inputs)
    diagnostic_task_ceiling = int(
        diagnostic_studies.get("diagnostic_task_ceiling", 0)
    )
    return {
        "schema_version": PROTOCOL_FREEZE_SCHEMA,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(paths.repo_root),
        "source_identity": source_identity,
        "dataset_release_id": options.dataset_release_id,
        "eval_protocol_version": options.protocol_version,
        "execution_phase": options.execution_phase,
        "policy_evidence_tier": options.policy_evidence_tier,
        "official_models": options.model_names,
        "smoke_models": options.smoke_model_names,
        "arms": options.arm_names,
        "repeats": options.repeats,
        "matrix_seed": options.matrix_seed,
        "execution_order": options.execution_order,
        "repeat_reduction": options.repeat_reduction,
        "retry_policy": {
            "llm_attempt_limit": options.llm_attempt_limit,
            "llm_retry_wait_seconds": options.llm_retry_wait_seconds,
            "connectivity_probe_timeout_seconds": (
                options.connectivity_probe_timeout_seconds
            ),
            "connectivity_probe_interval_seconds": (
                options.connectivity_probe_interval_seconds
            ),
            "connectivity_recovery_timeout_seconds": (
                options.connectivity_recovery_timeout_seconds
            ),
            "connectivity_required_successes": (
                options.connectivity_required_successes
            ),
            "retryable_failures": [
                "transport_error",
                "timeout",
                "rate_limit",
                "designated_server_error",
            ],
        },
        "incident_count": len(incident_ids),
        "primary_cell_count": primary_cell_count,
        "diagnostic_task_ceiling": diagnostic_task_ceiling,
        "scientific_task_ceiling": primary_cell_count + diagnostic_task_ceiling,
        "diagnostic_studies": diagnostic_studies,
        "incident_ids": incident_ids,
        "incident_partitions": (
            dict(sorted(campaign.incident_partitions.items()))
            if campaign is not None
            else {incident_id: "unpartitioned" for incident_id in incident_ids}
        ),
        "partition_counts": (
            {
                label: partition.incident_count
                for label, partition in sorted(campaign.partitions.items())
            }
            if campaign is not None
            else {"unpartitioned": len(incident_ids)}
        ),
        "incident_hashes": compute_incident_hashes(paths, incident_ids),
        "policy_rule_count": int(
            global_artifact_scope.get("policy_rule_count", 0) or 0
        ),
        "action_catalog_count": int(
            global_artifact_scope.get("action_catalog_count", 0) or 0
        ),
        "release_readiness_status": str(release_readiness.get("status", "unknown")),
        "dataset_audit_status": "ok" if dataset_audit_path.exists() else "missing",
        "constitutes_model_evidence": False,
        "state": "pre_execution_protocol_freeze",
        "input_hashes": input_hashes,
        "notes": [
            "This manifest defines the canonical provider campaign.",
            "Provider attempts include the initial request and are distinct from "
            "scientific repeats.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soc_llm_policy.protocol_freeze",
        description=(
            "Write a canonical protocol-freeze manifest with hashes and "
            "official settings."
        ),
    )
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--dataset-release-id", required=True)
    parser.add_argument("--protocol-version", required=True)
    parser.add_argument(
        "--execution-phase",
        required=True,
        choices=["primary"],
    )
    parser.add_argument("--models", required=True)
    parser.add_argument("--smoke-models", required=True)
    parser.add_argument("--arms", required=True)
    parser.add_argument(
        "--policy-evidence-tier",
        default="operational",
        choices=["operational"],
    )
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--matrix-seed", type=int, default=0)
    parser.add_argument(
        "--execution-order",
        default="paired_blocked",
        choices=["grouped", "randomized", "paired_blocked"],
    )
    parser.add_argument(
        "--repeat-reduction",
        default="last",
        choices=["first", "last", "majority", "any"],
    )
    parser.add_argument(
        "--llm-attempt-limit",
        type=int,
        default=3,
        help="Total attempts per cell, including the initial provider request.",
    )
    parser.add_argument("--llm-retry-wait-seconds", type=float, default=65.0)
    parser.add_argument(
        "--connectivity-probe-timeout-seconds", type=float, default=5.0
    )
    parser.add_argument(
        "--connectivity-probe-interval-seconds", type=float, default=10.0
    )
    parser.add_argument(
        "--connectivity-recovery-timeout-seconds", type=float, default=900.0
    )
    parser.add_argument(
        "--connectivity-required-successes", type=int, default=3
    )
    parser.add_argument(
        "--output-json",
        default="results/analysis/protocol_freeze.json",
    )
    parser.add_argument("--incidents", default="")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--policy-preflight-manifest", default="")
    parser.add_argument("--corpus-campaign-manifest", default="")
    parser.add_argument("--execution-source-manifest", default="")
    parser.add_argument("--campaign-protocol", default="")
    return parser


def _parse_csv(value: str) -> list[str]:
    items: list[str] = []
    for part in value.split(","):
        item = part.strip()
        if item and item not in items:
            items.append(item)
    return items


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = resolve_repo_root(args.repo_root)
    paths = RepoPaths(repo_root=repo_root)
    output_json = Path(args.output_json)
    if not output_json.is_absolute():
        output_json = repo_root / output_json
    output_json.parent.mkdir(parents=True, exist_ok=True)
    policy_preflight_manifest = (
        Path(args.policy_preflight_manifest).expanduser().resolve()
        if args.policy_preflight_manifest
        else None
    )
    corpus_campaign_manifest = (
        Path(args.corpus_campaign_manifest).expanduser().resolve()
        if args.corpus_campaign_manifest
        else None
    )
    execution_source_manifest = (
        Path(args.execution_source_manifest).expanduser().resolve()
        if args.execution_source_manifest
        else None
    )
    campaign_protocol_path = (
        Path(args.campaign_protocol).expanduser().resolve()
        if args.campaign_protocol
        else None
    )

    manifest = build_protocol_freeze_manifest(
        paths=paths,
        options=ProtocolFreezeOptions(
            dataset_release_id=str(args.dataset_release_id).strip(),
            protocol_version=str(args.protocol_version).strip(),
            execution_phase=str(args.execution_phase).strip(),
            model_names=_parse_csv(args.models),
            smoke_model_names=_parse_csv(args.smoke_models),
            arm_names=_parse_csv(args.arms),
            policy_evidence_tier=str(args.policy_evidence_tier).strip(),
            repeats=int(args.repeats),
            matrix_seed=int(args.matrix_seed),
            execution_order=str(args.execution_order).strip(),
            llm_attempt_limit=int(args.llm_attempt_limit),
            llm_retry_wait_seconds=float(args.llm_retry_wait_seconds),
            connectivity_probe_timeout_seconds=float(
                args.connectivity_probe_timeout_seconds
            ),
            connectivity_probe_interval_seconds=float(
                args.connectivity_probe_interval_seconds
            ),
            connectivity_recovery_timeout_seconds=float(
                args.connectivity_recovery_timeout_seconds
            ),
            connectivity_required_successes=int(
                args.connectivity_required_successes
            ),
            output_json=output_json,
            include_all_incidents=bool(args.all),
            incident_ids=_parse_csv(args.incidents),
            policy_preflight_manifest=policy_preflight_manifest,
            corpus_campaign_manifest=corpus_campaign_manifest,
            execution_source_manifest=execution_source_manifest,
            campaign_protocol_path=campaign_protocol_path,
            repeat_reduction=str(args.repeat_reduction),
        ),
    )
    atomic_write_json(output_json, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
