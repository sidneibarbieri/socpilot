from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from soc_llm_policy.adversarial_scenarios import (
    bind_adversarial_manifest_to_packages,
    load_adversarial_manifest,
    validate_adversarial_bank,
)
from soc_llm_policy.artifact_verify import verify_artifact_package
from soc_llm_policy.campaign_evidence import (
    CampaignEvidence,
    load_campaign_evidence,
)
from soc_llm_policy.corpus_campaign import (
    CampaignPartition,
    CorpusCampaignManifest,
    materialize_campaign_view,
)
from soc_llm_policy.corpus_hash import canonical_corpus_sha256
from soc_llm_policy.dashboard_preview import verify_preview
from soc_llm_policy.dataset_audit import audit_dataset
from soc_llm_policy.execution_evidence import (
    ExecutionEvidence,
)
from soc_llm_policy.execution_source import build_execution_source_manifest
from soc_llm_policy.frozen_inputs import (
    PROVIDER_INPUT_ROLES,
    require_active_input_paths,
    resolve_frozen_input_bindings,
)
from soc_llm_policy.global_artifact_assessment import (
    build_global_artifact_assessment,
)
from soc_llm_policy.global_provenance import build_global_provenance
from soc_llm_policy.json_stability import write_stable_json
from soc_llm_policy.official_public_recheck import build_report as build_public_recheck
from soc_llm_policy.paths import (
    RepoPaths,
    resolve_repo_relative_path,
    resolve_repo_root,
)
from soc_llm_policy.policy_ir import (
    dump_policy_graph,
    load_policy_document,
    project_policy_graph,
)
from soc_llm_policy.policy_predicates import predicate_digest
from soc_llm_policy.policy_surface_audit import AuditPaths
from soc_llm_policy.policy_surface_audit import (
    build_report as build_policy_surface_report,
)
from soc_llm_policy.preflight_evidence import validate_policy_preflight_reports
from soc_llm_policy.protocol_freeze import (
    PROTOCOL_FREEZE_SCHEMA,
    sha256_incident_package,
)
from soc_llm_policy.publication_correlation import (
    collision_free_ip_aliases,
    new_private_projection,
    read_incident_times,
    write_private_map,
)
from soc_llm_policy.publication_redaction import (
    DATA_TREES,
    OUTPUT_TREES,
    RedactionVocabulary,
    project_loose_files_into_record,
    read_public_aliases,
    redact_release_trees,
    write_record,
)
from soc_llm_policy.publication_residue import (
    audit_raw_private_ip_overlap,
    audit_release_residue,
)
from soc_llm_policy.publication_tracebacks import publish_tracebacks
from soc_llm_policy.release_audit import audit_release
from soc_llm_policy.scenario_bank import (
    bind_manifest_to_packages,
    package_sha256,
)
from soc_llm_policy.scenario_bank import (
    load_manifest as load_scenario_manifest,
)
from soc_llm_policy.scenario_bank import (
    write_manifest as write_scenario_manifest,
)

_ROOT_FILES = [
    "LICENSE",
    "DATA_LICENSE.md",
    ".env.example",
    "pyproject.toml",
    "requirements.txt",
    "requirements-dev.txt",
    "reference_data/mitre/manifest.json",
    "docs/POLICY_REVIEW_PROTOCOL.md",
    "docs/REVIEWER_GUIDE.md",
    "docs/EXTENDING.md",
    "docs/EXECUTION_BACKUP.md",
    "tests/test_connectivity_evidence.py",
    "tests/test_connectivity_recovery.py",
    "tests/test_evaluation_plan.py",
    "tests/test_execution_storage.py",
    "tests/test_execution_backup.py",
    "tests/test_llm_adapter.py",
    "tests/test_public_artifact_runner.py",
    "tests/test_provider_status.py",
    "tests/test_provider_plan.py",
    "tests/test_policy_example.py",
]
_ROOT_DIRS = [
    "src",
    "scripts",
    "config",
    "scenario_banks/adversarial_2026_07",
    "protocol_extensions/projection_fidelity_probe",
]
_EXCLUDE_RELATIVE_FILE_PATHS = {
    "scripts/analyze_approval_mode_sensitivity.py",
    "scripts/analyze_paid_execution_candidates.py",
    "scripts/analyze_protocol_extension_candidates.py",
    "scripts/audit_mapping_samples.py",
    "scripts/audit_raw_soar_human_traces.py",
    "scripts/audit_reviewer_surface.py",
    "scripts/build_paper_workspace.sh",
    "scripts/build_campaign_source.py",
    "scripts/build_experiment_source.py",
    "scripts/check_template_lock.sh",
    "scripts/render_adversarial_tex.py",
    "scripts/render_evaluation_assets.py",
    "scripts/render_judge_ablation_tex.py",
    "scripts/render_measurement_tex.py",
    "scripts/render_policy_derivation_tex.py",
    "scripts/render_policy_coverage_table.py",
    "scripts/render_reviewer_diagnostics_tables.py",
    "scripts/render_reference_response_tex.py",
    "scripts/render_separation_tex.py",
    "scripts/render_values_tex.py",
    "scripts/replay_approval_modes_from_verifier_outputs.py",
    "scripts/separation_smoke.sh",
    "scripts/public_artifact_runner.sh",
    "scripts/public_provider_runner.sh",
    "scripts/adversarial_smoke.sh",
    "scripts/sync_artifact.sh",
    "scripts/suggest_global_policy_updates.py",
    "scripts/validate_fresh_clone_reproduction.py",
    "scripts/verify_paper_numbers.py",
    "config/models.local.yaml",
    "src/soc_llm_policy/select_minibatch.py",
}
_ANALYSIS_FILES = [
    "analysis_bundle.json",
    "summary.json",
    "data_preflight.json",
    "data_quality_report.json",
    "dataset_audit.json",
    "corpus_readiness.json",
    "global_artifact_assessment.json",
    "global_artifact_assessment.md",
    "global_artifact_provenance.json",
    "mini_batch_selection.json",
    "official_llm_analysis_bundle.json",
    "official_public_consistency.json",
    "official_evaluation_summary.json",
    "official_runs_manifest.json",
    "official_pairwise_tests.json",
    "release_readiness.json",
    "separation_input.schema.json",
    "separation_report.json",
    "separation_report.schema.json",
    "adversarial_report.json",
    "projection_fidelity_report.json",
    "projection_diagnostics.json",
    "stability_diagnostics.json",
    "judge_ablation_report.json",
    "repeat_reduction_sensitivity.json",
    "policy_surface_audit.json",
    "policy_surface_audit.md",
]
_POLICY_PREFLIGHT_REPORT_FILES = (
    "corpus_macros.tex",
    "corpus_partition_audit.json",
    "corpus_partition_audit.md",
    "heldout_activation.json",
    "heldout_activation.md",
    "normalization_ablation.json",
    "normalization_ablation_macros.tex",
    "policy_bundle_manifest.json",
    "policy_derivation_macros.tex",
    "policy_derivation_table.tex",
    "policy_preflight.json",
)
_POLICY_PREFLIGHT_SOURCE_FILES = tuple(
    filename
    for filename in _POLICY_PREFLIGHT_REPORT_FILES
    if not filename.endswith(".tex")
)
_ANALYSIS_DIRS = ["repeat_stability"]
_ANALYSIS_FIGURES = [
    "violation_rate_by_model.png",
    "violations_by_rule_model.png",
    "attack_tactic_frequency.png",
    "usage_budget.png",
]
_EXCLUDE_DIR_NAMES = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
_EXCLUDE_DIR_SUFFIXES = (".egg-info",)
_EXCLUDE_FILE_SUFFIXES = {".pyc"}
_EXCLUDE_FILE_NAMES = {".DS_Store"}
_MANIFEST_EXCLUDE_FILE_NAMES = {
    "artifact_manifest.json",
    "release_candidate_checklist.json",
    "release_readiness.json",
}
_MANIFEST_EXCLUDE_FILE_SUFFIXES = {"_recheck.json"}
_PUBLIC_ARTIFACT_LABEL = "socpilot_public_snapshot"
_PUBLIC_PROTOCOL_LABEL = "official"
_SCHEMA_WITH_INPUT_HASHES = 3
_PRIVATE_PLACEHOLDER = "<private-source>"
_RAW_EXPORT_PLACEHOLDER = "<private-raw-export>"
_REDACTED_EXPORT_PLACEHOLDER = "<private-redacted-export>"
_LOCAL_PATH_RE = re.compile(r"(/" r"Users/[^\s`\"']+)")
_SECRET_ASSIGNMENT_RE = re.compile(
    r"\b(api[-_ ]?key|access[-_ ]?token|authorization|password|passwd|secret)"
    r"(\s*[:=]\s*)(?:Bearer\s+)?([A-Za-z0-9_./+=-]{12,})",
    flags=re.IGNORECASE,
)
_RUN_TAG_TIMESTAMP_RE = re.compile(r"_[0-9]{8}T[0-9]{6,}Z$")
_LOCAL_TOOL_MARKER_REPLACEMENTS = {
    "Anti" + "gravity": "redacted_application",
    ".lm" + "studio": ".redacted_cli",
    "Oll" + "ama": "redacted_runtime",
    "Microsoft VS" + " Code": "redacted_editor",
    "Z" + "ed": "redacted_editor",
    "W" + "arp": "redacted_terminal",
}
_REPO_PATH_OVERRIDE_NAMES = (
    "SOC_LLM_POLICY_ACTION_CATALOG_PATH",
    "SOC_LLM_POLICY_CONSTRAINTS_PATH",
    "SOC_LLM_POLICY_REVIEW_RECORD_PATH",
    "SOC_LLM_POLICY_ACTION_MAPPING_BANK_PATH",
    "SOC_LLM_POLICY_INBOX_INCIDENTS_DIR",
    "SOC_LLM_POLICY_OUTPUTS_DIR",
)
_EVIDENCE_EXPORT_FILES = {
    "analysis_bundle": "official_llm_analysis_bundle.json",
    "connectivity_recovery": "connectivity_recovery.jsonl",
    "data_preflight": "data_preflight.json",
    "data_quality": "data_quality_report.json",
    "ledger": "run_ledger.json",
    "manifest": "run_manifest.json",
    "provider_attempts": "provider_attempts.jsonl",
    "run_matrix": "run_matrix.json",
    "summary": "official_evaluation_summary.json",
}
_FORBIDDEN_TERMINAL_EVIDENCE_ROOTS = {
    ".git",
    ".local",
    "dataset",
    "incoming",
    "paper",
    "policy",
    "private",
    "results",
}


@dataclass(frozen=True)
class PackageOptions:
    dataset_release_id: str
    incidents: list[str]
    include_analysis: bool
    run_id: str | None
    output_dir: Path
    protocol_freeze_path: Path | None = None
    campaign_evidence_path: Path | None = None
    # A freeze taken before preflight states existed binds no policy-derivation
    # evidence. Naming one here is an explicit, audited choice: the predicate
    # digest must match the fixed policy, so the reports decide over the same
    # rules the run was verified against.
    policy_preflight_manifest: Path | None = None
    # Legacy deterministic runs can be named only for non-provider analysis
    # packages. A controlled provider campaign is selected by one campaign
    # evidence manifest instead.
    experiment_runs: tuple[str, ...] = ()
    # The two acquisition batches form one campaign and retain provenance labels.
    heldout_dir: Path | None = None
    pre_execution_source: bool = False
    public_category_aliases_path: Path | None = None
    private_correlation_map_path: Path | None = None
    raw_dir: Path | None = None


@dataclass(frozen=True)
class _PackageContext:
    campaign: CampaignEvidence | None
    evidence: ExecutionEvidence | None
    freeze_src: Path
    heldout_dir: Path | None
    heldout_incidents: list[str]
    provider_incidents: list[str]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _sanitize_yaml_banner(text: str) -> str:
    lines = text.splitlines()
    while lines:
        stripped = lines[0].strip().lower()
        if not stripped.startswith("#"):
            break
        if any(
            token in stripped
            for token in ("candidate", "non-canonical", "legacy freeze")
        ):
            lines.pop(0)
            continue
        break
    out = "\n".join(lines).lstrip("\n")
    if text.endswith("\n") and not out.endswith("\n"):
        out += "\n"
    return out


def _copy_tree(src: Path, dst: Path, *, package_rel_root: str | None = None) -> None:
    if not src.exists():
        return
    for item in src.rglob("*"):
        if any(
            part in _EXCLUDE_DIR_NAMES or part.endswith(_EXCLUDE_DIR_SUFFIXES)
            for part in item.parts
        ):
            continue
        if not item.is_file():
            continue
        if item.name in _EXCLUDE_FILE_NAMES:
            continue
        if item.suffix in _EXCLUDE_FILE_SUFFIXES:
            continue
        rel = item.relative_to(src)
        if package_rel_root:
            dst_rel = f"{package_rel_root}/{rel.as_posix()}"
        else:
            dst_rel = rel.as_posix()
        if dst_rel in _EXCLUDE_RELATIVE_FILE_PATHS:
            continue
        _copy_file(item, dst / rel)


def _resolve_incidents(
    paths: RepoPaths, incidents_csv: str | None, use_all: bool
) -> list[str]:
    if use_all:
        out = sorted(
            path.name
            for path in paths.inbox_incidents_dir.glob("INC_*")
            if path.is_dir()
        )
    else:
        if not incidents_csv:
            raise ValueError("Provide --incidents or --all.")
        reader = csv.reader([incidents_csv])
        values = next(reader, [])
        out = sorted({value.strip() for value in values if value.strip()})
    if not out:
        raise ValueError("No incidents selected for packaging.")
    missing = [inc for inc in out if not paths.inbox_incident_dir(inc).exists()]
    if missing:
        raise FileNotFoundError(
            f"Selected incidents not found in dataset/incidents: {missing}"
        )
    return out


def _copy_dataset(
    paths: RepoPaths,
    release_root: Path,
    incidents: list[str],
    heldout_dir: Path | None,
    *,
    preserve_execution_source: bool = False,
) -> None:
    dataset_root = release_root / "artifact_data" / "dataset"
    global_root = release_root / "artifact_data" / "global"

    _copy_file(paths.inbox_action_catalog_path, global_root / "action_catalog.yaml")
    _copy_file(paths.inbox_constraints_path, global_root / "constraints.yaml")
    if preserve_execution_source:
        _copy_file(
            paths.inbox_action_catalog_path.parent / "policy_graph.json",
            global_root / "policy_graph.json",
        )
    else:
        policy_document = load_policy_document(
            paths.inbox_action_catalog_path,
            paths.inbox_constraints_path,
        )
        dump_policy_graph(
            project_policy_graph(policy_document),
            global_root / "policy_graph.json",
        )
    # The full policy bundle ships: rules, review record with per-rule
    # provenance, and the normalization instrument the content rules fire on.
    review_record = paths.inbox_action_catalog_path.parent / "review_record.yaml"
    if review_record.exists():
        _copy_file(review_record, global_root / "review_record.yaml")
    normalization = (
        paths.inbox_action_catalog_path.parent / "telemetry_normalization.yaml"
    )
    if normalization.exists():
        _copy_file(normalization, global_root / "telemetry_normalization.yaml")
    # The specification-coverage scenarios drive the rule-coverage command a
    # reviewer runs. They live in the policy source dir, not the active bundle,
    # so resolve both; shipping neither leaves rule-coverage without its input.
    coverage_scenarios = next(
        (
            candidate
            for candidate in (
                paths.inbox_action_catalog_path.parent / "coverage_scenarios.yaml",
                paths.policy_dir / "coverage_scenarios.yaml",
            )
            if candidate.exists()
        ),
        None,
    )
    if coverage_scenarios is None:
        raise FileNotFoundError(
            "coverage_scenarios.yaml is required for the rule-coverage command "
            "but was found in neither the active bundle nor the policy directory."
        )
    _copy_file(coverage_scenarios, global_root / "coverage_scenarios.yaml")
    playbook_src = paths.inbox_global_dir / "playbook"
    if playbook_src.exists():
        _copy_tree(
            playbook_src,
            global_root / "playbook",
            package_rel_root="artifact_data/global/playbook",
        )

    for incident_id in incidents:
        src_incident = paths.inbox_incident_dir(incident_id)
        dst_incident = dataset_root / incident_id
        _copy_tree(
            src_incident,
            dst_incident,
            package_rel_root=f"artifact_data/dataset/{incident_id}",
        )

    if heldout_dir is None:
        return
    heldout_root = release_root / "artifact_data" / "heldout"
    for src_incident in sorted(path for path in heldout_dir.iterdir() if path.is_dir()):
        _copy_tree(
            src_incident,
            heldout_root / src_incident.name,
            package_rel_root=f"artifact_data/heldout/{src_incident.name}",
        )


def _heldout_incident_ids(heldout_dir: Path | None) -> list[str]:
    if heldout_dir is None:
        return []
    if not heldout_dir.is_dir():
        raise FileNotFoundError(
            f"Second acquisition-batch directory not found: {heldout_dir}"
        )
    incidents = sorted(path.name for path in heldout_dir.iterdir() if path.is_dir())
    if not incidents:
        raise ValueError(f"Second acquisition-batch directory is empty: {heldout_dir}")
    return incidents


def _copy_project_files(
    paths: RepoPaths,
    release_root: Path,
    *,
    preserve_execution_source: bool = False,
    include_dashboard_preview: bool = True,
) -> None:
    for filename in ("LICENSE", "DATA_LICENSE.md"):
        license_path = paths.repo_root / filename
        if (
            not license_path.is_file()
            or not license_path.read_text(encoding="utf-8").strip()
        ):
            raise ValueError(f"A nonempty license file is required: {filename}")
    for rel in _ROOT_FILES:
        src = paths.repo_root / rel
        if src.exists():
            _copy_file(src, release_root / rel)
    preview_image = paths.repo_root / "docs/img/reviewer_dashboard.png"
    preview_receipt = preview_image.with_suffix(".preview.json")
    if (
        include_dashboard_preview
        and preview_image.is_file()
        and preview_receipt.is_file()
    ):
        _copy_file(preview_image, release_root / "docs/img/reviewer_dashboard.png")
        _copy_file(
            preview_receipt, release_root / "docs/img/reviewer_dashboard.preview.json"
        )
    for rel in _ROOT_DIRS:
        src_dir = paths.repo_root / rel
        if src_dir.exists():
            _copy_tree(src_dir, release_root / rel, package_rel_root=rel)
    if preserve_execution_source:
        for filename in ("run.sh", "provider_run.sh"):
            _copy_file(paths.repo_root / filename, release_root / filename)
        return
    public_runner = paths.repo_root / "scripts" / "public_artifact_runner.sh"
    if not public_runner.exists():
        raise FileNotFoundError(f"Public artifact runner not found: {public_runner}")
    packaged_runner = release_root / "run.sh"
    _copy_file(public_runner, packaged_runner)
    packaged_runner.chmod(0o755)
    provider_runner = paths.repo_root / "scripts" / "public_provider_runner.sh"
    if not provider_runner.exists():
        raise FileNotFoundError(f"Provider runner not found: {provider_runner}")
    packaged_provider_runner = release_root / "provider_run.sh"
    _copy_file(provider_runner, packaged_provider_runner)
    packaged_provider_runner.chmod(0o755)


def _iter_analysis_source_dirs(paths: RepoPaths) -> list[Path]:
    dirs = [paths.outputs_analysis_dir]
    public_analysis_dir = paths.repo_root / "artifact" / "artifact_outputs" / "analysis"
    if public_analysis_dir.exists():
        dirs.append(public_analysis_dir)
    return dirs


def _resolve_analysis_file_source(paths: RepoPaths, name: str) -> Path:
    for analysis_dir in _iter_analysis_source_dirs(paths):
        candidate = analysis_dir / name
        if candidate.exists():
            return candidate
    return paths.outputs_analysis_dir / name


def _reproject_runs_manifest_counts(paths: RepoPaths, analysis_dst: Path) -> None:
    """Give the shipped runs manifest the integer counts its rates come from.

    The historical manifest carries only rounded rates. The preserved bundle
    contains the integer counts, so the shipped manifest states those counts
    and derives downstream rates from them.
    """
    manifest_path = analysis_dst / "official_runs_manifest.json"
    bundle_path = _resolve_analysis_file_source(paths, "official_bundle.json")
    if not manifest_path.exists() or not bundle_path.exists():
        return
    bundle = _load_json(bundle_path)
    counts = {
        row["model_label"]: row
        for row in bundle.get("by_model", [])
        if isinstance(row, dict) and "model_label" in row
    }
    manifest = _load_json(manifest_path)
    for row in manifest.get("by_model_and_arm", []):
        label = f"{row.get('model')}|{row.get('arm_internal')}"
        source = counts.get(label)
        if source is None:
            raise ValueError(
                f"Runs manifest row {label} has no counts in the preserved bundle."
            )
        row["runs_with_violations"] = source["runs_with_violations"]
        row["enforcement_modified_run_count"] = source["enforcement_modified_run_count"]
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _copy_experiment_runs(
    paths: RepoPaths, release_root: Path, run_ids: tuple[str, ...]
) -> None:
    """Ship the per-cell verdict record of each named run, or say which is absent."""
    for run_id in run_ids:
        run_dir = paths.outputs_experiment_dir(run_id)
        if not run_dir.is_dir():
            raise FileNotFoundError(
                f"Experiment run requested for the release does not exist: {run_id}"
            )
        _copy_tree(
            run_dir,
            release_root / "artifact_outputs" / "experiments" / run_id,
            package_rel_root=f"artifact_outputs/experiments/{run_id}",
        )


def _copy_analysis(paths: RepoPaths, release_root: Path, run_id: str | None) -> None:
    analysis_dst = release_root / "artifact_outputs" / "analysis"
    for name in _ANALYSIS_FILES:
        src = _resolve_analysis_file_source(paths, name)
        if src.exists():
            _copy_file(src, analysis_dst / name)
    _reproject_runs_manifest_counts(paths, analysis_dst)
    for name in _ANALYSIS_DIRS:
        src_dir = paths.outputs_analysis_dir / name
        if src_dir.exists():
            _copy_tree(
                src_dir,
                analysis_dst / name,
                package_rel_root=f"artifact_outputs/analysis/{name}",
            )

    figures_src = paths.outputs_analysis_dir / "figures"
    if figures_src.exists():
        for name in _ANALYSIS_FIGURES:
            src = figures_src / name
            if src.exists():
                _copy_file(src, analysis_dst / "figures" / name)

    if run_id:
        run_dir = paths.outputs_experiment_dir(run_id)
        if run_dir.exists():
            _copy_tree(
                run_dir,
                release_root / "artifact_outputs" / "experiments" / run_id,
                package_rel_root=f"artifact_outputs/experiments/{run_id}",
            )
        checklist = paths.outputs_analysis_dir / f"official_checklist_{run_id}.json"
        if checklist.exists():
            _copy_file(checklist, analysis_dst / checklist.name)


def _copy_execution_evidence(
    evidence: ExecutionEvidence,
    release_root: Path,
) -> dict[str, Path]:
    analysis_dst = release_root / "artifact_outputs" / "analysis"
    copied: dict[str, Path] = {}
    for role, public_name in _EVIDENCE_EXPORT_FILES.items():
        destination = analysis_dst / public_name
        _copy_file(evidence.files[role].path, destination)
        copied[role] = destination
    return copied


def _repo_relative_source(path: Path, repo_root: Path, *, label: str) -> Path:
    resolved_root = repo_root.expanduser().resolve()
    resolved_path = path.expanduser().resolve()
    if resolved_path == resolved_root or resolved_root not in resolved_path.parents:
        raise ValueError(f"{label} must remain inside the artifact root.")
    return resolved_path.relative_to(resolved_root)


def _copy_exact_repo_file(
    source: Path,
    *,
    repo_root: Path,
    release_root: Path,
) -> Path:
    relative_path = _repo_relative_source(
        source,
        repo_root,
        label="Terminal campaign file",
    )
    if relative_path.parts[0] in _FORBIDDEN_TERMINAL_EVIDENCE_ROOTS:
        raise ValueError(
            "Terminal campaign evidence references a non-public path: "
            f"{relative_path.as_posix()}"
        )
    destination = release_root / relative_path
    if destination.exists():
        if not destination.is_file() or _sha256_file(destination) != _sha256_file(
            source
        ):
            raise ValueError(
                "Terminal campaign file collides with different packaged bytes: "
                f"{relative_path.as_posix()}"
            )
        return destination
    _copy_file(source, destination)
    if _sha256_file(destination) != _sha256_file(source):
        raise ValueError(
            f"Terminal campaign file changed while copying: {relative_path.as_posix()}"
        )
    return destination


def _copy_exact_repo_tree(
    source_dir: Path,
    *,
    repo_root: Path,
    release_root: Path,
) -> None:
    _repo_relative_source(source_dir, repo_root, label="Terminal campaign directory")
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Terminal campaign directory is missing: {source_dir}")
    for source in sorted(path for path in source_dir.rglob("*") if path.is_file()):
        _copy_exact_repo_file(
            source,
            repo_root=repo_root,
            release_root=release_root,
        )


def _copy_execution_output_files(
    evidence: ExecutionEvidence,
    *,
    repo_root: Path,
    release_root: Path,
) -> None:
    ledger = _load_json(evidence.files["ledger"].path)
    cells = ledger.get("cells")
    if not isinstance(cells, list):
        raise ValueError("Execution evidence ledger has no cells list.")
    for cell in cells:
        if not isinstance(cell, dict):
            raise ValueError("Execution evidence ledger contains a non-object cell.")
        output_evidence = cell.get("output_evidence")
        if output_evidence is None:
            continue
        if not isinstance(output_evidence, dict):
            raise ValueError("Execution output evidence must be an object.")
        files = output_evidence.get("files")
        if not isinstance(files, list) or not files:
            raise ValueError("Execution output evidence has no files.")
        for entry in files:
            if not isinstance(entry, dict):
                raise ValueError("Execution output evidence file must be an object.")
            raw_path = entry.get("path")
            expected_hash = entry.get("sha256")
            if not isinstance(raw_path, str) or not isinstance(expected_hash, str):
                raise ValueError(
                    "Execution output evidence file binding is incomplete."
                )
            source = resolve_repo_relative_path(raw_path, repo_root)
            if not source.is_file() or _sha256_file(source) != expected_hash:
                raise ValueError(f"Execution output evidence changed: {raw_path}")
            _copy_exact_repo_file(
                source,
                repo_root=repo_root,
                release_root=release_root,
            )


def _copy_frozen_inputs_exact(
    *,
    protocol_freeze_path: Path,
    repo_root: Path,
    release_root: Path,
) -> None:
    freeze = _load_json(protocol_freeze_path)
    bindings = resolve_frozen_input_bindings(
        freeze.get("input_hashes"),
        repo_root,
        required_roles=PROVIDER_INPUT_ROLES,
    )
    for role, binding in sorted(bindings.items()):
        relative_path = Path(binding.path)
        if relative_path.parts[0] in _FORBIDDEN_TERMINAL_EVIDENCE_ROOTS:
            raise ValueError(
                f"Fixed input {role} is outside the public source boundary: "
                f"{binding.path}"
            )
        _copy_exact_repo_file(
            binding.resolved_path,
            repo_root=repo_root,
            release_root=release_root,
        )

    preflight = bindings.get("policy_preflight_manifest")
    if preflight is not None:
        _validate_policy_preflight_manifest(
            paths=RepoPaths(repo_root=repo_root),
            manifest_path=preflight.resolved_path,
        )
        for filename in _POLICY_PREFLIGHT_REPORT_FILES:
            source = preflight.resolved_path.parent / filename
            if not source.is_file():
                raise FileNotFoundError(
                    f"Policy preflight is missing required file: {source}"
                )
            _copy_exact_repo_file(
                source,
                repo_root=repo_root,
                release_root=release_root,
            )


def _copy_terminal_campaign_evidence(
    campaign: CampaignEvidence,
    *,
    repo_root: Path,
    release_root: Path,
) -> Path:
    for source in (
        campaign.manifest_path,
        campaign.plan_path,
        campaign.protocol_freeze_path,
        campaign.provider_readiness_path,
        campaign.provider_readiness_attempt_log_path,
        campaign.primary_evidence.manifest_path,
        campaign.adversarial_evidence.manifest_path,
        *campaign.report_files.values(),
    ):
        _copy_exact_repo_file(
            source,
            repo_root=repo_root,
            release_root=release_root,
        )
    _copy_frozen_inputs_exact(
        protocol_freeze_path=campaign.protocol_freeze_path,
        repo_root=repo_root,
        release_root=release_root,
    )
    for evidence in (campaign.primary_evidence, campaign.adversarial_evidence):
        _copy_exact_repo_tree(
            evidence.run_dir,
            repo_root=repo_root,
            release_root=release_root,
        )
        _copy_execution_output_files(
            evidence,
            repo_root=repo_root,
            release_root=release_root,
        )
    for run_dir in campaign.diagnostic_run_dirs.values():
        _copy_exact_repo_tree(
            run_dir,
            repo_root=repo_root,
            release_root=release_root,
        )

    figures_dir = repo_root / "artifact_outputs" / "analysis" / "figures"
    if figures_dir.is_dir():
        _copy_exact_repo_tree(
            figures_dir,
            repo_root=repo_root,
            release_root=release_root,
        )

    campaign_manifest = (
        _load_json(campaign.protocol_freeze_path)
        .get("input_hashes", {})
        .get("corpus_campaign_manifest")
    )
    if not isinstance(campaign_manifest, dict) or not isinstance(
        campaign_manifest.get("path"), str
    ):
        raise ValueError("Terminal campaign freeze does not bind its corpus manifest.")
    packaged_campaign_manifest = release_root / campaign_manifest["path"]
    materialize_campaign_view(
        repo_root=release_root,
        manifest_path=packaged_campaign_manifest,
    )
    return release_root / campaign.manifest_path.relative_to(repo_root)


def _write_derived_public_reports(
    *,
    release_root: Path,
    evidence: ExecutionEvidence,
) -> dict[str, Path]:
    analysis_dir = release_root / "artifact_outputs" / "analysis"
    bundle = _load_json(evidence.files["analysis_bundle"].path)
    ledger = _load_json(evidence.files["ledger"].path)
    freeze = _load_json(evidence.protocol_freeze.path)

    by_model_and_arm = []
    for row in bundle.get("by_model", []):
        if not isinstance(row, dict):
            raise ValueError("Analysis bundle by_model rows must be objects.")
        model_label = str(row.get("model_label", ""))
        if "|" not in model_label:
            raise ValueError("Analysis bundle model label must include its arm.")
        model, arm = model_label.rsplit("|", 1)
        by_model_and_arm.append(
            {
                "model": model,
                "arm_internal": arm,
                "corpus_stratum": row.get("corpus_stratum", "all_incidents"),
                "completed_runs": row.get("run_count"),
                # The integer counts travel with the row. A rate rounded once in
                # the bundle and formatted again downstream loses a digit the
                # report prints from the counts, and the two surfaces disagree.
                "runs_with_violations": row.get("runs_with_violations"),
                "enforcement_modified_run_count": row.get(
                    "enforcement_modified_run_count"
                ),
                "run_violation_rate": row.get("run_violation_rate"),
                "enforcement_modification_rate": row.get(
                    "enforcement_modification_rate"
                ),
                "task_coverage_enforced_avg": row.get("task_coverage_enforced_avg"),
                "delta_jaccard_avg": row.get("delta_jaccard_avg"),
                "llm_total_tokens_total": row.get("llm_total_tokens_total"),
                "llm_cost_estimated_usd_total": row.get("llm_cost_estimated_usd_total"),
            }
        )

    status_counts = ledger.get("status_counts")
    if not isinstance(status_counts, dict):
        raise ValueError("Run ledger is missing status counts.")
    runs_manifest = {
        "manifest_version": 2,
        "reported_evaluation_run_id": evidence.run_id,
        "dataset_release_id": freeze.get("dataset_release_id"),
        "eval_protocol_version": freeze.get("eval_protocol_version"),
        "incident_count": freeze.get("incident_count"),
        "official_models": freeze.get("official_models"),
        "arms_internal": freeze.get("arms"),
        "repeats": freeze.get("repeats"),
        "execution_accounting": {
            "llm_trajectories_planned": ledger.get("cell_count"),
            "llm_trajectories_completed": status_counts.get("completed", 0),
            "llm_trajectories_failed_total": sum(
                status_counts.get(status, 0)
                for status in ("failed", "parse_failed", "refused")
            ),
            "execution_failures": sum(
                status_counts.get(status, 0)
                for status in ("failed", "parse_failed", "refused")
            ),
            "terminal_status_counts": status_counts,
        },
        "by_model_and_arm": by_model_and_arm,
    }
    pairwise_rows = bundle.get("pairwise", [])
    if not isinstance(pairwise_rows, list):
        raise ValueError("Analysis bundle pairwise rows must be a list.")
    pairwise_tests = {
        "row_count": len(pairwise_rows),
        "rows": pairwise_rows,
    }

    outputs = {
        "official_runs_manifest": analysis_dir / "official_runs_manifest.json",
        "official_pairwise_tests": analysis_dir / "official_pairwise_tests.json",
    }
    for role, payload in (
        ("official_runs_manifest", runs_manifest),
        ("official_pairwise_tests", pairwise_tests),
    ):
        outputs[role].write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return outputs


def _validate_evidence_release_binding(
    *,
    paths: RepoPaths,
    options: PackageOptions,
    evidence: ExecutionEvidence,
    heldout_dir: Path | None,
    heldout_incidents: list[str],
) -> dict[str, Any]:
    freeze = _load_json(evidence.protocol_freeze.path)
    freeze_incidents = freeze.get("incident_ids")
    if not isinstance(freeze_incidents, list) or not all(
        isinstance(item, str) and item for item in freeze_incidents
    ):
        raise ValueError("Execution evidence freeze has invalid incident IDs.")
    derivation_set = set(options.incidents)
    full_corpus_set = derivation_set | set(heldout_incidents)
    freeze_set = set(freeze_incidents)
    if freeze_set not in (derivation_set, full_corpus_set):
        raise ValueError(
            "Packaged partitions differ from the execution-evidence freeze."
        )
    if options.dataset_release_id != freeze.get("dataset_release_id"):
        raise ValueError(
            "Package dataset release differs from the execution-evidence freeze."
        )
    if options.run_id is not None and options.run_id != evidence.run_id:
        raise ValueError("Package run_id differs from the execution evidence.")
    if options.protocol_freeze_path is not None and (
        options.protocol_freeze_path.expanduser().resolve()
        != evidence.protocol_freeze.path
    ):
        raise ValueError(
            "Selected protocol freeze differs from the execution evidence."
        )

    bindings = resolve_frozen_input_bindings(
        freeze.get("input_hashes"),
        paths.repo_root,
        required_roles=PROVIDER_INPUT_ROLES,
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
    observed_hashes: dict[str, str] = {}
    for incident_id in freeze_incidents:
        source = paths.inbox_incident_dir(incident_id)
        if not source.is_dir() and heldout_dir is not None:
            source = heldout_dir / incident_id
        observed_hashes[incident_id] = sha256_incident_package(source)
    if freeze.get("incident_hashes") != observed_hashes:
        raise ValueError("Packaged incident content differs from the execution freeze.")
    return freeze


def _write_public_evidence_provenance(
    *,
    release_root: Path,
    campaign: CampaignEvidence,
    copied_files: dict[str, Path],
    derived_files: dict[str, Path],
) -> Path:
    evidence = campaign.primary_evidence
    output_path = (
        release_root
        / "artifact_outputs"
        / "analysis"
        / "public_evidence_provenance.json"
    )
    payload = {
        "schema_version": 2,
        "state": "public_campaign_evidence_projection",
        "constitutes_model_evidence": True,
        "campaign_id": campaign.campaign_id,
        "source_campaign_evidence_sha256": _sha256_file(campaign.manifest_path),
        "source_execution_evidence_sha256": _sha256_file(evidence.manifest_path),
        "source_protocol_freeze_sha256": evidence.protocol_freeze.sha256,
        "source_files": {
            role: {"sha256": evidence.files[role].sha256}
            for role in sorted(copied_files)
        },
        "public_files": {
            role: {
                "path": path.relative_to(release_root).as_posix(),
                "sha256": _sha256_file(path),
            }
            for role, path in sorted(copied_files.items())
        },
        "derived_files": {
            role: {
                "path": path.relative_to(release_root).as_posix(),
                "sha256": _sha256_file(path),
            }
            for role, path in sorted(derived_files.items())
        },
        "transformations": [
            "primary evidence files were copied to stable reviewer-facing names",
            "paired-test and run-accounting summaries were derived from the bound primary evidence",
        ],
        "terminal_evidence_bytes_preserved": True,
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def _write_package_analysis_reports(
    *,
    release_root: Path,
    incidents: list[str],
) -> dict[str, Path]:
    with _without_repo_path_overrides():
        with _package_incident_paths(release_root, incidents) as paths:
            return _write_package_analysis_reports_for_paths(
                paths=paths,
                incidents=incidents,
            )


def _write_package_analysis_reports_for_paths(
    *,
    paths: RepoPaths,
    incidents: list[str],
) -> dict[str, Path]:
    analysis_dir = paths.outputs_analysis_dir
    dataset_report = audit_dataset(paths=paths, incidents=incidents)
    if dataset_report.get("ok") is not True:
        raise ValueError("Packaged dataset failed validation or privacy checks.")

    outputs = {
        "dataset_audit": analysis_dir / "dataset_audit.json",
        "corpus_readiness": analysis_dir / "corpus_readiness.json",
        "global_artifact_assessment": (
            analysis_dir / "global_artifact_assessment.json"
        ),
        "official_public_consistency": (
            analysis_dir / "official_public_consistency.json"
        ),
    }
    assessment = build_global_artifact_assessment(
        paths=paths,
        incidents=incidents,
        official_summary_path=analysis_dir / "official_evaluation_summary.json",
    )
    consistency = build_public_recheck(
        official_bundle=_load_json(analysis_dir / "official_llm_analysis_bundle.json"),
        official_summary=_load_json(analysis_dir / "official_evaluation_summary.json"),
        runs_manifest=_load_json(analysis_dir / "official_runs_manifest.json"),
        pairwise_tests=_load_json(analysis_dir / "official_pairwise_tests.json"),
    )
    if consistency.get("status") != "pass":
        raise ValueError("Packaged provider evidence failed its consistency check.")

    payloads = {
        "dataset_audit": dataset_report,
        "corpus_readiness": dataset_report["corpus_readiness"],
        "global_artifact_assessment": assessment,
        "official_public_consistency": consistency,
    }
    for role, payload in payloads.items():
        outputs[role].write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return outputs


@contextmanager
def _package_incident_paths(
    release_root: Path,
    incident_ids: list[str],
) -> Iterator[RepoPaths]:
    derivation_root = release_root / "artifact_data" / "dataset"
    heldout_root = release_root / "artifact_data" / "heldout"
    if all((derivation_root / incident_id).is_dir() for incident_id in incident_ids):
        yield RepoPaths(repo_root=release_root)
        return

    view = release_root / "artifact_outputs" / ".package_campaign"
    if view.exists():
        raise FileExistsError(f"Temporary package campaign exists: {view}")
    view.mkdir(parents=True)
    try:
        for incident_id in incident_ids:
            candidates = (
                derivation_root / incident_id,
                heldout_root / incident_id,
            )
            sources = [candidate for candidate in candidates if candidate.is_dir()]
            if len(sources) != 1:
                raise ValueError(
                    f"Incident {incident_id} must occur in exactly one package partition."
                )
            target = os.path.relpath(sources[0], start=view)
            (view / incident_id).symlink_to(target, target_is_directory=True)
        yield RepoPaths(
            repo_root=release_root,
            inbox_incidents_override=view,
        )
    finally:
        shutil.rmtree(view, ignore_errors=True)


@contextmanager
def _without_repo_path_overrides() -> Iterator[None]:
    saved = {
        name: os.environ.pop(name)
        for name in _REPO_PATH_OVERRIDE_NAMES
        if name in os.environ
    }
    try:
        yield
    finally:
        os.environ.update(saved)


def _protocol_freeze_score(path: Path) -> tuple[int, int]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return (0, 0)
    input_hashes = payload.get("input_hashes")
    if not isinstance(input_hashes, dict):
        return (0, 0)
    score = sum(
        1
        for key in (
            "official_evaluation_summary",
            "official_llm_analysis_bundle",
            "official_runs_manifest",
            "official_pairwise_tests",
        )
        if key in input_hashes
    )
    return (score, len(input_hashes))


def _resolve_protocol_freeze_source(
    paths: RepoPaths,
    requested_path: Path | None = None,
) -> Path:
    if requested_path is not None:
        resolved_path = requested_path.resolve()
        if not resolved_path.is_file():
            raise FileNotFoundError(
                f"Selected protocol freeze manifest not found: {resolved_path}"
            )
        return resolved_path

    candidates: list[Path] = []
    for candidate in [
        paths.outputs_analysis_dir / "protocol_freeze.json",
        *sorted(paths.outputs_analysis_dir.glob("protocol_freeze*.json")),
        paths.repo_root
        / "artifact"
        / "artifact_outputs"
        / "analysis"
        / "protocol_freeze.json",
    ]:
        if candidate.exists() and candidate not in candidates:
            candidates.append(candidate)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        paths_text = ", ".join(str(candidate) for candidate in candidates)
        raise ValueError(
            "Multiple protocol freeze manifests are available. Select the "
            f"intended lineage with --protocol-freeze-json: {paths_text}"
        )
    raise FileNotFoundError(
        "Protocol manifest (`protocol_freeze.json`) not found. Run "
        "`./run_experiments.sh protocol-freeze --all --dataset-release-id <ID> "
        "--protocol-version <VERSION> --execution-phase main --repeats <N>` "
        "before packaging."
    )


def _require_packageable_freeze(
    freeze_src: Path,
    *,
    allow_pre_execution: bool,
) -> None:
    freeze = _load_json(freeze_src)
    is_pre_execution = freeze.get("state") == "pre_execution_protocol_freeze"
    if is_pre_execution and not allow_pre_execution:
        raise ValueError(
            "Cannot package a pre-execution protocol freeze. It does not bind "
            "provider results; complete a controlled execution and package a "
            "post-execution evidence state instead."
        )
    if allow_pre_execution and not is_pre_execution:
        raise ValueError(
            "A pre-execution source package requires a pre-execution protocol freeze."
        )


def _validate_pre_execution_package_options(options: PackageOptions) -> None:
    if not options.pre_execution_source:
        return
    if options.include_analysis:
        raise ValueError("A pre-execution source package cannot include analysis.")
    if options.campaign_evidence_path is not None or options.experiment_runs:
        raise ValueError("A pre-execution source package cannot include run evidence.")
    if options.run_id is not None:
        raise ValueError("A pre-execution source package cannot claim a run ID.")
    if options.heldout_dir is None:
        raise ValueError(
            "A pre-execution source package requires both acquisition batches."
        )
    if options.policy_preflight_manifest is None:
        raise ValueError(
            "A pre-execution source package requires an explicit policy preflight."
        )


def _copy_protocol_freeze(
    paths: RepoPaths,
    release_root: Path,
    freeze_src: Path,
) -> Path:
    analysis_dst = release_root / "artifact_outputs" / "analysis"
    freeze_dst = analysis_dst / "protocol_freeze.json"
    _copy_file(freeze_src, freeze_dst)
    return freeze_dst


def _source_campaign_manifest(
    *,
    paths: RepoPaths,
    freeze_src: Path,
) -> CorpusCampaignManifest:
    freeze = _load_json(freeze_src)
    input_hashes = freeze.get("input_hashes")
    if not isinstance(input_hashes, dict):
        raise ValueError("Protocol freeze is missing input hashes.")
    entry = input_hashes.get("corpus_campaign_manifest")
    if not isinstance(entry, dict):
        raise ValueError("Protocol freeze does not bind a corpus campaign.")
    raw_path = entry.get("path")
    expected_hash = entry.get("sha256")
    if not isinstance(raw_path, str) or not isinstance(expected_hash, str):
        raise ValueError("Protocol freeze has an invalid corpus-campaign binding.")
    source_path = resolve_repo_relative_path(raw_path, paths.repo_root)
    if _sha256_file(source_path) != expected_hash:
        raise ValueError("Corpus campaign hash does not match the protocol freeze.")
    return CorpusCampaignManifest.model_validate_json(
        source_path.read_text(encoding="utf-8")
    )


def _write_packaged_campaign_manifest(
    *,
    paths: RepoPaths,
    release_root: Path,
    freeze_src: Path,
) -> Path:
    source = _source_campaign_manifest(paths=paths, freeze_src=freeze_src)
    public_roots = {
        "artifact_data/dataset": release_root / "artifact_data" / "dataset",
        "artifact_data/heldout": release_root / "artifact_data" / "heldout",
    }
    public_ids = {
        relative: sorted(path.name for path in root.iterdir() if path.is_dir())
        for relative, root in public_roots.items()
    }
    partitions: dict[str, CampaignPartition] = {}
    used_public_roots: set[str] = set()
    for label, partition in sorted(source.partitions.items()):
        matches = [
            relative
            for relative, incident_ids in public_ids.items()
            if incident_ids == partition.incident_ids
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Campaign partition {label} does not map to one public corpus batch."
            )
        relative = matches[0]
        if relative in used_public_roots:
            raise ValueError("Two campaign partitions map to the same public batch.")
        used_public_roots.add(relative)
        partitions[label] = CampaignPartition(
            source_path=relative,
            incident_count=len(public_ids[relative]),
            incident_ids=public_ids[relative],
            corpus_sha256=canonical_corpus_sha256(public_roots[relative]),
        )
    if used_public_roots != set(public_roots):
        raise ValueError("The packaged campaign does not cover both public batches.")

    manifest = CorpusCampaignManifest(
        view_path=(
            "artifact_outputs/provider_state/canonical_campaign/campaign_corpus"
        ),
        total_incident_count=source.total_incident_count,
        partitions=partitions,
        incident_partitions=source.incident_partitions,
    )
    destination = (
        release_root / "artifact_outputs" / "analysis" / "corpus_campaign.json"
    )
    write_stable_json(destination, manifest.model_dump(mode="json"))
    return destination


def _bind_packaged_campaign(
    *,
    protocol_freeze_path: Path,
    campaign_manifest_path: Path,
    release_root: Path,
) -> None:
    payload = _load_json(protocol_freeze_path)
    input_hashes = payload.get("input_hashes")
    if not isinstance(input_hashes, dict):
        raise ValueError("Protocol freeze is missing input hashes.")
    input_hashes["corpus_campaign_manifest"] = {
        "path": campaign_manifest_path.relative_to(release_root).as_posix(),
        "sha256": _sha256_file(campaign_manifest_path),
    }
    payload["incident_hashes"] = {
        incident_id: sha256_incident_package(
            release_root / manifest_partition.source_path / incident_id
        )
        for manifest_partition in CorpusCampaignManifest.model_validate_json(
            campaign_manifest_path.read_text(encoding="utf-8")
        ).partitions.values()
        for incident_id in manifest_partition.incident_ids
    }
    protocol_freeze_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _bind_packaged_execution_source(
    *,
    protocol_freeze_path: Path,
    release_root: Path,
) -> Path:
    manifest_path = (
        release_root
        / "artifact_outputs"
        / "analysis"
        / "execution_source_manifest.json"
    )
    source = build_execution_source_manifest(
        repo_root=release_root,
        output_path=manifest_path,
    )
    payload = _load_json(protocol_freeze_path)
    input_hashes = payload.get("input_hashes")
    if not isinstance(input_hashes, dict):
        raise ValueError("Protocol freeze is missing input hashes.")
    relative_path = manifest_path.relative_to(release_root).as_posix()
    input_hashes["execution_source_manifest"] = {
        "path": relative_path,
        "sha256": source.sha256,
    }
    payload["source_identity"] = {
        "kind": "execution_source_manifest",
        "path": relative_path,
        "sha256": source.sha256,
        "file_count": source.file_count,
    }
    protocol_freeze_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _copy_and_bind_packaged_campaign(
    *,
    paths: RepoPaths,
    release_root: Path,
    freeze_src: Path | None,
    protocol_freeze_path: Path | None,
    required: bool,
) -> None:
    if freeze_src is None or protocol_freeze_path is None:
        if required:
            raise ValueError(
                "A pre-execution source package requires a protocol freeze."
            )
        return
    freeze_input_hashes = _load_json(freeze_src).get("input_hashes")
    if not isinstance(freeze_input_hashes, dict) or (
        "corpus_campaign_manifest" not in freeze_input_hashes
    ):
        if required:
            raise ValueError(
                "A pre-execution source package requires one 400-incident "
                "corpus-campaign binding."
            )
        return
    campaign_manifest_path = _write_packaged_campaign_manifest(
        paths=paths,
        release_root=release_root,
        freeze_src=freeze_src,
    )
    _bind_packaged_campaign(
        protocol_freeze_path=protocol_freeze_path,
        campaign_manifest_path=campaign_manifest_path,
        release_root=release_root,
    )


def _packaged_freeze_input_path(
    *,
    freeze: dict[str, Any],
    role: str,
    release_root: Path,
) -> Path:
    input_hashes = freeze.get("input_hashes")
    if not isinstance(input_hashes, dict):
        raise ValueError("Protocol freeze is missing input hashes.")
    entry = input_hashes.get(role)
    if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
        raise ValueError(f"Protocol freeze is missing input: {role}.")
    relative_path = Path(entry["path"])
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"Protocol freeze input escapes release: {role}.")
    path = release_root / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Protocol freeze input was not packaged: {role}.")
    return path


def _rebind_packaged_adversarial_bank(
    *,
    protocol_freeze_path: Path | None,
    release_root: Path,
) -> None:
    if protocol_freeze_path is None:
        return
    freeze = _load_json(protocol_freeze_path)
    input_hashes = freeze.get("input_hashes")
    if not isinstance(input_hashes, dict):
        return
    adversarial_roles = {
        "adversarial_pair_manifest",
        "adversarial_scenario_manifest",
    }
    present_roles = adversarial_roles.intersection(input_hashes)
    if not present_roles:
        return
    if present_roles != adversarial_roles:
        raise ValueError("Protocol freeze has an incomplete adversarial bank binding.")

    studies = freeze.get("diagnostic_studies")
    adversarial = studies.get("adversarial") if isinstance(studies, dict) else None
    source_partition = (
        adversarial.get("source_partition") if isinstance(adversarial, dict) else None
    )
    if not isinstance(source_partition, str) or not source_partition:
        raise ValueError("Adversarial study has no fixed source partition.")

    campaign_path = _packaged_freeze_input_path(
        freeze=freeze,
        role="corpus_campaign_manifest",
        release_root=release_root,
    )
    campaign = CorpusCampaignManifest.model_validate_json(
        campaign_path.read_text(encoding="utf-8")
    )
    partition = campaign.partitions.get(source_partition)
    if partition is None:
        raise ValueError(f"Adversarial source partition is absent: {source_partition}.")
    source_dir = release_root / partition.source_path

    pair_path = _packaged_freeze_input_path(
        freeze=freeze,
        role="adversarial_pair_manifest",
        release_root=release_root,
    )
    scenario_path = _packaged_freeze_input_path(
        freeze=freeze,
        role="adversarial_scenario_manifest",
        release_root=release_root,
    )
    if pair_path.parent != scenario_path.parent:
        raise ValueError("Adversarial manifests must share one bank directory.")
    bank_dir = pair_path.parent

    pair_manifest = bind_adversarial_manifest_to_packages(
        manifest=load_adversarial_manifest(pair_path),
        source_incidents_dir=source_dir,
        bank_dir=bank_dir,
    )
    scenario_manifest = bind_manifest_to_packages(
        manifest=load_scenario_manifest(scenario_path),
        source_incidents_dir=source_dir,
        bank_dir=bank_dir,
    )
    pair_scenario_ids = {
        scenario_id
        for pair in pair_manifest.pairs
        for scenario_id in (pair.clean_scenario_id, pair.injected_scenario_id)
    }
    scenario_ids = {record.scenario_id for record in scenario_manifest.records}
    if pair_scenario_ids != scenario_ids:
        raise ValueError("Adversarial pair and scenario manifests disagree.")
    if pair_manifest.source_corpus_id != scenario_manifest.source_corpus_id:
        raise ValueError("Adversarial manifests name different source corpora.")

    write_stable_json(pair_path, pair_manifest.model_dump(mode="json"))
    write_scenario_manifest(scenario_manifest, scenario_path)
    validate_adversarial_bank(pair_manifest, bank_dir)
    for record in scenario_manifest.records:
        package_dir = (
            source_dir / record.scenario_id
            if record.stratum == "real"
            else bank_dir / "packages" / record.scenario_id
        )
        if package_sha256(package_dir) != record.package_sha256:
            raise ValueError(
                f"Packaged adversarial scenario hash mismatch: {record.scenario_id}"
            )


def _frozen_constraints_entry(protocol_freeze_path: Path) -> dict[str, Any]:
    freeze = _load_json(protocol_freeze_path)
    input_hashes = freeze.get("input_hashes")
    if not isinstance(input_hashes, dict):
        raise ValueError("Protocol freeze is missing input hashes.")
    constraints = input_hashes.get("constraints")
    if not isinstance(constraints, dict):
        raise ValueError("Protocol freeze is missing the constraints hash.")
    if not isinstance(constraints.get("sha256"), str) or not isinstance(
        constraints.get("path"), str
    ):
        raise ValueError("Protocol freeze has an invalid constraints entry.")
    return constraints


def _frozen_predicate_digest(protocol_freeze_path: Path, repo_root: Path) -> str:
    """Return the policy-predicate digest authorized by this protocol.

    The file hash still pins the run: the policy arm's prompt carried the
    bundle's prose, so model evidence is evidence about that text. Deterministic
    reports read predicates alone, so they are checked against the predicates.
    """
    entry = _frozen_constraints_entry(protocol_freeze_path)
    constraints_path = resolve_repo_relative_path(entry["path"], repo_root)
    if _sha256_file(constraints_path) != entry["sha256"]:
        raise ValueError(
            f"{entry['path']} has changed since the protocol freeze was taken; "
            "the fixed run inputs are no longer reproducible from it."
        )
    return predicate_digest(constraints_path)


def _validate_preflight_reports(
    analysis_dir: Path,
    protocol_freeze_path: Path,
    repo_root: Path,
) -> None:
    """Reject a release that combines a freeze with another policy state."""
    report_paths = {
        "heldout_activation": analysis_dir / "heldout_activation.json",
        "partition_audit": analysis_dir / "corpus_partition_audit.json",
        "normalization_ablation": analysis_dir / "normalization_ablation.json",
    }
    present = [path.exists() for path in report_paths.values()]
    if not any(present):
        return
    if not all(present):
        raise ValueError(
            "A release with policy-preflight evidence must contain activation, "
            "partition, and normalization reports together."
        )
    validate_policy_preflight_reports(
        heldout_activation=_load_json(report_paths["heldout_activation"]),
        partition_audit=_load_json(report_paths["partition_audit"]),
        normalization_ablation=_load_json(report_paths["normalization_ablation"]),
        expected_predicate_sha256=_frozen_predicate_digest(
            protocol_freeze_path, repo_root
        ),
    )


def _write_release_boundary(
    release_root: Path, *, includes_terminal_campaign: bool
) -> None:
    evidence_scope = (
        "- Includes the exact per-task model inputs, model outputs, verifier "
        "records, provider-transport metadata, diagnostic ledgers, and aggregate "
        "reports bound by the terminal evidence manifest.\n"
        "- Excludes credentials, authorization headers, private raw exports, and "
        "private incident lineage."
        if includes_terminal_campaign
        else "- Contains no provider observations or model-result claims.\n"
        "- Excludes credentials, provider payloads, private raw exports, and "
        "private incident lineage."
    )
    text = f"""# Release Boundary

This package is the public artifact boundary for reproducibility. It is the
software and data package used to validate and rerun SOCpilot experiments. It is
not a publication workspace and does not include literature reviews, private
notes, or internal planning material.

- Contains only anonymized inputs under `artifact_data/`: two disjoint export
  batches in `dataset/` and `heldout/`, plus global policy inputs in `global/`.
- Excludes raw incoming data and secret-bearing local environment files.
- Includes the deterministic anonymization and conversion code under `local_redaction/`.
- Excludes local secrets (`.env`) and developer caches.
- Excludes publication workspaces, literature folders, private notes, and other
  author-only administrative resources.
- Includes optional analysis outputs under `artifact_outputs/` when available.
- The terminal evidence manifest, when present, is the sole authority for model
  evidence and binds every study to one protocol freeze.
{evidence_scope}
"""
    boundary = release_root / "ARTIFACT_BOUNDARY.md"
    boundary.write_text(text, encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def _policy_preflight_source(
    paths: RepoPaths,
    freeze_src: Path,
) -> Path | None:
    freeze = _load_json(freeze_src)
    input_hashes = freeze.get("input_hashes")
    if not isinstance(input_hashes, dict):
        # Schema versions before the preflight binding existed remain readable so
        # historical public snapshots can still be reconstructed. New freezes
        # must carry input hashes and are rejected by protocol_freeze.py.
        if freeze.get("schema_version", 1) >= _SCHEMA_WITH_INPUT_HASHES:
            raise ValueError("Protocol freeze is missing input hashes.")
        return None
    entry = input_hashes.get("policy_preflight_manifest")
    if entry is None:
        return None
    if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
        raise ValueError("Protocol freeze has an invalid policy-preflight entry.")
    manifest_path = resolve_repo_relative_path(entry["path"], paths.repo_root)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Policy preflight manifest referenced by freeze is missing: {manifest_path}"
        )
    if _sha256_file(manifest_path) != entry.get("sha256"):
        raise ValueError("Policy preflight manifest hash does not match the freeze.")
    return manifest_path


def _copy_policy_preflight(
    paths: RepoPaths,
    release_root: Path,
    freeze_src: Path | None,
    explicit_manifest: Path | None = None,
    *,
    source_only: bool = False,
) -> Path | None:
    manifest_path = (
        _policy_preflight_source(paths, freeze_src) if freeze_src is not None else None
    )
    if manifest_path is None:
        manifest_path = explicit_manifest
    elif explicit_manifest is not None and manifest_path != explicit_manifest:
        raise ValueError(
            "The protocol freeze binds a policy preflight manifest that differs "
            "from the one requested; refusing to package two policy states."
        )
    if manifest_path is None:
        return None
    preflight_dir = manifest_path.parent
    _validate_policy_preflight_manifest(
        paths=paths,
        manifest_path=manifest_path,
    )
    if freeze_src is not None:
        _validate_preflight_reports(preflight_dir, freeze_src, paths.repo_root)
    files_to_copy = (
        _POLICY_PREFLIGHT_SOURCE_FILES
        if source_only
        else _POLICY_PREFLIGHT_REPORT_FILES
    )
    for filename in files_to_copy:
        if not (preflight_dir / filename).is_file():
            raise FileNotFoundError(
                f"Policy preflight is missing required report: {filename}"
            )
    target_dir = (
        release_root
        / "artifact_outputs"
        / "analysis"
        / "policy_preflight"
        / preflight_dir.name
    )
    target_dir.mkdir(parents=True, exist_ok=False)
    for filename in files_to_copy:
        _copy_file(preflight_dir / filename, target_dir / filename)
    return target_dir


def _validate_policy_preflight_manifest(
    *,
    paths: RepoPaths,
    manifest_path: Path,
) -> None:
    """Validate one complete preflight directly against the active policy."""
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Policy preflight manifest not found: {manifest_path}")
    preflight_dir = manifest_path.parent
    manifest = _load_json(manifest_path)
    if manifest.get("state") != "policy_and_corpus_preflight":
        raise ValueError("Policy preflight manifest has an invalid state.")
    if manifest.get("constitutes_model_evidence") is not False:
        raise ValueError("Policy preflight cannot claim model evidence.")

    constraints_path = paths.inbox_constraints_path.resolve()
    constraints_sha256 = _sha256_file(constraints_path)
    if manifest.get("constraints_sha256") != constraints_sha256:
        raise ValueError("Policy preflight does not bind the active policy file.")

    report_entries = manifest.get("reports")
    if not isinstance(report_entries, dict):
        raise ValueError("Policy preflight manifest has no report bindings.")
    for role in (
        "corpus_partition_audit",
        "heldout_activation",
        "normalization_ablation",
        "policy_bundle",
    ):
        entry = report_entries.get(role)
        if not isinstance(entry, dict):
            raise ValueError(f"Policy preflight is missing report binding: {role}")
        relative_path = entry.get("path")
        expected_sha256 = entry.get("sha256")
        if not isinstance(relative_path, str) or not isinstance(expected_sha256, str):
            raise ValueError(f"Policy preflight has an invalid report binding: {role}")
        report_path = (preflight_dir / relative_path).resolve()
        if report_path.parent != preflight_dir.resolve() or not report_path.is_file():
            raise ValueError(f"Policy preflight report escapes its directory: {role}")
        if _sha256_file(report_path) != expected_sha256:
            raise ValueError(f"Policy preflight report hash changed: {role}")

    evidence = validate_policy_preflight_reports(
        heldout_activation=_load_json(preflight_dir / "heldout_activation.json"),
        partition_audit=_load_json(preflight_dir / "corpus_partition_audit.json"),
        normalization_ablation=_load_json(
            preflight_dir / "normalization_ablation.json"
        ),
        expected_predicate_sha256=predicate_digest(constraints_path),
    )
    expected_values = {
        "constraints_sha256": evidence.constraints_sha256,
        "derivation_corpus_sha256": evidence.derivation_corpus_sha256,
        "heldout_corpus_sha256": evidence.heldout_corpus_sha256,
        "derivation_incident_count": evidence.derivation_incident_count,
        "heldout_incident_count": evidence.heldout_incident_count,
        "raw_export_count": evidence.raw_export_count,
    }
    mismatches = [
        field
        for field, expected in expected_values.items()
        if manifest.get(field) != expected
    ]
    if mismatches:
        raise ValueError(
            "Policy preflight manifest disagrees with its reports: "
            + ", ".join(mismatches)
        )


def _bind_packaged_policy_preflight(
    protocol_freeze_path: Path,
    preflight_dir: Path | None,
    release_root: Path,
) -> None:
    if preflight_dir is None:
        return
    manifest_path = preflight_dir / "policy_preflight.json"
    payload = _load_json(protocol_freeze_path)
    input_hashes = payload.get("input_hashes")
    if not isinstance(input_hashes, dict):
        raise ValueError("Protocol freeze is missing input hashes.")
    if "policy_preflight_manifest" not in input_hashes:
        raise ValueError("Protocol freeze does not bind a policy preflight.")
    input_hashes["policy_preflight_manifest"] = {
        "path": manifest_path.relative_to(release_root).as_posix(),
        "sha256": _sha256_file(manifest_path),
    }
    protocol_freeze_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _public_incident_path(normalized: str, incident_hint: str | None) -> str | None:
    partitions = (
        ("dataset/incidents_heldout", "artifact_data/heldout"),
        ("dataset/incidents", "artifact_data/dataset"),
    )
    for source_root, public_root in partitions:
        if normalized == source_root:
            return public_root
        markers = (f"/{source_root}/", f"{source_root}/")
        for marker in markers:
            if marker not in normalized:
                continue
            incident_id = incident_hint
            if not incident_id:
                tail = normalized.split(marker, 1)[1]
                incident_id = tail.split("/", 1)[0]
            return f"{public_root}/{incident_id}" if incident_id else public_root
    return None


def _relative_for_public(value: str, incident_hint: str | None = None) -> str:
    normalized = value.replace("\\", "/")
    if "incoming/raw/" in normalized:
        return _RAW_EXPORT_PLACEHOLDER
    # Batch 2 redacts into incoming/redacted_heldout/, so match the
    # incoming/redacted prefix rather than a trailing slash that only the
    # Batch 1's incoming/redacted/ carries.
    if "incoming/redacted" in normalized or "redacted/" in normalized:
        return _REDACTED_EXPORT_PLACEHOLDER
    if "anonymized_staging/" in normalized:
        return _REDACTED_EXPORT_PLACEHOLDER
    preflight_marker = "results/policy_preflight/"
    if preflight_marker in normalized:
        suffix = normalized.split(preflight_marker, 1)[1]
        return f"artifact_outputs/analysis/policy_preflight/{suffix}"
    public_incident_path = _public_incident_path(normalized, incident_hint)
    if public_incident_path is not None:
        return public_incident_path
    path_map = {
        "policy/action_catalog.yaml": "artifact_data/global/action_catalog.yaml",
        "policy/constraints.yaml": "artifact_data/global/constraints.yaml",
        "results/analysis/dataset_audit.json": "artifact_outputs/analysis/dataset_audit.json",
        "results/analysis/corpus_readiness.json": "artifact_outputs/analysis/corpus_readiness.json",
        "results/analysis/global_artifact_assessment.json": "artifact_outputs/analysis/global_artifact_assessment.json",
        "results/analysis/global_artifact_assessment.md": "artifact_outputs/analysis/global_artifact_assessment.md",
        "results/analysis/official_llm_analysis_bundle.json": "artifact_outputs/analysis/official_llm_analysis_bundle.json",
        "results/analysis/official_evaluation_summary.json": "artifact_outputs/analysis/official_evaluation_summary.json",
        "results/analysis/release_readiness.json": "artifact_outputs/analysis/release_readiness.json",
        "action_catalog.yaml": "artifact_data/global/action_catalog.yaml",
        "constraints.yaml": "artifact_data/global/constraints.yaml",
        "models.freeze.yaml": "config/models.freeze.yaml",
        "attack_mapping.yaml": "config/attack_mapping.yaml",
        "anonymization_policy.yaml": "config/anonymization_policy.yaml",
        "dataset_audit.json": "artifact_outputs/analysis/dataset_audit.json",
        "corpus_readiness.json": "artifact_outputs/analysis/corpus_readiness.json",
        "global_artifact_assessment.json": "artifact_outputs/analysis/global_artifact_assessment.json",
        "global_artifact_assessment.md": "artifact_outputs/analysis/global_artifact_assessment.md",
        "official_llm_analysis_bundle.json": "artifact_outputs/analysis/official_llm_analysis_bundle.json",
        "official_evaluation_summary.json": "artifact_outputs/analysis/official_evaluation_summary.json",
        "official_runs_manifest.json": "artifact_outputs/analysis/official_runs_manifest.json",
        "official_pairwise_tests.json": "artifact_outputs/analysis/official_pairwise_tests.json",
        "review_record.yaml": "artifact_data/global/review_record.yaml",
        "telemetry_normalization.yaml": "artifact_data/global/telemetry_normalization.yaml",
        "release_readiness.json": "artifact_outputs/analysis/release_readiness.json",
    }
    for suffix, public_path in path_map.items():
        if normalized.endswith(suffix):
            return public_path
    if normalized.startswith("/"):
        return _PRIVATE_PLACEHOLDER
    return value


def _scrub_string(
    value: str,
    *,
    dataset_release_id: str,
    preserve_runtime_names: bool = False,
) -> str:
    out = value
    if dataset_release_id:
        out = out.replace(dataset_release_id, _PUBLIC_ARTIFACT_LABEL)
    for marker, replacement in _LOCAL_TOOL_MARKER_REPLACEMENTS.items():
        if preserve_runtime_names and marker == "Ollama":
            continue
        out = out.replace(marker, replacement)
    out = out.replace("financial_2026q1_", _PUBLIC_ARTIFACT_LABEL)
    out = out.replace(
        "Earlier exploratory bundles are intentionally excluded from the official provenance of this package.",
        "",
    )
    out = out.replace(
        "Exploratory bundles are excluded from the official experimental provenance.",
        "",
    )
    out = out.replace(
        "Exploratory bundles must not be mixed with the official evaluation "
        "bundle as if they were one experiment.",
        "",
    )
    out = out.replace("mini-batch", "evaluation subset")
    out = out.replace("used by the paper", "used by the reported analysis")
    out = out.replace(
        "needed to audit the paper's paired-tests table",
        "needed to audit the paired statistical tests",
    )
    out = _SECRET_ASSIGNMENT_RE.sub(
        lambda match: f"{match.group(1)}{match.group(2)}<redacted-secret>",
        out,
    )
    out = _LOCAL_PATH_RE.sub(_PRIVATE_PLACEHOLDER, out)
    return out


def _scrub_json_value(
    value: Any,
    *,
    dataset_release_id: str,
    parent_key: str | None = None,
    incident_hint: str | None = None,
) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if key in {
                "mapping_rules_path",
                "mapping_rules_sha256",
                "action_mapping_quality",
            }:
                continue
            if key == "dataset_release_id":
                out[key] = _PUBLIC_ARTIFACT_LABEL
                continue
            if key == "eval_protocol_version":
                out[key] = _PUBLIC_PROTOCOL_LABEL
                continue
            if key == "notes" and isinstance(item, list):
                out[key] = [
                    "This manifest defines the fixed protocol for the public artifact."
                ]
                continue
            out[key] = _scrub_json_value(
                item,
                dataset_release_id=dataset_release_id,
                parent_key=key,
                incident_hint=incident_hint,
            )
        return out
    if isinstance(value, list):
        return [
            _scrub_json_value(
                item,
                dataset_release_id=dataset_release_id,
                parent_key=parent_key,
                incident_hint=incident_hint,
            )
            for item in value
        ]
    if isinstance(value, str):
        if parent_key in {
            "path",
            "source_raw_file",
            "redacted_export",
            "incident_dir",
            "source_path",
            "official_summary_source",
            "repo_root",
            "legacy_root",
        }:
            if parent_key == "official_summary_source":
                return "artifact_outputs/analysis/official_evaluation_summary.json"
            return _relative_for_public(value, incident_hint=incident_hint)
        if parent_key == "dataset_release_id":
            return _PUBLIC_ARTIFACT_LABEL
        if parent_key == "eval_protocol_version":
            return _PUBLIC_PROTOCOL_LABEL
        return _scrub_string(value, dataset_release_id=dataset_release_id)
    return value


def _sanitize_json_file(
    path: Path, *, dataset_release_id: str, incident_hint: str | None = None
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload = _scrub_json_value(
        payload,
        dataset_release_id=dataset_release_id,
        incident_hint=incident_hint,
    )
    if path.name in {"analysis_bundle.json", "official_llm_analysis_bundle.json"}:
        payload = _stabilize_public_analysis_bundle(payload)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _sanitize_jsonl_file(path: Path, *, dataset_release_id: str) -> None:
    sanitized_lines: list[str] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
        sanitized_payload = _scrub_json_value(
            payload,
            dataset_release_id=dataset_release_id,
        )
        sanitized_lines.append(
            json.dumps(sanitized_payload, ensure_ascii=False, separators=(",", ":"))
        )
    trailing_newline = "\n" if sanitized_lines else ""
    path.write_text(
        "\n".join(sanitized_lines) + trailing_newline,
        encoding="utf-8",
    )


def _stabilize_public_run_tag(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        return normalized
    return _RUN_TAG_TIMESTAMP_RE.sub("_public_bundle", normalized)


def _stabilize_public_analysis_bundle(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    incidents = payload.get("incidents")
    if not isinstance(incidents, list):
        return payload
    for item in incidents:
        if not isinstance(item, dict):
            continue
        run_tag = item.get("run_tag")
        if isinstance(run_tag, str):
            item["run_tag"] = _stabilize_public_run_tag(run_tag)
    return payload


def _sanitize_global_artifacts(release_root: Path) -> None:
    for rel in ["artifact_data/global/constraints.yaml"]:
        path = release_root / rel
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        path.write_text(_sanitize_yaml_banner(text), encoding="utf-8")


def _write_package_readme(release_root: Path) -> None:
    text = """# Artifact Package

This package is the executable SOCpilot research artifact. It contains the
400-incident anonymized corpus, acquisition-batch provenance, declared policy inputs, verifier, experiment
scripts, and non-private evidence needed to validate the reported results.

Incident identifiers are opaque aliases (`INC_PUBLIC_...`) and every incident's
event times are shifted into the past by a random offset. The shift preserves the
order and spacing of events within an incident; calendar dates and the timing
between unrelated incidents carry no information. Verdicts do not read times, so
the recorded decision digests are unchanged.

The package is independent of any publication workspace or event template. It
does not include literature collections, private notes, raw exports, local
provider payloads, credentials, or publication-workspace automation.

## Environment setup
Python 3.12 or newer is required. Check `python3 --version` before creating a
virtual environment inside this package. If it reports an older version, use a
Python 3.12+ interpreter for the command below. The bundled
`reproduce.sh` selects the offline analysis runtime and auto-detects `.venv/`
when present.

```bash
python3 -m venv .venv
source .venv/bin/activate
.venv/bin/pip install -r requirements.txt
```

`requirements-dev.txt` is optional and intended for maintainers working from the
full source repository. Use `reproduce.sh` for the zero-cost checks below: it
selects the bundled `analysis_runtime`. The separate `run.sh` retains the frozen
collection source for provenance and is not the offline replay entry point.

## Recommended verification order
Open `REVIEWER_DASHBOARD.html` first when reviewing a downloaded copy of this
package. It is a static, self-contained entry point for the public evidence
surface: corpus readiness, model-execution accounting, policy coverage, paired
contrasts, replay stability, and direct links to the files behind the reported
claims.

Regenerate it with:

```bash
bash run.sh reproduce-results
```

This zero-cost command validates the public artifact, recomputes corpus
partition, transfer activation, and normalization-sensitivity evidence from the
released incident packages, regenerates evidence summaries, writes a
reproduction report, audits the policy-evidence boundary, and renders both
`REVIEWER_DASHBOARD.html` and
`artifact_outputs/dashboard/index.html`. To run only the integrity and dataset
checks, use `./run.sh validate-public-artifact`.

The reviewed YAML policy is the semantic source. Its graph is a deterministic
projection for inspection, not a second policy definition. Verify the graph and
typed rules agree with:

```bash
bash run.sh policy-graph-validate
```

`docs/POLICY_REVIEW_PROTOCOL.md` separates executable policy checks from the
domain judgment required to accept, revise, or reject a rule.

The validation path runs artifact integrity checks, dataset audit, global-artifact
assessment, and release-hygiene rechecks and writes fresh verification reports
under `artifact_outputs/analysis/`.
When `official_llm_analysis_bundle.json` is present, the command also rechecks
that the sanitized official row-level metrics agree with the shipped official
summary, run manifest, and paired-test rows.
Local transient files such as `.venv/` and `__pycache__/` are ignored by
the release-surface checks for the packaged public path.
It also verifies that `artifact_outputs/analysis/protocol_freeze.json` points
to packaged files whose SHA-256 digests match the published bundle.
The terminal model-execution evidence includes the anonymized per-task model inputs,
model outputs, verifier records, worker-execution histories, and provider
transport-attempt metadata needed to recount the results. It excludes
credentials, authorization headers, private raw exports, and private incident
lineage. `artifact_outputs/evidence/` contains the sole terminal evidence
manifest; convenience reports in `artifact_outputs/analysis/` are derived from
evidence bound by that manifest. The released package does not mutate this
evidence. A study with another model, policy, or corpus must use a separate copy
of the source code and a new protocol manifest.

Readers should interpret the rule set across two evidence tiers. On the real
corpus, the evaluation uses only the 15 rules marked `operational`. Four
approval rules are exercised only by fixed specification-coverage scenarios
because the exports contain no authenticated approval record. Model behavior is
reported for the complete 400-incident corpus and with both export-batch
sensitivity estimates.

## Package layout
- `artifact_data/dataset/`: first export batch; policy-derivation source
- `artifact_data/heldout/`: second acquisition batch; policy-transfer check
- `artifact_data/global/`: policy and normalization inputs used by the run
- `artifact_outputs/analysis/`: non-private analysis outputs and the official protocol manifest
- `src/`, `scripts/`, `config/`, `local_redaction/`: reproducibility and data-preparation code

Use `README.md` for the package entry point and `EVAL_PROTOCOL.md` for the
evaluation setup. The model registry used by the reported run is recorded in
`config/models.freeze.yaml`; the filename is retained for compatibility. Local
execution overrides are intentionally excluded.
"""
    preview_path = release_root / "docs/img/reviewer_dashboard.png"
    if preview_path.is_file():
        preview_link = (
            "![Reviewer dashboard preview](docs/img/reviewer_dashboard.png)\n\n"
        )
        text = text.replace(
            "Regenerate it with:\n", preview_link + "Regenerate it with:\n", 1
        )
    (release_root / "README.md").write_text(text, encoding="utf-8")


def _write_package_eval_protocol(
    *, release_root: Path, protocol_freeze_path: Path
) -> None:
    freeze_manifest = _load_json(protocol_freeze_path)
    official_models = freeze_manifest.get("official_models", [])
    if not isinstance(official_models, list):
        official_models = []
    arms = freeze_manifest.get("arms", [])
    if not isinstance(arms, list):
        arms = []
    incident_count = int(freeze_manifest.get("incident_count", 0) or 0)
    partition_counts = freeze_manifest.get("partition_counts", {})
    if not isinstance(partition_counts, dict):
        partition_counts = {}
    batch_one_count = int(partition_counts.get("export_batch_1", 0) or 0)
    batch_two_count = int(partition_counts.get("export_batch_2", 0) or 0)
    models_text = ", ".join(str(item) for item in official_models) or "unknown"
    arms_text = ", ".join(str(item) for item in arms) or "unknown"
    text = f"""# Evaluation Protocol

## Objective
Evaluate whether LLM-generated incident-response plans violate declared SOC
policy rules, where those violations concentrate, and how deterministic
enforcement changes the resulting plans before analyst review.

## Reported configuration
- Evaluation incident count: `{incident_count}`
- Export Batch 1: `{batch_one_count}`
- Export Batch 2: `{batch_two_count}`
- Official models: `{models_text}`
- Prompt conditions: `{arms_text}`
- Operational policy tier: `operational`

## Primary outputs
- `artifact_outputs/analysis/protocol_freeze.json`
- `artifact_outputs/analysis/summary.json`
- `artifact_outputs/analysis/analysis_bundle.json`
- `artifact_outputs/analysis/official_llm_analysis_bundle.json`
- `artifact_outputs/analysis/official_evaluation_summary.json`
- `artifact_outputs/analysis/official_pairwise_tests.json`
- `artifact_outputs/analysis/global_artifact_assessment.json`
- `artifact_outputs/analysis/global_artifact_provenance.json`
- `artifact_outputs/analysis/repeat_stability/` (when present)

## Public-bundle scope
- The package includes one terminal evidence manifest, one protocol manifest, the
  complete per-task evidence it binds, and deterministic analysis projections.
- Worker executions and provider transport attempts are operational records;
  neither changes the three scientific repeats per model condition.
- The zero-cost path validates all evidence hashes, recounts aggregates, and
  regenerates deterministic reports without contacting a provider.
- Credentials, authorization headers, private raw exports, and private incident
  lineage are excluded.
- Reviewers may supply their own provider keys and run new experiments with the
  same artifact or with their own incident packages. Such runs are new execution
  lineages unless the incident slice, policy files, model registry, prompt
  template, and verifier manifest all match the reported state.

## Rule-activation interpretation
- The real-corpus evaluation uses the 15 rules marked `operational`.
- Four approval rules are specification-only because the exports contain no authenticated approval record; fixed coverage scenarios test their implementation semantics without contributing to corpus rates.
- Rules whose behavior classes recur weakly in the transfer-check batch remain local operational rules and are also exercised by a fixed specification-coverage set; they do not support cross-batch transfer claims.

## Inspectable prompt-condition surfaces
- `src/soc_llm_policy/pipeline.py`: binds `llm_zero` to `policy_prompt_mode=none` and `llm_policy_prompt` to `policy_prompt_mode=inline_constraints`.
- `src/soc_llm_policy/llm_adapter.py`: defines the shared prompt template, requested JSON format, and inline policy-constraint block.
- `config/models.freeze.yaml`: records the model registry used in the study; the filename is retained for compatibility.

## Planned statistical contrasts
- Three repeats for each incident-model-condition group reduce by majority verdict.
- A paired contrast uses only groups in which all three repeats produced a plan; refusals and parse failures remain in terminal accounting.
- Within each analysis stratum, two contrasts compare prompt conditions within a model and two compare models within a prompt condition.
- No contrast changes model and prompt condition at the same time.
- Holm adjustment covers the complete planned family across the full corpus and both batch-specific sensitivity strata.
- First-repeat, last-repeat, and any-violation reductions are sensitivity checks; they do not replace the majority estimate.

## Reproducibility requirements
- Preserve the packaged anonymized dataset under `artifact_data/`
- Treat `artifact_outputs/analysis/protocol_freeze.json` as the official protocol manifest
- Re-run verification and dataset audit from the package root to confirm integrity before any additional analysis
"""
    (release_root / "EVAL_PROTOCOL.md").write_text(text, encoding="utf-8")


def _write_pre_execution_docs(
    *,
    release_root: Path,
    batch_one_incident_count: int,
    batch_two_incident_count: int,
) -> None:
    freeze = _load_json(
        release_root / "artifact_outputs" / "analysis" / "protocol_freeze.json"
    )
    primary_cell_count = int(freeze["primary_cell_count"])
    diagnostic_task_ceiling = int(freeze["diagnostic_task_ceiling"])
    scientific_task_ceiling = int(freeze["scientific_task_ceiling"])
    diagnostic_studies = freeze["diagnostic_studies"]
    official_models = freeze["official_models"]
    model_list = " and ".join(f"`{model}`" for model in official_models)
    readme = f"""# Model Execution Source

This package is the content-addressed source for a controlled model evaluation. It
is publication-independent and contains no model results.

## Scope
- Export Batch 1: `{batch_one_incident_count}` incidents.
- Export Batch 2: `{batch_two_incident_count}` incidents.
- Total evaluation corpus: `{batch_one_incident_count + batch_two_incident_count}` incidents.
- Reviewed policy: `artifact_data/global/`.
- Model registry: `config/models.freeze.yaml` (compatibility filename).

## Zero-cost validation
```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
bash run.sh validate-public-artifact
bash run.sh policy-graph-validate
bash run.sh rule-coverage
bash provider_run.sh prepare
PLAN=artifact_outputs/provider_state/canonical_campaign/provider_execution_plan.json
bash provider_run.sh execute --plan "$PLAN" --validate-only
bash provider_run.sh status --plan "$PLAN"
bash provider_run.sh dashboard --plan "$PLAN"
```

Open `artifact_outputs/provider_state/canonical_campaign/execution_dashboard.html`
in a browser. This offline page displays a timestamped, read-only snapshot from the
same records as CLI status. Missing records remain unknown, not zero. It neither
starts model calls nor interprets partial scientific outcomes.

The package already contains the only protocol manifest. `prepare` materializes
its declared corpus view, derives a draft plan without changing any scientific
setting, and makes no provider calls. Inspect the {primary_cell_count:,}-cell matrix and cost
estimate before approval.
Only an explicitly approved `execute` command can start model calls. Interrupted
transport failures resume under the same fixed matrix only while their original
attempt allowance remains; valid observations are never replaced.
Extract into a persistent local directory, not `/tmp` or the system temporary
directory. Keep the source archive and back up all execution records.
Before scientific cell 1, one bounded live request must pass for every declared
model profile. A failed readiness gate creates no scientific experiment state.

## Runtime regression tests
The packaged tests cover recovery logging, transport attempts, refusal and parse
handling, and exclusive worker ownership. They use synthetic inputs, mocked
provider responses, and a loopback TCP listener; they make no external model calls
and need no credentials. They are a portable subset of the maintainer test suite.

```bash
.venv/bin/pip install -r requirements-dev.txt
PYTHONPATH=src .venv/bin/python -m pytest -q
```

The protocol contains `{primary_cell_count}` primary cells and
`{diagnostic_task_ceiling}` diagnostic tasks, for a scientific-task ceiling of
`{scientific_task_ceiling}`. Provider readiness and transport-attempt ceilings
are derived explicitly in the draft provider plan.

This source package supports the corpus, policy-selection, and verifier claims
listed in `CLAIM_EVIDENCE_MATRIX.md`. Model-result claims remain pending. The
final artifact must add the completed task records and derived reports while
preserving the protocol manifest and execution-source digest.

## Reviewer paths and resources

Read [the reviewer guide](docs/REVIEWER_GUIDE.md) for the environment, minimal
functional test, expected outputs, CLI and dashboard journeys, security concerns,
and evidence-preservation steps. Python 3.12 and Bash are required. Offline checks
need no GPU or credentials. The declared local model profiles use approximately
39 GB of model weights; the reference execution machine has 64 GB of memory.
Whole-run duration and output storage have not yet been measured.

## Organization and extension

`src/soc_llm_policy/` contains the implementation, `scripts/` the entry points,
`config/` the fixed model settings, `artifact_data/` the redacted inputs, and
`artifact_outputs/` the evidence and derived reports. See
[the architecture and extension guide](docs/EXTENDING.md) for component boundaries,
contributor checks, and requirements for new models, policies, corpora, and metrics.

## Evaluation status

| Dimension | What this source package provides | What still requires evidence |
| --- | --- | --- |
| Availability | A hash-verifiable source archive | Continued availability and archival preservation |
| Functionality | Offline checks, specification scenarios, and runtime regression tests | Successful live readiness and complete model execution |
| Maintainability | Typed boundaries, extension documentation, tests, and code/data terms | Independent reuse |
| Result reproducibility | Claim-to-evidence mapping and deterministic audit commands | Terminal model records and a clean reproduction of reported results |

These are evidence categories, not awarded badges or a technology-readiness rating.

## Security and licensing

Normal model execution sends rendered incident context to the declared endpoints
and may incur charges. Check data-sharing permissions before supplying credentials.
Keep keys outside the distributed package. Preserve outputs on durable storage and
back them up to a separate device or approved protected destination.

Source code is licensed under [MIT](LICENSE). Anonymized incident data may be used
and redistributed for research and reproducibility under [DATA_LICENSE.md](DATA_LICENSE.md):
preserve anonymization and do not attempt reidentification. These data terms are
purpose-restricted, not an unrestricted open-data license. Dependency and model
licenses remain separate. See the reviewer guide before reusing data.
"""
    protocol = f"""# Evaluation Protocol

## Design fixed before provider access
- Corpus: one `{batch_one_incident_count + batch_two_incident_count}`-incident evaluation corpus received in two disjoint export batches (`{batch_one_incident_count}` and `{batch_two_incident_count}`).
- Factors: incident, export batch, model, prompt condition, and repeat.
- Models: {model_list}.
- Prompt conditions: `llm_zero` and `llm_policy_prompt`.
- Repeats: `3`; paired groups reduce by majority verdict.
- Primary plan cells: `{primary_cell_count}`.
- Independent-judge tasks: `{diagnostic_studies["independent_judge"]["task_count_ceiling"]}`.
- Judge-ablation tasks: `{diagnostic_studies["judge_ablation"]["task_count_ceiling"]}`.
- Adversarial tasks: `{diagnostic_studies["adversarial"]["task_count_ceiling"]}`.
- Projection-fidelity tasks: `{diagnostic_studies["projection_fidelity"]["task_count_ceiling"]}`.
- Scientific-task ceiling: `{scientific_task_ceiling}`.

## Planned statistical contrasts
- Three repeats for each incident-model-condition group reduce by majority verdict.
- A paired contrast uses only groups in which all three repeats produced a plan; refusals and parse failures remain in terminal accounting.
- Within each analysis stratum, two contrasts compare prompt conditions within a model and two compare models within a prompt condition.
- No contrast changes model and prompt condition at the same time.
- Holm adjustment covers the complete planned family across the full corpus and both batch-specific sensitivity strata.
- First-repeat, last-repeat, and any-violation reductions are sensitivity checks; they do not replace the majority estimate.

The complete corpus is the model-behavior estimand; acquisition batch is a
mandatory heterogeneity factor.
Policy candidates were derived on Batch 1 and checked for transfer activation
on Batch 2. Batch 2 is not a pristine holdout for the complete construction
pipeline because its vocabulary informed the disclosed normalization audit.

## Evidence status
This source package contains no model observations and supports no empirical
model-behavior claim. `provider_run.sh prepare` consumes the published manifest and
creates its content-addressed matrix without network access. `execute
--validate-only` rehearses every binding without credentials. Normal execution
requires an explicitly approved plan.

## Failure policy
Only declared transport failures that produced no observation permit another
attempt. The attempt limit includes the initial request and does not
change the scientific repeat count.
Connection and transport errors trigger bounded, credential-free TCP probes
before an allowed in-process retry. The protocol fixes the probe timeout,
interval, recovery window, and required consecutive successes. Reachability
does not guarantee API availability or a valid model response. Probe records
are persisted and checked against the provider-attempt ledger. Recovery timeout
stops the worker; it does not consume a new provider attempt. HTTP rate-limit
and server-error responses retain the fixed retry wait. Exhausted attempts
block finalization rather than authorizing replacement observations.
Execution uses a single-host kernel lock that remains held across long waits;
an expired lease cannot evict a live worker. Model calls require persistent
local storage outside temporary directories.
Valid responses, refusals, parse outcomes, and policy violations are immutable.
Any change to corpus, policy, catalog, prompt, model binding, provider-attempt
policy, or analysis creates a new protocol digest.
"""
    (release_root / "README.md").write_text(readme, encoding="utf-8")
    (release_root / "EVAL_PROTOCOL.md").write_text(protocol, encoding="utf-8")


def _write_claim_evidence_matrix(
    *,
    release_root: Path,
    provider_evidence_available: bool,
    bound_evidence_paths: dict[str, Path] | None = None,
) -> None:
    """Write the public claim boundary in machine- and reviewer-readable forms."""

    def evidence_path(filename: str) -> str:
        if bound_evidence_paths and filename in bound_evidence_paths:
            selected = bound_evidence_paths[filename]
            relative = _repo_relative_source(
                selected, release_root, label="Claim evidence"
            )
            if not selected.is_file():
                raise FileNotFoundError(f"Missing bound claim evidence: {filename}")
            return relative.as_posix()
        matches = sorted(release_root.rglob(filename))
        if len(matches) != 1:
            raise ValueError(
                f"Claim evidence {filename} must resolve to exactly one packaged file."
            )
        return matches[0].relative_to(release_root).as_posix()

    model_status = "supported" if provider_evidence_available else "pending"
    audit_validate_command = (
        "bash reproduce.sh validate-public-artifact"
        if provider_evidence_available
        else "bash run.sh validate-public-artifact"
    )
    model_evidence = (
        [
            evidence_path("run_ledger.json"),
            evidence_path("official_evaluation_summary.json"),
        ]
        if provider_evidence_available
        else []
    )
    matrix = {
        "schema_version": 1,
        "artifact_stage": (
            "reported_results"
            if provider_evidence_available
            else "pre_execution_source"
        ),
        "claims": [
            {
                "claim_id": "corpus_scope",
                "claim": "The evaluation corpus contains 400 incidents in two disjoint export batches.",
                "status": "supported",
                "evidence": [
                    evidence_path("corpus_campaign.json"),
                    evidence_path("corpus_partition_audit.json"),
                ],
                "audit_command": audit_validate_command,
            },
            {
                "claim_id": "policy_selection",
                "claim": "The reviewed bundle separates operational rules from specification-only approval gates and records the transfer check.",
                "status": "supported",
                "evidence": [
                    evidence_path("constraints.yaml"),
                    evidence_path("policy_preflight.json"),
                ],
                "audit_command": audit_validate_command,
            },
            {
                "claim_id": "verifier_semantics",
                "claim": "The shipped verifier implements deterministic typed checks and repairs over ordered action lists.",
                "status": "supported",
                "evidence": [
                    evidence_path("verify.py"),
                    evidence_path("protocol_freeze.json"),
                ],
                "audit_command": audit_validate_command,
            },
            {
                "claim_id": "model_results",
                "claim": "The reported model comparisons, violation rates, judge results, adversarial results, and projection-fidelity results follow from one terminal model execution.",
                "status": model_status,
                "evidence": model_evidence,
                "audit_command": (
                    "bash reproduce.sh reproduce-results"
                    if provider_evidence_available
                    else "Provider evidence not present"
                ),
            },
        ],
    }
    analysis_dir = release_root / "artifact_outputs" / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = analysis_dir / "claim_evidence_matrix.json"
    matrix_path.write_text(
        json.dumps(matrix, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    rows = [
        "# Claim-to-Evidence Matrix",
        "",
        "This table states what the package can support at its current stage.",
        "A pending claim must not be reported as a result.",
        "",
        "| Claim | Status | Audit |",
        "| --- | --- | --- |",
    ]
    rows.extend(
        f"| {claim['claim']} | `{claim['status']}` | `{claim['audit_command']}` |"
        for claim in matrix["claims"]
    )
    rows.extend(
        [
            "",
            "The machine-readable matrix and evidence paths are in",
            "`artifact_outputs/analysis/claim_evidence_matrix.json`.",
            "",
        ]
    )
    (release_root / "CLAIM_EVIDENCE_MATRIX.md").write_text(
        "\n".join(rows), encoding="utf-8"
    )


def _write_package_release_readiness(release_root: Path) -> None:
    manifest_path = release_root / "artifact_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Release readiness requires the final manifest.")
    report = audit_release(RepoPaths(repo_root=release_root))
    output_path = (
        release_root / "artifact_outputs" / "analysis" / "release_readiness.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if report.get("status") != "ok":
        issue_codes = sorted(
            str(issue.get("code"))
            for issue in report.get("issues", [])
            if isinstance(issue, dict)
        )
        raise ValueError("Release readiness failed: " + ", ".join(issue_codes))


def _refresh_freeze_input_hashes(
    protocol_freeze_path: Path,
    release_root: Path,
    *,
    strict: bool = True,
) -> None:
    """Recompute input_hashes SHA-256 values to match sanitized artifact files."""
    payload = json.loads(protocol_freeze_path.read_text(encoding="utf-8"))
    input_hashes = payload.get("input_hashes")
    if not isinstance(input_hashes, dict):
        return
    changed = False
    for role, entry in input_hashes.items():
        if not isinstance(entry, dict):
            if strict:
                raise ValueError(f"Protocol freeze input is not an object: {role}.")
            continue
        rel_path = entry.get("path")
        if not isinstance(rel_path, str):
            if strict:
                raise ValueError(f"Protocol freeze input has no path: {role}.")
            continue
        relative_path = Path(rel_path)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            if strict:
                raise ValueError(f"Protocol freeze input path escapes release: {role}.")
            continue
        file_path = release_root / relative_path
        if not file_path.is_file():
            if strict:
                raise FileNotFoundError(
                    f"Protocol freeze input was not packaged: {role} -> {rel_path}"
                )
            continue
        new_hash = _sha256_file(file_path)
        if entry.get("sha256") != new_hash:
            entry["sha256"] = new_hash
            changed = True
    if changed:
        protocol_freeze_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _sanitize_release_surface(
    *,
    release_root: Path,
    dataset_release_id: str,
    protocol_freeze_path: Path | None,
    pre_execution_source: bool,
) -> None:
    _sanitize_global_artifacts(release_root)
    if protocol_freeze_path is not None:
        _sanitize_json_file(protocol_freeze_path, dataset_release_id=dataset_release_id)

    manifest_files = sorted(
        (release_root / "artifact_data").glob("*/INC_*/*/source_manifest.json")
    )
    for manifest_path in manifest_files:
        incident_id = manifest_path.parts[-3]
        _sanitize_json_file(
            manifest_path,
            dataset_release_id=dataset_release_id,
            incident_hint=incident_id,
        )

    for json_path in sorted(release_root.rglob("*.json")):
        if (
            protocol_freeze_path is not None and json_path == protocol_freeze_path
        ) or json_path.name == "source_manifest.json":
            continue
        if "policy_preflight" in json_path.relative_to(release_root).parts:
            continue
        _sanitize_json_file(json_path, dataset_release_id=dataset_release_id)

    for jsonl_path in sorted(release_root.rglob("*.jsonl")):
        _sanitize_jsonl_file(
            jsonl_path,
            dataset_release_id=dataset_release_id,
        )

    for suffix in ("*.md", "*.txt", "*.yaml", "*.yml", "*.sh", "*.toml"):
        for text_path in sorted(release_root.rglob(suffix)):
            if text_path.name in {
                "README.md",
                "EVAL_PROTOCOL.md",
                "ARTIFACT_BOUNDARY.md",
            }:
                continue
            text = text_path.read_text(encoding="utf-8")
            text = _scrub_string(
                text,
                dataset_release_id=dataset_release_id,
                preserve_runtime_names=text_path.suffix == ".md",
            )
            text_path.write_text(text, encoding="utf-8")

    if protocol_freeze_path is not None:
        if pre_execution_source:
            _write_pre_execution_docs(
                release_root=release_root,
                batch_one_incident_count=len(
                    list((release_root / "artifact_data" / "dataset").glob("INC_*"))
                ),
                batch_two_incident_count=len(
                    list((release_root / "artifact_data" / "heldout").glob("INC_*"))
                ),
            )
        else:
            _write_package_readme(release_root)
            _write_package_eval_protocol(
                release_root=release_root, protocol_freeze_path=protocol_freeze_path
            )


def _is_controlled_protocol_freeze(path: Path | None) -> bool:
    return path is not None and _load_json(path).get("schema_version") == (
        PROTOCOL_FREEZE_SCHEMA
    )


def _finalize_packaged_freeze(
    *,
    protocol_freeze_path: Path | None,
    release_root: Path,
) -> None:
    if protocol_freeze_path is None:
        return
    if _is_controlled_protocol_freeze(protocol_freeze_path):
        _bind_packaged_execution_source(
            protocol_freeze_path=protocol_freeze_path,
            release_root=release_root,
        )
        _refresh_freeze_input_hashes(protocol_freeze_path, release_root)
        return
    _refresh_freeze_input_hashes(
        protocol_freeze_path,
        release_root,
        strict=False,
    )


def _write_packaged_global_provenance(
    *,
    release_root: Path,
    incident_ids: list[str],
) -> None:
    with _without_repo_path_overrides():
        with _package_incident_paths(release_root, incident_ids) as paths:
            report = build_global_provenance(
                paths=paths,
                incident_ids=incident_ids,
            )
    write_stable_json(
        release_root
        / "artifact_outputs"
        / "analysis"
        / "global_artifact_provenance.json",
        report,
    )


def _build_manifest(
    *,
    release_root: Path,
    incidents: list[str],
    include_analysis: bool,
    run_id: str | None,
    heldout_incidents: list[str] | None = None,
    artifact_stage: str = "reported_results",
    provider_incidents: list[str] | None = None,
    campaign_evidence_path: str | None = None,
) -> dict[str, Any]:
    files: list[dict[str, str]] = []
    for path in sorted(release_root.rglob("*")):
        if not path.is_file():
            continue
        rel_path = path.relative_to(release_root)
        if any(
            part in _EXCLUDE_DIR_NAMES or part.endswith(_EXCLUDE_DIR_SUFFIXES)
            for part in rel_path.parts
        ):
            continue
        if path.name in _EXCLUDE_FILE_NAMES or path.suffix in _EXCLUDE_FILE_SUFFIXES:
            continue
        if path.name in _MANIFEST_EXCLUDE_FILE_NAMES:
            continue
        if any(
            path.name.endswith(suffix) for suffix in _MANIFEST_EXCLUDE_FILE_SUFFIXES
        ):
            continue
        rel = rel_path.as_posix()
        files.append({"path": rel, "sha256": _sha256_file(path)})
    heldout_incidents = heldout_incidents or []
    provider_incidents = provider_incidents or []
    provider_incident_count = len(provider_incidents)
    total_incident_count = len(incidents) + len(heldout_incidents)
    if provider_incident_count == 0:
        provider_corpus_scope = "none"
    elif provider_incident_count == total_incident_count:
        provider_corpus_scope = "all_incidents"
    else:
        raise ValueError(
            "Provider incident count does not cover the complete campaign corpus."
        )
    batch_one_root = release_root / "artifact_data" / "dataset"
    batch_one_partition: dict[str, Any] = {
        "path": "artifact_data/dataset",
        "incident_count": len(incidents),
    }
    if batch_one_root.is_dir():
        batch_one_partition["corpus_sha256"] = canonical_corpus_sha256(batch_one_root)
    partitions: dict[str, Any] = {"export_batch_1": batch_one_partition}
    if heldout_incidents:
        batch_two_root = release_root / "artifact_data" / "heldout"
        partitions["export_batch_2"] = {
            "path": "artifact_data/heldout",
            "incident_count": len(heldout_incidents),
            "corpus_sha256": canonical_corpus_sha256(batch_two_root),
        }
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "dataset_release_id": _PUBLIC_ARTIFACT_LABEL,
        "incident_count": total_incident_count,
        "measurement_incident_count": provider_incident_count,
        "provider_incident_count": provider_incident_count,
        "provider_corpus_scope": provider_corpus_scope,
        "provider_incidents": provider_incidents,
        "incidents": incidents,
        "heldout_incidents": heldout_incidents,
        "campaign_incidents": sorted([*incidents, *heldout_incidents]),
        "partitions": partitions,
        "include_analysis": include_analysis,
        "run_id": run_id,
        "campaign_evidence_path": campaign_evidence_path,
        "artifact_stage": artifact_stage,
        "file_count": len(files),
        "files": files,
    }


def _provider_incidents_from_freeze(freeze_path: Path | None) -> list[str]:
    if freeze_path is None:
        return []
    payload = _load_json(freeze_path)
    raw_incident_ids = payload.get("incident_ids")
    if raw_incident_ids is None and payload.get("schema_version") is None:
        return []
    if not isinstance(raw_incident_ids, list) or not all(
        isinstance(incident_id, str) and incident_id for incident_id in raw_incident_ids
    ):
        raise ValueError("Protocol freeze has invalid or missing incident_ids.")
    if len(set(raw_incident_ids)) != len(raw_incident_ids):
        raise ValueError("Protocol freeze contains duplicate incident_ids.")
    return raw_incident_ids


def _render_packaged_reviewer_dashboard(release_root: Path) -> None:
    """Render the static reviewer dashboard before manifest hashing."""
    runtime_root = release_root / "analysis_runtime"
    runtime_available = (runtime_root / "runtime_manifest.json").is_file()
    script_path = (
        (runtime_root if runtime_available else release_root)
        / "scripts"
        / "render_audit_dashboard.py"
    )
    if not script_path.exists():
        return
    spec = importlib.util.spec_from_file_location(
        "socpilot_packaged_dashboard_renderer",
        script_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load dashboard renderer: {script_path}")
    module = importlib.util.module_from_spec(spec)
    # Executing the packaged script writes __pycache__ into the release tree,
    # after _copy_tree has already filtered it, so the package would ship
    # bytecode and the release audit would flag a transient directory.
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous

    analysis_dir = release_root / "artifact_outputs" / "analysis"
    # The rule table is driven from the shipped policy file, which _copy_dataset
    # has already placed in the release tree, so the packaged dashboard binds to
    # the same rule set the packaged verifier loads.
    constraints_path = release_root / "artifact_data" / "global" / "constraints.yaml"
    freeze = _load_json(analysis_dir / "protocol_freeze.json")
    input_hashes = freeze.get("input_hashes")
    preflight_dir = None
    if isinstance(input_hashes, dict):
        preflight = input_hashes.get("policy_preflight_manifest")
        if isinstance(preflight, dict) and isinstance(preflight.get("path"), str):
            preflight_path = release_root / preflight["path"]
            preflight_dir = preflight_path.parent
    data = module._load_dashboard_data(analysis_dir, preflight_dir)
    render_options = {"artifact_root": release_root}
    if runtime_available:
        render_options["runner"] = "reproduce.sh"
    landing_html = module._render_html(data, constraints_path, **render_options)
    dashboard_html = module._render_html(
        data,
        constraints_path,
        artifact_prefix="../../",
        command_prefix="../../",
        **render_options,
    )
    dashboard_path = release_root / "artifact_outputs" / "dashboard" / "index.html"
    landing_path = release_root / "REVIEWER_DASHBOARD.html"
    dashboard_path.parent.mkdir(parents=True, exist_ok=True)
    dashboard_path.write_text(dashboard_html, encoding="utf-8")
    landing_path.write_text(landing_html, encoding="utf-8")
    preview_path = release_root / "docs/img/reviewer_dashboard.png"
    if preview_path.exists():
        verify_preview(
            landing_path, preview_path, preview_path.with_suffix(".preview.json")
        )


def _resolve_package_context(
    *, paths: RepoPaths, options: PackageOptions
) -> _PackageContext:
    campaign: CampaignEvidence | None = None
    evidence: ExecutionEvidence | None = None
    if options.campaign_evidence_path is not None:
        campaign = load_campaign_evidence(
            repo_root=paths.repo_root,
            manifest_path=options.campaign_evidence_path,
        )
        evidence = campaign.primary_evidence
        freeze_src = campaign.protocol_freeze_path
    else:
        freeze_src = _resolve_protocol_freeze_source(
            paths,
            options.protocol_freeze_path,
        )
        if (
            options.include_analysis
            and _load_json(freeze_src).get("schema_version") == PROTOCOL_FREEZE_SCHEMA
        ):
            raise ValueError("Provider analysis requires --campaign-evidence-json.")
        _require_packageable_freeze(
            freeze_src,
            allow_pre_execution=options.pre_execution_source,
        )
    _validate_preflight_reports(freeze_src.parent, freeze_src, paths.repo_root)

    heldout_dir = options.heldout_dir
    if heldout_dir is None:
        default_heldout = paths.dataset_dir / "incidents_heldout"
        if default_heldout.is_dir():
            heldout_dir = default_heldout
    heldout_incidents = _heldout_incident_ids(heldout_dir)
    if evidence is not None and not heldout_incidents:
        raise ValueError(
            "An evidence-bound package must include both declared export batches."
        )
    if evidence is not None:
        _validate_evidence_release_binding(
            paths=paths,
            options=options,
            evidence=evidence,
            heldout_dir=heldout_dir,
            heldout_incidents=heldout_incidents,
        )
    if campaign is not None and options.run_id not in (None, campaign.campaign_id):
        raise ValueError("Package run_id differs from the terminal campaign evidence.")
    return _PackageContext(
        campaign=campaign,
        evidence=evidence,
        freeze_src=freeze_src,
        heldout_dir=heldout_dir,
        heldout_incidents=heldout_incidents,
        provider_incidents=_provider_incidents_from_freeze(freeze_src),
    )


def _review_cell_count(value: Any) -> int:
    if type(value) is not int or value < 1:
        raise ValueError("Offline review requires a positive integer cell count.")
    return value


def _offline_verdict_readme(release_root: Path) -> str:
    manifest_path = (
        release_root / "artifact_outputs/analysis/official_runs_manifest.json"
    )
    if not manifest_path.is_file():
        return ""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_id = manifest["reported_evaluation_run_id"]
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("Invalid reported run identity for offline review.")
    completed = _review_cell_count(
        manifest["execution_accounting"]["llm_trajectories_completed"]
    )
    commands = [
        f"bash reproduce.sh offline-verdict-audit --run-id {shlex.quote(run_id)} "
        f"--expected-cells {completed}"
    ]
    adversarial_run = f"{run_id}__adversarial"
    experiment = Path("artifact_outputs/experiments") / adversarial_run
    summary_path = release_root / experiment / "summary.json"
    scenario_root = experiment / "scenario_input/dataset"
    if summary_path.is_file() and (release_root / scenario_root).is_dir():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        adversarial_cells = _review_cell_count(summary["run_count"])
        commands.append(
            f"bash reproduce.sh offline-verdict-audit --run-id {shlex.quote(adversarial_run)} "
            f"--expected-cells {adversarial_cells} "
            f"--incident-input-root {shlex.quote(scenario_root.as_posix())}"
        )
    return (
        "\n## Offline decision replay\n\n"
        "Recompute the preserved decisions without calling a provider. Expected "
        "counts cover preserved decisions, not planned cells. Missing outputs "
        "remain in failure accounting rather than being replaced. The "
        "adversarial study uses its own frozen scenario inputs, not the primary "
        "corpus. These checks do not certify publication safety or operational effects.\n\n"
        "```bash\n" + "\n".join(commands) + "\n```\n"
    )


def _copy_analysis_runtime(release_root: Path) -> None:
    maintenance_root = Path(__file__).resolve().parents[2]
    runtime_root = release_root / "analysis_runtime"
    for directory in ("src", "scripts"):
        _copy_tree(
            maintenance_root / directory,
            runtime_root / directory,
            package_rel_root=directory,
        )
    for renderer in ("render_measurement_tex.py", "render_judge_ablation_tex.py"):
        relative = Path("scripts") / renderer
        _copy_file(maintenance_root / relative, runtime_root / relative)
    for filename in ("requirements.txt", "requirements-dev.txt", "pyproject.toml"):
        _copy_file(maintenance_root / filename, runtime_root / filename)
    _copy_file(
        maintenance_root / "scripts/public_artifact_runner.sh",
        release_root / "reproduce.sh",
    )
    (release_root / "reproduce.sh").chmod(0o755)
    files = [
        {
            "path": path.relative_to(runtime_root).as_posix(),
            "sha256": _sha256_file(path),
        }
        for path in sorted(runtime_root.rglob("*"))
        if path.is_file()
    ]
    (runtime_root / "runtime_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "purpose": "offline_reanalysis_not_collection_source",
                "files": files,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    readme = release_root / "README.md"
    content = readme.read_text(encoding="utf-8")
    content = content.replace("bash run.sh", "bash reproduce.sh")
    content = content.replace(
        "./run.sh validate-public-artifact",
        "bash reproduce.sh validate-public-artifact",
    )
    runtime_note = (
        "\n## Collection source and offline reanalysis\n\n"
        "The root `src/`, `scripts/`, `run.sh`, and `provider_run.sh` retain the "
        "manifest-bound collection implementation. Use `bash reproduce.sh` for "
        "offline review. It selects the separately hashed `analysis_runtime/` "
        "maintenance code; this code did not collect the model responses. "
        "Its dependency declarations are in `analysis_runtime/requirements.txt` "
        "and `analysis_runtime/requirements-dev.txt`. The package manifest binds "
        "both implementations. No collection inputs or observations are replaced.\n"
    )
    if "\n## Collection source and offline reanalysis\n" not in content:
        content += runtime_note
    if "\n## Offline decision replay\n" not in content:
        content += _offline_verdict_readme(release_root)
    if "\n## Repair stability and residual conflicts\n" not in content:
        content += (
            "\n## Repair stability and residual conflicts\n\n"
            "Stable does not mean compliant. The offline audit separately counts "
            "post-repair violations and "
            "`active_requirement_prohibition_conflict_cells`: contexts where "
            "active rules both require and prohibit the same action. This count "
            "screens direct conflicts, not general policy satisfiability. "
            "Such conflicts require policy review; retrying the same repair "
            "does not resolve them. Historical decisions and failures remain "
            "part of the preserved evidence.\n"
        )
    if "\n## Dashboard preview maintenance\n" not in content:
        content += (
            "\n## Dashboard preview maintenance\n\n"
            "Opening the shipped dashboard requires no browser automation. "
            "To regenerate its documentation preview, provide an installed "
            "headless-capable Chrome/Chromium and the reviewer Python environment:\n\n"
            "```bash\n"
            "PYTHON=.venv/bin/python CHROME_BIN=/path/to/chromium "
            "bash analysis_runtime/scripts/render_dashboard_screenshot.sh "
            "REVIEWER_DASHBOARD.html /path/to/preview.png\n"
            "```\n\n"
            "Write outside the verified package. The adjacent `.preview.json` "
            "binds the HTML and PNG hashes; a preview is not experiment evidence. "
            "The maintained verifier checks the shipped preview binding as well "
            "as manifest integrity.\n"
        )
    if "\n## Explicit preview backend selection\n" not in content:
        content += (
            "\n## Explicit preview backend selection\n\n"
            "The default preview engine is `chrome-cli`. If the installed browser "
            "cannot finish that capture, select the optional Playwright backend "
            "explicitly. It requires Node.js and the `playwright` module available "
            "to Node; use `NODE_PATH` for an existing installation outside this "
            "package. This command uses the browser you supply, downloads no browser, "
            "makes no provider calls, and does not silently switch engines after a "
            "failure:\n\n"
            "```bash\n"
            "PREVIEW_ENGINE=playwright PYTHON=.venv/bin/python "
            "CHROME_BIN=/path/to/chromium "
            "bash analysis_runtime/scripts/render_dashboard_screenshot.sh "
            "REVIEWER_DASHBOARD.html /path/to/preview.png\n"
            "```\n\n"
            "Write the preview outside this verified package. A missing dependency, "
            "browser error or timeout fails the capture without a success receipt. "
            "Opening the existing HTML needs neither Node.js nor Playwright.\n"
        )
    readme.write_text(content, encoding="utf-8")


def _populate_terminal_campaign_release(
    *,
    paths: RepoPaths,
    options: PackageOptions,
    context: _PackageContext,
    release_root: Path,
) -> Path:
    campaign = context.campaign
    evidence = context.evidence
    if campaign is None or evidence is None:
        raise ValueError("Terminal campaign packaging requires campaign evidence.")
    packaged_campaign_path = _copy_terminal_campaign_evidence(
        campaign,
        repo_root=paths.repo_root,
        release_root=release_root,
    )
    copied_files = _copy_execution_evidence(evidence, release_root)
    derived_files = _write_derived_public_reports(
        release_root=release_root,
        evidence=evidence,
    )
    protocol_freeze_dst = release_root / context.freeze_src.relative_to(paths.repo_root)
    if not protocol_freeze_dst.is_file() or _sha256_file(
        protocol_freeze_dst
    ) != _sha256_file(context.freeze_src):
        raise ValueError("Packaged campaign changed the protocol freeze.")
    _write_release_boundary(release_root, includes_terminal_campaign=True)
    _write_package_readme(release_root)
    _write_package_eval_protocol(
        release_root=release_root,
        protocol_freeze_path=protocol_freeze_dst,
    )
    _write_packaged_global_provenance(
        release_root=release_root,
        incident_ids=context.provider_incidents,
    )
    derived_files.update(
        _write_package_analysis_reports(
            release_root=release_root,
            incidents=context.provider_incidents,
        )
    )
    policy_audit_path = (
        release_root / "artifact_outputs/analysis/policy_surface_audit.json"
    )
    policy_audit = build_policy_surface_report(
        AuditPaths(
            dataset_dir=release_root / "artifact_data/dataset",
            additional_dataset_dirs=(release_root / "artifact_data/heldout",),
            constraints_path=release_root / "artifact_data/global/constraints.yaml",
            official_bundle_path=copied_files["analysis_bundle"],
            repeat_assignments_path=None,
            output_json_path=policy_audit_path,
            output_markdown_path=policy_audit_path.with_suffix(".md"),
        )
    )
    policy_audit_path.write_text(
        json.dumps(policy_audit, indent=2) + "\n", encoding="utf-8"
    )
    derived_files["policy_surface_audit"] = policy_audit_path
    _write_public_evidence_provenance(
        release_root=release_root,
        campaign=campaign,
        copied_files=copied_files,
        derived_files=derived_files,
    )
    claim_paths = {
        Path(entry["path"]).name: release_root / entry["path"]
        for entry in _load_json(protocol_freeze_dst)["input_hashes"].values()
    }
    claim_paths.update(
        {
            "run_ledger.json": copied_files["ledger"],
            "official_evaluation_summary.json": copied_files["summary"],
            "protocol_freeze.json": protocol_freeze_dst,
        }
    )
    _write_claim_evidence_matrix(
        release_root=release_root,
        provider_evidence_available=True,
        bound_evidence_paths=claim_paths,
    )
    publish_tracebacks(release_root, packaged_campaign_path)
    _copy_analysis_runtime(release_root)
    _render_packaged_reviewer_dashboard(release_root)
    return packaged_campaign_path


def _populate_source_or_legacy_release(
    *,
    paths: RepoPaths,
    options: PackageOptions,
    context: _PackageContext,
    release_root: Path,
) -> None:
    if options.include_analysis:
        _copy_analysis(paths, release_root, options.run_id)
        _copy_experiment_runs(paths, release_root, options.experiment_runs)
    protocol_freeze_dst = _copy_protocol_freeze(
        paths,
        release_root,
        context.freeze_src,
    )
    packaged_preflight_dir = _copy_policy_preflight(
        paths,
        release_root,
        context.freeze_src,
        options.policy_preflight_manifest,
        source_only=options.pre_execution_source,
    )
    _bind_packaged_policy_preflight(
        protocol_freeze_dst,
        packaged_preflight_dir,
        release_root,
    )
    _write_release_boundary(release_root, includes_terminal_campaign=False)
    _sanitize_release_surface(
        release_root=release_root,
        dataset_release_id=options.dataset_release_id,
        protocol_freeze_path=protocol_freeze_dst,
        pre_execution_source=options.pre_execution_source,
    )
    _copy_and_bind_packaged_campaign(
        paths=paths,
        release_root=release_root,
        freeze_src=context.freeze_src,
        protocol_freeze_path=protocol_freeze_dst,
        required=options.pre_execution_source,
    )
    _rebind_packaged_adversarial_bank(
        protocol_freeze_path=protocol_freeze_dst,
        release_root=release_root,
    )
    _finalize_packaged_freeze(
        protocol_freeze_path=protocol_freeze_dst,
        release_root=release_root,
    )
    if options.pre_execution_source:
        _write_claim_evidence_matrix(
            release_root=release_root,
            provider_evidence_available=False,
        )
    if not options.pre_execution_source:
        _write_packaged_global_provenance(
            release_root=release_root,
            incident_ids=(
                context.provider_incidents
                or sorted([*options.incidents, *context.heldout_incidents])
            ),
        )
        _render_packaged_reviewer_dashboard(release_root)


def _packaged_redaction_vocabulary(
    release_root: Path, alias_path: Path | None = None
) -> RedactionVocabulary:
    global_root = release_root / "artifact_data" / "global"
    aliases = read_public_aliases(alias_path) if alias_path is not None else ()
    return RedactionVocabulary.from_policy(
        global_root / "constraints.yaml",
        global_root / "telemetry_normalization.yaml",
        aliases,
    )


def _require_private_alias_path(
    repo_root: Path,
    heldout_dir: Path | None,
    alias_path: Path | None,
) -> None:
    if alias_path is None:
        return
    source_roots = (repo_root, heldout_dir)
    for source_root in source_roots:
        if source_root is None:
            continue
        if alias_path.absolute().is_relative_to(source_root.absolute()) or (
            alias_path.resolve().is_relative_to(source_root.resolve())
        ):
            raise ValueError(
                "Private publication maps must be outside copied source trees"
            )


def _require_residue_free_release(
    release_root: Path, *, raw_dir: Path | None = None
) -> None:
    """Refuse a package whose data or outputs still carry tenant residue."""
    findings = audit_release_residue(release_root)
    if raw_dir is not None:
        findings.extend(audit_raw_private_ip_overlap(raw_dir, release_root))
    if not findings:
        return
    examples = ", ".join(sorted({finding.kind.value for finding in findings}))
    raise ValueError(
        f"Public packaging refused: {len(findings)} residue finding(s) after redaction. {examples}"
    )


def _require_verified_terminal_release(release_root: Path) -> None:
    report = verify_artifact_package(release_root)
    if report["status"] != "ok":
        issues = sorted(
            f"{issue['code']}: {issue.get('message', '')}" for issue in report["issues"]
        )
        raise ValueError("Artifact verification failed: " + "; ".join(issues))

    runtime_source = release_root / "analysis_runtime" / "src"
    if not runtime_source.is_dir():
        raise FileNotFoundError(
            f"Terminal analysis runtime is missing: {runtime_source}"
        )
    environment = {**os.environ, "PYTHONPATH": str(runtime_source)}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "soc_llm_policy.artifact_verify",
            "--package-root",
            str(release_root),
        ],
        cwd=release_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Packaged artifact verifier failed (exit {result.returncode}): "
            f"{result.stderr.strip()}"
        )


def build_package(*, paths: RepoPaths, options: PackageOptions) -> Path:
    for private_path in (
        options.public_category_aliases_path,
        options.private_correlation_map_path,
    ):
        _require_private_alias_path(paths.repo_root, options.heldout_dir, private_path)
    _validate_pre_execution_package_options(options)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    final_release_root = (
        options.output_dir / f"artifact_{options.dataset_release_id}_{ts}"
    )
    staging_root = options.output_dir / f".{final_release_root.name}.staging"
    if final_release_root.exists() or staging_root.exists():
        raise FileExistsError(f"Release directory already exists: {final_release_root}")

    context = _resolve_package_context(paths=paths, options=options)

    release_root = staging_root
    release_root.mkdir(parents=True, exist_ok=False)
    try:
        preserve_execution_source = context.campaign is not None
        _copy_project_files(
            paths,
            release_root,
            preserve_execution_source=preserve_execution_source,
            include_dashboard_preview=not options.pre_execution_source,
        )
        _copy_dataset(
            paths,
            release_root,
            options.incidents,
            context.heldout_dir,
            preserve_execution_source=preserve_execution_source,
        )
        packaged_campaign_path: Path | None = None
        if context.campaign is not None:
            packaged_campaign_path = _populate_terminal_campaign_release(
                paths=paths,
                options=options,
                context=context,
                release_root=release_root,
            )
        else:
            _populate_source_or_legacy_release(
                paths=paths,
                options=options,
                context=context,
                release_root=release_root,
            )

        vocabulary = _packaged_redaction_vocabulary(
            release_root, options.public_category_aliases_path
        )
        raw_dir = options.raw_dir
        if raw_dir is None:
            local_raw_dir = paths.repo_root / "incoming" / "raw"
            raw_dir = local_raw_dir if local_raw_dir.is_dir() else None
        projection = new_private_projection(
            read_incident_times(release_root / "artifact_data"),
            ip_aliases=(
                collision_free_ip_aliases(release_root, raw_dir)
                if raw_dir is not None
                else ()
            ),
        )
        record = redact_release_trees(
            release_root, (*DATA_TREES, *OUTPUT_TREES), vocabulary, projection
        )
        record = project_loose_files_into_record(
            release_root,
            record,
            projection,
            excluded_trees=(*DATA_TREES, *OUTPUT_TREES),
        )
        write_record(release_root, record)
        if options.private_correlation_map_path is not None:
            write_private_map(projection, options.private_correlation_map_path)
        _require_residue_free_release(release_root, raw_dir=raw_dir)

        def published_ids(incident_ids: list[str] | None) -> list[str] | None:
            if incident_ids is None:
                return None
            return [
                projection.project_text(incident_id) for incident_id in incident_ids
            ]

        manifest = _build_manifest(
            release_root=release_root,
            incidents=published_ids(options.incidents),
            include_analysis=options.include_analysis,
            run_id=(
                context.campaign.campaign_id
                if context.campaign is not None
                else options.run_id
            ),
            heldout_incidents=published_ids(context.heldout_incidents),
            artifact_stage=(
                "pre_execution_source"
                if options.pre_execution_source
                else "reported_results"
            ),
            provider_incidents=published_ids(context.provider_incidents),
            campaign_evidence_path=(
                packaged_campaign_path.relative_to(release_root).as_posix()
                if packaged_campaign_path is not None
                else None
            ),
        )
        manifest_path = release_root / "artifact_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _write_package_release_readiness(release_root)
        if packaged_campaign_path is not None:
            _require_verified_terminal_release(release_root)
        release_root.rename(final_release_root)
        return final_release_root
    except Exception:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soc_llm_policy.artifact_package",
        description="Build a publishable artifact package with anonymized inputs.",
    )
    parser.add_argument("--repo-root", default=None)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        help="Authorized raw corpus for a private source-overlap audit.",
    )
    parser.add_argument("--dataset-release-id", required=True)
    parser.add_argument("--incidents", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--include-analysis", action="store_true")
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--public-category-aliases-json",
        default=None,
        help="Private source-to-public category aliases; never copied into the package.",
    )
    parser.add_argument(
        "--private-correlation-map-json",
        default=None,
        help="Where to write the private incident-ID and time crosswalk; never packaged.",
    )
    parser.add_argument(
        "--heldout-dir",
        default=None,
        help=(
            "Second acquisition batch. Defaults to dataset/incidents_heldout "
            "when that directory exists."
        ),
    )
    parser.add_argument(
        "--experiment-run",
        action="append",
        default=[],
        dest="experiment_runs",
        help=(
            "Legacy deterministic run to include in a non-provider analysis "
            "package. Provider campaigns use --campaign-evidence-json."
        ),
    )
    parser.add_argument(
        "--protocol-freeze-json",
        default=None,
        help="Explicit protocol-freeze manifest to package.",
    )
    parser.add_argument(
        "--policy-preflight-manifest",
        default=None,
        help=(
            "Policy-derivation evidence to package when the freeze predates the "
            "preflight binding. Its predicate digest must match the fixed policy."
        ),
    )
    parser.add_argument(
        "--campaign-evidence-json",
        default=None,
        help="Sole terminal campaign-evidence manifest for provider results.",
    )
    parser.add_argument(
        "--output-dir",
        default="releases",
        help="Directory where artifact package will be created.",
    )
    parser.add_argument(
        "--print-path-only",
        action="store_true",
        help="Print only the created package path (for scripting).",
    )
    parser.add_argument(
        "--pre-execution-source",
        action="store_true",
        help=(
            "Build a source-only campaign package with no provider evidence or "
            "published report values."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.all and not args.incidents:
        raise SystemExit("Provide --incidents or --all.")

    repo_root = resolve_repo_root(args.repo_root)
    paths = RepoPaths(repo_root=repo_root)
    incidents = _resolve_incidents(paths, args.incidents, args.all)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    protocol_freeze_path = None
    if args.protocol_freeze_json:
        protocol_freeze_path = Path(args.protocol_freeze_json).expanduser()
        if not protocol_freeze_path.is_absolute():
            protocol_freeze_path = repo_root / protocol_freeze_path
    policy_preflight_manifest = None
    if args.policy_preflight_manifest:
        policy_preflight_manifest = Path(args.policy_preflight_manifest).expanduser()
        if not policy_preflight_manifest.is_absolute():
            policy_preflight_manifest = repo_root / policy_preflight_manifest
    campaign_evidence_path = None
    if args.campaign_evidence_json:
        campaign_evidence_path = Path(args.campaign_evidence_json).expanduser()
        if not campaign_evidence_path.is_absolute():
            campaign_evidence_path = repo_root / campaign_evidence_path

    release_root = build_package(
        paths=paths,
        options=PackageOptions(
            dataset_release_id=str(args.dataset_release_id).strip(),
            incidents=incidents,
            include_analysis=bool(args.include_analysis),
            run_id=(str(args.run_id).strip() or None)
            if args.run_id is not None
            else None,
            output_dir=output_dir,
            protocol_freeze_path=protocol_freeze_path,
            campaign_evidence_path=campaign_evidence_path,
            policy_preflight_manifest=policy_preflight_manifest,
            experiment_runs=tuple(args.experiment_runs),
            heldout_dir=(
                resolve_repo_relative_path(args.heldout_dir, repo_root)
                if args.heldout_dir
                else None
            ),
            pre_execution_source=bool(args.pre_execution_source),
            public_category_aliases_path=(
                resolve_repo_relative_path(args.public_category_aliases_json, repo_root)
                if args.public_category_aliases_json
                else None
            ),
            private_correlation_map_path=(
                Path(args.private_correlation_map_json).expanduser().absolute()
                if args.private_correlation_map_json
                else None
            ),
            raw_dir=args.raw_dir.expanduser().resolve() if args.raw_dir else None,
        ),
    )
    if args.print_path_only:
        print(str(release_root))
    else:
        print(f"Artifact package created at: {release_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
