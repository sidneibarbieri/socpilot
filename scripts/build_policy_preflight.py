#!/usr/bin/env python3
"""Build an immutable, zero-cost policy-and-corpus preflight state."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from soc_llm_policy.io import parse_rules, read_yaml_list
from soc_llm_policy.paths import resolve_repo_root
from soc_llm_policy.policy_predicates import predicate_digest
from soc_llm_policy.policy_review import validate_policy_review
from soc_llm_policy.preflight_evidence import (
    sha256_file,
    validate_policy_preflight_reports,
)


@dataclass(frozen=True)
class PreflightInputs:
    repo_root: Path
    bundle_dir: Path
    derivation_dir: Path
    heldout_dir: Path
    raw_dir: Path
    output_dir: Path


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def _run(repo_root: Path, *arguments: str) -> None:
    subprocess.run(
        [sys.executable, *arguments],
        cwd=repo_root,
        check=True,
    )


def _bundle_constraints_sha256(bundle_manifest: dict[str, Any]) -> str:
    files = bundle_manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("Policy bundle manifest is missing file digests.")
    for entry in files:
        if isinstance(entry, dict) and entry.get("filename") == "constraints.yaml":
            value = entry.get("sha256")
            if isinstance(value, str):
                return value
    raise ValueError("Policy bundle manifest is missing constraints.yaml.")


def _normalization_sensitivity(
    normalization_report: dict[str, Any],
    review_status: str,
) -> dict[str, Any]:
    all_survive = normalization_report.get("leave_one_out_all_survive")
    minimum = normalization_report.get("leave_one_out_min_heldout")
    if not isinstance(all_survive, bool):
        raise ValueError("Normalization ablation is missing leave-one-out status.")
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 0:
        raise ValueError("Normalization ablation has an invalid held-out minimum.")
    expected_status = "robust" if all_survive else "acknowledged_sensitive"
    if review_status != expected_status:
        raise ValueError(
            "Policy review normalization status does not match the ablation result."
        )
    return {
        "leave_one_out_all_survive": all_survive,
        "leave_one_out_min_heldout": minimum,
        "review_status": review_status,
    }


def build_preflight(inputs: PreflightInputs) -> Path:
    repo_root = inputs.repo_root
    bundle_dir = inputs.bundle_dir
    derivation_dir = inputs.derivation_dir
    heldout_dir = inputs.heldout_dir
    raw_dir = inputs.raw_dir
    output_dir = inputs.output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Preflight output directory must be new or empty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    action_catalog = bundle_dir / "action_catalog.yaml"
    constraints = bundle_dir / "constraints.yaml"
    review_record = bundle_dir / "review_record.yaml"
    normalization = bundle_dir / "telemetry_normalization.yaml"
    review = validate_policy_review(review_record, constraints, action_catalog)
    rules = parse_rules(read_yaml_list(constraints))
    operational_rule_count = sum(
        rule.evidence_tier == "operational" for rule in rules
    )
    specification_rule_count = sum(
        rule.evidence_tier == "specification" for rule in rules
    )
    operational_approval_rule_count = sum(
        rule.evidence_tier == "operational" and rule.type == "require_approval"
        for rule in rules
    )
    if (operational_rule_count, specification_rule_count) != (15, 4):
        raise ValueError(
            "The campaign requires exactly 15 operational and 4 specification rules."
        )
    if operational_approval_rule_count != 0:
        raise ValueError(
            "Approval rules require authenticated records and cannot enter the "
            "operational campaign."
        )

    partition_json = output_dir / "corpus_partition_audit.json"
    activation_json = output_dir / "heldout_activation.json"
    normalization_json = output_dir / "normalization_ablation.json"
    bundle_json = output_dir / "policy_bundle_manifest.json"

    _run(
        repo_root,
        "scripts/audit_corpus_partitions.py",
        "--repo-root",
        str(repo_root),
        "--derivation-dir",
        str(derivation_dir.relative_to(repo_root)),
        "--heldout-dir",
        str(heldout_dir.relative_to(repo_root)),
        "--raw-dir",
        str(raw_dir.relative_to(repo_root)),
        "--output-json",
        str(partition_json),
        "--output-md",
        str(output_dir / "corpus_partition_audit.md"),
        "--macros-output",
        str(output_dir / "corpus_macros.tex"),
    )
    _run(
        repo_root,
        "scripts/validate_heldout_activation.py",
        "--repo-root",
        str(repo_root),
        "--derivation-dir",
        str(derivation_dir.relative_to(repo_root)),
        "--heldout-dir",
        str(heldout_dir.relative_to(repo_root)),
        "--constraints",
        str(constraints.relative_to(repo_root)),
        "--output-json",
        str(activation_json),
        "--output-md",
        str(output_dir / "heldout_activation.md"),
    )
    _run(
        repo_root,
        "scripts/run_normalization_ablation.py",
        "--repo-root",
        str(repo_root),
        "--derivation-dir",
        str(derivation_dir.relative_to(repo_root)),
        "--heldout-dir",
        str(heldout_dir.relative_to(repo_root)),
        "--constraints",
        str(constraints.relative_to(repo_root)),
        "--normalization-yaml",
        str(normalization.relative_to(repo_root)),
        "--output-json",
        str(normalization_json),
        "--macros-output",
        str(output_dir / "normalization_ablation_macros.tex"),
    )

    partition = _read_json(partition_json)
    activation = _read_json(activation_json)
    normalization_report = _read_json(normalization_json)
    evidence = validate_policy_preflight_reports(
        heldout_activation=activation,
        partition_audit=partition,
        normalization_ablation=normalization_report,
        expected_predicate_sha256=predicate_digest(constraints),
    )
    normalization_sensitivity = _normalization_sensitivity(
        normalization_report,
        (
            review.normalization_sensitivity.leave_one_out_status
            if review.normalization_sensitivity is not None
            else ""
        ),
    )
    _run(
        repo_root,
        "scripts/build_policy_bundle_manifest.py",
        "--repo-root",
        str(repo_root),
        "--bundle-dir",
        str(bundle_dir.relative_to(repo_root)),
        "--derived-from-corpus-sha256",
        evidence.derivation_corpus_sha256,
        "--heldout-validation-sha256",
        sha256_file(activation_json),
        "--output-json",
        str(bundle_json),
    )
    _run(
        repo_root,
        "scripts/render_policy_derivation_tex.py",
        "--report-json",
        str(activation_json),
        "--normalization-yaml",
        str(normalization.relative_to(repo_root)),
        "--output-table",
        str(output_dir / "policy_derivation_table.tex"),
        "--output-macros",
        str(output_dir / "policy_derivation_macros.tex"),
    )

    bundle = _read_json(bundle_json)
    if _bundle_constraints_sha256(bundle) != evidence.constraints_sha256:
        raise ValueError("Policy bundle manifest does not match preflight constraints.")
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "constitutes_model_evidence": False,
        "state": "policy_and_corpus_preflight",
        "policy_digest": bundle.get("policy_digest"),
        "constraints_sha256": evidence.constraints_sha256,
        "derivation_corpus_sha256": evidence.derivation_corpus_sha256,
        "heldout_corpus_sha256": evidence.heldout_corpus_sha256,
        "derivation_incident_count": evidence.derivation_incident_count,
        "heldout_incident_count": evidence.heldout_incident_count,
        "raw_export_count": evidence.raw_export_count,
        "policy_evidence_tier": "operational",
        "policy_rule_counts": {
            "operational": operational_rule_count,
            "specification": specification_rule_count,
        },
        "normalization_sensitivity": normalization_sensitivity,
        "reports": {
            name: {"path": path.name, "sha256": sha256_file(path)}
            for name, path in {
                "corpus_partition_audit": partition_json,
                "heldout_activation": activation_json,
                "normalization_ablation": normalization_json,
                "policy_bundle": bundle_json,
            }.items()
        },
        "checks": {
            "policy_review_valid": True,
            "policy_tiers_valid": True,
            "raw_exports_map_bijectively": True,
            "preflight_reports_share_policy_and_corpora": True,
        },
    }
    manifest_path = output_dir / "policy_preflight.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument(
        "--bundle-dir", default="policy"
    )
    parser.add_argument("--derivation-dir", default="dataset/incidents")
    parser.add_argument("--heldout-dir", default="dataset/incidents_heldout")
    parser.add_argument("--raw-dir", default="incoming/raw")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    repo_root = resolve_repo_root(args.repo_root)
    manifest_path = build_preflight(
        PreflightInputs(
            repo_root=repo_root,
            bundle_dir=(repo_root / args.bundle_dir).resolve(),
            derivation_dir=(repo_root / args.derivation_dir).resolve(),
            heldout_dir=(repo_root / args.heldout_dir).resolve(),
            raw_dir=(repo_root / args.raw_dir).resolve(),
            output_dir=(repo_root / args.output_dir).resolve(),
        )
    )
    print(f"Wrote policy preflight manifest: {manifest_path}")


if __name__ == "__main__":
    main()
