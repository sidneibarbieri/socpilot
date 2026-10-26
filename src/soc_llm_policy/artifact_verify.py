from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import yaml

from soc_llm_policy.campaign_evidence import load_campaign_evidence
from soc_llm_policy.corpus_campaign import validate_campaign_sources
from soc_llm_policy.execution_source import validate_execution_source_manifest
from soc_llm_policy.global_provenance import provenance_consistency_issues
from soc_llm_policy.paths import RepoPaths
from soc_llm_policy.protocol_freeze import PROTOCOL_FREEZE_SCHEMA

_FORBIDDEN_PATH_PARTS = {
    ".env",
    "raw",
    "redacted",
    "private",
}
_LOCAL_TRANSIENT_DIR_NAMES = {
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}
_LOCAL_TRANSIENT_SUFFIXES = {".pyc"}
_LOCAL_TRANSIENT_FILE_NAMES = {"release_candidate_checklist.json"}
_LOCAL_TRANSIENT_FILE_SUFFIXES = {"_recheck.json", "_recheck.md"}


def _absolute_path_field(field: str) -> str:
    """A JSON field whose value starts at the filesystem root."""
    return '"' + field + '"' + ': "/'


# Assembled from fragments, never written whole, because this module ships inside
# the release and the scan below reads it. A marker spelled out here would make
# the verifier report itself and hide every real finding behind its own noise.
_FORBIDDEN_TEXT_MARKERS = frozenset(
    {
        "/" + "Users" + "/",
        "private" + "/derived",
        "internal" + "/candidates",
        "dataset_release_id" + '": "financial_',
        _absolute_path_field("source_raw_file"),
        _absolute_path_field("redacted_export"),
        _absolute_path_field("incident_dir"),
        _absolute_path_field("mapping_rules_path"),
        _absolute_path_field("repo_root"),
        _absolute_path_field("legacy_root"),
    }
)
_FORBIDDEN_VENUE_MARKERS = (
    "".join(("n", "dss")),
    "".join(("us", "enix")),
    "".join(("i", "eee")),
    "".join(("c", "cs")),
    "".join(("a", "cm")),
    "".join(("s", "&", "p")),
    " ".join(("security", "symposium")),
    " ".join(("network", "and", "distributed")),
    " ".join(("computer", "and", "communications", "security")),
)
# A leak is a leak wherever it hides, so the text scan covers every text format
# the release ships, including the corpus telemetry. Event names are prose, not
# data: an anonymized hostname can spell an acronym a conference also uses, and
# scanning the corpus for them would fail the release on its own evidence. The
# markers above are assembled from fragments so that this module, which the
# release ships and this scan reads, does not report itself.
_LEAK_SCANNED_SUFFIXES = frozenset(
    {".md", ".txt", ".json", ".jsonl", ".yaml", ".yml", ".toml", ".sh", ".py", ".tex"}
)
_VENUE_SCANNED_SUFFIXES = frozenset({".md", ".txt", ".sh", ".py", ".tex"})
_FORBIDDEN_VENUE_PATTERN = re.compile(
    r"\b(?:"
    + "|".join(re.escape(marker) for marker in _FORBIDDEN_VENUE_MARKERS)
    + r")\b",
    flags=re.IGNORECASE,
)
_SHA256_LENGTH = 64


def _is_local_transient_path(rel_path: Path) -> bool:
    parts = set(rel_path.parts)
    if any(name in parts for name in _LOCAL_TRANSIENT_DIR_NAMES):
        return True
    if rel_path.name in _LOCAL_TRANSIENT_FILE_NAMES:
        return True
    if any(rel_path.name.endswith(suffix) for suffix in _LOCAL_TRANSIENT_FILE_SUFFIXES):
        return True
    return rel_path.suffix in _LOCAL_TRANSIENT_SUFFIXES


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_forbidden_paths(package_root: Path) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    for path in package_root.rglob("*"):
        rel_path = path.relative_to(package_root)
        if _is_local_transient_path(rel_path):
            continue
        rel = rel_path.as_posix()
        parts = set(rel.split("/"))
        if any(name in parts for name in _FORBIDDEN_PATH_PARTS):
            issues.append(
                {
                    "code": "forbidden_path",
                    "path": rel,
                    "message": (
                        "Forbidden path/component found inside artifact package."
                    ),
                }
            )
            continue
    return issues


def _check_required_structure(package_root: Path) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    manifest_path = package_root / "artifact_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {}
    )
    freeze_path = (
        package_root / "artifact_outputs" / "analysis" / "protocol_freeze.json"
    )
    freeze = (
        json.loads(freeze_path.read_text(encoding="utf-8"))
        if freeze_path.is_file()
        else {}
    )
    freeze_inputs = freeze.get("input_hashes")
    campaign_is_bound = isinstance(freeze_inputs, dict) and isinstance(
        freeze_inputs.get("corpus_campaign_manifest"), dict
    )
    controlled_freeze = freeze.get("schema_version") == PROTOCOL_FREEZE_SCHEMA
    required_files = [
        manifest_path,
        package_root / "ARTIFACT_BOUNDARY.md",
        package_root / "README.md",
        package_root / "EVAL_PROTOCOL.md",
        package_root / "artifact_data" / "global" / "action_catalog.yaml",
        package_root / "artifact_data" / "global" / "constraints.yaml",
        freeze_path,
    ]
    if controlled_freeze:
        required_files.append(
            package_root
            / "artifact_outputs"
            / "analysis"
            / "execution_source_manifest.json"
        )
    if manifest.get("artifact_stage") == "pre_execution_source" or campaign_is_bound:
        required_files.append(
            package_root
            / "artifact_outputs"
            / "analysis"
            / "corpus_campaign.json"
        )
    for file_path in required_files:
        if not file_path.exists():
            issues.append(
                {
                    "code": "missing_required_file",
                    "path": str(file_path.relative_to(package_root)),
                    "message": "Required artifact file is missing.",
                }
            )

    partitions = manifest.get("partitions")
    roots = [package_root / "artifact_data" / "dataset"]
    if isinstance(partitions, dict) and (
        "export_batch_2" in partitions or "heldout" in partitions
    ):
        roots.append(package_root / "artifact_data" / "heldout")

    incidents: list[Path] = []
    for dataset_root in roots:
        partition_incidents = sorted(
            path for path in dataset_root.glob("INC_*") if path.is_dir()
        )
        if not partition_incidents:
            issues.append(
                {
                    "code": "missing_incidents",
                    "path": str(dataset_root.relative_to(package_root)),
                    "message": "No incident folders found in declared partition.",
                }
            )
        incidents.extend(partition_incidents)
    if not incidents:
        return issues

    for incident_dir in incidents:
        meta = incident_dir / "incident_meta.json"
        human = incident_dir / "incident_human_actions.jsonl"
        if not meta.exists():
            issues.append(
                {
                    "code": "missing_incident_meta",
                    "path": str(meta.relative_to(package_root)),
                    "message": "Incident metadata file is missing.",
                }
            )
        if not human.exists():
            issues.append(
                {
                    "code": "missing_incident_human_actions",
                    "path": str(human.relative_to(package_root)),
                    "message": "Incident human actions file is missing.",
                }
            )
    return issues


def _check_claim_evidence_matrix(package_root: Path) -> list[dict[str, str]]:
    """Keep reported claims within the evidence shipped at each artifact stage."""
    manifest_path = package_root / "artifact_manifest.json"
    if not manifest_path.is_file():
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stage = manifest.get("artifact_stage")
    terminal_evidence = isinstance(manifest.get("campaign_evidence_path"), str)
    if stage != "pre_execution_source" and not terminal_evidence:
        return []

    matrix_path = (
        package_root
        / "artifact_outputs"
        / "analysis"
        / "claim_evidence_matrix.json"
    )
    readable_path = package_root / "CLAIM_EVIDENCE_MATRIX.md"
    missing = [path for path in (matrix_path, readable_path) if not path.is_file()]
    if missing:
        return [
            {
                "code": "missing_claim_evidence_matrix",
                "path": path.relative_to(package_root).as_posix(),
                "message": "The artifact does not state which claims its evidence supports.",
            }
            for path in missing
        ]

    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    issues: list[dict[str, str]] = []
    expected_stage = "reported_results" if terminal_evidence else "pre_execution_source"
    if matrix.get("schema_version") != 1 or matrix.get("artifact_stage") != expected_stage:
        issues.append(
            {
                "code": "claim_evidence_stage_mismatch",
                "path": matrix_path.relative_to(package_root).as_posix(),
                "message": "The claim matrix does not match the artifact evidence stage.",
            }
        )

    claims = matrix.get("claims")
    if not isinstance(claims, list):
        return [
            *issues,
            {
                "code": "invalid_claim_evidence_matrix",
                "path": matrix_path.relative_to(package_root).as_posix(),
                "message": "The claim matrix must contain a claims list.",
            },
        ]

    required_claim_ids = {
        "corpus_scope",
        "policy_selection",
        "verifier_semantics",
        "model_results",
    }
    seen_claim_ids: set[str] = set()
    for claim in claims:
        if not isinstance(claim, dict):
            issues.append(
                {
                    "code": "invalid_claim_evidence_matrix",
                    "path": matrix_path.relative_to(package_root).as_posix(),
                    "message": "Each claim entry must be an object.",
                }
            )
            continue
        claim_id = claim.get("claim_id")
        status = claim.get("status")
        evidence = claim.get("evidence")
        if not isinstance(claim_id, str) or claim_id in seen_claim_ids:
            issues.append(
                {
                    "code": "invalid_claim_evidence_matrix",
                    "path": matrix_path.relative_to(package_root).as_posix(),
                    "message": "Claim identifiers must be unique non-empty strings.",
                }
            )
            continue
        seen_claim_ids.add(claim_id)
        if status not in {"supported", "pending"} or not isinstance(evidence, list):
            issues.append(
                {
                    "code": "invalid_claim_evidence_matrix",
                    "path": matrix_path.relative_to(package_root).as_posix(),
                    "message": f"Claim {claim_id} has an invalid status or evidence list.",
                }
            )
            continue
        if status == "pending" and evidence:
            issues.append(
                {
                    "code": "pending_claim_has_evidence",
                    "path": matrix_path.relative_to(package_root).as_posix(),
                    "message": f"Pending claim {claim_id} must not imply shipped evidence.",
                }
            )
        if status == "supported":
            if not evidence:
                issues.append(
                    {
                        "code": "supported_claim_has_no_evidence",
                        "path": matrix_path.relative_to(package_root).as_posix(),
                        "message": f"Supported claim {claim_id} has no evidence.",
                    }
                )
            for relative_evidence in evidence:
                if not isinstance(relative_evidence, str) or not (
                    package_root / relative_evidence
                ).is_file():
                    issues.append(
                        {
                            "code": "claim_evidence_missing",
                            "path": str(relative_evidence),
                            "message": f"Supported claim {claim_id} cites missing evidence.",
                        }
                    )

    missing_claim_ids = sorted(required_claim_ids - seen_claim_ids)
    if missing_claim_ids:
        issues.append(
            {
                "code": "claim_evidence_coverage_gap",
                "path": matrix_path.relative_to(package_root).as_posix(),
                "message": "Missing required claims: " + ", ".join(missing_claim_ids),
            }
        )
    model_claims = [
        claim
        for claim in claims
        if isinstance(claim, dict) and claim.get("claim_id") == "model_results"
    ]
    expected_model_status = "supported" if terminal_evidence else "pending"
    if len(model_claims) != 1 or model_claims[0].get("status") != expected_model_status:
        issues.append(
            {
                "code": "model_claim_evidence_mismatch",
                "path": matrix_path.relative_to(package_root).as_posix(),
                "message": "Model-result claims do not match the terminal evidence state.",
            }
        )
    return issues


def _check_execution_campaign(package_root: Path) -> list[dict[str, str]]:
    campaign_path = (
        package_root / "artifact_outputs" / "analysis" / "corpus_campaign.json"
    )
    source_path = (
        package_root
        / "artifact_outputs"
        / "analysis"
        / "execution_source_manifest.json"
    )
    freeze_path = (
        package_root / "artifact_outputs" / "analysis" / "protocol_freeze.json"
    )
    if not all(path.is_file() for path in (campaign_path, source_path, freeze_path)):
        return []
    issues: list[dict[str, str]] = []
    try:
        campaign = validate_campaign_sources(
            repo_root=package_root,
            manifest_path=campaign_path,
        )
    except (FileNotFoundError, OSError, UnicodeError, ValueError) as exc:
        issues.append(
            {
                "code": "invalid_corpus_campaign",
                "path": campaign_path.relative_to(package_root).as_posix(),
                "message": str(exc),
            }
        )
        return issues
    try:
        validate_execution_source_manifest(
            repo_root=package_root,
            manifest_path=source_path,
        )
    except (FileNotFoundError, OSError, UnicodeError, ValueError) as exc:
        issues.append(
            {
                "code": "invalid_execution_source_manifest",
                "path": source_path.relative_to(package_root).as_posix(),
                "message": str(exc),
            }
        )
    try:
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        issues.append(
            {
                "code": "invalid_protocol_freeze_json",
                "path": freeze_path.relative_to(package_root).as_posix(),
                "message": str(exc),
            }
        )
        return issues
    expected_count = campaign.total_incident_count
    expected_ids = sorted(campaign.incident_partitions)
    freeze_incident_ids = freeze.get("incident_ids")
    if not isinstance(freeze_incident_ids, list) or not all(
        isinstance(incident_id, str) for incident_id in freeze_incident_ids
    ):
        freeze_incident_ids = []
    if freeze.get("incident_count") != expected_count or sorted(
        freeze_incident_ids
    ) != expected_ids:
        issues.append(
            {
                "code": "protocol_campaign_incident_mismatch",
                "path": freeze_path.relative_to(package_root).as_posix(),
                "message": "Protocol freeze and corpus campaign select different incidents.",
            }
        )
    if freeze.get("incident_partitions") != campaign.incident_partitions:
        issues.append(
            {
                "code": "protocol_campaign_partition_mismatch",
                "path": freeze_path.relative_to(package_root).as_posix(),
                "message": "Protocol freeze and corpus campaign assign different batches.",
            }
        )
    models = freeze.get("official_models")
    arms = freeze.get("arms")
    repeats = freeze.get("repeats")
    dimensions_are_valid = (
        isinstance(models, list)
        and all(isinstance(model, str) for model in models)
        and isinstance(arms, list)
        and all(isinstance(arm, str) for arm in arms)
        and isinstance(repeats, int)
        and not isinstance(repeats, bool)
        and repeats > 0
    )
    expected_cells = (
        expected_count * len(models) * len(arms) * repeats
        if dimensions_are_valid
        else 0
    )
    if freeze.get("primary_cell_count") != expected_cells:
        issues.append(
            {
                "code": "inconsistent_provider_matrix",
                "path": freeze_path.relative_to(package_root).as_posix(),
                "message": "Protocol freeze primary_cell_count is inconsistent.",
            }
        )
    return issues


def _check_policy_review_hashes(package_root: Path) -> list[dict[str, str]]:
    review_path = package_root / "artifact_data" / "global" / "review_record.yaml"
    if not review_path.is_file():
        return []
    try:
        payload = yaml.safe_load(review_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        return [
            {
                "code": "invalid_policy_review_record",
                "path": str(review_path.relative_to(package_root)),
                "message": f"Policy review record cannot be read: {exc}",
            }
        ]
    if not isinstance(payload, dict):
        return [
            {
                "code": "invalid_policy_review_record",
                "path": str(review_path.relative_to(package_root)),
                "message": "Policy review record must be a YAML mapping.",
            }
        ]

    issues: list[dict[str, str]] = []
    bindings = {
        "action_catalog_sha256": "action_catalog.yaml",
        "constraints_sha256": "constraints.yaml",
    }
    for hash_field, filename in bindings.items():
        expected = payload.get(hash_field)
        bound_path = review_path.parent / filename
        if not isinstance(expected, str) or len(expected) != _SHA256_LENGTH:
            issues.append(
                {
                    "code": "invalid_policy_review_hash",
                    "path": str(review_path.relative_to(package_root)),
                    "message": f"Policy review record lacks a valid {hash_field}.",
                }
            )
            continue
        if bound_path.is_file() and _sha256_file(bound_path) != expected:
            issues.append(
                {
                    "code": "policy_review_hash_mismatch",
                    "path": str(bound_path.relative_to(package_root)),
                    "message": f"Published {filename} differs from its reviewed hash.",
                }
            )
    return issues


def _check_protocol_freeze(package_root: Path) -> list[dict[str, str]]:
    freeze_path = (
        package_root / "artifact_outputs" / "analysis" / "protocol_freeze.json"
    )
    if not freeze_path.exists():
        return []
    try:
        payload = json.loads(freeze_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return [
            {
                "code": "invalid_protocol_freeze_json",
                "path": "artifact_outputs/analysis/protocol_freeze.json",
                "message": "Protocol freeze manifest is not valid JSON.",
            }
        ]
    if not isinstance(payload, dict):
        return [
            {
                "code": "invalid_protocol_freeze_shape",
                "path": "artifact_outputs/analysis/protocol_freeze.json",
                "message": "Protocol freeze manifest must be a JSON object.",
            }
        ]

    issues: list[dict[str, str]] = []
    required_keys = [
        "dataset_release_id",
        "eval_protocol_version",
        "official_models",
        "incident_count",
        "input_hashes",
    ]
    for key in required_keys:
        if key not in payload:
            issues.append(
                {
                    "code": "missing_protocol_freeze_field",
                    "path": "artifact_outputs/analysis/protocol_freeze.json",
                    "message": (
                        f"Protocol freeze manifest is missing required field `{key}`."
                    ),
                }
            )

    official_models = payload.get("official_models")
    if isinstance(official_models, list) and not official_models:
        issues.append(
            {
                "code": "empty_official_models",
                "path": "artifact_outputs/analysis/protocol_freeze.json",
                "message": (
                    "Protocol freeze manifest must list at least one official model."
                ),
            }
        )
    input_hashes = payload.get("input_hashes")
    if isinstance(input_hashes, dict):
        for name, entry in input_hashes.items():
            if not isinstance(entry, dict):
                issues.append(
                    {
                        "code": "invalid_protocol_freeze_input_hash_entry",
                        "path": "artifact_outputs/analysis/protocol_freeze.json",
                        "message": (
                            "Protocol freeze input hash entry "
                            f"`{name}` must be an object."
                        ),
                    }
                )
                continue
            rel = entry.get("path")
            checksum = entry.get("sha256")
            if not isinstance(rel, str) or not isinstance(checksum, str):
                issues.append(
                    {
                        "code": "invalid_protocol_freeze_input_hash_fields",
                        "path": "artifact_outputs/analysis/protocol_freeze.json",
                        "message": (
                            f"Protocol freeze input hash entry `{name}` must define "
                            "string `path` and `sha256` fields."
                        ),
                    }
                )
                continue
            file_path = package_root / rel
            if not file_path.exists():
                issues.append(
                    {
                        "code": "protocol_freeze_missing_input_file",
                        "path": rel,
                        "message": (
                            "Protocol freeze input hash entry "
                            f"`{name}` points to a missing file."
                        ),
                    }
                )
                continue
            if _sha256_file(file_path) != checksum:
                issues.append(
                    {
                        "code": "protocol_freeze_input_hash_mismatch",
                        "path": rel,
                        "message": (
                            "Protocol freeze input hash entry "
                            f"`{name}` does not match file content."
                        ),
                    }
                )
    return issues


def _check_textual_leaks(package_root: Path) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    for path in package_root.rglob("*"):
        rel_path = path.relative_to(package_root)
        if _is_local_transient_path(rel_path):
            continue
        if not path.is_file():
            continue
        if path.suffix not in _LEAK_SCANNED_SUFFIXES:
            continue
        # The suffix filter already selected text formats. One that does not
        # decode is a corrupt release, and a scan that skips it reports clean.
        text = path.read_text(encoding="utf-8")
        for marker in _FORBIDDEN_TEXT_MARKERS:
            if marker not in text:
                continue
            issues.append(
                {
                    "code": "forbidden_text_marker",
                    "path": str(rel_path),
                    "message": f"Artifact text contains forbidden marker `{marker}`.",
                }
            )
            break
        if (
            path.suffix in _VENUE_SCANNED_SUFFIXES
            and _FORBIDDEN_VENUE_PATTERN.search(text)
        ):
            issues.append(
                {
                    "code": "forbidden_venue_marker",
                    "path": str(rel_path),
                    "message": "Artifact text contains event-specific language.",
                }
            )
    return issues


def _check_manifest_integrity(package_root: Path) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    manifest_path = package_root / "artifact_manifest.json"
    if not manifest_path.exists():
        return issues
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return [
            {
                "code": "invalid_manifest_json",
                "path": "artifact_manifest.json",
                "message": "artifact_manifest.json is not valid JSON.",
            }
        ]

    if not isinstance(manifest, dict):
        return [
            {
                "code": "invalid_manifest_json",
                "path": "artifact_manifest.json",
                "message": "artifact_manifest.json is not a JSON object.",
            }
        ]

    entries = manifest.get("files", [])
    if not isinstance(entries, list):
        return [
            {
                "code": "invalid_manifest_files",
                "path": "artifact_manifest.json",
                "message": "Manifest 'files' must be a list.",
            }
        ]

    expected_count = manifest.get("file_count")
    if isinstance(expected_count, int) and expected_count != len(entries):
        issues.append(
            {
                "code": "manifest_file_count_mismatch",
                "path": "artifact_manifest.json",
                "message": "Manifest file_count does not match files list length.",
            }
        )

    for entry in entries:
        if not isinstance(entry, dict):
            issues.append(
                {
                    "code": "invalid_manifest_entry",
                    "path": "artifact_manifest.json",
                    "message": "Manifest entry must be an object with path and sha256.",
                }
            )
            continue
        rel = entry.get("path")
        checksum = entry.get("sha256")
        if not isinstance(rel, str) or not isinstance(checksum, str):
            issues.append(
                {
                    "code": "invalid_manifest_entry_fields",
                    "path": "artifact_manifest.json",
                    "message": "Manifest entry missing valid path/sha256.",
                }
            )
            continue
        file_path = package_root / rel
        if not file_path.exists():
            issues.append(
                {
                    "code": "manifest_missing_file",
                    "path": rel,
                    "message": "Manifest references a file that does not exist.",
                }
            )
            continue
        if _sha256_file(file_path) != checksum:
            issues.append(
                {
                    "code": "manifest_sha256_mismatch",
                    "path": rel,
                    "message": "Manifest SHA-256 does not match file content.",
                }
            )
    return issues


def _check_campaign_projection_source(
    *,
    package_root: Path,
    artifact_manifest: dict[str, Any],
    payload: dict[str, Any],
    provenance_path: Path,
) -> list[dict[str, str]]:
    if payload.get("state") != "public_campaign_evidence_projection":
        return []
    digest = payload.get("source_campaign_evidence_sha256")
    if not isinstance(digest, str) or len(digest) != _SHA256_LENGTH:
        return [
            {
                "code": "invalid_public_campaign_evidence_hash",
                "path": provenance_path.as_posix(),
                "message": "Campaign evidence projection has no source hash.",
            }
        ]
    campaign_path = artifact_manifest.get("campaign_evidence_path")
    campaign_file = (
        package_root / campaign_path if isinstance(campaign_path, str) else None
    )
    if campaign_file is None or not campaign_file.is_file():
        return [
            {
                "code": "missing_source_campaign_evidence",
                "path": provenance_path.as_posix(),
                "message": "Campaign projection source is unavailable.",
            }
        ]
    if _sha256_file(campaign_file) == digest:
        return []
    return [
        {
            "code": "source_campaign_evidence_hash_mismatch",
            "path": str(campaign_path),
            "message": "Campaign projection source hash differs.",
        }
    ]


def _check_public_evidence_provenance(
    package_root: Path,
) -> list[dict[str, str]]:
    artifact_manifest_path = package_root / "artifact_manifest.json"
    artifact_manifest: dict[str, Any] = {}
    if artifact_manifest_path.is_file():
        artifact_manifest = json.loads(
            artifact_manifest_path.read_text(encoding="utf-8")
        )
        if artifact_manifest.get("artifact_stage") == "pre_execution_source":
            return []
    freeze_path = (
        package_root / "artifact_outputs" / "analysis" / "protocol_freeze.json"
    )
    if not freeze_path.is_file():
        return []
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (
        not isinstance(freeze, dict)
        or freeze.get("schema_version") != PROTOCOL_FREEZE_SCHEMA
    ):
        return []

    relative_manifest = Path(
        "artifact_outputs/analysis/public_evidence_provenance.json"
    )
    manifest_path = package_root / relative_manifest
    if not manifest_path.is_file():
        return [
            {
                "code": "missing_public_evidence_provenance",
                "path": relative_manifest.as_posix(),
                "message": "Frozen result packages require public evidence provenance.",
            }
        ]
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return [
            {
                "code": "invalid_public_evidence_provenance",
                "path": relative_manifest.as_posix(),
                "message": "Public evidence provenance is not valid JSON.",
            }
        ]
    if not isinstance(payload, dict):
        return [
            {
                "code": "invalid_public_evidence_provenance",
                "path": relative_manifest.as_posix(),
                "message": "Public evidence provenance must be a JSON object.",
            }
        ]

    issues: list[dict[str, str]] = []
    valid_states = {
        "public_evidence_export",
        "public_campaign_evidence_projection",
    }
    if (
        payload.get("state") not in valid_states
        or payload.get("constitutes_model_evidence") is not True
    ):
        issues.append(
            {
                "code": "invalid_public_evidence_state",
                "path": relative_manifest.as_posix(),
                "message": "Public evidence provenance has an invalid state.",
            }
        )
    for hash_name in (
        "source_execution_evidence_sha256",
        "source_protocol_freeze_sha256",
    ):
        digest = payload.get(hash_name)
        if not isinstance(digest, str) or len(digest) != _SHA256_LENGTH:
            issues.append(
                {
                    "code": "invalid_public_evidence_source_hash",
                    "path": relative_manifest.as_posix(),
                    "message": f"Public evidence provenance has invalid {hash_name}.",
                }
            )
    issues.extend(
        _check_campaign_projection_source(
            package_root=package_root,
            artifact_manifest=artifact_manifest,
            payload=payload,
            provenance_path=relative_manifest,
        )
    )

    for section_name in ("public_files", "derived_files"):
        entries = payload.get(section_name)
        if not isinstance(entries, dict) or not entries:
            issues.append(
                {
                    "code": "missing_public_evidence_files",
                    "path": relative_manifest.as_posix(),
                    "message": f"Public evidence provenance lacks {section_name}.",
                }
            )
            continue
        for role, entry in sorted(entries.items()):
            if not isinstance(entry, dict):
                issues.append(
                    {
                        "code": "invalid_public_evidence_file",
                        "path": relative_manifest.as_posix(),
                        "message": f"Public evidence entry is invalid: {role}.",
                    }
                )
                continue
            raw_path = entry.get("path")
            expected_hash = entry.get("sha256")
            relative_path = Path(raw_path) if isinstance(raw_path, str) else None
            if (
                relative_path is None
                or relative_path.is_absolute()
                or ".." in relative_path.parts
                or not isinstance(expected_hash, str)
                or len(expected_hash) != _SHA256_LENGTH
            ):
                issues.append(
                    {
                        "code": "invalid_public_evidence_file",
                        "path": relative_manifest.as_posix(),
                        "message": f"Public evidence entry fields are invalid: {role}.",
                    }
                )
                continue
            file_path = package_root / relative_path
            if not file_path.is_file():
                issues.append(
                    {
                        "code": "missing_public_evidence_file",
                        "path": relative_path.as_posix(),
                        "message": f"Public evidence file is missing: {role}.",
                    }
                )
                continue
            if _sha256_file(file_path) != expected_hash:
                issues.append(
                    {
                        "code": "public_evidence_hash_mismatch",
                        "path": relative_path.as_posix(),
                        "message": f"Public evidence file hash differs: {role}.",
                    }
                )
    return issues


def _check_terminal_campaign_evidence(
    package_root: Path,
) -> list[dict[str, str]]:
    artifact_manifest_path = package_root / "artifact_manifest.json"
    if not artifact_manifest_path.is_file():
        return []
    try:
        artifact_manifest = json.loads(
            artifact_manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return []
    if not isinstance(artifact_manifest, dict):
        return []
    raw_path = artifact_manifest.get("campaign_evidence_path")
    issue: dict[str, str] | None = None
    if artifact_manifest.get("artifact_stage") == "pre_execution_source":
        if raw_path is not None:
            issue = {
                "code": "pre_execution_campaign_evidence_present",
                "path": "artifact_manifest.json",
                "message": "A source package cannot claim terminal campaign evidence.",
            }
    elif artifact_manifest.get("provider_incident_count", 0) == 0 and raw_path is None:
        issue = None
    elif not isinstance(raw_path, str) or not raw_path:
        issue = {
                "code": "missing_terminal_campaign_evidence",
                "path": "artifact_manifest.json",
                "message": "Reported provider results require one campaign manifest.",
            }
    else:
        relative_path = Path(raw_path)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            issue = {
                "code": "invalid_terminal_campaign_evidence_path",
                "path": "artifact_manifest.json",
                "message": "Campaign evidence path escapes the artifact.",
            }
        else:
            try:
                evidence = load_campaign_evidence(
                    repo_root=package_root,
                    manifest_path=package_root / relative_path,
                )
            except (
                FileNotFoundError,
                OSError,
                UnicodeDecodeError,
                ValueError,
            ) as exc:
                issue = {
                    "code": "terminal_campaign_evidence_invalid",
                    "path": relative_path.as_posix(),
                    "message": f"Terminal campaign evidence failed validation: {exc}",
                }
            else:
                if artifact_manifest.get("run_id") != evidence.campaign_id:
                    issue = {
                        "code": "terminal_campaign_id_mismatch",
                        "path": relative_path.as_posix(),
                        "message": "Artifact run ID differs from campaign evidence.",
                    }
    return [issue] if issue is not None else []


@contextmanager
def _provenance_paths(
    package_root: Path,
    incident_ids: list[str],
) -> Iterator[RepoPaths]:
    batch_one = package_root / "artifact_data" / "dataset"
    batch_two = package_root / "artifact_data" / "heldout"
    if all((batch_one / incident_id).is_dir() for incident_id in incident_ids):
        yield RepoPaths(repo_root=package_root)
        return
    with TemporaryDirectory(prefix="socpilot-campaign-") as temporary:
        view = Path(temporary)
        for incident_id in incident_ids:
            sources = [
                candidate
                for candidate in (
                    batch_one / incident_id,
                    batch_two / incident_id,
                )
                if candidate.is_dir()
            ]
            if len(sources) != 1:
                raise ValueError(
                    f"Incident {incident_id} must occur in exactly one package batch."
                )
            (view / incident_id).symlink_to(sources[0], target_is_directory=True)
        yield RepoPaths(
            repo_root=package_root,
            inbox_incidents_override=view,
        )


def _check_global_artifact_provenance(
    package_root: Path,
) -> list[dict[str, str]]:
    provenance_path = (
        package_root
        / "artifact_outputs"
        / "analysis"
        / "global_artifact_provenance.json"
    )
    if not provenance_path.is_file():
        return []

    artifact_manifest_path = package_root / "artifact_manifest.json"
    try:
        published = json.loads(provenance_path.read_text(encoding="utf-8"))
        artifact_manifest = json.loads(
            artifact_manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return [
            {
                "code": "global_provenance_unreadable",
                "path": provenance_path.relative_to(package_root).as_posix(),
                "message": f"Global provenance cannot be audited: {exc}",
            }
        ]
    incident_ids = artifact_manifest.get("provider_incidents") or artifact_manifest.get(
        "incidents"
    )
    if not isinstance(published, dict) or not isinstance(incident_ids, list) or any(
        not isinstance(incident_id, str) for incident_id in incident_ids
    ):
        return [
            {
                "code": "global_provenance_invalid",
                "path": provenance_path.relative_to(package_root).as_posix(),
                "message": "Global provenance or its incident scope is invalid.",
            }
        ]
    try:
        with _provenance_paths(package_root, incident_ids) as paths:
            mismatches = provenance_consistency_issues(
                paths=paths,
                incident_ids=incident_ids,
                published=published,
            )
    except (FileNotFoundError, OSError, ValueError) as exc:
        return [
            {
                "code": "global_provenance_recompute_failed",
                "path": provenance_path.relative_to(package_root).as_posix(),
                "message": f"Global provenance cannot be recomputed: {exc}",
            }
        ]
    if not mismatches:
        return []
    return [
        {
            "code": "global_provenance_mismatch",
            "path": provenance_path.relative_to(package_root).as_posix(),
            "message": (
                "Published global provenance differs from released inputs: "
                + ", ".join(mismatches)
            ),
        }
    ]


def verify_artifact_package(package_root: Path) -> dict[str, Any]:
    issues = [
        *_check_required_structure(package_root),
        *_check_claim_evidence_matrix(package_root),
        *_check_policy_review_hashes(package_root),
        *_check_protocol_freeze(package_root),
        *_check_execution_campaign(package_root),
        *_check_forbidden_paths(package_root),
        *_check_textual_leaks(package_root),
        *_check_manifest_integrity(package_root),
        *_check_terminal_campaign_evidence(package_root),
        *_check_public_evidence_provenance(package_root),
        *_check_global_artifact_provenance(package_root),
    ]
    status = "ok" if not issues else "failed"
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": status,
        "issue_count": len(issues),
        "issues": issues,
    }


def _stabilize_timestamp(
    existing: dict[str, Any],
    report: dict[str, Any],
) -> dict[str, Any]:
    comparable_existing = {
        key: value for key, value in existing.items() if key != "generated_at_utc"
    }
    comparable_report = {
        key: value for key, value in report.items() if key != "generated_at_utc"
    }
    if comparable_existing == comparable_report and isinstance(
        existing.get("generated_at_utc"), str
    ):
        report = dict(report)
        report["generated_at_utc"] = existing["generated_at_utc"]
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soc_llm_policy.artifact_verify",
        description="Verify release-candidate artifact package integrity and boundary.",
    )
    parser.add_argument("--package-root", required=True)
    parser.add_argument(
        "--output-json",
        default="release_candidate_checklist.json",
    )
    return parser


def _report_issues(report: dict[str, Any]) -> None:
    """Say what failed. A gate that exits nonzero in silence teaches nothing."""
    issues = report.get("issues") or []
    print(f"Artifact verification failed with {len(issues)} issue(s):", file=sys.stderr)
    for issue in issues:
        code = issue.get("code", "unknown")
        path = issue.get("path", "")
        print(f"  {code}: {path}".rstrip(), file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    package_root = Path(args.package_root).expanduser().resolve()
    report = verify_artifact_package(package_root)
    output_path = Path(args.output_json)
    if not output_path.is_absolute():
        output_path = package_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if isinstance(existing, dict):
            report = _stabilize_timestamp(existing, report)
    output_text = json.dumps(report, ensure_ascii=False, indent=2)
    if not output_path.exists() or output_path.read_text(encoding="utf-8") != output_text:
        output_path.write_text(output_text, encoding="utf-8")
    if report["status"] == "ok":
        return 0
    _report_issues(report)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
