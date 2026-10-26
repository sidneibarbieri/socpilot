from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from soc_llm_policy.paths import RepoPaths, repo_relative_path, resolve_repo_root

_DATASET_MARKERS = ("/dataset/incidents/", "/inbox/incidents/")
_RAW_MARKERS = ("/incoming/raw/", "/external_data/raw_incoming/")
_REDACTED_MARKERS = ("/incoming/redacted/", "/external_data/anonymized_staging/")
_MAPPING_BANK_MARKERS = (
    "/local_redaction/action_mapping_bank.yaml",
    "/private_anonymizer/action_mapping_bank.yaml",
)
_ACTION_CATALOG_MARKERS = (
    "/policy/action_catalog.yaml",
    "/inbox/global/action_catalog.yaml",
)
_CONSTRAINTS_MARKERS = (
    "/policy/constraints.yaml",
    "/inbox/global/constraints.yaml",
)
_OUTPUTS_INCIDENTS_MARKERS = ("/results/incidents", "/outputs/incidents")
_OUTPUTS_ANALYSIS_MARKERS = ("/results/analysis", "/outputs/analysis")
_ANONYMIZATION_POLICY_MARKERS = ("/config/anonymization_policy.yaml",)
_MITRE_STIX_MARKERS = (
    "/reference_data/mitre/enterprise-attack.json",
    "/data/mitre/enterprise-attack.json",
)
_REPO_ANCHOR = "/policy-verifier/"
_PATH_KEYS = {
    "source_raw_file",
    "redacted_export",
    "incident_dir",
    "mapping_rules_path",
    "stix_path",
    "path",
    "privacy_policy_path",
    "outputs_incidents_dir",
    "raw_incoming_dir",
    "action_catalog_path",
    "constraints_path",
    "output_dir",
}


@dataclass(frozen=True)
class RepoSurfaceReport:
    scanned_json_count: int
    updated_json_count: int


def _relative_tail(
    value: str, markers: tuple[str, ...], target_prefix: str
) -> str | None:
    normalized = value.replace("\\", "/")
    for marker in markers:
        if marker not in normalized:
            continue
        tail = normalized.split(marker, 1)[1].lstrip("/")
        return f"{target_prefix}/{tail}" if tail else target_prefix
    return None


def _normalize_path_value(value: str, repo_root: Path) -> str:
    normalized = value.replace("\\", "/").strip()
    normalized_value = _normalize_known_path_value(normalized)
    if normalized_value is not None:
        return normalized_value
    return _normalize_filesystem_path_value(
        normalized=normalized,
        original_value=value,
        repo_root=repo_root,
    )


def _normalize_known_path_value(normalized: str) -> str | None:
    if not normalized:
        return None

    marker_groups = [
        (_DATASET_MARKERS, "dataset/incidents"),
        (_RAW_MARKERS, "incoming/raw"),
        (_REDACTED_MARKERS, "incoming/redacted"),
    ]
    for markers, target_prefix in marker_groups:
        marker_path = _relative_tail(normalized, markers, target_prefix)
        if marker_path is not None:
            return marker_path

    fixed_paths = [
        (_MAPPING_BANK_MARKERS, "local_redaction/action_mapping_bank.yaml"),
        (_ACTION_CATALOG_MARKERS, "policy/action_catalog.yaml"),
        (_CONSTRAINTS_MARKERS, "policy/constraints.yaml"),
        (_ANONYMIZATION_POLICY_MARKERS, "config/anonymization_policy.yaml"),
        (_OUTPUTS_INCIDENTS_MARKERS, "results/incidents"),
        (_MITRE_STIX_MARKERS, "reference_data/mitre/enterprise-attack.json"),
    ]
    for markers, target_path in fixed_paths:
        if any(marker in normalized for marker in markers):
            return target_path

    if any(marker in normalized for marker in _OUTPUTS_ANALYSIS_MARKERS):
        tail = normalized.split("/analysis", 1)[1].lstrip("/")
        return f"results/analysis/{tail}" if tail else "results/analysis"

    if _REPO_ANCHOR in normalized:
        return normalized.split(_REPO_ANCHOR, 1)[1].lstrip("/")
    return None


def _normalize_filesystem_path_value(
    *,
    normalized: str,
    original_value: str,
    repo_root: Path,
) -> str:
    if not normalized:
        return original_value
    candidate = Path(normalized).expanduser()
    if not candidate.is_absolute():
        return normalized
    try:
        return repo_relative_path(candidate, repo_root)
    except ValueError:
        return original_value


def _normalize_json_value(value: Any, *, key: str | None, repo_root: Path) -> Any:
    if isinstance(value, dict):
        return {
            item_key: _normalize_json_value(
                item_value, key=item_key, repo_root=repo_root
            )
            for item_key, item_value in value.items()
        }
    if isinstance(value, list):
        return [
            _normalize_json_value(item, key=key, repo_root=repo_root) for item in value
        ]
    if isinstance(value, str) and key in _PATH_KEYS:
        return _normalize_path_value(value, repo_root)
    return value


def _sanitize_json_file(path: Path, repo_root: Path) -> bool:
    payload = json.loads(path.read_text(encoding="utf-8"))
    normalized_payload = _normalize_json_value(payload, key=None, repo_root=repo_root)
    rendered_original = json.dumps(payload, ensure_ascii=False, indent=2)
    rendered_normalized = json.dumps(normalized_payload, ensure_ascii=False, indent=2)
    if rendered_normalized == rendered_original:
        return False
    path.write_text(rendered_normalized, encoding="utf-8")
    return True


def sanitize_repo_surface(paths: RepoPaths) -> RepoSurfaceReport:
    candidate_files = sorted(
        set(paths.dataset_dir.glob("incidents/INC_*/evidence/source_manifest.json"))
        | set(paths.outputs_dir.glob("incidents/INC_*/evidence/source_manifest.json"))
        | set(paths.outputs_dir.rglob("*.json"))
        | {paths.mitre_manifest_path}
    )

    scanned_json_count = 0
    updated_json_count = 0
    for json_path in candidate_files:
        if not json_path.exists():
            continue
        scanned_json_count += 1
        if _sanitize_json_file(json_path, paths.repo_root):
            updated_json_count += 1

    return RepoSurfaceReport(
        scanned_json_count=scanned_json_count,
        updated_json_count=updated_json_count,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soc_llm_policy.repo_surface",
        description=(
            "Normalize tracked repository metadata to release-safe relative paths."
        ),
    )
    parser.add_argument("--repo-root", default=".")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = resolve_repo_root(args.repo_root)
    report = sanitize_repo_surface(RepoPaths(repo_root=repo_root))
    print(
        "Sanitized repository surface JSON files: "
        f"{report.updated_json_count}/{report.scanned_json_count}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
