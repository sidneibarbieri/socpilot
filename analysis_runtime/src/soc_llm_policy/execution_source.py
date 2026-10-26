from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from soc_llm_policy.durable_io import atomic_write_bytes, atomic_write_json
from soc_llm_policy.preflight_evidence import sha256_file
from soc_llm_policy.publication_redaction import expected_digest

_CANONICAL_SOURCE_SCHEMA = 1
_CANONICAL_SOURCE_STATE = "frozen_execution_source"
_SOURCE_EXACT_PATHS = frozenset(
    {
        "provider_run.sh",
        "pyproject.toml",
        "requirements.txt",
        "run.sh",
    }
)
_SOURCE_PREFIXES = (
    "artifact_data/",
    "artifact_outputs/analysis/policy_preflight/",
    "config/",
    "local_redaction/",
    "protocol_extensions/projection_fidelity_probe/",
    "reference_data/",
    "scenario_banks/",
    "scripts/",
    "src/",
)
_SOURCE_EXCLUDED_PATHS = frozenset(
    {
        "artifact_outputs/analysis/corpus_campaign.json",
        "artifact_outputs/analysis/execution_source_manifest.json",
        "artifact_outputs/analysis/protocol_freeze.json",
    }
)


@dataclass(frozen=True)
class ExecutionSourceManifest:
    path: Path
    sha256: str
    file_count: int


def _is_execution_source_path(relative_path: Path) -> bool:
    normalized = relative_path.as_posix()
    if normalized in _SOURCE_EXCLUDED_PATHS:
        return False
    return normalized in _SOURCE_EXACT_PATHS or normalized.startswith(_SOURCE_PREFIXES)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_execution_source_manifest(
    *,
    repo_root: Path,
    output_path: Path,
) -> ExecutionSourceManifest:
    root = repo_root.expanduser().resolve()
    output = output_path.expanduser().resolve()
    if output != root and root not in output.parents:
        raise ValueError("Execution source manifest must stay inside the artifact.")
    files = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": _sha256_path(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and _is_execution_source_path(path.relative_to(root))
    ]
    if not files:
        raise ValueError("Execution source manifest selected no files.")
    payload = {
        "schema_version": _CANONICAL_SOURCE_SCHEMA,
        "state": _CANONICAL_SOURCE_STATE,
        "file_count": len(files),
        "files": files,
    }
    atomic_write_json(output, payload)
    return validate_execution_source_manifest(repo_root=root, manifest_path=output)


def _read_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Execution source manifest must be an object: {path}")
    return payload


def validate_execution_source_manifest(
    *,
    repo_root: Path,
    manifest_path: Path,
) -> ExecutionSourceManifest:
    root = repo_root.expanduser().resolve()
    path = manifest_path.expanduser().resolve()
    if path != root and root not in path.parents:
        raise ValueError("Execution source manifest must stay inside the artifact.")

    payload = _read_manifest(path)
    schema_version = payload.get("schema_version")
    state = payload.get("state")
    if schema_version != _CANONICAL_SOURCE_SCHEMA:
        raise ValueError("Execution source manifest has an unsupported schema.")
    if state != _CANONICAL_SOURCE_STATE:
        raise ValueError("Execution source manifest has an invalid state.")
    files = payload.get("files")
    file_count = payload.get("file_count")
    if not isinstance(files, list) or isinstance(file_count, bool) or not isinstance(
        file_count, int
    ):
        raise ValueError("Execution source manifest has invalid file accounting.")
    if file_count != len(files):
        raise ValueError("Execution source manifest file_count does not match files.")

    seen_paths: set[str] = set()
    for entry in files:
        if not isinstance(entry, dict):
            raise ValueError("Execution source manifest entries must be objects.")
        relative_value = entry.get("path")
        expected_hash = entry.get("sha256")
        if not isinstance(relative_value, str) or not isinstance(expected_hash, str):
            raise ValueError("Execution source manifest entry is incomplete.")
        relative_path = Path(relative_value)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(
                f"Execution source path escapes the artifact: {relative_value}"
            )
        normalized = relative_path.as_posix()
        if normalized in seen_paths:
            raise ValueError(f"Duplicate execution source path: {normalized}")
        seen_paths.add(normalized)
        source_path = (root / relative_path).resolve()
        if source_path != root and root not in source_path.parents:
            raise ValueError(
                f"Execution source path resolves outside the artifact: {normalized}"
            )
        if not source_path.is_file():
            raise FileNotFoundError(f"Execution source file is missing: {normalized}")
        if sha256_file(source_path) != expected_digest(root, source_path, expected_hash):
            raise ValueError(f"Execution source file hash changed: {normalized}")

    return ExecutionSourceManifest(
        path=path,
        sha256=sha256_file(path),
        file_count=file_count,
    )


def seal_execution_source_manifest(
    *,
    repo_root: Path,
    artifact_manifest_path: Path,
    output_path: Path,
) -> ExecutionSourceManifest:
    validate_execution_source_manifest(
        repo_root=repo_root,
        manifest_path=artifact_manifest_path,
    )
    atomic_write_bytes(output_path, artifact_manifest_path.read_bytes())
    return validate_execution_source_manifest(
        repo_root=repo_root,
        manifest_path=output_path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Seal or validate the exact artifact used for provider execution."
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.expanduser().resolve()
    manifest_path = args.manifest
    if not manifest_path.is_absolute():
        manifest_path = repo_root / manifest_path
    if args.validate_only:
        sealed = validate_execution_source_manifest(
            repo_root=repo_root,
            manifest_path=manifest_path,
        )
    else:
        if args.output is None:
            raise ValueError("--output is required when sealing a manifest.")
        output_path = args.output
        if not output_path.is_absolute():
            output_path = repo_root / output_path
        sealed = seal_execution_source_manifest(
            repo_root=repo_root,
            artifact_manifest_path=manifest_path,
            output_path=output_path,
        )
    print(
        f"Execution source manifest OK: {sealed.file_count} files, "
        f"sha256={sealed.sha256}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
