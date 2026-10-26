from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from soc_llm_policy.corpus_hash import canonical_corpus_sha256
from soc_llm_policy.durable_io import atomic_write_json
from soc_llm_policy.paths import repo_relative_path, resolve_repo_root

_LABEL_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_MIN_PARTITIONS = 2
_REQUIRED_INCIDENT_FILES = frozenset(
    {
        "incident_meta.json",
        "incident_telemetry.jsonl",
        "incident_human_actions.jsonl",
    }
)


class CampaignPartition(BaseModel):
    source_path: str = Field(..., min_length=1)
    incident_count: int = Field(..., ge=1)
    incident_ids: list[str] = Field(..., min_length=1)
    corpus_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")


class CorpusCampaignManifest(BaseModel):
    schema_version: Literal[1] = 1
    state: Literal["frozen_corpus_campaign"] = "frozen_corpus_campaign"
    view_path: str = Field(..., min_length=1)
    total_incident_count: int = Field(..., ge=1)
    partitions: dict[str, CampaignPartition] = Field(..., min_length=2)
    incident_partitions: dict[str, str] = Field(..., min_length=1)


def _incident_ids(root: Path) -> list[str]:
    if not root.is_dir():
        raise FileNotFoundError(f"Corpus partition does not exist: {root}")
    incident_ids: list[str] = []
    for incident_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        missing = sorted(
            filename
            for filename in _REQUIRED_INCIDENT_FILES
            if not (incident_dir / filename).is_file()
        )
        if missing:
            raise FileNotFoundError(
                f"Incident {incident_dir.name} is missing files: {missing}"
            )
        incident_ids.append(incident_dir.name)
    if not incident_ids:
        raise ValueError(f"Corpus partition is empty: {root}")
    return incident_ids


def _resolve_partitions(
    repo_root: Path, partitions: dict[str, Path]
) -> dict[str, Path]:
    if len(partitions) < _MIN_PARTITIONS:
        raise ValueError("A campaign requires at least two named partitions.")
    resolved: dict[str, Path] = {}
    for label, path in sorted(partitions.items()):
        if not _LABEL_PATTERN.fullmatch(label):
            raise ValueError(f"Invalid partition label: {label!r}")
        candidate = path.expanduser()
        if not candidate.is_absolute():
            candidate = repo_root / candidate
        candidate = candidate.resolve()
        if repo_root != candidate and repo_root not in candidate.parents:
            raise ValueError(f"Partition escapes the artifact root: {candidate}")
        resolved[label] = candidate
    return resolved


def build_campaign(
    *,
    repo_root: Path,
    partitions: dict[str, Path],
    view_path: Path,
    manifest_path: Path,
) -> CorpusCampaignManifest:
    repo_root = repo_root.expanduser().resolve()
    resolved_partitions = _resolve_partitions(repo_root, partitions)
    view_path = view_path.expanduser().resolve()
    manifest_path = manifest_path.expanduser().resolve()
    for output in (view_path, manifest_path):
        if repo_root != output and repo_root not in output.parents:
            raise ValueError(f"Campaign output escapes the artifact root: {output}")
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"Campaign output already exists: {output}")

    partition_ids = {
        label: _incident_ids(root) for label, root in resolved_partitions.items()
    }
    incident_partitions: dict[str, str] = {}
    for label, incident_ids in partition_ids.items():
        for incident_id in incident_ids:
            previous = incident_partitions.get(incident_id)
            if previous is not None:
                raise ValueError(
                    f"Incident {incident_id} appears in both {previous} and {label}."
                )
            incident_partitions[incident_id] = label

    partition_records = {
        label: CampaignPartition(
            source_path=repo_relative_path(root, repo_root),
            incident_count=len(partition_ids[label]),
            incident_ids=partition_ids[label],
            corpus_sha256=canonical_corpus_sha256(root),
        )
        for label, root in resolved_partitions.items()
    }
    manifest = CorpusCampaignManifest(
        view_path=repo_relative_path(view_path, repo_root),
        total_incident_count=len(incident_partitions),
        partitions=partition_records,
        incident_partitions=dict(sorted(incident_partitions.items())),
    )
    atomic_write_json(manifest_path, manifest.model_dump(mode="json"))
    materialize_campaign_view(repo_root=repo_root, manifest_path=manifest_path)
    validate_campaign(repo_root=repo_root, manifest_path=manifest_path)
    return manifest


def _validated_campaign_sources(
    *, repo_root: Path, manifest_path: Path
) -> tuple[CorpusCampaignManifest, dict[str, Path]]:
    repo_root = repo_root.expanduser().resolve()
    manifest = CorpusCampaignManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    source_roots: dict[str, Path] = {}
    observed_ids: set[str] = set()
    for label, partition in sorted(manifest.partitions.items()):
        source_root = (repo_root / partition.source_path).resolve()
        if source_root != repo_root and repo_root not in source_root.parents:
            raise ValueError(f"Campaign partition escapes the artifact: {label}.")
        incident_ids = _incident_ids(source_root)
        if incident_ids != partition.incident_ids:
            raise ValueError(f"Partition {label} incident list changed.")
        if canonical_corpus_sha256(source_root) != partition.corpus_sha256:
            raise ValueError(f"Partition {label} corpus hash changed.")
        for incident_id in incident_ids:
            if manifest.incident_partitions.get(incident_id) != label:
                raise ValueError(f"Partition assignment changed for {incident_id}.")
            observed_ids.add(incident_id)
        source_roots[label] = source_root
    if len(observed_ids) != manifest.total_incident_count:
        raise ValueError("Campaign total does not match its partitions.")
    if observed_ids != set(manifest.incident_partitions):
        raise ValueError("Campaign assignment map differs from its partitions.")
    return manifest, source_roots


def validate_campaign_sources(
    *, repo_root: Path, manifest_path: Path
) -> CorpusCampaignManifest:
    manifest, _ = _validated_campaign_sources(
        repo_root=repo_root,
        manifest_path=manifest_path,
    )
    return manifest


def materialize_campaign_view(
    *, repo_root: Path, manifest_path: Path
) -> CorpusCampaignManifest:
    repo_root = repo_root.expanduser().resolve()
    manifest, source_roots = _validated_campaign_sources(
        repo_root=repo_root,
        manifest_path=manifest_path,
    )
    view_path = (repo_root / manifest.view_path).resolve()
    if view_path != repo_root and repo_root not in view_path.parents:
        raise ValueError("Campaign view escapes the artifact root.")
    if view_path.exists() or view_path.is_symlink():
        return validate_campaign(repo_root=repo_root, manifest_path=manifest_path)

    staging = view_path.with_name(f".{view_path.name}.{os.getpid()}.tmp")
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(f"Campaign staging path already exists: {staging}")
    staging.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    try:
        for incident_id, label in sorted(manifest.incident_partitions.items()):
            source = source_roots[label] / incident_id
            relative_target = os.path.relpath(source, start=view_path)
            (staging / incident_id).symlink_to(
                relative_target,
                target_is_directory=True,
            )
        os.replace(staging, view_path)
    except (OSError, ValueError):
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return validate_campaign(repo_root=repo_root, manifest_path=manifest_path)


def validate_campaign(
    *, repo_root: Path, manifest_path: Path
) -> CorpusCampaignManifest:
    repo_root = repo_root.expanduser().resolve()
    manifest, source_roots = _validated_campaign_sources(
        repo_root=repo_root,
        manifest_path=manifest_path,
    )
    view_path = (repo_root / manifest.view_path).resolve()
    if view_path != repo_root and repo_root not in view_path.parents:
        raise ValueError("Campaign view escapes the artifact root.")
    for incident_id, label in sorted(manifest.incident_partitions.items()):
        link = view_path / incident_id
        if not link.is_symlink() or link.resolve() != source_roots[label] / incident_id:
            raise ValueError(f"Campaign view changed for {incident_id}.")
    return manifest


def _partition_arg(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("partition must use LABEL=PATH")
    return label, Path(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soc_llm_policy.corpus_campaign",
        description="Build or validate a content-addressed multi-partition corpus view.",
    )
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--partition", action="append", type=_partition_arg, default=[])
    parser.add_argument("--view", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--materialize-view", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = resolve_repo_root(args.repo_root)
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = repo_root / manifest_path
    if args.validate_only and args.materialize_view:
        raise ValueError("Choose either --validate-only or --materialize-view.")
    if args.validate_only:
        validate_campaign(repo_root=repo_root, manifest_path=manifest_path)
        return 0
    if args.materialize_view:
        materialize_campaign_view(repo_root=repo_root, manifest_path=manifest_path)
        return 0
    view_path = Path(args.view)
    if not view_path.is_absolute():
        view_path = repo_root / view_path
    partitions = dict(args.partition)
    if len(partitions) != len(args.partition):
        raise ValueError("Campaign partition labels must be unique.")
    build_campaign(
        repo_root=repo_root,
        partitions=partitions,
        view_path=view_path,
        manifest_path=manifest_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
