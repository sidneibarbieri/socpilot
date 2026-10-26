"""Create and verify private, quiescent execution backups without model calls."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import stat
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from soc_llm_policy.corpus_campaign import materialize_campaign_view, validate_campaign
from soc_llm_policy.durable_io import atomic_write_json
from soc_llm_policy.execution_source import validate_execution_source_manifest
from soc_llm_policy.preflight_evidence import sha256_file
from soc_llm_policy.provider_plan import ProviderExecutionPlan, load_provider_plan

type Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
_MANIFEST = "backup_manifest.json"
_FORBIDDEN = {".env", ".git", ".venv", "__pycache__", "raw", "private"}
_KEY_PATTERN = re.compile(
    rb"sk-(?:proj-|ant-api)[A-Za-z0-9_-]{20,}|-----BEGIN "
    rb"(?:[A-Z0-9]+ )*PRIVATE KEY-----"
)


def _relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if (not value or path.is_absolute() or ".." in path.parts or "\\" in value
            or path.as_posix() != value or value == "."):
        raise ValueError("Backup paths must be normalized relative file paths.")
    if any(part in _FORBIDDEN or (part.startswith(".env.") and part != ".env.example")
           for part in path.parts):
        raise ValueError("Backup selection contains a credential or private path.")
    return value


class BackupFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    sha256: Digest
    size_bytes: Annotated[int, Field(strict=True, ge=0)]
    mode: Annotated[int, Field(strict=True, ge=0, le=0o777)]

    _validate_path = field_validator("path")(_relative_path)


class BackupManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    created_at_utc: str
    plan_id: str
    protocol_sha256: Digest
    plan_sha256: Digest
    plan_path: str
    different_filesystem: bool
    files: list[BackupFile] = Field(min_length=1)

    _validate_plan_path = field_validator("plan_path")(_relative_path)


def _regular_file(root: Path, relative: str) -> Path:
    path = root / _relative_path(relative)
    for candidate in (path, *path.parents):
        if candidate == root:
            break
        if candidate.is_symlink():
            raise ValueError("Backup files must not resolve through symbolic links.")
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError("Backup selection contains a non-regular file.")
    return path


def _corpus_view(root: Path, plan: ProviderExecutionPlan) -> Path | None:
    binding = plan.frozen_inputs.get("corpus_campaign_manifest")
    if binding is None:
        return None
    manifest = validate_campaign(repo_root=root, manifest_path=_regular_file(root, binding.path))
    view = root / _relative_path(manifest.view_path)
    if view.is_symlink() or {path.name for path in view.iterdir()} != set(manifest.incident_partitions):
        raise ValueError("Corpus view contains unexpected entries.")
    return view


def _selected_files(root: Path, plan_path: Path, view: Path | None) -> list[str]:
    source = json.loads((root / "artifact_manifest.json").read_text(encoding="utf-8"))
    names = {entry["path"] for entry in source["files"]}
    names.update({"artifact_manifest.json", plan_path.relative_to(root).as_posix()})
    for path in (root / "artifact_outputs").rglob("*"):
        if view is not None and path.is_relative_to(view):
            continue
        if path.is_symlink():
            raise ValueError("Execution outputs contain a symbolic link.")
        if not path.is_dir():
            names.add(path.relative_to(root).as_posix())
    for name in names:
        _regular_file(root, name)
    return sorted(names)


@contextmanager
def _idle_workers(root: Path) -> Iterator[None]:
    directories = [path for path in (root / "artifact_outputs/experiments").glob("*")
                   if path.is_dir()]
    directories.extend((root / "artifact_outputs/provider_state").glob("*/derived"))
    with ExitStack() as stack:
        for directory in sorted(directories):
            descriptor = os.open(directory, os.O_RDONLY)
            stack.callback(os.close, descriptor)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("Worker is active; back up only at a quiescent checkpoint.") from exc
        yield


def _copy_file(source: Path, target: Path, relative: str, *, scan_keys: bool) -> BackupFile:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    digest = hashlib.sha256()
    size = 0
    overlap = b""
    with source.open("rb") as reader, os.fdopen(descriptor, "wb") as writer:
        for chunk in iter(lambda: reader.read(65536), b""):
            if scan_keys and _KEY_PATTERN.search(overlap + chunk):
                raise ValueError("Possible credential in selected content; backup remains incomplete.")
            overlap = chunk[-256:]
            digest.update(chunk)
            size += len(chunk)
            writer.write(chunk)
        writer.flush()
        os.fsync(writer.fileno())
    return BackupFile(path=relative, sha256=digest.hexdigest(), size_bytes=size,
                      mode=stat.S_IMODE(source.stat().st_mode) & 0o777)


def _sync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sync_directories(root: Path) -> None:
    directories = [root, *(path for path in root.rglob("*") if path.is_dir())]
    for directory in reversed(directories):
        _sync_directory(directory)


def _validate_inputs(root: Path, plan_path: Path) -> ProviderExecutionPlan:
    plan = load_provider_plan(plan_path)
    for binding in plan.frozen_inputs.values():
        if sha256_file(_regular_file(root, binding.path)) != binding.sha256:
            raise ValueError("A fixed input differs from the execution plan.")
    freeze = json.loads((root / plan.frozen_inputs["protocol_freeze"].path).read_text())
    source_binding = freeze["input_hashes"]["execution_source_manifest"]
    source_manifest = _regular_file(root, source_binding["path"])
    if sha256_file(source_manifest) != source_binding["sha256"]:
        raise ValueError("Execution source inventory differs from the protocol.")
    validate_execution_source_manifest(repo_root=root, manifest_path=source_manifest)
    return plan


def create_backup(root: Path, plan_path: Path, destination: Path) -> Path:
    root = root.resolve()
    plan_path = plan_path if plan_path.is_absolute() else root / plan_path
    plan_relative = plan_path.relative_to(root).as_posix()
    _regular_file(root, plan_relative)
    destination = destination.absolute()
    if destination.resolve().is_relative_to(root):
        raise ValueError("Backup destination must be outside the execution workspace.")
    with _idle_workers(root):
        plan_digest = sha256_file(plan_path)
        plan = _validate_inputs(root, plan_path)
        view = _corpus_view(root, plan)
        selected = _selected_files(root, plan_path, view)
        before = {name: sha256_file(root / name) for name in selected}
        if before[plan_relative] != plan_digest:
            raise RuntimeError("Plan changed while preparing the backup.")
        destination.mkdir(mode=0o700, parents=False, exist_ok=False)
        copied = [_copy_file(root / name, destination / "workspace" / name, name, scan_keys=True)
                  for name in selected]
        after = {name: sha256_file(root / name) for name in _selected_files(root, plan_path, view)}
        if before != after or before != {entry.path: entry.sha256 for entry in copied}:
            raise RuntimeError("Source changed during backup; incomplete destination retained.")
        _validate_inputs(root, plan_path)
        _corpus_view(root, plan)
        manifest = BackupManifest(
            created_at_utc=datetime.now(UTC).isoformat(), plan_id=plan.plan_id,
            protocol_sha256=plan.frozen_inputs["protocol_freeze"].sha256,
            plan_sha256=before[plan_relative], plan_path=plan_relative,
            different_filesystem=root.stat().st_dev != destination.stat().st_dev, files=copied,
        )
        _sync_directories(destination)
        atomic_write_json(destination / _MANIFEST, manifest.model_dump(mode="json"))
        _sync_directory(destination.parent)
    return destination / _MANIFEST


def verify_backup(backup: Path, expected_manifest_sha256: str) -> BackupManifest:
    backup = backup.resolve()
    manifest_path = _regular_file(backup, _MANIFEST)
    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise ValueError("Backup manifest does not match the independently retained digest.")
    manifest = BackupManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    names = [entry.path for entry in manifest.files]
    if len(names) != len(set(names)):
        raise ValueError("Backup manifest contains duplicate paths.")
    root = backup / "workspace"
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if not path.is_dir()}
    if actual != set(names) or root.is_symlink() or any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("Backup inventory differs from its manifest.")
    for entry in manifest.files:
        path = _regular_file(root, entry.path)
        if path.stat().st_size != entry.size_bytes or sha256_file(path) != entry.sha256:
            raise ValueError(f"Backup content differs from its manifest: {entry.path}")
    plan_path = _regular_file(root, manifest.plan_path)
    plan = load_provider_plan(plan_path)
    if (sha256_file(plan_path), plan.plan_id, plan.frozen_inputs["protocol_freeze"].sha256) != (
        manifest.plan_sha256, manifest.plan_id, manifest.protocol_sha256,
    ):
        raise ValueError("Backup plan binding differs from its manifest.")
    return manifest


def restore_backup(backup: Path, expected_manifest_sha256: str, destination: Path) -> None:
    manifest = verify_backup(backup, expected_manifest_sha256)
    destination.mkdir(mode=0o700, parents=False, exist_ok=False)
    for entry in manifest.files:
        source = _regular_file(backup.resolve() / "workspace", entry.path)
        target = destination / entry.path
        copied = _copy_file(source, target, entry.path, scan_keys=False)
        if (copied.sha256, copied.size_bytes) != (entry.sha256, entry.size_bytes):
            raise RuntimeError("Backup changed during restoration; incomplete destination retained.")
        target.chmod(entry.mode)
    plan = _validate_inputs(destination, destination / manifest.plan_path)
    corpus_binding = plan.frozen_inputs.get("corpus_campaign_manifest")
    if corpus_binding is not None:
        materialize_campaign_view(repo_root=destination, manifest_path=destination / corpus_binding.path)
    _sync_directories(destination)
    _sync_directory(destination.parent)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("--repo-root", type=Path, default=Path.cwd())
    create.add_argument("--plan", type=Path, required=True)
    create.add_argument("--destination", type=Path, required=True)
    for name in ("verify", "restore"):
        command = commands.add_parser(name)
        command.add_argument("--backup", type=Path, required=True)
        command.add_argument("--manifest-sha256", required=True)
        if name == "restore":
            command.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "create":
        path = create_backup(args.repo_root, args.plan, args.destination)
        print(json.dumps({"manifest": str(path), "manifest_sha256": sha256_file(path)}))
    elif args.command == "verify":
        report = verify_backup(args.backup, args.manifest_sha256)
        print(json.dumps({"verified_files": len(report.files), "plan_id": report.plan_id,
                          "different_filesystem": report.different_filesystem}))
    else:
        restore_backup(args.backup, args.manifest_sha256, args.destination)
        print("Restored and hash-checked. No model call, lease removal, or resume performed.")


if __name__ == "__main__":
    main()
