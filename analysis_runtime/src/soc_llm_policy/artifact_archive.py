"""Archive verified package files and reconstructible corpus views, without publishing."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import os
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, Field, field_validator

from soc_llm_policy.artifact_verify import verify_artifact_package
from soc_llm_policy.corpus_campaign import materialize_campaign_view

_MANIFEST = "artifact_manifest.json"
_CAMPAIGN = "artifact_outputs/analysis/corpus_campaign.json"


class ArchiveFile(BaseModel):
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("path")
    @classmethod
    def normalized_path(cls, value: str) -> str:
        relative = PurePosixPath(value)
        if (not value or relative.is_absolute() or ".." in relative.parts
                or relative.as_posix() != value or "\\" in value or value == "."):
            raise ValueError("Archive paths must be normalized package-relative paths.")
        return value


class ArchiveInventory(BaseModel):
    files: list[ArchiveFile] = Field(min_length=1)

    @field_validator("files")
    @classmethod
    def unique_files(cls, entries: list[ArchiveFile]) -> list[ArchiveFile]:
        names = [entry.path for entry in entries]
        if len(names) != len(set(names)) or _MANIFEST in names:
            raise ValueError("Archive inventory has duplicate or reserved paths.")
        return entries


def _digest(path: Path) -> str:
    with path.open("rb") as reader:
        return hashlib.file_digest(reader, "sha256").hexdigest()


def _require_verified(root: Path) -> None:
    report = verify_artifact_package(root)
    if report["status"] != "ok":
        raise ValueError(f"Package verification failed: {report['issues']}")


def _regular_file(root: Path, relative: str) -> Path:
    path = root / relative
    if any(parent.is_symlink() for parent in (path, *path.parents) if parent != root):
        raise ValueError(f"Archive file resolves through a symbolic link: {relative}")
    if not path.is_file() or not path.resolve().is_relative_to(root):
        raise ValueError(f"Archive file is not an internal regular file: {relative}")
    return path


def _copy_inventory(source: Path, staging: Path, inventory: ArchiveInventory) -> None:
    for entry in inventory.files:
        original = _regular_file(source, entry.path)
        target = staging / entry.path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
        if _digest(target) != entry.sha256 or _digest(original) != entry.sha256:
            raise ValueError(f"Manifest file changed during archive preparation: {entry.path}")


def _portable_header(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    info.pax_headers = {}
    return info


def _write_archive(staging: Path, target: Path) -> None:
    with target.open("wb") as writer, gzip.GzipFile(
        filename="", fileobj=writer, mode="wb", mtime=0
    ) as compressed, tarfile.open(fileobj=compressed, mode="w", dereference=False) as archive:
        for path in sorted(staging.rglob("*")):
            if path.is_symlink():
                if Path(os.readlink(path)).is_absolute() or not path.resolve(strict=True).is_relative_to(staging):
                    raise ValueError("Derived corpus link escapes the archive.")
            archive.add(path, arcname=path.relative_to(staging).as_posix(),
                        recursive=False, filter=_portable_header)


def archive_package(source: Path, destination: Path) -> Path:
    """Create a new private archive; never modify source evidence or an existing output."""
    source = source.resolve(strict=True)
    destination = destination.absolute()
    if destination.resolve().is_relative_to(source):
        raise ValueError("Archive destination must be outside the source package.")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    manifest = _regular_file(source, _MANIFEST).read_bytes()
    inventory = ArchiveInventory.model_validate_json(manifest)
    _require_verified(source)
    with tempfile.TemporaryDirectory(prefix=".artifact-archive-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "package"
        staging.mkdir()
        _copy_inventory(source, staging, inventory)
        (staging / _MANIFEST).write_bytes(manifest)
        if any(entry.path == _CAMPAIGN for entry in inventory.files):
            materialize_campaign_view(repo_root=staging, manifest_path=staging / _CAMPAIGN)
        _require_verified(staging)
        archive = Path(temporary) / "package.tgz"
        _write_archive(staging, archive)
        archive.chmod(0o600)
        if (source / _MANIFEST).read_bytes() != manifest:
            raise ValueError("Source manifest changed during archive preparation.")
        _require_verified(source)
        os.link(archive, destination)
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args(argv)
    print(archive_package(args.source, args.destination))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
