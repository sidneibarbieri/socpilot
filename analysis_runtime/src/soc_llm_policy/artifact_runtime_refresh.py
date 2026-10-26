"""Refresh offline reanalysis without replacing collection evidence."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from soc_llm_policy.artifact_package import (
    _build_manifest,
    _copy_analysis_runtime,
    _render_packaged_reviewer_dashboard,
    _require_residue_free_release,
    _sha256_file,
)
from soc_llm_policy.artifact_verify import verify_artifact_package
from soc_llm_policy.corpus_campaign import materialize_campaign_view


class ManifestFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        relative = Path(value)
        if not value or relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Manifest paths must remain inside the package.")
        return value


def _require_verified(
    package_root: Path, *, allow_legacy_preview: bool = False
) -> None:
    report = verify_artifact_package(package_root)
    if report["status"] == "ok":
        return
    if allow_legacy_preview:
        manifest = json.loads((package_root / "artifact_manifest.json").read_text())
        bound_paths = {entry["path"] for entry in manifest["files"]}
        issues = report["issues"]
        if (
            len(issues) == 1
            and issues[0]["code"] == "invalid_dashboard_preview_binding"
            and issues[0]["path"] == "docs/img/reviewer_dashboard.preview.json"
            and "docs/img/reviewer_dashboard.png" in bound_paths
            and "docs/img/reviewer_dashboard.preview.json" not in bound_paths
        ):
            return
    raise ValueError(f"Package verification failed: {report['issues']}")


def _replaceable(relative: str) -> bool:
    return relative in {
        "README.md",
        "reproduce.sh",
        "REVIEWER_DASHBOARD.html",
        "artifact_outputs/dashboard/index.html",
        "artifact_outputs/analysis/runtime_refresh_receipt.json",
        "docs/img/reviewer_dashboard.png",
        "docs/img/reviewer_dashboard.preview.json",
    } or relative.startswith("analysis_runtime/")


def _copy_bound_files(source: Path, staging: Path, entries: list[ManifestFile]) -> None:
    for entry in entries:
        if _replaceable(entry.path) and entry.path != "README.md":
            continue
        original = source / entry.path
        if original.is_symlink() or not original.resolve().is_relative_to(source):
            raise ValueError(f"Manifest file escapes package: {entry.path}")
        target = staging / entry.path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
        if _sha256_file(target) != entry.sha256:
            raise ValueError(f"Manifest file changed during copy: {entry.path}")


def _copy_pre_execution_source_files(
    source: Path, staging: Path, entries: list[ManifestFile]
) -> None:
    """Copy immutable source-package inputs while dropping a legacy preview."""
    preview_path = "docs/img/reviewer_dashboard.png"
    for entry in entries:
        if entry.path == preview_path:
            continue
        original = source / entry.path
        if original.is_symlink() or not original.resolve().is_relative_to(source):
            raise ValueError(f"Manifest file escapes package: {entry.path}")
        target = staging / entry.path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
        if _sha256_file(target) != entry.sha256:
            raise ValueError(f"Manifest file changed during copy: {entry.path}")


def _write_refreshed_manifest(staging: Path, original: dict[str, Any]) -> None:
    rebuilt = _build_manifest(
        release_root=staging,
        incidents=original["incidents"],
        include_analysis=original["include_analysis"],
        run_id=original["run_id"],
        heldout_incidents=original.get("heldout_incidents", []),
        artifact_stage=original["artifact_stage"],
        provider_incidents=original.get("provider_incidents", []),
        campaign_evidence_path=original.get("campaign_evidence_path"),
    )
    manifest = {**original, **rebuilt}
    (staging / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def _copy_preview(
    staging: Path,
    entries: list[ManifestFile],
    preview_root: Path | None,
) -> None:
    preview_relative = "docs/img/reviewer_dashboard.png"
    has_preview = any(entry.path == preview_relative for entry in entries)
    if preview_root is not None and not has_preview:
        raise ValueError("Parent package has no bound dashboard preview.")
    if not has_preview:
        return
    selected = (
        preview_root.resolve(strict=True)
        if preview_root is not None
        else Path(__file__).resolve().parents[2] / "docs/img"
    )
    for name in ("reviewer_dashboard.png", "reviewer_dashboard.preview.json"):
        target = staging / "docs/img" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(selected / name, target)


def refresh_runtime(
    source: Path,
    destination: Path,
    *,
    preview_root: Path | None = None,
) -> Path:
    """Create a verified derivative, leaving the original package unchanged."""
    source = source.resolve(strict=True)
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(destination)
    if destination.is_relative_to(source):
        raise ValueError("Destination must not be inside the source package.")
    parent_manifest = source / "artifact_manifest.json"
    parent_manifest_sha256 = _sha256_file(parent_manifest)
    original = json.loads(parent_manifest.read_text())
    if original.get("artifact_stage") != "reported_results":
        raise ValueError("Runtime refresh requires a reported-results package.")
    entries = [ManifestFile.model_validate(entry) for entry in original["files"]]
    if len({entry.path for entry in entries}) != len(entries):
        raise ValueError("Manifest contains duplicate paths.")
    _require_verified(source, allow_legacy_preview=preview_root is not None)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".runtime-refresh-", dir=destination.parent
    ) as temporary:
        staging = Path(temporary) / "package"
        staging.mkdir()
        _copy_bound_files(source, staging, entries)
        campaign_manifest = "artifact_outputs/analysis/corpus_campaign.json"
        if any(entry.path == campaign_manifest for entry in entries):
            materialize_campaign_view(
                repo_root=staging, manifest_path=staging / campaign_manifest
            )
        _copy_analysis_runtime(staging)
        _copy_preview(staging, entries, preview_root)
        _render_packaged_reviewer_dashboard(staging)
        preserved = [entry for entry in entries if not _replaceable(entry.path)]
        for entry in preserved:
            if _sha256_file(staging / entry.path) != entry.sha256:
                raise ValueError(f"Collection evidence changed: {entry.path}")
        receipt = {
            "schema_version": 1,
            "purpose": "offline_runtime_refresh_not_new_collection",
            "parent_manifest_sha256": parent_manifest_sha256,
            "preserved_file_count": len(preserved),
        }
        receipt_path = (
            staging / "artifact_outputs/analysis/runtime_refresh_receipt.json"
        )
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        _require_residue_free_release(staging)
        _write_refreshed_manifest(staging, original)
        _require_verified(staging)
        if _sha256_file(parent_manifest) != parent_manifest_sha256:
            raise ValueError("Parent manifest changed during runtime refresh.")
        staging.rename(destination)
    return destination


def refresh_pre_execution_source(source: Path, destination: Path) -> Path:
    """Repair only a pre-execution package's unbound legacy preview.

    The derivative preserves every manifest-bound source and protocol file. It
    intentionally drops the optional dashboard image because a pre-execution
    package contains no rendered result dashboard to which the image can bind.
    """
    source = source.resolve(strict=True)
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(destination)
    if destination.is_relative_to(source):
        raise ValueError("Destination must not be inside the source package.")
    parent_manifest = source / "artifact_manifest.json"
    parent_manifest_sha256 = _sha256_file(parent_manifest)
    original = json.loads(parent_manifest.read_text())
    if original.get("artifact_stage") != "pre_execution_source":
        raise ValueError("Source refresh requires a pre-execution package.")
    entries = [ManifestFile.model_validate(entry) for entry in original["files"]]
    if len({entry.path for entry in entries}) != len(entries):
        raise ValueError("Manifest contains duplicate paths.")
    if "docs/img/reviewer_dashboard.png" not in {entry.path for entry in entries}:
        raise ValueError("Source package has no legacy dashboard preview to remove.")
    _require_verified(source, allow_legacy_preview=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".source-refresh-", dir=destination.parent
    ) as temporary:
        staging = Path(temporary) / "package"
        staging.mkdir()
        _copy_pre_execution_source_files(source, staging, entries)
        _require_residue_free_release(staging)
        _write_refreshed_manifest(staging, original)
        _require_verified(staging)
        if _sha256_file(parent_manifest) != parent_manifest_sha256:
            raise ValueError("Parent manifest changed during source refresh.")
        staging.rename(destination)
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument(
        "--preview-root",
        type=Path,
        help="Directory with reviewer_dashboard.png and its receipt bound to refreshed HTML.",
    )
    parser.add_argument(
        "--pre-execution-source",
        action="store_true",
        help="Remove only an unbound legacy preview from a pre-execution package.",
    )
    args = parser.parse_args(argv)
    if args.pre_execution_source:
        if args.preview_root is not None:
            raise ValueError("A pre-execution source refresh cannot accept a preview.")
        refreshed = refresh_pre_execution_source(args.source, args.destination)
    else:
        refreshed = refresh_runtime(
            args.source, args.destination, preview_root=args.preview_root
        )
    print(refreshed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
