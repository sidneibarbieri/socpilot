from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from soc_llm_policy.paths import repo_relative_path


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _load_stix_bundle(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid STIX bundle (expected object): {path}")
    return data


def _extract_external_id(obj: dict[str, Any]) -> str | None:
    refs = obj.get("external_references", [])
    if not isinstance(refs, list):
        return None
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        if ref.get("source_name") != "mitre-attack":
            continue
        external_id = ref.get("external_id")
        if isinstance(external_id, str) and external_id:
            return external_id
    return None


def _is_active(obj: dict[str, Any]) -> bool:
    return not bool(obj.get("revoked")) and not bool(obj.get("x_mitre_deprecated"))


def build_mitre_manifest(stix_path: Path, *, repo_root: Path | None = None) -> dict[str, Any]:
    """Generate reproducible metadata for the MITRE ATT&CK base used in runs."""
    bundle = _load_stix_bundle(stix_path)
    objects = bundle.get("objects", [])
    if not isinstance(objects, list):
        raise ValueError(f"STIX bundle missing 'objects' list: {stix_path}")

    techniques: dict[str, str] = {}
    tactics: dict[str, str] = {}
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        if not _is_active(obj):
            continue
        obj_type = obj.get("type")
        external_id = _extract_external_id(obj)
        name = obj.get("name")
        if not isinstance(external_id, str) or not isinstance(name, str):
            continue
        if obj_type == "attack-pattern":
            techniques[external_id] = name
        elif obj_type == "x-mitre-tactic":
            tactics[external_id] = name
    modified_values = sorted(
        str(obj.get("modified", ""))
        for obj in objects
        if isinstance(obj, dict) and obj.get("modified")
    )
    latest_modified = modified_values[-1] if modified_values else ""

    stix_reference = (
        repo_relative_path(stix_path, repo_root)
        if repo_root is not None
        else str(stix_path)
    )

    return {
        "stix_path": stix_reference,
        "sha256": _sha256(stix_path),
        "object_count": len(objects),
        "technique_count": len(techniques),
        "tactic_count": len(tactics),
        "latest_modified": latest_modified,
    }


def write_mitre_manifest(
    stix_path: Path,
    manifest_path: Path,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    manifest = build_mitre_manifest(stix_path, repo_root=repo_root)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def read_mitre_manifest(manifest_path: Path) -> dict[str, Any]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid MITRE manifest (expected object): {manifest_path}")
    return data
