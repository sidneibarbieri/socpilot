"""Publish failure diagnostics without machine paths, retaining source commitments."""
from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
from pathlib import Path

_RECEIPT = "artifact_outputs/analysis/traceback_publication.json"
_FRAME_PATH = re.compile(r'(File ")/[^"\n]*/([^/"\n]+)(", line [0-9]+)')


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Publication evidence must be a JSON object.")
    return value


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _path(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if Path(relative).is_absolute() or root.resolve() not in candidate.parents:
        raise ValueError("Publication path must remain inside the package.")
    return candidate


def _without_tracebacks(payload: dict) -> dict:
    result = copy.deepcopy(payload)
    for failure in result["failures"]:
        failure.pop("traceback", None)
    return result


def _semantic_digest(payload: dict) -> str:
    encoded = json.dumps(_without_tracebacks(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def redact_frame_paths(payload: dict) -> tuple[dict, int]:
    result = copy.deepcopy(payload)
    count = 0
    for failure in result["failures"]:
        traceback = failure.get("traceback")
        if traceback is None:
            continue
        if not isinstance(traceback, str):
            raise ValueError("Failure traceback must be text.")
        failure["traceback"], replacements = _FRAME_PATH.subn(
            r'\1<local-source>/\2\3', traceback,
        )
        count += replacements
    return result, count


def publish_tracebacks(root: Path, campaign_path: Path) -> None:
    """Transform only a staging publication copy, never the collection source."""
    campaign = _read(campaign_path)
    primary_entry = campaign["studies"]["primary"]["execution_evidence"]
    primary_path = _path(root, primary_entry["path"])
    primary = _read(primary_path)
    failure_entry = primary["snapshot"]["files"]["experiment_failures"]
    failure_path = _path(root, failure_entry["path"])
    original = _read(failure_path)
    projected, count = redact_frame_paths(original)
    if not count:
        return
    if _digest(primary_path) != primary_entry["sha256"] or _digest(failure_path) != failure_entry["sha256"]:
        raise ValueError("Publication source hashes differ before transformation.")
    archive = root / "artifact_outputs/analysis/publication_sources"
    archive.mkdir(parents=True, exist_ok=False)
    originals = {}
    for role, source in (("campaign", campaign_path), ("primary", primary_path)):
        destination = archive / f"{role}_manifest.json"
        shutil.copy2(source, destination)
        originals[role] = {"path": destination.relative_to(root).as_posix(), "sha256": _digest(source)}
    original_failure_hash = _digest(failure_path)
    _write(failure_path, projected)
    failure_entry["sha256"] = _digest(failure_path)
    _write(primary_path, primary)
    primary_entry["sha256"] = _digest(primary_path)
    _write(campaign_path, campaign)
    receipt = {
        "schema_version": 1,
        "transformation": "absolute_python_frame_paths_to_local_source_basename",
        "original_manifests": originals,
        "campaign_path": campaign_path.relative_to(root).as_posix(),
        "public_campaign_sha256": _digest(campaign_path),
        "original_failure_sha256": original_failure_hash,
        "public_failure_sha256": _digest(failure_path),
        "non_traceback_fields_sha256": _semantic_digest(original),
        "redacted_frame_count": count,
        "limitation": "Original machine paths are withheld; their removal cannot be independently reversed from the public copy.",
    }
    _write(root / _RECEIPT, receipt)
    provenance_path = root / "artifact_outputs/analysis/public_evidence_provenance.json"
    provenance = _read(provenance_path)
    provenance["terminal_evidence_bytes_preserved"] = False
    provenance["traceback_publication"] = {"path": _RECEIPT, "sha256": _digest(root / _RECEIPT)}
    provenance["transformations"].append(receipt["transformation"])
    _write(provenance_path, provenance)
    validate_traceback_projection(root, campaign_path, provenance)
    readme = root / "README.md"
    if readme.is_file():
        with readme.open("a", encoding="utf-8") as stream:
            stream.write(
                "\n## Published failure diagnostics\n\n"
                "Absolute machine paths in Python traceback frames are replaced by "
                "`<local-source>/filename`. Failure records, messages, identifiers, "
                "line numbers, and counts remain. The original manifests and hashes "
                "are retained under `artifact_outputs/analysis/publication_sources/`; "
                "`artifact_outputs/analysis/traceback_publication.json` records the "
                "transformation and new bindings. The public copy cannot recover "
                "or independently verify the withheld directory names. Original "
                "collection files remain privately preserved.\n"
            )


def validate_traceback_projection(root: Path, campaign_path: Path, provenance: dict) -> None:
    binding = provenance["traceback_publication"]
    receipt_path = _path(root, binding["path"])
    if binding["path"] != _RECEIPT or _digest(receipt_path) != binding["sha256"]:
        raise ValueError("Traceback publication receipt hash differs.")
    receipt = _read(receipt_path)
    if receipt["schema_version"] != 1 or receipt["transformation"] != "absolute_python_frame_paths_to_local_source_basename":
        raise ValueError("Unsupported traceback publication transformation.")
    originals = {}
    for role in ("campaign", "primary"):
        entry = receipt["original_manifests"][role]
        source = _path(root, entry["path"])
        if _digest(source) != entry["sha256"]:
            raise ValueError("Original publication manifest hash differs.")
        originals[role] = _read(source)
    source_campaign = receipt["original_manifests"]["campaign"]["sha256"]
    source_primary = receipt["original_manifests"]["primary"]["sha256"]
    if source_campaign != provenance["source_campaign_evidence_sha256"] or source_primary != provenance["source_execution_evidence_sha256"]:
        raise ValueError("Publication original manifests do not match provenance.")
    primary_entry = originals["campaign"]["studies"]["primary"]["execution_evidence"]
    if primary_entry["sha256"] != source_primary:
        raise ValueError("Original campaign does not bind original primary evidence.")
    primary_path = _path(root, primary_entry["path"])
    failure_entry = originals["primary"]["snapshot"]["files"]["experiment_failures"]
    if failure_entry["sha256"] != receipt["original_failure_sha256"]:
        raise ValueError("Original failure commitment differs.")
    failure_path = _path(root, failure_entry["path"])
    if _digest(failure_path) != receipt["public_failure_sha256"]:
        raise ValueError("Public failure hash differs.")
    if _semantic_digest(_read(failure_path)) != receipt["non_traceback_fields_sha256"]:
        raise ValueError("Non-traceback failure fields changed.")
    failure_entry["sha256"] = _digest(failure_path)
    if _read(primary_path) != originals["primary"]:
        raise ValueError("Primary evidence changed beyond the failure-file binding.")
    primary_entry["sha256"] = _digest(primary_path)
    if campaign_path.resolve() != _path(root, receipt["campaign_path"]) or _digest(campaign_path) != receipt["public_campaign_sha256"]:
        raise ValueError("Public campaign binding differs.")
    if _read(campaign_path) != originals["campaign"]:
        raise ValueError("Campaign changed beyond the primary-evidence binding.")
    if provenance["terminal_evidence_bytes_preserved"] is not False:
        raise ValueError("Publication must disclose changed diagnostic bytes.")
