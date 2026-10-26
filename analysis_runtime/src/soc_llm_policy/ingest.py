"""
ETL layer: converts raw SIEM dataset events into the normalized format
expected by the engine (incident_telemetry.jsonl).

Output format (per line):
{
    "event_type": str,          # mapped from "Event Name"
    "category": str,            # mapped from "Low Level Category"
    "timestamp": str,           # mapped from "Start Time"
    "source_type": str,         # source tag: EDR, IPS, LINUX_OS, etc.
    "source_ip": str | null,
    "dest_ip": str | null,
    "username": str | null,
    "details": {
        "command": str | null,  # field queried by the engine
        "severity": str | null,
        "log_source": str | null,
        "source_event": dict    # source event after local sanitization
    }
}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_SOURCE_MAP: dict[str, str] = {
    "dataset-alsd": "ALSD",
    "dataset-bloodhoundv2": "BLOODHOUND",
    "dataset-eap": "EAP",
    "dataset-edr": "EDR",
    "dataset-ips": "IPS",
    "dataset-rbmws": "RBMWS",
    "dataset-utm": "UTM",
    "datset-linuxos": "LINUX_OS",  # intentional typo in original filename
    "dataset-linuxos": "LINUX_OS",
}


def _resolve_source_type(path: Path) -> str:
    stem = path.stem.lower()
    for prefix, stype in _SOURCE_MAP.items():
        if stem.startswith(prefix):
            return stype
    return stem.upper()


def _get(raw: dict[str, Any], *keys: str) -> str | None:
    """Return the first non-empty value found among keys."""
    for key in keys:
        value = raw.get(key)
        if value and str(value).strip() not in ("", "N/A", "null", "None"):
            return str(value).strip()
    return None


def _strip_quotes(value: str | None) -> str | None:
    """Remove extra quotes added by some Fortinet exporters."""
    if value is None:
        return None
    return value.strip('"').strip("'")


def _extract_severity(raw: dict[str, Any]) -> str | None:
    candidates = [
        _get(raw, "CS-Severity (custom)"),
        _strip_quotes(_get(raw, "FortinetSeverity (custom)")),
        _get(raw, "Threat Severity (custom)"),
        _strip_quotes(_get(raw, "Fortinet Level (custom)")),
    ]
    for candidate in candidates:
        if candidate:
            return candidate
    return None


def _extract_command(raw: dict[str, Any]) -> str | None:
    """
    Extract the most relevant command field depending on source.
    Priority: Command (custom) -> Command Line (custom) -> Parent Command Line.
    """
    return _get(
        raw,
        "Command (custom)",
        "Command Line (custom)",
        "Parent Command Line (custom)",
        "Grandparent Command Line (custom)",
    )


def normalize_event(raw: dict[str, Any], source_type: str) -> dict[str, Any]:
    """Convert a raw SIEM event to the pipeline canonical format."""
    return {
        "event_type": _get(raw, "Event Name") or "unknown",
        "category": _get(raw, "Low Level Category") or "unknown",
        "timestamp": _get(raw, "Start Time"),
        "source_type": source_type,
        "source_ip": _get(raw, "Source IP"),
        "dest_ip": _get(raw, "Destination IP"),
        "username": _get(raw, "Username", "Account Name (custom)"),
        "details": {
            "command": _extract_command(raw),
            "severity": _extract_severity(raw),
            "log_source": _get(raw, "Log Source", "Fortinet Device Name (custom)"),
            "source_event": raw,
        },
    }


def load_dataset(path: Path) -> list[dict[str, Any]]:
    """Load a SIEM dataset JSON file (list of events)."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Dataset must be a JSON list: {path}")
    return data


def convert_dataset_to_telemetry(
    dataset_path: Path,
    output_path: Path,
    *,
    overwrite: bool = False,
) -> int:
    """
    Read a dataset JSON file, normalize all events, and write JSONL.

    Returns the number of written events.
    """
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Telemetry file already exists (use overwrite=True): {output_path}"
        )

    source_type = _resolve_source_type(dataset_path)
    raw_events = load_dataset(dataset_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as out:
        for raw in raw_events:
            if not isinstance(raw, dict):
                continue
            normalized = normalize_event(raw, source_type)
            out.write(json.dumps(normalized, ensure_ascii=False) + "\n")
            count += 1

    return count


def merge_datasets_to_telemetry(
    dataset_paths: list[Path],
    output_path: Path,
    *,
    overwrite: bool = False,
    strict: bool = False,
) -> int:
    """
    Merge multiple datasets into one telemetry JSONL file.
    Useful to assemble one incident from multiple sources.

    Returns total number of written events.
    """
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Telemetry file already exists (use overwrite=True): {output_path}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0

    with output_path.open("w", encoding="utf-8") as out:
        for path in dataset_paths:
            source_type = _resolve_source_type(path)
            try:
                raw_events = load_dataset(path)
            except (FileNotFoundError, ValueError) as exc:
                if strict:
                    raise
                print(f"Warning: skipping {path}: {exc}")
                continue

            for raw in raw_events:
                if not isinstance(raw, dict):
                    if strict:
                        raise ValueError(
                            f"Invalid event (not an object) in {path}: {raw!r}"
                        )
                    continue
                normalized = normalize_event(raw, source_type)
                out.write(json.dumps(normalized, ensure_ascii=False) + "\n")
                total += 1

    return total
