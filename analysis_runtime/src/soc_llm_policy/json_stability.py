from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from soc_llm_policy.durable_io import atomic_write_bytes


def _strip_generated_at(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_generated_at(item)
            for key, item in value.items()
            if key != "generated_at_utc"
        }
    if isinstance(value, list):
        return [_strip_generated_at(item) for item in value]
    return value


def stabilize_generated_at(existing: Any, current: Any) -> Any:
    if _strip_generated_at(existing) == _strip_generated_at(current):
        return existing
    return current


def write_stable_json(path: Path, payload: Any) -> Any:
    stabilized = payload
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        stabilized = stabilize_generated_at(existing, payload)
    output_text = json.dumps(stabilized, ensure_ascii=False, indent=2)
    if path.exists() and path.read_text(encoding="utf-8") == output_text:
        return stabilized
    atomic_write_bytes(path, output_text.encode("utf-8"))
    return stabilized
