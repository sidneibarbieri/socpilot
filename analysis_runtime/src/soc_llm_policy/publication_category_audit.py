"""Check published telemetry categories against the declared policy vocabulary."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

NO_DECLARED_PATTERN = "<no declared pattern>"
SCANNED_TREES = ("artifact_data", "artifact_outputs", "scenario_banks")


class CategoryShapeFinding(BaseModel):
    """Location of a category outside the declared publication grammar."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str
    line: int = Field(ge=1)


def _declared_patterns(release_root: Path) -> frozenset[str]:
    global_dir = release_root / "artifact_data" / "global"
    rules = yaml.safe_load((global_dir / "constraints.yaml").read_text(encoding="utf-8"))
    patterns = {
        pattern.lower()
        for rule in rules
        for pattern in (rule.get("scope") or {}).get("category_contains") or []
    }
    normalization_path = global_dir / "telemetry_normalization.yaml"
    if normalization_path.is_file():
        normalization = yaml.safe_load(normalization_path.read_text(encoding="utf-8"))
        patterns.update(
            pattern.lower()
            for behavior in normalization["behavior_classes"]
            for pattern in behavior["category_patterns"]
        )
    if any(" " in pattern for pattern in patterns):
        raise ValueError("declared category vocabulary contains spaces")
    return frozenset(patterns)


def _valid_category(category: str, patterns: frozenset[str]) -> bool:
    if category in patterns:
        return True
    projection, separator, rank = category.rpartition(" #")
    if not separator or not rank.isascii() or not rank.isdecimal():
        return False
    if rank != f"{int(rank):03d}":
        return False
    if projection == NO_DECLARED_PATTERN:
        return True
    tokens = projection.split(" ")
    return tokens == sorted(set(tokens)) and all(token in patterns for token in tokens)


def audit_published_categories(release_root: Path) -> list[CategoryShapeFinding]:
    """Reject raw or decorated category prose without echoing its contents."""
    patterns = _declared_patterns(release_root)
    findings: list[CategoryShapeFinding] = []
    for tree_name in SCANNED_TREES:
        tree = release_root / tree_name
        if not tree.is_dir():
            continue
        for path in sorted(tree.rglob("incident_telemetry.jsonl")):
            relative_path = path.relative_to(release_root).as_posix()
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                event = json.loads(line)
                category = event.get("category") if isinstance(event, dict) else False
                if category is not None and not (
                    isinstance(category, str) and _valid_category(category, patterns)
                ):
                    findings.append(CategoryShapeFinding(path=relative_path, line=line_number))
    return findings
