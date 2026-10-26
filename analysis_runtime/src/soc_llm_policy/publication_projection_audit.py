"""Compare policy applicability before and after a publication transformation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from soc_llm_policy.engine import IncidentMeta, rule_scope_matches
from soc_llm_policy.publication_redaction import (
    CategoryIndex,
    RedactionVocabulary,
    project_normalization,
    project_policy_rules,
    read_public_aliases,
)


class ProjectionAudit(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    incident_count: int
    rule_count: int
    checks: int
    differences: int
    boundary: str = (
        "Rule applicability only; provider outputs and full verdicts are not replayed."
    )


def _telemetry(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _telemetry_paths(root: Path) -> list[Path]:
    trees = (root / "artifact_data", root / "scenario_banks")
    return sorted(
        path.relative_to(root)
        for tree in trees
        if tree.is_dir()
        for path in tree.rglob("incident_telemetry.jsonl")
    )


def _require_matching_event_inventories(
    source_events: list[dict], published_events: list[dict]
) -> None:
    identity_fields = ("event_id", "timestamp", "event_type", "source_type")
    source_ids = [
        tuple(event.get(field) for field in identity_fields) for event in source_events
    ]
    published_ids = [
        tuple(event.get(field) for field in identity_fields)
        for event in published_events
    ]
    if source_ids != published_ids:
        raise ValueError("Source and published event inventories differ")


def _require_matching_normalization(
    source_data: Path,
    published_data: Path,
    vocabulary: RedactionVocabulary | None,
) -> None:
    relative = Path("global/telemetry_normalization.yaml")
    source_path = source_data / relative
    published_path = published_data / relative
    if source_path.is_file() != published_path.is_file():
        raise ValueError("Source and published normalization inventories differ")
    if not source_path.is_file():
        return
    source_normalization = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    published_normalization = yaml.safe_load(published_path.read_text(encoding="utf-8"))
    expected = (
        project_normalization(source_normalization, vocabulary)
        if vocabulary is not None
        else source_normalization
    )
    if expected != published_normalization:
        raise ValueError("Published normalization differs from authorized projection")


def audit_projection(
    source: Path,
    published: Path,
    *,
    public_aliases: tuple[tuple[str, str], ...] = (),
) -> ProjectionAudit:
    """Check inventories and authorized rules, then compare rule applicability."""
    source_data = source / "artifact_data"
    published_data = published / "artifact_data"
    original_paths = _telemetry_paths(source)
    published_paths = _telemetry_paths(published)
    if not original_paths or original_paths != published_paths:
        raise ValueError(
            "Source and published incident inventories must be identical and nonempty"
        )
    rules = yaml.safe_load(
        (source_data / "global/constraints.yaml").read_text(encoding="utf-8")
    )
    published_rules = yaml.safe_load(
        (published_data / "global/constraints.yaml").read_text(encoding="utf-8")
    )
    expected_rules = rules
    vocabulary = None
    categories = None
    if public_aliases:
        global_dir = source_data / "global"
        vocabulary = RedactionVocabulary.from_policy(
            global_dir / "constraints.yaml",
            global_dir / "telemetry_normalization.yaml",
            public_aliases,
        )
        incident_types = {
            IncidentMeta.model_validate_json(
                (source / path).with_name("incident_meta.json").read_text()
            ).incident_type
            for path in original_paths
        }
        categories = CategoryIndex.build(set(), vocabulary, incident_types)
        expected_rules = project_policy_rules(rules, vocabulary, categories)
    if not rules or expected_rules != published_rules:
        raise ValueError("Published policy rules must match the authorized projection")
    _require_matching_normalization(source_data, published_data, vocabulary)
    differences = 0
    for relative_path in original_paths:
        source_path = source / relative_path
        published_path = published / relative_path
        original = _telemetry(source_path)
        redacted = _telemetry(published_path)
        _require_matching_event_inventories(original, redacted)
        source_meta = IncidentMeta.model_validate_json(
            source_path.with_name("incident_meta.json").read_text()
        )
        published_meta = IncidentMeta.model_validate_json(
            published_path.with_name("incident_meta.json").read_text()
        )
        expected_type = (
            categories.echo(source_meta.incident_type)
            if categories is not None
            else source_meta.incident_type
        )
        if published_meta.incident_type != expected_type:
            raise ValueError(
                "Published incident type differs from authorized projection"
            )
        for source_rule, published_rule in zip(rules, published_rules, strict=True):
            differences += rule_scope_matches(
                source_meta, original, source_rule.get("scope")
            ) != rule_scope_matches(
                published_meta, redacted, published_rule.get("scope")
            )
    return ProjectionAudit(
        incident_count=len(original_paths),
        rule_count=len(rules),
        checks=len(original_paths) * len(rules),
        differences=differences,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--published", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--public-aliases", type=Path)
    args = parser.parse_args()
    aliases = read_public_aliases(args.public_aliases) if args.public_aliases else ()
    report = audit_projection(args.source, args.published, public_aliases=aliases)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(report.model_dump_json())
    return 1 if report.differences else 0


if __name__ == "__main__":
    raise SystemExit(main())
