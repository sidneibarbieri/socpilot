from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from soc_llm_policy.io import parse_action_catalog, parse_rules, read_yaml_list
from soc_llm_policy.json_stability import write_stable_json
from soc_llm_policy.paths import RepoPaths, repo_relative_path, resolve_repo_root
from soc_llm_policy.pipeline import list_inbox_incidents
from soc_llm_policy.publication_redaction import expected_digest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_hash(path: Path) -> str | None:
    return _sha256_file(path) if path.exists() else None


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return raw


def _incident_ids(
    paths: RepoPaths, include_all: bool, incidents_csv: str | None
) -> list[str]:
    if include_all:
        return list_inbox_incidents(paths)
    if not incidents_csv:
        raise ValueError("Provide --incidents or --all.")
    return sorted({item.strip() for item in incidents_csv.split(",") if item.strip()})


def _attack_mapping_summary(
    paths: RepoPaths, summary: dict[str, Any]
) -> dict[str, Any]:
    raw = yaml.safe_load(paths.attack_mapping_path.read_text(encoding="utf-8"))
    items = raw if isinstance(raw, list) else []
    techniques = {
        str(item.get("technique_id"))
        for item in items
        if isinstance(item, dict) and str(item.get("technique_id") or "").strip()
    }
    tactics = {
        str(item.get("tactic_id"))
        for item in items
        if isinstance(item, dict) and str(item.get("tactic_id") or "").strip()
    }
    matched_techniques = summary.get("attack_technique_counts", {})
    matched_tactics = summary.get("attack_tactic_counts", {})
    return {
        "path": repo_relative_path(paths.attack_mapping_path, paths.repo_root),
        "status": "heuristic_coverage_enrichment",
        "rule_count": len(items),
        "technique_count": len(techniques),
        "tactic_count": len(tactics),
        "corpus_matched_technique_count": (
            len(matched_techniques) if isinstance(matched_techniques, dict) else 0
        ),
        "corpus_matched_tactic_count": (
            len(matched_tactics) if isinstance(matched_tactics, dict) else 0
        ),
    }


def build_global_provenance(
    *, paths: RepoPaths, incident_ids: list[str]
) -> dict[str, Any]:
    freeze = _read_json_object(paths.outputs_analysis_dir / "protocol_freeze.json")
    summary = _read_json_object(paths.outputs_analysis_dir / "summary.json")
    catalog = parse_action_catalog(read_yaml_list(paths.inbox_action_catalog_path))
    rules = parse_rules(read_yaml_list(paths.inbox_constraints_path))

    constrained_actions: dict[str, list[str]] = defaultdict(list)
    for rule in rules:
        constrained_actions[rule.action].append(rule.rule_id)

    action_items = [
        {
            "action_id": action.action_id,
            "requires_approval": action.requires_approval,
            "reversible": action.reversible,
            "constrained_by_rule_ids": sorted(
                constrained_actions.get(action.action_id, [])
            ),
            "provenance_basis": [
                "research_authored_action_catalog",
                "content_addressed_protocol_input",
            ],
        }
        for action in catalog
    ]
    rule_items = [
        {
            "rule_id": rule.rule_id,
            "type": rule.type,
            "severity": rule.severity,
            "action": rule.action,
            "condition_action": rule.condition_action,
            "scope": rule.scope or {},
            "evidence_tier": rule.evidence_tier,
            "decision_record": "docs/POLICY_SURFACE_DERIVATION.md",
            "provenance_basis": [
                "research_reviewed_experimental_policy",
                "content_addressed_protocol_input",
            ],
        }
        for rule in rules
    ]
    operational_count = sum(
        1 for rule in rule_items if rule["evidence_tier"] == "operational"
    )
    specification_count = len(rule_items) - operational_count
    attack_mapping = _attack_mapping_summary(paths, summary)

    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "dataset_release_id": freeze.get("dataset_release_id", "unknown"),
        "eval_protocol_version": freeze.get("eval_protocol_version", "unknown"),
        "incident_count": len(incident_ids),
        "input_hashes": {
            "action_catalog": _sha256_file(paths.inbox_action_catalog_path),
            "constraints": _sha256_file(paths.inbox_constraints_path),
            "policy_review_record": _optional_hash(paths.policy_review_record_path),
            "attack_mapping": _sha256_file(paths.attack_mapping_path),
        },
        "global_artifact_scope": {
            "action_catalog_count": len(action_items),
            "policy_rule_count": len(rule_items),
            "operational_policy_rule_count": operational_count,
            "specification_policy_rule_count": specification_count,
            "attack_mapping_rule_count": attack_mapping["rule_count"],
        },
        "action_catalog": {
            "path": repo_relative_path(
                paths.inbox_action_catalog_path, paths.repo_root
            ),
            "items": action_items,
        },
        "policy_rules": {
            "path": repo_relative_path(paths.inbox_constraints_path, paths.repo_root),
            "items": rule_items,
        },
        "attack_mapping": attack_mapping,
        "notes": [
            "The corpus contains no observed incident-response action sequence.",
            "Workflow tasks are not treated as response actions or approvals.",
            "Only operational-tier rules enter prompts, verdicts, and rates.",
        ],
    }


def provenance_consistency_issues(
    *, paths: RepoPaths, incident_ids: list[str], published: dict[str, Any]
) -> list[str]:
    expected = build_global_provenance(paths=paths, incident_ids=incident_ids)
    compared_fields = (
        "dataset_release_id",
        "eval_protocol_version",
        "incident_count",
        "global_artifact_scope",
        "action_catalog",
        "policy_rules",
        "attack_mapping",
        "notes",
    )
    issues = [
        field_name
        for field_name in compared_fields
        if published.get(field_name) != expected.get(field_name)
    ]
    published_hashes = published.get("input_hashes")
    expected_hashes = expected["input_hashes"]
    input_paths = {
        "action_catalog": paths.inbox_action_catalog_path,
        "constraints": paths.inbox_constraints_path,
        "policy_review_record": paths.policy_review_record_path,
        "attack_mapping": paths.attack_mapping_path,
    }
    if not isinstance(published_hashes, dict) or set(published_hashes) != set(input_paths):
        issues.append("input_hashes")
    elif any(
        expected_hashes[role]
        != (
            expected_digest(paths.repo_root, path, published_hashes[role])
            if isinstance(published_hashes[role], str)
            else published_hashes[role]
        )
        for role, path in input_paths.items()
    ):
        issues.append("input_hashes")
    return issues


def _render_markdown(report: dict[str, Any]) -> str:
    scope = report["global_artifact_scope"]
    lines = [
        "# Global Artifact Provenance",
        "",
        f"- Dataset release: `{report['dataset_release_id']}`",
        f"- Protocol version: `{report['eval_protocol_version']}`",
        f"- Incident count: `{report['incident_count']}`",
        f"- Action catalog entries: `{scope['action_catalog_count']}`",
        f"- Operational policy rules: `{scope['operational_policy_rule_count']}`",
        f"- Specification-only rules: `{scope['specification_policy_rule_count']}`",
        "",
        "## Evidence boundary",
        "",
        "- The corpus contains no observed incident-response action sequence.",
        "- Workflow tasks are not response actions or approval records.",
        "- Specification-only rules are excluded from prompts, verdicts, and rates.",
        "",
        "## Policy rules",
    ]
    for item in report["policy_rules"]["items"]:
        lines.append(
            f"- `{item['rule_id']}`: `{item['type']}` on "
            f"`{item['action']}` (`{item['evidence_tier']}`)."
        )
    attack = report["attack_mapping"]
    lines.extend(
        [
            "",
            "## ATT&CK enrichment",
            "",
            f"- Mapping rules: `{attack['rule_count']}`",
            f"- Techniques: `{attack['technique_count']}`",
            f"- Tactics: `{attack['tactic_count']}`",
            "- Status: heuristic coverage enrichment, not ground truth.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="soc_llm_policy.global_provenance")
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--incidents", default=None)
    parser.add_argument(
        "--output-json", default="results/analysis/global_artifact_provenance.json"
    )
    parser.add_argument(
        "--output-md", default="results/analysis/global_artifact_provenance.md"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = resolve_repo_root(args.repo_root)
    paths = RepoPaths(repo_root=repo_root)
    incident_ids = _incident_ids(paths, bool(args.all), args.incidents)
    report = build_global_provenance(paths=paths, incident_ids=incident_ids)
    output_json = Path(args.output_json).expanduser().resolve()
    output_md = Path(args.output_md).expanduser().resolve()
    write_stable_json(output_json, report)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(_render_markdown(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
