from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from soc_llm_policy.engine import telemetry_matches_scope
from soc_llm_policy.io import (
    HumanAction,
    PolicyRule,
    TelemetryEvent,
    parse_human_actions_strict_order,
    parse_rules,
    parse_telemetry,
    read_json,
    read_jsonl,
    read_yaml_list,
)
from soc_llm_policy.paths import resolve_repo_root


@dataclass(frozen=True)
class IncidentTrace:
    actions: tuple[str, ...]
    telemetry: tuple[TelemetryEvent, ...]


@dataclass(frozen=True)
class AuditPaths:
    dataset_dir: Path
    constraints_path: Path
    official_bundle_path: Path
    repeat_assignments_path: Path | None
    output_json_path: Path
    output_markdown_path: Path
    additional_dataset_dirs: tuple[Path, ...] = ()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_incident_traces(dataset_dir: Path) -> list[IncidentTrace]:
    traces: list[IncidentTrace] = []
    for incident_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
        human_actions: list[HumanAction] = parse_human_actions_strict_order(
            read_jsonl(
                incident_dir / "incident_human_actions.jsonl",
                strict=True,
            )
        )
        telemetry = parse_telemetry(
            read_jsonl(
                incident_dir / "incident_telemetry.jsonl",
                strict=True,
            )
        )
        traces.append(
            IncidentTrace(
                actions=tuple(action.action_id for action in human_actions),
                telemetry=tuple(telemetry),
            )
        )
    return traces


def _scope_support(rule: PolicyRule, traces: list[IncidentTrace]) -> int:
    if (rule.scope or {}).get("global") is True:
        return len(traces)
    return sum(
        telemetry_matches_scope(list(trace.telemetry), rule.scope) for trace in traces
    )


def _human_rule_support(
    rule: PolicyRule, traces: list[IncidentTrace]
) -> dict[str, int]:
    trigger_count = 0
    violation_count = 0
    for trace in traces:
        actions = trace.actions
        if rule.type == "mandatory":
            if not telemetry_matches_scope(list(trace.telemetry), rule.scope):
                continue
            trigger_count += 1
            violation_count += rule.action not in actions
        elif rule.type == "prohibit_before":
            if rule.action not in actions:
                continue
            trigger_count += 1
            prerequisite = rule.condition_action
            violation_count += (
                prerequisite is None
                or prerequisite not in actions
                or actions.index(prerequisite) > actions.index(rule.action)
            )
        elif rule.type == "require_approval":
            trigger_count += rule.action in actions
    return {
        "trigger_count": trigger_count,
        "violation_count": violation_count,
    }


def _official_bundle_evidence(bundle: dict[str, Any]) -> dict[str, Any]:
    rows = bundle.get("incidents", [])
    if not isinstance(rows, list):
        rows = []
    rows_with_actions = sum(
        isinstance(row, dict) and isinstance(row.get("llm_actions"), list)
        for row in rows
    )
    summary = bundle.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}
    violations_by_rule = summary.get("violations_by_rule", {})
    if not isinstance(violations_by_rule, dict):
        violations_by_rule = {}
    return {
        "row_count": len(rows),
        "rows_with_llm_actions": rows_with_actions,
        "violations_by_rule": {
            str(rule_id): int(count) for rule_id, count in violations_by_rule.items()
        },
    }


def _assignment_evidence(assignments_payload: dict[str, Any]) -> dict[str, int]:
    assignments = assignments_payload.get("assignments", [])
    if not isinstance(assignments, list):
        assignments = []
    existing_source_count = 0
    for assignment in assignments:
        if not isinstance(assignment, dict):
            continue
        source_path = assignment.get("source_path")
        if isinstance(source_path, str) and Path(source_path).is_file():
            existing_source_count += 1
    return {
        "assignment_count": len(assignments),
        "directly_available_source_count": existing_source_count,
        "missing_source_count": len(assignments) - existing_source_count,
    }


def _precedence_support(
    traces: list[IncidentTrace],
    prerequisite: str,
    action: str,
) -> dict[str, Any]:
    action_support = [trace for trace in traces if action in trace.actions]
    both_support = [trace for trace in action_support if prerequisite in trace.actions]
    prerequisite_first = sum(
        trace.actions.index(prerequisite) < trace.actions.index(action)
        for trace in both_support
    )
    return {
        "prerequisite": prerequisite,
        "action": action,
        "action_trace_count": len(action_support),
        "both_actions_trace_count": len(both_support),
        "prerequisite_first_count": prerequisite_first,
        "prerequisite_first_rate": (
            round(prerequisite_first / len(both_support), 4) if both_support else None
        ),
        "interpretation": "descriptive_regularities_only",
    }


def build_report(paths: AuditPaths) -> dict[str, Any]:
    dataset_dirs = (paths.dataset_dir, *paths.additional_dataset_dirs)
    incident_names = [path.name for directory in dataset_dirs for path in directory.iterdir() if path.is_dir()]
    if len(incident_names) != len(set(incident_names)):
        raise ValueError("Policy audit datasets contain duplicate incident identifiers.")
    traces = [trace for directory in dataset_dirs for trace in _load_incident_traces(directory)]
    rules = parse_rules(read_yaml_list(paths.constraints_path))
    bundle = read_json(paths.official_bundle_path)
    assignments = read_json(paths.repeat_assignments_path) if paths.repeat_assignments_path else {"assignments": []}
    official = _official_bundle_evidence(bundle)
    assignment_evidence = _assignment_evidence(assignments)
    violations_by_rule = official["violations_by_rule"]

    rule_rows = []
    for rule in rules:
        observed_violation_count = int(violations_by_rule.get(rule.rule_id, 0))
        rule_rows.append(
            {
                "rule_id": rule.rule_id,
                "type": rule.type,
                "action": rule.action,
                "condition_action": rule.condition_action,
                "scope_support_incident_count": _scope_support(rule, traces),
                "human_trace_support": _human_rule_support(rule, traces),
                "official_llm_observed_violation_count": observed_violation_count,
                "official_llm_status": (
                    "observed_violation"
                    if observed_violation_count > 0
                    else "no_observed_violation"
                ),
            }
        )

    bundle_action_trace_ready = (
        official["row_count"] > 0
        and official["rows_with_llm_actions"] == official["row_count"]
    )
    assigned_raw_outputs_ready = (
        assignment_evidence["assignment_count"] > 0
        and assignment_evidence["missing_source_count"] == 0
    )
    posthoc_ready = bundle_action_trace_ready or assigned_raw_outputs_ready
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_hashes": {
            "constraints_sha256": _sha256(paths.constraints_path),
            "official_bundle_sha256": _sha256(paths.official_bundle_path),
            "repeat_assignments_sha256": _sha256(paths.repeat_assignments_path) if paths.repeat_assignments_path else None,
        },
        "corpus": {
            "incident_count": len(traces),
            "human_action_counts": dict(
                sorted(
                    Counter(
                        action for trace in traces for action in trace.actions
                    ).items()
                )
            ),
        },
        "official_evidence": {
            **official,
            **assignment_evidence,
            "bundle_action_trace_ready": bundle_action_trace_ready,
            "assigned_raw_outputs_ready": assigned_raw_outputs_ready,
            "posthoc_candidate_rule_evaluation_ready": posthoc_ready,
            "raw_output_availability_basis": "direct_source_path_exists" if paths.repeat_assignments_path else "not_assessed_bundle_actions_only",
            "limitation": (
                None
                if posthoc_ready
                else (
                    "Sanitized rows omit action lists or assigned raw outputs "
                    "are missing; new predicates cannot be evaluated on the "
                    "fixed provider campaign."
                )
            ),
        },
        "declared_rule_surface": rule_rows,
        "descriptive_human_regularities": [
            _precedence_support(traces, "collect_forensics", "isolate_host"),
            _precedence_support(
                traces,
                "collect_forensics",
                "reset_admin_credentials",
            ),
        ],
        "claim_boundary": {
            "official_rule_results": "Only observed violations in the preserved campaign evidence.",
            "human_regularities": (
                "Hypothesis generation, not normative policy evidence."
            ),
            "authored_witnesses": (
                "Conformance evidence only, never corpus-rate estimates."
            ),
        },
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    evidence = report["official_evidence"]
    rule_lines = [
        "| Rule | Type | Corpus scope | Human triggers | "
        "Official violations | Status |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in report["declared_rule_surface"]:
        row_template = (
            "| {rule_id} | {type} | {scope} | {triggers} | {violations} | {status} |"
        )
        rule_lines.append(
            row_template.format(
                rule_id=row["rule_id"],
                type=row["type"],
                scope=row["scope_support_incident_count"],
                triggers=row["human_trace_support"]["trigger_count"],
                violations=row["official_llm_observed_violation_count"],
                status=row["official_llm_status"],
            )
        )
    lines = [
        "# Policy Rule Audit",
        "",
        "## Evidence availability",
        "",
        f"- Preserved provider rows: `{evidence['row_count']}`",
        f"- Rows retaining action lists: `{evidence['rows_with_llm_actions']}`",
        f"- Repeat assignments: `{evidence['assignment_count']}`",
        "- Directly available assigned raw outputs: "
        f"`{evidence['directly_available_source_count']}`",
        "- Post-hoc candidate-rule evaluation ready: "
        f"`{str(evidence['posthoc_candidate_rule_evaluation_ready']).lower()}`",
        f"- Limitation: {evidence['limitation'] or 'none'}",
        "",
        "## Declared rule set",
        "",
        *rule_lines,
        "",
        "## Claim boundary",
        "",
        "- Official rule results come only from observed violations in the "
        "preserved campaign evidence.",
        "- Human ordering frequencies generate hypotheses; they do not "
        "establish policy.",
        "- Authored witnesses test conformance and are not corpus-rate estimators.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="soc_llm_policy.policy_surface_audit")
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--dataset-dir", default="dataset/incidents")
    parser.add_argument("--additional-dataset-dir", action="append", default=[])
    parser.add_argument("--bundle-actions-only", action="store_true", help="Assess retained action lists without a legacy repeat-assignment source.")
    parser.add_argument(
        "--constraints",
        default="policy/constraints.yaml",
    )
    parser.add_argument(
        "--official-bundle",
        default="results/analysis/official_llm_analysis_bundle.json",
    )
    parser.add_argument(
        "--repeat-assignments",
        default="results/analysis/repeat_stability/repeat_assignments.json",
    )
    parser.add_argument(
        "--output-json",
        default="results/analysis/policy_surface_audit.json",
    )
    parser.add_argument(
        "--output-markdown",
        default="results/analysis/policy_surface_audit.md",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    repo_root = resolve_repo_root(args.repo_root)
    paths = AuditPaths(
        dataset_dir=(repo_root / args.dataset_dir).resolve(),
        constraints_path=(repo_root / args.constraints).resolve(),
        official_bundle_path=(repo_root / args.official_bundle).resolve(),
        repeat_assignments_path=None if args.bundle_actions_only else (repo_root / args.repeat_assignments).resolve(),
        additional_dataset_dirs=tuple((repo_root / item).resolve() for item in args.additional_dataset_dir),
        output_json_path=(repo_root / args.output_json).resolve(),
        output_markdown_path=(repo_root / args.output_markdown).resolve(),
    )
    report = build_report(paths)
    _write_json(paths.output_json_path, report)
    _write_markdown(paths.output_markdown_path, report)
    print(f"Wrote policy rule audit: {paths.output_json_path}")
    print(
        "Post-hoc candidate-rule evaluation ready: "
        f"{report['official_evidence']['posthoc_candidate_rule_evaluation_ready']}"
    )


if __name__ == "__main__":
    main()
