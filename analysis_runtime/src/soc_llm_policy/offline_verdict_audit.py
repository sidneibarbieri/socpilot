"""Recompute recorded policy decisions without provider calls or source writes."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import replace
from functools import cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from soc_llm_policy.engine import EnforcementRequest, enforce_policy, rule_scope_matches
from soc_llm_policy.family_metrics import evaluate_cell_families
from soc_llm_policy.io import (
    IncidentMeta,
    TelemetryEvent,
    parse_action_catalog,
    parse_incident_meta,
    parse_rules,
    parse_telemetry,
    read_json,
    read_jsonl,
    read_yaml_list,
)
from soc_llm_policy.paths import RepoPaths
from soc_llm_policy.policy_selection import operational_rules


class RepairCellOutcome(BaseModel):
    """Paired verifier outcomes under the same incident and policy context."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    original_violating: bool
    changed_plan: bool
    stable_repaired_plan: bool
    residual_violating: bool
    active_requirement_prohibition_conflict: bool
    empty_proposed: bool
    empty_repaired: bool


class RepairAuditSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    total_cells: int = Field(ge=1)
    original_violating_cells: int = Field(default=0, ge=0)
    changed_plan_cells: int = Field(default=0, ge=0)
    stable_repaired_plan_cells: int = Field(default=0, ge=0)
    residual_violating_cells: int = Field(default=0, ge=0)
    active_requirement_prohibition_conflict_cells: int = Field(default=0, ge=0)
    empty_proposed_cells: int = Field(default=0, ge=0)
    empty_repaired_cells: int = Field(default=0, ge=0)
    scope: str = (
        "Violating means at least one verifier decision, including deferrals. "
        "Plan changes use exact ordered-list comparison. Empty lists are not "
        "classified as model refusals. Counts cover only audited saved cells."
        " Active requirement/prohibition conflicts count co-applicable rules "
        "that require and forbid the same action, not general unsatisfiability."
    )


class VerdictAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    cells_checked: int
    records_sha256: str
    decisions_sha256: str
    inputs_sha256: str
    verifier_source_sha256: str
    repair_outcomes: RepairAuditSummary
    scope: str = (
        "Recorded actions, violations, repaired plans, and policy-family verdicts "
        "recomputed with the maintained verifier. This does not rerun model inference, "
        "validate historical model acquisition, or certify publication safety."
    )


def _digest_files(root: Path, files: set[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _has_active_requirement_prohibition_conflict(request: EnforcementRequest) -> bool:
    """Identify a direct contradiction in the evaluated incident context."""
    required: set[str] = set()
    prohibited: set[str] = set()
    for rule in request.rules:
        if rule.type not in {"mandatory", "prohibit_when"}:
            continue
        if not rule_scope_matches(request.incident_meta, request.telemetry, rule.scope):
            continue
        if rule.type == "mandatory":
            required.add(rule.action)
        else:
            prohibited.add(rule.action)
    return bool(required & prohibited)


def compare_decision(request: EnforcementRequest, recorded: dict) -> RepairCellOutcome:
    """Reject any difference in the decision, including repair order."""
    violations, repaired = enforce_policy(request)
    families = [
        item.model_dump(mode="json") for item in evaluate_cell_families(request)
    ]
    expected = {
        "llm_actions": request.llm_actions,
        "violations": [
            dict(item.to_dict(), missing_before=item.missing_before)
            for item in violations
        ],
        "enforced_actions": repaired,
        "policy_family_outcomes": families,
    }
    for field, value in expected.items():
        preserved = recorded.get(field)
        if field == "violations" and isinstance(preserved, list):
            if not all(isinstance(item, dict) for item in preserved):
                raise ValueError(
                    "Recorded violations differs from the required object schema"
                )
            preserved = [
                dict(item, missing_before=item.get("missing_before"))
                for item in preserved
            ]
        if preserved != value:
            raise ValueError(f"Recorded {field} differs from offline recomputation")
    residual, second_plan = enforce_policy(replace(request, llm_actions=repaired))
    return RepairCellOutcome(
        original_violating=bool(violations),
        changed_plan=request.llm_actions != repaired,
        stable_repaired_plan=second_plan == repaired,
        residual_violating=bool(residual),
        active_requirement_prohibition_conflict=_has_active_requirement_prohibition_conflict(request),
        empty_proposed=not request.llm_actions,
        empty_repaired=not repaired,
    )


class VerdictInputPaths(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_directory: Path
    incident_directories: tuple[Path, ...] = ()


def _inside_workspace(root: Path, selected: Path) -> Path:
    path = (selected if selected.is_absolute() else root / selected).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Audit inputs must remain inside the declared workspace")
    return path


def _input_paths(
    paths: RepoPaths, policy_root: Path | None, incident_roots: tuple[Path, ...]
) -> VerdictInputPaths:
    if (policy_root is None) != (not incident_roots):
        raise ValueError("Declare both policy and incident input roots")
    return VerdictInputPaths(
        policy_directory=(
            _inside_workspace(paths.repo_root, policy_root)
            if policy_root is not None
            else paths.outputs_action_catalog_path.parent
        ),
        incident_directories=tuple(
            _inside_workspace(paths.repo_root, source) for source in incident_roots
        ),
    )


def _incident_package(
    paths: RepoPaths, inputs: VerdictInputPaths, incident_id: str
) -> Path:
    if not inputs.incident_directories:
        return paths.outputs_incident_dir(incident_id)
    matches = [
        source / incident_id for source in inputs.incident_directories
        if (source / incident_id).is_dir()
    ]
    if len(matches) != 1:
        raise ValueError("Incident input is missing or ambiguous across declared roots")
    return _inside_workspace(paths.repo_root, matches[0])


def audit_verdicts(
    root: Path, run_id: str, expected_cells: int, *,
    policy_input_root: Path | None = None,
    incident_input_roots: tuple[Path, ...] = (),
    record_order: Literal["ascending", "descending"] = "ascending",
) -> VerdictAudit:
    """Audit every versioned decision in one declared study, retaining repeats."""
    paths = RepoPaths(repo_root=root.resolve())
    if record_order not in {"ascending", "descending"}:
        raise ValueError("Unknown verdict record traversal order")
    inputs = _input_paths(paths, policy_input_root, incident_input_roots)
    catalog_path = inputs.policy_directory / "action_catalog.yaml"
    constraints_path = inputs.policy_directory / "constraints.yaml"
    catalog = parse_action_catalog(read_yaml_list(catalog_path))
    rules = operational_rules(parse_rules(read_yaml_list(constraints_path)))

    @cache
    def incident_inputs(incident_id: str) -> tuple[list[TelemetryEvent], IncidentMeta]:
        package = _incident_package(paths, inputs, incident_id)
        return (
            parse_telemetry(
                read_jsonl(
                    package / "incident_telemetry.jsonl", strict=True
                )
            ),
            parse_incident_meta(
                read_json(package / "incident_meta.json")
            ),
        )

    cell_ids: set[str] = set()
    decisions: dict[str, dict[str, object]] = {}
    digest = hashlib.sha256()
    input_files = {catalog_path, constraints_path}
    repair_counts: Counter[str] = Counter()
    audited_paths: set[Path] = set()
    record_paths = sorted(
        paths.outputs_incidents_dir.glob("*/verifier_output_*.json"),
        reverse=record_order == "descending",
    )
    for path in record_paths:
        recorded = read_json(path)
        if recorded.get("experiment_run_id") != run_id:
            continue
        cell_id = recorded.get("experiment_cell_id")
        if not isinstance(cell_id, str) or not cell_id or cell_id in cell_ids:
            raise ValueError("Missing or duplicate experiment cell identity")
        cell_ids.add(cell_id)
        raw_path = path.with_name(
            path.name.replace("verifier_output_", "llm_raw_actions_", 1)
        )
        raw = read_json(raw_path)
        identity_matches = (
            raw.get("incident_id") == recorded.get("incident_id") == path.parent.name
            and raw.get("arm") == recorded.get("llm_arm")
            and raw.get("model") == recorded.get("llm_deployment")
            and raw.get("actions") == recorded.get("llm_actions")
        )
        if not identity_matches:
            raise ValueError(
                "Raw actions and recorded decision have different identity or actions"
            )
        telemetry, meta = incident_inputs(path.parent.name)
        package = _incident_package(paths, inputs, path.parent.name)
        input_files.update(
            {
                package / "incident_telemetry.jsonl",
                package / "incident_meta.json",
            }
        )
        request = EnforcementRequest(
            llm_actions=raw["actions"],
            telemetry=telemetry,
            rules=rules,
            catalog=catalog,
            incident_approved_actions=set(recorded["incident_approved_actions"]),
            approval_policy_mode=recorded["approval_policy_mode"],
            incident_meta=meta,
        )
        outcome = compare_decision(request, recorded)
        decisions[cell_id] = {
            field: recorded[field]
            for field in (
                "llm_actions",
                "violations",
                "enforced_actions",
                "policy_family_outcomes",
            )
        }
        repair_counts.update(
            f"{name}_cells" for name, observed in outcome.model_dump().items()
            if observed
        )
        audited_paths.add(path)
    if len(cell_ids) != expected_cells:
        raise ValueError(
            f"Expected {expected_cells} unique cells, found {len(cell_ids)}"
        )
    for path in sorted(audited_paths):
        raw_path = path.with_name(
            path.name.replace("verifier_output_", "llm_raw_actions_", 1)
        )
        for evidence in (path, raw_path):
            digest.update(str(evidence.relative_to(root.resolve())).encode())
            digest.update(b"\0")
            digest.update(hashlib.sha256(evidence.read_bytes()).digest())
    source_root = Path(__file__).resolve().parent
    return VerdictAudit(
        run_id=run_id,
        cells_checked=len(cell_ids),
        records_sha256=digest.hexdigest(),
        decisions_sha256=hashlib.sha256(
            json.dumps(decisions, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        inputs_sha256=_digest_files(root.resolve(), input_files),
        verifier_source_sha256=_digest_files(
            source_root, set(source_root.glob("*.py"))
        ),
        repair_outcomes=RepairAuditSummary(total_cells=len(cell_ids), **repair_counts),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-cells", type=int, required=True)
    parser.add_argument("--record-order", choices=("ascending", "descending"), default="ascending")
    parser.add_argument(
        "--policy-input-root", type=Path,
        help="Policy input directory inside the workspace (requires incident roots).",
    )
    parser.add_argument(
        "--incident-input-root", type=Path, action="append", default=[],
        help="Incident input directory inside the workspace; repeat for each partition.",
    )
    args = parser.parse_args()
    if args.expected_cells < 1:
        parser.error("--expected-cells must be positive")
    print(
        audit_verdicts(
            args.repo_root, args.run_id, args.expected_cells,
            policy_input_root=args.policy_input_root,
            incident_input_roots=tuple(args.incident_input_root),
            record_order=args.record_order,
        ).model_dump_json(indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
