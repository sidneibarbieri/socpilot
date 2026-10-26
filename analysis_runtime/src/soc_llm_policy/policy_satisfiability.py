"""Static satisfiability analysis for a policy rule set.

The engine rejects ordering cycles before it evaluates a plan. A rule set can
still be unsatisfiable without a cycle: a mandatory action that another rule
prohibits, or a mandatory action whose required predecessor is prohibited.
This module finds those conflicts over the rules alone, before any incident is
seen, so an inconsistent bundle is refused at authoring time instead of
surfacing as an unrepairable plan.

The analysis over-approximates. Two scoped rules are treated as able to fire
together unless their scopes are identical, so every reported conflict is
labelled ``certain`` (both rules fire whenever either does) or ``possible``
(they fire together only on incidents that match both scopes). A bundle with no
``certain`` and no ``possible`` conflict is satisfiable for every incident.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from itertools import product
from pathlib import Path

import yaml

from soc_llm_policy.engine import rule_scope_matches
from soc_llm_policy.io import IncidentMeta, PolicyRule


class ConflictKind(StrEnum):
    """The ways two or three rules can leave no satisfying action list."""

    required_and_prohibited = "required_and_prohibited"
    predecessor_prohibited = "predecessor_prohibited"
    required_needs_approval = "required_needs_approval"


class Certainty(StrEnum):
    certain = "certain"
    possible = "possible"


@dataclass(frozen=True)
class RuleConflict:
    kind: ConflictKind
    certainty: Certainty
    action: str
    rule_ids: tuple[str, ...]
    explanation: str

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "certainty": self.certainty.value,
            "action": self.action,
            "rule_ids": list(self.rule_ids),
            "explanation": self.explanation,
        }


def _scope_key(rule: PolicyRule) -> str:
    return json.dumps(rule.scope or {}, sort_keys=True)


def _co_application(rules: Iterable[PolicyRule]) -> Certainty:
    """Global rules always co-apply; scoped rules only when scopes coincide."""
    scoped = {_scope_key(rule) for rule in rules if rule.scope}
    return Certainty.certain if len(scoped) <= 1 else Certainty.possible


def _by_type(rules: list[PolicyRule], rule_type: str) -> list[PolicyRule]:
    return [rule for rule in rules if rule.type == rule_type]


def _required_and_prohibited(rules: list[PolicyRule]) -> list[RuleConflict]:
    conflicts = []
    for required, prohibited in product(
        _by_type(rules, "mandatory"), _by_type(rules, "prohibit_when")
    ):
        if required.action != prohibited.action:
            continue
        conflicts.append(
            RuleConflict(
                kind=ConflictKind.required_and_prohibited,
                certainty=_co_application((required, prohibited)),
                action=required.action,
                rule_ids=(required.rule_id, prohibited.rule_id),
                explanation=(
                    f"{required.rule_id} requires {required.action} and "
                    f"{prohibited.rule_id} prohibits it."
                ),
            )
        )
    return conflicts


def _predecessor_prohibited(rules: list[PolicyRule]) -> list[RuleConflict]:
    conflicts = []
    mandatory_actions = {rule.action: rule for rule in _by_type(rules, "mandatory")}
    for ordering, prohibited in product(
        _by_type(rules, "prohibit_before"), _by_type(rules, "prohibit_when")
    ):
        required = mandatory_actions.get(ordering.action)
        if required is None or ordering.condition_action != prohibited.action:
            continue
        conflicts.append(
            RuleConflict(
                kind=ConflictKind.predecessor_prohibited,
                certainty=_co_application((required, ordering, prohibited)),
                action=required.action,
                rule_ids=(required.rule_id, ordering.rule_id, prohibited.rule_id),
                explanation=(
                    f"{required.rule_id} requires {required.action}, "
                    f"{ordering.rule_id} needs {prohibited.action} before it, and "
                    f"{prohibited.rule_id} prohibits {prohibited.action}."
                ),
            )
        )
    return conflicts


def _required_needs_approval(rules: list[PolicyRule]) -> list[RuleConflict]:
    conflicts = []
    for required, gated in product(
        _by_type(rules, "mandatory"), _by_type(rules, "require_approval")
    ):
        if required.action != gated.action:
            continue
        conflicts.append(
            RuleConflict(
                kind=ConflictKind.required_needs_approval,
                certainty=_co_application((required, gated)),
                action=required.action,
                rule_ids=(required.rule_id, gated.rule_id),
                explanation=(
                    f"{required.rule_id} requires {required.action}, which "
                    f"{gated.rule_id} removes when approval is absent; the plan "
                    "is satisfiable only with an authenticated approval."
                ),
            )
        )
    return conflicts


def find_conflicts(rules: list[PolicyRule]) -> list[RuleConflict]:
    """Return every conflict the rule set admits, most severe kind first."""
    return [
        *_required_and_prohibited(rules),
        *_predecessor_prohibited(rules),
        *_required_needs_approval(rules),
    ]


def load_rules(path: Path) -> list[PolicyRule]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [PolicyRule.model_validate(entry) for entry in raw]


def _is_unconditional(conflict: RuleConflict) -> bool:
    return conflict.kind is not ConflictKind.required_needs_approval


def _load_incidents(corpus_dirs: list[Path]) -> list[tuple[IncidentMeta, list[dict]]]:
    incidents = []
    for corpus_dir in corpus_dirs:
        for incident_dir in sorted(path for path in corpus_dir.iterdir() if path.is_dir()):
            meta = IncidentMeta.model_validate_json(
                (incident_dir / "incident_meta.json").read_text(encoding="utf-8")
            )
            telemetry = [
                json.loads(line)
                for line in (incident_dir / "incident_telemetry.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            incidents.append((meta, telemetry))
    return incidents


def co_applicable_incidents(
    conflicts: list[RuleConflict],
    rules: list[PolicyRule],
    incidents: list[tuple[IncidentMeta, list[dict]]],
) -> list[set[int]]:
    """For each conflict, the incidents on which all of its rules apply at once."""
    by_id = {rule.rule_id: rule for rule in rules}
    return [
        {
            position
            for position, (meta, telemetry) in enumerate(incidents)
            if all(
                rule_scope_matches(meta, telemetry, by_id[rule_id].scope)
                for rule_id in conflict.rule_ids
            )
        }
        for conflict in conflicts
    ]


def build_report(
    rules: list[PolicyRule], corpus_dirs: list[Path] | None = None
) -> dict[str, object]:
    conflicts = find_conflicts(rules)
    unconditional = [conflict for conflict in conflicts if _is_unconditional(conflict)]
    entries = [conflict.as_dict() for conflict in conflicts]
    report: dict[str, object] = {
        "rule_count": len(rules),
        "conflict_count": len(conflicts),
        "certain_unconditional_conflict_count": sum(
            conflict.certainty is Certainty.certain for conflict in unconditional
        ),
        "possible_unconditional_conflict_count": sum(
            conflict.certainty is Certainty.possible for conflict in unconditional
        ),
        "approval_dependent_conflict_count": len(conflicts) - len(unconditional),
        "conflicts": entries,
    }
    if corpus_dirs:
        incidents = _load_incidents(corpus_dirs)
        affected = co_applicable_incidents(conflicts, rules, incidents)
        report["corpus_incident_count"] = len(incidents)
        for entry, positions in zip(entries, affected, strict=True):
            entry["co_applicable_incidents"] = len(positions)
        report["unconditional_affected_incident_count"] = len(
            set().union(*(positions for conflict, positions in zip(conflicts, affected, strict=True)
                          if _is_unconditional(conflict)))
        )
        report["approval_dependent_affected_incident_count"] = len(
            set().union(*(positions for conflict, positions in zip(conflicts, affected, strict=True)
                          if not _is_unconditional(conflict)))
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m soc_llm_policy.policy_satisfiability",
        description="Report rule subsets that no action list can satisfy.",
    )
    parser.add_argument("constraints", type=Path, help="constraints.yaml to analyze")
    parser.add_argument(
        "--corpus",
        type=Path,
        action="append",
        default=[],
        help="Incident directory to measure co-application on; repeatable.",
    )
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    report = build_report(load_rules(args.constraints), args.corpus)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(
        f"{report['rule_count']} rules: {report['certain_unconditional_conflict_count']} certain, "
        f"{report['possible_unconditional_conflict_count']} possible, "
        f"{report['approval_dependent_conflict_count']} approval-dependent conflict(s)."
    )
    for conflict in report["conflicts"]:
        measured = conflict.get("co_applicable_incidents")
        suffix = "" if measured is None else (
            f" [co-applies on {measured} of {report['corpus_incident_count']} incidents]"
        )
        print(f"  [{conflict['certainty']}] {conflict['kind']}: {conflict['explanation']}{suffix}")
    return 1 if report["certain_unconditional_conflict_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
