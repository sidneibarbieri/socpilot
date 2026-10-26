#!/usr/bin/env python3
"""Emit specification-coverage evidence: every frozen rule must activate under a
documented scenario, so no rule is left inert. Deterministic, no model calls.

Reads the frozen policy and the frozen coverage scenarios, runs the verifier on
each scenario, and records which rules fire. Exits nonzero if any scenario fails
to activate the rule it targets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from soc_llm_policy.engine import enforce_policy
from soc_llm_policy.io import parse_action_catalog, parse_rules, read_yaml_list


def _repo_root(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def _evaluate_scenario(
    scenario: dict[str, Any],
    rules: list[Any],
    catalog: list[Any],
) -> dict[str, Any]:
    violations, enforced = enforce_policy(
        llm_actions=list(scenario["proposed_plan"]),
        telemetry=list(scenario.get("telemetry", [])),
        rules=rules,
        catalog=catalog,
    )
    fired = sorted({violation.rule_id for violation in violations})
    expected = list(scenario.get("expect_fires", []))
    missing = [rule_id for rule_id in expected if rule_id not in fired]
    return {
        "scenario_id": scenario["scenario_id"],
        "targets_rule": scenario["targets_rule"],
        "family": scenario["family"],
        "rules_fired": fired,
        "expected_fires": expected,
        "activation_ok": not missing,
        "missing": missing,
        "violations": [
            {"rule_id": v.rule_id, "type": v.type, "action": v.action, "repair": v.repair}
            for v in violations
        ],
        "enforced_plan": enforced,
    }


def _resolve_policy_dir(repo_root: Path) -> Path:
    """The frozen policy lives in policy/ in the source tree and in
    artifact_data/global/ inside the packaged artifact."""
    for candidate in (repo_root / "policy", repo_root / "artifact_data" / "global"):
        if (candidate / "action_catalog.yaml").exists():
            return candidate
    raise FileNotFoundError(
        f"No policy directory with action_catalog.yaml under {repo_root}"
    )


def build_report(repo_root: Path) -> dict[str, Any]:
    policy_dir = _resolve_policy_dir(repo_root)
    catalog = parse_action_catalog(read_yaml_list(policy_dir / "action_catalog.yaml"))
    rules = parse_rules(read_yaml_list(policy_dir / "constraints.yaml"))
    scenarios = read_yaml_list(policy_dir / "coverage_scenarios.yaml")

    results = [_evaluate_scenario(scenario, rules, catalog) for scenario in scenarios]
    families_covered = sorted({r["family"] for r in results if r["activation_ok"]})
    rules_activated = sorted(
        {rule_id for r in results if r["activation_ok"] for rule_id in r["expected_fires"]}
    )
    return {
        "declared_rules": [rule.rule_id for rule in rules],
        "rules_activated_under_coverage": rules_activated,
        "families_covered": families_covered,
        "all_declared_rules_activate": set(rules_activated)
        == {rule.rule_id for rule in rules},
        "scenarios": results,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument(
        "--output-json",
        default="results/analysis/rule_coverage_report.json",
    )
    args = parser.parse_args(argv)

    repo_root = _repo_root(args.repo_root)
    report = build_report(repo_root)

    output_path = repo_root / args.output_json
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Rule coverage report saved at: {output_path}")
    print(f"Declared rules:     {report['declared_rules']}")
    print(f"Activated under coverage: {report['rules_activated_under_coverage']}")
    print(f"All declared rules activate: {report['all_declared_rules_activate']}")

    failures = [r for r in report["scenarios"] if not r["activation_ok"]]
    if failures:
        for failure in failures:
            print(
                f"  [FAIL] {failure['scenario_id']} did not activate "
                f"{failure['missing']}"
            )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
