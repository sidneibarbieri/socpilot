"""Measured scope support of policy rules over an incident corpus.

Each rule's scope is evaluated against every incident with the same matcher the
enforcement engine uses. This produces an exact incident-level support count
with an explicit denominator. It does not claim that a structural rule is
applicable or violated before a proposed plan exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from soc_llm_policy.corpus_hash import canonical_corpus_sha256
from soc_llm_policy.engine import rule_scope_matches
from soc_llm_policy.io import (
    HumanAction,
    IncidentMeta,
    PolicyRule,
    TelemetryEvent,
    parse_human_actions,
    parse_incident_meta,
    parse_rules,
    parse_telemetry,
    read_json,
    read_jsonl,
    read_yaml_list,
)
from soc_llm_policy.paths import repo_relative_path, resolve_repo_root

_META_FILENAME = "incident_meta.json"
_TELEMETRY_FILENAME = "incident_telemetry.jsonl"
_HUMAN_ACTIONS_FILENAME = "incident_human_actions.jsonl"


@dataclass(frozen=True)
class IncidentPackage:
    incident_id: str
    meta: IncidentMeta
    telemetry: list[TelemetryEvent]
    human_actions: list[HumanAction]


class RuleScopeSupport(BaseModel):
    rule_id: str = Field(..., min_length=1)
    rule_type: str = Field(..., min_length=1)
    action: str = Field(..., min_length=1)
    scope_supported_incident_count: int = Field(..., ge=0)
    scope_support_fraction: float = Field(..., ge=0.0, le=1.0)


class PolicyScopeSupportReport(BaseModel):
    total_incidents: int = Field(..., ge=0)
    rules: list[RuleScopeSupport]


class ScopeSupportEvidence(BaseModel):
    schema_version: Literal[1] = 1
    corpus_path: str = Field(..., min_length=1)
    corpus_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    constraints_path: str = Field(..., min_length=1)
    constraints_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    report: PolicyScopeSupportReport


def load_incident_corpus(incidents_dir: Path) -> list[IncidentPackage]:
    """Load every incident package under a corpus directory."""
    packages: list[IncidentPackage] = []
    for incident_dir in sorted(incidents_dir.iterdir()):
        if not incident_dir.is_dir():
            continue
        packages.append(
            IncidentPackage(
                incident_id=incident_dir.name,
                meta=parse_incident_meta(read_json(incident_dir / _META_FILENAME)),
                telemetry=parse_telemetry(
                    read_jsonl(incident_dir / _TELEMETRY_FILENAME)
                ),
                human_actions=parse_human_actions(
                    read_jsonl(incident_dir / _HUMAN_ACTIONS_FILENAME)
                ),
            )
        )
    return packages


def measure_scope_support(
    corpus: list[IncidentPackage],
    rules: list[PolicyRule],
) -> PolicyScopeSupportReport:
    """Count, for each rule, the incidents whose scope the rule matches."""
    total = len(corpus)
    rows: list[RuleScopeSupport] = []
    for rule in rules:
        supported_count = sum(
            rule_scope_matches(package.meta, package.telemetry, rule.scope)
            for package in corpus
        )
        rows.append(
            RuleScopeSupport(
                rule_id=rule.rule_id,
                rule_type=rule.type,
                action=rule.action,
                scope_supported_incident_count=supported_count,
                scope_support_fraction=(
                    round(supported_count / total, 4) if total else 0.0
                ),
            )
        )
    return PolicyScopeSupportReport(total_incidents=total, rules=rows)


def build_scope_support_evidence(
    *,
    repo_root: Path,
    incidents_dir: Path,
    constraints_path: Path,
) -> ScopeSupportEvidence:
    corpus = load_incident_corpus(incidents_dir)
    rules = parse_rules(read_yaml_list(constraints_path))
    return ScopeSupportEvidence(
        corpus_path=repo_relative_path(incidents_dir, repo_root),
        corpus_sha256=canonical_corpus_sha256(incidents_dir),
        constraints_path=repo_relative_path(constraints_path, repo_root),
        constraints_sha256=hashlib.sha256(constraints_path.read_bytes()).hexdigest(),
        report=measure_scope_support(corpus, rules),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="soc_llm_policy.policy_scope_support")
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--incidents-dir", required=True)
    parser.add_argument("--constraints", required=True)
    parser.add_argument("--output-json", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    repo_root = resolve_repo_root(args.repo_root)
    incidents_dir = (repo_root / str(args.incidents_dir)).resolve()
    constraints_path = (repo_root / str(args.constraints)).resolve()
    output_path = (repo_root / str(args.output_json)).resolve()
    evidence = build_scope_support_evidence(
        repo_root=repo_root,
        incidents_dir=incidents_dir,
        constraints_path=constraints_path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(evidence.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
