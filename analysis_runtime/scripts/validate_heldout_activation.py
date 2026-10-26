"""Cross-batch activation transfer for content-triggered policy rules.

This tool measures each rule's scope support on the policy-development corpus
and on a disjoint transfer-check export batch. It uses the identical
scope matcher as the verifier and records whether support reaches a
prespecified transfer floor. The result is activation transfer, not evidence
of cross-organization or temporal generalization.

Determinism: the report is a pure function of the two frozen corpora and the
constraints file; every input is hashed into the output so the verdict replays.
No provider calls, no randomness.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from soc_llm_policy.io import parse_rules, read_yaml_list
from soc_llm_policy.paths import repo_relative_path, resolve_repo_root
from soc_llm_policy.policy_predicates import predicate_digest
from soc_llm_policy.policy_scope_support import (
    ScopeSupportEvidence,
    build_scope_support_evidence,
)

# The floor separates recurring support from a single observed category form.
# It does not establish external validity beyond these two export batches.
_TRANSFER_MIN_INCIDENTS = 3


class RuleTransfer(BaseModel):
    rule_id: str = Field(..., min_length=1)
    rule_type: str = Field(..., min_length=1)
    action: str = Field(..., min_length=1)
    # A global rule matches every incident by construction; only a
    # content_triggered rule carries genuine, discriminating corpus evidence.
    scope_kind: str = Field(..., pattern=r"^(global|content_triggered)$")
    derivation_incidents: int = Field(..., ge=0)
    derivation_fraction: float = Field(..., ge=0.0, le=1.0)
    heldout_incidents: int = Field(..., ge=0)
    heldout_fraction: float = Field(..., ge=0.0, le=1.0)
    meets_transfer_floor: bool


class HeldoutActivationReport(BaseModel):
    schema_version: int = 1
    generated_at_utc: str
    constraints_path: str
    constraints_sha256: str
    constraints_predicate_sha256: str
    derivation_corpus_path: str
    derivation_corpus_sha256: str
    derivation_incident_count: int = Field(..., ge=0)
    heldout_corpus_path: str
    heldout_corpus_sha256: str
    heldout_incident_count: int = Field(..., ge=0)
    rules: list[RuleTransfer]
    transfer_rule_ids: list[str]


# Resolve the string annotations produced by ``from __future__ import
# annotations`` so the model works whether imported or run as a script.
HeldoutActivationReport.model_rebuild()


def _scope_kinds(constraints_path: Path) -> dict[str, str]:
    """Map each rule id to global or content_triggered by its declared scope."""
    kinds: dict[str, str] = {}
    for rule in parse_rules(read_yaml_list(constraints_path)):
        kinds[rule.rule_id] = "content_triggered" if rule.scope else "global"
    return kinds


def _classify(
    derivation: ScopeSupportEvidence,
    heldout: ScopeSupportEvidence,
    scope_kinds: dict[str, str],
) -> list[RuleTransfer]:
    """Join the two per-rule reports and apply the transfer floor.

    A global rule matches unconditionally, so transfer-check support carries no
    discriminating evidence and is never counted as transfer coverage.
    """
    heldout_by_id = {row.rule_id: row for row in heldout.report.rules}
    rows: list[RuleTransfer] = []
    for deriv_row in derivation.report.rules:
        held_row = heldout_by_id[deriv_row.rule_id]
        kind = scope_kinds[deriv_row.rule_id]
        meets_transfer_floor = (
            kind == "content_triggered"
            and held_row.scope_supported_incident_count
            >= _TRANSFER_MIN_INCIDENTS
        )
        rows.append(
            RuleTransfer(
                rule_id=deriv_row.rule_id,
                rule_type=deriv_row.rule_type,
                action=deriv_row.action,
                scope_kind=kind,
                derivation_incidents=deriv_row.scope_supported_incident_count,
                derivation_fraction=deriv_row.scope_support_fraction,
                heldout_incidents=held_row.scope_supported_incident_count,
                heldout_fraction=held_row.scope_support_fraction,
                meets_transfer_floor=meets_transfer_floor,
            )
        )
    return rows


def build_report(
    *,
    repo_root: Path,
    derivation_dir: Path,
    heldout_dir: Path,
    constraints_path: Path,
) -> HeldoutActivationReport:
    derivation = build_scope_support_evidence(
        repo_root=repo_root,
        incidents_dir=derivation_dir,
        constraints_path=constraints_path,
    )
    heldout = build_scope_support_evidence(
        repo_root=repo_root,
        incidents_dir=heldout_dir,
        constraints_path=constraints_path,
    )
    rows = _classify(derivation, heldout, _scope_kinds(constraints_path))
    return HeldoutActivationReport(
        generated_at_utc=datetime.now(UTC).isoformat(),
        constraints_path=repo_relative_path(constraints_path, repo_root),
        constraints_sha256=derivation.constraints_sha256,
        constraints_predicate_sha256=predicate_digest(constraints_path),
        derivation_corpus_path=derivation.corpus_path,
        derivation_corpus_sha256=derivation.corpus_sha256,
        derivation_incident_count=derivation.report.total_incidents,
        heldout_corpus_path=heldout.corpus_path,
        heldout_corpus_sha256=heldout.corpus_sha256,
        heldout_incident_count=heldout.report.total_incidents,
        rules=rows,
        transfer_rule_ids=[r.rule_id for r in rows if r.meets_transfer_floor],
    )


def render_markdown(report: HeldoutActivationReport) -> str:
    lines = [
        "# Cross-Batch Policy Activation",
        "",
        f"Generated: `{report.generated_at_utc}`",
        "",
        f"- Policy-development batch: `{report.derivation_incident_count}` incidents "
        f"(`{report.derivation_corpus_sha256[:12]}`)",
        f"- Transfer-check batch: `{report.heldout_incident_count}` incidents "
        f"(`{report.heldout_corpus_sha256[:12]}`)",
        f"- Constraints: `{report.constraints_path}` "
        f"(`{report.constraints_sha256[:12]}`)",
        "",
    ]

    def _table(
        title: str,
        rows: Sequence[RuleTransfer],
        note: str,
    ) -> list[str]:
        block = [f"## {title}", ""]
        if not rows:
            return [*block, "(none)", ""]
        block += [
            "| Rule | Action | Policy development | Transfer check | Meets floor |",
            "| --- | --- | --- | --- | --- |",
        ]
        for row in rows:
            block.append(
                f"| {row.rule_id} | {row.action} | "
                f"{row.derivation_incidents} ({row.derivation_fraction:.2f}) | "
                f"{row.heldout_incidents} ({row.heldout_fraction:.2f}) | "
                f"{'yes' if row.meets_transfer_floor else 'no'} |"
            )
        return [*block, "", note, ""]

    content = [r for r in report.rules if r.scope_kind == "content_triggered"]
    global_rules = [r for r in report.rules if r.scope_kind == "global"]
    lines += _table(
        "Content-triggered rules (transfer evidence)",
        content,
        "A content-triggered rule meets the transfer floor when it activates on "
        f"at least {_TRANSFER_MIN_INCIDENTS} transfer-check incidents. "
        "Rules below the floor remain local operational rules and do not support "
        "the transfer result.",
    )
    lines += _table(
        "Global rules (always-applicable policy, not coverage)",
        global_rules,
        "Global rules match every incident by construction. Their support is "
        "reported for completeness only and is never counted as coverage "
        "evidence, which would inflate transfer support.",
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="validate_heldout_activation",
        description="Measure content-triggered rule activation on a disjoint "
        "transfer-check export batch.",
    )
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--derivation-dir", default="dataset/incidents")
    parser.add_argument("--heldout-dir", default="dataset/incidents_heldout")
    parser.add_argument(
        "--constraints",
        default="policy/constraints.yaml",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    repo_root = resolve_repo_root(args.repo_root)
    report = build_report(
        repo_root=repo_root,
        derivation_dir=(repo_root / str(args.derivation_dir)).resolve(),
        heldout_dir=(repo_root / str(args.heldout_dir)).resolve(),
        constraints_path=(repo_root / str(args.constraints)).resolve(),
    )
    output_json = (repo_root / str(args.output_json)).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote JSON report: {output_json}")
    if args.output_md:
        output_md = (repo_root / str(args.output_md)).resolve()
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(render_markdown(report), encoding="utf-8")
        print(f"Wrote Markdown report: {output_md}")
    transferring = ", ".join(report.transfer_rule_ids) or "(none)"
    print(f"Rules meeting the transfer-check floor: {transferring}")


if __name__ == "__main__":
    main()
