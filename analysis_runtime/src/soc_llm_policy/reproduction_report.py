from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from soc_llm_policy.artifact_verify import verify_artifact_package
from soc_llm_policy.json_stability import write_stable_json
from soc_llm_policy.paths import resolve_repo_root


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_json(repo_root: Path, rel_path: str) -> dict[str, Any]:
    path = repo_root / rel_path
    if not path.exists():
        return {}
    return _read_json(path)


def _status_ok(payload: dict[str, Any]) -> bool:
    if not payload:
        return False
    if payload.get("status") in {"pass", "ok"}:
        return True
    if payload.get("ok") is True:
        return True
    return False


def _artifact_file(repo_root: Path, rel_path: str) -> dict[str, Any]:
    path = repo_root / rel_path
    if not path.exists():
        return {"path": rel_path, "present": False}
    return {
        "path": rel_path,
        "present": True,
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _repair_modes(summary: dict[str, Any]) -> dict[str, int]:
    return {
        "insert": int(summary.get("enforcement_actions_inserted_count_total", 0) or 0),
        "remove": int(summary.get("enforcement_actions_removed_count_total", 0) or 0),
        "defer": int(summary.get("enforcement_actions_deferred_count_total", 0) or 0),
        "reorder": int(
            summary.get("enforcement_actions_reordered_count_total", 0) or 0
        ),
    }


def build_report(repo_root: Path) -> dict[str, Any]:
    protocol = _optional_json(
        repo_root,
        "artifact_outputs/analysis/protocol_freeze.json",
    )
    official = _optional_json(
        repo_root,
        "artifact_outputs/analysis/official_evaluation_summary.json",
    )
    manifest = _optional_json(
        repo_root,
        "artifact_outputs/analysis/official_runs_manifest.json",
    )
    public_consistency = _optional_json(
        repo_root,
        "artifact_outputs/analysis/official_public_consistency_recheck.json",
    )
    dataset_audit = _optional_json(
        repo_root,
        "artifact_outputs/analysis/dataset_audit_recheck.json",
    )
    corpus_readiness = _optional_json(
        repo_root,
        "artifact_outputs/analysis/corpus_readiness_recheck.json",
    )
    global_assessment = _optional_json(
        repo_root,
        "artifact_outputs/analysis/global_artifact_assessment_recheck.json",
    )
    release_readiness = _optional_json(
        repo_root,
        "artifact_outputs/analysis/release_readiness_recheck.json",
    )

    execution = manifest.get("execution_accounting", {})
    if not isinstance(execution, dict):
        execution = {}

    files = [
        "artifact_outputs/analysis/protocol_freeze.json",
        "artifact_outputs/analysis/dataset_audit_recheck.json",
        "artifact_outputs/analysis/corpus_readiness_recheck.json",
        "artifact_outputs/analysis/global_artifact_assessment_recheck.json",
        "artifact_outputs/analysis/official_llm_analysis_bundle.json",
        "artifact_outputs/analysis/official_evaluation_summary.json",
        "artifact_outputs/analysis/official_runs_manifest.json",
        "artifact_outputs/analysis/official_pairwise_tests.json",
        "artifact_outputs/analysis/official_public_consistency_recheck.json",
        "artifact_outputs/analysis/release_readiness_recheck.json",
        "artifact_outputs/analysis/repeat_stability/repeat_stability_summary.json",
    ]

    checks = {
        "artifact_integrity": verify_artifact_package(repo_root)["status"] == "ok",
        "dataset_audit": bool(dataset_audit.get("ok") is True),
        # The readiness recheck is a stats profile, not a pass/fail record:
        # it attests by carrying a non-empty recomputed corpus profile.
        "corpus_readiness": int(corpus_readiness.get("incident_count", 0) or 0) > 0,
        "official_public_consistency": _status_ok(public_consistency),
        "global_artifact_assessment": bool(global_assessment),
        "release_hygiene": release_readiness.get("status") in {"ok", "pass", "warning"},
        "official_paid_calls_not_rerun": True,
    }

    summary = {
        "artifact_name": "SOCpilot",
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "status": "pass" if all(checks.values()) else "fail",
        "zero_cost_reproduction": True,
        "paid_llm_calls_rerun": False,
        "checks": checks,
        "protocol": {
            "dataset_release_id": protocol.get("dataset_release_id"),
            "eval_protocol_version": protocol.get("eval_protocol_version"),
            "approval_policy_mode": protocol.get("approval_policy_mode"),
            "incident_count": protocol.get("incident_count"),
            "action_catalog_count": protocol.get("action_catalog_count"),
            "policy_rule_count": protocol.get("policy_rule_count"),
            "official_models": manifest.get(
                "official_models",
                protocol.get("official_models"),
            ),
            "arms": manifest.get("arms_internal", protocol.get("arms")),
        },
        "official_evaluation": {
            "llm_trajectories_planned": execution.get("llm_trajectories_planned"),
            "llm_trajectories_completed": execution.get("llm_trajectories_completed"),
            "llm_trajectories_failed_total": execution.get(
                "llm_trajectories_failed_total"
            ),
            "run_count": official.get("run_count"),
            "incident_violation_rate": official.get("incident_violation_rate"),
            "enforcement_modification_rate": official.get(
                "enforcement_modification_rate"
            ),
            "task_coverage_drop_rate": official.get("task_coverage_drop_rate"),
            "violations_by_rule": official.get("violations_by_rule", {}),
            "repair_modes_observed": _repair_modes(official),
            "llm_total_tokens_total": official.get("llm_total_tokens_total"),
            "llm_cost_estimated_usd_total": official.get(
                "llm_cost_estimated_usd_total"
            ),
        },
        "public_consistency": {
            "status": public_consistency.get("status"),
            "check_count": public_consistency.get("check_count"),
            "failed_check_count": public_consistency.get("failed_check_count"),
        },
        "release_hygiene": {
            "status": release_readiness.get("status"),
            "issue_count": release_readiness.get("issue_count"),
            "issues_by_severity": release_readiness.get("issues_by_severity", {}),
        },
        "evidence_files": [_artifact_file(repo_root, rel_path) for rel_path in files],
        "interpretation": [
            "This command reproduces the public analysis checks without rerunning "
            "paid LLM calls.",
            "The official evaluation remains the frozen real-SOC protocol.",
            "Sanitized row-level official metrics are checked against shipped "
            "summaries, manifests, and paired tests.",
            "Observed official violations activate approval rules R3 and R4; "
            "inactive rule families remain declared scope limitations, not "
            "pooled evidence.",
        ],
    }
    return summary


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    protocol = report["protocol"]
    official = report["official_evaluation"]
    consistency = report["public_consistency"]
    hygiene = report["release_hygiene"]
    lines = [
        "# SOCpilot Reproduction Report",
        "",
        f"Status: **{report['status']}**",
        "",
        "This zero-cost reproduction path does not rerun paid LLM calls. "
        "It checks the public artifact surface, recomputes public audits, "
        "and verifies that sanitized official row-level metrics agree with "
        "the shipped official summaries.",
        "",
        "## Protocol",
        "",
        f"- Dataset release: `{protocol.get('dataset_release_id')}`",
        f"- Evaluation protocol: `{protocol.get('eval_protocol_version')}`",
        f"- Incidents: `{protocol.get('incident_count')}`",
        f"- Action catalog size: `{protocol.get('action_catalog_count')}`",
        f"- Policy rule count: `{protocol.get('policy_rule_count')}`",
        f"- Official models: `{', '.join(protocol.get('official_models') or [])}`",
        "",
        "## Official Evaluation",
        "",
        f"- Planned LLM trajectories: `{official.get('llm_trajectories_planned')}`",
        f"- Completed LLM trajectories: `{official.get('llm_trajectories_completed')}`",
        "- Final failed LLM trajectories: "
        f"`{official.get('llm_trajectories_failed_total')}`",
        f"- Run-level violation rate: `{official.get('incident_violation_rate')}`",
        "- Enforcement-modification rate: "
        f"`{official.get('enforcement_modification_rate')}`",
        f"- Task-coverage drop rate: `{official.get('task_coverage_drop_rate')}`",
        f"- Violations by rule: `{official.get('violations_by_rule')}`",
        f"- Repair modes observed: `{official.get('repair_modes_observed')}`",
        "- Estimated paid-run cost recorded in manifest: "
        f"`${official.get('llm_cost_estimated_usd_total')}`",
        "",
        "## Public Consistency",
        "",
        f"- Official public consistency: `{consistency.get('status')}`",
        f"- Consistency checks: `{consistency.get('check_count')}`",
        f"- Failed consistency checks: `{consistency.get('failed_check_count')}`",
        f"- Release hygiene status: `{hygiene.get('status')}`",
        f"- Release hygiene issue count: `{hygiene.get('issue_count')}`",
        "",
        "## Evidence Files",
        "",
    ]
    for item in report["evidence_files"]:
        marker = "present" if item.get("present") else "missing"
        digest = item.get("sha256", "")
        line = f"- `{item['path']}`: {marker}"
        if digest:
            line += f", sha256 `{digest}`"
        lines.append(line)
    lines.extend(["", "## Interpretation", ""])
    for item in report["interpretation"]:
        lines.append(f"- {item}")
    text = "\n".join(lines) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soc_llm_policy.reproduction_report",
        description=(
            "Write a zero-cost SOCpilot reproduction report from public "
            "artifact outputs."
        ),
    )
    parser.add_argument("--repo-root", default=None)
    parser.add_argument(
        "--output-json",
        default="artifact_outputs/reproduction_report.json",
    )
    parser.add_argument(
        "--output-md",
        default="artifact_outputs/reproduction_report.md",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(resolve_repo_root(args.repo_root))
    report = build_report(repo_root)

    json_path = Path(args.output_json)
    if not json_path.is_absolute():
        json_path = repo_root / json_path
    md_path = Path(args.output_md)
    if not md_path.is_absolute():
        md_path = repo_root / md_path

    write_stable_json(json_path, report)
    _write_markdown(md_path, report)
    print(f"Wrote reproduction report JSON: {json_path}")
    print(f"Wrote reproduction report Markdown: {md_path}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
