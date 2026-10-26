from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from soc_llm_policy.dataset_audit import audit_dataset
from soc_llm_policy.global_provenance import build_global_provenance
from soc_llm_policy.json_stability import write_stable_json
from soc_llm_policy.paths import RepoPaths, repo_relative_path, resolve_repo_root
from soc_llm_policy.pipeline import list_inbox_incidents

MIN_DUAL_MODEL_COUNT = 2
MIN_DUAL_ARM_COUNT = 2


def _parse_csv_list(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return list(dict.fromkeys(item.strip() for item in raw.split(",") if item.strip()))


def _resolve_incidents(
    paths: RepoPaths, incidents_arg: str | None, include_all: bool
) -> list[str]:
    if include_all:
        return list_inbox_incidents(paths)
    incidents = _parse_csv_list(incidents_arg)
    if not incidents:
        raise ValueError("Provide --incidents INC_001,INC_002 or use --all.")
    return incidents


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return raw


def _active_eval_protocol(paths: RepoPaths) -> str | None:
    freeze = _load_json_object(paths.outputs_analysis_dir / "protocol_freeze.json")
    value = str(freeze.get("eval_protocol_version", "")).strip()
    return value or None


def _active_model_names(paths: RepoPaths) -> list[str]:
    freeze = _load_json_object(paths.outputs_analysis_dir / "protocol_freeze.json")
    values = freeze.get("official_models", [])
    if not isinstance(values, list):
        return []
    return sorted(str(value).strip() for value in values if str(value).strip())


def _active_repeat_count(paths: RepoPaths) -> int | None:
    freeze = _load_json_object(paths.outputs_analysis_dir / "protocol_freeze.json")
    value = freeze.get("repeats")
    if isinstance(value, int) and not isinstance(value, bool) and value >= 1:
        return value
    return None


def _score_official_bundle(
    bundle: dict[str, Any],
    *,
    bundle_path: Path,
    active_protocol: str | None,
    expected_models: list[str] | None,
    expected_repeat_count: int | None,
    require_dual_model: bool,
) -> tuple[int, int, int, int, int, int, int, int, float] | None:
    summary = bundle.get("summary", {})
    experiment = bundle.get("experiment", {})
    if not isinstance(summary, dict) or not isinstance(experiment, dict):
        return None
    coverage = experiment.get("coverage", {})
    mode_counts = summary.get("mode_counts", {})
    if not isinstance(coverage, dict) or not isinstance(mode_counts, dict):
        return None
    incident_count = int(summary.get("incident_count", 0) or 0)
    llm_run_count = int(mode_counts.get("LLM", 0) or 0)
    if incident_count <= 0 or llm_run_count <= 0:
        return None

    selected_models = experiment.get("selected_models", [])
    selected_names = sorted(
        str(item.get("name", "")).strip()
        for item in selected_models
        if isinstance(item, dict)
    )
    model_count = int(coverage.get("selected_model_count", len(selected_names)) or 0)
    arm_count = int(coverage.get("planned_arm_count", 0) or 0)
    repeat_count = int(coverage.get("planned_repeat_count", 1) or 1)
    successful = int(coverage.get("successful_run_count", llm_run_count) or 0)
    failures = int(coverage.get("execution_failure_count", 0) or 0)
    protocol = str(experiment.get("eval_protocol_version", "")).strip()
    if require_dual_model and (
        model_count < MIN_DUAL_MODEL_COUNT or arm_count < MIN_DUAL_ARM_COUNT
    ):
        return None
    return (
        int(bool(active_protocol) and protocol == active_protocol),
        int(bool(expected_models) and selected_names == expected_models),
        int(successful == llm_run_count),
        int(expected_repeat_count is not None and repeat_count == expected_repeat_count),
        incident_count,
        successful,
        model_count,
        -failures,
        bundle_path.stat().st_mtime,
    )


def select_best_official_bundle(
    bundle_paths: list[Path],
    *,
    active_protocol: str | None,
    expected_models: list[str] | None = None,
    expected_repeat_count: int | None = None,
    require_dual_model: bool = True,
) -> Path | None:
    scored: list[
        tuple[tuple[int, int, int, int, int, int, int, int, float], Path]
    ] = []
    for path in bundle_paths:
        bundle = _load_json_object(path)
        if not bundle:
            continue
        score = _score_official_bundle(
            bundle,
            bundle_path=path,
            active_protocol=active_protocol,
            expected_models=expected_models,
            expected_repeat_count=expected_repeat_count,
            require_dual_model=require_dual_model,
        )
        if score is not None:
            scored.append((score, path))
    if not scored:
        return None
    return max(scored, key=lambda item: item[0])[1].resolve()


def _resolve_official_summary_source(
    paths: RepoPaths, explicit_path: str | None
) -> Path | None:
    if explicit_path:
        candidate = Path(explicit_path).expanduser().resolve()
        return candidate if candidate.exists() else None
    bundle = select_best_official_bundle(
        sorted(paths.outputs_experiments_dir.glob("*/analysis_bundle.json")),
        active_protocol=_active_eval_protocol(paths),
        expected_models=_active_model_names(paths),
        expected_repeat_count=_active_repeat_count(paths),
    )
    if bundle is not None:
        summary = bundle.parent / "summary.json"
        if summary.exists():
            return summary.resolve()
    copied = paths.outputs_analysis_dir / "official_evaluation_summary.json"
    return copied.resolve() if copied.exists() else None


def _copy_official_summary(source: Path | None, output: Path | None) -> None:
    if source is None or output is None:
        return
    write_stable_json(output, _load_json_object(source))


def _official_scope(
    summary: dict[str, Any], operational_rule_ids: list[str]
) -> dict[str, Any]:
    if not summary:
        return {"available": False}
    violations = summary.get("violations_by_rule", {})
    if not isinstance(violations, dict):
        raise ValueError("Official summary violations_by_rule must be an object.")
    unexpected = sorted(set(violations) - set(operational_rule_ids))
    return {
        "available": True,
        "run_count": int(summary.get("run_count", 0) or 0),
        "incident_count": int(summary.get("incident_count", 0) or 0),
        "violations_by_rule": violations,
        "unexpected_rule_ids": unexpected,
        "llm_total_tokens_total": int(summary.get("llm_total_tokens_total", 0) or 0),
        "llm_cost_estimated_usd_total": float(
            summary.get("llm_cost_estimated_usd_total", 0.0) or 0.0
        ),
        "run_success_rate": float(
            (summary.get("experiment_coverage", {}) or {}).get(
                "run_success_rate", 0.0
            )
            or 0.0
        ),
    }


def build_global_artifact_assessment(
    *,
    paths: RepoPaths,
    incidents: list[str],
    official_summary_path: Path | None,
) -> dict[str, Any]:
    dataset = audit_dataset(paths=paths, incidents=incidents)
    provenance = build_global_provenance(paths=paths, incident_ids=incidents)
    summary = (
        _load_json_object(official_summary_path) if official_summary_path else {}
    )
    rule_items = provenance["policy_rules"]["items"]
    operational_rule_ids = sorted(
        item["rule_id"]
        for item in rule_items
        if item["evidence_tier"] == "operational"
    )
    specification_rule_ids = sorted(
        item["rule_id"]
        for item in rule_items
        if item["evidence_tier"] == "specification"
    )
    readiness = dataset.get("corpus_readiness", {})
    conversion = readiness.get("conversion_quality", {})
    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "dataset_release_id": provenance["dataset_release_id"],
        "eval_protocol_version": provenance["eval_protocol_version"],
        "incident_count": len(incidents),
        "official_summary_source": (
            repo_relative_path(official_summary_path, paths.repo_root)
            if official_summary_path is not None and official_summary_path.exists()
            else None
        ),
        "dataset": {
            "ok": bool(dataset.get("ok")),
            "privacy_issue_count": int(dataset.get("privacy_issue_count", 0) or 0),
            "invalid_incident_count": int(
                dataset.get("invalid_incident_count", 0) or 0
            ),
            "event_count_total": int(readiness.get("event_count_total", 0) or 0),
            "observed_action_count": int(readiness.get("action_count_total", 0) or 0),
            "workflow_task_count": int(conversion.get("task_count_total", 0) or 0),
            "empty_action_profile_count": int(
                conversion.get("empty_action_profile_count", 0) or 0
            ),
        },
        "policy_surface": {
            "action_catalog_count": len(provenance["action_catalog"]["items"]),
            "operational_rule_ids": operational_rule_ids,
            "specification_rule_ids": specification_rule_ids,
        },
        "official_evaluation": _official_scope(summary, operational_rule_ids),
    }
    if report["official_evaluation"].get("unexpected_rule_ids"):
        report["dataset"]["ok"] = False
    return report


def _render_markdown(report: dict[str, Any]) -> str:
    dataset = report["dataset"]
    policy = report["policy_surface"]
    evaluation = report["official_evaluation"]
    lines = [
        "# Global Artifact Assessment",
        "",
        f"- Incidents: `{report['incident_count']}`",
        f"- Dataset gate: `{'PASS' if dataset['ok'] else 'FAIL'}`",
        f"- Privacy issues: `{dataset['privacy_issue_count']}`",
        f"- Invalid incidents: `{dataset['invalid_incident_count']}`",
        f"- Telemetry events: `{dataset['event_count_total']}`",
        f"- Observed response actions: `{dataset['observed_action_count']}`",
        f"- Workflow tasks: `{dataset['workflow_task_count']}`",
        "",
        "## Evidence boundary",
        "",
        "- Workflow tasks are not treated as response actions or approvals.",
        "- The corpus does not support a paired human-action baseline.",
        "- Specification-only rules are excluded from prompts, verdicts, and rates.",
        "",
        "## Policy surface",
        "",
        f"- Action catalog entries: `{policy['action_catalog_count']}`",
        f"- Operational rules: `{len(policy['operational_rule_ids'])}`",
        f"- Specification-only rules: `{len(policy['specification_rule_ids'])}`",
        "",
        "## Official evaluation",
        "",
    ]
    if evaluation.get("available"):
        lines.extend(
            [
                f"- Runs: `{evaluation['run_count']}`",
                f"- Incidents: `{evaluation['incident_count']}`",
                f"- Unexpected rule ids: `{evaluation['unexpected_rule_ids']}`",
                f"- Estimated cost: `USD {evaluation['llm_cost_estimated_usd_total']:.6f}`",
            ]
        )
    else:
        lines.append("- No compatible official evaluation is present.")
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="soc_llm_policy.global_artifact_assessment")
    parser.add_argument("--repo-root", default=None)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--incidents", default=None)
    group.add_argument("--all", action="store_true")
    parser.add_argument(
        "--output-json", default="results/analysis/global_artifact_assessment.json"
    )
    parser.add_argument(
        "--output-md", default="results/analysis/global_artifact_assessment.md"
    )
    parser.add_argument("--official-summary-json", default=None)
    parser.add_argument(
        "--official-summary-output-json",
        default="results/analysis/official_evaluation_summary.json",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    repo_root = resolve_repo_root(args.repo_root)
    paths = RepoPaths(repo_root=repo_root)
    incidents = _resolve_incidents(paths, args.incidents, bool(args.all))
    summary_path = _resolve_official_summary_source(paths, args.official_summary_json)
    summary_output = Path(args.official_summary_output_json).expanduser().resolve()
    _copy_official_summary(summary_path, summary_output)
    report = build_global_artifact_assessment(
        paths=paths,
        incidents=incidents,
        official_summary_path=summary_path,
    )
    output_json = Path(args.output_json).expanduser().resolve()
    output_md = Path(args.output_md).expanduser().resolve()
    write_stable_json(output_json, report)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(_render_markdown(report), encoding="utf-8")


if __name__ == "__main__":
    main()
