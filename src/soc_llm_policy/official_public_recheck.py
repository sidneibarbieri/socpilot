from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from soc_llm_policy.json_stability import write_stable_json
from soc_llm_policy.paths import resolve_repo_root
from soc_llm_policy.reference_metrics import summarize_reference_metrics


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "detail": self.detail}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _close(left: float, right: float, *, tolerance: float = 1e-4) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def _same_measurement(left: Any, right: Any, *, tolerance: float = 1e-4) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return _close(left, right, tolerance=tolerance)


def _as_int(value: Any, default: int = -1) -> int:
    if value is None:
        return default
    return int(value)


def _rows_by_model(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        deployment = str(row.get("llm_deployment", ""))
        arm = str(row.get("llm_arm", ""))
        if deployment and arm:
            grouped[(deployment, arm)].append(row)
    return grouped


def _model_metrics(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    run_count = len(rows)
    if run_count == 0:
        return {
            "completed_runs": 0,
            "run_violation_rate": 0.0,
            "enforcement_modification_rate": 0.0,
            "task_coverage_enforced_avg": None,
            "delta_jaccard_avg": None,
            "llm_total_tokens_total": 0,
            "llm_cost_estimated_usd_total": 0.0,
        }
    violation_count = sum(
        1 for row in rows if int(row.get("violation_count", 0) or 0) > 0
    )
    modified_count = sum(
        1 for row in rows if bool(row.get("enforcement_modified", False))
    )
    references = summarize_reference_metrics(rows)
    return {
        "completed_runs": run_count,
        "run_violation_rate": round(violation_count / run_count, 4),
        "enforcement_modification_rate": round(modified_count / run_count, 4),
        "task_coverage_enforced_avg": references["task_coverage_enforced_avg"],
        "delta_jaccard_avg": references["delta_jaccard_avg"],
        "llm_total_tokens_total": sum(
            int(row.get("llm_total_tokens", 0) or 0) for row in rows
        ),
        "llm_cost_estimated_usd_total": round(
            sum(float(row.get("llm_cost_estimated_usd", 0.0) or 0.0) for row in rows),
            6,
        ),
    }


def _pairwise_signature(
    rows: list[dict[str, Any]],
) -> list[tuple[int, float, float, float]]:
    return sorted(
        (
            int(row.get("paired_incident_count", 0) or 0),
            round(float(row.get("rate_diff_a_minus_b", 0.0) or 0.0), 6),
            round(float(row.get("mcnemar_p_value_two_sided", 0.0) or 0.0), 6),
            round(float(row.get("cohens_h", 0.0) or 0.0), 6),
        )
        for row in rows
    )


def build_report(
    *,
    official_bundle: dict[str, Any],
    official_summary: dict[str, Any],
    runs_manifest: dict[str, Any],
    pairwise_tests: dict[str, Any],
) -> dict[str, Any]:
    checks: list[Check] = []
    incidents = official_bundle.get("incidents", [])
    if not isinstance(incidents, list):
        incidents = []
    summary = official_bundle.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}
    bundle_pairwise = official_bundle.get("pairwise", [])
    if not isinstance(bundle_pairwise, list):
        bundle_pairwise = []
    public_pairwise = pairwise_tests.get("rows", [])
    if not isinstance(public_pairwise, list):
        public_pairwise = []

    row_count = len(incidents)
    checks.append(
        Check(
            "row_count_matches_bundle",
            row_count == _as_int(official_bundle.get("row_count"), -1),
            f"rows={row_count}, declared={official_bundle.get('row_count')}",
        )
    )
    checks.append(
        Check(
            "row_count_matches_summary",
            row_count == _as_int(summary.get("run_count"), -1),
            f"rows={row_count}, summary.run_count={summary.get('run_count')}",
        )
    )

    manifest_accounting = runs_manifest.get("execution_accounting", {})
    if not isinstance(manifest_accounting, dict):
        manifest_accounting = {}
    checks.append(
        Check(
            "manifest_completed_count_matches_rows",
            row_count
            == _as_int(
                manifest_accounting.get("llm_trajectories_completed"),
                -1,
            ),
            (
                f"rows={row_count}, "
                f"manifest.completed={manifest_accounting.get('llm_trajectories_completed')}"
            ),
        )
    )
    checks.append(
        Check(
            "manifest_final_failures_zero",
            _as_int(manifest_accounting.get("llm_trajectories_failed_total"), -1) == 0
            and _as_int(manifest_accounting.get("execution_failures"), -1) == 0,
            (
                "failed_total="
                f"{manifest_accounting.get('llm_trajectories_failed_total')}, "
                f"execution_failures={manifest_accounting.get('execution_failures')}"
            ),
        )
    )

    for field in [
        "run_count",
        "incident_count",
        "incident_violation_rate",
        "enforcement_modification_rate",
        "task_coverage_drop_rate",
        "llm_total_tokens_total",
        "llm_cost_estimated_usd_total",
    ]:
        left = summary.get(field)
        right = official_summary.get(field)
        ok = _same_measurement(left, right)
        checks.append(
            Check(
                f"summary_field_{field}",
                ok,
                f"bundle={left}, official={right}",
            )
        )

    grouped = _rows_by_model([row for row in incidents if isinstance(row, dict)])
    for manifest_row in runs_manifest.get("by_model_and_arm", []):
        if not isinstance(manifest_row, dict):
            continue
        key = (
            str(manifest_row.get("model", "")),
            str(manifest_row.get("arm_internal", "")),
        )
        metrics = _model_metrics(grouped.get(key, []))
        for field, computed in metrics.items():
            expected = manifest_row.get(field)
            ok = _same_measurement(computed, expected, tolerance=1e-3)
            checks.append(
                Check(
                    f"manifest_model_{key[0]}_{key[1]}_{field}",
                    ok,
                    f"computed={computed}, manifest={expected}",
                )
            )

    checks.append(
        Check(
            "pairwise_row_count_matches",
            len(bundle_pairwise)
            == _as_int(pairwise_tests.get("row_count"), -1)
            == len(public_pairwise),
            (
                f"bundle={len(bundle_pairwise)}, public_rows={len(public_pairwise)}, "
                f"declared={pairwise_tests.get('row_count')}"
            ),
        )
    )
    checks.append(
        Check(
            "pairwise_signatures_match",
            _pairwise_signature(bundle_pairwise)
            == _pairwise_signature(public_pairwise),
            "compared paired count, rate difference, McNemar p-value, and Cohen h",
        )
    )

    ok = all(check.ok for check in checks)
    return {
        "status": "pass" if ok else "fail",
        "check_count": len(checks),
        "failed_check_count": sum(1 for check in checks if not check.ok),
        "checks": [check.to_dict() for check in checks],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soc_llm_policy.official_public_recheck",
        description=(
            "Verify that public official row-level metrics match shipped "
            "official summaries."
        ),
    )
    parser.add_argument("--repo-root", default=None)
    parser.add_argument(
        "--official-bundle-json",
        default="artifact_outputs/analysis/official_llm_analysis_bundle.json",
    )
    parser.add_argument(
        "--official-summary-json",
        default="artifact_outputs/analysis/official_evaluation_summary.json",
    )
    parser.add_argument(
        "--official-runs-manifest-json",
        default="artifact_outputs/analysis/official_runs_manifest.json",
    )
    parser.add_argument(
        "--official-pairwise-json",
        default="artifact_outputs/analysis/official_pairwise_tests.json",
    )
    parser.add_argument(
        "--output-json",
        default="artifact_outputs/analysis/official_public_consistency.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(resolve_repo_root(args.repo_root))

    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else repo_root / path

    report = build_report(
        official_bundle=_read_json(resolve(str(args.official_bundle_json))),
        official_summary=_read_json(resolve(str(args.official_summary_json))),
        runs_manifest=_read_json(resolve(str(args.official_runs_manifest_json))),
        pairwise_tests=_read_json(resolve(str(args.official_pairwise_json))),
    )
    output_path = resolve(str(args.output_json))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_stable_json(output_path, report)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
