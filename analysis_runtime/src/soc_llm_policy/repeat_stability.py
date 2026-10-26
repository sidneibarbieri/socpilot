from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

from soc_llm_policy.analyze import (
    AnalysisBundleInput,
    _build_analysis_bundle,
    analyze_incident_outputs_full,
)
from soc_llm_policy.json_stability import write_stable_json
from soc_llm_policy.paths import resolve_repo_root
from soc_llm_policy.result_models import VerifierOutputModel

RunKey = tuple[str, str, str]
RUN_ID_TIMESTAMP_LENGTH = 16


@dataclass(frozen=True)
class RunRecord:
    incident_id: str
    llm_deployment: str
    llm_arm: str
    source_path: Path
    run_tag: str
    timestamp_tag: str
    timestamp: datetime
    source_kind: str
    repair_run_id: str | None = None

    @property
    def key(self) -> RunKey:
        return (self.incident_id, self.llm_deployment, self.llm_arm)


def _parse_base_run_timestamp(run_id: str) -> datetime:
    parts = run_id.split("_")
    for part in parts:
        if (
            len(part) == RUN_ID_TIMESTAMP_LENGTH
            and part.endswith("Z")
            and "T" in part
        ):
            return datetime.strptime(part, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
    raise ValueError(f"Could not parse timestamp from run id: {run_id}")


def _parse_timestamp_tag(value: str) -> datetime:
    return datetime.strptime(value, "%Y%m%dT%H%M%S%fZ").replace(tzinfo=UTC)


def _extract_timestamp_tag(path: Path) -> str:
    name = path.name
    if not name.startswith("verifier_output_") or not name.endswith(".json"):
        raise ValueError(f"Unexpected verifier output filename: {path}")
    stem = name[:-5]
    timestamp_tag = stem.rsplit("_", 1)[-1]
    if "T" not in timestamp_tag or not timestamp_tag.endswith("Z"):
        raise ValueError(f"Could not parse timestamp tag from: {path}")
    return timestamp_tag


def _load_verifier(path: Path) -> VerifierOutputModel:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return VerifierOutputModel.model_validate(raw)


def _collect_base_records(
    *,
    outputs_incidents_dir: Path,
    base_run_started_at: datetime,
) -> dict[RunKey, list[RunRecord]]:
    grouped: dict[RunKey, list[RunRecord]] = defaultdict(list)
    for incident_dir in sorted(
        path for path in outputs_incidents_dir.glob("INC_*") if path.is_dir()
    ):
        for path in sorted(incident_dir.glob("verifier_output_*.json")):
            timestamp_tag = _extract_timestamp_tag(path)
            timestamp = _parse_timestamp_tag(timestamp_tag)
            if timestamp < base_run_started_at:
                continue
            verifier = _load_verifier(path)
            if not verifier.llm_deployment or not verifier.llm_arm:
                continue
            record = RunRecord(
                incident_id=verifier.incident_id,
                llm_deployment=verifier.llm_deployment,
                llm_arm=verifier.llm_arm,
                source_path=path.resolve(),
                run_tag=verifier.run_tag,
                timestamp_tag=timestamp_tag,
                timestamp=timestamp,
                source_kind="base",
            )
            grouped[record.key].append(record)
    for records in grouped.values():
        records.sort(key=lambda item: item.timestamp_tag)
    return grouped


def _load_failures(path: Path) -> tuple[dict[RunKey, list[int]], list[dict[str, Any]]]:
    if not path.exists():
        return {}, []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("failures", [])
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON failure list at {path}")
    grouped: dict[RunKey, list[int]] = defaultdict(list)
    for item in payload:
        if not isinstance(item, dict):
            continue
        incident_id = str(item.get("incident_id", "")).strip()
        deployment = str(item.get("deployment", "")).strip()
        arm = str(item.get("arm", "")).strip()
        repeat_index_raw = item.get("repeat_index")
        if not incident_id or not deployment or not arm:
            continue
        if repeat_index_raw is None:
            continue
        repeat_index = int(repeat_index_raw)
        grouped[(incident_id, deployment, arm)].append(repeat_index)
    for values in grouped.values():
        values.sort()
    return grouped, payload


def _assign_base_records_to_repeats(
    *,
    base_records: dict[RunKey, list[RunRecord]],
    failures_by_key: dict[RunKey, list[int]],
    planned_repeat_count: int,
) -> dict[RunKey, dict[int, RunRecord]]:
    assignments: dict[RunKey, dict[int, RunRecord]] = {}
    for key, records in base_records.items():
        missing = set(failures_by_key.get(key, []))
        available_slots = [
            slot for slot in range(1, planned_repeat_count + 1) if slot not in missing
        ]
        if len(records) != len(available_slots):
            raise ValueError(
                "Base repeat assignment mismatch for "
                f"{key}: records={len(records)} available_slots={len(available_slots)}"
            )
        assignments[key] = {
            slot: record for slot, record in zip(available_slots, records, strict=True)
        }
    return assignments


def _collect_repair_records(
    repair_run_dirs: list[Path],
) -> dict[RunKey, list[RunRecord]]:
    grouped: dict[RunKey, list[RunRecord]] = defaultdict(list)
    for repair_dir in repair_run_dirs:
        run_id = repair_dir.name
        incidents_root = repair_dir / "results" / "incidents"
        if not incidents_root.exists():
            raise FileNotFoundError(f"Repair incidents dir not found: {incidents_root}")
        for incident_dir in sorted(
            path for path in incidents_root.glob("INC_*") if path.is_dir()
        ):
            for path in sorted(incident_dir.glob("verifier_output_*.json")):
                timestamp_tag = _extract_timestamp_tag(path)
                timestamp = _parse_timestamp_tag(timestamp_tag)
                verifier = _load_verifier(path)
                if not verifier.llm_deployment or not verifier.llm_arm:
                    continue
                record = RunRecord(
                    incident_id=verifier.incident_id,
                    llm_deployment=verifier.llm_deployment,
                    llm_arm=verifier.llm_arm,
                    source_path=path.resolve(),
                    run_tag=verifier.run_tag,
                    timestamp_tag=timestamp_tag,
                    timestamp=timestamp,
                    source_kind="repair",
                    repair_run_id=run_id,
                )
                grouped[record.key].append(record)
    for records in grouped.values():
        records.sort(key=lambda item: item.timestamp_tag)
    return grouped


def _merge_repair_records(
    *,
    assignments: dict[RunKey, dict[int, RunRecord]],
    repair_records: dict[RunKey, list[RunRecord]],
    planned_repeat_count: int,
) -> dict[RunKey, dict[int, RunRecord]]:
    merged: dict[RunKey, dict[int, RunRecord]] = {
        key: dict(value) for key, value in assignments.items()
    }
    for key, records in repair_records.items():
        slot_map = merged.setdefault(key, {})
        missing_slots = [
            slot for slot in range(1, planned_repeat_count + 1) if slot not in slot_map
        ]
        if len(records) != len(missing_slots):
            raise ValueError(
                "Repair repeat assignment mismatch for "
                f"{key}: repairs={len(records)} missing_slots={len(missing_slots)}"
            )
        for slot, record in zip(missing_slots, records, strict=True):
            slot_map[slot] = record
    for key, slot_map in merged.items():
        missing_slots = [
            slot for slot in range(1, planned_repeat_count + 1) if slot not in slot_map
        ]
        if missing_slots:
            raise ValueError(
                f"Incomplete repeat coverage for {key}: missing {missing_slots}"
            )
    return merged


def _safe_symlink(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        target.symlink_to(source)
    except OSError:
        shutil.copy2(source, target)


def _stage_repeat_outputs(
    *,
    assignments: dict[RunKey, dict[int, RunRecord]],
    output_dir: Path,
    planned_repeat_count: int,
) -> dict[int, Path]:
    staged: dict[int, Path] = {}
    for repeat_index in range(1, planned_repeat_count + 1):
        repeat_dir = output_dir / "staged" / f"repeat_{repeat_index}"
        if repeat_dir.exists():
            shutil.rmtree(repeat_dir)
        repeat_dir.mkdir(parents=True, exist_ok=True)
        for (incident_id, _deployment, _arm), slot_map in assignments.items():
            record = slot_map[repeat_index]
            target = repeat_dir / incident_id / record.source_path.name
            _safe_symlink(record.source_path, target)
        staged[repeat_index] = repeat_dir
    return staged


def _metric_series(values: list[float | None]) -> dict[str, Any]:
    rounded = [round(value, 6) if value is not None else None for value in values]
    available = [value for value in values if value is not None]
    return {
        "values": rounded,
        "available_count": len(available),
        "mean": round(mean(available), 6) if available else None,
        "min": round(min(available), 6) if available else None,
        "max": round(max(available), 6) if available else None,
        "range": round(max(available) - min(available), 6) if available else None,
    }


def _build_stability_summary(
    *,
    repeat_outputs: list[dict[str, Any]],
    repaired_slot_count: int,
    original_failures: list[dict[str, Any]],
) -> dict[str, Any]:
    summaries = [item["summary"] for item in repeat_outputs]
    by_rule_series: dict[str, list[float]] = defaultdict(list)
    for summary in summaries:
        rules = summary.get("violations_by_rule", {})
        if isinstance(rules, dict):
            for rule_id, value in rules.items():
                by_rule_series[str(rule_id)].append(float(value))
    by_model_series: dict[str, dict[str, list[float | None]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for repeat in repeat_outputs:
        for row in repeat.get("by_model", []):
            model_label = str(row.get("model_label", "unknown"))
            by_model_series[model_label]["run_violation_rate"].append(
                float(row.get("run_violation_rate", 0.0))
            )
            by_model_series[model_label]["enforcement_modification_rate"].append(
                float(row.get("enforcement_modification_rate", 0.0))
            )
            by_model_series[model_label]["task_coverage_drop_rate"].append(
                row.get("task_coverage_drop_rate")
            )

    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "repeat_count": len(repeat_outputs),
        "repeat_indices": [item["repeat_index"] for item in repeat_outputs],
        "completed_repeat_row_count": sum(
            int(summary.get("run_count", 0) or 0) for summary in summaries
        ),
        "pre_repair_failure_count": len(original_failures),
        "repaired_slot_count": repaired_slot_count,
        "final_unresolved_failure_count": 0,
        "repair_note": (
            "Repeat summaries are computed after targeted repair records fill "
            "any transient provider/API failures from the base repeated run."
        ),
        "incident_violation_rate": _metric_series(
            [float(summary["incident_violation_rate"]) for summary in summaries]
        ),
        "enforcement_modification_rate": _metric_series(
            [float(summary["enforcement_modification_rate"]) for summary in summaries]
        ),
        "task_coverage_drop_rate": _metric_series(
            [summary.get("task_coverage_drop_rate") for summary in summaries]
        ),
        "llm_total_tokens_total": _metric_series(
            [float(summary["llm_total_tokens_total"]) for summary in summaries]
        ),
        "llm_cost_estimated_usd_total": _metric_series(
            [float(summary["llm_cost_estimated_usd_total"]) for summary in summaries]
        ),
        "violations_by_rule": {
            rule_id: _metric_series(values)
            for rule_id, values in sorted(by_rule_series.items())
        },
        "by_model": {
            model_label: {
                metric_name: _metric_series(values)
                for metric_name, values in sorted(metric_map.items())
            }
            for model_label, metric_map in sorted(by_model_series.items())
        },
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    write_stable_json(path, payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soc_llm_policy.repeat_stability",
        description=(
            "Reconstruct repeat-aware summaries from a campaign run and its "
            "explicit recovery segments."
        ),
    )
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--base-run-id", required=True)
    parser.add_argument(
        "--repair-run-dirs",
        required=True,
        help="CSV list of repair run directories under .local/repair_runs.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/analysis/repeat_stability",
        help="Directory to write staged repeat outputs and repeat-aware summaries.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = resolve_repo_root(args.repo_root)
    repo_root = Path(repo_root)
    base_run_id = str(args.base_run_id).strip()
    repair_run_dirs = [
        (repo_root / item.strip()).resolve()
        if not Path(item.strip()).is_absolute()
        else Path(item.strip()).resolve()
        for item in str(args.repair_run_dirs).split(",")
        if item.strip()
    ]
    output_dir = (
        (repo_root / str(args.output_dir)).resolve()
        if not Path(str(args.output_dir)).is_absolute()
        else Path(str(args.output_dir)).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    base_run_dir = repo_root / "results" / "experiments" / base_run_id
    base_bundle_path = (base_run_dir / "analysis_bundle.json").resolve()
    if not base_bundle_path.exists():
        raise FileNotFoundError(f"Base analysis bundle not found: {base_bundle_path}")
    base_bundle = json.loads(base_bundle_path.read_text(encoding="utf-8"))
    coverage = base_bundle.get("experiment", {}).get("coverage", {})
    planned_repeat_count = int(coverage.get("planned_repeat_count", 0) or 0)
    if planned_repeat_count <= 0:
        raise ValueError(f"Invalid planned repeat count in {base_bundle_path}")

    base_run_started_at = _parse_base_run_timestamp(base_run_id)
    outputs_incidents_dir = (repo_root / "results" / "incidents").resolve()
    base_records = _collect_base_records(
        outputs_incidents_dir=outputs_incidents_dir,
        base_run_started_at=base_run_started_at,
    )
    failures_by_key, original_failures = _load_failures(
        base_run_dir / "experiment_failures.json"
    )
    assignments = _assign_base_records_to_repeats(
        base_records=base_records,
        failures_by_key=failures_by_key,
        planned_repeat_count=planned_repeat_count,
    )
    repairs = _collect_repair_records(repair_run_dirs)
    completed = _merge_repair_records(
        assignments=assignments,
        repair_records=repairs,
        planned_repeat_count=planned_repeat_count,
    )

    staged_repeat_dirs = _stage_repeat_outputs(
        assignments=completed,
        output_dir=output_dir,
        planned_repeat_count=planned_repeat_count,
    )

    repeat_outputs: list[dict[str, Any]] = []
    repeat_assignment_rows: list[dict[str, Any]] = []
    for repeat_index in range(1, planned_repeat_count + 1):
        repeat_dir = staged_repeat_dirs[repeat_index]
        repeat_summary_path = output_dir / f"repeat_{repeat_index}_summary.json"
        repeat_bundle_path = output_dir / f"repeat_{repeat_index}_analysis_bundle.json"
        (
            summary,
            rows,
            by_model_rows,
            by_rule_rows,
            by_rule_treatment_rows,
            pairwise_rows,
        ) = analyze_incident_outputs_full(repeat_dir)
        _write_json(repeat_summary_path, summary)
        bundle = _build_analysis_bundle(
            AnalysisBundleInput(
                eval_protocol_version=str(
                    base_bundle.get("eval_protocol_version", "official")
                ),
                repo_root=repo_root,
                outputs_incidents_dir=repeat_dir,
                incident_filter=None,
                summary=summary,
                rows=rows,
                by_model_rows=by_model_rows,
                by_rule_rows=by_rule_rows,
                by_rule_treatment_rows=by_rule_treatment_rows,
                pairwise_rows=pairwise_rows,
            )
        )
        _write_json(repeat_bundle_path, bundle)
        repeat_outputs.append(
            {
                "repeat_index": repeat_index,
                "summary": summary,
                "by_model": by_model_rows,
                "summary_path": str(repeat_summary_path),
                "analysis_bundle_path": str(repeat_bundle_path),
            }
        )
        for key, slot_map in sorted(completed.items()):
            record = slot_map[repeat_index]
            repeat_assignment_rows.append(
                {
                    "repeat_index": repeat_index,
                    "incident_id": key[0],
                    "llm_deployment": key[1],
                    "llm_arm": key[2],
                    "run_tag": record.run_tag,
                    "timestamp_tag": record.timestamp_tag,
                    "source_kind": record.source_kind,
                    "source_path": str(record.source_path),
                    "repair_run_id": record.repair_run_id,
                }
            )

    repeat_stability_summary = _build_stability_summary(
        repeat_outputs=repeat_outputs,
        repaired_slot_count=len(original_failures),
        original_failures=original_failures,
    )
    _write_json(
        output_dir / "repeat_assignments.json", {"assignments": repeat_assignment_rows}
    )
    _write_json(output_dir / "repeat_stability_summary.json", repeat_stability_summary)
    _write_json(
        output_dir / "repeat_stability_manifest.json",
        {
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "base_run_id": base_run_id,
            "base_run_dir": str(base_run_dir.resolve()),
            "base_run_started_at_utc": base_run_started_at.isoformat(),
            "planned_repeat_count": planned_repeat_count,
            "repair_run_dirs": [str(path) for path in repair_run_dirs],
            "pre_repair_failure_count": len(original_failures),
            "repaired_slot_count": len(original_failures),
            "final_unresolved_failure_count": 0,
            "note": (
                "The base repeated run may contain transient provider/API "
                "failures; repeat_outputs and repeat_assignments describe the "
                "completed post-repair evidence surface used for reporting."
            ),
            "repeat_outputs": repeat_outputs,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
