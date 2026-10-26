"""Reference-dependent measurements; missing references are not zero scores."""

from typing import Any

from soc_llm_policy.result_models import IncidentMetrics

REFERENCE_FIELDS = (
    "precision_raw", "recall_raw", "task_coverage_raw", "f1_raw", "jaccard_raw",
    "precision_enforced", "recall_enforced", "task_coverage_enforced",
    "f1_enforced", "jaccard_enforced", "delta_jaccard", "task_coverage_delta",
)


def reference_values(metrics: IncidentMetrics) -> dict[str, float | None]:
    if not metrics.human_action_count:
        return dict.fromkeys(REFERENCE_FIELDS)
    values = {name: getattr(metrics, name) for name in REFERENCE_FIELDS
              if not name.startswith("task_coverage")}
    values["task_coverage_raw"] = metrics.recall_raw
    values["task_coverage_enforced"] = metrics.recall_enforced
    values["task_coverage_delta"] = (
        round(metrics.recall_enforced - metrics.recall_raw, 4)
        if metrics.recall_enforced is not None and metrics.recall_raw is not None
        else None
    )
    return values


def summarize_reference_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {}
    for field in REFERENCE_FIELDS:
        values = [float(row[field]) for row in rows if row.get(field) is not None]
        summary[f"{field}_avg"] = round(sum(values) / len(values), 4) if values else None
    deltas = [float(row["task_coverage_delta"]) for row in rows
              if row.get("task_coverage_delta") is not None]
    summary["reference_run_count"] = len(deltas)
    summary["task_coverage_drop_run_count"] = sum(value < 0 for value in deltas) if deltas else None
    summary["task_coverage_drop_rate"] = (
        round(sum(value < 0 for value in deltas) / len(deltas), 4) if deltas else None
    )
    return summary
