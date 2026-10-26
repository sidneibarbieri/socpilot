"""Score projection-fidelity extractions against the authored ground truth.

Deterministic, no provider calls. For each model's extractions it computes
precision, recall, exact-match rate over the proposed action set, and a separate
distractor-rejection rate (the fraction of catalog actions that the prose names
but explicitly does not propose which the projection correctly excludes). The
distractor rate is the load-bearing signal: it measures whether the projection
recovers proposed actions rather than merely mentioned ones.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def _blank_cell() -> dict:
    return {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "exact": 0,
        "scored": 0,
        "distractor_total": 0,
        "distractor_rejected": 0,
        "refused": 0,
        "parse_failed": 0,
        "content_filtered": 0,
        "output_unavailable": 0,
        "out_of_catalog": 0,
    }


def _score_model(probe: dict, extractions: dict) -> dict:
    targets = {item["plan_id"]: set(item["target_actions"]) for item in probe["items"]}
    distractors = {
        item["plan_id"]: set(item.get("distractor_actions", []))
        for item in probe["items"]
    }
    phenom = {
        item["plan_id"]: item.get("phenomenon", "untagged") for item in probe["items"]
    }
    tp = fp = fn = exact = 0
    distractor_total = distractor_rejected = 0
    refused = 0
    parse_failed = 0
    content_filtered = 0
    output_unavailable = 0
    # Per-phenomenon accumulation so precision/recall can be reported for the
    # specific linguistic stressors (negation, paraphrase, distractor, ...) a
    # reviewer asks about, not only in aggregate.
    by_phenomenon: dict[str, dict] = {}
    for record in extractions["records"]:
        plan_id = record["plan_id"]
        cell = by_phenomenon.setdefault(phenom[plan_id], _blank_cell())
        if record.get("output_unavailable"):
            output_unavailable += 1
            cell["output_unavailable"] += 1
            continue
        if record.get("content_filtered"):
            # Blocked by a provider content filter (deployment policy), not a
            # projection outcome: excluded from scoring and reported separately.
            content_filtered += 1
            cell["content_filtered"] += 1
            continue
        if record.get("refused"):
            refused += 1
            cell["refused"] += 1
            continue
        if record.get("parse_failed"):
            parse_failed += 1
            cell["parse_failed"] += 1
            continue
        extracted = set(record["extracted_actions"])
        out_of_catalog = set(record.get("out_of_catalog", []))
        target = targets[plan_id]
        c_tp = len(extracted & target)
        c_fp = len(extracted - target) + len(out_of_catalog)
        c_fn = len(target - extracted)
        tp += c_tp
        fp += c_fp
        fn += c_fn
        cell["tp"] += c_tp
        cell["fp"] += c_fp
        cell["fn"] += c_fn
        cell["out_of_catalog"] += len(out_of_catalog)
        cell["scored"] += 1
        if extracted == target and not out_of_catalog:
            exact += 1
            cell["exact"] += 1
        for distractor in distractors[plan_id]:
            distractor_total += 1
            cell["distractor_total"] += 1
            if distractor not in extracted:
                distractor_rejected += 1
                cell["distractor_rejected"] += 1
    scored = (
        len(extractions["records"])
        - refused
        - parse_failed
        - content_filtered
        - output_unavailable
    )
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    return {
        "model": extractions["model"],
        "scored_plans": scored,
        "refused_plans": refused,
        "parse_failed_plans": parse_failed,
        "content_filtered_plans": content_filtered,
        "output_unavailable_plans": output_unavailable,
        "exact_match_bounds_denominator": scored + output_unavailable,
        "exact_match_bounds_with_unavailable": (
            [
                exact / (scored + output_unavailable),
                (exact + output_unavailable) / (scored + output_unavailable),
            ]
            if scored + output_unavailable
            else None
        ),
        "out_of_catalog_count": sum(
            len(set(record.get("out_of_catalog", [])))
            for record in extractions["records"]
            if not record.get("refused")
            and not record.get("parse_failed")
            and not record.get("content_filtered")
            and not record.get("output_unavailable")
        ),
        "exact_match_count": exact,
        "precision": precision,
        "recall": recall,
        "distractor_total": distractor_total,
        "distractor_rejected": distractor_rejected,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "by_phenomenon": by_phenomenon,
    }


def _aggregate_phenomena(per_model: list[dict]) -> dict[str, dict]:
    merged: dict[str, dict] = {}
    for model in per_model:
        for phenomenon, cell in model["by_phenomenon"].items():
            acc = merged.setdefault(phenomenon, _blank_cell())
            for key, value in cell.items():
                acc[key] += value
    out: dict[str, dict] = {}
    for phenomenon, cell in merged.items():
        tp, fp, fn = cell["tp"], cell["fp"], cell["fn"]
        out[phenomenon] = {
            **cell,
            "precision": tp / (tp + fp) if (tp + fp) else None,
            "recall": tp / (tp + fn) if (tp + fn) else None,
        }
    return dict(sorted(out.items()))


def _validate_extractions(probe: dict, extractions: dict, path: Path) -> None:
    for record in extractions["records"]:
        if record.get("output_unavailable"):
            if not record.get("unavailable_reason") or any(
                record.get(key)
                for key in (
                    "extracted_actions",
                    "out_of_catalog",
                    "refused",
                    "parse_failed",
                    "content_filtered",
                )
            ):
                raise ValueError(
                    "Unavailable output must have a reason and no scored outcome"
                )
    expected_ids = [str(item["plan_id"]) for item in probe["items"]]
    observed_ids = [str(record.get("plan_id")) for record in extractions["records"]]
    if len(observed_ids) != len(set(observed_ids)):
        raise ValueError(f"Extraction file contains duplicate plan ids: {path}")
    if set(observed_ids) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(observed_ids))
        unexpected = sorted(set(observed_ids) - set(expected_ids))
        raise ValueError(
            f"Extraction coverage differs from the probe ({path}): "
            f"missing={missing}, unexpected={unexpected}"
        )


def render_phenomenon_table(by_phenomenon: dict[str, dict]) -> str:
    """A generated tabular fragment: precision/recall/exact/distractor per
    phenomenon, so reports can render the measured values directly."""

    def pct(value: float | None) -> str:
        return f"{value * 100:.0f}" if value is not None else "--"

    rows = []
    for phenomenon, cell in by_phenomenon.items():
        label = phenomenon.replace("_", " ")
        distractor = (
            f"{cell['distractor_rejected']}/{cell['distractor_total']}"
            if cell["distractor_total"]
            else "--"
        )
        rows.append(
            f"{label} & {cell['scored']} & {pct(cell['precision'])} & "
            f"{pct(cell['recall'])} & {cell['exact']} & {distractor} \\\\"
        )
    header = (
        "% Generated by build_projection_fidelity_report.py\n"
        "\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}lrrrrr@{}}\n\\toprule\n"
        "\\textbf{Phenomenon} & \\textbf{Ext.} & \\textbf{Prec.\\%} & \\textbf{Rec.\\%} & \\textbf{Exact} & \\textbf{Distr. rej.} \\\\\n"
        "\\midrule\n"
    )
    return header + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular*}\n"


def render_macros(report: dict[str, object]) -> str:
    """Render fidelity macros derived only from the report."""

    def percent(value: float | None) -> str:
        return f"{value * 100:.0f}" if value is not None else "0"

    values = {
        "FidelityPlanCount": report["plan_count"],
        "FidelityModelCount": report["model_count"],
        "FidelityPhenomenonCount": report["phenomenon_count"],
        "FidelityPrecisionPct": percent(report["overall_precision"]),
        "FidelityRecallPct": percent(report["overall_recall"]),
        "FidelityExactMatch": report["overall_exact_match_count"],
        "FidelityScoredPlans": report["overall_scored_plans"],
        "FidelityUnavailableOutputs": report.get("overall_output_unavailable_plans", 0),
        "FidelityDistractorTotal": report["distractor_total"],
        "FidelityDistractorRejected": report["distractor_rejected"],
    }
    header = "% Generated by build_projection_fidelity_report.py\n"
    body = "\n".join(
        f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in values.items()
    )
    return header + body + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-yaml", type=Path, required=True)
    parser.add_argument(
        "--extractions-json",
        type=Path,
        nargs="+",
        required=True,
        help="One extraction file per model.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--macros-output", type=Path)
    parser.add_argument(
        "--phenomenon-table",
        type=Path,
        help="Optional generated LaTeX tabular with the per-phenomenon breakdown.",
    )
    args = parser.parse_args(argv)

    probe = yaml.safe_load(args.probe_yaml.read_text(encoding="utf-8"))
    per_model = []
    observed_models: set[str] = set()
    for path in args.extractions_json:
        extractions = json.loads(path.read_text(encoding="utf-8"))
        _validate_extractions(probe, extractions, path)
        model = str(extractions["model"])
        if model in observed_models:
            raise ValueError(f"Duplicate extraction model: {model}")
        observed_models.add(model)
        per_model.append(_score_model(probe, extractions))

    plan_count = len(probe["items"])
    tp = sum(m["true_positive"] for m in per_model)
    fp = sum(m["false_positive"] for m in per_model)
    fn = sum(m["false_negative"] for m in per_model)
    by_phenomenon = _aggregate_phenomena(per_model)
    report = {
        "probe_id": probe["probe_id"],
        "plan_count": plan_count,
        "model_count": len(per_model),
        "phenomenon_count": len(by_phenomenon),
        "per_model": per_model,
        "by_phenomenon": by_phenomenon,
        "overall_precision": tp / (tp + fp) if (tp + fp) else None,
        "overall_recall": tp / (tp + fn) if (tp + fn) else None,
        "overall_exact_match_count": sum(m["exact_match_count"] for m in per_model),
        "overall_scored_plans": sum(m["scored_plans"] for m in per_model),
        "overall_output_unavailable_plans": sum(
            m["output_unavailable_plans"] for m in per_model
        ),
        "overall_refused_plans": sum(m["refused_plans"] for m in per_model),
        "overall_parse_failed_plans": sum(m["parse_failed_plans"] for m in per_model),
        "overall_content_filtered_plans": sum(
            m["content_filtered_plans"] for m in per_model
        ),
        "out_of_catalog_count": sum(m["out_of_catalog_count"] for m in per_model),
        "distractor_total": sum(m["distractor_total"] for m in per_model),
        "distractor_rejected": sum(m["distractor_rejected"] for m in per_model),
    }
    if args.phenomenon_table:
        args.phenomenon_table.parent.mkdir(parents=True, exist_ok=True)
        args.phenomenon_table.write_text(
            render_phenomenon_table(by_phenomenon), encoding="utf-8"
        )
        print(f"Wrote per-phenomenon table: {args.phenomenon_table}")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote projection-fidelity report: {args.output_json}")
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "overall_precision",
                    "overall_recall",
                    "overall_exact_match_count",
                    "overall_scored_plans",
                    "distractor_total",
                    "distractor_rejected",
                )
            },
            indent=2,
        )
    )

    if args.macros_output:
        args.macros_output.parent.mkdir(parents=True, exist_ok=True)
        args.macros_output.write_text(render_macros(report), encoding="utf-8")
        print(f"Wrote fidelity macros: {args.macros_output}")


if __name__ == "__main__":
    main()
