"""Test declared paired contrasts under alternative repeat reductions.

Each incident is evaluated three times per model and prompt condition. The
primary analysis uses the reduction named in the analysis bundle. This tool
repeats the same predefined one-factor contrasts under four reductions and
records whether their direction and significance classification change. It
reads preserved evidence, makes no provider calls, and is deterministic.

  first     the earliest repeat of the group
  last      the latest repeat
  majority  the verdict at least two of three repeats agree on
  any       the incident counts as violating if any repeat violated
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RepeatObservation:
    repeat_index: int | None
    run_tag: str
    violated: bool


RepeatGroup = list[RepeatObservation]
RepeatGroupKey = tuple[str | None, str, str]
Reduction = Callable[[RepeatGroup], bool]
_ALL_INCIDENTS_STRATUM = "all_incidents"


@dataclass(frozen=True)
class PairwiseResult:
    """One planned comparison under one reduction of the repeat group."""

    label_a: str
    label_b: str
    corpus_stratum: str | None
    contrast_type: str
    incident_count: int
    rate_difference: float
    cohens_h: float
    p_value_holm: float


def _first(group: RepeatGroup) -> bool:
    return _ordered_group(group)[0].violated


def _last(group: RepeatGroup) -> bool:
    return _ordered_group(group)[-1].violated


def _majority(group: RepeatGroup) -> bool:
    if len(group) % 2 == 0:
        raise ValueError("Majority reduction requires an odd repeat count.")
    return sum(item.violated for item in group) * 2 > len(group)


def _any(group: RepeatGroup) -> bool:
    return any(item.violated for item in group)


def _ordered_group(group: RepeatGroup) -> list[RepeatObservation]:
    return sorted(
        group,
        key=lambda item: (
            item.repeat_index is None,
            item.repeat_index if item.repeat_index is not None else 0,
            item.run_tag,
        ),
    )


REDUCTIONS: dict[str, Reduction] = {
    "first": _first,
    "last": _last,
    "majority": _majority,
    "any": _any,
}
_SIGNIFICANCE_LEVEL = 0.05


def _mcnemar_p_value(discordant_a: int, discordant_b: int) -> float:
    """Two-sided McNemar with the continuity correction the pipeline applies."""
    total = discordant_a + discordant_b
    if total == 0:
        return 1.0
    chi_square = (abs(discordant_a - discordant_b) - 1) ** 2 / total
    return math.erfc(math.sqrt(chi_square / 2))


def _cohens_h(rate_a: float, rate_b: float) -> float:
    return 2 * math.asin(math.sqrt(rate_a)) - 2 * math.asin(math.sqrt(rate_b))


def _holm_adjust(p_values: list[float]) -> list[float]:
    ranked = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(ranked):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def _analysis_rows(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    rows = list(bundle["incidents"])
    if not any(row.get("corpus_stratum") is not None for row in rows):
        return rows
    aggregate_rows = [
        {**row, "corpus_stratum": _ALL_INCIDENTS_STRATUM}
        for row in rows
        if row.get("corpus_stratum") != _ALL_INCIDENTS_STRATUM
    ]
    return [*rows, *aggregate_rows]


def _group_rows(rows: list[dict[str, Any]]) -> dict[RepeatGroupKey, RepeatGroup]:
    groups: dict[RepeatGroupKey, RepeatGroup] = defaultdict(list)
    for row in rows:
        incident = str(row.get("incident_dir_id") or row["incident_id"])
        raw_stratum = row.get("corpus_stratum")
        stratum = str(raw_stratum) if raw_stratum is not None else None
        key = (stratum, str(row["model_label"]), incident)
        repeat_index = row.get("experiment_repeat_index")
        if isinstance(repeat_index, bool) or not isinstance(repeat_index, int):
            repeat_index = None
        groups[key].append(
            RepeatObservation(
                repeat_index=repeat_index,
                run_tag=str(row["run_tag"]),
                violated=int(row["violation_count"]) > 0,
            )
        )
    return groups


def group_repeats(bundle: dict[str, Any]) -> dict[RepeatGroupKey, RepeatGroup]:
    """Collect each incident-model-condition group in every analysis stratum."""
    return _group_rows(_analysis_rows(bundle))


def _expected_repeats(bundle: dict[str, Any]) -> int | None:
    summary = bundle.get("summary")
    if isinstance(summary, dict):
        value = summary.get("expected_repeats")
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    experiment = bundle.get("experiment")
    if isinstance(experiment, dict):
        value = experiment.get("repeats")
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def _complete_groups(
    groups: dict[RepeatGroupKey, RepeatGroup], expected_repeats: int | None
) -> tuple[dict[RepeatGroupKey, RepeatGroup], int]:
    if expected_repeats is None:
        return groups, 0
    expected_indices = set(range(1, expected_repeats + 1))
    complete: dict[RepeatGroupKey, RepeatGroup] = {}
    incomplete_count = 0
    for key, group in groups.items():
        indices = [item.repeat_index for item in group]
        indexed = all(index is not None for index in indices)
        if indexed:
            if len(set(indices)) != len(indices):
                raise ValueError(f"Repeat group {key} has duplicate repeat indices.")
            unexpected = set(indices) - expected_indices
            if unexpected:
                raise ValueError(
                    f"Repeat group {key} exceeds the declared schedule: {unexpected}"
                )
            is_complete = set(indices) == expected_indices
        else:
            # Compatibility for older evidence that predates repeat-index fields.
            is_complete = len(group) == expected_repeats
        if is_complete:
            complete[key] = group
        else:
            incomplete_count += 1
    return complete, incomplete_count


def _label_factors(
    bundle: dict[str, Any],
) -> dict[tuple[str | None, str], tuple[str, str] | None]:
    factors: dict[tuple[str | None, str], tuple[str, str] | None] = {}
    for row in _analysis_rows(bundle):
        label = str(row["model_label"])
        raw_stratum = row.get("corpus_stratum")
        stratum = str(raw_stratum) if raw_stratum is not None else None
        deployment = str(row.get("llm_deployment") or "")
        condition = str(row.get("llm_arm") or "")
        value = (deployment, condition) if deployment and condition else None
        factor_key = (stratum, label)
        previous = factors.setdefault(factor_key, value)
        if previous != value:
            raise ValueError(f"Model label {label!r} maps to conflicting factors.")
    return factors


def pairwise_under(
    groups: dict[RepeatGroupKey, RepeatGroup],
    factors: dict[tuple[str | None, str], tuple[str, str] | None],
    reduce_group: Reduction,
) -> list[PairwiseResult]:
    """Evaluate the planned one-factor contrasts, paired within incident."""
    verdicts: dict[str | None, dict[str, dict[str, bool]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for (stratum, label, incident), group in groups.items():
        verdicts[stratum][label][incident] = reduce_group(group)

    raw: list[tuple[str | None, str, str, str, list[str]]] = []
    for stratum in sorted(verdicts, key=lambda value: value or ""):
        labels = sorted(verdicts[stratum])
        for index_a in range(len(labels)):
            for index_b in range(index_a + 1, len(labels)):
                label_a, label_b = labels[index_a], labels[index_b]
                factors_a = factors.get((stratum, label_a))
                factors_b = factors.get((stratum, label_b))
                if factors_a is not None and factors_b is not None:
                    same_model = factors_a[0] == factors_b[0]
                    same_condition = factors_a[1] == factors_b[1]
                    if same_model == same_condition:
                        continue
                    contrast_type = "prompt_condition" if same_model else "model"
                else:
                    contrast_type = "generic"
                shared = sorted(
                    set(verdicts[stratum][label_a])
                    & set(verdicts[stratum][label_b])
                )
                if shared:
                    raw.append(
                        (stratum, label_a, label_b, contrast_type, shared)
                    )

    p_values = []
    partials = []
    for stratum, label_a, label_b, contrast_type, shared in raw:
        only_a = sum(
            verdicts[stratum][label_a][incident]
            and not verdicts[stratum][label_b][incident]
            for incident in shared
        )
        only_b = sum(
            verdicts[stratum][label_b][incident]
            and not verdicts[stratum][label_a][incident]
            for incident in shared
        )
        rate_a = sum(
            verdicts[stratum][label_a][incident] for incident in shared
        ) / len(shared)
        rate_b = sum(
            verdicts[stratum][label_b][incident] for incident in shared
        ) / len(shared)
        p_values.append(_mcnemar_p_value(only_a, only_b))
        partials.append(
            (
                label_a,
                label_b,
                stratum,
                contrast_type,
                len(shared),
                rate_a - rate_b,
                _cohens_h(rate_a, rate_b),
            )
        )

    adjusted = _holm_adjust(p_values)
    return [
        PairwiseResult(
            label_a,
            label_b,
            stratum,
            contrast_type,
            count,
            difference,
            effect,
            adjusted[index],
        )
        for index, (
            label_a,
            label_b,
            stratum,
            contrast_type,
            count,
            difference,
            effect,
        ) in enumerate(partials)
    ]


def _comparison_key(result: PairwiseResult) -> tuple[str | None, str, str]:
    return result.corpus_stratum, result.label_a, result.label_b


def _direction(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def build_report(bundle: dict[str, Any]) -> dict[str, Any]:
    observed_groups = group_repeats(bundle)
    expected_repeats = _expected_repeats(bundle)
    groups, incomplete_group_count = _complete_groups(
        observed_groups, expected_repeats
    )
    source_observed_groups = _group_rows(list(bundle["incidents"]))
    source_groups, source_incomplete_group_count = _complete_groups(
        source_observed_groups, expected_repeats
    )
    factors = _label_factors(bundle)
    summary = bundle.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("Analysis bundle has no summary.")
    reported_reduction = summary.get("repeat_reduction")
    if reported_reduction not in REDUCTIONS:
        raise ValueError("Analysis bundle has no supported repeat reduction.")

    by_reduction: dict[str, list[dict[str, Any]]] = {}
    for name, reduce_group in REDUCTIONS.items():
        by_reduction[name] = [
            {
                "label_a": result.label_a,
                "label_b": result.label_b,
                "corpus_stratum": result.corpus_stratum,
                "contrast_type": result.contrast_type,
                "incident_count": result.incident_count,
                "rate_difference": result.rate_difference,
                "cohens_h": result.cohens_h,
                "p_value_holm": result.p_value_holm,
            }
            for result in pairwise_under(groups, factors, reduce_group)
        ]

    comparison_keys = {
        name: {
            (entry["corpus_stratum"], entry["label_a"], entry["label_b"])
            for entry in entries
        }
        for name, entries in by_reduction.items()
    }
    baseline_keys = comparison_keys[reported_reduction]
    if any(keys != baseline_keys for keys in comparison_keys.values()):
        raise ValueError("Repeat reductions do not cover one comparison family.")

    baseline = {
        (entry["corpus_stratum"], entry["label_a"], entry["label_b"]): entry
        for entry in by_reduction[reported_reduction]
    }
    agreement_checks = 0
    direction_agreements = 0
    significance_agreements = 0
    for name, entries in by_reduction.items():
        if name == reported_reduction:
            continue
        for entry in entries:
            key = (
                entry["corpus_stratum"],
                entry["label_a"],
                entry["label_b"],
            )
            reference = baseline[key]
            agreement_checks += 1
            direction_agreements += int(
                _direction(entry["rate_difference"])
                == _direction(reference["rate_difference"])
            )
            significance_agreements += int(
                (entry["p_value_holm"] < _SIGNIFICANCE_LEVEL)
                == (reference["p_value_holm"] < _SIGNIFICANCE_LEVEL)
            )

    return {
        "schema_version": 2,
        "repeat_group_count": len(groups),
        "observed_repeat_group_count": len(observed_groups),
        "incomplete_repeat_group_count": incomplete_group_count,
        "source_repeat_group_count": len(source_groups),
        "source_observed_repeat_group_count": len(source_observed_groups),
        "source_incomplete_repeat_group_count": source_incomplete_group_count,
        "expected_repeats": expected_repeats,
        "reported_reduction": reported_reduction,
        "comparison_count": len(baseline_keys),
        "comparison_family": "single_factor",
        "corpus_strata": sorted(
            {key[0] for key in baseline_keys}, key=lambda value: value or ""
        ),
        "by_reduction": by_reduction,
        "agreement_check_count": agreement_checks,
        "direction_agreement_count": direction_agreements,
        "significance_agreement_count": significance_agreements,
        "same_comparison_family_under_every_reduction": True,
    }


def render_macros(report: dict[str, Any]) -> str:
    """Render compact accounting macros from the sensitivity report."""
    values = {
        "ReductionCount": len(report["by_reduction"]),
        "ReductionIncidentGroups": report["repeat_group_count"],
        "ReductionContrastCount": report["comparison_count"],
        "ReductionAgreementChecks": report["agreement_check_count"],
        "ReductionDirectionAgreements": report["direction_agreement_count"],
        "ReductionSignificanceAgreements": report[
            "significance_agreement_count"
        ],
    }
    header = "% Generated by run_repeat_reduction_sensitivity.py\n"
    body = "\n".join(
        f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in values.items()
    )
    return header + body + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--macros-output")
    args = parser.parse_args(argv)

    bundle = json.loads(Path(args.bundle_json).read_text(encoding="utf-8"))
    report = build_report(bundle)

    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote reduction sensitivity: {output}")
    for name, entries in sorted(report["by_reduction"].items()):
        marker = " (reported)" if name == report["reported_reduction"] else ""
        print(f"  {name:9s} comparisons={len(entries)}{marker}")
    if args.macros_output:
        macros = Path(args.macros_output)
        macros.parent.mkdir(parents=True, exist_ok=True)
        macros.write_text(render_macros(report), encoding="utf-8")
        print(f"Wrote macros: {macros}")


if __name__ == "__main__":
    main()
