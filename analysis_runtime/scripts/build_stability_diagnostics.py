"""Compute repeat-stability diagnostics from a completed run.

Groups plan-producing cells by incident, model, and prompt condition. For each
complete repeat group, it compares verdict stability with plan variability.
Every number comes from preserved per-cell outputs.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _hard_violation(cell: dict) -> bool:
    return any(
        isinstance(v, dict) and v.get("severity") == "hard"
        for v in cell.get("violations", [])
    )


def _complete_group(cells: list[dict], expected_repeats: int) -> bool:
    indices = [cell.get("experiment_repeat_index") for cell in cells]
    indexed = all(
        isinstance(index, int) and not isinstance(index, bool) for index in indices
    )
    if indexed:
        if len(set(indices)) != len(indices):
            raise ValueError("A stability group contains duplicate repeat indices.")
        expected_indices = set(range(1, expected_repeats + 1))
        unexpected = set(indices) - expected_indices
        if unexpected:
            raise ValueError(
                f"A stability group exceeds the declared schedule: {unexpected}"
            )
        return set(indices) == expected_indices
    return len(cells) == expected_repeats


def build_stability(
    outputs_dir: Path, run_ids: set[str], expected_repeats: int | None = None
) -> dict:
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for path in sorted(outputs_dir.rglob("verifier_output_llm_*.json")):
        cell = json.loads(path.read_text(encoding="utf-8"))
        if cell.get("experiment_run_id") not in run_ids:
            continue
        key = (
            str(cell.get("incident_id")),
            str(cell.get("llm_deployment")),
            str(cell.get("llm_arm")),
        )
        groups[key].append(cell)

    if expected_repeats is None:
        repeat_groups = {key: cells for key, cells in groups.items() if len(cells) > 1}
        incomplete_group_count = 0
    else:
        if expected_repeats < 1:
            raise ValueError("Expected repeat count must be positive.")
        repeat_groups = {
            key: cells
            for key, cells in groups.items()
            if _complete_group(cells, expected_repeats)
        }
        incomplete_group_count = len(groups) - len(repeat_groups)
    total = len(repeat_groups)
    verdict_stable = 0
    sequence_varies = 0
    composition_varies = 0
    for cells in repeat_groups.values():
        verdicts = {_hard_violation(cell) for cell in cells}
        sequences = {tuple(cell.get("llm_actions", [])) for cell in cells}
        compositions = {frozenset(cell.get("llm_actions", [])) for cell in cells}
        if len(verdicts) == 1:
            verdict_stable += 1
        if len(sequences) > 1:
            sequence_varies += 1
        if len(compositions) > 1:
            composition_varies += 1
    return {
        "schema_version": 2,
        "run_ids": sorted(run_ids),
        "source_cell_count": sum(len(cells) for cells in groups.values()),
        "repeat_group_count": total,
        "observed_repeat_group_count": len(groups),
        "incomplete_repeat_group_count": incomplete_group_count,
        "expected_repeats": expected_repeats,
        "verdict_stable_count": verdict_stable,
        "verdict_stable_rate": verdict_stable / total if total else None,
        "plan_sequence_varies_count": sequence_varies,
        "plan_sequence_varies_rate": sequence_varies / total if total else None,
        "plan_composition_varies_count": composition_varies,
        "plan_composition_varies_rate": composition_varies / total if total else None,
    }


def render_macros(stability: dict) -> str:
    def pct(value: float | None) -> str:
        return f"{value * 100:.1f}" if value is not None else "0"

    values = {
        "StabilityRepeatGroups": stability["repeat_group_count"],
        "StabilityVerdictStablePct": pct(stability["verdict_stable_rate"]),
        "StabilityPlanSequenceVariesPct": pct(stability["plan_sequence_varies_rate"]),
        "StabilityPlanCompositionVariesPct": pct(
            stability["plan_composition_varies_rate"]
        ),
    }
    return "\n".join(
        f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in values.items()
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-incidents-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True, help="Comma-separated run ids.")
    parser.add_argument("--expected-repeats", type=int)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--macros-output", type=Path)
    args = parser.parse_args(argv)

    run_ids = {rid.strip() for rid in str(args.run_id).split(",") if rid.strip()}
    stability = build_stability(
        args.outputs_incidents_dir.resolve(), run_ids, args.expected_repeats
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(stability, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote stability diagnostics: {args.output_json}")
    print(json.dumps(stability, indent=2))
    if args.macros_output:
        header = "% Generated by build_stability_diagnostics.py\n"
        args.macros_output.parent.mkdir(parents=True, exist_ok=True)
        args.macros_output.write_text(
            header + render_macros(stability) + "\n", encoding="utf-8"
        )
        print(f"Wrote stability macros: {args.macros_output}")


if __name__ == "__main__":
    main()
