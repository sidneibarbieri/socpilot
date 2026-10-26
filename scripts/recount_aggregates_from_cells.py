"""Recount the reported aggregates from the per-cell verdicts that produced them.

Reported rates are ratios of integers, and those integers live in the analysis
bundle. A reviewer who trusts the bundle is trusting a summary. This tool goes one
level down: it reads the per-cell verdict record the run wrote as it executed, one
line per evaluated cell, and recounts each rate from first principles.

A campaign may span more than one ledger when an interrupted worker execution
is resumed. Every named record is deduplicated by content-addressed cell ID.
Canonical campaigns are additionally keyed by acquisition batch, so
heterogeneity estimates are never pooled silently.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from soc_llm_policy.paths import resolve_repo_root


@dataclass(frozen=True)
class CellTally:
    """How many cells a model and arm produced, and how many violated policy."""

    cell_count: int
    violating_cell_count: int


def _cell_violates(cell: dict[str, Any]) -> bool:
    return any(
        outcome["violated"]
        for family in cell["family_outcomes"]
        for outcome in family["rule_outcomes"]
    )


type TallyKey = tuple[str | None, str]


def tally_cells(cell_records: list[Path]) -> dict[TallyKey, CellTally]:
    """Count each model and arm once per distinct cell across every record."""
    cells = Counter()
    violations = Counter()
    seen: set[str] = set()
    for record in cell_records:
        with record.open(encoding="utf-8") as handle:
            for line in handle:
                cell = json.loads(line)
                if cell["cell_id"] in seen:
                    continue
                seen.add(cell["cell_id"])
                key = (
                    cell.get("corpus_stratum"),
                    f"{cell['deployment']}|{cell['arm']}",
                )
                cells[key] += 1
                if _cell_violates(cell):
                    violations[key] += 1
    return {
        key: CellTally(cells[key], violations[key])
        for key in sorted(cells, key=lambda item: (item[0] or "", item[1]))
    }


def compare_with_bundle(
    tallies: dict[TallyKey, CellTally], bundle: dict[str, Any]
) -> list[str]:
    """Return one message per aggregate the per-cell record does not support."""
    problems = []
    reported_keys: set[TallyKey] = set()
    for row in bundle["by_model"]:
        label = row["model_label"]
        key = (row.get("corpus_stratum"), label)
        reported_keys.add(key)
        tally = tallies.get(key)
        prefix = f"{key[0] or 'unstratified'} / {label}"
        if tally is None:
            problems.append(
                f"{prefix}: the bundle reports cells the record has none of"
            )
            continue
        if tally.cell_count != row["run_count"]:
            problems.append(
                f"{prefix}: bundle counts {row['run_count']} cells, "
                f"the record holds {tally.cell_count}"
            )
        if tally.violating_cell_count != row["runs_with_violations"]:
            problems.append(
                f"{prefix}: bundle counts {row['runs_with_violations']} violating "
                f"cells, the record holds {tally.violating_cell_count}"
            )
    unreported = sorted(
        set(tallies) - reported_keys,
        key=lambda item: (item[0] or "", item[1]),
    )
    problems.extend(
        f"{stratum or 'unstratified'} / {label}: "
        "the record holds cells the bundle never reports"
        for stratum, label in unreported
    )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--bundle-json", required=True)
    parser.add_argument(
        "--cells-jsonl",
        nargs="+",
        required=True,
        help="Per-cell verdict records from every worker execution segment.",
    )
    args = parser.parse_args(argv)
    repo_root = resolve_repo_root(args.repo_root)

    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else repo_root / path

    tallies = tally_cells([resolve(value) for value in args.cells_jsonl])
    bundle = json.loads(resolve(args.bundle_json).read_text(encoding="utf-8"))
    problems = compare_with_bundle(tallies, bundle)

    total = sum(tally.cell_count for tally in tallies.values())
    for problem in problems:
        print(problem)
    print(f"Recounted {total} cells from the per-cell verdict record.")
    if problems:
        print(f"{len(problems)} aggregate(s) the record does not support.")
        return 1
    print("Every reported aggregate recounts from the cells that produced it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
