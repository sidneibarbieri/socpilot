#!/usr/bin/env python3
"""Build a cell-traceable paired adversarial report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from soc_llm_policy.adversarial_analysis import (
    AdversarialCell,
    analyze_adversarial_pairs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-matrix-json", type=Path, required=True)
    parser.add_argument("--adversarial-manifest-json", type=Path, required=True)
    parser.add_argument("--outputs-incidents-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def hard_violation(payload: dict[str, Any]) -> bool:
    return any(
        violation.get("severity") == "hard"
        for violation in payload.get("violations", [])
        if isinstance(violation, dict)
    )


def scenario_index(
    manifest_path: Path,
) -> dict[str, tuple[str, str, str]]:
    manifest = read_json(manifest_path)
    index: dict[str, tuple[str, str, str]] = {}
    for pair in manifest.get("pairs", []):
        base_incident_id = str(pair["base_incident_id"])
        family = str(pair["injection_family"])
        clean_id = str(pair["clean_scenario_id"])
        injected_id = str(pair["injected_scenario_id"])
        for scenario_id in (clean_id, injected_id):
            if scenario_id in index:
                raise ValueError(
                    f"Duplicate scenario id in adversarial manifest: {scenario_id}"
                )
        index[clean_id] = (base_incident_id, "clean", family)
        index[injected_id] = (base_incident_id, "injected", family)
    if not index:
        raise ValueError("Adversarial manifest contains no scenario pairs")
    return index


def load_cells(
    run_matrix_path: Path,
    manifest_path: Path,
    outputs_dir: Path,
) -> list[AdversarialCell]:
    matrix = read_json(run_matrix_path)
    scenarios = scenario_index(manifest_path)
    matrix_cells: dict[str, Any] = {}
    for cell in matrix.get("cells", []):
        cell_id = str(cell["cell_id"])
        if cell_id in matrix_cells:
            raise ValueError(f"Duplicate cell_id in adversarial run matrix: {cell_id}")
        matrix_cells[cell_id] = cell
    # results/incidents is shared across runs, so scope strictly to this run's
    # matrix: an output is kept only if its cell_id belongs to the adversarial
    # run matrix. Cells in the matrix with no output are unobserved (model
    # refusals or failures); they are left out here and surface downstream as
    # incomplete pairs, never silently treated as compliant.
    outputs: dict[str, dict[str, Any]] = {}
    for path in sorted(outputs_dir.rglob("verifier_output_llm_*.json")):
        payload = read_json(path)
        cell_id = payload.get("experiment_cell_id")
        if not (isinstance(cell_id, str) and cell_id in matrix_cells):
            continue
        if cell_id in outputs:
            raise ValueError(f"Duplicate verifier output for cell {cell_id}")
        outputs[cell_id] = payload

    cells: list[AdversarialCell] = []
    for cell_id, matrix_cell in matrix_cells.items():
        output = outputs.get(cell_id)
        if output is None:
            continue
        incident_id = str(matrix_cell["incident_id"])
        if incident_id not in scenarios:
            raise ValueError(
                f"Run-matrix incident {incident_id!r} is absent from manifest"
            )
        base_incident_id, stratum, injection_family = scenarios[incident_id]
        cells.append(
            AdversarialCell(
                cell_id=cell_id,
                base_incident_id=base_incident_id,
                stratum=stratum,
                injection_family=injection_family,
                model_name=str(matrix_cell["model_name"]),
                arm=str(matrix_cell["arm"]),
                repeat_index=int(matrix_cell["repeat_index"]),
                proposed_actions=list(output.get("llm_actions", [])),
                verifier_compliant=not hard_violation(output),
            )
        )
    return cells


def planned_pair_count(run_matrix_path: Path, manifest_path: Path) -> int:
    matrix = read_json(run_matrix_path)
    scenarios = scenario_index(manifest_path)
    groups: dict[tuple[str, str, str, int], set[str]] = {}
    for cell in matrix.get("cells", []):
        incident_id = str(cell["incident_id"])
        if incident_id not in scenarios:
            raise ValueError(f"Unknown adversarial scenario in run matrix: {incident_id}")
        base_incident_id, stratum, _ = scenarios[incident_id]
        key = (
            base_incident_id,
            str(cell["model_name"]),
            str(cell["arm"]),
            int(cell["repeat_index"]),
        )
        strata = groups.setdefault(key, set())
        if stratum in strata:
            raise ValueError(f"Duplicate {stratum} matrix cell for pair {key}")
        strata.add(stratum)
    incomplete = [key for key, strata in groups.items() if strata != {"clean", "injected"}]
    if incomplete:
        raise ValueError(f"Run matrix has incomplete adversarial pairs: {incomplete}")
    return len(groups)


def planned_base_scenario_count(manifest_path: Path) -> int:
    manifest = read_json(manifest_path)
    base_incident_ids = [
        str(pair["base_incident_id"]) for pair in manifest.get("pairs", [])
    ]
    if not base_incident_ids:
        raise ValueError("Adversarial manifest contains no scenario pairs")
    if len(base_incident_ids) != len(set(base_incident_ids)):
        raise ValueError("Adversarial manifest contains duplicate base incidents")
    return len(base_incident_ids)


def main() -> None:
    args = parse_args()
    report = analyze_adversarial_pairs(
        load_cells(
            args.run_matrix_json.resolve(),
            args.adversarial_manifest_json.resolve(),
            args.outputs_incidents_dir.resolve(),
        ),
        planned_pair_count=planned_pair_count(
            args.run_matrix_json.resolve(),
            args.adversarial_manifest_json.resolve(),
        ),
        planned_base_scenario_count=planned_base_scenario_count(
            args.adversarial_manifest_json.resolve()
        ),
    )
    output_path = args.output_json.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        report.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote adversarial report: {output_path}")


if __name__ == "__main__":
    main()
