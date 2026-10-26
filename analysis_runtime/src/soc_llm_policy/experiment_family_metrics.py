"""Persist policy-family measurements for one complete experiment run."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from soc_llm_policy.evaluation_plan import EvaluationCell
from soc_llm_policy.family_metrics import aggregate_families
from soc_llm_policy.result_models import (
    PolicyBundleReference,
    PolicyFamilyMetricsReport,
    VerifierOutputModel,
)


class ExperimentFamilyMetricsReport(BaseModel):
    schema_version: Literal[1] = 1
    experiment_run_id: str = Field(..., min_length=1)
    policy_bundle: PolicyBundleReference
    expected_cell_count: int = Field(..., ge=1)
    observed_cell_count: int = Field(..., ge=1)
    metrics: PolicyFamilyMetricsReport


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _load_run_outputs(
    outputs_incidents_dir: Path,
    experiment_run_id: str,
) -> dict[str, VerifierOutputModel]:
    outputs: dict[str, VerifierOutputModel] = {}
    for path in sorted(outputs_incidents_dir.glob("*/verifier_output_*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("experiment_run_id") != experiment_run_id:
            continue
        verifier = VerifierOutputModel.model_validate(raw)
        cell_id = verifier.experiment_cell_id
        if not cell_id:
            raise ValueError(
                f"Experiment output has no experiment_cell_id: {path}"
            )
        if cell_id in outputs:
            raise ValueError(
                f"Multiple verifier outputs found for experiment cell {cell_id!r}"
            )
        outputs[cell_id] = verifier
    return outputs


def persist_experiment_family_metrics(
    *,
    outputs_incidents_dir: Path,
    run_dir: Path,
    experiment_run_id: str,
    completed_cells: list[EvaluationCell],
) -> ExperimentFamilyMetricsReport:
    if not completed_cells:
        raise ValueError("No completed cells available for family aggregation")

    cell_ids = [cell.cell_id for cell in completed_cells]
    duplicate_ids = sorted({cid for cid in cell_ids if cell_ids.count(cid) > 1})
    if duplicate_ids:
        raise ValueError(
            f"Duplicate completed cells in family aggregation: {duplicate_ids}"
        )

    outputs = _load_run_outputs(outputs_incidents_dir, experiment_run_id)
    expected_ids = set(cell_ids)
    observed_ids = set(outputs)
    if observed_ids != expected_ids:
        missing = sorted(expected_ids - observed_ids)
        unexpected = sorted(observed_ids - expected_ids)
        raise ValueError(
            "Experiment verifier outputs do not match completed cells: "
            f"missing={missing}, unexpected={unexpected}"
        )

    bundles = []
    family_cells = []
    serialized_cells = []
    for cell in completed_cells:
        verifier = outputs[cell.cell_id]
        if verifier.policy_bundle is None:
            raise ValueError(f"Cell {cell.cell_id!r} has no policy bundle")
        if not verifier.policy_family_outcomes:
            raise ValueError(
                f"Cell {cell.cell_id!r} has no policy-family outcomes"
            )
        if (
            cell.policy_bundle_sha256 is not None
            and verifier.policy_bundle.bundle_sha256
            != cell.policy_bundle_sha256
        ):
            raise ValueError(
                f"Cell {cell.cell_id!r} executed under a different policy bundle"
            )
        bundles.append(verifier.policy_bundle)
        family_cells.append(verifier.policy_family_outcomes)
        serialized_cells.append(
            {
                "cell_id": cell.cell_id,
                "order_index": cell.order_index,
                "incident_id": cell.incident_id,
                "model_name": cell.model_name,
                "deployment": cell.deployment,
                "provider": cell.provider,
                "arm": cell.arm,
                "repeat_index": cell.repeat_index,
                "corpus_stratum": cell.corpus_stratum,
                "policy_bundle_sha256": verifier.policy_bundle.bundle_sha256,
                "family_outcomes": [
                    outcome.model_dump(mode="json")
                    for outcome in verifier.policy_family_outcomes
                ],
            }
        )

    bundle_hashes = {bundle.bundle_sha256 for bundle in bundles}
    if len(bundle_hashes) != 1:
        raise ValueError(
            "Cannot aggregate cells executed under different policy bundles"
        )

    report = ExperimentFamilyMetricsReport(
        experiment_run_id=experiment_run_id,
        policy_bundle=bundles[0],
        expected_cell_count=len(completed_cells),
        observed_cell_count=len(outputs),
        metrics=aggregate_families(family_cells),
    )
    cells_jsonl = "".join(
        f"{json.dumps(item, ensure_ascii=False, sort_keys=True)}\n"
        for item in serialized_cells
    )
    _atomic_write(run_dir / "policy_family_cells.jsonl", cells_jsonl)
    _atomic_write(
        run_dir / "policy_family_metrics.json",
        json.dumps(
            report.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        ),
    )
    return report
