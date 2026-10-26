#!/usr/bin/env python3
"""Build a versioned separation report from persisted verifier outputs.

Compares a prompt-only accept-all baseline and an optional model judge against
the typed verifier over already-produced plans, reporting false-accept and
false-reject with explicit denominators. An optional, separately produced
LLM-judge verdict set can join by experiment cell ID.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from soc_llm_policy.separation_analysis import (
    BASELINE_DESCRIPTIONS,
    LLMJudgeVerdictSet,
    SeparationReportEnvelope,
    SeparationVerifierOutput,
    attach_llm_judge_verdicts,
    build_separation_cells,
    separation_cell_traces,
    separation_metrics,
)

_EXCLUDED_BASELINES = [
    "llm_judge (model-produced; attached by the provider campaign)",
]


class _RecommendationSource(BaseModel):
    kind: str = Field(..., min_length=1)
    constitutes_model_evidence: bool


class _ExperimentMetadata(BaseModel):
    run_id: str = Field(..., min_length=1)
    dataset_release_id: str = Field(..., min_length=1)
    git_commit: str | None = None
    recommendation_source: _RecommendationSource


class _AnalysisBundle(BaseModel):
    experiment: _ExperimentMetadata
    generated_at_utc: str | None = None


_AnalysisBundle.model_rebuild()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must be a JSON object")
    return payload


def _stable_schema_payload(model: type[BaseModel]) -> tuple[dict[str, Any], str]:
    schema = model.model_json_schema()
    encoded = json.dumps(
        schema,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return schema, hashlib.sha256(encoded).hexdigest()


def _limitations(
    metadata: _ExperimentMetadata,
    llm_judge_included: bool,
) -> list[str]:
    limitations = []
    if not llm_judge_included:
        limitations.extend(
            [
                (
                    "The LLM-judge baseline is excluded until model-produced "
                    "verdicts are available."
                ),
                "This report evaluates deterministic separation baselines only.",
            ]
        )
    if not metadata.recommendation_source.constitutes_model_evidence:
        limitations.insert(
            0,
            (
                "Fixture replay validates analysis plumbing and does not "
                "constitute model evidence."
            ),
        )
    return limitations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-incidents-dir", type=Path, required=True)
    parser.add_argument("--analysis-bundle-json", type=Path, required=True)
    parser.add_argument("--llm-judge-verdicts-json", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "Comma-separated experiment run ids (primary,recovery). Restricts "
            "which persisted outputs join and supplies the run identity when "
            "the bundle carries no experiment block."
        ),
    )
    parser.add_argument("--dataset-release-id", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    bundle_payload = _read_json(args.analysis_bundle_json.expanduser().resolve())
    run_ids = [
        rid.strip() for rid in str(args.run_id or "").split(",") if rid.strip()
    ]
    if "experiment" in bundle_payload:
        analysis_bundle = _AnalysisBundle.model_validate(bundle_payload)
        metadata = analysis_bundle.experiment
        generated_at_utc = analysis_bundle.generated_at_utc
    else:
        # Bundles from the current analysis carry no experiment block; the
        # run identity comes from the CLI and the plans are real model output.
        if not run_ids or not args.dataset_release_id:
            raise ValueError(
                "A bundle without an experiment block requires --run-id and "
                "--dataset-release-id."
            )
        metadata = _ExperimentMetadata(
            run_id=run_ids[0],
            dataset_release_id=str(args.dataset_release_id),
            recommendation_source=_RecommendationSource(
                kind="provider_llm",
                constitutes_model_evidence=True,
            ),
        )
        generated_at_utc = bundle_payload.get("generated_at_utc")
    output_files = sorted(
        args.outputs_incidents_dir.expanduser().resolve().rglob(
            "verifier_output_llm_*.json"
        )
    )
    if not output_files:
        raise FileNotFoundError(
            f"No verifier outputs under {args.outputs_incidents_dir}"
        )

    verifier_outputs = [_read_json(path) for path in output_files]
    if run_ids:
        verifier_outputs = [
            payload
            for payload in verifier_outputs
            if payload.get("experiment_run_id") in set(run_ids)
        ]
        if not verifier_outputs:
            raise ValueError(f"No verifier outputs match run ids {run_ids}")
    cells = build_separation_cells(verifier_outputs)
    llm_judge = None
    if args.llm_judge_verdicts_json is not None:
        llm_judge = LLMJudgeVerdictSet.model_validate(
            _read_json(args.llm_judge_verdicts_json.expanduser().resolve())
        )
        if llm_judge.run_id != metadata.run_id:
            raise ValueError("LLM-judge run_id does not match analysis bundle")
        cells = attach_llm_judge_verdicts(cells, llm_judge)
    report = separation_metrics(cells)
    input_schema, input_schema_sha256 = _stable_schema_payload(
        SeparationVerifierOutput
    )

    payload = SeparationReportEnvelope(
        input_schema_sha256=input_schema_sha256,
        run_id=metadata.run_id,
        dataset_release_id=metadata.dataset_release_id,
        generated_at=(generated_at_utc or datetime.now(UTC).isoformat()),
        git_commit=metadata.git_commit,
        constitutes_model_evidence=(
            metadata.recommendation_source.constitutes_model_evidence
        ),
        recommendation_source_kind=metadata.recommendation_source.kind,
        limitations=_limitations(metadata, llm_judge is not None),
        baseline_descriptions=BASELINE_DESCRIPTIONS,
        excluded_baselines=(
            [] if llm_judge is not None else _EXCLUDED_BASELINES
        ),
        report=report,
        cells=separation_cell_traces(cells),
    )
    output_path = args.output_json.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    input_schema_path = output_path.with_name("separation_input.schema.json")
    input_schema_path.write_text(
        json.dumps(input_schema, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_schema_path = output_path.with_name("separation_report.schema.json")
    report_schema_path.write_text(
        json.dumps(
            SeparationReportEnvelope.model_json_schema(),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    judge_schema_path = output_path.with_name("llm_judge_input.schema.json")
    judge_schema_path.write_text(
        json.dumps(
            LLMJudgeVerdictSet.model_json_schema(),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved separation report at: {output_path}")


if __name__ == "__main__":
    main()
