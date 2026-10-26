"""Convert the adversarial paired-bank manifest into a portable scenario bank.

`build_adversarial_scenario_bank.py` writes `adversarial_bank_manifest.json`
in a bank-specific schema (a list of clean/injected pairs). The experiment
runner's `--scenario-manifest` flag instead consumes the generic
`ScenarioBankManifest` schema (a flat, sorted `records` list). This script
reshapes one into the other without recomputing any hash: every package hash
already in the pairs manifest is reused verbatim, and the conversion is
validated by re-parsing the output through the same model the runner uses.

Each clean and injected package is recorded as an `authored_witness`
scenario: it is not a real corpus incident on its own path (so it must be
sourced from the bank's `packages/` directory, not `dataset/incidents/`) and
it is not a mechanical field-flip counterfactual (so it carries no
derivation fields). This is the accurate provenance label for a real base
incident with authored adversarial content layered onto it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from soc_llm_policy.scenario_bank import ScenarioBankManifest, ScenarioRecord


def convert(pairs_manifest: dict) -> ScenarioBankManifest:
    records = []
    for pair in pairs_manifest["pairs"]:
        records.append(
            ScenarioRecord(
                scenario_id=pair["clean_scenario_id"],
                stratum="authored_witness",
                package_sha256=pair["clean_package_sha256"],
            )
        )
        records.append(
            ScenarioRecord(
                scenario_id=pair["injected_scenario_id"],
                stratum="authored_witness",
                package_sha256=pair["injected_package_sha256"],
            )
        )
    records.sort(key=lambda record: record.scenario_id)
    return ScenarioBankManifest(
        schema_version=pairs_manifest["schema_version"],
        bank_id=pairs_manifest["bank_id"],
        source_corpus_id=pairs_manifest["source_corpus_id"],
        source_corpus_sha256=pairs_manifest["source_corpus_sha256"],
        seed=pairs_manifest["seed"],
        records=records,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adversarial-bank-manifest", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args(argv)

    pairs_manifest = json.loads(
        args.adversarial_bank_manifest.read_text(encoding="utf-8")
    )
    manifest = convert(pairs_manifest)

    # Round-trip through the exact validator the runner uses before writing,
    # so a schema drift fails here rather than silently at execution time.
    payload = manifest.model_dump_json(indent=2)
    ScenarioBankManifest.model_validate_json(payload)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(payload + "\n", encoding="utf-8")
    print(f"Wrote {args.output_json} ({len(manifest.records)} records)")


if __name__ == "__main__":
    main()
