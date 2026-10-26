"""Portable, provenance-separated scenario bank."""

from __future__ import annotations

import hashlib
import json
import random
import shutil
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from soc_llm_policy.io import parse_incident_meta, read_json

ScenarioStratum = Literal["real", "derived_counterfactual", "authored_witness"]

_ANTECEDENT_TOGGLE = {
    "critical": "low",
    "high": "low",
    "medium": "high",
    "low": "high",
}
_FLIP_PREFIX = {"asset_criticality": "CF_CRIT", "severity": "CF_SEV"}
_META_FILENAME = "incident_meta.json"
_PACKAGES_DIRNAME = "packages"


class ScenarioBankDesign(BaseModel):
    bank_id: str = Field(..., min_length=1)
    source_corpus_id: str = Field(..., min_length=1)
    seed: int
    extreme_levels: tuple[str, ...] = ("low", "critical")
    middle_sample_size: int = Field(default=40, ge=0)


class ScenarioRecord(BaseModel):
    scenario_id: str = Field(..., min_length=1)
    stratum: ScenarioStratum
    package_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    base_incident_id: str | None = None
    flipped_field: str | None = None
    flip_from: str | None = None
    flip_to: str | None = None

    @model_validator(mode="after")
    def validate_provenance(self) -> ScenarioRecord:
        flip_values = (self.flipped_field, self.flip_from, self.flip_to)
        if self.stratum == "derived_counterfactual":
            if self.base_incident_id is None or any(
                value is None for value in flip_values
            ):
                raise ValueError(
                    "derived_counterfactual records require base_incident_id "
                    "and all flip fields"
                )
        elif self.base_incident_id is not None or any(
            value is not None for value in flip_values
        ):
            raise ValueError(
                f"{self.stratum} records must not carry derivation fields"
            )
        return self


class ScenarioBankManifest(BaseModel):
    schema_version: Literal[1] = 1
    bank_id: str = Field(..., min_length=1)
    source_corpus_id: str = Field(..., min_length=1)
    source_corpus_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    seed: int
    records: list[ScenarioRecord] = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_records(self) -> ScenarioBankManifest:
        scenario_ids = [record.scenario_id for record in self.records]
        if len(scenario_ids) != len(set(scenario_ids)):
            raise ValueError("scenario_ids must be unique within a bank")
        if scenario_ids != sorted(scenario_ids):
            raise ValueError("scenario records must be sorted by scenario_id")
        return self

    def count_by_stratum(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self.records:
            counts[record.stratum] = counts.get(record.stratum, 0) + 1
        return counts


def toggle_level(level: str) -> str:
    return _ANTECEDENT_TOGGLE[level]


def package_sha256(package_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(package_dir.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(package_dir).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def corpus_sha256(incidents_dir: Path) -> str:
    digest = hashlib.sha256()
    incident_dirs = sorted(
        path for path in incidents_dir.iterdir() if path.is_dir()
    )
    for incident_dir in incident_dirs:
        digest.update(incident_dir.name.encode())
        digest.update(package_sha256(incident_dir).encode())
    return digest.hexdigest()


def _incident_ids_by_level(incidents_dir: Path) -> dict[str, list[str]]:
    by_level: dict[str, list[str]] = {}
    incident_dirs = sorted(
        path for path in incidents_dir.iterdir() if path.is_dir()
    )
    for incident_dir in incident_dirs:
        meta = parse_incident_meta(read_json(incident_dir / _META_FILENAME))
        by_level.setdefault(meta.asset_criticality, []).append(incident_dir.name)
    return by_level


def _select_base_incidents(
    incidents_dir: Path,
    design: ScenarioBankDesign,
) -> list[str]:
    by_level = _incident_ids_by_level(incidents_dir)
    selected: list[str] = []
    for level in design.extreme_levels:
        selected.extend(by_level.get(level, []))
    middle = sorted(
        incident_id
        for level, incident_ids in by_level.items()
        if level not in design.extreme_levels
        for incident_id in incident_ids
    )
    sample_size = min(design.middle_sample_size, len(middle))
    selected.extend(random.Random(design.seed).sample(middle, sample_size))
    return sorted(selected)


def _derive_package(
    *,
    base_dir: Path,
    derived_dir: Path,
    scenario_id: str,
    flipped_field: str,
) -> tuple[str, str]:
    shutil.copytree(base_dir, derived_dir)
    meta_path = derived_dir / _META_FILENAME
    meta = read_json(meta_path)
    flip_from = str(meta[flipped_field])
    flip_to = toggle_level(flip_from)
    meta[flipped_field] = flip_to
    meta["incident_id"] = scenario_id
    parse_incident_meta(meta)
    meta_path.write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return flip_from, flip_to


def build_scenario_bank(
    *,
    incidents_dir: Path,
    output_dir: Path,
    design: ScenarioBankDesign,
    authored_witnesses_dir: Path | None = None,
) -> ScenarioBankManifest:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Output directory is not empty: {output_dir}")
    packages_dir = output_dir / _PACKAGES_DIRNAME
    packages_dir.mkdir(parents=True, exist_ok=True)

    records = [
        ScenarioRecord(
            scenario_id=incident_dir.name,
            stratum="real",
            package_sha256=package_sha256(incident_dir),
        )
        for incident_dir in sorted(
            path for path in incidents_dir.iterdir() if path.is_dir()
        )
    ]

    if authored_witnesses_dir is not None:
        for witness_dir in sorted(
            path for path in authored_witnesses_dir.iterdir() if path.is_dir()
        ):
            scenario_id = witness_dir.name
            meta = parse_incident_meta(read_json(witness_dir / _META_FILENAME))
            if meta.incident_id != scenario_id:
                raise ValueError(
                    "Authored-witness directory and incident_id disagree: "
                    f"{scenario_id!r} != {meta.incident_id!r}"
                )
            destination = packages_dir / scenario_id
            if destination.exists():
                raise ValueError(f"Duplicate scenario_id: {scenario_id}")
            shutil.copytree(witness_dir, destination)
            records.append(
                ScenarioRecord(
                    scenario_id=scenario_id,
                    stratum="authored_witness",
                    package_sha256=package_sha256(destination),
                )
            )

    for base_incident_id in _select_base_incidents(incidents_dir, design):
        for flipped_field, prefix in _FLIP_PREFIX.items():
            scenario_id = f"{prefix}__{base_incident_id}"
            derived_dir = packages_dir / scenario_id
            flip_from, flip_to = _derive_package(
                base_dir=incidents_dir / base_incident_id,
                derived_dir=derived_dir,
                scenario_id=scenario_id,
                flipped_field=flipped_field,
            )
            records.append(
                ScenarioRecord(
                    scenario_id=scenario_id,
                    stratum="derived_counterfactual",
                    package_sha256=package_sha256(derived_dir),
                    base_incident_id=base_incident_id,
                    flipped_field=flipped_field,
                    flip_from=flip_from,
                    flip_to=flip_to,
                )
            )

    manifest = ScenarioBankManifest(
        bank_id=design.bank_id,
        source_corpus_id=design.source_corpus_id,
        source_corpus_sha256=corpus_sha256(incidents_dir),
        seed=design.seed,
        records=sorted(records, key=lambda record: record.scenario_id),
    )
    write_manifest(manifest, output_dir / "scenario_bank_manifest.json")
    return manifest


def write_manifest(manifest: ScenarioBankManifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            manifest.model_dump(mode="json"),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def load_manifest(path: Path) -> ScenarioBankManifest:
    return ScenarioBankManifest.model_validate_json(path.read_text(encoding="utf-8"))


def bind_manifest_to_packages(
    *,
    manifest: ScenarioBankManifest,
    source_incidents_dir: Path,
    bank_dir: Path,
) -> ScenarioBankManifest:
    """Bind a manifest to the exact package bytes at a release boundary."""
    records = [
        record.model_copy(
            update={
                "package_sha256": package_sha256(
                    _source_package(
                        record=record,
                        source_incidents_dir=source_incidents_dir,
                        bank_dir=bank_dir,
                    )
                )
            }
        )
        for record in manifest.records
    ]
    return manifest.model_copy(
        update={
            "source_corpus_sha256": corpus_sha256(source_incidents_dir),
            "records": records,
        }
    )


def _source_package(
    *,
    record: ScenarioRecord,
    source_incidents_dir: Path,
    bank_dir: Path,
) -> Path:
    if record.stratum == "real":
        return source_incidents_dir / record.scenario_id
    return bank_dir / _PACKAGES_DIRNAME / record.scenario_id


def materialize_evaluation_dataset(
    *,
    manifest: ScenarioBankManifest,
    source_incidents_dir: Path,
    bank_dir: Path,
    destination: Path,
) -> Path:
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"Destination directory is not empty: {destination}")
    observed_corpus_sha256 = corpus_sha256(source_incidents_dir)
    if observed_corpus_sha256 != manifest.source_corpus_sha256:
        raise ValueError(
            "Source corpus hash mismatch: "
            f"manifest={manifest.source_corpus_sha256} "
            f"observed={observed_corpus_sha256}"
        )

    dataset_dir = destination / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for record in manifest.records:
        source_dir = _source_package(
            record=record,
            source_incidents_dir=source_incidents_dir,
            bank_dir=bank_dir,
        )
        observed_package_sha256 = package_sha256(source_dir)
        if observed_package_sha256 != record.package_sha256:
            raise ValueError(
                f"Package hash mismatch for {record.scenario_id}: "
                f"manifest={record.package_sha256} "
                f"observed={observed_package_sha256}"
            )
        shutil.copytree(source_dir, dataset_dir / record.scenario_id)

    write_manifest(manifest, destination / "scenario_bank_manifest.json")
    return dataset_dir
