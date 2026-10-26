from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from soc_llm_policy.adversarial_scenarios import (
    load_adversarial_manifest,
    validate_adversarial_bank,
)
from soc_llm_policy.paths import repo_relative_path
from soc_llm_policy.scenario_bank import (
    load_manifest as load_scenario_manifest,
)

CAMPAIGN_PROTOCOL_SCHEMA = 1


class TokenAssumption(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int = Field(..., ge=1)
    completion_tokens: int = Field(..., ge=1)


class ProviderTransportProtocol(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    attempt_limit: int = Field(..., ge=1)
    retry_wait_seconds: float = Field(..., ge=0)
    connectivity_probe_timeout_seconds: float = Field(..., gt=0)
    connectivity_probe_interval_seconds: float = Field(..., gt=0)
    connectivity_recovery_timeout_seconds: float = Field(..., gt=0)
    connectivity_required_successes: int = Field(..., ge=1)


class PrimaryProtocol(BaseModel):
    model_config = ConfigDict(extra="forbid")

    incident_count: int = Field(..., ge=1)
    policy_evidence_tier: Literal["operational"] = "operational"
    models: list[str] = Field(..., min_length=1)
    arms: list[str] = Field(..., min_length=1)
    repeats: int = Field(..., ge=1)
    repeat_reduction: Literal["first", "last", "majority", "any"]
    execution_order: Literal["grouped", "randomized", "paired_blocked"]
    matrix_seed: int

    @model_validator(mode="after")
    def validate_majority(self) -> PrimaryProtocol:
        if self.repeat_reduction == "majority" and self.repeats % 2 == 0:
            raise ValueError("Majority repeat reduction requires an odd repeat count.")
        return self


class IndependentJudgeProtocol(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(..., min_length=1)
    input_population: Literal["primary_plan_cells"] = "primary_plan_cells"
    token_assumption: TokenAssumption


class JudgeAblationProtocol(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(..., min_length=1)
    sample_size: int = Field(..., ge=1)
    seed: int
    stratify_by: list[Literal["model", "arm"]] = Field(..., min_length=1)
    token_assumption: TokenAssumption


class AdversarialProtocol(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pair_manifest: str = Field(..., min_length=1)
    scenario_manifest: str = Field(..., min_length=1)
    source_partition: str = Field(..., min_length=1)
    models: list[str] = Field(..., min_length=1)
    arms: list[str] = Field(..., min_length=1)
    repeats: int = Field(default=1, ge=1)
    repeat_reduction: Literal["first", "last", "majority", "any"]
    execution_order: Literal["grouped", "randomized", "paired_blocked"]
    matrix_seed: int
    token_assumptions: dict[str, TokenAssumption]


class ProjectionFidelityProtocol(BaseModel):
    model_config = ConfigDict(extra="forbid")

    probe: str = Field(..., min_length=1)
    models: list[str] = Field(..., min_length=1)
    token_assumptions: dict[str, TokenAssumption]


class DiagnosticProtocols(BaseModel):
    model_config = ConfigDict(extra="forbid")

    independent_judge: IndependentJudgeProtocol
    judge_ablation: JudgeAblationProtocol
    adversarial: AdversarialProtocol
    projection_fidelity: ProjectionFidelityProtocol


class CampaignProtocol(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int
    execution_sequence: list[
        Literal[
            "primary",
            "independent_judge",
            "judge_ablation",
            "adversarial",
            "projection_fidelity",
        ]
    ]
    provider_transport: ProviderTransportProtocol
    primary: PrimaryProtocol
    diagnostics: DiagnosticProtocols

    @model_validator(mode="after")
    def validate_roles(self) -> CampaignProtocol:
        if self.schema_version != CAMPAIGN_PROTOCOL_SCHEMA:
            raise ValueError("Unsupported campaign protocol schema.")
        expected_sequence = [
            "primary",
            "independent_judge",
            "judge_ablation",
            "adversarial",
            "projection_fidelity",
        ]
        if self.execution_sequence != expected_sequence:
            raise ValueError(
                "Campaign execution sequence must list every study once in "
                "dependency order."
            )
        if self.diagnostics.adversarial.models != self.primary.models:
            raise ValueError("Adversarial models must match the primary model set.")
        if self.diagnostics.adversarial.arms != self.primary.arms:
            raise ValueError("Adversarial arms must match the primary arm set.")
        if (
            self.diagnostics.adversarial.repeat_reduction == "majority"
            and self.diagnostics.adversarial.repeats % 2 == 0
        ):
            raise ValueError(
                "Adversarial majority reduction requires an odd repeat count."
            )
        if self.diagnostics.projection_fidelity.models != self.primary.models:
            raise ValueError("Projection-fidelity models must match the primary set.")
        required_adversarial_models = set(self.diagnostics.adversarial.models)
        if set(self.diagnostics.adversarial.token_assumptions) != required_adversarial_models:
            raise ValueError("Adversarial token assumptions must cover every model.")
        required_fidelity_models = set(self.diagnostics.projection_fidelity.models)
        if set(self.diagnostics.projection_fidelity.token_assumptions) != required_fidelity_models:
            raise ValueError("Projection-fidelity token assumptions must cover every model.")
        return self


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_campaign_protocol(path: Path) -> CampaignProtocol:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return CampaignProtocol.model_validate(payload)


def _repo_file(repo_root: Path, value: str, label: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} must be a repository-relative path.")
    path = (repo_root / relative).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    return path


def _projection_probe_count(path: Path) -> int:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise ValueError("Projection-fidelity probe must contain an items list.")
    plan_ids = [item.get("plan_id") for item in payload["items"] if isinstance(item, dict)]
    if len(plan_ids) != len(payload["items"]) or any(
        not isinstance(plan_id, str) or not plan_id for plan_id in plan_ids
    ):
        raise ValueError("Every projection-fidelity item must have a plan_id.")
    if len(plan_ids) != len(set(plan_ids)):
        raise ValueError("Projection-fidelity plan IDs must be unique.")
    return len(plan_ids)


def build_diagnostic_freeze(
    *,
    repo_root: Path,
    campaign_protocol_path: Path,
    protocol: CampaignProtocol,
    primary_cell_count: int,
) -> tuple[dict[str, object], dict[str, dict[str, str]]]:
    if primary_cell_count != (
        protocol.primary.incident_count
        * len(protocol.primary.models)
        * len(protocol.primary.arms)
        * protocol.primary.repeats
    ):
        raise ValueError("Primary cell count differs from the campaign protocol.")
    if protocol.diagnostics.judge_ablation.sample_size > primary_cell_count:
        raise ValueError("Judge-ablation sample exceeds the primary cell count.")

    adversarial = protocol.diagnostics.adversarial
    pair_manifest_path = _repo_file(
        repo_root, adversarial.pair_manifest, "Adversarial pair manifest"
    )
    scenario_manifest_path = _repo_file(
        repo_root, adversarial.scenario_manifest, "Adversarial scenario manifest"
    )
    bank_dir = pair_manifest_path.parent
    pair_manifest = load_adversarial_manifest(pair_manifest_path)
    validate_adversarial_bank(pair_manifest, bank_dir)
    scenario_manifest = load_scenario_manifest(scenario_manifest_path)
    scenario_ids = [record.scenario_id for record in scenario_manifest.records]
    pair_count = len(pair_manifest.pairs)
    pair_scenario_ids = sorted(
        scenario_id
        for pair in pair_manifest.pairs
        for scenario_id in (pair.clean_scenario_id, pair.injected_scenario_id)
    )
    if sorted(scenario_ids) != pair_scenario_ids:
        raise ValueError(
            "Adversarial scenario manifest must contain both variants of every pair."
        )
    if (
        pair_manifest.source_corpus_id != scenario_manifest.source_corpus_id
        or pair_manifest.source_corpus_sha256
        != scenario_manifest.source_corpus_sha256
    ):
        raise ValueError("Adversarial manifests must bind the same source corpus.")

    fidelity = protocol.diagnostics.projection_fidelity
    probe_path = _repo_file(repo_root, fidelity.probe, "Projection-fidelity probe")
    probe_count = _projection_probe_count(probe_path)
    independent_judge_count = primary_cell_count
    ablation_count = protocol.diagnostics.judge_ablation.sample_size
    adversarial_count = (
        len(scenario_ids)
        * len(adversarial.models)
        * len(adversarial.arms)
        * adversarial.repeats
    )
    fidelity_count = probe_count * len(fidelity.models)

    input_paths = {
        "campaign_protocol": campaign_protocol_path,
        "adversarial_pair_manifest": pair_manifest_path,
        "adversarial_scenario_manifest": scenario_manifest_path,
        "projection_fidelity_probe": probe_path,
        "llm_judge_protocol": repo_root / "src/soc_llm_policy/llm_judge.py",
        "campaign_diagnostics": repo_root
        / "src/soc_llm_policy/campaign_diagnostics.py",
    }
    input_hashes = {
        role: {
            "path": repo_relative_path(path, repo_root),
            "sha256": sha256_file(path),
        }
        for role, path in input_paths.items()
    }

    studies: dict[str, object] = {
        "independent_judge": {
            "kind": "dependent_diagnostic",
            "depends_on": "primary",
            "model": protocol.diagnostics.independent_judge.model,
            "input_population": "primary_plan_cells",
            "task_count_ceiling": independent_judge_count,
            "token_assumption": protocol.diagnostics.independent_judge.token_assumption.model_dump(),
        },
        "judge_ablation": {
            "kind": "fixed_sample_diagnostic",
            "depends_on": "primary",
            "model": protocol.diagnostics.judge_ablation.model,
            "sample_size": ablation_count,
            "seed": protocol.diagnostics.judge_ablation.seed,
            "stratify_by": protocol.diagnostics.judge_ablation.stratify_by,
            "task_count_ceiling": ablation_count,
            "token_assumption": protocol.diagnostics.judge_ablation.token_assumption.model_dump(),
        },
        "adversarial": {
            "kind": "fixed_matrix_diagnostic",
            "depends_on": "primary_protocol",
            "source_partition": adversarial.source_partition,
            "models": adversarial.models,
            "arms": adversarial.arms,
            "repeats": adversarial.repeats,
            "repeat_reduction": adversarial.repeat_reduction,
            "execution_order": adversarial.execution_order,
            "matrix_seed": adversarial.matrix_seed,
            "pair_count": pair_count,
            "scenario_count": len(scenario_ids),
            "scenario_ids": scenario_ids,
            "task_count_ceiling": adversarial_count,
            "pair_manifest": input_hashes["adversarial_pair_manifest"],
            "scenario_manifest": input_hashes["adversarial_scenario_manifest"],
            "token_assumptions": {
                name: assumption.model_dump()
                for name, assumption in adversarial.token_assumptions.items()
            },
        },
        "projection_fidelity": {
            "kind": "fixed_matrix_diagnostic",
            "depends_on": "primary_protocol",
            "models": fidelity.models,
            "plan_count": probe_count,
            "task_count_ceiling": fidelity_count,
            "probe": input_hashes["projection_fidelity_probe"],
            "token_assumptions": {
                name: assumption.model_dump()
                for name, assumption in fidelity.token_assumptions.items()
            },
        },
    }
    studies["diagnostic_task_ceiling"] = (
        independent_judge_count + ablation_count + adversarial_count + fidelity_count
    )
    studies["execution_sequence"] = protocol.execution_sequence
    return studies, input_hashes


def canonical_protocol_json(protocol: CampaignProtocol) -> str:
    return json.dumps(
        protocol.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
