# src/soc_llm_policy/paths.py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def repo_relative_path(path: Path, repo_root: Path) -> str:
    resolved_path = path.expanduser().resolve()
    resolved_root = repo_root.expanduser().resolve()
    if resolved_path == resolved_root or resolved_root in resolved_path.parents:
        return resolved_path.relative_to(resolved_root).as_posix()
    return resolved_path.as_posix()


def resolve_repo_relative_path(value: str, repo_root: Path) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (repo_root.expanduser().resolve() / candidate).resolve()


@dataclass(frozen=True)
class RepoPaths:
    repo_root: Path
    inbox_incidents_override: Path | None = None

    def _env_path(self, name: str, default: Path) -> Path:
        override = os.environ.get(name, "").strip()
        if not override:
            return default
        return Path(override).expanduser().resolve()

    @property
    def artifact_data_dir(self) -> Path:
        return self.repo_root / "artifact_data"

    @property
    def artifact_dataset_dir(self) -> Path:
        return self.artifact_data_dir / "dataset"

    @property
    def artifact_global_dir(self) -> Path:
        return self.artifact_data_dir / "global"

    @property
    def artifact_outputs_dir(self) -> Path:
        return self.repo_root / "artifact_outputs"

    @property
    def uses_artifact_layout(self) -> bool:
        return self.artifact_dataset_dir.is_dir() and self.artifact_global_dir.is_dir()

    @property
    def config_dir(self) -> Path:
        return self.repo_root / "config"

    @property
    def reference_data_dir(self) -> Path:
        return self.repo_root / "reference_data"

    @property
    def mitre_dir(self) -> Path:
        return self.reference_data_dir / "mitre"

    @property
    def dataset_dir(self) -> Path:
        return self.repo_root / "dataset"

    @property
    def policy_dir(self) -> Path:
        return self.repo_root / "policy"

    @property
    def incoming_dir(self) -> Path:
        return self.repo_root / "incoming"

    @property
    def incoming_raw_dir(self) -> Path:
        return self.incoming_dir / "raw"

    @property
    def incoming_redacted_dir(self) -> Path:
        return self.incoming_dir / "redacted"

    @property
    def private_dir(self) -> Path:
        return self.repo_root / "private"

    @property
    def private_derived_dir(self) -> Path:
        return self.private_dir / "derived"

    @property
    def local_redaction_dir(self) -> Path:
        return self.repo_root / "local_redaction"

    @property
    def releases_dir(self) -> Path:
        return self.repo_root / "releases"

    @property
    def artifact_dir(self) -> Path:
        return self.repo_root / "artifact"

    @property
    def inbox_dir(self) -> Path:
        return self.artifact_data_dir if self.uses_artifact_layout else self.dataset_dir

    @property
    def inbox_global_dir(self) -> Path:
        return (
            self.artifact_global_dir if self.uses_artifact_layout else self.policy_dir
        )

    @property
    def inbox_incidents_dir(self) -> Path:
        if self.inbox_incidents_override is not None:
            return self.inbox_incidents_override.expanduser().resolve()
        default = (
            self.artifact_dataset_dir
            if self.uses_artifact_layout
            else self.inbox_dir / "incidents"
        )
        return self._env_path(
            "SOC_LLM_POLICY_INBOX_INCIDENTS_DIR",
            default,
        )

    def inbox_incident_dir(self, incident_id: str) -> Path:
        return self.inbox_incidents_dir / incident_id

    @property
    def outputs_dir(self) -> Path:
        default = (
            self.artifact_outputs_dir
            if self.uses_artifact_layout
            else self.repo_root / "results"
        )
        return self._env_path(
            "SOC_LLM_POLICY_OUTPUTS_DIR",
            default,
        )

    @property
    def outputs_global_dir(self) -> Path:
        return self.outputs_dir / "global"

    @property
    def outputs_incidents_dir(self) -> Path:
        return self.outputs_dir / "incidents"

    @property
    def outputs_analysis_dir(self) -> Path:
        return self.outputs_dir / "analysis"

    @property
    def outputs_experiments_dir(self) -> Path:
        return self.outputs_dir / "experiments"

    def outputs_experiment_dir(self, run_id: str) -> Path:
        return self.outputs_experiments_dir / run_id

    def outputs_incident_dir(self, incident_id: str) -> Path:
        return self.outputs_incidents_dir / incident_id

    def outputs_incident_meta_path(self, incident_id: str) -> Path:
        return self.outputs_incident_dir(incident_id) / "incident_meta.json"

    def outputs_incident_telemetry_path(self, incident_id: str) -> Path:
        return self.outputs_incident_dir(incident_id) / "incident_telemetry.jsonl"

    def outputs_incident_human_actions_path(self, incident_id: str) -> Path:
        return self.outputs_incident_dir(incident_id) / "incident_human_actions.jsonl"

    def outputs_llm_actions_path(self, incident_id: str) -> Path:
        """Raw LLM-recommended actions before policy verification."""
        return self.outputs_incident_dir(incident_id) / "llm_raw_actions.json"

    def outputs_llm_actions_versioned_path(
        self,
        incident_id: str,
        run_tag: str,
    ) -> Path:
        return (
            self.outputs_incident_dir(incident_id) / f"llm_raw_actions_{run_tag}.json"
        )

    def outputs_llm_input_snapshot_path(self, incident_id: str) -> Path:
        return self.outputs_incident_dir(incident_id) / "llm_input_snapshot.json"

    def outputs_llm_input_snapshot_versioned_path(
        self,
        incident_id: str,
        run_tag: str,
    ) -> Path:
        return (
            self.outputs_incident_dir(incident_id)
            / f"llm_input_snapshot_{run_tag}.json"
        )

    def outputs_verifier_output_path(self, incident_id: str) -> Path:
        return self.outputs_incident_dir(incident_id) / "verifier_output.json"

    def outputs_verifier_output_versioned_path(
        self,
        incident_id: str,
        run_tag: str,
    ) -> Path:
        return (
            self.outputs_incident_dir(incident_id) / f"verifier_output_{run_tag}.json"
        )

    @property
    def inbox_action_catalog_path(self) -> Path:
        return self._env_path(
            "SOC_LLM_POLICY_ACTION_CATALOG_PATH",
            self.inbox_global_dir / "action_catalog.yaml",
        )

    @property
    def inbox_constraints_path(self) -> Path:
        return self._env_path(
            "SOC_LLM_POLICY_CONSTRAINTS_PATH",
            self.inbox_global_dir / "constraints.yaml",
        )

    @property
    def policy_review_record_path(self) -> Path:
        return self._env_path(
            "SOC_LLM_POLICY_REVIEW_RECORD_PATH",
            self.inbox_global_dir / "review_record.yaml",
        )

    @property
    def anonymization_policy_path(self) -> Path:
        return self.config_dir / "anonymization_policy.yaml"

    @property
    def outputs_action_catalog_path(self) -> Path:
        return self.outputs_global_dir / "action_catalog.yaml"

    @property
    def outputs_constraints_path(self) -> Path:
        return self.outputs_global_dir / "constraints.yaml"

    @property
    def attack_mapping_path(self) -> Path:
        return self.config_dir / "attack_mapping.yaml"

    @property
    def models_freeze_path(self) -> Path:
        return self.config_dir / "models.freeze.yaml"

    @property
    def models_local_path(self) -> Path:
        return self.config_dir / "models.local.yaml"

    @property
    def models_registry_path(self) -> Path:
        override = os.environ.get("SOC_LLM_POLICY_MODELS_REGISTRY_PATH", "").strip()
        if override:
            return Path(override).expanduser().resolve()
        for candidate in (self.models_local_path, self.models_freeze_path):
            if candidate.exists():
                return candidate
        return self.models_local_path

    @property
    def mitre_stix_path(self) -> Path:
        return self.mitre_dir / "enterprise-attack.json"

    @property
    def mitre_manifest_path(self) -> Path:
        return self.mitre_dir / "manifest.json"


def resolve_repo_root(repo_root: str | None) -> Path:
    if repo_root:
        return Path(repo_root).expanduser().resolve()
    return Path.cwd().resolve()
