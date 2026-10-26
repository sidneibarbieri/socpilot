from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

PolicyScopeKind = Literal["global", "content_triggered"]
PolicyInputRole = Literal[
    "action_catalog",
    "policy_rules",
    "proportionality_policy",
]


class PolicyInputReference(BaseModel):
    role: PolicyInputRole
    path: str = Field(..., min_length=1)
    sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    policy_id: str | None = None
    review_status: Literal[
        "declared",
        "pending_review",
        "approved_for_experiment",
    ] | None = None


class PolicyBundleReference(BaseModel):
    schema_version: Literal[1] = 1
    bundle_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    inputs: list[PolicyInputReference] = Field(..., min_length=2)

    @model_validator(mode="after")
    def validate_unique_roles(self) -> PolicyBundleReference:
        roles = [item.role for item in self.inputs]
        if len(roles) != len(set(roles)):
            raise ValueError("Policy bundle input roles must be unique")
        required_roles = {"action_catalog", "policy_rules"}
        if not required_roles.issubset(roles):
            raise ValueError(
                "Policy bundle must include action_catalog and policy_rules"
            )
        return self


class PolicyRuleOutcome(BaseModel):
    rule_id: str = Field(..., min_length=1)
    applicable: bool
    violated: bool
    violation_type: str | None = None

    @model_validator(mode="after")
    def validate_outcome(self) -> PolicyRuleOutcome:
        if self.violated and not self.applicable:
            raise ValueError("A rule cannot be violated when it is not applicable")
        if self.violated and not self.violation_type:
            raise ValueError("A violated rule must declare violation_type")
        if not self.violated and self.violation_type:
            raise ValueError("A non-violated rule cannot declare violation_type")
        return self


class PolicyFamilyOutcome(BaseModel):
    family_id: str = Field(..., min_length=1)
    scope_kind: PolicyScopeKind
    applicable: bool
    violated: bool
    rule_outcomes: list[PolicyRuleOutcome] = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_outcome(self) -> PolicyFamilyOutcome:
        if self.violated and not self.applicable:
            raise ValueError("A family cannot be violated when it is not applicable")
        if self.applicable != any(item.applicable for item in self.rule_outcomes):
            raise ValueError("Family applicability disagrees with rule outcomes")
        if self.violated != any(item.violated for item in self.rule_outcomes):
            raise ValueError("Family violation disagrees with rule outcomes")
        return self


class PolicyFamilyAggregate(BaseModel):
    family_id: str = Field(..., min_length=1)
    scope_kind: PolicyScopeKind
    cell_count: int = Field(..., ge=0)
    applicable_count: int = Field(..., ge=0)
    violated_count: int = Field(..., ge=0)
    applicability_rate: float = Field(..., ge=0.0, le=1.0)
    violation_rate: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_counts(self) -> PolicyFamilyAggregate:
        if self.applicable_count > self.cell_count:
            raise ValueError("applicable_count cannot exceed cell_count")
        if self.violated_count > self.applicable_count:
            raise ValueError("violated_count cannot exceed applicable_count")
        expected_applicability_rate = (
            round(self.applicable_count / self.cell_count, 4)
            if self.cell_count
            else 0.0
        )
        if self.applicability_rate != expected_applicability_rate:
            raise ValueError("applicability_rate disagrees with counts")
        expected_violation_rate = (
            round(self.violated_count / self.applicable_count, 4)
            if self.applicable_count
            else None
        )
        if self.violation_rate != expected_violation_rate:
            raise ValueError("violation_rate disagrees with counts")
        return self


class PolicyFamilyMetricsReport(BaseModel):
    schema_version: Literal[1] = 1
    total_cells: int = Field(..., ge=0)
    families: list[PolicyFamilyAggregate] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_cell_counts(self) -> PolicyFamilyMetricsReport:
        if any(item.cell_count != self.total_cells for item in self.families):
            raise ValueError("Family cell_count disagrees with total_cells")
        return self


class ViolationOutput(BaseModel):
    rule_id: str = Field(..., min_length=1)
    severity: Literal["hard", "soft", "warning"]
    type: str = Field(..., min_length=1)
    action: str = Field(..., min_length=1)
    repair: str | None = None
    missing_before: str | None = None


class AttackTechniqueRef(BaseModel):
    technique_id: str = Field(..., min_length=1)
    technique_name: str = Field(..., min_length=1)


class AttackTacticRef(BaseModel):
    tactic_id: str = Field(..., min_length=1)
    tactic_name: str = Field(..., min_length=1)


class AttackTechniqueCount(BaseModel):
    technique_id: str = Field(..., min_length=1)
    technique_name: str = Field(..., min_length=1)
    count: int = Field(..., ge=0)


class AttackTacticCount(BaseModel):
    tactic_id: str = Field(..., min_length=1)
    tactic_name: str = Field(..., min_length=1)
    count: int = Field(..., ge=0)


class AttackCandidate(BaseModel):
    rule_id: str = Field(..., min_length=1)
    technique_id: str = Field(..., min_length=1)
    technique_name: str = Field(..., min_length=1)
    tactic_id: str = Field(..., min_length=1)
    tactic_name: str = Field(..., min_length=1)
    event_type: str = Field(..., min_length=1)
    score: float = Field(..., ge=0.0, le=1.0)
    matched_signals: list[str] = Field(default_factory=list)
    evidence: str = Field(..., min_length=1)


class AttackMatch(BaseModel):
    rule_id: str = Field(..., min_length=1)
    technique_id: str = Field(..., min_length=1)
    technique_name: str = Field(..., min_length=1)
    tactic_id: str = Field(..., min_length=1)
    tactic_name: str = Field(..., min_length=1)
    event_type: str = Field(..., min_length=1)
    confidence: float = Field(..., ge=0.0, le=1.0)
    matched_signals: list[str] = Field(default_factory=list)
    evidence: str = Field(..., min_length=1)


class AttackReference(BaseModel):
    stix_path: str = Field(..., min_length=1)
    sha256: str = Field(..., min_length=1)
    object_count: int = Field(..., ge=0)
    technique_count: int = Field(..., ge=0)
    tactic_count: int = Field(..., ge=0)
    latest_modified: str = Field(..., min_length=1)


class AttackCandidatesSummary(BaseModel):
    rule_count: int = Field(..., ge=0)
    candidate_count: int = Field(..., ge=0)
    average_score: float = Field(..., ge=0.0, le=1.0)
    technique_counts: list[AttackTechniqueCount] = Field(default_factory=list)
    tactic_counts: list[AttackTacticCount] = Field(default_factory=list)
    candidates: list[AttackCandidate] = Field(default_factory=list)


class AttackContextSummary(BaseModel):
    rule_count: int = Field(..., ge=0)
    match_count: int = Field(..., ge=0)
    high_confidence_match_count: int = Field(..., ge=0)
    techniques: list[AttackTechniqueRef] = Field(default_factory=list)
    tactics: list[AttackTacticRef] = Field(default_factory=list)
    technique_counts: list[AttackTechniqueCount] = Field(default_factory=list)
    tactic_counts: list[AttackTacticCount] = Field(default_factory=list)
    matches: list[AttackMatch] = Field(default_factory=list)


class IncidentMetrics(BaseModel):
    human_action_count: int = Field(..., ge=0)
    llm_action_count: int = Field(..., ge=0)
    enforced_action_count: int = Field(..., ge=0)
    violation_count: int = Field(..., ge=0)
    hard_violation_count: int = Field(..., ge=0)
    soft_violation_count: int = Field(..., ge=0)
    warning_violation_count: int = Field(..., ge=0)
    llm_only_action_count: int = Field(..., ge=0)
    human_only_action_count: int = Field(..., ge=0)
    attack_match_count: int = Field(..., ge=0)
    attack_high_confidence_match_count: int = Field(..., ge=0)
    attack_candidate_count: int = Field(..., ge=0)
    attack_technique_count: int = Field(..., ge=0)
    attack_tactic_count: int = Field(..., ge=0)
    enforcement_actions_inserted_count: int = Field(default=0, ge=0)
    enforcement_actions_removed_count: int = Field(default=0, ge=0)
    enforcement_actions_deferred_count: int = Field(default=0, ge=0)
    enforcement_actions_reordered_count: int = Field(default=0, ge=0)
    enforcement_action_modification_count: int = Field(default=0, ge=0)
    enforcement_modified: bool = False
    llm_hallucinated_action_count: int = Field(default=0, ge=0)
    llm_hallucination_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    precision_raw: float | None = Field(default=None, ge=0.0, le=1.0)
    recall_raw: float | None = Field(default=None, ge=0.0, le=1.0)
    f1_raw: float | None = Field(default=None, ge=0.0, le=1.0)
    jaccard_raw: float | None = Field(default=None, ge=0.0, le=1.0)
    precision_enforced: float | None = Field(default=None, ge=0.0, le=1.0)
    recall_enforced: float | None = Field(default=None, ge=0.0, le=1.0)
    f1_enforced: float | None = Field(default=None, ge=0.0, le=1.0)
    jaccard_enforced: float | None = Field(default=None, ge=0.0, le=1.0)
    delta_jaccard: float | None = Field(default=None, ge=-1.0, le=1.0)


class LLMUsage(BaseModel):
    prompt_tokens: int = Field(..., ge=0)
    completion_tokens: int = Field(..., ge=0)
    total_tokens: int = Field(..., ge=0)


class VerifierOutputModel(BaseModel):
    incident_id: str = Field(..., min_length=1)
    incident_dir_id: str | None = None
    incident_type: str | None = None
    severity: Literal["low", "medium", "high", "critical"] | None = None
    asset_criticality: Literal["low", "medium", "high", "critical"] | None = None
    mode: Literal["LLM", "human"]
    run_tag: str = Field(..., min_length=1)
    approval_policy_mode: Literal["remove", "defer_to_human_approval"] | None = None
    llm_deployment: str | None = None
    llm_arm: Literal["llm_zero", "llm_policy_prompt"] | None = None
    experiment_run_id: str | None = None
    experiment_cell_id: str | None = None
    experiment_repeat_index: int | None = Field(default=None, ge=1)
    experiment_order_index: int | None = Field(default=None, ge=1)
    experiment_corpus_stratum: str | None = Field(default=None, min_length=1)
    human_actions: list[str] = Field(default_factory=list)
    incident_approved_actions: list[str] = Field(default_factory=list)
    llm_actions: list[str] = Field(default_factory=list)
    violations: list[ViolationOutput] = Field(default_factory=list)
    policy_family_outcomes: list[PolicyFamilyOutcome] = Field(default_factory=list)
    policy_bundle: PolicyBundleReference | None = None
    enforced_actions: list[str] = Field(default_factory=list)
    approval_pending_actions: list[str] = Field(default_factory=list)
    attack_reference: AttackReference
    attack_candidates: AttackCandidatesSummary
    attack_summary: AttackContextSummary
    metrics: IncidentMetrics
    llm_usage: LLMUsage | None = None
    llm_latency_ms: int | None = Field(default=None, ge=0)
    llm_cost_estimated_usd: float | None = None

    @field_validator("mode", mode="before")
    @classmethod
    def normalize_legacy_mode(cls, value: object) -> object:
        if value == "humano":
            return "human"
        return value
