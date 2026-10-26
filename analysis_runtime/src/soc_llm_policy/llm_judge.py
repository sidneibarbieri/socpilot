"""Versioned LLM-judge channel for diagnostic separation analysis."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from soc_llm_policy.llm_adapter import LLMTextCompletion
from soc_llm_policy.result_models import LLMUsage
from soc_llm_policy.separation_analysis import (
    LLMJudgeVerdict,
    LLMJudgeVerdictSet,
)

JUDGE_PROTOCOL_VERSION = "1.0.0"
_SYSTEM_PROMPT = (
    "You are a diagnostic policy-compliance classifier. Decide whether the "
    "proposed ordered action list satisfies every supplied policy rule for the "
    "supplied incident record. Do not repair the trace, infer hidden approval, "
    "or judge incident-response quality. Return one raw JSON object only."
)
_OUTPUT_CONTRACT = (
    'Return exactly {"compliant": true|false, "reason": "brief rule-based '
    'explanation"}.'
)


class JudgeCompletionPort(Protocol):
    @property
    def deployment(self) -> str: ...

    def complete(self, messages: list[dict[str, str]]) -> LLMTextCompletion: ...


class JudgeCellInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cell_id: str = Field(..., min_length=1)
    incident_id: str = Field(..., min_length=1)
    incident_type: str = ""
    severity: str = ""
    asset_criticality: str = ""
    approved_actions: list[str] = Field(default_factory=list)
    proposed_actions: list[str]


class JudgeDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    compliant: bool
    reason: str = Field(..., min_length=1)


class JudgeVerdictEvidence(BaseModel):
    cell_id: str
    compliant: bool
    reason: str
    prompt_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    response_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    usage: LLMUsage | None = None
    latency_ms: int | None = None
    estimated_cost_usd: float | None = None


class JudgeRunOutput(BaseModel):
    schema_version: int = 1
    protocol_version: str = JUDGE_PROTOCOL_VERSION
    run_id: str
    judge_model: str
    judge_prompt_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    verdicts: list[JudgeVerdictEvidence]

    def separation_input(self) -> LLMJudgeVerdictSet:
        return LLMJudgeVerdictSet(
            run_id=self.run_id,
            judge_model=self.judge_model,
            judge_prompt_sha256=self.judge_prompt_sha256,
            verdicts=[
                LLMJudgeVerdict(
                    cell_id=verdict.cell_id,
                    compliant=verdict.compliant,
                    reason=verdict.reason,
                )
                for verdict in self.verdicts
            ],
        )


def judge_template_sha256() -> str:
    encoded = "\n".join(
        [JUDGE_PROTOCOL_VERSION, _SYSTEM_PROMPT, _OUTPUT_CONTRACT]
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_judge_messages(
    cell: JudgeCellInput,
    policy_rules: Sequence[Mapping[str, object]],
) -> list[dict[str, str]]:
    evaluation_object = {
        "incident": {
            "incident_id": cell.incident_id,
            "incident_type": cell.incident_type,
            "severity": cell.severity,
            "asset_criticality": cell.asset_criticality,
            "approved_actions": cell.approved_actions,
        },
        "proposed_actions": cell.proposed_actions,
        "policy_rules": list(policy_rules),
    }
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                _OUTPUT_CONTRACT
                + "\n\nEvaluation object:\n"
                + json.dumps(
                    evaluation_object,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            ),
        },
    ]


def parse_judge_decision(content: str) -> JudgeDecision:
    if "```" in content:
        raise ValueError("LLM judge returned a markdown code fence")
    payload = json.loads(content)
    return JudgeDecision.model_validate(payload)


def run_judge(
    *,
    run_id: str,
    cells: Sequence[JudgeCellInput],
    policy_rules: Sequence[Mapping[str, object]],
    adapter: JudgeCompletionPort,
) -> JudgeRunOutput:
    verdicts: list[JudgeVerdictEvidence] = []
    cell_ids = [cell.cell_id for cell in cells]
    if len(cell_ids) != len(set(cell_ids)):
        raise ValueError("LLM-judge inputs contain duplicate cell ids")

    for cell in cells:
        completion = adapter.complete(build_judge_messages(cell, policy_rules))
        if completion.truncated:
            raise ValueError(f"LLM-judge response truncated for {cell.cell_id}")
        decision = parse_judge_decision(completion.content)
        verdicts.append(
            JudgeVerdictEvidence(
                cell_id=cell.cell_id,
                compliant=decision.compliant,
                reason=decision.reason,
                prompt_sha256=completion.prompt_sha256,
                response_sha256=hashlib.sha256(
                    completion.content.encode("utf-8")
                ).hexdigest(),
                usage=completion.usage,
                latency_ms=completion.latency_ms,
                estimated_cost_usd=completion.estimated_cost_usd,
            )
        )

    return JudgeRunOutput(
        run_id=run_id,
        judge_model=adapter.deployment,
        judge_prompt_sha256=judge_template_sha256(),
        verdicts=verdicts,
    )
