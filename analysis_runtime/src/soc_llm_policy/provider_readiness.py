"""Gate a provider campaign on one bounded live request per declared model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from soc_llm_policy.dotenv_utils import load_project_dotenv
from soc_llm_policy.durable_io import atomic_write_json
from soc_llm_policy.evaluation_plan import RunLease
from soc_llm_policy.llm_adapter import LLMAdapter, LLMAdapterError, LLMConfig
from soc_llm_policy.models_registry import (
    ModelProfile,
    load_model_registry,
    model_environment,
    normalize_provider,
)
from soc_llm_policy.preflight_evidence import sha256_file
from soc_llm_policy.provider_cost import ProviderCostTracker
from soc_llm_policy.provider_plan import ProviderExecutionPlan, load_provider_plan
from soc_llm_policy.publication_redaction import expected_digest

READINESS_MAX_COMPLETION_TOKENS = 64
READINESS_PROMPT_TOKEN_BUDGET = 256
READINESS_STUDY_ID = "provider_readiness"
_READINESS_MESSAGES = [
    {
        "role": "system",
        "content": "This is a provider readiness check. Reply with OK only.",
    },
    {"role": "user", "content": "OK"},
]

type ProbeFn = Callable[[ModelProfile, Path, int, float], "ReadinessObservation"]


class ProviderReadinessError(RuntimeError):
    """The campaign cannot begin because a provider profile is not ready."""


class ReadinessObservation(BaseModel):
    model_name: str = Field(..., min_length=1)
    provider: str = Field(..., min_length=1)
    deployment: str = Field(..., min_length=1)
    status: Literal["passed"] = "passed"
    returned_model: str = Field(..., min_length=1)
    model_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    response_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    latency_ms: int | None = Field(default=None, ge=0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0)
    truncated: bool


class AttemptLogBinding(BaseModel):
    path: str = Field(..., min_length=1)
    sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")


class ProviderReadinessReport(BaseModel):
    schema_version: Literal[1] = 1
    state: Literal["provider_readiness_passed"] = "provider_readiness_passed"
    constitutes_model_evidence: Literal[False] = False
    campaign_id: str = Field(..., min_length=1)
    provider_plan_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    prompt_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    generated_at_utc: str = Field(..., min_length=1)
    provider_attempt_log: AttemptLogBinding
    results: list[ReadinessObservation] = Field(..., min_length=1)
    provider_request_count: int = Field(..., ge=1)
    provider_response_count: int = Field(..., ge=1)
    realized_cost_usd: float = Field(..., ge=0)


def readiness_prompt_sha256() -> str:
    encoded = json.dumps(
        _READINESS_MESSAGES,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@contextmanager
def _temporary_environment(values: dict[str, str]) -> Iterator[None]:
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, old_value in previous.items():
            if old_value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old_value


def _resolve_repo_file(repo_root: Path, relative_path: str, label: str) -> Path:
    candidate = Path(relative_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"{label} path escapes the repository.")
    path = (repo_root / candidate).resolve()
    if repo_root != path and repo_root not in path.parents:
        raise ValueError(f"{label} path escapes the repository.")
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    return path


def _require_repo_path(repo_root: Path, path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if resolved != repo_root and repo_root not in resolved.parents:
        raise ValueError(f"{label} must remain inside the repository.")
    return resolved


def _load_readiness_profiles(
    repo_root: Path,
    plan: ProviderExecutionPlan,
) -> list[ModelProfile]:
    binding = plan.frozen_inputs.get("models_registry")
    if binding is None:
        raise ValueError("Provider plan does not bind its model registry.")
    registry_path = _resolve_repo_file(repo_root, binding.path, "Model registry")
    if sha256_file(registry_path) != expected_digest(
            repo_root, registry_path, binding.sha256
        ):
        raise ValueError("Model registry changed after the provider plan was frozen.")
    profiles = load_model_registry(registry_path)
    profiles_by_name = {profile.name: profile for profile in profiles}
    missing = [
        model_name
        for model_name in plan.provider_readiness.model_names
        if model_name not in profiles_by_name
    ]
    if missing:
        raise ValueError(f"Provider-readiness models are missing: {missing}")
    return [profiles_by_name[name] for name in plan.provider_readiness.model_names]


def _read_retry_wait(repo_root: Path, plan: ProviderExecutionPlan) -> float:
    binding = plan.frozen_inputs.get("protocol_freeze")
    if binding is None:
        raise ValueError("Provider plan does not bind its protocol freeze.")
    freeze_path = _resolve_repo_file(repo_root, binding.path, "Protocol freeze")
    if sha256_file(freeze_path) != expected_digest(
            repo_root, freeze_path, binding.sha256
        ):
        raise ValueError("Protocol freeze changed after the provider plan was created.")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    retry_policy = freeze.get("retry_policy")
    if not isinstance(retry_policy, dict):
        raise ValueError("Protocol freeze has no retry policy.")
    attempt_limit = retry_policy.get("llm_attempt_limit")
    if attempt_limit != plan.provider_readiness.provider_attempt_limit:
        raise ValueError("Provider-readiness attempt limit differs from the freeze.")
    retry_wait = retry_policy.get("llm_retry_wait_seconds")
    if not isinstance(retry_wait, (int, float)) or retry_wait < 0:
        raise ValueError("Protocol freeze has an invalid retry wait.")
    return float(retry_wait)


def _verify_local_model_identity(profile: ModelProfile) -> str | None:
    if normalize_provider(profile.provider) != "ollama":
        return None
    expected_digest = profile.model_digest.strip()
    if not expected_digest:
        raise ProviderReadinessError(
            f"Local model {profile.name} has no content digest."
        )
    endpoint = os.environ.get(
        "OLLAMA_API_BASE", "http://127.0.0.1:11434"
    ).rstrip("/")
    with urllib.request.urlopen(f"{endpoint}/api/tags", timeout=10) as response:
        body = json.loads(response.read().decode("utf-8"))
    models = body.get("models") if isinstance(body, dict) else None
    if not isinstance(models, list):
        raise ProviderReadinessError("Ollama returned an invalid model inventory.")
    matching_model = next(
        (
            item
            for item in models
            if isinstance(item, dict)
            and item.get("name") in {profile.deployment, profile.model}
        ),
        None,
    )
    if matching_model is None:
        raise ProviderReadinessError(
            f"Local model {profile.name} is not installed in Ollama."
        )
    observed_digest = matching_model.get("digest")
    if observed_digest != expected_digest:
        raise ProviderReadinessError(
            f"Local model {profile.name} digest differs from the registry."
        )
    return expected_digest


def _probe_model(
    profile: ModelProfile,
    attempt_log: Path,
    attempt_limit: int,
    retry_wait_seconds: float,
) -> ReadinessObservation:
    model_digest = _verify_local_model_identity(profile)
    environment = {
        **model_environment(profile),
        "SOC_LLM_ATTEMPT_LIMIT": str(attempt_limit),
        "SOC_LLM_RETRY_WAIT_SECONDS": str(retry_wait_seconds),
        "SOC_EXPERIMENT_RUN_ID": READINESS_STUDY_ID,
        "SOC_EXPERIMENT_CELL_ID": f"readiness::{profile.name}",
        "SOC_EXPERIMENT_EXECUTION_ID": f"readiness::{profile.name}",
        "SOC_EXPERIMENT_ATTEMPT_LOG": str(attempt_log),
    }
    with _temporary_environment(environment):
        config = replace(
            LLMConfig.from_env(),
            max_tokens=READINESS_MAX_COMPLETION_TOKENS,
            attempt_limit=attempt_limit,
            retry_wait_seconds=retry_wait_seconds,
        )
        completion = LLMAdapter(config).complete(_READINESS_MESSAGES)
    if completion.truncated:
        raise ProviderReadinessError(
            f"Model {profile.name} truncated the readiness response."
        )
    if not completion.content.strip():
        raise ProviderReadinessError(
            f"Model {profile.name} returned an empty readiness response."
        )
    returned_model = completion.returned_model
    expected_models = {profile.deployment, profile.model}
    expected_models.discard("")
    if returned_model not in expected_models:
        raise ProviderReadinessError(
            f"Model {profile.name} returned unexpected identity {returned_model!r}."
        )
    usage = completion.usage
    return ReadinessObservation(
        model_name=profile.name,
        provider=normalize_provider(profile.provider),
        deployment=profile.deployment,
        returned_model=returned_model,
        model_digest=model_digest,
        response_sha256=hashlib.sha256(completion.content.encode("utf-8")).hexdigest(),
        latency_ms=completion.latency_ms,
        prompt_tokens=(usage.prompt_tokens if usage is not None else None),
        completion_tokens=(usage.completion_tokens if usage is not None else None),
        estimated_cost_usd=completion.estimated_cost_usd,
        truncated=completion.truncated,
    )


def _validate_observation(
    observation: ReadinessObservation,
    profile: ModelProfile,
) -> None:
    expected_models = {profile.deployment, profile.model}
    expected_models.discard("")
    if (
        observation.model_name != profile.name
        or observation.provider != normalize_provider(profile.provider)
        or observation.deployment != profile.deployment
        or observation.returned_model not in expected_models
        or observation.model_digest
        != (profile.model_digest if profile.billing_mode == "local" else None)
    ):
        raise ProviderReadinessError(
            f"Provider-readiness observation differs from profile {profile.name}."
        )


def _attempt_accounting(path: Path) -> tuple[int, int, float]:
    tracker = ProviderCostTracker(path)
    tracker.refresh()
    if tracker.unknown_cost_count:
        raise ValueError("Readiness has incomplete provider-cost accounting.")
    return (
        tracker.request_count,
        tracker.response_count,
        round(float(tracker.accrued_usd), 6),
    )


def _write_failed_report(
    *,
    output_json: Path,
    plan: ProviderExecutionPlan,
    plan_path: Path,
    completed: list[ReadinessObservation],
    failed_profile: ModelProfile,
    error: Exception,
) -> None:
    payload = {
        "schema_version": 1,
        "state": "provider_readiness_failed",
        "constitutes_model_evidence": False,
        "campaign_id": plan.plan_id,
        "provider_plan_sha256": sha256_file(plan_path),
        "prompt_sha256": readiness_prompt_sha256(),
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "completed_results": [item.model_dump(mode="json") for item in completed],
        "failed_model": failed_profile.name,
        "exception_type": type(error).__name__,
        "error": str(error),
    }
    atomic_write_json(output_json, payload)


def execute_provider_readiness(
    *,
    repo_root: Path,
    plan_path: Path,
    output_json: Path,
    attempt_log: Path,
    probe: ProbeFn = _probe_model,
) -> ProviderReadinessReport:
    repo_root = repo_root.expanduser().resolve()
    output_json = _require_repo_path(repo_root, output_json, "Readiness report")
    with RunLease(
        output_json.parent / "provider_readiness_lease.json",
        "provider_readiness",
        lease_seconds=900,
        allow_stale_takeover=True,
    ):
        return _execute_provider_readiness_locked(
            repo_root=repo_root,
            plan_path=plan_path,
            output_json=output_json,
            attempt_log=attempt_log,
            probe=probe,
        )


def _execute_provider_readiness_locked(
    *,
    repo_root: Path,
    plan_path: Path,
    output_json: Path,
    attempt_log: Path,
    probe: ProbeFn,
) -> ProviderReadinessReport:
    repo_root = repo_root.expanduser().resolve()
    plan_path = _require_repo_path(repo_root, plan_path, "Provider plan")
    output_json = _require_repo_path(repo_root, output_json, "Readiness report")
    attempt_log = _require_repo_path(repo_root, attempt_log, "Readiness attempt log")
    if output_json.exists():
        return load_provider_readiness(
            repo_root=repo_root,
            plan_path=plan_path,
            report_path=output_json,
        )
    if attempt_log.exists():
        raise ProviderReadinessError(
            "Provider-readiness attempts exist without a terminal report. "
            "Do not risk duplicate operational requests; start from a clean source package."
        )
    plan = load_provider_plan(plan_path)
    if plan.approval.status != "approved":
        raise ValueError("Provider readiness requires an approved campaign plan.")
    if plan.provider_readiness.prompt_sha256 != readiness_prompt_sha256():
        raise ValueError("Provider-readiness prompt differs from the approved plan.")
    if plan.provider_readiness.max_completion_tokens != READINESS_MAX_COMPLETION_TOKENS:
        raise ValueError("Provider-readiness token cap differs from the implementation.")
    profiles = _load_readiness_profiles(repo_root, plan)
    retry_wait_seconds = _read_retry_wait(repo_root, plan)
    load_project_dotenv(repo_root)
    budget = plan.cost_estimate.study_budget_ceiling_usd.get(READINESS_STUDY_ID)
    if not isinstance(budget, (int, float)) or budget <= 0:
        raise ValueError("Provider plan has no readiness budget ceiling.")

    output_json.parent.mkdir(parents=True, exist_ok=True)
    attempt_log.parent.mkdir(parents=True, exist_ok=True)
    tracker = ProviderCostTracker(attempt_log)
    completed: list[ReadinessObservation] = []
    for profile in profiles:
        budget_profile = profile.model_copy(
            update={
                "planning_prompt_tokens": READINESS_PROMPT_TOKEN_BUDGET,
                "planning_completion_tokens": READINESS_MAX_COMPLETION_TOKENS,
            }
        )
        tracker.require_budget_for_next_request(
            budget_profile,
            float(budget),
            max_completion_tokens=READINESS_MAX_COMPLETION_TOKENS,
        )
        try:
            with _temporary_environment(
                {"SOC_EXPERIMENT_APPROVED_BUDGET_USD": str(budget)}
            ):
                observation = probe(
                    budget_profile,
                    attempt_log,
                    plan.provider_readiness.provider_attempt_limit,
                    retry_wait_seconds,
                )
            _validate_observation(observation, profile)
            completed.append(observation)
        except (LLMAdapterError, ProviderReadinessError) as error:
            _write_failed_report(
                output_json=output_json,
                plan=plan,
                plan_path=plan_path,
                completed=completed,
                failed_profile=profile,
                error=error,
            )
            raise ProviderReadinessError(
                f"Provider readiness failed for model {profile.name}."
            ) from error

    request_count, response_count, realized_cost = _attempt_accounting(attempt_log)
    if response_count != len(completed):
        raise ProviderReadinessError(
            "Provider-readiness responses do not cover every declared model."
        )
    if request_count > plan.provider_readiness.provider_attempt_ceiling:
        raise ProviderReadinessError(
            "Provider-readiness requests exceed the approved attempt ceiling."
        )
    report = ProviderReadinessReport(
        campaign_id=plan.plan_id,
        provider_plan_sha256=sha256_file(plan_path),
        prompt_sha256=readiness_prompt_sha256(),
        generated_at_utc=datetime.now(UTC).isoformat(),
        provider_attempt_log=AttemptLogBinding(
            path=attempt_log.relative_to(repo_root).as_posix(),
            sha256=sha256_file(attempt_log),
        ),
        results=completed,
        provider_request_count=request_count,
        provider_response_count=response_count,
        realized_cost_usd=realized_cost,
    )
    atomic_write_json(output_json, report.model_dump(mode="json"))
    return report


def load_provider_readiness(
    *,
    repo_root: Path,
    plan_path: Path,
    report_path: Path,
) -> ProviderReadinessReport:
    repo_root = repo_root.expanduser().resolve()
    plan_path = _require_repo_path(repo_root, plan_path, "Provider plan")
    report_path = _require_repo_path(repo_root, report_path, "Readiness report")
    report = ProviderReadinessReport.model_validate_json(
        report_path.read_text(encoding="utf-8")
    )
    plan = load_provider_plan(plan_path)
    if report.campaign_id != plan.plan_id:
        raise ValueError("Provider-readiness report belongs to another campaign.")
    if sha256_file(plan_path) != expected_digest(
        repo_root, plan_path, report.provider_plan_sha256
    ):
        raise ValueError("Provider plan changed after the readiness gate passed.")
    if report.prompt_sha256 != plan.provider_readiness.prompt_sha256:
        raise ValueError("Provider-readiness prompt differs from the approved plan.")
    profiles = _load_readiness_profiles(repo_root, plan)
    if [item.model_name for item in report.results] != plan.provider_readiness.model_names:
        raise ValueError("Provider readiness does not cover the declared model set.")
    for observation, profile in zip(report.results, profiles, strict=True):
        _validate_observation(observation, profile)
    attempt_log = _resolve_repo_file(
        repo_root,
        report.provider_attempt_log.path,
        "Provider-readiness attempt log",
    )
    if sha256_file(attempt_log) != expected_digest(
            repo_root, attempt_log, report.provider_attempt_log.sha256
        ):
        raise ValueError("Provider-readiness attempt log changed after the gate passed.")
    request_count, response_count, realized_cost = _attempt_accounting(attempt_log)
    if (
        request_count != report.provider_request_count
        or response_count != report.provider_response_count
        or realized_cost != report.realized_cost_usd
    ):
        raise ValueError("Provider-readiness accounting differs from its attempt log.")
    if response_count != plan.provider_readiness.request_count:
        raise ValueError("Provider readiness has the wrong response count.")
    if request_count > plan.provider_readiness.provider_attempt_ceiling:
        raise ValueError("Provider readiness exceeded its attempt ceiling.")
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    subparsers = parser.add_subparsers(dest="command", required=True)
    execute = subparsers.add_parser("execute")
    execute.add_argument("--plan", type=Path, required=True)
    execute.add_argument("--output-json", type=Path, required=True)
    execute.add_argument("--attempt-log", type=Path, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--plan", type=Path, required=True)
    validate.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "execute":
        report = execute_provider_readiness(
            repo_root=args.repo_root,
            plan_path=args.plan,
            output_json=args.output_json,
            attempt_log=args.attempt_log,
        )
    else:
        report = load_provider_readiness(
            repo_root=args.repo_root,
            plan_path=args.plan,
            report_path=args.report,
        )
    print(
        json.dumps(
            {
                "campaign_id": report.campaign_id,
                "state": report.state,
                "provider_request_count": report.provider_request_count,
                "realized_cost_usd": report.realized_cost_usd,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
