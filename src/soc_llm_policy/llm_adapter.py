"""
LLM adapter for multiple providers.

Supported providers:
  - azure_openai
  - openai
  - anthropic
  - ollama

Single responsibility: receive incident metadata + telemetry, build the prompt,
call the provider API, and return a list of action_ids.

It does not handle filesystem operations, policy enforcement, or engine logic.
That orchestration belongs to pipeline.py.

Usage:
    from soc_llm_policy.llm_adapter import LLMAdapter

    adapter = LLMAdapter.from_env()
    rec = adapter.recommend(meta, telemetry, catalog)
    # rec.actions -> ["isolate_host", "collect_forensics", "reset_admin_credentials"]
"""

from __future__ import annotations

import hashlib
import http.client
import json
import math
import os
import re
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from soc_llm_policy.connectivity_recovery import (
    ConnectivityRecoveryConfig,
    ConnectivityRecoveryTimeout,
    wait_for_stable_connectivity,
)
from soc_llm_policy.interfaces import LLMRecommendation
from soc_llm_policy.io import (
    ActionCatalogItem,
    IncidentMeta,
    PolicyRule,
    TelemetryEvent,
)
from soc_llm_policy.provider_cost import provider_cost_tracker
from soc_llm_policy.result_models import LLMUsage

type ProviderName = Literal["azure_openai", "openai", "anthropic", "ollama"]
type BillingMode = Literal["metered", "local"]


class LLMAdapterError(Exception):
    """Generic error raised by the LLM adapter."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.usage: LLMUsage | None = None
        self.returned_model: str | None = None


class LLMTransportError(LLMAdapterError):
    """A provider request failed without producing a model observation."""


class LLMProviderAccessError(LLMTransportError):
    """Provider authentication or authorization failed before an observation."""


class LLMRefusalError(LLMAdapterError):
    """The provider explicitly reported that the model declined to answer.

    A refusal is a permanent, recordable outcome under the attempt policy:
    it is never retried, and the ledger's exception type distinguishes it
    from transport or format failures so it gets its own denominator.
    """


class LLMResponseParseError(LLMAdapterError):
    """The LLM replied, but valid action_ids could not be extracted."""


DEFAULT_MAX_COMPLETION_TOKENS = 4000
_PROVIDER_ERROR_TEXT_LIMIT = 320
_PROMPT_RESERVATION_OVERHEAD_TOKENS = 512
_SECRET_LIKE_PATTERN = re.compile(
    r"(?i)(?:bearer\s+)?(?:sk|key|token)[-_]?[a-z0-9]{12,}"
)


def _positive_float_env(name: str, default: str) -> float:
    raw_value = os.environ.get(name, default).strip()
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise LLMAdapterError(f"{name} must be numeric.") from exc
    if not math.isfinite(value) or value <= 0:
        raise LLMAdapterError(f"{name} must be finite and positive.")
    return value


def _positive_int_env(name: str, default: str) -> int:
    raw_value = os.environ.get(name, default).strip()
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise LLMAdapterError(f"{name} must be an integer.") from exc
    if value < 1:
        raise LLMAdapterError(f"{name} must be positive.")
    return value


def _connectivity_recovery_from_env() -> ConnectivityRecoveryConfig | None:
    if not _read_bool_env("SOC_LLM_CONNECTIVITY_RECOVERY_ENABLED", default=False):
        return None
    log_value = os.environ.get("SOC_EXPERIMENT_CONNECTIVITY_LOG", "").strip()
    if not log_value:
        raise LLMAdapterError("Connectivity recovery requires an append-only log.")
    return ConnectivityRecoveryConfig(
        probe_timeout_seconds=_positive_float_env(
            "SOC_LLM_CONNECTIVITY_PROBE_TIMEOUT_SECONDS", "5"
        ),
        probe_interval_seconds=_positive_float_env(
            "SOC_LLM_CONNECTIVITY_PROBE_INTERVAL_SECONDS", "10"
        ),
        recovery_timeout_seconds=_positive_float_env(
            "SOC_LLM_CONNECTIVITY_RECOVERY_TIMEOUT_SECONDS", "900"
        ),
        required_consecutive_successes=_positive_int_env(
            "SOC_LLM_CONNECTIVITY_REQUIRED_SUCCESSES", "3"
        ),
        log_path=Path(log_value) if log_value else None,
    )


def _text_identity(value: str) -> str:
    encoded = value.encode("utf-8")
    return f"sha256={hashlib.sha256(encoded).hexdigest()}, bytes={len(encoded)}"


def _json_identity(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256={hashlib.sha256(encoded).hexdigest()}, bytes={len(encoded)}"


def _record_provider_attempt(record: dict[str, object]) -> None:
    """Append secret-free transport metadata for the active experiment cell."""
    path_value = os.environ.get("SOC_EXPERIMENT_ATTEMPT_LOG", "").strip()
    if not path_value:
        return
    payload = {
        "schema_version": 1,
        "cell_id": os.environ.get("SOC_EXPERIMENT_CELL_ID"),
        "execution_id": os.environ.get("SOC_EXPERIMENT_EXECUTION_ID"),
        **record,
    }
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, sort_keys=True) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        remaining = memoryview(encoded)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("Provider-attempt ledger append made no progress.")
            remaining = remaining[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _provider_quota_exhausted(raw_body: str) -> bool:
    """Recognize structured account exhaustion, not transient rate limiting."""
    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError:
        return False
    error = payload.get("error") if isinstance(payload, dict) else None
    if not isinstance(error, dict):
        return False
    exhausted_codes = {"insufficient_quota", "credit_balance_exhausted"}
    return any(
        isinstance(error.get(field), str) and error[field] in exhausted_codes
        for field in ("type", "code")
    )


def _provider_error_summary(raw_body: str, api_key: str) -> str:
    """Return bounded structured diagnostics without retaining response bodies."""
    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError:
        return ""
    if not isinstance(payload, dict):
        return ""
    error = payload.get("error")
    if isinstance(error, str):
        values = {"message": error}
    elif isinstance(error, dict):
        values = {
            field: error.get(field)
            for field in ("type", "code", "message")
            if error.get(field) not in (None, "")
        }
    else:
        return ""
    parts: list[str] = []
    for field, value in values.items():
        text = " ".join(str(value).split())
        if api_key:
            text = text.replace(api_key, "[redacted]")
        text = _SECRET_LIKE_PATTERN.sub("[redacted]", text)
        parts.append(f"{field}={text[:_PROVIDER_ERROR_TEXT_LIMIT]}")
    return ", ".join(parts)


def _read_bool_env(name: str, *, default: bool) -> bool:
    """Read a strict boolean environment flag."""
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    normalized_value = raw_value.strip().lower()
    if normalized_value in {"1", "true", "yes", "on"}:
        return True
    if normalized_value in {"0", "false", "no", "off"}:
        return False
    raise LLMAdapterError(
        f"{name} must be one of: 1, true, yes, on, 0, false, no, off."
    )


def _provider_and_billing_mode() -> tuple[ProviderName, BillingMode]:
    provider = os.environ.get("SOC_LLM_PROVIDER", "azure_openai").strip().lower()
    if provider not in {"azure_openai", "openai", "anthropic", "ollama"}:
        raise LLMAdapterError(
            "SOC_LLM_PROVIDER must be one of: azure_openai, openai, "
            "anthropic, ollama."
        )
    billing_mode = os.environ.get(
        "SOC_LLM_BILLING_MODE",
        "local" if provider == "ollama" else "metered",
    ).strip().lower()
    if billing_mode not in {"metered", "local"}:
        raise LLMAdapterError(
            "SOC_LLM_BILLING_MODE must be one of: metered, local."
        )
    if (provider == "ollama") != (billing_mode == "local"):
        raise LLMAdapterError(
            "The Ollama provider requires local billing mode, and local billing "
            "mode requires Ollama."
        )
    return cast(ProviderName, provider), cast(BillingMode, billing_mode)


@dataclass(frozen=True)
class _ProviderConnection:
    endpoint: str
    deployment: str
    api_key: str = ""
    api_version: str = ""
    missing: tuple[str, ...] = ()


def _provider_connection(provider: ProviderName) -> _ProviderConnection:
    if provider == "azure_openai":
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()
        deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "").strip()
        api_key = os.environ.get("AZURE_OPENAI_API_KEY", "").strip()
        api_version = os.environ.get(
            "AZURE_OPENAI_API_VERSION", "2024-02-15-preview"
        ).strip()
        values = {
            "AZURE_OPENAI_ENDPOINT": endpoint,
            "AZURE_OPENAI_DEPLOYMENT": deployment,
            "AZURE_OPENAI_API_KEY": api_key,
        }
    elif provider == "openai":
        endpoint = os.environ.get(
            "OPENAI_API_BASE", "https://api.openai.com/v1"
        ).strip()
        deployment = os.environ.get("OPENAI_MODEL", "").strip()
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        api_version = ""
        values = {"OPENAI_MODEL": deployment, "OPENAI_API_KEY": api_key}
    elif provider == "anthropic":
        endpoint = os.environ.get(
            "ANTHROPIC_API_BASE", "https://api.anthropic.com/v1"
        ).strip()
        deployment = os.environ.get("ANTHROPIC_MODEL", "").strip()
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        api_version = os.environ.get(
            "ANTHROPIC_API_VERSION", "2023-06-01"
        ).strip()
        values = {"ANTHROPIC_MODEL": deployment, "ANTHROPIC_API_KEY": api_key}
    else:
        endpoint = os.environ.get(
            "OLLAMA_API_BASE", "http://127.0.0.1:11434"
        ).strip()
        deployment = os.environ.get("OLLAMA_MODEL", "").strip()
        api_key = ""
        api_version = ""
        values = {"OLLAMA_MODEL": deployment}
    return _ProviderConnection(
        endpoint=endpoint,
        deployment=deployment,
        api_key=api_key,
        api_version=api_version,
        missing=tuple(name for name, value in values.items() if not value),
    )


@dataclass(frozen=True)
class LLMConfig:
    """Connection and generation settings for the provider adapter."""

    endpoint: str
    deployment: str
    api_key: str
    provider: ProviderName = "azure_openai"
    billing_mode: BillingMode = "metered"
    api_version: str = "2024-02-15-preview"
    azure_token_parameter: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"
    # Reasoning and visible output share this ceiling; truncation is recorded.
    max_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS
    # None omits the field so the provider default applies. GPT-5.5 and
    # Claude Opus 4.8 rejects non-default sampling settings.
    temperature: float | None = None
    # Maximum number of telemetry events sent to the LLM.
    # Large datasets (1000+ events) can exceed token limits.
    # Truncation preserves the most relevant events (see _truncate).
    max_telemetry_events: int = 40
    # Total provider attempts, including the initial request.
    attempt_limit: int = 3
    retry_wait_seconds: float = 65.0
    connectivity_recovery: ConnectivityRecoveryConfig | None = None
    # Official runs require a raw JSON object, not markdown-fenced JSON.
    allow_markdown_json_fence: bool = False
    # Pricing used to estimate cost per request (USD per 1k tokens).
    prompt_price_per_1k_usd: float = 0.0
    completion_price_per_1k_usd: float = 0.0
    planning_prompt_tokens: int | None = None
    approved_budget_usd: float | None = None
    attempt_log_path: Path | None = None
    thinking: bool | None = None
    response_format: Literal["text", "json"] = "text"

    @classmethod
    def from_env(cls) -> LLMConfig:
        """
        Load credentials from environment variables.

        Common variable:
            SOC_LLM_PROVIDER         one of: azure_openai, openai, anthropic, ollama
            SOC_LLM_BILLING_MODE     one of: metered, local

        Provider-specific variables:
            azure_openai:
              AZURE_OPENAI_ENDPOINT
              AZURE_OPENAI_DEPLOYMENT
              AZURE_OPENAI_API_KEY
              AZURE_OPENAI_API_VERSION (optional)
            openai:
              OPENAI_API_KEY
              OPENAI_MODEL
              OPENAI_API_BASE (optional, default https://api.openai.com/v1)
            anthropic:
              ANTHROPIC_API_KEY
              ANTHROPIC_MODEL
              ANTHROPIC_API_BASE (optional, default https://api.anthropic.com/v1)
              ANTHROPIC_API_VERSION (optional, default 2023-06-01)
            ollama:
              OLLAMA_MODEL
              OLLAMA_API_BASE (optional, default http://127.0.0.1:11434)
              OLLAMA_THINK (optional)
              OLLAMA_RESPONSE_FORMAT (optional, text or json)
        """
        provider, billing_mode = _provider_and_billing_mode()
        response_format = os.environ.get(
            "OLLAMA_RESPONSE_FORMAT", "text"
        ).strip().lower()
        if response_format not in {"text", "json"}:
            raise LLMAdapterError(
                "OLLAMA_RESPONSE_FORMAT must be one of: text, json."
            )
        if provider != "ollama" and response_format != "text":
            raise LLMAdapterError(
                "OLLAMA_RESPONSE_FORMAT applies only to the Ollama provider."
            )

        attempt_limit_raw = os.environ.get("SOC_LLM_ATTEMPT_LIMIT", "3").strip()
        retry_wait_raw = os.environ.get("SOC_LLM_RETRY_WAIT_SECONDS", "65").strip()
        connectivity = _connectivity_recovery_from_env()
        try:
            prompt_price = float(
                os.environ.get("SOC_LLM_PROMPT_PRICE_PER_1K_USD", "0")
            )
            completion_price = float(
                os.environ.get("SOC_LLM_COMPLETION_PRICE_PER_1K_USD", "0")
            )
        except ValueError as exc:
            raise LLMAdapterError("Provider prices must be numeric.") from exc
        if not math.isfinite(prompt_price) or not math.isfinite(completion_price):
            raise LLMAdapterError("Provider prices must be finite.")
        planning_prompt_raw = os.environ.get(
            "SOC_LLM_PLANNING_PROMPT_TOKENS", ""
        ).strip()
        approved_budget_raw = os.environ.get(
            "SOC_EXPERIMENT_APPROVED_BUDGET_USD", ""
        ).strip()
        attempt_log_raw = os.environ.get("SOC_EXPERIMENT_ATTEMPT_LOG", "").strip()
        try:
            attempt_limit = int(attempt_limit_raw)
        except ValueError as exc:
            raise LLMAdapterError("SOC_LLM_ATTEMPT_LIMIT must be an integer.") from exc
        try:
            retry_wait_seconds = float(retry_wait_raw)
        except ValueError as exc:
            raise LLMAdapterError(
                "SOC_LLM_RETRY_WAIT_SECONDS must be a finite number."
            ) from exc
        if not math.isfinite(retry_wait_seconds):
            raise LLMAdapterError("SOC_LLM_RETRY_WAIT_SECONDS must be a finite number.")
        if attempt_limit < 1 or retry_wait_seconds < 0:
            raise LLMAdapterError(
                "SOC_LLM_ATTEMPT_LIMIT must be >= 1 and "
                "SOC_LLM_RETRY_WAIT_SECONDS >= 0."
            )
        try:
            planning_prompt_tokens = (
                int(planning_prompt_raw) if planning_prompt_raw else None
            )
        except ValueError as exc:
            raise LLMAdapterError(
                "SOC_LLM_PLANNING_PROMPT_TOKENS must be an integer."
            ) from exc
        try:
            approved_budget_usd = (
                float(approved_budget_raw) if approved_budget_raw else None
            )
        except ValueError as exc:
            raise LLMAdapterError(
                "SOC_EXPERIMENT_APPROVED_BUDGET_USD must be numeric."
            ) from exc
        if planning_prompt_tokens is not None and planning_prompt_tokens < 1:
            raise LLMAdapterError(
                "SOC_LLM_PLANNING_PROMPT_TOKENS must be positive."
            )
        if billing_mode == "local":
            if prompt_price != 0 or completion_price != 0:
                raise LLMAdapterError("Local model prices must be zero.")
            approved_budget_usd = None
        elif approved_budget_usd is not None:
            if not math.isfinite(approved_budget_usd) or approved_budget_usd <= 0:
                raise LLMAdapterError(
                    "SOC_EXPERIMENT_APPROVED_BUDGET_USD must be finite and positive."
                )
            if planning_prompt_tokens is None or not attempt_log_raw:
                raise LLMAdapterError(
                    "A provider budget requires SOC_LLM_PLANNING_PROMPT_TOKENS "
                    "and SOC_EXPERIMENT_ATTEMPT_LOG."
                )
            if prompt_price <= 0 or completion_price <= 0:
                raise LLMAdapterError(
                    "A provider budget requires positive frozen provider prices."
                )

        azure_token_parameter = os.environ.get(
            "AZURE_OPENAI_TOKEN_PARAMETER", "max_tokens"
        ).strip()
        if azure_token_parameter not in {"max_tokens", "max_completion_tokens"}:
            raise LLMAdapterError("Invalid AZURE_OPENAI_TOKEN_PARAMETER.")

        connection = _provider_connection(provider)
        if connection.missing:
            provider_hint = f" (provider={provider})"
            raise LLMAdapterError(
                "Missing environment variables"
                f"{provider_hint}: {', '.join(connection.missing)}\n"
                "Set them in the repository-root .env or export them in "
                "your shell before running."
            )

        return cls(
            provider=provider,
            billing_mode=billing_mode,
            endpoint=connection.endpoint,
            deployment=connection.deployment,
            api_key=connection.api_key,
            api_version=connection.api_version,
            azure_token_parameter=cast(
                Literal["max_tokens", "max_completion_tokens"], azure_token_parameter
            ),
            attempt_limit=attempt_limit,
            retry_wait_seconds=retry_wait_seconds,
            connectivity_recovery=connectivity,
            allow_markdown_json_fence=_read_bool_env(
                "SOC_LLM_ALLOW_MARKDOWN_JSON_FENCE",
                default=False,
            ),
            prompt_price_per_1k_usd=prompt_price,
            completion_price_per_1k_usd=completion_price,
            planning_prompt_tokens=planning_prompt_tokens,
            approved_budget_usd=approved_budget_usd,
            attempt_log_path=(Path(attempt_log_raw) if attempt_log_raw else None),
            thinking=(
                _read_bool_env("OLLAMA_THINK", default=False)
                if provider == "ollama"
                else None
            ),
            response_format=cast(Literal["text", "json"], response_format),
        )


_HTTP_RATE_LIMIT = 429
_HTTP_SERVER_ERROR_MIN = 500
_HTTP_SERVER_ERROR_MAX_EXCLUSIVE = 600
_REQUEST_TIMEOUT_SECONDS = 120


@dataclass(frozen=True)
class _LLMAPIResponse:
    content: str
    usage: LLMUsage | None
    latency_ms: int | None
    output_truncated: bool = False
    returned_model: str | None = None


@dataclass(frozen=True)
class LLMTextCompletion:
    """Auditable text completion for non-recommendation evaluation tasks."""

    content: str
    deployment: str
    usage: LLMUsage | None
    latency_ms: int | None
    estimated_cost_usd: float | None
    prompt_sha256: str
    truncated: bool
    returned_model: str | None = None


# Adapter


class LLMAdapter:
    """Provider-agnostic LLM adapter.

    Receives an LLMConfig with all connection and generation settings.
    """

    def __init__(self, config: LLMConfig) -> None:
        self._config = config

    @classmethod
    def from_env(cls) -> LLMAdapter:
        """Build adapter from environment variables via LLMConfig."""
        return cls(LLMConfig.from_env())

    @property
    def deployment(self) -> str:
        """Model or deployment identifier used for traceability."""
        return self._config.deployment

    def _truncate_telemetry(
        self,
        telemetry: list[TelemetryEvent],
    ) -> tuple[list[TelemetryEvent], bool]:
        """
        Limit telemetry to max_telemetry_events.

        Strategy: prioritize events with relevant command signals (bash, tcp,
        passwd, privilege), then fill with remaining events up to the limit.
        This keeps critical context even for large datasets.

        Returns (selected_events, was_truncated).
        """
        limit = self._config.max_telemetry_events
        if len(telemetry) <= limit:
            return telemetry, False

        priority_keywords = [
            "bash",
            "/dev/tcp",
            "passwd",
            "shadow",
            "credential",
            "escalat",
            "lateral",
            "exfil",
            "ransom",
            "malware",
            "c2",
            "beacon",
            "inject",
            "persistence",
        ]

        priority_indices: list[int] = []
        rest_indices: list[int] = []

        for index, event in enumerate(telemetry):
            command_text = (event.details.command or "").lower()
            event_type_text = event.event_type.lower()
            searchable_text = command_text + " " + event_type_text
            if any(keyword in searchable_text for keyword in priority_keywords):
                priority_indices.append(index)
            else:
                rest_indices.append(index)

        # Choose which events to keep by priority, then emit them in the original
        # chronological order so the model still sees a time-ordered sequence.
        kept = set((priority_indices + rest_indices)[:limit])
        selected = [event for index, event in enumerate(telemetry) if index in kept]
        return selected, True

    def _format_catalog(self, catalog: list[ActionCatalogItem]) -> str:
        """Render the bounded action interface used by the copilot."""
        lines: list[str] = []
        for item in catalog:
            attributes = [
                f"approval={'yes' if item.requires_approval else 'no'}",
                f"reversible={'yes' if item.reversible else 'no'}",
            ]
            if item.phase:
                attributes.append(f"phase={item.phase}")
            if item.operational_role:
                attributes.append(f"role={item.operational_role}")
            if item.baseline_support:
                attributes.append(f"baseline_support={item.baseline_support}")
            lines.append(f"  - {item.action_id} ({', '.join(attributes)})")
        return "\n".join(lines)

    def _format_telemetry(
        self,
        telemetry: list[TelemetryEvent],
        total_events: int,
    ) -> tuple[str, str]:
        """Render telemetry and the truncation note for the prompt."""
        telemetry_lines: list[str] = []
        for event in telemetry:
            event_type = event.event_type
            command_text = event.details.command or ""
            timestamp = (event.timestamp or "")[:19]
            line = f"  [{timestamp}] {event_type}"
            if command_text:
                line += f" -> {command_text[:80]}"
            telemetry_lines.append(line)

        truncation_note = ""
        if total_events > len(telemetry):
            truncation_note = (
                f"\n(Sample: {len(telemetry)} of {total_events} total events,"
                " prioritizing the most relevant SOC context)"
            )

        return "\n".join(telemetry_lines), truncation_note

    def _build_prompt(
        self,
        meta: IncidentMeta,
        telemetry: list[TelemetryEvent],
        catalog: list[ActionCatalogItem],
        total_events: int,
        *,
        policy_constraints_block: str = "",
    ) -> list[dict[str, str]]:
        """Build the chat messages sent to the provider API."""
        catalog_text = self._format_catalog(catalog)
        telemetry_text, truncation_note = self._format_telemetry(
            telemetry,
            total_events,
        )
        time_window = f"{meta.time_window_start} -> {meta.time_window_end}"

        system_prompt = (
            "You are the proposal component of SOCpilot, a non-autonomous SOC "
            "copilot. You propose a bounded, ordered incident-response action list for "
            "analyst review. You do not execute tools, claim approvals, alter "
            "systems, or expand the action vocabulary. Reply with one raw JSON "
            "object only: no markdown, no prose outside JSON, and no code fences."
        )

        user_prompt = f"""
Construct a proposed, ordered incident-response action list for the case below.

## Incident metadata
- ID:               {meta.incident_id}
- Type:             {meta.incident_type}
- Severity:         {meta.severity}
- Asset criticality:{meta.asset_criticality}
- Asset role:       {meta.asset_role}
- Time window:      {time_window}

## Correlated telemetry ({len(telemetry)} events){truncation_note}
{telemetry_text}

## Bounded SOC action catalog
{catalog_text}

## Policy context supplied to the copilot
{policy_constraints_block or "No additional rules provided in prompt."}

## Copilot contract
- Output a proposal for analyst review, not an executable procedure.
- Use only action_id values from the bounded catalog.
- Preserve recommended execution order in the JSON array.
- Do not invent actions, synonyms, composite labels, host commands, or
  approval evidence.
- If evidence is insufficient for a catalog action, omit that action.

## Required output
Reply ONLY with JSON using this exact format:
{{
  "recommended_actions": ["action_id_1", "action_id_2", "action_id_3"],
  "reasoning": "Brief rationale with one sentence per recommended action."
}}
""".strip()

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _build_policy_constraints_block(self, rules: list[PolicyRule]) -> str:
        lines: list[str] = []
        for rule in rules:
            rule_parts = [
                f"- [{rule.rule_id}] type={rule.type}",
                f"action={rule.action}",
                f"severity={rule.severity}",
            ]
            if rule.condition_action:
                rule_parts.append(f"prerequisite_action={rule.condition_action}")
            if rule.repair_operator:
                rule_parts.append(f"verifier_repair={rule.repair_operator}")
            if rule.alternative_repair_operator:
                rule_parts.append(
                    f"alternative_repair={rule.alternative_repair_operator}"
                )
            # Provenance belongs to the audited policy bundle, not to the
            # model's decision context. This keeps empirical counts from
            # changing a normative prompt when a corpus is corrected.
            if rule.approval_evidence:
                approval_source = rule.approval_evidence.get("source", "")
                approval_default = rule.approval_evidence.get(
                    "default_when_absent",
                    "",
                )
                if approval_source or approval_default:
                    rule_parts.append(
                        "approval_evidence="
                        f"{approval_source or 'unspecified'}"
                        f"/{approval_default or 'unspecified'}"
                    )
            if rule.rationale:
                rule_parts.append(f"rationale={rule.rationale}")
            lines.append(", ".join(rule_parts))
        return "\n".join(lines)

    def _openai_url(self) -> str:
        endpoint = self._config.endpoint.rstrip("/")
        if endpoint.endswith("/chat/completions"):
            return endpoint
        return f"{endpoint}/chat/completions"

    def _anthropic_url(self) -> str:
        endpoint = self._config.endpoint.rstrip("/")
        if endpoint.endswith("/messages"):
            return endpoint
        return f"{endpoint}/messages"

    def _ollama_url(self) -> str:
        endpoint = self._config.endpoint.rstrip("/")
        if endpoint.endswith("/api/chat"):
            return endpoint
        return f"{endpoint}/api/chat"

    @staticmethod
    def _openai_truncated(body: dict[str, object]) -> bool:
        """True when the model hit the completion-token ceiling (output cut off)."""
        try:
            return body["choices"][0].get("finish_reason") == "length"  # type: ignore[index]
        except (KeyError, IndexError, TypeError):
            return False

    def _extract_openai_content(self, body: dict[str, object]) -> str:
        try:
            message = body["choices"][0]["message"]  # type: ignore[index]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMResponseParseError(
                f"Unexpected OpenAI response format ({_json_identity(body)})."
            ) from exc

        if isinstance(message, dict) and message.get("refusal"):
            raise LLMRefusalError("Model declined to answer.")
        if not isinstance(message, dict) or "content" not in message:
            raise LLMResponseParseError(
                f"OpenAI response has no message content ({_json_identity(body)})."
            )
        content_obj = message["content"]

        if isinstance(content_obj, str):
            return content_obj
        if isinstance(content_obj, list):
            parts: list[str] = []
            for block in content_obj:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
            if parts:
                return "\n".join(parts)
        raise LLMResponseParseError(
            f"Unexpected OpenAI content format ({_json_identity(content_obj)})."
        )

    def _extract_anthropic_content(self, body: dict[str, object]) -> str:
        if body.get("stop_reason") == "refusal":
            raise LLMRefusalError("Model declined to answer (stop_reason=refusal).")
        content_obj = body.get("content")
        if not isinstance(content_obj, list):
            raise LLMResponseParseError(
                f"Unexpected Anthropic response format ({_json_identity(body)})."
            )
        parts: list[str] = []
        for block in content_obj:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        if not parts:
            raise LLMResponseParseError(
                f"Anthropic response has no text blocks ({_json_identity(body)})."
            )
        return "\n".join(parts)

    def _parse_anthropic_usage(self, usage_raw: object) -> LLMUsage | None:
        if not isinstance(usage_raw, dict):
            return None
        prompt_tokens = usage_raw.get("input_tokens")
        completion_tokens = usage_raw.get("output_tokens")
        if not (isinstance(prompt_tokens, int) and isinstance(completion_tokens, int)):
            return None
        return LLMUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

    def _parse_ollama_usage(self, body: dict[str, object]) -> LLMUsage | None:
        prompt_tokens = body.get("prompt_eval_count")
        completion_tokens = body.get("eval_count")
        if not (isinstance(prompt_tokens, int) and isinstance(completion_tokens, int)):
            return None
        return LLMUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

    def _extract_ollama_content(self, body: dict[str, object]) -> str:
        message = body.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise LLMResponseParseError(
                f"Unexpected Ollama response format ({_json_identity(body)})."
            )
        return str(message["content"])

    def _to_anthropic_payload(
        self,
        messages: list[dict[str, str]],
    ) -> dict[str, object]:
        system_parts: list[str] = []
        convo: list[dict[str, str]] = []
        for message in messages:
            role = message.get("role", "").strip()
            content = str(message.get("content", ""))
            if role == "system":
                system_parts.append(content)
                continue
            if role in {"user", "assistant"}:
                convo.append({"role": role, "content": content})
        if not convo:
            convo = [{"role": "user", "content": "Return valid JSON only."}]

        payload: dict[str, object] = {
            "model": self._config.deployment,
            "max_tokens": self._config.max_tokens,
            "messages": convo,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        return payload

    def _build_request(self, messages: list[dict[str, str]]) -> urllib.request.Request:
        cfg = self._config
        body: dict[str, object]
        headers: dict[str, str]
        if cfg.provider == "azure_openai":
            url = (
                f"{cfg.endpoint.rstrip('/')}/openai/deployments/{cfg.deployment}"
                f"/chat/completions?api-version={cfg.api_version}"
            )
            body = {"messages": messages, cfg.azure_token_parameter: cfg.max_tokens}
            headers = {"Content-Type": "application/json", "api-key": cfg.api_key}
        elif cfg.provider == "openai":
            url = self._openai_url()
            body = {
                "model": cfg.deployment,
                "messages": messages,
                "max_completion_tokens": cfg.max_tokens,
            }
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {cfg.api_key}",
            }
        elif cfg.provider == "anthropic":
            url = self._anthropic_url()
            body = self._to_anthropic_payload(messages)
            headers = {
                "Content-Type": "application/json",
                "x-api-key": cfg.api_key,
                "anthropic-version": cfg.api_version or "2023-06-01",
            }
        else:
            url = self._ollama_url()
            body = {
                "model": cfg.deployment,
                "messages": messages,
                "stream": False,
                "think": bool(cfg.thinking),
                "options": {"num_predict": cfg.max_tokens},
            }
            if cfg.response_format == "json":
                body["format"] = "json"
            headers = {"Content-Type": "application/json"}
        if cfg.temperature is not None:
            body["temperature"] = cfg.temperature
        return urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )

    def _decode_provider_response(self, body: dict[str, object]) -> _LLMAPIResponse:
        if self._config.provider in {"azure_openai", "openai"}:
            content = self._extract_openai_content(body)
            usage = self._parse_usage(body.get("usage"))
            output_truncated = self._openai_truncated(body)
        elif self._config.provider == "anthropic":
            content = self._extract_anthropic_content(body)
            usage = self._parse_anthropic_usage(body.get("usage"))
            output_truncated = body.get("stop_reason") == "max_tokens"
        else:
            content = self._extract_ollama_content(body)
            usage = self._parse_ollama_usage(body)
            output_truncated = body.get("done_reason") == "length"
        return _LLMAPIResponse(
            content=content,
            usage=usage,
            latency_ms=None,
            output_truncated=output_truncated,
            returned_model=(str(body["model"]) if body.get("model") else None),
        )

    def _execute_request(self, messages: list[dict[str, str]]) -> _LLMAPIResponse:
        request = self._build_request(messages)
        started_at = time.perf_counter()
        with urllib.request.urlopen(
            request, timeout=_REQUEST_TIMEOUT_SECONDS
        ) as response:
            raw_body = response.read()
        try:
            body = json.loads(raw_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LLMResponseParseError(
                "Provider returned malformed JSON "
                f"(sha256={hashlib.sha256(raw_body).hexdigest()}, bytes={len(raw_body)})."
            ) from exc
        if not isinstance(body, dict):
            raise LLMResponseParseError(
                f"Provider response is not a JSON object ({_json_identity(body)})."
            )
        try:
            decoded = self._decode_provider_response(body)
        except LLMAdapterError as exc:
            exc.usage = (
                self._parse_usage(body.get("usage"))
                if self._config.provider in {"azure_openai", "openai"}
                else (
                    self._parse_anthropic_usage(body.get("usage"))
                    if self._config.provider == "anthropic"
                    else self._parse_ollama_usage(body)
                )
            )
            exc.returned_model = str(body["model"]) if body.get("model") else None
            raise
        return _LLMAPIResponse(
            content=decoded.content,
            usage=decoded.usage,
            latency_ms=round((time.perf_counter() - started_at) * 1000),
            output_truncated=decoded.output_truncated,
            returned_model=decoded.returned_model,
        )

    def _retry_or_raise(
        self,
        *,
        attempt: int,
        retryable: bool,
        error: LLMAdapterError,
        cause: Exception,
        context: str,
        wait_for_connectivity: bool = False,
    ) -> LLMAdapterError:
        cfg = self._config
        if not retryable or attempt >= cfg.attempt_limit:
            raise error from cause
        if wait_for_connectivity and cfg.connectivity_recovery is not None:
            print(
                f"   [warn] {cfg.provider} {context} - waiting for "
                f"{cfg.connectivity_recovery.required_consecutive_successes} consecutive TCP "
                "checks before the next attempt..."
            )
            try:
                wait_for_stable_connectivity(
                    endpoint=cfg.endpoint,
                    provider=cfg.provider,
                    provider_attempt_index=attempt,
                    config=cfg.connectivity_recovery,
                )
            except ConnectivityRecoveryTimeout as exc:
                raise LLMTransportError(
                    "Connectivity recovery window expired before the next "
                    "provider attempt."
                ) from exc
            return error
        print(
            f"   [warn] {cfg.provider} {context} - waiting "
            f"{cfg.retry_wait_seconds}s before attempt "
            f"{attempt + 1}/{cfg.attempt_limit}..."
        )
        time.sleep(cfg.retry_wait_seconds)
        return error

    def _call_api(self, messages: list[dict[str, str]]) -> _LLMAPIResponse:
        """Call the provider API, retrying transient errors (429, 5xx, network)."""
        cfg = self._config
        last_error: LLMAdapterError | None = None
        request_sha256 = self._prompt_sha256(messages)
        serialized_prompt_bytes = len(
            json.dumps(
                messages,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        prompt_tokens_reserved = max(
            cfg.planning_prompt_tokens or 1,
            serialized_prompt_bytes + _PROMPT_RESERVATION_OVERHEAD_TOKENS,
        )
        for attempt in range(1, cfg.attempt_limit + 1):
            reserved_cost_usd: float | None = (
                0.0 if cfg.billing_mode == "local" else None
            )
            if cfg.approved_budget_usd is not None:
                if cfg.attempt_log_path is None:
                    raise LLMAdapterError(
                        "Provider budget has no append-only attempt log."
                    )
                reserved_cost_usd = provider_cost_tracker(
                    cfg.attempt_log_path
                ).require_budget_for_next_attempt(
                    approved_budget_usd=cfg.approved_budget_usd,
                    prompt_tokens_reserved=prompt_tokens_reserved,
                    max_completion_tokens=cfg.max_tokens,
                    prompt_price_per_1k_usd=cfg.prompt_price_per_1k_usd,
                    completion_price_per_1k_usd=cfg.completion_price_per_1k_usd,
                )
            started_at = datetime.now(UTC).isoformat()
            _record_provider_attempt(
                {
                    "provider": cfg.provider,
                    "billing_mode": cfg.billing_mode,
                    "provider_attempt_index": attempt,
                    "configured_attempt_limit": cfg.attempt_limit,
                    "started_at_utc": started_at,
                    "outcome": "request_started",
                    "request_sha256": request_sha256,
                    "prompt_tokens_reserved": prompt_tokens_reserved,
                    "max_completion_tokens_reserved": cfg.max_tokens,
                    "reserved_cost_usd": reserved_cost_usd,
                }
            )
            try:
                response = self._execute_request(messages)
                _record_provider_attempt(
                    {
                        "provider": cfg.provider,
                        "billing_mode": cfg.billing_mode,
                        "provider_attempt_index": attempt,
                        "configured_attempt_limit": cfg.attempt_limit,
                        "started_at_utc": started_at,
                        "finished_at_utc": datetime.now(UTC).isoformat(),
                        "outcome": "response_received",
                        "request_sha256": request_sha256,
                        "returned_model": response.returned_model,
                        "usage": (
                            response.usage.model_dump(mode="json")
                            if response.usage is not None
                            else None
                        ),
                        "estimated_cost_usd": self._estimate_cost_usd(response.usage),
                        "retryable": False,
                    }
                )
                return response
            except urllib.error.HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="replace")
                error_identity = _text_identity(error_body)
                error_summary = _provider_error_summary(error_body, cfg.api_key)
                quota_exhausted = exc.code == _HTTP_RATE_LIMIT and _provider_quota_exhausted(error_body)
                retryable = not quota_exhausted and (exc.code == _HTTP_RATE_LIMIT or (
                    _HTTP_SERVER_ERROR_MIN
                    <= exc.code
                    < _HTTP_SERVER_ERROR_MAX_EXCLUSIVE
                ))
                access_error = exc.code in {401, 403} or quota_exhausted
                if retryable:
                    error_type = LLMTransportError
                elif access_error:
                    error_type = LLMProviderAccessError
                else:
                    error_type = LLMAdapterError
                _record_provider_attempt(
                    {
                        "provider": cfg.provider,
                        "billing_mode": cfg.billing_mode,
                        "provider_attempt_index": attempt,
                        "configured_attempt_limit": cfg.attempt_limit,
                        "started_at_utc": started_at,
                        "finished_at_utc": datetime.now(UTC).isoformat(),
                        "outcome": "http_error",
                        "http_status": exc.code,
                        "request_sha256": request_sha256,
                        "retryable": retryable,
                    }
                )
                last_error = self._retry_or_raise(
                    attempt=attempt,
                    retryable=retryable,
                    error=error_type(
                        f"{self._config.provider} HTTP {exc.code} "
                        f"({error_summary + '; ' if error_summary else ''}"
                        f"{error_identity})."
                    ),
                    cause=exc,
                    context=f"HTTP {exc.code}",
                )
            except urllib.error.URLError as exc:
                _record_provider_attempt(
                    {
                        "provider": cfg.provider,
                        "billing_mode": cfg.billing_mode,
                        "provider_attempt_index": attempt,
                        "configured_attempt_limit": cfg.attempt_limit,
                        "started_at_utc": started_at,
                        "finished_at_utc": datetime.now(UTC).isoformat(),
                        "outcome": "connection_error",
                        "exception_type": type(exc).__name__,
                        "request_sha256": request_sha256,
                        "retryable": True,
                    }
                )
                error = LLMTransportError(
                    f"Connection error ({type(exc.reason).__name__})."
                )
                if (
                    isinstance(exc.reason, socket.gaierror)
                    and cfg.connectivity_recovery is None
                ):
                    raise error from exc
                last_error = self._retry_or_raise(
                    attempt=attempt,
                    retryable=True,
                    error=error,
                    cause=exc,
                    context=f"connection error ({type(exc.reason).__name__})",
                    wait_for_connectivity=True,
                )
            except (http.client.HTTPException, ConnectionError, TimeoutError) as exc:
                _record_provider_attempt(
                    {
                        "provider": cfg.provider,
                        "billing_mode": cfg.billing_mode,
                        "provider_attempt_index": attempt,
                        "configured_attempt_limit": cfg.attempt_limit,
                        "started_at_utc": started_at,
                        "finished_at_utc": datetime.now(UTC).isoformat(),
                        "outcome": "transport_error",
                        "exception_type": type(exc).__name__,
                        "request_sha256": request_sha256,
                        "retryable": True,
                    }
                )
                last_error = self._retry_or_raise(
                    attempt=attempt,
                    retryable=True,
                    error=LLMTransportError(f"Transport error ({type(exc).__name__})."),
                    cause=exc,
                    context=f"transport error ({type(exc).__name__})",
                    wait_for_connectivity=True,
                )
            except LLMAdapterError as exc:
                _record_provider_attempt(
                    {
                        "provider": cfg.provider,
                        "billing_mode": cfg.billing_mode,
                        "provider_attempt_index": attempt,
                        "configured_attempt_limit": cfg.attempt_limit,
                        "started_at_utc": started_at,
                        "finished_at_utc": datetime.now(UTC).isoformat(),
                        "outcome": "terminal_model_response",
                        "exception_type": type(exc).__name__,
                        "request_sha256": request_sha256,
                        "returned_model": exc.returned_model,
                        "usage": (
                            exc.usage.model_dump(mode="json")
                            if exc.usage is not None
                            else None
                        ),
                        "estimated_cost_usd": self._estimate_cost_usd(exc.usage),
                        "retryable": False,
                    }
                )
                raise

        raise LLMAdapterError(
            f"Provider attempt limit reached ({self._config.attempt_limit})."
        ) from last_error

    def _parse_usage(self, usage_raw: object) -> LLMUsage | None:
        if not isinstance(usage_raw, dict):
            return None
        prompt_tokens = usage_raw.get("prompt_tokens")
        completion_tokens = usage_raw.get("completion_tokens")
        total_tokens = usage_raw.get("total_tokens")
        if not all(
            isinstance(value, int)
            for value in (prompt_tokens, completion_tokens, total_tokens)
        ):
            return None
        return LLMUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

    def _estimate_cost_usd(self, usage: LLMUsage | None) -> float | None:
        if self._config.billing_mode == "local":
            return 0.0
        if usage is None:
            return None
        cfg = self._config
        prompt_cost = (usage.prompt_tokens / 1000) * cfg.prompt_price_per_1k_usd
        completion_cost = (
            usage.completion_tokens / 1000
        ) * cfg.completion_price_per_1k_usd
        return round(prompt_cost + completion_cost, 6)

    def _prompt_sha256(self, messages: list[dict[str, str]]) -> str:
        payload = json.dumps(messages, ensure_ascii=False, sort_keys=True).encode(
            "utf-8"
        )
        return hashlib.sha256(payload).hexdigest()

    def _parse_response(
        self,
        raw: str,
        valid_actions: set[str],
    ) -> tuple[list[str], str, list[str]]:
        """Extract recommended_actions and reasoning from the LLM JSON response.

        Returns (action_ids, reasoning, hallucinated_action_ids).
        Filters action_ids not present in the catalog.
        """
        text = raw.strip()
        response_identity = (
            f"sha256={hashlib.sha256(raw.encode('utf-8')).hexdigest()}, "
            f"characters={len(raw)}"
        )
        if text.startswith("```"):
            if not self._config.allow_markdown_json_fence:
                raise LLMResponseParseError(
                    "LLM returned markdown-fenced JSON. Official runs require "
                    f"a raw JSON object ({response_identity})."
                )
            text = "\n".join(
                line for line in text.splitlines() if not line.strip().startswith("```")
            ).strip()

        try:
            obj = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMResponseParseError(
                f"LLM did not return valid JSON ({response_identity}; error={exc})."
            ) from exc

        if not isinstance(obj, dict):
            raise LLMResponseParseError(
                f"LLM response is not a JSON object ({response_identity})."
            )
        if "recommended_actions" not in obj:
            raise LLMResponseParseError(
                f"Missing 'recommended_actions' key in response ({response_identity})."
            )

        raw_actions = obj["recommended_actions"]
        if not isinstance(raw_actions, list):
            raise LLMResponseParseError(
                f"'recommended_actions' must be a list, got: {type(raw_actions)}"
            )

        seen: set[str] = set()
        actions: list[str] = []
        skipped: list[str] = []
        for raw_action in raw_actions:
            action_id = str(raw_action).strip()
            if action_id in valid_actions and action_id not in seen:
                actions.append(action_id)
                seen.add(action_id)
            elif action_id not in valid_actions:
                skipped.append(action_id)

        if skipped:
            print(f"   [warn] LLM suggested {len(skipped)} out-of-catalog action(s).")

        reasoning = str(obj.get("reasoning", ""))
        return actions, reasoning, skipped

    def recommend(
        self,
        meta: IncidentMeta,
        telemetry: list[TelemetryEvent],
        catalog: list[ActionCatalogItem],
        *,
        policy_rules: list[PolicyRule] | None = None,
        policy_prompt_mode: Literal["none", "inline_constraints"] = "none",
    ) -> LLMRecommendation:
        """Request incident-response recommendations from the LLM.

        Args:
            meta: incident metadata parsed from incident_meta.json
            telemetry: list of events from incident_telemetry.jsonl
            catalog: list of action catalog items from action_catalog.yaml

        Returns:
            LLMRecommendation with actions, reasoning, usage, and estimated cost.
        """
        total_events = len(telemetry)
        truncated, was_truncated = self._truncate_telemetry(telemetry)

        if was_truncated:
            print(
                f"   [INFO] Telemetry truncated: {total_events} -> {len(truncated)} events"
                " (prioritizing high-relevance events)"
            )

        valid_actions = {item.action_id for item in catalog}
        policy_constraints_block = ""
        if policy_prompt_mode == "inline_constraints" and policy_rules:
            policy_constraints_block = self._build_policy_constraints_block(
                policy_rules
            )
        messages = self._build_prompt(
            meta,
            truncated,
            catalog,
            total_events,
            policy_constraints_block=policy_constraints_block,
        )
        api_response = self._call_api(messages)
        actions, reasoning, hallucinated_actions = self._parse_response(
            api_response.content,
            valid_actions,
        )
        return LLMRecommendation(
            actions=actions,
            reasoning=reasoning,
            hallucinated_actions=hallucinated_actions,
            usage=api_response.usage,
            latency_ms=api_response.latency_ms,
            estimated_cost_usd=self._estimate_cost_usd(api_response.usage),
            prompt_messages=messages,
            prompt_sha256=self._prompt_sha256(messages),
            returned_model=api_response.returned_model,
            truncated=api_response.output_truncated,
        )

    def complete(self, messages: list[dict[str, str]]) -> LLMTextCompletion:
        """Call the configured model while preserving prompt and cost metadata."""
        api_response = self._call_api(messages)
        return LLMTextCompletion(
            content=api_response.content,
            deployment=self.deployment,
            usage=api_response.usage,
            latency_ms=api_response.latency_ms,
            estimated_cost_usd=self._estimate_cost_usd(api_response.usage),
            prompt_sha256=self._prompt_sha256(messages),
            truncated=api_response.output_truncated,
            returned_model=api_response.returned_model,
        )
