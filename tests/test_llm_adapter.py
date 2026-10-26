from __future__ import annotations

import http.client
import io
import json
import socket
import urllib.error
from pathlib import Path

import pytest

from soc_llm_policy.connectivity_recovery import (
    ConnectivityRecoveryConfig,
    ConnectivityRecoveryTimeout,
)
from soc_llm_policy.io import PolicyRule
from soc_llm_policy.llm_adapter import (
    LLMAdapter,
    LLMAdapterError,
    LLMConfig,
    LLMProviderAccessError,
    LLMRefusalError,
    LLMResponseParseError,
    LLMTransportError,
)
from soc_llm_policy.provider_cost import (
    ProviderCostLimitError,
    ProviderCostTracker,
)

_MIN_COMPLETION_TOKEN_BUDGET = 4000
_RETRY_SUCCESS_ATTEMPTS = 2


class _FakeResponse:
    """Minimal context-manager stand-in for a urlopen response."""

    def __init__(self, body: dict[str, object]) -> None:
        self._payload = json.dumps(body).encode("utf-8")

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def read(self) -> bytes:
        return self._payload


class _FakeRawResponse(_FakeResponse):
    def __init__(self, payload: bytes) -> None:
        self._payload = payload


def _set_required_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "chat")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "dummy")


def test_llm_config_reads_attempt_controls_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_required_env(monkeypatch)
    monkeypatch.setenv("SOC_LLM_ATTEMPT_LIMIT", "1")
    monkeypatch.setenv("SOC_LLM_RETRY_WAIT_SECONDS", "0.5")

    config = LLMConfig.from_env()

    assert config.attempt_limit == 1
    assert config.retry_wait_seconds == 0.5


def test_llm_config_requires_complete_cost_guard_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_required_env(monkeypatch)
    monkeypatch.setenv("SOC_EXPERIMENT_APPROVED_BUDGET_USD", "1")

    with pytest.raises(LLMAdapterError, match="requires"):
        LLMConfig.from_env()


def test_llm_config_disallows_markdown_fenced_json_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_required_env(monkeypatch)

    config = LLMConfig.from_env()

    assert config.allow_markdown_json_fence is False


def test_llm_config_can_enable_markdown_fenced_json_for_local_debug(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_required_env(monkeypatch)
    monkeypatch.setenv("SOC_LLM_ALLOW_MARKDOWN_JSON_FENCE", "true")

    config = LLMConfig.from_env()

    assert config.allow_markdown_json_fence is True


def test_llm_config_rejects_invalid_attempt_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_required_env(monkeypatch)
    monkeypatch.setenv("SOC_LLM_ATTEMPT_LIMIT", "zero")

    with pytest.raises(LLMAdapterError):
        LLMConfig.from_env()


def test_llm_config_rejects_non_finite_retry_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_required_env(monkeypatch)
    monkeypatch.setenv("SOC_LLM_RETRY_WAIT_SECONDS", "nan")

    with pytest.raises(LLMAdapterError, match="finite number"):
        LLMConfig.from_env()


def test_llm_config_supports_openai_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SOC_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.2-mini")

    config = LLMConfig.from_env()

    assert config.provider == "openai"
    assert config.endpoint == "https://api.openai.com/v1"
    assert config.deployment == "gpt-5.2-mini"
    assert config.api_key == "dummy"


def test_llm_config_supports_anthropic_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SOC_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")
    monkeypatch.setenv("ANTHROPIC_API_VERSION", "2023-06-01")

    config = LLMConfig.from_env()

    assert config.provider == "anthropic"
    assert config.endpoint == "https://api.anthropic.com/v1"
    assert config.deployment == "claude-sonnet-4-5"
    assert config.api_version == "2023-06-01"


def test_llm_config_supports_local_ollama_without_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SOC_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3.6:latest")
    monkeypatch.setenv("OLLAMA_THINK", "false")
    monkeypatch.setenv("OLLAMA_RESPONSE_FORMAT", "json")

    config = LLMConfig.from_env()

    assert config.provider == "ollama"
    assert config.billing_mode == "local"
    assert config.endpoint == "http://127.0.0.1:11434"
    assert config.deployment == "qwen3.6:latest"
    assert config.api_key == ""
    assert config.thinking is False
    assert config.response_format == "json"


def test_llm_config_rejects_paid_ollama_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SOC_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("SOC_LLM_BILLING_MODE", "metered")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3.6:latest")

    with pytest.raises(LLMAdapterError, match="requires local billing mode"):
        LLMConfig.from_env()


def test_llm_config_rejects_invalid_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SOC_LLM_PROVIDER", "invalid_provider")

    with pytest.raises(LLMAdapterError):
        LLMConfig.from_env()


def test_llm_response_parser_rejects_markdown_fence_by_default() -> None:
    adapter = LLMAdapter(
        LLMConfig(
            endpoint="https://example.openai.azure.com",
            deployment="chat",
            api_key="dummy",
        )
    )

    with pytest.raises(LLMResponseParseError) as error:
        adapter._parse_response(
            '```json\n{"recommended_actions": ["collect_forensics"]}\n```',
            {"collect_forensics"},
        )

    assert "collect_forensics" not in str(error.value)
    assert "sha256=" in str(error.value)


def test_openai_refusal_is_a_typed_terminal_observation() -> None:
    adapter = LLMAdapter(
        LLMConfig(
            endpoint="https://api.openai.com/v1",
            deployment="model",
            api_key="dummy",
        )
    )
    body = {"choices": [{"message": {"content": None, "refusal": "Cannot assist."}}]}

    with pytest.raises(LLMRefusalError, match="declined"):
        adapter._extract_openai_content(body)


def test_openai_refusal_does_not_require_a_content_field() -> None:
    adapter = LLMAdapter(
        LLMConfig(
            endpoint="https://api.openai.com/v1",
            deployment="model",
            api_key="dummy",
        )
    )

    with pytest.raises(LLMRefusalError, match="declined"):
        adapter._extract_openai_content(
            {"choices": [{"message": {"refusal": "private refusal text"}}]}
        )


def test_unexpected_provider_response_is_hashed_not_logged() -> None:
    adapter = LLMAdapter(
        LLMConfig(
            endpoint="https://api.openai.com/v1",
            deployment="model",
            api_key="dummy",
        )
    )

    with pytest.raises(LLMResponseParseError) as error:
        adapter._extract_openai_content({"private": "sensitive response"})

    assert "sensitive response" not in str(error.value)
    assert "sha256=" in str(error.value)


def test_missing_action_list_is_hashed_not_logged() -> None:
    adapter = LLMAdapter(
        LLMConfig(
            endpoint="https://api.openai.com/v1",
            deployment="model",
            api_key="dummy",
        )
    )

    with pytest.raises(LLMResponseParseError) as error:
        adapter._parse_response('{"reasoning":"sensitive response"}', set())

    assert "sensitive response" not in str(error.value)
    assert "sha256=" in str(error.value)


def test_llm_response_parser_accepts_markdown_fence_when_explicitly_enabled() -> None:
    adapter = LLMAdapter(
        LLMConfig(
            endpoint="https://example.openai.azure.com",
            deployment="chat",
            api_key="dummy",
            allow_markdown_json_fence=True,
        )
    )

    actions, _reasoning, skipped = adapter._parse_response(
        '```json\n{"recommended_actions": ["collect_forensics"]}\n```',
        {"collect_forensics"},
    )

    assert actions == ["collect_forensics"]
    assert skipped == []


def test_openai_truncation_detected_from_finish_reason() -> None:
    """A length finish_reason marks the run as output-truncated."""
    assert (
        LLMAdapter._openai_truncated({"choices": [{"finish_reason": "length"}]}) is True
    )
    assert (
        LLMAdapter._openai_truncated({"choices": [{"finish_reason": "stop"}]}) is False
    )
    assert LLMAdapter._openai_truncated({}) is False


def test_completion_budget_default_is_generous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The completion cap is high enough that it does not bind a reasoning model."""
    _set_required_env(monkeypatch)
    assert LLMConfig.from_env().max_tokens >= _MIN_COMPLETION_TOKEN_BUDGET


def test_policy_prompt_excludes_empirical_provenance() -> None:
    adapter = LLMAdapter(
        LLMConfig(
            endpoint="https://example.openai.azure.com",
            deployment="chat",
            api_key="dummy",
        )
    )
    rules = [
        PolicyRule(
            rule_id="isolate_on_tampering",
            type="mandatory",
            action="isolate_host",
            evidence_basis="derivation 17/200, held-out 54/200",
            rationale="Contain the host before encryption.",
        )
    ]

    policy_prompt = adapter._build_policy_constraints_block(rules)

    assert "evidence_basis" not in policy_prompt
    assert "derivation 17/200" not in policy_prompt
    assert "rationale=Contain the host before encryption." in policy_prompt


@pytest.mark.parametrize(
    ("config", "expected_url", "expected_header", "expected_value", "body_key"),
    [
        (
            LLMConfig(
                endpoint="https://azure.example",
                deployment="deployment-a",
                api_key="azure-key",
                api_version="2025-01-01-preview",
            ),
            "https://azure.example/openai/deployments/deployment-a/"
            "chat/completions?api-version=2025-01-01-preview",
            "Api-key",
            "azure-key",
            "max_tokens",
        ),
        (
            LLMConfig(
                endpoint="http://127.0.0.1:11434",
                deployment="qwen3.6:latest",
                api_key="",
                provider="ollama",
                billing_mode="local",
                thinking=False,
                response_format="json",
            ),
            "http://127.0.0.1:11434/api/chat",
            "Content-type",
            "application/json",
            "options",
        ),
        (
            LLMConfig(
                endpoint="https://api.example/v1",
                deployment="model-a",
                api_key="openai-key",
                provider="openai",
            ),
            "https://api.example/v1/chat/completions",
            "Authorization",
            "Bearer openai-key",
            "max_completion_tokens",
        ),
        (
            LLMConfig(
                endpoint="https://anthropic.example/v1",
                deployment="model-b",
                api_key="anthropic-key",
                provider="anthropic",
                api_version="2023-06-01",
            ),
            "https://anthropic.example/v1/messages",
            "X-api-key",
            "anthropic-key",
            "max_tokens",
        ),
    ],
)
def test_build_request_encodes_provider_contract(
    config: LLMConfig,
    expected_url: str,
    expected_header: str,
    expected_value: str,
    body_key: str,
) -> None:
    adapter = LLMAdapter(config)

    request = adapter._build_request(
        [
            {"role": "system", "content": "system instruction"},
            {"role": "user", "content": "user input"},
        ]
    )

    assert request.full_url == expected_url
    assert request.get_header(expected_header) == expected_value
    assert request.data is not None
    payload = json.loads(request.data.decode("utf-8"))
    if config.provider == "ollama":
        assert payload[body_key]["num_predict"] == config.max_tokens
        assert payload["stream"] is False
        assert payload["think"] is False
        assert payload["format"] == "json"
    else:
        assert payload[body_key] == config.max_tokens


def test_decode_ollama_response_preserves_usage_and_truncation() -> None:
    adapter = LLMAdapter(
        LLMConfig(
            endpoint="http://127.0.0.1:11434",
            deployment="qwen3.6:latest",
            api_key="",
            provider="ollama",
            billing_mode="local",
        )
    )

    response = adapter._decode_provider_response(
        {
            "model": "qwen3.6:latest",
            "message": {"role": "assistant", "content": '{"ok":true}'},
            "done": True,
            "done_reason": "length",
            "prompt_eval_count": 21,
            "eval_count": 34,
        }
    )

    assert response.content == '{"ok":true}'
    assert response.returned_model == "qwen3.6:latest"
    assert response.output_truncated is True
    assert response.usage is not None
    assert response.usage.total_tokens == 55


def test_local_attempt_cost_is_known_zero(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("SOC_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3.6:latest")
    monkeypatch.setenv("SOC_LLM_ATTEMPT_LIMIT", "1")
    attempt_log = tmp_path / "provider_attempts.jsonl"
    monkeypatch.setenv("SOC_EXPERIMENT_ATTEMPT_LOG", str(attempt_log))
    monkeypatch.setenv("SOC_EXPERIMENT_CELL_ID", "cell-local")
    monkeypatch.setenv("SOC_EXPERIMENT_EXECUTION_ID", "cell-local:1")
    adapter = LLMAdapter.from_env()
    response_body = {
        "model": "qwen3.6:latest",
        "message": {"role": "assistant", "content": "OK"},
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 2,
        "eval_count": 1,
    }
    monkeypatch.setattr(
        "soc_llm_policy.llm_adapter.urllib.request.urlopen",
        lambda _request, timeout=0: _FakeResponse(response_body),
    )

    response = adapter._call_api([{"role": "user", "content": "OK"}])

    assert adapter._estimate_cost_usd(response.usage) == 0.0
    tracker = ProviderCostTracker(attempt_log)
    tracker.refresh()
    assert tracker.unknown_cost_count == 0
    assert float(tracker.accrued_usd) == 0.0


def test_call_api_retries_on_remote_disconnected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A socket closed mid-response is a transport failure the adapter retries.

    urllib does not wrap the getresponse() phase, so RemoteDisconnected reaches
    the adapter raw rather than as URLError. The first attempt fails at the
    transport layer; the second returns a valid completion.
    """
    monkeypatch.setenv("SOC_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.2-mini")
    monkeypatch.setenv("SOC_LLM_ATTEMPT_LIMIT", "2")
    monkeypatch.setenv("SOC_LLM_RETRY_WAIT_SECONDS", "0")
    attempt_log = tmp_path / "provider_attempts.jsonl"
    monkeypatch.setenv("SOC_EXPERIMENT_ATTEMPT_LOG", str(attempt_log))
    monkeypatch.setenv("SOC_EXPERIMENT_CELL_ID", "cell-a")
    monkeypatch.setenv("SOC_EXPERIMENT_EXECUTION_ID", "cell-a:1")
    monkeypatch.setenv("SOC_EXPERIMENT_APPROVED_BUDGET_USD", "1")
    monkeypatch.setenv("SOC_LLM_PLANNING_PROMPT_TOKENS", "100")
    monkeypatch.setenv("SOC_LLM_PROMPT_PRICE_PER_1K_USD", "0.001")
    monkeypatch.setenv("SOC_LLM_COMPLETION_PRICE_PER_1K_USD", "0.001")
    adapter = LLMAdapter.from_env()

    completion = {
        "model": "gpt-5.2-mini",
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    attempts = {"count": 0}

    def _fake_urlopen(_req: object, timeout: int = 0) -> _FakeResponse:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise http.client.RemoteDisconnected("Remote end closed connection")
        return _FakeResponse(completion)

    monkeypatch.setattr(
        "soc_llm_policy.llm_adapter.urllib.request.urlopen", _fake_urlopen
    )

    response = adapter._call_api([{"role": "user", "content": "hi"}])

    assert attempts["count"] == _RETRY_SUCCESS_ATTEMPTS
    assert response.content == "ok"
    records = [
        json.loads(line)
        for line in attempt_log.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["outcome"] for record in records] == [
        "request_started",
        "transport_error",
        "request_started",
        "response_received",
    ]
    assert {record["cell_id"] for record in records} == {"cell-a"}
    assert len({record["request_sha256"] for record in records}) == 1
    assert records[-1]["returned_model"] == "gpt-5.2-mini"
    assert records[-1]["usage"] == {
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "total_tokens": 2,
    }
    assert records[-1]["estimated_cost_usd"] == 0.000002
    assert all(record["reserved_cost_usd"] > 0 for record in records[::2])
    assert all("api_key" not in record for record in records)
    tracker = ProviderCostTracker(attempt_log)
    tracker.refresh()
    assert tracker.response_count == 1
    assert tracker.reserved_exposure_usd > 0


def test_experiment_transport_retry_waits_for_stable_connectivity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("SOC_LLM_PROVIDER", "openai")
    monkeypatch.setenv("SOC_LLM_CONNECTIVITY_RECOVERY_ENABLED", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.2-mini")
    monkeypatch.setenv("SOC_LLM_ATTEMPT_LIMIT", "2")
    monkeypatch.setenv("SOC_LLM_RETRY_WAIT_SECONDS", "0")
    monkeypatch.setenv("SOC_LLM_CONNECTIVITY_PROBE_TIMEOUT_SECONDS", "2")
    monkeypatch.setenv("SOC_LLM_CONNECTIVITY_PROBE_INTERVAL_SECONDS", "3")
    monkeypatch.setenv("SOC_LLM_CONNECTIVITY_RECOVERY_TIMEOUT_SECONDS", "40")
    monkeypatch.setenv("SOC_LLM_CONNECTIVITY_REQUIRED_SUCCESSES", "3")
    monkeypatch.setenv(
        "SOC_EXPERIMENT_CONNECTIVITY_LOG", str(tmp_path / "connectivity.jsonl")
    )
    adapter = LLMAdapter.from_env()
    calls = {"provider": 0, "recovery": 0}

    completion = {
        "model": "gpt-5.2-mini",
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }

    def request(_req: object, timeout: int = 0) -> _FakeResponse:
        calls["provider"] += 1
        if calls["provider"] == 1:
            raise TimeoutError("network unavailable")
        return _FakeResponse(completion)

    def recover(
        *, endpoint: str, provider: str, provider_attempt_index: int, config: ConnectivityRecoveryConfig
    ) -> None:
        calls["recovery"] += 1
        assert endpoint == "https://api.openai.com/v1"
        assert provider == "openai"
        assert provider_attempt_index == 1
        assert config.required_consecutive_successes == 3

    monkeypatch.setattr(
        "soc_llm_policy.llm_adapter.urllib.request.urlopen", request
    )
    monkeypatch.setattr(
        "soc_llm_policy.llm_adapter.wait_for_stable_connectivity", recover
    )

    response = adapter._call_api([{"role": "user", "content": "hi"}])

    assert response.content == "ok"
    assert calls == {"provider": 2, "recovery": 1}


@pytest.mark.parametrize("attempt_limit", [1, 3])
def test_network_recovery_never_dispatches_after_expiry(
    monkeypatch, tmp_path, attempt_limit
):
    monkeypatch.setenv("SOC_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setenv("SOC_LLM_CONNECTIVITY_RECOVERY_ENABLED", "1")
    monkeypatch.setenv("SOC_LLM_ATTEMPT_LIMIT", str(attempt_limit))
    monkeypatch.setenv("SOC_EXPERIMENT_CONNECTIVITY_LOG", str(tmp_path / "probes.jsonl"))
    log_path = tmp_path / "attempts.jsonl"
    monkeypatch.setenv("SOC_EXPERIMENT_ATTEMPT_LOG", str(log_path))
    recoveries = []

    def offline(*args, **kwargs):
        raise urllib.error.URLError(socket.gaierror("offline"))

    def recovery_timeout(**kwargs):
        recoveries.append(kwargs)
        raise ConnectivityRecoveryTimeout("deadline reached")

    monkeypatch.setattr("soc_llm_policy.llm_adapter.urllib.request.urlopen", offline)
    monkeypatch.setattr("soc_llm_policy.llm_adapter.wait_for_stable_connectivity", recovery_timeout)
    with pytest.raises(LLMTransportError):
        LLMAdapter.from_env()._call_api([{"role": "user", "content": "test"}])
    assert len(recoveries) == (0 if attempt_limit == 1 else 1)
    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert [row["outcome"] for row in records] == ["request_started", "connection_error"]


def test_connectivity_logging_does_not_enable_recovery(monkeypatch, tmp_path):
    monkeypatch.setenv("SOC_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setenv("SOC_LLM_CONNECTIVITY_RECOVERY_ENABLED", "0")
    monkeypatch.setenv("SOC_EXPERIMENT_CONNECTIVITY_LOG", str(tmp_path / "probes.jsonl"))
    assert LLMConfig.from_env().connectivity_recovery is None


def test_enabled_recovery_requires_a_ledger(monkeypatch):
    monkeypatch.setenv("SOC_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setenv("SOC_LLM_CONNECTIVITY_RECOVERY_ENABLED", "1")
    monkeypatch.delenv("SOC_EXPERIMENT_CONNECTIVITY_LOG", raising=False)
    with pytest.raises(LLMAdapterError, match="append-only log"):
        LLMConfig.from_env()


def test_call_api_checks_budget_before_each_transport_attempt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("SOC_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.2-mini")
    monkeypatch.setenv("SOC_LLM_ATTEMPT_LIMIT", "2")
    monkeypatch.setenv("SOC_LLM_RETRY_WAIT_SECONDS", "0")
    monkeypatch.setenv("SOC_LLM_PROMPT_PRICE_PER_1K_USD", "0.001")
    monkeypatch.setenv("SOC_LLM_COMPLETION_PRICE_PER_1K_USD", "0.001")
    monkeypatch.setenv("SOC_LLM_PLANNING_PROMPT_TOKENS", "100")
    monkeypatch.setenv("SOC_EXPERIMENT_APPROVED_BUDGET_USD", "0.007")
    attempt_log = tmp_path / "provider_attempts.jsonl"
    monkeypatch.setenv("SOC_EXPERIMENT_ATTEMPT_LOG", str(attempt_log))
    monkeypatch.setenv("SOC_EXPERIMENT_CELL_ID", "cell-budget")
    monkeypatch.setenv("SOC_EXPERIMENT_EXECUTION_ID", "cell-budget:1")
    adapter = LLMAdapter.from_env()
    call_count = 0

    def _disconnect(_req: object, timeout: int = 0) -> _FakeResponse:
        nonlocal call_count
        call_count += 1
        raise http.client.RemoteDisconnected("Remote end closed connection")

    monkeypatch.setattr(
        "soc_llm_policy.llm_adapter.urllib.request.urlopen", _disconnect
    )

    with pytest.raises(ProviderCostLimitError, match="ambiguous USD"):
        adapter._call_api([{"role": "user", "content": "hi"}])

    assert call_count == 1
    records = [
        json.loads(line)
        for line in attempt_log.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["outcome"] for record in records] == [
        "request_started",
        "transport_error",
    ]


def test_call_api_raises_after_transport_retries_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persistent transport failure surfaces as a typed adapter error, not raw."""
    monkeypatch.setenv("SOC_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.2-mini")
    monkeypatch.setenv("SOC_LLM_ATTEMPT_LIMIT", "2")
    monkeypatch.setenv("SOC_LLM_RETRY_WAIT_SECONDS", "0")
    adapter = LLMAdapter.from_env()

    def _always_disconnect(_req: object, timeout: int = 0) -> _FakeResponse:
        raise http.client.RemoteDisconnected("Remote end closed connection")

    monkeypatch.setattr(
        "soc_llm_policy.llm_adapter.urllib.request.urlopen", _always_disconnect
    )

    with pytest.raises(LLMTransportError, match="Transport error"):
        adapter._call_api([{"role": "user", "content": "hi"}])


def test_call_api_stops_immediately_when_dns_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("SOC_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.2-mini")
    monkeypatch.setenv("SOC_LLM_ATTEMPT_LIMIT", "3")
    attempt_log = tmp_path / "provider_attempts.jsonl"
    monkeypatch.setenv("SOC_EXPERIMENT_ATTEMPT_LOG", str(attempt_log))
    monkeypatch.setenv("SOC_EXPERIMENT_CELL_ID", "cell-dns")
    monkeypatch.setenv("SOC_EXPERIMENT_EXECUTION_ID", "cell-dns:1")
    adapter = LLMAdapter.from_env()
    call_count = 0

    def dns_failure(_req: object, timeout: int = 0) -> _FakeResponse:
        nonlocal call_count
        call_count += 1
        raise urllib.error.URLError(socket.gaierror("name resolution failed"))

    monkeypatch.setattr(
        "soc_llm_policy.llm_adapter.urllib.request.urlopen", dns_failure
    )

    with pytest.raises(LLMTransportError, match="gaierror"):
        adapter._call_api([{"role": "user", "content": "hi"}])

    assert call_count == 1
    records = [
        json.loads(line)
        for line in attempt_log.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["outcome"] for record in records] == [
        "request_started",
        "connection_error",
    ]


@pytest.mark.parametrize("status,error_body", [
    (401, b'{"error":"invalid credential"}'),
    (429, b'{"error":{"type":"insufficient_quota"}}'),
    (429, b'{"error":{"code":"credit_balance_exhausted"}}'),
])
def test_call_api_does_not_retry_provider_access_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    status: int,
    error_body: bytes,
) -> None:
    monkeypatch.setenv("SOC_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "invalid")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.2-mini")
    monkeypatch.setenv("SOC_LLM_ATTEMPT_LIMIT", "3")
    attempt_log = tmp_path / "provider_attempts.jsonl"
    monkeypatch.setenv("SOC_EXPERIMENT_ATTEMPT_LOG", str(attempt_log))
    monkeypatch.setenv("SOC_EXPERIMENT_CELL_ID", "cell-access")
    monkeypatch.setenv("SOC_EXPERIMENT_EXECUTION_ID", "cell-access:1")
    adapter = LLMAdapter.from_env()
    call_count = 0

    def reject_access(_req: object, timeout: int = 0) -> _FakeResponse:
        nonlocal call_count
        call_count += 1
        raise urllib.error.HTTPError(
            url="https://api.openai.com/v1/chat/completions",
            code=status,
            msg="Unauthorized",
            hdrs=None,
            fp=io.BytesIO(error_body),
        )

    monkeypatch.setattr(
        "soc_llm_policy.llm_adapter.urllib.request.urlopen", reject_access
    )

    with pytest.raises(LLMProviderAccessError, match=f"HTTP {status}"):
        adapter._call_api([{"role": "user", "content": "hi"}])

    assert call_count == 1
    records = [
        json.loads(line)
        for line in attempt_log.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["outcome"] for record in records] == [
        "request_started",
        "http_error",
    ]
    assert records[-1]["http_status"] == status
    assert records[-1]["retryable"] is False


def test_call_api_surfaces_bounded_provider_credit_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SOC_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "private-test-key-value")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-opus-4-8")
    adapter = LLMAdapter.from_env()

    def reject_credit(_req: object, timeout: int = 0) -> _FakeResponse:
        raise urllib.error.HTTPError(
            url="https://api.anthropic.com/v1/messages",
            code=400,
            msg="Bad Request",
            hdrs=None,
            fp=io.BytesIO(
                json.dumps(
                    {
                        "type": "error",
                        "error": {
                            "type": "invalid_request_error",
                            "message": "Your credit balance is too low.",
                        },
                    }
                ).encode("utf-8")
            ),
        )

    monkeypatch.setattr(
        "soc_llm_policy.llm_adapter.urllib.request.urlopen", reject_credit
    )

    with pytest.raises(LLMAdapterError, match="credit balance is too low") as error:
        adapter._call_api([{"role": "user", "content": "hi"}])

    assert "private-test-key-value" not in str(error.value)


def test_call_api_classifies_malformed_provider_json_as_terminal_parse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SOC_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.2-mini")
    adapter = LLMAdapter.from_env()

    monkeypatch.setattr(
        "soc_llm_policy.llm_adapter.urllib.request.urlopen",
        lambda _req, timeout=0: _FakeRawResponse(b"private malformed response"),
    )

    with pytest.raises(LLMResponseParseError) as error:
        adapter._call_api([{"role": "user", "content": "hi"}])

    assert "private malformed response" not in str(error.value)
    assert "sha256=" in str(error.value)


@pytest.mark.parametrize("token_parameter", ["max_tokens", "max_completion_tokens"])
def test_azure_explicit_token_parameter(token_parameter: str) -> None:
    config = LLMConfig(
        endpoint="https://azure.example", deployment="arbitrary-deployment",
        api_key="test-key", azure_token_parameter=token_parameter, max_tokens=64,
    )
    request = LLMAdapter(config)._build_request([{ "role": "user", "content": "OK"}])
    payload = json.loads(request.data)
    assert payload[token_parameter] == 64
    other = "max_tokens" if token_parameter == "max_completion_tokens" else "max_completion_tokens"
    assert other not in payload


def test_invalid_azure_token_parameter_fails_before_request(monkeypatch) -> None:
    monkeypatch.setenv("AZURE_OPENAI_TOKEN_PARAMETER", "unlimited")
    with pytest.raises(LLMAdapterError, match="Invalid AZURE_OPENAI_TOKEN_PARAMETER"):
        LLMConfig.from_env()
