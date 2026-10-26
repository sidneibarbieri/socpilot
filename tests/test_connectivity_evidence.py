from __future__ import annotations

import json

import pytest

from soc_llm_policy.connectivity_evidence import validate_connectivity_evidence


def _state(tmp_path):
    identity = dict(cell_id="cell", execution_id="worker", provider_attempt_index=1)
    failure = dict(
        **identity,
        provider="openai",
        outcome="transport_error",
        finished_at_utc="2026-08-28T00:00:00+00:00",
    )
    retry = dict(
        **(identity | {"provider_attempt_index": 2}),
        provider="openai",
        outcome="request_started",
        started_at_utc="2026-08-28T00:00:10+00:00",
    )
    probes = [
        dict(
            **identity,
            schema_version=1,
            run_id="run",
            provider="openai",
            probe_index=index,
            required_consecutive_successes=3,
            consecutive_successes=index,
            outcome="reachable",
            exception_type=None,
            started_at_utc=f"2026-08-28T00:00:0{index}+00:00",
            finished_at_utc=f"2026-08-28T00:00:0{index}+00:00",
            endpoint_sha256="a" * 64,
        )
        for index in range(1, 4)
    ]
    return [failure, retry], probes


def _validate(tmp_path, attempts, probes):
    attempt_path = tmp_path / "attempts.jsonl"
    log_path = tmp_path / "probes.jsonl"
    attempt_path.write_text("".join(json.dumps(row) + "\n" for row in attempts))
    log_path.write_text("".join(json.dumps(row) + "\n" for row in probes))
    validate_connectivity_evidence(
        log_path=log_path,
        attempt_path=attempt_path,
        required_successes=3,
    )


def test_recovery_is_bound_to_actual_retry(tmp_path):
    _validate(tmp_path, *_state(tmp_path))


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "incomplete",
        "duplicate",
        "wrong_threshold",
        "skipped_index",
        "wrong_count",
        "wrong_worker",
        "wrong_provider",
        "wrong_endpoint",
        "late",
        "naive_time",
        "replaced_response",
        "error_metadata",
    ],
)
def test_corrupted_recovery_is_rejected(tmp_path, mutation):
    attempts, probes = _state(tmp_path)
    if mutation == "missing":
        probes = []
    elif mutation == "incomplete":
        probes.pop()
    elif mutation == "duplicate":
        probes.append(probes[-1])
    elif mutation == "wrong_threshold":
        probes[0]["required_consecutive_successes"] = 1
    elif mutation == "skipped_index":
        probes[0]["probe_index"] = 2
    elif mutation == "wrong_count":
        probes[0]["consecutive_successes"] = 2
    elif mutation == "wrong_worker":
        probes[0]["execution_id"] = "other"
    elif mutation == "wrong_provider":
        probes[0]["provider"] = "other"
    elif mutation == "wrong_endpoint":
        probes[1]["endpoint_sha256"] = "b" * 64
    elif mutation == "late":
        attempts[-1]["started_at_utc"] = "2026-08-28T00:00:02+00:00"
    elif mutation == "naive_time":
        probes[0]["started_at_utc"] = "2026-08-28T00:00:01"
    elif mutation == "replaced_response":
        attempts[0]["outcome"] = "response_received"
    elif mutation == "error_metadata":
        probes[0]["exception_type"] = "TimeoutError"
    with pytest.raises(ValueError):
        _validate(tmp_path, attempts, probes)


def test_stopped_recovery_does_not_require_a_model_retry(tmp_path):
    attempts, probes = _state(tmp_path)
    _validate(tmp_path, attempts[:1], probes[:1])


def test_no_transport_failure_requires_no_probes(tmp_path):
    _validate(tmp_path, [], [])
