from __future__ import annotations

import json
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from soc_llm_policy import connectivity_recovery as recovery
from soc_llm_policy.connectivity_recovery import (
    ConnectivityRecoveryConfig,
    ConnectivityRecoveryTimeout,
    wait_for_stable_connectivity,
)


class _Clock:
    now = 0.0

    def sleep(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    result = _Clock()
    monkeypatch.setattr(recovery.time, "monotonic", lambda: result.now)
    monkeypatch.setattr(recovery.time, "sleep", result.sleep)
    return result


def _config(tmp_path: Path, **overrides: object) -> ConnectivityRecoveryConfig:
    values = dict(
        probe_timeout_seconds=2,
        probe_interval_seconds=1,
        recovery_timeout_seconds=10,
        required_consecutive_successes=3,
        log_path=tmp_path / "connectivity.jsonl",
    )
    return ConnectivityRecoveryConfig(**(values | overrides))


def _wait(config: ConnectivityRecoveryConfig) -> None:
    wait_for_stable_connectivity(
        endpoint="https://api.example.test/v1",
        provider="openai",
        provider_attempt_index=1,
        config=config,
    )


def test_recovery_resets_success_count_after_failure(monkeypatch, tmp_path, clock):
    outcomes = iter([None, "TimeoutError", None, None, None])
    monkeypatch.setattr(recovery, "_probe_endpoint", lambda *args: next(outcomes))
    config = _config(tmp_path)
    _wait(config)
    records = [json.loads(line) for line in config.log_path.read_text().splitlines()]
    assert [row["consecutive_successes"] for row in records] == [1, 0, 1, 2, 3]
    assert clock.now == 4
    assert "api.example.test" not in config.log_path.read_text()


def test_recovery_does_not_probe_after_deadline(monkeypatch, tmp_path, clock):
    timeouts = []

    def probe(host, port, timeout):
        timeouts.append(timeout)
        clock.now += timeout
        return "TimeoutError"

    monkeypatch.setattr(recovery, "_probe_endpoint", probe)
    with pytest.raises(ConnectivityRecoveryTimeout):
        _wait(_config(tmp_path, recovery_timeout_seconds=1))
    assert timeouts == [1]
    assert clock.now == 1


def test_late_success_cannot_authorize_retry(monkeypatch, tmp_path, clock):
    def probe(*args):
        clock.now = 11

    monkeypatch.setattr(recovery, "_probe_endpoint", probe)
    with pytest.raises(ConnectivityRecoveryTimeout):
        _wait(_config(tmp_path, required_consecutive_successes=1))


@pytest.mark.parametrize(
    "field",
    [
        "probe_timeout_seconds",
        "probe_interval_seconds",
        "recovery_timeout_seconds",
    ],
)
@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan"), True])
def test_recovery_rejects_invalid_time_limits(tmp_path, field, value):
    with pytest.raises(ValueError):
        _config(tmp_path, **{field: value})


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_recovery_rejects_invalid_threshold(tmp_path, value):
    with pytest.raises(ValueError):
        _config(tmp_path, required_consecutive_successes=value)


def test_probe_enforces_timeout_including_dns(monkeypatch):
    def blocked_probe(command, **options):
        assert command[0] == sys.executable
        assert options["timeout"] == 0.1
        raise subprocess.TimeoutExpired(command, options["timeout"])

    monkeypatch.setattr(recovery.subprocess, "run", blocked_probe)
    assert recovery._probe_endpoint("example.test", 443, 0.1) == "TimeoutError"


def test_probe_does_not_mask_worker_bugs(monkeypatch):
    def broken_probe(command, **options):
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(recovery.subprocess, "run", broken_probe)
    with pytest.raises(subprocess.CalledProcessError):
        recovery._probe_endpoint("example.test", 443, 1)


def test_probe_runs_against_local_tcp_listener():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        assert (
            recovery._probe_endpoint("127.0.0.1", listener.getsockname()[1], 5) is None
        )
