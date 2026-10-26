from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse


class ConnectivityRecoveryTimeout(RuntimeError):
    """The provider endpoint did not become stable within the fixed window."""


@dataclass(frozen=True)
class ConnectivityRecoveryConfig:
    probe_timeout_seconds: float
    probe_interval_seconds: float
    recovery_timeout_seconds: float
    required_consecutive_successes: int
    log_path: Path | None = None

    def __post_init__(self) -> None:
        for value in (
            self.probe_timeout_seconds,
            self.probe_interval_seconds,
            self.recovery_timeout_seconds,
        ):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(
                    "Connectivity time limits must be finite and positive."
                )
        if (
            isinstance(self.required_consecutive_successes, bool)
            or not isinstance(self.required_consecutive_successes, int)
            or self.required_consecutive_successes < 1
        ):
            raise ValueError(
                "Connectivity success threshold must be a positive integer."
            )


def _probe_endpoint(host: str, port: int, timeout: float) -> str | None:
    # Isolate DNS and socket work so the parent can enforce one wall-clock limit.
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("connectivity_probe.py")),
                host,
                str(port),
                str(timeout),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return "TimeoutError"
    payload = json.loads(result.stdout)
    return payload["exception_type"]


def _append_record(path: Path | None, record: dict[str, object]) -> None:
    if path is None:
        return
    payload = {
        "schema_version": 1,
        "run_id": os.environ.get("SOC_EXPERIMENT_RUN_ID"),
        "cell_id": os.environ.get("SOC_EXPERIMENT_CELL_ID"),
        "execution_id": os.environ.get("SOC_EXPERIMENT_EXECUTION_ID"),
        **record,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, sort_keys=True) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        remaining = memoryview(encoded)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("Connectivity ledger append made no progress.")
            remaining = remaining[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _endpoint_socket(endpoint: str) -> tuple[str, int]:
    parsed = urlparse(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Provider endpoint must be an HTTP(S) URL.")
    default_port = 443 if parsed.scheme == "https" else 80
    return parsed.hostname, parsed.port or default_port


def wait_for_stable_connectivity(
    *,
    endpoint: str,
    provider: str,
    provider_attempt_index: int,
    config: ConnectivityRecoveryConfig,
) -> None:
    """Wait for consecutive TCP successes without issuing a model request."""
    host, port = _endpoint_socket(endpoint)
    endpoint_sha256 = hashlib.sha256(f"{host}:{port}".encode()).hexdigest()
    deadline = time.monotonic() + config.recovery_timeout_seconds
    consecutive_successes = 0
    probe_index = 0

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ConnectivityRecoveryTimeout(
                "Provider connectivity did not stabilize within the fixed window."
            )
        probe_index += 1
        started_at = datetime.now(UTC).isoformat()
        error_type = _probe_endpoint(
            host, port, min(config.probe_timeout_seconds, remaining)
        )
        if error_type is None:
            consecutive_successes += 1
            outcome = "reachable"
        else:
            consecutive_successes = 0
            outcome = "unreachable"

        _append_record(
            config.log_path,
            {
                "provider": provider,
                "provider_attempt_index": provider_attempt_index,
                "probe_index": probe_index,
                "started_at_utc": started_at,
                "finished_at_utc": datetime.now(UTC).isoformat(),
                "endpoint_sha256": endpoint_sha256,
                "outcome": outcome,
                "exception_type": error_type,
                "consecutive_successes": consecutive_successes,
                "required_consecutive_successes": (
                    config.required_consecutive_successes
                ),
            },
        )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ConnectivityRecoveryTimeout(
                "Provider connectivity did not stabilize within the fixed window."
            )
        if consecutive_successes >= config.required_consecutive_successes:
            return
        time.sleep(min(config.probe_interval_seconds, remaining))
