"""Credential-free TCP probe, isolated from the experiment worker."""

from __future__ import annotations

import json
import socket
import sys


def probe(host: str, port: int, timeout: float) -> str | None:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return None
    except OSError as exc:
        return type(exc).__name__


if __name__ == "__main__":
    error_type = probe(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]))
    print(json.dumps({"exception_type": error_type}))
