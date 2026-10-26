# src/soc_llm_policy/verify.py
from __future__ import annotations

import os

from soc_llm_policy.pipeline import main


def _argv_from_env() -> list[str]:
    incident_id = os.environ.get("INCIDENT_ID", "")
    if not incident_id:
        raise RuntimeError(
            "Set INCIDENT_ID in the environment to use python -m soc_llm_policy.verify"
        )
    return ["--incident", incident_id]


def cli() -> None:
    main(_argv_from_env())


if __name__ == "__main__":
    cli()
