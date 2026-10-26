#!/usr/bin/env python3
"""Build a zero-cost fixture covering a corpus slice for the dry run.

Emits one canned response per (incident, prompt mode) so the fixture provider can
drive the full evaluation pipeline without any paid call. The responses are
declared placeholders for plumbing validation; they are not model evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

FixtureProfile = Literal["all_compliant", "mixed", "all_noncompliant"]

_ACTIONS_BY_PROFILE: dict[
    FixtureProfile,
    dict[str, tuple[str, ...]],
] = {
    "all_compliant": {
        "none": ("collect_forensics",),
        "inline_constraints": ("collect_forensics",),
    },
    "mixed": {
        "none": ("collect_forensics", "restore_host"),
        "inline_constraints": ("collect_forensics", "isolate_host"),
    },
    "all_noncompliant": {
        "none": ("collect_forensics", "restore_host"),
        "inline_constraints": ("collect_forensics", "restore_host"),
    },
}


def build_fixture(
    incident_ids: list[str],
    profile: FixtureProfile = "mixed",
    actions_by_mode: dict[str, tuple[str, ...]] | None = None,
) -> dict:
    modes = (
        actions_by_mode
        if actions_by_mode is not None
        else _ACTIONS_BY_PROFILE[profile]
    )
    responses = [
        {
            "incident_id": incident_id,
            "policy_prompt_mode": prompt_mode,
            "recommended_actions": list(actions),
            "reasoning": "Declared zero-cost placeholder for pipeline validation.",
        }
        for incident_id in incident_ids
        for prompt_mode, actions in modes.items()
    ]
    return {
        "schema_version": 1,
        "deployment": "fixture-replay",
        "fixture_profile": profile,
        "constitutes_model_evidence": False,
        "responses": responses,
    }


def resolve_incident_ids(
    incidents_dir: Path,
    explicit_ids: list[str],
    count: int,
) -> list[str]:
    available = sorted(path.name for path in incidents_dir.iterdir() if path.is_dir())
    if explicit_ids:
        missing = [item for item in explicit_ids if item not in set(available)]
        if missing:
            raise ValueError(f"Unknown incident ids: {missing}")
        return explicit_ids
    if count <= 0:
        raise ValueError("Provide --incident-ids or a positive --count")
    return available[:count]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--incidents-dir", type=Path, required=True)
    parser.add_argument("--incident-ids", default="")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument(
        "--profile",
        choices=sorted(_ACTIONS_BY_PROFILE),
        default="mixed",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    explicit_ids = [
        item.strip()
        for item in args.incident_ids.split(",")
        if item.strip()
    ]
    incident_ids = resolve_incident_ids(args.incidents_dir, explicit_ids, args.count)
    fixture = build_fixture(incident_ids, profile=args.profile)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote fixture covering {len(incident_ids)} incidents "
        f"x {len(_ACTIONS_BY_PROFILE[args.profile])} prompt modes "
        f"using profile {args.profile!r}: {args.output}"
    )


if __name__ == "__main__":
    main()
