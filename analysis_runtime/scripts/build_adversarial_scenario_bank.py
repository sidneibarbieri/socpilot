#!/usr/bin/env python3
"""Build a hashed bank of paired clean and inert injected scenarios."""

from __future__ import annotations

import argparse
from pathlib import Path

from soc_llm_policy.adversarial_scenarios import (
    AdversarialBankDesign,
    build_adversarial_bank,
    validate_adversarial_bank,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--incidents-dir", type=Path, required=True)
    parser.add_argument("--incident-ids", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bank-id", required=True)
    parser.add_argument("--source-corpus-id", required=True)
    parser.add_argument("--seed", type=int, default=20260703)
    return parser.parse_args()


def parse_incident_ids(value: str) -> list[str]:
    incident_ids = [item.strip() for item in value.split(",") if item.strip()]
    if not incident_ids:
        raise ValueError("At least one incident ID is required.")
    return incident_ids


def main() -> None:
    args = parse_args()
    design = AdversarialBankDesign(
        bank_id=args.bank_id,
        source_corpus_id=args.source_corpus_id,
        seed=args.seed,
        incident_ids=parse_incident_ids(args.incident_ids),
    )
    manifest = build_adversarial_bank(
        incidents_dir=args.incidents_dir.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        design=design,
    )
    validate_adversarial_bank(manifest, args.output_dir.expanduser().resolve())
    print(
        f"Built {len(manifest.pairs)} clean/injected pairs at "
        f"{args.output_dir.expanduser().resolve()}"
    )


if __name__ == "__main__":
    main()
