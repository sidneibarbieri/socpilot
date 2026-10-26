#!/usr/bin/env python3
"""Build post-execution evidence from one terminally accounted provider run."""

from __future__ import annotations

import argparse
from pathlib import Path

from soc_llm_policy.execution_evidence import build_execution_evidence_manifest
from soc_llm_policy.paths import resolve_repo_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    repo_root = resolve_repo_root(args.repo_root)
    output_json = Path(args.output_json)
    if not output_json.is_absolute():
        output_json = repo_root / output_json
    manifest_path = build_execution_evidence_manifest(
        repo_root=repo_root,
        run_id=args.run_id,
        output_json=output_json,
    )
    print(f"Wrote execution evidence manifest: {manifest_path}")


if __name__ == "__main__":
    main()
