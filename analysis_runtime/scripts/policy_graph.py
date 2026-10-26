#!/usr/bin/env python3
"""Export or validate the deterministic policy graph projection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from soc_llm_policy.policy_ir import (
    PolicyGraph,
    compile_policy_graph,
    dump_constraints,
    dump_policy_graph,
    load_policy_document,
    project_policy_graph,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    export = subparsers.add_parser("export")
    export.add_argument("--catalog", type=Path, required=True)
    export.add_argument("--constraints", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--graph", type=Path, required=True)

    compile_parser = subparsers.add_parser("compile")
    compile_parser.add_argument("--graph", type=Path, required=True)
    compile_parser.add_argument("--output", type=Path, required=True)
    return parser


def _load_graph(path: Path) -> PolicyGraph:
    return PolicyGraph.model_validate(json.loads(path.read_text(encoding="utf-8")))


def main() -> int:
    args = _parser().parse_args()
    if args.command == "export":
        document = load_policy_document(args.catalog, args.constraints)
        dump_policy_graph(project_policy_graph(document), args.output)
        return 0

    graph = _load_graph(args.graph)
    document = compile_policy_graph(graph)
    if args.command == "compile":
        dump_constraints(document, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
