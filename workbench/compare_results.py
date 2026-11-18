"""Compare complete exported JSON results; no policy checking is reimplemented."""

import argparse
import json
from pathlib import Path


def unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"Duplicate JSON key: {name}")
        result[name] = value
    return result


def canonical_result(path: Path) -> str:
    value = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=unique_object
    )
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def compare_results(gui_result: Path, cli_result: Path) -> None:
    if canonical_result(gui_result) != canonical_result(cli_result):
        raise ValueError("Complete GUI and CLI JSON results differ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gui_result", type=Path)
    parser.add_argument("cli_result", type=Path)
    args = parser.parse_args()
    compare_results(args.gui_result, args.cli_result)
    print("PASS: complete GUI and CLI JSON results are equal")
