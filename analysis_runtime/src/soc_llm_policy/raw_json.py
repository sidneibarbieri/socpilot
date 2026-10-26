from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_VALID_SIMPLE_JSON_ESCAPES = frozenset('"\\/bfnrt')
_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")


def escape_invalid_json_backslashes(text: str) -> tuple[str, int]:
    repaired: list[str] = []
    repair_count = 0
    index = 0
    while index < len(text):
        char = text[index]
        if char != "\\":
            repaired.append(char)
            index += 1
            continue
        next_index = index + 1
        if next_index >= len(text):
            repaired.append("\\\\")
            repair_count += 1
            index += 1
            continue
        next_char = text[next_index]
        if next_char in _VALID_SIMPLE_JSON_ESCAPES:
            repaired.append(text[index : index + 2])
            index += 2
            continue
        if (
            next_char == "u"
            and next_index + 4 < len(text)
            and all(ch in _HEX_DIGITS for ch in text[next_index + 1 : next_index + 5])
        ):
            repaired.append(text[index : index + 6])
            index += 6
            continue
        repaired.append("\\\\")
        repair_count += 1
        index += 1
    return "".join(repaired), repair_count


def load_json_with_invalid_escape_repair(path: Path) -> tuple[Any, int]:
    raw_text = path.read_text(encoding="utf-8")
    try:
        return json.loads(raw_text), 0
    except json.JSONDecodeError:
        repaired_text, repair_count = escape_invalid_json_backslashes(raw_text)
        if repair_count <= 0 or repaired_text == raw_text:
            raise
        return json.loads(repaired_text), repair_count


def load_json_object_with_invalid_escape_repair(path: Path) -> tuple[dict[str, Any], int]:
    payload, repair_count = load_json_with_invalid_escape_repair(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must be a JSON object")
    return payload, repair_count
