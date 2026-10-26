"""Make a diagnostic redacted copy of supported data files.

The residue gate only detects. This module removes what it detects, then runs the gate again
on the copy and refuses to produce output while any finding remains. The source tree is never
modified. This diagnostic copy is not publication authorization.

Two transformations apply to JSON and JSONL files:

- a case-note event keeps only the declared label of its note class;
- every other string fragment the gate flags becomes a typed placeholder, so a reader still
  sees that something was there and what kind it was.

Files the gate cannot read are refused, not copied unchanged. Unknown private
facts can still evade pattern-based detection, so a separate release review is required.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from soc_llm_policy.case_note_kinds import classify_case_note, publishable_note_text
from soc_llm_policy.publication_residue import (
    CASE_NOTE_EVENT_TYPES,
    ResidueFinding,
    audit_package,
    audit_release_residue,
    scan_string,
)

PLACEHOLDER = "[redacted-{kind}]"


class ResidueRemains(RuntimeError):
    """The anonymized copy still carries findings, so nothing was written."""

    def __init__(self, findings: list[ResidueFinding]) -> None:
        self.findings = findings
        kinds = Counter(finding.kind.value for finding in findings)
        super().__init__(f"{len(findings)} residue findings remain: {dict(kinds)}")


def redact_string(text: str, key: str) -> tuple[str, Counter[str]]:
    redacted: Counter[str] = Counter()
    for _ in range(len(text)):
        hits = scan_string(text, key=key)
        if not hits:
            break
        kind, value = hits[0]
        if not value or value not in text:
            break
        text = text.replace(value, PLACEHOLDER.format(kind=kind.value))
        redacted[kind.value] += 1
    return text, redacted


def redact_value(value: Any, counter: Counter[str], key: str = "") -> Any:
    if isinstance(value, dict):
        return {
            child_key: redact_value(child, counter, str(child_key))
            for child_key, child in value.items()
        }
    if isinstance(value, list):
        return [redact_value(child, counter, key) for child in value]
    if isinstance(value, str):
        text, redacted = redact_string(value, key)
        counter.update(redacted)
        return text
    return value


def redact_event(event: dict[str, Any], counter: Counter[str]) -> dict[str, Any]:
    details = event.get("details")
    if event.get("event_type") in CASE_NOTE_EVENT_TYPES and isinstance(details, dict):
        note = details.get("command")
        if isinstance(note, str):
            label = publishable_note_text(classify_case_note(note))
            if label != note:
                event = {**event, "details": {**details, "command": label}}
                counter["case_note_text"] += 1
    return redact_value(event, counter)


def _rewrite_jsonl(path: Path, counter: Counter[str]) -> None:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    path.write_text(
        "".join(
            json.dumps(redact_event(row, counter), ensure_ascii=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _rewrite_json(path: Path, counter: Counter[str]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(
        json.dumps(redact_value(payload, counter), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


REWRITERS: dict[str, Callable[[Path, Counter[str]], None]] = {
    ".jsonl": _rewrite_jsonl,
    ".json": _rewrite_json,
}
AUDITED_TREES = frozenset({"artifact_data", "artifact_outputs", "scenario_banks"})


def _require_audited_inputs(root: Path) -> None:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if relative.parts[0] not in AUDITED_TREES or not (
            path.suffix == ".json" or path.name == "incident_telemetry.jsonl"
        ):
            raise ValueError(
                "unsupported export input: file is outside the audited data contract"
            )


def audit_tree(root: Path) -> list[ResidueFinding]:
    """The release gate plus a package audit of every directory holding telemetry."""
    findings = list(audit_release_residue(root))
    seen = {(finding.path, finding.location, finding.kind) for finding in findings}
    for telemetry in sorted(root.rglob("incident_telemetry.jsonl")):
        package = telemetry.parent
        for finding in audit_package(
            package, display_root=package.relative_to(root).as_posix()
        ):
            identity = (finding.path, finding.location, finding.kind)
            if identity not in seen:
                seen.add(identity)
                findings.append(finding)
    return findings


def export(source: Path, destination: Path) -> Counter[str]:
    """Copy `source` to `destination` anonymized, or raise without writing anything."""
    source = source.resolve(strict=True)
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    if destination.is_relative_to(source):
        raise ValueError("destination must not be inside the source tree")
    counter: Counter[str] = Counter()
    with tempfile.TemporaryDirectory(dir=destination.parent) as staging:
        copy = Path(staging) / "tree"
        shutil.copytree(
            source, copy, symlinks=True, ignore=shutil.ignore_patterns(".git")
        )
        if any(path.is_symlink() for path in copy.rglob("*")):
            raise ValueError("publication export refuses symbolic links")
        _require_audited_inputs(copy)
        for path in sorted(copy.rglob("*")):
            rewriter = REWRITERS.get(path.suffix)
            if path.is_file() and rewriter is not None:
                rewriter(path, counter)
        remaining = audit_tree(copy)
        if remaining:
            raise ResidueRemains(remaining)
        copy.rename(destination)
    return counter


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    try:
        counter = export(args.source.resolve(), args.destination.resolve())
    except ResidueRemains as refusal:
        print(f"[publication-export] refused: {refusal}")
        return 1
    print(f"[publication-export] wrote {args.destination}; redactions: {dict(counter)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
