"""Integrity audit for a measurement run ledger.

Classifies every cell so the official denominator is explicit and the failure
recovery policy is uniform and reviewer-reproducible, not a post-hoc manual
intervention (decision D12):

  completed        a plan was produced and verified.
  model_refusal    the model declined (LLMRefusalError). A valid observation:
                   no plan was produced, kept as-is, never re-executed.
  transport_failed a network/transport error left the cell with no observation
                   after the adapter's in-call retries were exhausted.
                   Recoverable: re-executed once under the identical protocol.
  other_failed     anything else; surfaced for manual inspection, never
                   silently dropped.

The violation-rate denominator is the completed cells (cells that produced a
plan). Refusals are reported in their own denominator. This tool is read-only
over the ledger and prints the recovery plan; it does not itself re-execute.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import BaseModel, Field

# Exhausted-transport exceptions: the adapter retries these in-call, so a cell
# reaching the ledger with one of these produced no observation and is a
# design-completeness gap, not an experimental outcome.
_TRANSPORT_EXCEPTIONS = frozenset(
    {
        "RemoteDisconnected",
        "TimeoutError",
        "ConnectionResetError",
        "ConnectionError",
        "IncompleteRead",
        "URLError",
        "HTTPError",
    }
)
_REFUSAL_EXCEPTIONS = frozenset({"LLMRefusalError"})


class RecoverableCell(BaseModel):
    incident_id: str
    deployment: str
    arm: str
    repeat_index: int | None = None
    exception_type: str


class RunIntegrityReport(BaseModel):
    schema_version: int = 1
    run_id: str
    planned_cells: int = Field(..., ge=0)
    completed_cells: int = Field(..., ge=0)
    model_refusal_cells: int = Field(..., ge=0)
    transport_failed_cells: int = Field(..., ge=0)
    other_failed_cells: int = Field(..., ge=0)
    running_or_planned_cells: int = Field(..., ge=0)
    # Denominator for compliance rates: cells that produced a plan.
    violation_rate_denominator: int = Field(..., ge=0)
    complete: bool
    recoverable: list[RecoverableCell]
    other_failures: list[RecoverableCell]


# Resolve the string annotations from ``from __future__ import annotations`` so
# the model works whether imported by a test or run as a script.
RunIntegrityReport.model_rebuild()


def _classify(entry: dict) -> str:
    status = entry.get("status")
    if status == "completed":
        return "completed"
    if status in {"running", "planned"}:
        return "pending"
    exception = str(entry.get("exception_type") or "")
    if exception in _REFUSAL_EXCEPTIONS:
        return "model_refusal"
    if exception in _TRANSPORT_EXCEPTIONS:
        return "transport_failed"
    return "other_failed"


def build_report(run_id: str, entries: list[dict]) -> RunIntegrityReport:
    counts = {
        "completed": 0,
        "model_refusal": 0,
        "transport_failed": 0,
        "other_failed": 0,
        "pending": 0,
    }
    recoverable: list[RecoverableCell] = []
    other: list[RecoverableCell] = []
    for entry in entries:
        kind = _classify(entry)
        counts[kind] += 1
        if kind in {"transport_failed", "other_failed"}:
            cell = RecoverableCell(
                incident_id=str(entry.get("incident_id")),
                deployment=str(entry.get("deployment")),
                arm=str(entry.get("arm")),
                repeat_index=entry.get("repeat_index"),
                exception_type=str(entry.get("exception_type") or ""),
            )
            (recoverable if kind == "transport_failed" else other).append(cell)
    return RunIntegrityReport(
        run_id=run_id,
        planned_cells=len(entries),
        completed_cells=counts["completed"],
        model_refusal_cells=counts["model_refusal"],
        transport_failed_cells=counts["transport_failed"],
        other_failed_cells=counts["other_failed"],
        running_or_planned_cells=counts["pending"],
        violation_rate_denominator=counts["completed"],
        complete=counts["pending"] == 0 and counts["transport_failed"] == 0,
        recoverable=recoverable,
        other_failures=other,
    )


def _load_entries(ledger_path: Path) -> tuple[str, list[dict]]:
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    entries = ledger if isinstance(ledger, list) else ledger.get(
        "entries", ledger.get("cells", [])
    )
    run_id = ""
    for entry in entries:
        if entry.get("run_id"):
            run_id = str(entry["run_id"])
            break
    return run_id, entries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger-json", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Exit nonzero if pending or transport-failed cells remain.",
    )
    args = parser.parse_args(argv)

    run_id, entries = _load_entries(Path(args.ledger_json))
    report = build_report(run_id, entries)
    payload = report.model_dump(mode="json")
    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    print(
        f"run {report.run_id}: completed {report.completed_cells}, "
        f"refusals {report.model_refusal_cells}, "
        f"transport-failed {report.transport_failed_cells}, "
        f"other-failed {report.other_failed_cells}, "
        f"pending {report.running_or_planned_cells}"
    )
    print(f"violation-rate denominator: {report.violation_rate_denominator}")
    if report.recoverable:
        print("recoverable (transport) cells:")
        for cell in report.recoverable:
            print(
                f"  {cell.incident_id} {cell.deployment} {cell.arm} "
                f"repeat={cell.repeat_index} ({cell.exception_type})"
            )
    if args.require_complete and not report.complete:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
