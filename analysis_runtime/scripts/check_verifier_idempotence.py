#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from soc_llm_policy.engine import EnforcementRequest, enforce_policy
from soc_llm_policy.io import (
    parse_action_catalog,
    parse_incident_meta,
    parse_rules,
    read_json,
    read_jsonl,
    read_yaml_list,
)


def _iter_verifier_outputs(incidents_root: Path) -> list[Path]:
    files = sorted(incidents_root.glob("INC_*/verifier_output_*_*.json"))
    files.extend(sorted(incidents_root.glob("INC_*/verifier_output.json")))
    # Preserve deterministic order and remove duplicates.
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in files:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def _violation_signature(violations: list[Any]) -> list[dict[str, str | None]]:
    signature: list[dict[str, str | None]] = []
    for violation in violations:
        if hasattr(violation, "to_dict"):
            record = violation.to_dict()
        elif isinstance(violation, dict):
            record = violation
        else:
            raise TypeError(f"unsupported violation record type: {type(violation)!r}")
        signature.append(
            {
                "rule_id": record.get("rule_id"),
                "severity": record.get("severity"),
                "type": record.get("type"),
                "action": record.get("action"),
                "repair": record.get("repair"),
                "missing_before": record.get("missing_before"),
            }
        )
    return sorted(
        signature,
        key=lambda item: (
            str(item["rule_id"]),
            str(item["type"]),
            str(item["action"]),
            str(item["repair"]),
            str(item["missing_before"]),
        ),
    )


def _expected_second_pass_violations(
    payload: dict[str, Any],
) -> list[dict[str, str | None]]:
    original_violations = list(payload.get("violations", []))
    deferred_only = []
    for violation in original_violations:
        if isinstance(violation, dict) and violation.get("type") == "approval_deferred":
            deferred_only.append(violation)
    return _violation_signature(deferred_only)


def run_check(repo_root: Path) -> dict[str, Any]:
    outputs_root = repo_root / "results"
    incidents_root = outputs_root / "incidents"
    rules_path = outputs_root / "global" / "constraints.yaml"
    catalog_path = outputs_root / "global" / "action_catalog.yaml"

    rules = parse_rules(read_yaml_list(rules_path))
    catalog = parse_action_catalog(read_yaml_list(catalog_path))

    files = _iter_verifier_outputs(incidents_root)
    checked = 0
    changed = 0
    second_pass_violations = 0
    changed_violations = 0
    errors: list[dict[str, str]] = []

    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            incident_id = str(payload.get("incident_dir_id") or "").strip()
            if not incident_id:
                raise ValueError("missing incident_dir_id in verifier output")
            enforced = list(payload.get("enforced_actions", []))
            approval_mode = str(payload.get("approval_policy_mode", "remove")).strip()
            approved_actions = {
                str(action_id).strip()
                for action_id in payload.get("incident_approved_actions", [])
                if str(action_id).strip()
            }
            telemetry_path = incidents_root / incident_id / "incident_telemetry.jsonl"
            telemetry = read_jsonl(telemetry_path, strict=True)
            incident_meta = parse_incident_meta(read_json(
                incidents_root / incident_id / "incident_meta.json"
            ))
            violations2, enforced2 = enforce_policy(
                EnforcementRequest(
                    llm_actions=enforced,
                    telemetry=telemetry,
                    rules=rules,
                    catalog=catalog,
                    incident_meta=incident_meta,
                    incident_approved_actions=approved_actions,
                    approval_policy_mode=approval_mode,  # type: ignore[arg-type]
                )
            )
            checked += 1
            if enforced2 != enforced:
                changed += 1
            if violations2:
                second_pass_violations += 1
            if _violation_signature(violations2) != _expected_second_pass_violations(
                payload
            ):
                changed_violations += 1
        except (OSError, ValueError, TypeError) as exc:
            errors.append({"file": str(path), "error": str(exc)})

    return {
        "files_checked": checked,
        "changed_after_second_pass": changed,
        "nonempty_violations_on_second_pass": second_pass_violations,
        "changed_violations_after_second_pass": changed_violations,
        "errors": errors,
        "idempotence_pass": checked > 0
        and changed == 0
        and changed_violations == 0
        and len(errors) == 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="check_verifier_idempotence",
        description=(
            "Re-apply policy enforcement to already enforced plans and report "
            "whether a second pass changes outputs."
        ),
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--output-json",
        default="results/analysis/idempotence_sanity.json",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).expanduser().resolve()
    report = run_check(repo_root)
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved idempotence report: {output_path}")
    print(json.dumps(report, indent=2))
    if not report["idempotence_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
