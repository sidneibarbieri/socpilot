from __future__ import annotations

import argparse
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from soc_llm_policy.attack import AttackRule, load_attack_rules
from soc_llm_policy.mitre import write_mitre_manifest
from soc_llm_policy.paths import resolve_repo_root

_DEFAULT_STIX_URL = (
    "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/"
    "enterprise-attack/enterprise-attack.json"
)
_MITRE_ATTACK_SOURCE = "mitre-attack"


@dataclass(frozen=True)
class ValidationIssue:
    level: str  # "error" | "warning"
    rule_id: str
    field: str
    message: str


@dataclass(frozen=True)
class AttackCatalog:
    techniques: dict[str, str]
    tactics: dict[str, str]


def _download_stix(url: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read()
    output_path.write_bytes(data)
    return output_path


def _load_stix_bundle(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        bundle = json.load(handle)
    if not isinstance(bundle, dict):
        raise ValueError(f"Invalid STIX bundle (expected object): {path}")
    if not isinstance(bundle.get("objects"), list):
        raise ValueError(f"STIX bundle missing 'objects' list: {path}")
    return bundle


def _extract_external_id(obj: dict[str, Any]) -> str | None:
    refs = obj.get("external_references", [])
    if not isinstance(refs, list):
        return None
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        if ref.get("source_name") != _MITRE_ATTACK_SOURCE:
            continue
        external_id = ref.get("external_id")
        if isinstance(external_id, str) and external_id:
            return external_id
    return None


def _is_active(obj: dict[str, Any]) -> bool:
    return not bool(obj.get("revoked")) and not bool(obj.get("x_mitre_deprecated"))


def load_attack_catalog_from_stix(path: Path) -> AttackCatalog:
    bundle = _load_stix_bundle(path)

    techniques: dict[str, str] = {}
    tactics: dict[str, str] = {}

    for obj in bundle["objects"]:
        if not isinstance(obj, dict):
            continue
        if not _is_active(obj):
            continue

        obj_type = obj.get("type")
        external_id = _extract_external_id(obj)
        name = obj.get("name")
        if not isinstance(external_id, str) or not isinstance(name, str):
            continue

        if obj_type == "attack-pattern":
            techniques[external_id] = name
        elif obj_type == "x-mitre-tactic":
            tactics[external_id] = name

    return AttackCatalog(techniques=techniques, tactics=tactics)


def validate_attack_mapping(
    *,
    mapping_path: Path,
    stix_path: Path,
) -> tuple[list[ValidationIssue], list[AttackRule]]:
    rules = load_attack_rules(mapping_path)
    catalog = load_attack_catalog_from_stix(stix_path)

    issues: list[ValidationIssue] = []
    for rule in rules:
        expected_technique_name = catalog.techniques.get(rule.technique_id)
        expected_tactic_name = catalog.tactics.get(rule.tactic_id)

        if expected_technique_name is None:
            issues.append(
                ValidationIssue(
                    level="error",
                    rule_id=rule.rule_id,
                    field="technique_id",
                    message=(
                        "Technique missing/revoked in ATT&CK: "
                        f"{rule.technique_id}"
                    ),
                )
            )
        elif expected_technique_name != rule.technique_name:
            issues.append(
                ValidationIssue(
                    level="warning",
                    rule_id=rule.rule_id,
                    field="technique_name",
                    message=(
                        "Technique name mismatch "
                        f"{rule.technique_id} (current: {rule.technique_name!r}, "
                        f"official: {expected_technique_name!r})"
                    ),
                )
            )

        if expected_tactic_name is None:
            issues.append(
                ValidationIssue(
                    level="error",
                    rule_id=rule.rule_id,
                    field="tactic_id",
                    message=f"Tactic missing/revoked in ATT&CK: {rule.tactic_id}",
                )
            )
        elif expected_tactic_name != rule.tactic_name:
            issues.append(
                ValidationIssue(
                    level="warning",
                    rule_id=rule.rule_id,
                    field="tactic_name",
                    message=(
                        "Tactic name mismatch "
                        f"{rule.tactic_id} (current: {rule.tactic_name!r}, "
                        f"official: {expected_tactic_name!r})"
                    ),
                )
            )

    return issues, rules


def refresh_attack_mapping_names(
    *,
    mapping_path: Path,
    stix_path: Path,
    output_path: Path | None = None,
) -> Path:
    with mapping_path.open("r", encoding="utf-8") as handle:
        raw_mapping = yaml.safe_load(handle)
    if not isinstance(raw_mapping, list):
        raise ValueError(f"Invalid ATT&CK mapping (expected list): {mapping_path}")

    catalog = load_attack_catalog_from_stix(stix_path)

    refreshed: list[dict[str, Any]] = []
    for item in raw_mapping:
        if not isinstance(item, dict):
            raise ValueError("Each mapping rule must be a YAML object.")
        rule = dict(item)
        technique_id = str(rule.get("technique_id", ""))
        tactic_id = str(rule.get("tactic_id", ""))

        if technique_id in catalog.techniques:
            rule["technique_name"] = catalog.techniques[technique_id]
        if tactic_id in catalog.tactics:
            rule["tactic_name"] = catalog.tactics[tactic_id]
        refreshed.append(rule)

    target = output_path or mapping_path
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            refreshed,
            handle,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="soc_llm_policy.attack_sync")
    parser.add_argument(
        "--mapping",
        default="config/attack_mapping.yaml",
        help="Path to local ATT&CK mapping.",
    )
    parser.add_argument(
        "--stix",
        default="reference_data/mitre/enterprise-attack.json",
        help="Path to local STIX bundle.",
    )
    parser.add_argument(
        "--manifest",
        default="reference_data/mitre/manifest.json",
        help="Path to generated MITRE manifest for traceability.",
    )
    parser.add_argument(
        "--download-stix",
        action="store_true",
        help="Download MITRE STIX bundle before validate/refresh.",
    )
    parser.add_argument(
        "--stix-url",
        default=_DEFAULT_STIX_URL,
        help="STIX bundle URL for download.",
    )
    parser.add_argument(
        "--mode",
        choices=("validate", "refresh"),
        default="validate",
        help="validate: check only; refresh: update names in mapping.",
    )
    parser.add_argument(
        "--output-mapping",
        default=None,
        help="Output path for updated mapping (refresh mode only).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    repo_root = resolve_repo_root(None)
    mapping_path = Path(args.mapping).resolve()
    stix_path = Path(args.stix).resolve()
    manifest_path = Path(args.manifest).resolve()

    if args.download_stix:
        _download_stix(args.stix_url, stix_path)
        print(f"STIX downloaded to: {stix_path}")

    manifest = write_mitre_manifest(stix_path, manifest_path, repo_root=repo_root)
    technique_count = manifest["technique_count"]
    tactic_count = manifest["tactic_count"]
    print(
        "MITRE manifest saved at: "
        f"{manifest_path} "
        f"(techniques={technique_count}, tactics={tactic_count})"
    )

    if args.mode == "refresh":
        output_path = (
            Path(args.output_mapping).resolve() if args.output_mapping else None
        )
        target = refresh_attack_mapping_names(
            mapping_path=mapping_path,
            stix_path=stix_path,
            output_path=output_path,
        )
        print(f"Mapping updated at: {target}")
        # Validate the refreshed output, not the pre-refresh mapping.
        mapping_path = Path(target).resolve()

    issues, rules = validate_attack_mapping(
        mapping_path=mapping_path,
        stix_path=stix_path,
    )
    errors = [issue for issue in issues if issue.level == "error"]
    warnings = [issue for issue in issues if issue.level == "warning"]

    print(f"Validated rules: {len(rules)}")
    print(f"Errors: {len(errors)} | Warnings: {len(warnings)}")
    for issue in issues:
        print(f"[{issue.level.upper()}] {issue.rule_id}.{issue.field}: {issue.message}")

    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
