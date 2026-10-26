from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

PUBLICATION_SAFE_CLASSIFICATIONS = frozenset(
    {"synthetic", "public", "authorized_for_publication"}
)


@dataclass(frozen=True)
class PublicationSafetyIssue:
    path: str
    message: str


def version_controlled_exclusions(root: Path, candidates: list[Path]) -> set[Path]:
    """Return the candidates that version control excludes from this tree.

    A bank the repository ignores cannot reach a package, so it is quarantined
    rather than unapproved. Inside a built package there is no repository, so
    nothing is excluded and every unapproved bank is refused.
    """
    if not candidates:
        return set()
    result = subprocess.run(
        ["git", "-C", str(root), "check-ignore", "--stdin"],
        input="\n".join(str(path) for path in candidates),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode not in (0, 1):
        return set()
    return {Path(line).resolve() for line in result.stdout.splitlines() if line.strip()}


def scan_scenario_publication_safety(root: Path) -> list[PublicationSafetyIssue]:
    """Find source-derived scenario data lacking explicit release authority."""
    scenario_root = root / "scenario_banks"
    if not scenario_root.is_dir():
        return []

    manifest_paths = sorted(scenario_root.rglob("source_manifest.json"))
    excluded = version_controlled_exclusions(root, manifest_paths)

    issues: list[PublicationSafetyIssue] = []
    for manifest_path in manifest_paths:
        if manifest_path.resolve() in excluded:
            continue
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("incident_id_assignment") != "source_derived":
            continue
        publication_class = payload.get("publication_classification")
        if publication_class in PUBLICATION_SAFE_CLASSIFICATIONS:
            continue
        issues.append(
            PublicationSafetyIssue(
                path=manifest_path.relative_to(root).as_posix(),
                message=(
                    "Source-derived scenario telemetry has no explicit publication "
                    "authorization. Pseudonymization is not release authority."
                ),
            )
        )
    return issues
