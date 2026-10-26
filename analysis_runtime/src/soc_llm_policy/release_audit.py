from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from soc_llm_policy.json_stability import write_stable_json
from soc_llm_policy.paths import RepoPaths, resolve_repo_root
from soc_llm_policy.publication_category_audit import (
    SCANNED_TREES,
    audit_published_categories,
)
from soc_llm_policy.publication_residue import audit_scenario_bank
from soc_llm_policy.publication_safety import (
    PUBLICATION_SAFE_CLASSIFICATIONS,
    scan_scenario_publication_safety,
    version_controlled_exclusions,
)

_TRANSIENT_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
}
_TRANSIENT_DIR_PREFIXES = ("tmp_test_outputs",)
_TRANSIENT_FILE_SUFFIXES = (".pyc",)
_TRANSIENT_FILE_NAMES = {".DS_Store"}
_SENSITIVE_FILE_PREFIXES = (".env",)
_PUBLIC_ENV_TEMPLATE_NAMES = {".env.example"}
_PRIVATE_COMPONENTS = {
    "local_redaction",
}
_NON_PUBLIC_DATA_DIRS = {
    "incoming/raw",
    "incoming/redacted",
    "dataset",
    "results",
    "private",
}
_TRACKED_SENSITIVE_FILE_SUFFIXES = (".pem", ".key", ".p12", ".pfx")
_TRACKED_LOCAL_OVERRIDE_SUFFIXES = (
    ".local.yaml",
    ".local.yml",
    ".local.json",
    ".local.toml",
)
_PUBLIC_TEXT_EXTENSIONS = {".md", ".py", ".yaml", ".yml", ".sh", ".tex"}
_AUTHORSHIP_MARKERS = (
    "".join(("chat", "gpt")),
    "".join(("co", "dex")),
    " ".join(("claude", "code")),
    "-".join(("ai", "assisted")),
    " ".join(("ai", "assisted")),
    "-".join(("ai", "generated")),
    " ".join(("ai", "generated")),
    " ".join(("written", "by", "ai")),
    " ".join(("authored", "by", "ai")),
    " ".join(("generated", "with", "ai")),
    " ".join(("generated", "by", "ai")),
)
_AUTHORSHIP_MARKER_EXCLUDED_PREFIXES = (
    "reference_data/",
    "artifact_data/dataset/",
)
_AUTHORSHIP_MARKER_EXCLUDED_RELATIVE_PATHS = {
    "src/soc_llm_policy/release_audit.py",
}
_VENUE_MARKERS = (
    "".join(("n", "dss")),
    "".join(("us", "enix")),
    "".join(("i", "eee")),
    "".join(("c", "cs")),
    "".join(("a", "cm")),
    "".join(("s", "&", "p")),
    " ".join(("security", "symposium")),
    " ".join(("network", "and", "distributed")),
    " ".join(("computer", "and", "communications", "security")),
)
_VENUE_MARKER_PATTERN = re.compile(
    r"(?<![\w-])(?:" + "|".join(re.escape(marker) for marker in _VENUE_MARKERS)
    + r")(?![\w-])",
    flags=re.IGNORECASE,
)
_SECRET_SCAN_PREFIXES = ("artifact_data/", "artifact_outputs/", "config/")
_SECRET_LITERAL_RE = re.compile(
    r"[\"']?\b(api[-_ ]?key|access[-_ ]?token|authorization|password|passwd|secret)"
    r"\b[\"']?\s*[:=]\s*[\"']?(?:Bearer\s+)?([A-Za-z0-9_./+=-]{12,})",
    flags=re.IGNORECASE,
)


def _is_release_bundle(repo_root: Path) -> bool:
    return (repo_root / "artifact_manifest.json").exists()


@dataclass(frozen=True)
class AuditIssue:
    severity: str
    code: str
    path: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {
            "severity": self.severity,
            "code": self.code,
            "path": self.path,
            "message": self.message,
        }


def _is_non_hidden(path: Path) -> bool:
    return not any(part.startswith(".") for part in path.parts)


def _has_hidden_or_transient_component(path: Path) -> bool:
    for part in path.parts:
        if part.startswith("."):
            return True
        if part in _TRANSIENT_DIR_NAMES:
            return True
        if any(part.startswith(prefix) for prefix in _TRANSIENT_DIR_PREFIXES):
            return True
    return False


def _scan_sensitive_files(
    repo_root: Path,
    *,
    allow_local_env: bool,
) -> list[AuditIssue]:
    issues: list[AuditIssue] = []
    for entry in repo_root.iterdir():
        if not entry.is_file():
            continue
        if entry.name in _PUBLIC_ENV_TEMPLATE_NAMES:
            continue
        if any(
            entry.name == prefix or entry.name.startswith(f"{prefix}.")
            for prefix in _SENSITIVE_FILE_PREFIXES
        ):
            severity = "low" if allow_local_env else "high"
            issues.append(
                AuditIssue(
                    severity=severity,
                    code="sensitive_file_present",
                    path=str(entry),
                    message=(
                        "Sensitive environment file found in repository root. "
                        "Rotate credentials and keep this file out of release "
                        "artifacts."
                    ),
                )
            )
    return issues


def _is_local_runtime_path(path: Path, repo_root: Path) -> bool:
    try:
        relative_path = path.relative_to(repo_root)
    except ValueError:
        return False
    return any(
        part in _TRANSIENT_DIR_NAMES for part in relative_path.parts
    ) or relative_path.suffix in _TRANSIENT_FILE_SUFFIXES


def _scan_transient_artifacts(
    repo_root: Path,
    *,
    allow_local_runtime: bool,
) -> list[AuditIssue]:
    issues: list[AuditIssue] = []
    release_bundle = _is_release_bundle(repo_root)
    for path in repo_root.rglob("*"):
        if allow_local_runtime and _is_local_runtime_path(path, repo_root):
            continue
        name = path.name
        if path.is_dir():
            if name == ".venv" and not release_bundle:
                continue
            if name in _TRANSIENT_DIR_NAMES or any(
                name.startswith(prefix) for prefix in _TRANSIENT_DIR_PREFIXES
            ):
                issues.append(
                    AuditIssue(
                        severity="medium",
                        code="transient_dir_present",
                        path=str(path),
                        message=(
                            "Transient directory should be removed before "
                            "release packaging."
                        ),
                    )
                )
            continue
        if name in _TRANSIENT_FILE_NAMES or path.suffix in _TRANSIENT_FILE_SUFFIXES:
            issues.append(
                AuditIssue(
                    severity="medium",
                    code="transient_file_present",
                    path=str(path),
                    message=(
                        "Transient file should be removed before release "
                        "packaging."
                    ),
                )
            )
    return issues


def _scan_private_components(repo_root: Path) -> list[AuditIssue]:
    if _is_release_bundle(repo_root):
        return []
    issues: list[AuditIssue] = []
    for rel in _PRIVATE_COMPONENTS:
        path = repo_root / rel
        if path.exists():
            issues.append(
                AuditIssue(
                    severity="medium",
                    code="private_component_in_repo",
                    path=str(path),
                    message=(
                        "Private preprocessing component is inside repository. "
                        "Keep it outside the public artifact boundary."
                    ),
                )
            )
    return issues


def _scan_non_public_data(repo_root: Path) -> list[AuditIssue]:
    issues: list[AuditIssue] = []
    for rel in _NON_PUBLIC_DATA_DIRS:
        path = repo_root / rel
        if not path.exists() or not path.is_dir():
            continue
        has_files = any(item.is_file() for item in path.rglob("*"))
        if not has_files:
            continue
        issues.append(
            AuditIssue(
                severity="medium",
                code="non_public_data_present",
                path=str(path),
                message=(
                    "Directory contains data generated or managed outside the "
                    "public artifact. "
                    "Ensure release package excludes it."
                ),
            )
        )
    return issues


def _git_tracked_paths(repo_root: Path) -> tuple[Path, list[str]] | None:
    git_root_result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
    )
    if git_root_result.returncode != 0:
        return None
    git_root = Path(git_root_result.stdout.strip()).resolve()
    ls_files_result = subprocess.run(
        ["git", "-C", str(repo_root), "ls-files", "-z"],
        capture_output=True,
        text=False,
        check=False,
    )
    if ls_files_result.returncode != 0:
        return None
    tracked_paths = [
        item.decode("utf-8") for item in ls_files_result.stdout.split(b"\x00") if item
    ]
    return git_root, tracked_paths


def _is_tracked_sensitive_path(rel_path: str) -> bool:
    normalized = rel_path.replace("\\", "/")
    name = Path(normalized).name
    if normalized == ".env":
        return True
    if name.startswith(".env.") and name != ".env.example":
        return True
    if normalized.endswith(_TRACKED_SENSITIVE_FILE_SUFFIXES):
        return True
    if normalized.endswith(_TRACKED_LOCAL_OVERRIDE_SUFFIXES):
        return True
    return False


def _scan_tracked_sensitive_files(repo_root: Path) -> list[AuditIssue]:
    if _is_release_bundle(repo_root):
        return []
    tracked = _git_tracked_paths(repo_root)
    if tracked is None:
        return []
    git_root, tracked_paths = tracked
    issues: list[AuditIssue] = []
    for rel_path in tracked_paths:
        if not _is_tracked_sensitive_path(rel_path):
            continue
        path = (git_root / rel_path).resolve()
        issues.append(
            AuditIssue(
                severity="high",
                code="tracked_sensitive_file",
                path=str(path),
                message=(
                    "Sensitive or local-override file is tracked by git. "
                    "Keep secrets and local overrides out of the published repository."
                ),
            )
        )
    return issues


def _scan_authorship_markers(repo_root: Path) -> list[AuditIssue]:
    issues: list[AuditIssue] = []
    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(repo_root).as_posix()
        rel_path = path.relative_to(repo_root)
        if _has_hidden_or_transient_component(rel_path):
            continue
        if rel in _AUTHORSHIP_MARKER_EXCLUDED_RELATIVE_PATHS:
            continue
        if any(
            rel.startswith(prefix) for prefix in _AUTHORSHIP_MARKER_EXCLUDED_PREFIXES
        ):
            continue
        if path.suffix not in _PUBLIC_TEXT_EXTENSIONS:
            continue
        text = path.read_text(encoding="utf-8")
        lowered = text.lower()
        for marker in _AUTHORSHIP_MARKERS:
            if marker not in lowered:
                continue
            issues.append(
                AuditIssue(
                    severity="medium",
                    code="authorship_marker_present",
                    path=str(path),
                    message=(
                        "Public artifact surface contains tool-authorship or "
                        "tool-assistance language. Keep public docs and code free "
                        "of tool-brand authorship markers."
                    ),
                )
            )
            break
    return issues


def _bank_claims_publication_authority(bank_dir: Path) -> bool:
    """Report whether any package of this bank declares itself publishable."""
    for manifest_path in sorted(bank_dir.rglob("source_manifest.json")):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("publication_classification") in PUBLICATION_SAFE_CLASSIFICATIONS:
            return True
    return False


def _scan_authorized_scenario_residue(repo_root: Path) -> list[AuditIssue]:
    """Check the content of every bank that claims publication authority.

    A publication classification is an assertion by whoever wrote the manifest.
    The residue gate is the independent reading of what the files actually
    contain, so a bank has to satisfy both before it can ship.
    """
    scenario_root = repo_root / "scenario_banks"
    if not scenario_root.is_dir():
        return []

    issues: list[AuditIssue] = []
    for bank_dir in sorted(path for path in scenario_root.iterdir() if path.is_dir()):
        if not _bank_claims_publication_authority(bank_dir):
            continue
        for finding in audit_scenario_bank(bank_dir):
            location = f"{finding.path}:{finding.location}"
            locator = hashlib.sha256(location.encode("utf-8")).hexdigest()
            issues.append(
                AuditIssue(
                    severity="high",
                    code="institutional_residue_in_authorized_scenario",
                    path=f"scenario_banks (locator_sha256={locator})",
                    message=(
                        f"Publication-authorized scenario carries {finding.kind.value}."
                    ),
                )
            )
    return issues


def _scan_published_category_shapes(repo_root: Path) -> list[AuditIssue]:
    if not (repo_root / "artifact_data" / "global" / "constraints.yaml").is_file():
        if _is_release_bundle(repo_root) and any(
            any((repo_root / tree).rglob("incident_telemetry.jsonl"))
            for tree in SCANNED_TREES
        ):
            return [
                AuditIssue(
                    severity="high",
                    code="missing_publication_vocabulary",
                    path="artifact_data/global/constraints.yaml",
                    message="Packaged telemetry has no declared category vocabulary.",
                )
            ]
        return []
    return [
        AuditIssue(
            severity="high",
            code="invalid_published_category",
            path=f"{finding.path} (line {finding.line})",
            message="Telemetry category is outside the declared publication grammar.",
        )
        for finding in audit_published_categories(repo_root)
    ]


def _scan_publication_venue_markers(repo_root: Path) -> list[AuditIssue]:
    candidates = [path for path in repo_root.rglob("*") if path.is_file()]
    excluded = version_controlled_exclusions(repo_root, candidates)
    issues: list[AuditIssue] = []
    for path in candidates:
        if path.resolve() in excluded:
            continue
        rel_path = path.relative_to(repo_root)
        rel = rel_path.as_posix()
        if _has_hidden_or_transient_component(rel_path):
            continue
        if rel in _AUTHORSHIP_MARKER_EXCLUDED_RELATIVE_PATHS:
            continue
        if path.suffix not in _PUBLIC_TEXT_EXTENSIONS | {".json"}:
            continue
        text = path.read_text(encoding="utf-8")
        if not _VENUE_MARKER_PATTERN.search(text):
            continue
        issues.append(
            AuditIssue(
                severity="medium",
                code="publication_venue_marker_present",
                path=str(path),
                message=(
                    "Public artifact surface contains event-specific language. "
                    "Keep the artifact package publication-independent."
                ),
            )
        )
    return issues


def _scan_public_secret_literals(repo_root: Path) -> list[AuditIssue]:
    issues: list[AuditIssue] = []
    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue
        rel_path = path.relative_to(repo_root)
        if _has_hidden_or_transient_component(rel_path):
            continue
        rel = rel_path.as_posix()
        if not any(rel.startswith(prefix) for prefix in _SECRET_SCAN_PREFIXES):
            continue
        if path.suffix not in {".json", ".jsonl", ".yaml", ".yml"}:
            continue
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            match = _SECRET_LITERAL_RE.search(line)
            if match is None:
                continue
            issues.append(
                AuditIssue(
                    severity="high",
                    code="secret_literal_present",
                    path=f"{path}:{line_number}",
                    message=(
                        "Public data or configuration contains a secret-shaped "
                        "literal. Replace it with a deterministic redaction token."
                    ),
                )
            )
    return issues


def audit_release(
    paths: RepoPaths,
    *,
    allow_local_env: bool = False,
) -> dict[str, Any]:
    repo_root = paths.repo_root
    issues = [
        *_scan_sensitive_files(repo_root, allow_local_env=allow_local_env),
        *_scan_transient_artifacts(
            repo_root,
            allow_local_runtime=allow_local_env,
        ),
        *_scan_private_components(repo_root),
        *_scan_non_public_data(repo_root),
        *_scan_tracked_sensitive_files(repo_root),
        *_scan_authorship_markers(repo_root),
        *_scan_publication_venue_markers(repo_root),
        *_scan_public_secret_literals(repo_root),
        *_scan_authorized_scenario_residue(repo_root),
        *_scan_published_category_shapes(repo_root),
        *(
            AuditIssue(
                severity="high",
                code="unapproved_source_derived_scenario",
                path=issue.path,
                message=issue.message,
            )
            for issue in scan_scenario_publication_safety(repo_root)
        ),
    ]
    issue_dicts = [issue.as_dict() for issue in issues]
    by_severity: dict[str, int] = {"high": 0, "medium": 0, "low": 0}
    for issue in issues:
        by_severity[issue.severity] = by_severity.get(issue.severity, 0) + 1

    status = "failed" if by_severity.get("high", 0) > 0 else "warning"
    if not issue_dicts:
        status = "ok"

    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": status,
        "issue_count": len(issue_dicts),
        "issues_by_severity": by_severity,
        "issues": issue_dicts,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit release hygiene (secrets, transient artifacts, private "
            "boundaries)."
        )
    )
    parser.add_argument("--repo-root", default=None)
    parser.add_argument(
        "--output-json",
        default="results/analysis/release_readiness.json",
    )
    parser.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="Exit with non-zero code also when medium-severity issues exist.",
    )
    parser.add_argument(
        "--allow-local-env",
        action="store_true",
        help=(
            "Treat local .env presence as low-severity warning "
            "(for local development)."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    repo_root = resolve_repo_root(args.repo_root)
    paths = RepoPaths(repo_root=repo_root)
    report = audit_release(paths, allow_local_env=args.allow_local_env)
    output_path = Path(args.output_json)
    if not output_path.is_absolute():
        output_path = repo_root / output_path
    write_stable_json(output_path, report)
    if report["status"] == "failed":
        return 1
    if args.fail_on_warning and report["status"] == "warning":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
