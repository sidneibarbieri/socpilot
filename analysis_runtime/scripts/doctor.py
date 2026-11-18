#!/usr/bin/env python3
"""Read-only preflight for the offline review, with the exact fix for every failure.

It runs on any Python 3 using only the standard library, so it can explain a
missing Python 3.12 instead of failing on it. The interpreter it checks is the
one the runner selected; the stray-output rule is read from the release audit
the review itself applies. Nothing is created, installed, moved, or deleted.
"""

from __future__ import annotations

import argparse
import ast
import json
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

# Same imports as has_dependencies in the runner (tests/test_doctor.py checks it).
RUNTIME_MODULES = ("dotenv", "pydantic", "yaml")
MINIMUM = (3, 12)
INSTALL_HINT = ("brew install python@3.12   # Debian/Ubuntu: apt install python3.12 "
                "python3.12-venv; Windows: use WSL")


@dataclass
class Check:
    name: str
    status: str  # OK, WARN, FAIL
    detail: str
    fix: str = ""


def probe(interpreter: str, code: str) -> subprocess.CompletedProcess:
    try:
        return subprocess.run([interpreter, "-c", code], capture_output=True, text=True,
                              check=False, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as error:
        return subprocess.CompletedProcess([interpreter], 127, "", str(error))


def check_interpreter(selected: str) -> list[Check]:
    if not selected:
        return [Check("python", "FAIL", "no Python 3.12+ found (checked python3, python3.14, "
                      "python3.13, python3.12)", INSTALL_HINT)]
    version = probe(selected, "import sys; print('%d.%d.%d' % sys.version_info[:3])")
    if version.returncode != 0:
        return [Check("python", "FAIL", f"{selected} does not run", INSTALL_HINT)]
    found = tuple(int(part) for part in version.stdout.strip().split(".")[:2])
    if found < MINIMUM:
        fix = INSTALL_HINT
        if "/.venv/" in selected:
            fix = "rm -rf .venv   # it was created with an older Python; the next command recreates it"
        return [Check("python", "FAIL", f"{selected} is {version.stdout.strip()}", fix)]
    checks = [Check("python", "OK", f"{selected} ({version.stdout.strip()})")]
    if probe(selected, "import ensurepip, venv").returncode != 0 and "/.venv/" not in selected:
        checks.append(Check("venv module", "FAIL", "cannot create .venv with this interpreter",
                            "apt install python3.12-venv   # or the -venv package of your Python"))
    imports = "import " + ", ".join(RUNTIME_MODULES)
    if probe(selected, imports).returncode == 0:
        checks.append(Check("runtime packages", "OK", ", ".join(RUNTIME_MODULES) + " importable"))
    else:
        checks.append(Check("runtime packages", "WARN", "not installed yet",
                            "None needed: the first command creates .venv and installs them "
                            "(network access to PyPI)."))
    return checks


def non_public_directories(runtime_root: Path) -> tuple[str, ...]:
    """The directory list the release audit flags, read without importing it."""
    source = (runtime_root / "src" / "soc_llm_policy" / "release_audit.py").read_text(encoding="utf-8")
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "_NON_PUBLIC_DATA_DIRS"
                for target in node.targets):
            return tuple(sorted(ast.literal_eval(node.value)))
    raise ValueError("release_audit.py no longer defines _NON_PUBLIC_DATA_DIRS")


def check_stray_outputs(package_root: Path, runtime_root: Path) -> list[Check]:
    stray = []
    for relative in non_public_directories(runtime_root):
        path = package_root / relative
        if path.is_dir() and any(item.is_file() for item in path.rglob("*")):
            stray.append(relative)
    if not stray:
        return [Check("stray outputs", "OK", "no non-public directory in the package")]
    moves = " && ".join(f"mv {name} ../socpilot-{name.replace('/', '-')}-moved" for name in stray)
    return [Check("stray outputs", "FAIL",
                  "review fails on non-public directories: " + ", ".join(stray), moves)]


def check_package(package_root: Path) -> list[Check]:
    manifest = package_root / "artifact_manifest.json"
    if not manifest.is_file():
        return [Check("package", "WARN", "no artifact_manifest.json: development workspace",
                      "Run the review from the published package root.")]
    stage = json.loads(manifest.read_text(encoding="utf-8")).get("artifact_stage", "unknown")
    if stage != "reported_results":
        return [Check("package", "WARN", f"stage {stage}: review/replay need reported results")]
    return [Check("package", "OK", "reported-results package")]


def check_disk(package_root: Path) -> list[Check]:
    free = shutil.disk_usage(package_root).free / 1024**3
    if free < 1:
        return [Check("disk space", "FAIL", f"{free:.1f} GiB free (floor 1 GiB)",
                      "Free at least 1 GiB for .venv and the regenerated reports.")]
    return [Check("disk space", "OK", f"{free:.0f} GiB free (floor 1 GiB)")]


def collect(package_root: Path, runtime_root: Path, selected: str) -> list[Check]:
    return (check_interpreter(selected) + check_package(package_root)
            + check_stray_outputs(package_root, runtime_root) + check_disk(package_root))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=Path.cwd())
    parser.add_argument("--runtime-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--selected-python", default=sys.executable,
                        help="Interpreter the runner chose; empty when none qualifies.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable results.")
    args = parser.parse_args()
    checks = collect(args.package_root.resolve(), args.runtime_root.resolve(), args.selected_python)
    failed = sum(check.status == "FAIL" for check in checks)
    if args.json:
        print(json.dumps([asdict(check) for check in checks], indent=2))
        return 1 if failed else 0
    width = max(len(check.name) for check in checks)
    print(f"host: {platform.system()} {platform.machine()}")
    for check in checks:
        print(f"[{check.status:<4}] {check.name:<{width}}  {check.detail}")
        if check.fix and check.status != "OK":
            print(f"{'':7}{'':<{width}}  fix: {check.fix}")
    print(f"\n{len(checks)} checks, {failed} failing" + ("" if failed else ": ready"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
