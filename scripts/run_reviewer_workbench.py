"""Install and run the policy workbench against this artifact's own engine."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SUFFIXES = {".py", ".toml", ".json", ".html", ".js", ".css", ".svg"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_files(root: Path) -> list[Path]:
    runtime = root / "analysis_runtime"
    files = list((runtime / "src").rglob("*.py"))
    files += [
        runtime / name for name in ("pyproject.toml", "LICENSE", "requirements.txt")
    ]
    files += [
        path
        for path in (root / "workbench").rglob("*")
        if path.is_file() and path.suffix in SOURCE_SUFFIXES
    ]
    files += [root / "scripts/run_reviewer_workbench.py", root / "reproduce.sh"]
    return sorted(set(files))


def source_fingerprint(root: Path) -> str:
    manifest = root / "artifact_manifest.json"
    payload = json.loads(manifest.read_text())
    entries = {entry["path"]: entry["sha256"] for entry in payload["files"]}
    for path in source_files(root):
        relative = path.relative_to(root).as_posix()
        if entries.get(relative) != sha256(path):
            raise ValueError(f"Workbench source is not manifest-bound: {relative}")
    identity = [
        sha256(manifest),
        sys.version,
        sys.implementation.cache_tag,
        sys.platform,
        platform.machine(),
        str(Path(sys.executable).resolve()),
    ]
    return hashlib.sha256(json.dumps(identity).encode()).hexdigest()


def cache_directory() -> Path:
    parent = Path(
        os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))
    ).expanduser()
    if not parent.is_absolute():
        raise ValueError("XDG_CACHE_HOME must be an absolute path")
    return parent / "socpilot" / "reviewer-workbench"


@dataclass(frozen=True)
class InstallationReference:
    fingerprint: str
    instance: str
    schema_version: str = "1"

    @classmethod
    def read(cls, path: Path, fingerprint: str, cache: Path) -> InstallationReference:
        payload = json.loads(path.read_text())
        if set(payload) != {"fingerprint", "instance", "schema_version"}:
            raise ValueError("Invalid workbench cache reference; use --rebuild")
        if payload["schema_version"] != "1" or payload["fingerprint"] != fingerprint:
            raise ValueError("Workbench cache identity differs; use --rebuild")
        if not isinstance(payload["instance"], str):
            raise ValueError("Invalid workbench instance path")
        instance = Path(payload["instance"]).resolve()
        if not instance.is_relative_to((cache / "instances").resolve()):
            raise ValueError("Workbench instance escapes its cache directory")
        return cls(fingerprint, str(instance))


def publish_reference(path: Path, reference: InstallationReference) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
        json.dump(asdict(reference), handle, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


@dataclass(frozen=True)
class RuntimeMetadata:
    prefix: str
    engine: str
    interface: str
    engine_version: str
    interface_version: str
    dependencies: dict[str, str]


def locked_dependencies(root: Path) -> dict[str, str]:
    versions = {}
    for line in (root / "analysis_runtime/requirements.txt").read_text().splitlines():
        requirement = line.strip()
        if not requirement or requirement.startswith("#"):
            continue
        name, separator, version = requirement.partition("==")
        if (
            not separator
            or not name
            or not version
            or any(char.isspace() for char in requirement)
        ):
            raise ValueError(
                f"Runtime dependency must be exactly pinned: {requirement}"
            )
        versions[name] = version
    return versions


def installed_paths(
    python: Path, instance: Path, dependencies: list[str]
) -> RuntimeMetadata:
    code = (
        "import json,sys,soc_llm_policy,socpilot_playground; "
        "from importlib.metadata import version; "
        "print(json.dumps(dict(prefix=sys.prefix,engine=soc_llm_policy.__file__,"
        "interface=socpilot_playground.__file__,engine_version=version('soc-llm-policy'),"
        "interface_version=version('socpilot-playground'),"
        "dependencies={name:version(name) for name in json.loads(sys.argv[1])})))"
    )
    result = subprocess.run(
        [str(python), "-I", "-c", code, json.dumps(dependencies)],
        cwd=instance,
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    )
    return RuntimeMetadata(**json.loads(result.stdout))


def verify_runtime(instance: Path, root: Path) -> Path:
    receipt = json.loads((instance / "qualification.json").read_text())
    if receipt.get("status") != "passed":
        raise ValueError("Workbench installation did not qualify; use --rebuild")
    python = instance / "runtime-env/bin/python"
    dependencies = locked_dependencies(root)
    paths = installed_paths(python, instance, list(dependencies))
    for name, expected in dependencies.items():
        actual = paths.dependencies[name]
        if actual != expected:
            raise ValueError(
                f"Installed {name}={actual} differs from {expected}; use --rebuild"
            )
    environment = (instance / "runtime-env").resolve()
    for location in (paths.prefix, paths.engine, paths.interface):
        if not Path(location).resolve().is_relative_to(environment):
            raise ValueError("Workbench import escaped its qualified environment")
    for project, installed_version in (
        (root / "analysis_runtime", paths.engine_version),
        (root / "workbench", paths.interface_version),
    ):
        expected = tomllib.loads((project / "pyproject.toml").read_text())["project"][
            "version"
        ]
        if installed_version != expected:
            raise ValueError(f"Installed {project.name} version differs; use --rebuild")
    site_packages = Path(paths.engine).parent.parent
    for source in (root / "analysis_runtime/src").rglob("*.py"):
        relative = source.relative_to(root / "analysis_runtime/src")
        if sha256(site_packages / relative) != sha256(source):
            raise ValueError(f"Installed engine differs: {relative}; use --rebuild")
    package = root / "workbench/socpilot_playground"
    installed_package = Path(paths.interface).parent
    for source in package.rglob("*"):
        if source.is_file() and source.suffix in SOURCE_SUFFIXES:
            relative = source.relative_to(package)
            if sha256(installed_package / relative) != sha256(source):
                raise ValueError(
                    f"Installed interface differs: {relative}; use --rebuild"
                )
    return python


def prepare_installation(
    root: Path, cache: Path, fingerprint: str, rebuild: bool
) -> Path:
    reference_path = cache / "references" / (fingerprint + ".json")
    if reference_path.exists() and not rebuild:
        reference = InstallationReference.read(reference_path, fingerprint, cache)
        return verify_runtime(Path(reference.instance), root)
    instances = cache / "instances"
    instances.mkdir(parents=True, exist_ok=True, mode=0o700)
    instance = instances / (fingerprint[:12] + "-" + uuid4().hex)
    print(
        f"[workbench] qualifying a clean installation; logs: {instance}",
        file=sys.stderr,
    )
    subprocess.run(
        [
            sys.executable,
            str(root / "workbench/qualify_install.py"),
            "--core-root",
            str(root / "analysis_runtime"),
            "--output",
            str(instance),
        ],
        cwd=cache,
        stdout=sys.stderr,
        stderr=sys.stderr,
        check=True,
    )
    python = verify_runtime(instance, root)
    publish_reference(reference_path, InstallationReference(fingerprint, str(instance)))
    return python


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--rebuild", action="store_true")
    options, arguments = parser.parse_known_args()
    fingerprint = source_fingerprint(ROOT)
    python = prepare_installation(ROOT, cache_directory(), fingerprint, options.rebuild)
    environment = {
        name: value
        for name, value in os.environ.items()
        if name not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"}
    }
    os.execve(
        python,
        [str(python), "-I", "-m", "socpilot_playground", *arguments],
        environment,
    )


if __name__ == "__main__":
    main()
