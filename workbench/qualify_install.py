"""Build a local wheel bundle and qualify an isolated, offline runtime install."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import select
import shutil
import subprocess
import sys
import tomllib
import venv
from http import HTTPStatus
from http.client import HTTPConnection
from pathlib import Path
from urllib.parse import urlparse


def run_step(
    name: str, command: list[str], output: Path, environment: dict[str, str]
) -> str:
    print(name, flush=True)
    with (output / f"{name}.log").open("w") as log:
        subprocess.run(
            command,
            cwd=output,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=180,
        )
    return (output / f"{name}.log").read_text()


def inspect_server(
    python: Path,
    output: Path,
    environment: dict[str, str],
    expected: dict,
    expected_case: dict,
) -> None:
    with (output / "server.log").open("w") as log:
        process = subprocess.Popen(
            [str(python), "-m", "socpilot_playground", "--port", "0"],
            cwd=output,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=log,
            text=True,
        )
        try:
            if not select.select([process.stdout], [], [], 10)[0]:
                raise TimeoutError(
                    "Installed server did not announce readiness within 10 seconds"
                )
            announcement = process.stdout.readline()
            location = urlparse(announcement.strip().split()[-1])
            if (
                location.scheme != "http"
                or location.hostname != "127.0.0.1"
                or not location.port
            ):
                raise ValueError(
                    f"Unexpected local server announcement: {announcement!r}"
                )
            origin = f"http://127.0.0.1:{location.port}"

            def request(
                method: str, path: str, payload: str | None = None
            ) -> tuple[int, bytes]:
                connection = HTTPConnection("127.0.0.1", location.port, timeout=5)
                try:
                    connection.request(
                        method,
                        path,
                        payload,
                        {
                            "Origin": origin,
                            "Content-Type": "application/json",
                        },
                    )
                    response = connection.getresponse()
                    return response.status, response.read()
                finally:
                    connection.close()

            for path in (
                "/",
                "/app.js",
                "/style.css",
                "/api/example",
                "/api/case-example",
                "/api/case-schema",
            ):
                status, body = request("GET", path)
                if status != HTTPStatus.OK or not body:
                    raise RuntimeError(f"Installed asset failed: {path}, HTTP {status}")
            status, _ = request("POST", "/api/check", '{"actions":["unknown"]}')
            if status != HTTPStatus.UNPROCESSABLE_ENTITY:
                raise RuntimeError("Invalid action did not produce HTTP 422")
            status, body = request(
                "POST", "/api/check", '{"actions":["reimage_host","isolate_host"]}'
            )
            if status != HTTPStatus.OK or json.loads(body) != expected:
                raise RuntimeError(
                    "Installed HTTP and CLI results differ, or recovery failed"
                )
            status, body = request(
                "POST", "/api/cases/check", json.dumps(expected_case["input_case"])
            )
            if status != HTTPStatus.OK or json.loads(body) != expected_case:
                raise RuntimeError("Installed policy-case HTTP and CLI results differ")
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            process.stdout.close()


def qualify(core_root: Path, output: Path) -> None:
    source = Path(__file__).resolve().parent
    project_version = tomllib.loads((source / "pyproject.toml").read_text())["project"][
        "version"
    ]
    core_root = core_root.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    environment = {
        "PATH": os.defpath,
        "PYTHONNOUSERSITE": "1",
        "PIP_CONFIG_FILE": os.devnull,
    }
    wheels = output / "wheelhouse"
    wheels.mkdir()
    staged_core = output / "core-source"
    staged_core.mkdir()
    staged_interface = output / "interface-source"
    staged_interface.mkdir()
    shutil.copy2(source / "pyproject.toml", staged_interface / "pyproject.toml")
    shutil.copytree(
        source / "socpilot_playground",
        staged_interface / "socpilot_playground",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    for filename in ("pyproject.toml", "LICENSE"):
        shutil.copy2(core_root / filename, staged_core / filename)
    source_hashes = {}
    for path in sorted((core_root / "src").rglob("*.py")):
        relative = path.relative_to(core_root)
        destination = staged_core / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        source_hashes[str(relative)] = hashlib.sha256(
            destination.read_bytes()
        ).hexdigest()
    shutil.copy2(core_root / "requirements.txt", output / "requirements.lock")
    examples = output / "examples"
    examples.mkdir()
    for filename in ("violating-proposal.json", "compliant-proposal.json"):
        shutil.copy2(source / filename, examples / filename)
    shutil.copy2(
        source / "socpilot_playground/policy-case.json", examples / "policy-case.json"
    )

    venv.EnvBuilder(with_pip=True).create(output / "build-env")
    builder = output / "build-env/bin/python"
    run_step(
        "build-dependencies",
        [str(builder), "-m", "pip", "install", "setuptools==80.9.0", "wheel==0.45.1"],
        output,
        environment,
    )
    for name, project in (("core", staged_core), ("interface", staged_interface)):
        run_step(
            f"build-{name}",
            [
                str(builder),
                "-m",
                "pip",
                "wheel",
                "--no-deps",
                "--no-build-isolation",
                "--wheel-dir",
                str(wheels),
                str(project),
            ],
            output,
            environment,
        )
    run_step(
        "download-runtime",
        [
            str(builder),
            "-m",
            "pip",
            "download",
            "--only-binary=:all:",
            "--dest",
            str(wheels),
            "-r",
            str(output / "requirements.lock"),
        ],
        output,
        environment,
    )

    venv.EnvBuilder(with_pip=True).create(output / "runtime-env")
    python = output / "runtime-env/bin/python"
    run_step(
        "install-offline",
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-index",
            "--find-links",
            str(wheels),
            "-r",
            str(output / "requirements.lock"),
            f"socpilot-playground=={project_version}",
        ],
        output,
        environment,
    )
    run_step(
        "dependency-check", [str(python), "-m", "pip", "check"], output, environment
    )
    freeze = run_step(
        "runtime-freeze", [str(python), "-m", "pip", "freeze"], output, environment
    )
    installed_paths = run_step(
        "installed-paths",
        [
            str(python),
            "-c",
            "import socpilot_playground,soc_llm_policy; "
            "print(socpilot_playground.__file__); print(soc_llm_policy.__file__)",
        ],
        output,
        environment,
    )
    if any(
        not Path(path).is_relative_to(output / "runtime-env")
        for path in installed_paths.splitlines()
    ):
        raise RuntimeError("Runtime imports escaped the clean virtual environment")
    cli = run_step(
        "cli-violating",
        [
            str(python),
            "-m",
            "socpilot_playground",
            "--input",
            str(examples / "violating-proposal.json"),
        ],
        output,
        environment,
    )
    expected = json.loads(cli)
    if (
        expected["repaired_actions"] != ["collect_forensics", "reimage_host"]
        or expected["violated_rules_after"]
    ):
        raise RuntimeError("Installed CLI failed the expected repair")
    print("inspect-installed-server", flush=True)
    case_cli = run_step(
        "cli-policy-case",
        [
            str(python),
            "-m",
            "socpilot_playground",
            "--case-input",
            str(examples / "policy-case.json"),
        ],
        output,
        environment,
    )
    expected_case = json.loads(case_cli)
    if (
        expected_case["repaired_actions"] != ["capture_memory", "restart_service"]
        or expected_case["violated_rules_after"]
    ):
        raise RuntimeError("Installed policy case failed the expected repair")
    inspect_server(python, output, environment, expected, expected_case)
    wheel_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(wheels.glob("*.whl"))
    }
    (output / "SHA256SUMS").write_text(
        "".join(
            f"{digest}  wheelhouse/{name}\n" for name, digest in wheel_hashes.items()
        )
    )
    receipt = {
        "status": "passed",
        "evidence_kind": "local_clean_install_qualification",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "runtime_install": "offline_from_local_wheels",
        "interface_version": project_version,
        "dependencies": freeze.splitlines(),
        "wheel_sha256": wheel_hashes,
        "core_python_source_sha256": source_hashes,
        "distribution_input_sha256": {
            str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [output / "requirements.lock", *sorted(examples.glob("*.json"))]
        },
        "checks": [
            "clean_import_paths",
            "pip_check",
            "cli_expected_repair",
            "installed_static_assets",
            "invalid_input_rejected",
            "recovery_after_invalid_input",
            "http_cli_parity",
            "policy_case_http_cli_parity",
            "server_teardown",
        ],
        "scope": "synthetic guided and structured cases; four rule families; no model calls; no host actions; no campaign execution",
    }
    (output / "qualification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"PASS: {output / 'qualification.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    if sys.version_info < (3, 12):  # noqa: UP036 -- standalone installer diagnoses unsupported Python.
        parser.error("Python 3.12 or newer is required")
    qualify(arguments.core_root, arguments.output)
