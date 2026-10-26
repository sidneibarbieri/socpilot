from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def _runner_path() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    if (repo_root / "artifact_manifest.json").is_file():
        return repo_root / "run.sh"
    return repo_root / "scripts" / "public_artifact_runner.sh"


def test_public_runner_help_requires_no_python_environment() -> None:
    script_path = _runner_path()

    completed = subprocess.run(
        ["bash", str(script_path), "--help"],
        cwd=script_path.parent,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "validate-public-artifact" in completed.stdout
    assert "zero-cost" in completed.stdout
    assert completed.stderr == ""


@pytest.mark.parametrize("entrypoint", ["public", "maintainer"])
def test_validation_preserves_reviewer_caches(tmp_path: Path, entrypoint: str) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source = _runner_path()
    if entrypoint == "maintainer":
        if (repo_root / "artifact_manifest.json").exists():
            pytest.skip("Maintainer entry point is not part of the public package.")
        source = repo_root / "run.sh"
    runner = tmp_path / "run.sh"
    shutil.copyfile(source, runner)
    (tmp_path / "artifact_manifest.json").write_text("{}", encoding="utf-8")
    interpreter = tmp_path / "stub-python"
    interpreter.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "if '-m' not in sys.argv:\n"
        "    print('pre_execution_source')\n",
        encoding="utf-8",
    )
    interpreter.chmod(0o700)
    caches = [
        tmp_path / ".venv/lib/site-packages/__pycache__/package.pyc",
        tmp_path / "src/__pycache__/module.pyc",
        tmp_path / ".pytest_cache/state",
        tmp_path / ".ruff_cache/state",
    ]
    for cache in caches:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(b"reviewer-owned cache")

    subprocess.run(
        ["bash", str(runner), "validate-public-artifact"],
        cwd=tmp_path,
        env={**os.environ, "PYTHON": str(interpreter), "VIRTUAL_ENV": ""},
        capture_output=True,
        text=True,
        check=True,
    )

    for cache in caches:
        assert cache.read_bytes() == b"reviewer-owned cache"
