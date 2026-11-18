"""Qualification and cache failures must never become reusable installations."""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "reviewer_workbench", ROOT / "scripts/run_reviewer_workbench.py"
)
WORKBENCH = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = WORKBENCH
SPEC.loader.exec_module(WORKBENCH)


def write_manifest(root):
    files = WORKBENCH.source_files(root)
    entries = [
        {"path": str(path.relative_to(root)), "sha256": WORKBENCH.sha256(path)}
        for path in files
    ]
    (root / "artifact_manifest.json").write_text(json.dumps({"files": entries}))


def minimal_source(root):
    names = [
        "analysis_runtime/src/soc_llm_policy/__init__.py",
        "analysis_runtime/pyproject.toml",
        "analysis_runtime/LICENSE",
        "analysis_runtime/requirements.txt",
        "workbench/qualify_install.py",
        "scripts/run_reviewer_workbench.py",
        "reproduce.sh",
    ]
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n")
    write_manifest(root)


class WorkbenchInstallationTests(unittest.TestCase):
    def test_runtime_dependencies_must_be_exactly_pinned(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            minimal_source(root)
            (root / "analysis_runtime/requirements.txt").write_text(
                "pydantic>=2.13.4\n"
            )
            with self.assertRaisesRegex(ValueError, "exactly pinned"):
                WORKBENCH.locked_dependencies(root)

    def test_failed_rebuild_preserves_the_previous_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            instance = cache / "instances/existing"
            instance.mkdir(parents=True)
            path = cache / "references/source-id.json"
            reference = WORKBENCH.InstallationReference("source-id", str(instance))
            WORKBENCH.publish_reference(path, reference)
            before = path.read_bytes()
            error = subprocess.CalledProcessError(7, ["qualifier"])
            with patch.object(WORKBENCH.subprocess, "run", side_effect=error):
                with self.assertRaises(subprocess.CalledProcessError):
                    WORKBENCH.prepare_installation(ROOT, cache, "source-id", True)
            self.assertEqual(path.read_bytes(), before)
            self.assertTrue(instance.is_dir())

    def test_source_change_requires_updated_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            minimal_source(root)
            WORKBENCH.source_fingerprint(root)
            (root / "analysis_runtime/src/soc_llm_policy/__init__.py").write_text(
                "changed"
            )
            with self.assertRaisesRegex(ValueError, "not manifest-bound"):
                WORKBENCH.source_fingerprint(root)

    def test_source_and_python_abi_changes_do_not_reuse_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            minimal_source(root)
            original = WORKBENCH.source_fingerprint(root)
            (root / "reproduce.sh").write_text("changed")
            write_manifest(root)
            updated = WORKBENCH.source_fingerprint(root)
            self.assertNotEqual(original, updated)
            with patch.object(
                WORKBENCH.sys.implementation, "cache_tag", "different-python-abi"
            ):
                self.assertNotEqual(updated, WORKBENCH.source_fingerprint(root))

    def test_reference_rejects_unknown_fields_and_paths_outside_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            path = cache / "reference.json"
            payload = {
                "schema_version": "1",
                "fingerprint": "expected",
                "instance": "/outside",
            }
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "escapes"):
                WORKBENCH.InstallationReference.read(path, "expected", cache)
            payload["extra"] = True
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "Invalid"):
                WORKBENCH.InstallationReference.read(path, "expected", cache)

    def test_reference_identity_is_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            reference = WORKBENCH.InstallationReference(
                "old", str(cache / "instances/old")
            )
            path = cache / "reference.json"
            WORKBENCH.publish_reference(path, reference)
            with self.assertRaisesRegex(ValueError, "identity differs"):
                WORKBENCH.InstallationReference.read(path, "new", cache)

    def test_failure_retains_diagnostics_without_publishing_success(self):
        with tempfile.TemporaryDirectory(prefix="workbench with spaces ") as directory:
            root = Path(directory) / "source"
            root.mkdir()
            (root / "workbench").mkdir()
            builder = root / "workbench/qualify_install.py"
            builder.write_text(
                "import sys\nfrom pathlib import Path\n"
                "output=Path(sys.argv[sys.argv.index('--output')+1])\n"
                "output.mkdir()\n(output/'failure.log').write_text('build failed')\n"
                "raise SystemExit(7)\n"
            )
            cache = Path(directory) / "cache"
            with self.assertRaises(subprocess.CalledProcessError) as error:
                WORKBENCH.prepare_installation(root, cache, "source-id", False)
            self.assertEqual(error.exception.returncode, 7)
            self.assertFalse((cache / "references/source-id.json").exists())
            self.assertEqual(len(list((cache / "instances").glob("*/failure.log"))), 1)

    def test_corrupt_cache_is_not_silently_used_and_rebuild_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            instance = cache / "instances/existing"
            instance.mkdir(parents=True)
            reference = WORKBENCH.InstallationReference("source-id", str(instance))
            WORKBENCH.publish_reference(cache / "references/source-id.json", reference)
            with patch.object(
                WORKBENCH, "verify_runtime", side_effect=ValueError("corrupt")
            ):
                with self.assertRaisesRegex(ValueError, "corrupt"):
                    WORKBENCH.prepare_installation(ROOT, cache, "source-id", False)

    def test_relative_cache_root_is_rejected(self):
        with patch.dict(os.environ, {"XDG_CACHE_HOME": "relative"}):
            with self.assertRaisesRegex(ValueError, "absolute"):
                WORKBENCH.cache_directory()

    def test_help_does_not_install_or_require_provider_credentials(self):
        result = subprocess.run(
            ["bash", str(ROOT / "reproduce.sh"), "workbench", "--help"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("--case-input", result.stdout)
        self.assertIn("--rebuild", result.stdout)
        self.assertNotIn("qualifying a clean installation", result.stderr)


if __name__ == "__main__":
    unittest.main()
