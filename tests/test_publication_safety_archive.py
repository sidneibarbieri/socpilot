"""Archive safety checks must inspect every bank without requiring Git."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "analysis_runtime/src/soc_llm_policy/publication_safety.py"
)
SPEC = importlib.util.spec_from_file_location("publication_safety_archive", MODULE_PATH)
PUBLICATION_SAFETY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PUBLICATION_SAFETY
SPEC.loader.exec_module(PUBLICATION_SAFETY)


class ArchiveSafetyTests(unittest.TestCase):
    def test_archive_checks_unapproved_bank_without_git(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "scenario_banks/example/source_manifest.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(json.dumps({"incident_id_assignment": "source_derived"}))
            with patch.object(
                PUBLICATION_SAFETY.subprocess, "run",
                side_effect=AssertionError("An archive must not invoke Git"),
            ):
                issues = PUBLICATION_SAFETY.scan_scenario_publication_safety(root)
            self.assertEqual(len(issues), 1)
            self.assertEqual(issues[0].path, "scenario_banks/example/source_manifest.json")

    def test_repository_does_not_hide_missing_git(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".git").mkdir()
            with patch.object(
                PUBLICATION_SAFETY.subprocess, "run", side_effect=FileNotFoundError("git"),
            ):
                with self.assertRaises(FileNotFoundError):
                    PUBLICATION_SAFETY.version_controlled_exclusions(root, [root / "case"])


if __name__ == "__main__":
    unittest.main()
