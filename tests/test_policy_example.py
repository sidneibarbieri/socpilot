"""Exercise the documented library example through its command-line entry point."""

import json
import os
import subprocess
import sys
from pathlib import Path

from soc_llm_policy.policy_example import run_example


def test_example_checks_proposal_and_repaired_plan_separately():
    report = run_example()
    assert report["constitutes_model_evidence"] is False
    assert report["proposed_actions"] == ["reimage_host", "isolate_host"]
    assert report["repaired_actions"] == ["collect_forensics", "reimage_host"]
    assert sum(outcome["violated"] for outcome in report["before"]) == 2
    assert not any(outcome["violated"] for outcome in report["after"])
    assert len(report["edits"]) == 2
    assert report["second_pass_unchanged"] is True


def test_example_cli_needs_no_credentials_and_writes_no_files(tmp_path):
    source = Path(__file__).resolve().parents[1] / "src"
    completed = subprocess.run(
        [sys.executable, "-B", "-m", "soc_llm_policy.policy_example"],
        cwd=tmp_path,
        env={"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(source)},
        check=True, capture_output=True, text=True,
    )
    assert json.loads(completed.stdout) == run_example()
    assert completed.stderr == ""
    assert list(tmp_path.iterdir()) == []
