from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from soc_llm_policy.evaluation_plan import (
    RunLease,
    RunLedger,
    RunMatrixDesign,
    build_run_matrix,
    run_matrix_manifest,
)


def _single_cell_matrix(run_id: str = "exp_resume", repeats: int = 1) -> list:
    return build_run_matrix(
        RunMatrixDesign(
            run_id=run_id,
            dataset_release_id="release_a",
            eval_protocol_version="fixture_protocol",
            incidents=("INC_001", "INC_002"),
            models=(("model_a", "dep_a", "openai"),),
            arms=("llm_zero",),
            repeats=repeats,
            execution_order="sequential",
            seed=0,
        )
    )


def test_paired_blocked_plan_is_deterministic_and_complete() -> None:
    design = RunMatrixDesign(
        run_id="exp_001",
        dataset_release_id="release_a",
        eval_protocol_version="fixture_protocol",
        incidents=("INC_001", "INC_002"),
        models=(
            ("model_a", "dep_a", "openai"),
            ("model_b", "dep_b", "anthropic"),
        ),
        arms=("llm_zero", "llm_policy_prompt"),
        repeats=2,
        execution_order="paired_blocked",
        seed=17,
    )

    first = build_run_matrix(design)
    second = build_run_matrix(design)

    expected_cell_count = 16
    assert first == second
    assert len(first) == expected_cell_count
    assert len({cell.cell_id for cell in first}) == expected_cell_count
    assert [cell.order_index for cell in first] == list(
        range(1, expected_cell_count + 1)
    )

    for start in range(0, len(first), 4):
        block = first[start : start + 4]
        assert len({cell.incident_id for cell in block}) == 1
        assert len({cell.repeat_index for cell in block}) == 1
        assert {(cell.model_name, cell.arm) for cell in block} == {
            ("model_a", "llm_zero"),
            ("model_a", "llm_policy_prompt"),
            ("model_b", "llm_zero"),
            ("model_b", "llm_policy_prompt"),
        }


def test_matrix_seed_changes_order_not_identity() -> None:
    def design(seed: int) -> RunMatrixDesign:
        return RunMatrixDesign(
            run_id="exp_001",
            dataset_release_id="release_a",
            eval_protocol_version="fixture_protocol",
            incidents=("INC_001", "INC_002"),
            models=(
                ("model_a", "dep_a", "openai"),
                ("model_b", "dep_b", "anthropic"),
            ),
            arms=("llm_zero", "llm_policy_prompt"),
            repeats=1,
            execution_order="global_randomized",
            seed=seed,
        )

    first = build_run_matrix(design(1))
    second = build_run_matrix(design(2))

    assert [cell.cell_id for cell in first] != [cell.cell_id for cell in second]
    assert {cell.cell_id for cell in first} == {cell.cell_id for cell in second}


def test_policy_bundle_changes_cell_identity() -> None:
    def design(policy_bundle_sha256: str) -> RunMatrixDesign:
        return RunMatrixDesign(
            run_id="exp_001",
            dataset_release_id="release_a",
            eval_protocol_version="fixture_protocol",
            incidents=("INC_001",),
            models=(("model_a", "dep_a", "openai"),),
            arms=("llm_zero",),
            repeats=1,
            execution_order="sequential",
            seed=0,
            policy_bundle_sha256=policy_bundle_sha256,
        )

    first = build_run_matrix(design("a" * 64))[0]
    second = build_run_matrix(design("b" * 64))[0]

    assert first.cell_id != second.cell_id
    assert first.policy_bundle_sha256 == "a" * 64
    assert second.policy_bundle_sha256 == "b" * 64


def test_matrix_preserves_corpus_strata_in_cell_identity() -> None:
    design = RunMatrixDesign(
        run_id="exp_partitioned",
        dataset_release_id="release_a",
        eval_protocol_version="fixture_protocol",
        incidents=("INC_001", "INC_002"),
        models=(("model_a", "dep_a", "openai"),),
        arms=("llm_zero",),
        repeats=1,
        execution_order="sequential",
        seed=0,
        incident_partitions=(
            ("INC_001", "export_batch_1"),
            ("INC_002", "export_batch_2"),
        ),
    )

    cells = build_run_matrix(design)
    manifest = run_matrix_manifest(cells, execution_order="sequential", seed=0)

    assert [cell.corpus_stratum for cell in cells] == [
        "export_batch_1",
        "export_batch_2",
    ]
    assert manifest["corpus_stratum_cell_counts"] == {
        "export_batch_1": 1,
        "export_batch_2": 1,
    }


def test_matrix_rejects_incomplete_corpus_partition_assignments() -> None:
    design = RunMatrixDesign(
        run_id="exp_partitioned",
        dataset_release_id="release_a",
        eval_protocol_version="fixture_protocol",
        incidents=("INC_001", "INC_002"),
        models=(("model_a", "dep_a", "openai"),),
        arms=("llm_zero",),
        repeats=1,
        execution_order="sequential",
        seed=0,
        incident_partitions=(("INC_001", "export_batch_1"),),
    )

    with pytest.raises(ValueError, match="must match the run matrix incidents"):
        build_run_matrix(design)


def test_manifest_hash_covers_cells_not_schedule_metadata() -> None:
    cells = build_run_matrix(
        RunMatrixDesign(
            run_id="exp_001",
            dataset_release_id="release_a",
            eval_protocol_version="fixture_protocol",
            incidents=("INC_001",),
            models=(("model_a", "dep_a", "openai"),),
            arms=("llm_zero",),
            repeats=1,
            execution_order="sequential",
            seed=0,
        )
    )

    first = run_matrix_manifest(cells, execution_order="sequential", seed=0)
    second = run_matrix_manifest(cells, execution_order="sequential", seed=1)

    assert first["plan_sha256"] == second["plan_sha256"]
    assert first["seed"] != second["seed"]


def test_ledger_persists_cell_state_atomically(tmp_path: Path) -> None:
    cells = build_run_matrix(
        RunMatrixDesign(
            run_id="exp_001",
            dataset_release_id="release_a",
            eval_protocol_version="fixture_protocol",
            incidents=("INC_001",),
            models=(("model_a", "dep_a", "openai"),),
            arms=("llm_zero",),
            repeats=1,
            execution_order="sequential",
            seed=0,
        )
    )
    path = tmp_path / "ledger.json"
    ledger = RunLedger(path, cells)

    execution_id = ledger.mark_running(cells[0])
    ledger.mark_completed(cells[0], execution_id)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status_counts"]["completed"] == 1
    assert payload["cells"][0]["execution_count"] == 1
    assert payload["cells"][0]["status"] == "completed"


def test_resume_skips_completed_and_reopens_interrupted(tmp_path: Path) -> None:
    cells = _single_cell_matrix()
    path = tmp_path / "ledger.json"

    first = RunLedger(path, cells, worker_id="worker_a")
    done_id = first.mark_running(cells[0])
    first.mark_completed(cells[0], done_id)
    # The second cell begins but its worker dies before recording a terminal state.
    first.mark_running(cells[1])

    resumed = RunLedger(path, cells, resume=True, worker_id="worker_b")

    assert resumed.status(cells[0]) == "completed"
    assert resumed.completed_cells(cells) == [cells[0]]
    # The interrupted cell reopens for another attempt; its dead attempt is recorded.
    assert resumed.status(cells[1]) == "planned"
    assert resumed.exception_type(cells[1]) == "InterruptedExecution"

    with pytest.raises(ValueError, match="Terminal cell cannot be re-executed"):
        resumed.mark_running(cells[0])

    retry_id = resumed.mark_running(cells[1])
    resumed.mark_completed(cells[1], retry_id)
    payload = json.loads(path.read_text(encoding="utf-8"))
    interrupted = payload["cells"][1]["executions"]
    assert [execution["status"] for execution in interrupted] == [
        "interrupted",
        "completed",
    ]
    assert interrupted[-1]["worker_id"] == "worker_b"


def test_resume_preserves_a_terminal_refusal(tmp_path: Path) -> None:
    cells = _single_cell_matrix()
    path = tmp_path / "ledger.json"
    first = RunLedger(path, cells, worker_id="worker_a")
    attempt_id = first.mark_running(cells[0])
    first.mark_terminal_observation(
        cells[0], attempt_id, "refused", RuntimeError("model refusal")
    )

    resumed = RunLedger(path, cells, resume=True, worker_id="worker_b")

    assert resumed.status(cells[0]) == "refused"
    assert resumed.terminal_observation_cells(cells) == [cells[0]]
    with pytest.raises(ValueError, match="Terminal cell cannot be re-executed"):
        resumed.mark_running(cells[0])


def test_resume_rejects_a_matrix_that_drifted_from_the_frozen_plan(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ledger.json"
    RunLedger(path, _single_cell_matrix(), worker_id="worker_a")

    drifted = _single_cell_matrix(run_id="exp_other")
    with pytest.raises(ValueError, match="differ from the fixed run matrix"):
        RunLedger(path, drifted, resume=True, worker_id="worker_b")


def test_lease_denies_a_second_live_worker_and_allows_a_stale_takeover(
    tmp_path: Path,
) -> None:
    path = tmp_path / "run_lease.json"
    holder = RunLease(path, "exp_resume", lease_seconds=60, allow_stale_takeover=False)
    holder.__enter__()

    contender = RunLease(
        path, "exp_resume", lease_seconds=60, allow_stale_takeover=False
    )
    with pytest.raises(FileExistsError, match="owned by worker"):
        contender.__enter__()

    # Expiry must not permit replacement while the owner is alive.
    lease_state = json.loads(path.read_text(encoding="utf-8"))
    lease_state["lease_expires_at_unix"] = 0.0
    path.write_text(json.dumps(lease_state), encoding="utf-8")
    with pytest.raises(FileExistsError, match="owned by worker"):
        RunLease(
            path, "exp_resume", lease_seconds=60, allow_stale_takeover=False
        ).__enter__()

    successor = RunLease(
        path, "exp_resume", lease_seconds=60, allow_stale_takeover=True
    )
    with pytest.raises(FileExistsError, match="owned by worker"):
        successor.__enter__()
    holder.release()
    # Use a reaped child's PID to represent a stopped worker.
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait(timeout=5)
    lease_state["pid"] = child.pid
    path.write_text(json.dumps(lease_state), encoding="utf-8")
    successor.__enter__()
    assert (
        json.loads(path.read_text(encoding="utf-8"))["worker_id"] == successor.worker_id
    )
    successor.release()


def test_lease_rejects_stale_live_owner_without_kernel_lock(tmp_path):
    path = tmp_path / "run_lease.json"
    path.write_text(json.dumps({
        "worker_id": "old_worker", "hostname": socket.gethostname(),
        "pid": os.getpid(), "lease_expires_at_unix": 0,
    }))
    contender = RunLease(path, "run", lease_seconds=60, allow_stale_takeover=True)
    with pytest.raises(FileExistsError, match="still alive"):
        contender.__enter__()
    assert contender._lock_descriptor is None


def test_lease_cleans_up_lock_on_corrupt_state(tmp_path):
    path = tmp_path / "run_lease.json"
    path.write_text("invalid json")
    contender = RunLease(path, "run", lease_seconds=60, allow_stale_takeover=True)
    with pytest.raises(json.JSONDecodeError):
        contender.__enter__()
    assert contender._lock_descriptor is None
    path.unlink()
    with contender:
        assert path.exists()


def test_expired_lease_remains_exclusive_across_processes(tmp_path):
    path = tmp_path / "run_lease.json"
    child_code = """
import sys
from pathlib import Path
from soc_llm_policy.evaluation_plan import RunLease
try:
    with RunLease(Path(sys.argv[1]), 'run', lease_seconds=60, allow_stale_takeover=True):
        sys.exit(1)
except FileExistsError:
    sys.exit(0)
"""
    with RunLease(path, "run", lease_seconds=60, allow_stale_takeover=False):
        state = json.loads(path.read_text())
        state["lease_expires_at_unix"] = 0
        path.write_text(json.dumps(state))
        subprocess.run([sys.executable, "-c", child_code, str(path)], check=True, timeout=10)


def test_lease_rejects_a_horizon_below_the_floor(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least"):
        RunLease(
            tmp_path / "run_lease.json",
            "exp_resume",
            lease_seconds=1,
            allow_stale_takeover=False,
        )
