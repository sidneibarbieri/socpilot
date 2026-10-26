from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_provider_plan import _write_draft_plan

from soc_llm_policy import execution_backup
from soc_llm_policy.corpus_campaign import build_campaign, validate_campaign
from soc_llm_policy.evaluation_plan import RunLease
from soc_llm_policy.execution_backup import create_backup, restore_backup, verify_backup
from soc_llm_policy.preflight_evidence import sha256_file


@pytest.fixture
def workspace(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "source"
    root.mkdir()
    (root / "run.sh").write_text("#!/bin/sh\nexit 0\n")
    (root / "run.sh").chmod(0o755)
    outputs = root / "artifact_outputs"
    outputs.mkdir()
    source_manifest = outputs / "execution_source_manifest.json"
    source_manifest.write_text(json.dumps({
        "schema_version": 1, "state": "frozen_execution_source", "file_count": 1,
        "files": [{"path": "run.sh", "sha256": sha256_file(root / "run.sh")}],
    }))
    freeze = outputs / "protocol_freeze.json"
    freeze.write_text(json.dumps({"input_hashes": {"execution_source_manifest": {
        "path": source_manifest.relative_to(root).as_posix(), "sha256": sha256_file(source_manifest),
    }}}))
    plan = outputs / "plan.json"
    _write_draft_plan(plan)
    payload = json.loads(plan.read_text())
    payload["frozen_inputs"]["protocol_freeze"] = {
        "path": freeze.relative_to(root).as_posix(), "sha256": sha256_file(freeze),
    }
    plan.write_text(json.dumps(payload))
    (root / "artifact_manifest.json").write_text(json.dumps({"files": [{"path": "run.sh"}]}))
    return root, plan


def test_round_trip_preserves_bytes_modes_and_excludes_unlisted_credentials(workspace, tmp_path):
    root, plan = workspace
    (root / ".env").write_text("EXAMPLE_SECRET=not-for-backup")
    (root / ".venv").mkdir()
    (root / ".venv/marker").write_text("unlisted runtime")
    backup = tmp_path / "backup"
    manifest = create_backup(root, plan, backup)
    digest = sha256_file(manifest)
    report = verify_backup(backup, digest)
    assert report.different_filesystem is False
    assert report.plan_sha256 == sha256_file(plan)
    restored = tmp_path / "restored"
    restore_backup(backup, digest, restored)
    assert not (restored / ".env").exists()
    assert not (restored / ".venv").exists()
    assert (restored / "run.sh").stat().st_mode & 0o777 == 0o755
    for entry in report.files:
        assert (root / entry.path).read_bytes() == (restored / entry.path).read_bytes()
    assert backup.stat().st_mode & 0o777 == 0o700
    assert sha256_file(plan) == report.plan_sha256


@pytest.mark.parametrize("change", ["content", "manifest", "extra", "missing"])
def test_corruption_is_rejected_before_restore(workspace, tmp_path, change):
    root, plan = workspace
    backup = tmp_path / "backup"
    manifest = create_backup(root, plan, backup)
    digest = sha256_file(manifest)
    if change == "content":
        (backup / "workspace/run.sh").write_text("changed")
    elif change == "manifest":
        manifest.write_text("{}")
    elif change == "missing":
        (backup / "workspace/run.sh").unlink()
    else:
        (backup / "workspace/extra.json").write_text("{}")
    with pytest.raises(ValueError):
        restore_backup(backup, digest, tmp_path / "restored")
    assert not (tmp_path / "restored").exists()


def test_backup_and_restore_never_overwrite(workspace, tmp_path):
    root, plan = workspace
    backup = tmp_path / "backup"
    manifest = create_backup(root, plan, backup)
    digest = sha256_file(manifest)
    with pytest.raises(FileExistsError):
        create_backup(root, plan, backup)
    with pytest.raises(FileExistsError):
        restore_backup(backup, digest, root)
    assert sha256_file(manifest) == digest


@pytest.mark.parametrize("relative", ["../escape", "/escape", "artifact_outputs/.env", "./run.sh"])
def test_unsafe_inventory_is_rejected(workspace, tmp_path, relative):
    root, plan = workspace
    (root / "artifact_manifest.json").write_text(json.dumps({"files": [{"path": relative}]}))
    with pytest.raises(ValueError):
        create_backup(root, plan, tmp_path / "backup")
    assert not (tmp_path / "backup").exists()


def test_symbolic_link_in_outputs_is_rejected(workspace, tmp_path):
    root, plan = workspace
    (root / "artifact_outputs/link").symlink_to(root / "run.sh")
    with pytest.raises(ValueError, match="symbolic link"):
        create_backup(root, plan, tmp_path / "backup")


def test_active_worker_is_not_interrupted(workspace, tmp_path):
    root, plan = workspace
    lease_path = root / "artifact_outputs/experiments/official_campaign/run_lease.json"
    with RunLease(lease_path, "official_campaign", lease_seconds=900, allow_stale_takeover=False):
        before = lease_path.read_bytes()
        with pytest.raises(RuntimeError, match="Worker is active"):
            create_backup(root, plan, tmp_path / "backup")
        assert lease_path.read_bytes() == before
    assert not (tmp_path / "backup").exists()


def test_concurrent_mutation_retains_incomplete_backup(workspace, tmp_path, monkeypatch):
    root, plan = workspace
    original = execution_backup._copy_file

    def copy_and_mutate(source, target, relative, *, scan_keys):
        result = original(source, target, relative, scan_keys=scan_keys)
        if relative == "run.sh":
            source.write_text("changed concurrently")
        return result

    monkeypatch.setattr(execution_backup, "_copy_file", copy_and_mutate)
    with pytest.raises(RuntimeError, match="Source changed"):
        create_backup(root, plan, tmp_path / "backup")
    assert (tmp_path / "backup").is_dir()
    assert not (tmp_path / "backup/backup_manifest.json").exists()


def test_key_shaped_content_is_not_silently_archived(workspace, tmp_path):
    root, plan = workspace
    (root / "artifact_outputs/accidental.log").write_text("sk-" + "proj-" + "x" * 40)
    with pytest.raises(ValueError, match="Possible credential"):
        create_backup(root, plan, tmp_path / "backup")
    assert not (tmp_path / "backup/backup_manifest.json").exists()


def test_backup_scanner_accepts_its_own_source(workspace, tmp_path):
    root, plan = workspace
    source = Path(execution_backup.__file__).read_bytes()
    (root / "artifact_outputs/scanner.py").write_bytes(source)
    manifest = create_backup(root, plan, tmp_path / "backup")
    verify_backup(tmp_path / "backup", sha256_file(manifest))


@pytest.mark.parametrize("key_type", ["", "RSA ", "EC ", "DSA ", "OPENSSH ", "ENCRYPTED "])
@pytest.mark.parametrize("offset", [0, 65520])
def test_private_key_header_is_rejected_across_chunks(workspace, tmp_path, key_type, offset):
    root, plan = workspace
    header = "-----BEGIN " + key_type + "PRIVATE KEY-----"
    (root / "artifact_outputs/accidental.log").write_text(" " * offset + header)
    with pytest.raises(ValueError, match="Possible credential"):
        create_backup(root, plan, tmp_path / "backup")
    assert not (tmp_path / "backup/backup_manifest.json").exists()


def test_changed_scientific_source_is_rejected(workspace, tmp_path):
    root, plan = workspace
    (root / "run.sh").write_text("changed before backup")
    with pytest.raises(ValueError, match="hash changed"):
        create_backup(root, plan, tmp_path / "backup")
    assert not (tmp_path / "backup").exists()


def test_nested_backup_destination_is_rejected(workspace):
    root, plan = workspace
    with pytest.raises(ValueError, match="outside"):
        create_backup(root, plan, root / "backup")


def _add_corpus(root: Path, plan: Path) -> Path:
    partitions = {"first": root / "artifact_data/first", "second": root / "artifact_data/second"}
    source_inventory = json.loads((root / "artifact_manifest.json").read_text())
    for label, directory in partitions.items():
        incident = directory / f"INC_{label}"
        incident.mkdir(parents=True)
        for name in ("incident_meta.json", "incident_human_actions.jsonl", "incident_telemetry.jsonl"):
            path = incident / name
            path.write_text("{}\n")
            source_inventory["files"].append({"path": path.relative_to(root).as_posix()})
    (root / "artifact_manifest.json").write_text(json.dumps(source_inventory))
    manifest = root / "artifact_outputs/corpus_campaign.json"
    build_campaign(repo_root=root, partitions=partitions,
                   view_path=root / "artifact_outputs/corpus", manifest_path=manifest)
    payload = json.loads(plan.read_text())
    payload["frozen_inputs"]["corpus_campaign_manifest"] = {
        "path": manifest.relative_to(root).as_posix(), "sha256": sha256_file(manifest),
    }
    plan.write_text(json.dumps(payload))
    return manifest


def test_restore_rebuilds_only_validated_corpus_links(workspace, tmp_path):
    root, plan = workspace
    corpus_manifest = _add_corpus(root, plan)
    backup = tmp_path / "backup"
    manifest = create_backup(root, plan, backup)
    digest = sha256_file(manifest)
    assert not (backup / "workspace/artifact_outputs/corpus").exists()
    restored = tmp_path / "restored"
    restore_backup(backup, digest, restored)
    validate_campaign(repo_root=restored, manifest_path=restored / corpus_manifest.relative_to(root))
    for label in ("first", "second"):
        link = restored / "artifact_outputs/corpus" / f"INC_{label}"
        assert link.is_symlink()
        assert link.resolve() == restored / f"artifact_data/{label}/INC_{label}"


@pytest.mark.parametrize("change", ["extra", "wrong_target"])
def test_corpus_view_is_not_a_general_symlink_exception(workspace, tmp_path, change):
    root, plan = workspace
    _add_corpus(root, plan)
    view = root / "artifact_outputs/corpus"
    if change == "extra":
        (view / "extra").write_text("not an incident")
    else:
        (view / "INC_first").unlink()
        (view / "INC_first").symlink_to(root / "run.sh")
    with pytest.raises(ValueError):
        create_backup(root, plan, tmp_path / "backup")
