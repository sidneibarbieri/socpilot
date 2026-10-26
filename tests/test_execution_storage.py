from pathlib import Path

import pytest

from soc_llm_policy.execution_storage import require_persistent_workspace


@pytest.mark.parametrize("directory", ["/tmp/run", "/var/tmp/run", "/private/tmp/run"])
def test_temporary_execution_is_rejected(directory):
    if directory.startswith("/private") and not Path("/private/tmp").exists():
        pytest.skip("macOS temporary-directory alias")
    with pytest.raises(ValueError, match="persistent local workspace"):
        require_persistent_workspace(Path(directory))


def test_system_temp_execution_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="persistent local workspace"):
        require_persistent_workspace(tmp_path)


def test_non_temporary_workspace_is_allowed():
    require_persistent_workspace(Path.home() / "research-runs" / "protocol-digest")


def test_symlink_into_temporary_storage_is_rejected(tmp_path):
    alias = tmp_path / "alias"
    target = tmp_path / "run"
    target.mkdir()
    alias.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="persistent local workspace"):
        require_persistent_workspace(alias)
