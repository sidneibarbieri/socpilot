"""Reject disposable workspace locations before any experimental model call."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path


def require_persistent_workspace(workspace: Path) -> None:
    root = workspace.expanduser().resolve()
    temporary_roots = {Path("/tmp"), Path("/var/tmp"), Path(tempfile.gettempdir())}
    if any(root.is_relative_to(candidate.resolve()) for candidate in temporary_roots):
        raise ValueError(
            "Model execution requires a persistent local workspace outside temporary "
            "directories. Extract the source archive there and prepare the plan again. "
            "Temporary directories are allowed only for zero-cost validation."
        )


if __name__ == "__main__":
    require_persistent_workspace(Path(sys.argv[1]))
