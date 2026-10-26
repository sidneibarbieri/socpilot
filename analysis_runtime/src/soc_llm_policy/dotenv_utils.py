from __future__ import annotations

import os
from pathlib import Path

from soc_llm_policy.paths import resolve_repo_root

_DOTENV_OVERRIDE = "SOC_LLM_POLICY_DOTENV_PATH"


def iter_dotenv_candidates(repo_root: Path | None = None) -> list[Path]:
    root = repo_root.resolve() if repo_root is not None else resolve_repo_root(None)
    override = os.environ.get(_DOTENV_OVERRIDE, "").strip()
    if override:
        return [Path(override).expanduser().resolve()]
    return [(root / ".env").resolve()]


def find_dotenv_path(repo_root: Path | None = None) -> Path | None:
    for candidate in iter_dotenv_candidates(repo_root):
        if candidate.is_file():
            return candidate
    return None


def load_project_dotenv(repo_root: Path | None = None) -> Path | None:
    try:
        from dotenv import load_dotenv  # noqa: PLC0415
    except ImportError:
        return None

    dotenv_path = find_dotenv_path(repo_root)
    if dotenv_path is None:
        return None
    load_dotenv(dotenv_path=dotenv_path, override=False)
    return dotenv_path
