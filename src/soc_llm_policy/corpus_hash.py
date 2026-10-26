"""Canonical content identity for an incident-corpus directory tree."""

from __future__ import annotations

import hashlib
from pathlib import Path


def canonical_corpus_sha256(corpus_dir: Path) -> str:
    """Hash every file with an unambiguous, path-bound encoding."""
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"Corpus directory does not exist: {corpus_dir}")
    digest = hashlib.sha256()
    for path in sorted(corpus_dir.rglob("*")):
        if not path.is_file():
            continue
        digest.update(path.relative_to(corpus_dir).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
