from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

from soc_llm_policy.preflight_evidence import sha256_file
from soc_llm_policy.publication_redaction import expected_digest

PROVIDER_INPUT_ROLES = (
    "action_catalog",
    "constraints",
    "policy_review_record",
    "models_registry",
)
_SHA256_LENGTH = 64


@dataclass(frozen=True)
class FrozenInputBinding:
    path: str
    sha256: str
    resolved_path: Path


def resolve_frozen_input_bindings(
    input_hashes: object,
    repo_root: Path,
    *,
    required_roles: Collection[str] = (),
) -> dict[str, FrozenInputBinding]:
    if not isinstance(input_hashes, dict):
        raise ValueError("Frozen input hashes must be a JSON object.")

    missing_roles = sorted(set(required_roles) - set(input_hashes))
    if missing_roles:
        raise ValueError("Frozen inputs are missing roles: " + ", ".join(missing_roles))

    resolved_root = repo_root.expanduser().resolve()
    bindings: dict[str, FrozenInputBinding] = {}
    for role, raw_entry in sorted(input_hashes.items()):
        if not isinstance(role, str) or not role:
            raise ValueError("Frozen input roles must be non-empty strings.")
        if not isinstance(raw_entry, dict):
            raise ValueError(f"Frozen input entry must be an object: {role}.")

        raw_path = raw_entry.get("path")
        expected_hash = raw_entry.get("sha256")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"Frozen input path is invalid: {role}.")
        if not _is_sha256(expected_hash):
            raise ValueError(f"Frozen input SHA-256 is invalid: {role}.")

        relative_path = Path(raw_path)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(
                f"Frozen input path must stay inside the repository: {role}."
            )
        resolved_path = (resolved_root / relative_path).resolve()
        if (
            resolved_path != resolved_root
            and resolved_root not in resolved_path.parents
        ):
            raise ValueError(f"Frozen input path escapes the repository: {role}.")
        if not resolved_path.is_file():
            raise FileNotFoundError(
                f"Frozen input is unavailable: {role} -> {resolved_path}"
            )
        if sha256_file(resolved_path) != expected_digest(
            resolved_root, resolved_path, expected_hash
        ):
            raise ValueError(f"Frozen input hash changed: {role}.")

        bindings[role] = FrozenInputBinding(
            path=relative_path.as_posix(),
            sha256=expected_hash,
            resolved_path=resolved_path,
        )
    return bindings


def require_active_input_paths(
    bindings: dict[str, FrozenInputBinding],
    active_paths: dict[str, Path],
) -> None:
    for role, active_path in sorted(active_paths.items()):
        binding = bindings.get(role)
        if binding is None:
            raise ValueError(f"Active input has no frozen binding: {role}.")
        if active_path.expanduser().resolve() != binding.resolved_path:
            raise ValueError(f"Active input differs from the protocol freeze: {role}.")


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != _SHA256_LENGTH:
        return False
    return all(character in "0123456789abcdef" for character in value.lower())
