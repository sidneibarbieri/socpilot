"""Content-addressed identity for the policy inputs of one evaluation cell."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from soc_llm_policy.paths import repo_relative_path
from soc_llm_policy.proportionality import load_proportionality_policy
from soc_llm_policy.result_models import (
    PolicyBundleReference,
    PolicyInputReference,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_policy_bundle(
    *,
    repo_root: Path,
    action_catalog_path: Path,
    constraints_path: Path,
    proportionality_policy_path: Path | None = None,
) -> PolicyBundleReference:
    inputs = [
        PolicyInputReference(
            role="action_catalog",
            path=repo_relative_path(action_catalog_path, repo_root),
            sha256=_sha256_file(action_catalog_path),
            review_status="declared",
        ),
        PolicyInputReference(
            role="policy_rules",
            path=repo_relative_path(constraints_path, repo_root),
            sha256=_sha256_file(constraints_path),
            review_status="declared",
        ),
    ]
    if proportionality_policy_path is not None:
        policy = load_proportionality_policy(proportionality_policy_path)
        inputs.append(
            PolicyInputReference(
                role="proportionality_policy",
                path=repo_relative_path(proportionality_policy_path, repo_root),
                sha256=_sha256_file(proportionality_policy_path),
                policy_id=policy.policy_id,
                review_status=policy.review_status,
            )
        )

    identity = [
        {
            "role": item.role,
            "sha256": item.sha256,
        }
        for item in sorted(inputs, key=lambda item: item.role)
    ]
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return PolicyBundleReference(
        bundle_sha256=hashlib.sha256(encoded).hexdigest(),
        inputs=inputs,
    )
