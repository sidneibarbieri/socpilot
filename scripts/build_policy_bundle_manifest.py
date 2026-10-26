"""Content-addressed identity for a policy bundle.

A policy bundle is the triple {action_catalog.yaml, constraints.yaml,
review_record.yaml}. Its identity is not a nominal generation label but the
SHA-256 of the canonical concatenation of those files: the policy_digest. The
manifest also records provenance -- the corpus the rules were derived from and
the held-out validation that tested them -- so any bundle is traceable to the
data and the decision that produced it.

Determinism: the digest is a pure function of the bundle files; identical files
yield an identical digest on any machine, so a bundle can be referenced by
digest instead of by a version label that loses meaning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from soc_llm_policy.paths import repo_relative_path, resolve_repo_root

# The files that constitute a bundle, in a fixed order so the combined digest
# is deterministic regardless of filesystem enumeration. The normalization
# table is part of the bundle: it defines what the content-triggered rules
# mean, so changing it must change the policy identity.
_BUNDLE_FILES = (
    "action_catalog.yaml",
    "constraints.yaml",
    "review_record.yaml",
    "telemetry_normalization.yaml",
)


class BundleFileDigest(BaseModel):
    filename: str = Field(..., min_length=1)
    sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    present: bool


class PolicyBundleManifest(BaseModel):
    schema_version: int = 1
    generated_at_utc: str
    bundle_path: str
    policy_digest: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    files: list[BundleFileDigest]
    provenance: dict[str, str]


# Resolve the string annotations from ``from __future__ import annotations`` so
# the model works whether imported by a test or run as a script.
PolicyBundleManifest.model_rebuild()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def build_manifest(
    *,
    repo_root: Path,
    bundle_dir: Path,
    provenance: dict[str, str],
) -> PolicyBundleManifest:
    files: list[BundleFileDigest] = []
    combined = hashlib.sha256()
    for name in _BUNDLE_FILES:
        path = bundle_dir / name
        present = path.is_file()
        payload = path.read_bytes() if present else b""
        # Bind the filename into the combined digest so a rename cannot preserve
        # identity, and mark absence explicitly rather than silently skipping.
        combined.update(name.encode())
        combined.update(b"\x00")
        combined.update(payload)
        combined.update(b"\x00")
        files.append(
            BundleFileDigest(
                filename=name,
                sha256=_sha256_bytes(payload),
                present=present,
            )
        )
    return PolicyBundleManifest(
        generated_at_utc=datetime.now(UTC).isoformat(),
        bundle_path=repo_relative_path(bundle_dir, repo_root),
        policy_digest=combined.hexdigest(),
        files=files,
        provenance=provenance,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="build_policy_bundle_manifest",
        description="Compute the content-addressed policy_digest and provenance "
        "for a policy bundle.",
    )
    parser.add_argument("--repo-root", default=None)
    parser.add_argument(
        "--bundle-dir",
        default="policy",
    )
    parser.add_argument("--derived-from-corpus-sha256", default="")
    parser.add_argument("--heldout-validation-sha256", default="")
    parser.add_argument("--decision-ref", default="policy/review_record.yaml")
    parser.add_argument("--output-json", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    repo_root = resolve_repo_root(args.repo_root)
    provenance = {
        "derived_from_corpus_sha256": args.derived_from_corpus_sha256,
        "heldout_validation_sha256": args.heldout_validation_sha256,
        "decision_ref": args.decision_ref,
    }
    manifest = build_manifest(
        repo_root=repo_root,
        bundle_dir=(repo_root / str(args.bundle_dir)).resolve(),
        provenance=provenance,
    )
    output = (repo_root / str(args.output_json)).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"policy_digest: {manifest.policy_digest}")
    print(f"Wrote manifest: {output}")


if __name__ == "__main__":
    main()
