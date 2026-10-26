"""Validate the deterministic evidence required before a provider run."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SHA256_LENGTH = 64


@dataclass(frozen=True)
class PolicyPreflightEvidence:
    # The file the reports were produced against, and the decidable surface they
    # actually read. The first identifies the bundle; the second gates the reports.
    constraints_sha256: str
    constraints_predicate_sha256: str
    derivation_corpus_sha256: str
    heldout_corpus_sha256: str
    derivation_incident_count: int
    heldout_incident_count: int
    raw_export_count: int


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != _SHA256_LENGTH:
        raise ValueError(f"{label} must be a SHA-256 digest.")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer.")
    return value


def validate_policy_preflight_reports(
    *,
    heldout_activation: dict[str, Any],
    partition_audit: dict[str, Any],
    normalization_ablation: dict[str, Any],
    expected_predicate_sha256: str,
) -> PolicyPreflightEvidence:
    """Verify that deterministic preflight reports describe one policy state.

    These reports read rule predicates and nothing else, so they bind to the
    predicate digest rather than to the bundle's file hash. Rewriting a rule's
    provenance prose leaves them valid, which is true; widening a scope by one
    pattern invalidates them, which is also true. Model evidence keeps binding
    to the file, because the prompt carried the prose.
    """
    expected_hash = _sha256(expected_predicate_sha256, "Expected predicate digest")
    activation = _mapping(heldout_activation, "Held-out activation report")
    partition = _mapping(partition_audit, "Corpus partition audit")
    normalization = _mapping(normalization_ablation, "Normalization ablation report")

    if activation.get("constraints_predicate_sha256") != expected_hash:
        raise ValueError(
            "Held-out activation report decides over different rule predicates "
            "than the selected policy bundle."
        )
    if normalization.get("constraints_predicate_sha256") != expected_hash:
        raise ValueError(
            "Normalization ablation report decides over different rule predicates "
            "than the selected policy bundle."
        )

    observed = _mapping(partition.get("observed"), "Corpus partition audit observed")
    derivation = _mapping(observed.get("derivation"), "Derivation partition")
    heldout = _mapping(observed.get("heldout"), "Held-out partition")
    activation_derivation_hash = _sha256(
        activation.get("derivation_corpus_sha256"),
        "Held-out activation derivation corpus hash",
    )
    activation_heldout_hash = _sha256(
        activation.get("heldout_corpus_sha256"),
        "Held-out activation held-out corpus hash",
    )
    if derivation.get("corpus_sha256") != activation_derivation_hash:
        raise ValueError(
            "Corpus partition audit does not match the activation derivation corpus."
        )
    if heldout.get("corpus_sha256") != activation_heldout_hash:
        raise ValueError(
            "Corpus partition audit does not match the activation held-out corpus."
        )

    derivation_count = _nonnegative_int(
        derivation.get("incident_count"), "Derivation incident count"
    )
    heldout_count = _nonnegative_int(
        heldout.get("incident_count"), "Held-out incident count"
    )
    if activation.get("derivation_incident_count") != derivation_count:
        raise ValueError(
            "Held-out activation derivation count does not match the partition audit."
        )
    if activation.get("heldout_incident_count") != heldout_count:
        raise ValueError(
            "Held-out activation count does not match the partition audit."
        )

    checks = _mapping(partition.get("checks"), "Corpus partition audit checks")
    lineage = _mapping(partition.get("source_lineage"), "Corpus source lineage")
    if checks.get("raw_exports_map_bijectively") is not True:
        raise ValueError(
            "Corpus partition audit did not prove one-to-one source lineage."
        )
    raw_export_count = _nonnegative_int(
        lineage.get("raw_export_count"), "Raw export count"
    )
    manifest_count = _nonnegative_int(
        lineage.get("canonical_source_manifest_count"),
        "Canonical source manifest count",
    )
    if raw_export_count != manifest_count or raw_export_count != (
        derivation_count + heldout_count
    ):
        raise ValueError(
            "Raw export and canonical partition counts do not form a bijection."
        )

    return PolicyPreflightEvidence(
        constraints_sha256=_sha256(
            activation.get("constraints_sha256"), "Activation constraints hash"
        ),
        constraints_predicate_sha256=expected_hash,
        derivation_corpus_sha256=activation_derivation_hash,
        heldout_corpus_sha256=activation_heldout_hash,
        derivation_incident_count=derivation_count,
        heldout_incident_count=heldout_count,
        raw_export_count=raw_export_count,
    )
