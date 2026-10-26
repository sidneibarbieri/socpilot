"""Publish incident evidence as its projection onto the declared vocabulary.

Every text field an operator, a detector or a model wrote inside the tenant can
carry institutional residue: asset names, privileged accounts, internal log
sources, ticket numbers. No pattern list enumerates all of it. This module
therefore does not look for residue to remove; it keeps only what the policy
already declares and withholds the rest.

The one field the verifier reads, the event ``category``, is replaced by the
declared patterns it contains. Scope predicates are substring tests over those
same patterns, and no declared pattern contains a space, so a pattern occurs in
the projection exactly when it occurred in the original: every verdict is
preserved by construction. Every other free-text field is withheld, and a
redaction record binds each published file to the hash of the file it came
from, so the evidence chain stays checkable up to the privacy boundary.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict

from soc_llm_policy.case_note_kinds import (
    classify_case_note,
    parse_publishable_note_text,
    publishable_note_text,
)
from soc_llm_policy.io import TelemetryDetails, TelemetryEvent
from soc_llm_policy.policy_predicates import predicate_digest
from soc_llm_policy.publication_correlation import (
    CorrelationProjection,
    project_loose_files,
    project_package_trees,
)
from soc_llm_policy.publication_residue import scan_string

TRANSFORM_ID = "declared-vocabulary-projection/v1"
WITHHELD = "<withheld>"
NO_DECLARED_PATTERN = "<no declared pattern>"
SLUG_SEPARATOR = "_"
RECORD_RELATIVE_PATH = Path("artifact_outputs/publication/redaction_record.json")

TELEMETRY_FILENAME = "incident_telemetry.jsonl"
HUMAN_ACTIONS_FILENAME = "incident_human_actions.jsonl"
SOURCE_MANIFEST_FILENAME = "source_manifest.json"
CASE_NOTE = "case_note"
SYNTHETIC_SOURCE = "SYNTHETIC_EVALUATION"
PSEUDONYM_HOST_PATTERN = re.compile(r"host_[0-9]+")
PSEUDONYM_USER_PATTERN = re.compile(r"user_[0-9]+")

# Output fields that hold tenant text or text a model wrote about it.
WITHHELD_OUTPUT_KEYS = frozenset({"reasoning", "evidence"})
PROMPT_MESSAGES_KEY = "prompt_messages"
TELEMETRY_PREVIEW_KEY = "telemetry_preview"
WITHHELD_FIELDS = (
    "category, replaced by its declared-pattern projection and a stable rank",
    "details.command of detection events",
    "details.source_summary text of detection events",
    "case-note prose, replaced by a declared kind label",
    "username unless it is a declared pseudonym",
    "details.log_source text",
    "output fields evidence, reasoning and prompt_messages content",
)
DATA_TREES = ("artifact_data", "scenario_banks")
OUTPUT_TREES = ("artifact_outputs",)


class RedactionVocabulary(BaseModel):
    """The category patterns the policy and the normalization instrument declare."""

    model_config = ConfigDict(frozen=True)

    patterns: tuple[str, ...]
    public_aliases: tuple[tuple[str, str], ...] = ()

    def public_pattern(self, source_pattern: str) -> str:
        return dict(self.public_aliases).get(source_pattern, source_pattern)

    @classmethod
    def from_policy(
        cls,
        constraints_path: Path,
        normalization_path: Path,
        public_aliases: tuple[tuple[str, str], ...] = (),
    ) -> RedactionVocabulary:
        patterns: set[str] = set()
        for rule in yaml.safe_load(constraints_path.read_text(encoding="utf-8")):
            patterns.update((rule.get("scope") or {}).get("category_contains") or [])
        # A package without the normalization instrument has no analysis that
        # reads its patterns, so the rules alone declare the vocabulary.
        if normalization_path.is_file():
            normalization = yaml.safe_load(
                normalization_path.read_text(encoding="utf-8")
            )
            for behavior in normalization["behavior_classes"]:
                patterns.update(behavior["category_patterns"])
        vocabulary = cls(
            patterns=tuple(sorted(pattern.lower() for pattern in patterns)),
            public_aliases=public_aliases,
        )
        vocabulary.require_projection_is_exact()
        return vocabulary

    def require_projection_is_exact(self) -> None:
        """Refuse a vocabulary under which the projection could invent a match."""
        aliases = dict(self.public_aliases)
        if len(aliases) != len(self.public_aliases) or set(aliases) - set(
            self.patterns
        ):
            raise ValueError("Aliases must name unique declared source patterns")
        if any(
            scan_string(pattern, key="category") and pattern not in aliases
            for pattern in self.patterns
        ):
            raise ValueError("Declared category vocabulary contains residue")
        public_patterns = tuple(
            self.public_pattern(pattern) for pattern in self.patterns
        )
        if len(set(public_patterns)) != len(public_patterns):
            raise ValueError("Published category aliases overlap declared patterns")
        if any(scan_string(pattern, key="category") for pattern in public_patterns):
            raise ValueError("Published category aliases contain publication residue")
        for source_pattern, alias in self.public_aliases:
            if any(
                other != source_pattern
                and (
                    self.public_pattern(other) in alias
                    or alias in self.public_pattern(other)
                )
                for other in self.patterns
            ):
                raise ValueError("Published category aliases overlap declared patterns")
        spaced = [pattern for pattern in public_patterns if " " in pattern]
        if spaced:
            raise ValueError("Published category patterns contain spaces")
        leaking = [
            pattern for pattern in public_patterns if pattern in NO_DECLARED_PATTERN
        ]
        if leaking:
            raise ValueError("Sentinel contains published category patterns")

    @property
    def sha256(self) -> str:
        if not self.public_aliases:
            return hashlib.sha256("\n".join(self.patterns).encode("utf-8")).hexdigest()
        published = "\n".join(
            f"{source}={self.public_pattern(source)}" for source in self.patterns
        )
        return hashlib.sha256(published.encode("utf-8")).hexdigest()


def read_public_aliases(path: Path) -> tuple[tuple[str, str], ...]:
    """Load a private alias map without copying it into the release."""
    pairs = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=tuple)
    if (
        not isinstance(pairs, tuple)
        or not pairs
        or any(
            not isinstance(source, str) or not isinstance(alias, str)
            for source, alias in pairs
        )
    ):
        raise ValueError("Public aliases must be a string-to-string JSON object")
    if len({source for source, _ in pairs}) != len(pairs):
        raise ValueError("Public aliases contain duplicate source patterns")
    return tuple(sorted(pairs))


class DigestEncoding(StrEnum):
    """The hash encodings the evidence chain binds."""

    file = "file"
    canonical_json = "canonical_json"
    policy_predicate = "policy_predicate"
    incident_package = "incident_package"
    scenario_package = "scenario_package"
    canonical_corpus = "canonical_corpus"
    scenario_corpus = "scenario_corpus"
    governance_projection = "governance_projection"
    run_matrix_plan = "run_matrix_plan"


class RedactedDigest(BaseModel):
    path: str
    encoding: DigestEncoding
    source_sha256: str
    published_sha256: str


class RedactionRecord(BaseModel):
    """What was withheld, and the hashes that tie published bytes to the source."""

    schema_version: int = 1
    transform: str = TRANSFORM_ID
    vocabulary_sha256: str
    category_count: int
    withheld_fields: tuple[str, ...]
    digests: list[RedactedDigest]

    def published_hash_for(self, relative_path: str, chain_sha256: str) -> str:
        """Return the hash a verifier should expect for an object the chain binds.

        The record vouches for a published object only when it names the very
        source digest the evidence chain bound; otherwise the chain's own digest
        is expected and a redacted object fails verification.
        """
        for entry in self.digests:
            if entry.path == relative_path and entry.source_sha256 == chain_sha256:
                return entry.published_sha256
        return chain_sha256


def compose_redaction_records(
    original: RedactionRecord, later: RedactionRecord
) -> RedactionRecord:
    """Bind original source digests to bytes after a second private projection."""

    def index(
        record: RedactionRecord,
    ) -> dict[tuple[str, DigestEncoding], RedactedDigest]:
        entries = {(entry.path, entry.encoding): entry for entry in record.digests}
        if len(entries) != len(record.digests):
            raise ValueError("Redaction record contains duplicate object encodings")
        return entries

    original_entries = index(original)
    later_entries = index(later)
    composed: list[RedactedDigest] = []
    for key, first in original_entries.items():
        second = later_entries.pop(key, None)
        if second is not None and second.source_sha256 != first.published_sha256:
            raise ValueError("Redaction digest chain does not connect")
        final_sha256 = (
            second.published_sha256 if second is not None else first.published_sha256
        )
        if first.source_sha256 != final_sha256:
            composed.append(first.model_copy(update={"published_sha256": final_sha256}))
    composed.extend(later_entries.values())
    composed.sort(key=lambda entry: (entry.path, entry.encoding.value))
    return RedactionRecord(
        schema_version=2,
        transform=f"{original.transform} -> {later.transform}",
        vocabulary_sha256=later.vocabulary_sha256,
        category_count=later.category_count,
        withheld_fields=tuple(
            dict.fromkeys((*original.withheld_fields, *later.withheld_fields))
        ),
        digests=composed,
    )


def project_category(category: str, vocabulary: RedactionVocabulary) -> str:
    lowered = category.lower()
    matched = sorted(
        vocabulary.public_pattern(pattern)
        for pattern in vocabulary.patterns
        if pattern in lowered
    )
    return " ".join(matched) if matched else NO_DECLARED_PATTERN


def project_policy_rules(
    rules: list[dict[str, Any]],
    vocabulary: RedactionVocabulary,
    categories: CategoryIndex | None = None,
) -> list[dict[str, Any]]:
    """Project declared predicates without mutating source rules."""
    projected: list[dict[str, Any]] = []
    for rule in rules:
        scope = rule.get("scope") or {}
        projected_scope = dict(scope)
        if "category_contains" in scope:
            projected_scope["category_contains"] = [
                vocabulary.public_pattern(pattern.lower())
                for pattern in scope["category_contains"]
            ]
        if "incident_type_in" in scope:
            if categories is None:
                raise ValueError("Incident type projection requires source metadata")
            if not all(isinstance(item, str) for item in scope["incident_type_in"]):
                raise ValueError("Incident type predicates must contain strings")
            projected_scope["incident_type_in"] = [
                categories.publish_incident_type(incident_type)
                for incident_type in scope["incident_type_in"]
            ]
        projected.append({**rule, "scope": projected_scope} if scope else rule.copy())
    return projected


def project_normalization(
    normalization: dict[str, Any], vocabulary: RedactionVocabulary
) -> dict[str, Any]:
    """Project declared behavior patterns without changing other fields."""
    classes = normalization.get("behavior_classes")
    if not isinstance(classes, list):
        raise ValueError("Normalization has no behavior classes")
    projected_classes = []
    for behavior in classes:
        patterns = behavior.get("category_patterns")
        if not isinstance(patterns, list):
            raise ValueError("Normalization behavior has no category patterns")
        projected_classes.append(
            {
                **behavior,
                "category_patterns": [
                    vocabulary.public_pattern(pattern.lower()) for pattern in patterns
                ],
            }
        )
    return {**normalization, "behavior_classes": projected_classes}


class CategoryIndex(BaseModel):
    """Published form of every raw category: projection plus a stable rank.

    The rank keeps distinct categories distinct, so counts and cross-partition
    comparisons over categories survive publication; the projection keeps every
    declared-pattern predicate exact. The rank reveals only sorted position.
    """

    model_config = ConfigDict(frozen=True)

    published: dict[str, str]
    incident_types: dict[str, str]

    @classmethod
    def build(
        cls,
        raw_categories: set[str],
        vocabulary: RedactionVocabulary,
        raw_incident_types: set[str] | None = None,
    ) -> CategoryIndex:
        published = {
            raw: vocabulary.public_pattern(raw)
            if raw in vocabulary.patterns
            else f"{project_category(raw, vocabulary)} #{rank:03d}"
            for rank, raw in enumerate(sorted(raw_categories))
        }
        decorations = {
            value.rsplit(" #", 1)[1] for value in published.values() if " #" in value
        }
        for pattern in (vocabulary.public_pattern(raw) for raw in vocabulary.patterns):
            if any(pattern in f" #{decoration}" for decoration in decorations):
                raise ValueError(
                    f"Category rank decoration contains pattern {pattern!r}."
                )
        sensitive_types = sorted(
            incident_type
            for incident_type in (raw_incident_types or set())
            if scan_string(incident_type, key="incident_type")
        )
        incident_types = {
            incident_type: f"incident_type_{rank:03d}"
            for rank, incident_type in enumerate(sensitive_types, start=1)
        }
        if set(incident_types.values()) & (raw_incident_types or set()):
            raise ValueError("Incident type aliases collide with source labels")
        return cls(published=published, incident_types=incident_types)

    def publish(self, raw: str, vocabulary: RedactionVocabulary) -> str:
        return self.published.get(raw) or project_category(raw, vocabulary)

    def publish_report_category(self, raw: str) -> str:
        if raw in self.published:
            return self.published[raw]
        if scan_string(raw, key="category"):
            raise ValueError("Report category contains unmapped publication residue")
        return raw

    def publish_incident_type(self, raw: str) -> str:
        if raw in self.incident_types:
            return self.incident_types[raw]
        if scan_string(raw, key="incident_type"):
            raise ValueError("Policy has unmapped incident type")
        return raw

    def echo(self, value: str) -> str:
        """Republish a report string that repeats a raw category verbatim.

        Sensitive incident-type labels are mapped first. Other bare tokens
        may mean something besides a category and are left unchanged.
        """
        if value in self.incident_types:
            return self.publish_incident_type(value)
        if SLUG_SEPARATOR not in value:
            return value
        return self.published.get(value, value)


def _redact_case_note(details: dict[str, Any]) -> dict[str, Any]:
    text = str(details.get("command") or "")
    classification = parse_publishable_note_text(text) or classify_case_note(text)
    return {
        **details,
        "command": publishable_note_text(classification),
        "source_summary": {"note_kind": classification.kind.value},
    }


def _redact_detection(details: dict[str, Any]) -> dict[str, Any]:
    summary = details.get("source_summary") or {}
    host = summary.get("host")
    kept = (
        {"host": host}
        if isinstance(host, str) and PSEUDONYM_HOST_PATTERN.fullmatch(host)
        else {}
    )
    command = details.get("command")
    return {
        **details,
        "command": WITHHELD if command else command,
        "source_summary": kept,
    }


def _require_supported_event_shape(event: dict[str, Any]) -> dict[str, Any] | None:
    if set(event) - TelemetryEvent.model_fields.keys():
        raise ValueError("unsupported telemetry fields")
    if "details" not in event:
        return None
    details = event.get("details")
    if not isinstance(details, dict):
        raise ValueError("unsupported telemetry details")
    if (
        set(details) - TelemetryDetails.model_fields.keys()
        or details.get("raw") is not None
    ):
        raise ValueError("unsupported telemetry details")
    return details


def redact_event(
    event: dict[str, Any], vocabulary: RedactionVocabulary, categories: CategoryIndex
) -> dict[str, Any]:
    details = _require_supported_event_shape(event)
    redacted = dict(event)
    if isinstance(event.get("category"), str):
        redacted["category"] = categories.publish(event["category"], vocabulary)
    if event.get("source_type") == SYNTHETIC_SOURCE:
        return redacted
    if "username" in event:
        username = event["username"]
        if username not in (None, "") and not (
            isinstance(username, str) and PSEUDONYM_USER_PATTERN.fullmatch(username)
        ):
            redacted["username"] = WITHHELD
    if details is None:
        return redacted
    if event.get("event_type") == CASE_NOTE:
        redacted["details"] = _redact_case_note(details)
    else:
        redacted["details"] = _redact_detection(details)
    if "log_source" in details and details["log_source"] not in (None, ""):
        redacted["details"]["log_source"] = WITHHELD
    return redacted


def redact_telemetry_text(
    text: str, vocabulary: RedactionVocabulary, categories: CategoryIndex
) -> str:
    events = [json.loads(line) for line in text.splitlines() if line.strip()]
    return "".join(
        json.dumps(
            redact_event(event, vocabulary, categories),
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
        for event in events
    )


def redact_output_payload(
    value: Any, vocabulary: RedactionVocabulary, categories: CategoryIndex
) -> Any:
    if isinstance(value, str):
        return categories.echo(value)
    if isinstance(value, list):
        return [redact_output_payload(item, vocabulary, categories) for item in value]
    if not isinstance(value, dict):
        return value
    redacted: dict[str, Any] = {}
    for key, child in value.items():
        published_key = categories.echo(key)
        if key == "category" and isinstance(child, str) and vocabulary.public_aliases:
            redacted[published_key] = categories.publish(child, vocabulary)
        elif key == "incident_type" and isinstance(child, str):
            redacted[published_key] = categories.publish_incident_type(child)
        elif key in {
            "primary_detection_categories",
            "shared_primary_detection_categories",
        } and isinstance(child, list):
            if not all(isinstance(category, str) for category in child):
                raise ValueError("Report categories must contain strings")
            redacted[published_key] = [
                categories.publish_report_category(category) for category in child
            ]
        elif key == "worst_dropped_pattern" and isinstance(child, str):
            pattern = child.lower()
            if pattern not in vocabulary.patterns and scan_string(
                child, key="category"
            ):
                raise ValueError("Report pattern contains unmapped publication residue")
            redacted[published_key] = (
                vocabulary.public_pattern(pattern)
                if pattern in vocabulary.patterns
                else child
            )
        elif key in {"category_contains", "category_patterns"} and isinstance(
            child, list
        ):
            if not all(isinstance(pattern, str) for pattern in child):
                raise ValueError("Category predicates must contain strings")
            redacted[published_key] = [
                vocabulary.public_pattern(pattern.lower()) for pattern in child
            ]
        elif key in WITHHELD_OUTPUT_KEYS and isinstance(child, str):
            redacted[published_key] = WITHHELD
        elif (
            key == "normative_basis"
            and isinstance(child, str)
            and scan_string(child, key=key)
        ):
            redacted[published_key] = WITHHELD
        elif key == PROMPT_MESSAGES_KEY and isinstance(child, list):
            redacted[published_key] = [
                {**message, "content": WITHHELD}
                if isinstance(message, dict)
                else message
                for message in child
            ]
        elif key == TELEMETRY_PREVIEW_KEY and isinstance(child, list):
            redacted[published_key] = [
                redact_event(event, vocabulary, categories)
                if isinstance(event, dict)
                else event
                for event in child
            ]
        else:
            redacted[published_key] = redact_output_payload(
                child, vocabulary, categories
            )
    return redacted


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_sha256(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _redacted_bytes(
    path: Path, vocabulary: RedactionVocabulary, categories: CategoryIndex
) -> bytes | None:
    """Return the published form of a file, or None when it is published as-is."""
    if (
        (vocabulary.public_aliases or categories.incident_types)
        and path.parent.name == "global"
        and path.parent.parent.name == "artifact_data"
    ):
        if path.name == "constraints.yaml":
            rules = yaml.safe_load(path.read_text(encoding="utf-8"))
            return yaml.safe_dump(
                project_policy_rules(rules, vocabulary, categories), sort_keys=False
            ).encode("utf-8")
        if path.name == "telemetry_normalization.yaml":
            normalization = yaml.safe_load(path.read_text(encoding="utf-8"))
            return yaml.safe_dump(
                project_normalization(normalization, vocabulary), sort_keys=False
            ).encode("utf-8")
        if path.name == "coverage_scenarios.yaml":
            scenarios = yaml.safe_load(path.read_text(encoding="utf-8"))
            return yaml.safe_dump(
                redact_output_payload(scenarios, vocabulary, categories),
                sort_keys=False,
            ).encode("utf-8")
        if path.name == "review_record.yaml":
            review = yaml.safe_load(path.read_text(encoding="utf-8"))
            projected = redact_output_payload(review, vocabulary, categories)
            if projected != review:
                return yaml.safe_dump(projected, sort_keys=False).encode("utf-8")
    if path.name == TELEMETRY_FILENAME:
        return redact_telemetry_text(
            path.read_text(encoding="utf-8"), vocabulary, categories
        ).encode("utf-8")
    if path.suffix != ".json":
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    redacted = redact_output_payload(payload, vocabulary, categories)
    if redacted == payload:
        return None
    return (json.dumps(redacted, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _raw_categories(incident_dirs: list[Path]) -> set[str]:
    categories: set[str] = set()
    for incident_dir in incident_dirs:
        for line in (
            (incident_dir / TELEMETRY_FILENAME).read_text(encoding="utf-8").splitlines()
        ):
            if line.strip():
                category = json.loads(line).get("category")
                if isinstance(category, str):
                    categories.add(category)
    return categories


def _raw_incident_types(incident_dirs: list[Path]) -> set[str]:
    incident_types: set[str] = set()
    for incident_dir in incident_dirs:
        metadata_path = incident_dir / "incident_meta.json"
        if not metadata_path.is_file():
            continue
        incident_type = json.loads(metadata_path.read_text(encoding="utf-8")).get(
            "incident_type"
        )
        if isinstance(incident_type, str):
            incident_types.add(incident_type)
    return incident_types


def _incident_dirs(root: Path) -> list[Path]:
    return sorted({path.parent for path in root.rglob(TELEMETRY_FILENAME)})


def _authorize_redacted_scenario_manifests(tree_root: Path) -> None:
    """Classify redacted scenario packages; the residue gate still has the last word."""
    for manifest_path in sorted(tree_root.rglob(SOURCE_MANIFEST_FILENAME)):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("incident_id_assignment") != "source_derived":
            continue
        payload["publication_classification"] = "authorized_for_publication"
        payload["publication_basis"] = (
            f"{TRANSFORM_ID}; see {RECORD_RELATIVE_PATH.as_posix()}"
        )
        manifest_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def _directory_encodings() -> tuple[
    dict[DigestEncoding, Callable[[Path], str]],
    dict[DigestEncoding, Callable[[Path], str]],
    dict[DigestEncoding, Callable[[Path], str]],
]:
    # `adversarial_scenarios` imports expected_digest from this module. Keep
    # these imports local so the digest registry does not create a cycle at
    # module initialization.
    from soc_llm_policy.adversarial_scenarios import (  # noqa: I001, PLC0415
        governance_projection_sha256,
    )
    from soc_llm_policy.corpus_hash import canonical_corpus_sha256  # noqa: PLC0415
    from soc_llm_policy.protocol_freeze import sha256_incident_package  # noqa: PLC0415
    from soc_llm_policy.scenario_bank import corpus_sha256, package_sha256  # noqa: PLC0415

    package_encodings = {
        DigestEncoding.incident_package: sha256_incident_package,
        DigestEncoding.scenario_package: package_sha256,
    }
    governance_encodings = {
        DigestEncoding.governance_projection: governance_projection_sha256,
    }
    corpus_encodings = {
        DigestEncoding.canonical_corpus: canonical_corpus_sha256,
        DigestEncoding.scenario_corpus: corpus_sha256,
    }
    return package_encodings, governance_encodings, corpus_encodings


def run_matrix_plan_sha256(path: Path) -> str:
    """The digest a run matrix embeds as ``plan_sha256``: its canonical cells."""
    cells = json.loads(path.read_text(encoding="utf-8")).get("cells")
    return _sha256_bytes(
        json.dumps(cells, sort_keys=True, separators=(",", ":")).encode()
    )


def _snapshot(
    release_root: Path, roots: list[Path], incident_dirs: list[Path]
) -> dict[tuple[str, DigestEncoding], str]:
    """Digest every object the chain can bind, under each encoding it uses."""
    package_encodings, governance_encodings, corpus_encodings = _directory_encodings()
    digests: dict[tuple[str, DigestEncoding], str] = {}
    for root in roots:
        for path in root.rglob("*"):
            if path.is_file():
                key = (path.relative_to(release_root).as_posix(), DigestEncoding.file)
                digests[key] = _sha256_bytes(path.read_bytes())
                if path.name == "run_matrix.json":
                    digests[(key[0], DigestEncoding.run_matrix_plan)] = (
                        run_matrix_plan_sha256(path)
                    )
                if path.name == "task_matrix.json":
                    digests[(key[0], DigestEncoding.canonical_json)] = (
                        _canonical_json_sha256(path)
                    )
                if (
                    key[0] == "artifact_data/global/constraints.yaml"
                    and (
                        release_root / "artifact_outputs/analysis/protocol_freeze.json"
                    ).is_file()
                ):
                    digests[(key[0], DigestEncoding.policy_predicate)] = (
                        predicate_digest(path)
                    )
    for incident_dir in incident_dirs:
        relative = incident_dir.relative_to(release_root).as_posix()
        for encoding, digest in package_encodings.items():
            digests[(relative, encoding)] = digest(incident_dir)
        if (incident_dir / HUMAN_ACTIONS_FILENAME).is_file():
            for encoding, digest in governance_encodings.items():
                digests[(relative, encoding)] = digest(incident_dir)
    for corpus_dir in {path.parent for path in incident_dirs}:
        relative = corpus_dir.relative_to(release_root).as_posix()
        for encoding, digest in corpus_encodings.items():
            digests[(relative, encoding)] = digest(corpus_dir)
    return digests


def _published_path(path: str, projection: CorrelationProjection | None) -> str:
    if projection is None:
        return path
    return projection.project_path(Path(path)).as_posix()


def redact_release_trees(
    release_root: Path,
    trees: tuple[str, ...],
    vocabulary: RedactionVocabulary,
    projection: CorrelationProjection | None = None,
) -> RedactionRecord:
    """Rewrite every carrier of tenant text under ``trees`` and record the change.

    With a correlation ``projection``, incident IDs and event times are also
    replaced, in bytes and in paths. Each record entry then names the published
    path, so the public record never discloses a source incident ID.
    """
    roots = [release_root / tree for tree in trees if (release_root / tree).is_dir()]
    incident_dirs = sorted(path for root in roots for path in _incident_dirs(root))
    categories = CategoryIndex.build(
        _raw_categories(incident_dirs),
        vocabulary,
        _raw_incident_types(incident_dirs),
    )
    before = _snapshot(release_root, roots, incident_dirs)

    for root in roots:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.name != SOURCE_MANIFEST_FILENAME:
                published = _redacted_bytes(path, vocabulary, categories)
                if published is not None:
                    path.write_bytes(published)
        if root.name == "scenario_banks":
            _authorize_redacted_scenario_manifests(root)

    if projection is not None:
        project_package_trees(
            release_root, projection, tuple(root.name for root in roots)
        )
        incident_dirs = sorted(path for root in roots for path in _incident_dirs(root))

    after = _snapshot(release_root, roots, incident_dirs)
    published = {
        (_published_path(path, projection), encoding): source
        for (path, encoding), source in before.items()
    }
    if len(published) != len(before) or published.keys() != after.keys():
        raise ValueError("Publication projection lost or merged a bound object")
    digests = [
        RedactedDigest(
            path=path,
            encoding=encoding,
            source_sha256=source,
            published_sha256=after[(path, encoding)],
        )
        for (path, encoding), source in sorted(published.items())
        if after[(path, encoding)] != source
    ]
    return RedactionRecord(
        vocabulary_sha256=vocabulary.sha256,
        category_count=len(categories.published),
        withheld_fields=WITHHELD_FIELDS,
        digests=digests,
    )


def project_loose_files_into_record(
    release_root: Path,
    record: RedactionRecord,
    projection: CorrelationProjection,
    excluded_trees: tuple[str, ...],
) -> RedactionRecord:
    """Project text files outside the redacted trees and bind their new bytes."""

    def file_digests() -> dict[str, str]:
        return {
            path.relative_to(release_root).as_posix(): _sha256_bytes(path.read_bytes())
            for path in release_root.rglob("*")
            if path.is_file()
            and not path.is_symlink()
            and path.relative_to(release_root).parts[0] not in excluded_trees
        }

    before = file_digests()
    project_loose_files(release_root, projection, excluded_trees)
    after = file_digests()
    if before.keys() != after.keys():
        raise ValueError("Loose-file projection added or removed files")
    loose = [
        RedactedDigest(
            path=path,
            encoding=DigestEncoding.file,
            source_sha256=before[path],
            published_sha256=after[path],
        )
        for path in sorted(before)
        if before[path] != after[path]
    ]
    digests = sorted(
        [*record.digests, *loose], key=lambda entry: (entry.path, entry.encoding.value)
    )
    return record.model_copy(update={"digests": digests})


def write_record(release_root: Path, record: RedactionRecord) -> Path:
    path = release_root / RECORD_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def load_record(package_root: Path) -> RedactionRecord | None:
    path = package_root / RECORD_RELATIVE_PATH
    if not path.is_file():
        return None
    return RedactionRecord.model_validate_json(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class _DigestSnapshot:
    root: Path
    record_sha256: str | None
    published: dict[tuple[str, str], str]


_DIGEST_SNAPSHOT: ContextVar[_DigestSnapshot | None] = ContextVar(
    "publication_digest_snapshot", default=None
)


def _record_hash(root: Path) -> str | None:
    path = root / RECORD_RELATIVE_PATH
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


@contextmanager
def redaction_digest_scope(package_root: Path) -> Iterator[None]:
    """Reuse one record within an audit, rejecting changes before it returns."""
    root = package_root.resolve()
    active = _DIGEST_SNAPSHOT.get()
    if active is not None and active.root == root:
        yield
        return
    path = root / RECORD_RELATIVE_PATH
    raw = path.read_bytes() if path.is_file() else None
    record_hash = hashlib.sha256(raw).hexdigest() if raw is not None else None
    record = RedactionRecord.model_validate_json(raw) if raw is not None else None
    published: dict[tuple[str, str], str] = {}
    for entry in record.digests if record is not None else []:
        key = (entry.path, entry.source_sha256)
        if key in published and published[key] != entry.published_sha256:
            raise ValueError("Redaction record declares conflicting published digests")
        published[key] = entry.published_sha256
    token = _DIGEST_SNAPSHOT.set(_DigestSnapshot(root, record_hash, published))
    try:
        yield
        if _record_hash(root) != record_hash:
            raise ValueError("Redaction record changed during the audit")
    finally:
        _DIGEST_SNAPSHOT.reset(token)


def expected_digest(package_root: Path, target: Path, chain_sha256: str) -> str:
    """The digest a verifier should expect for a file or directory the chain binds.

    Outside a published package, or for anything the redaction left untouched,
    this is the chain's own digest.
    """
    try:
        relative = target.resolve().relative_to(package_root.resolve()).as_posix()
    except ValueError:
        return chain_sha256
    snapshot = _DIGEST_SNAPSHOT.get()
    if snapshot is not None and snapshot.root == package_root.resolve():
        return snapshot.published.get((relative, chain_sha256), chain_sha256)
    record = load_record(package_root)
    if record is None:
        return chain_sha256
    return record.published_hash_for(relative, chain_sha256)


def chain_digest(package_root: Path, target: Path, published_sha256: str) -> str:
    """The source digest the evidence chain recorded for a published file.

    The inverse of ``expected_digest``: a file whose bytes the publication
    rewrote is named by its source digest inside the chain. A file the record
    does not cover keeps its own digest.
    """
    try:
        relative = target.resolve().relative_to(package_root.resolve()).as_posix()
    except ValueError:
        return published_sha256
    record = load_record(package_root)
    if record is None:
        return published_sha256
    sources = {
        entry.source_sha256
        for entry in record.digests
        if entry.path == relative
        and entry.encoding is DigestEncoding.file
        and entry.published_sha256 == published_sha256
    }
    if len(sources) > 1:
        raise ValueError(f"Redaction record maps {relative} to several sources")
    return sources.pop() if sources else published_sha256
