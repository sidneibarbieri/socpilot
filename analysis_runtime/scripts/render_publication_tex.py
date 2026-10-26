"""Recompute the corpus-publication macros that a released package can support.

The paper's publication macros come from an audit of the private corpus. A
package carries only the published projection, so some values cannot be
recomputed from it: residue counts and the before/after applicability
comparison need the raw text. Those names are declared in PRIVATE_MACROS and
are reported, never silently skipped, by the paper-number verifier.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

import yaml

from soc_llm_policy.publication_redaction import RedactionVocabulary

MACRO_PREFIX = "SocCorpus"
PRIVATE_MACROS = frozenset(
    {
        f"{MACRO_PREFIX}PackagesWithResidue",
        f"{MACRO_PREFIX}ApplicabilityDifferences",
    }
)
UNCLASSIFIED_NOTE_KIND = "unclassified"


def _incident_dirs(partition: Path) -> list[Path]:
    return sorted(path for path in partition.iterdir() if path.is_dir())


def _events(incident: Path) -> list[dict]:
    lines = (incident / "incident_telemetry.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in lines.splitlines() if line.strip()]


def _is_case_note(event: dict) -> bool:
    return event.get("event_type") == "case_note"


def _note_kind(event: dict) -> str:
    return str(event["details"]["source_summary"].get("note_kind"))


def published_profile(
    partitions: list[Path], constraints: Path, normalization: Path
) -> dict[str, int]:
    """Count what the published projection still shows about the corpus."""
    incidents = [incident for partition in partitions for incident in _incident_dirs(partition)]
    metas = [json.loads((incident / "incident_meta.json").read_text()) for incident in incidents]
    severity = collections.Counter(meta.get("severity") or "unspecified" for meta in metas)
    incident_type = collections.Counter(meta.get("incident_type") or "unspecified" for meta in metas)
    events = [event for incident in incidents for event in _events(incident)]
    notes = [event for event in events if _is_case_note(event)]
    categories = {event["category"] for event in events if isinstance(event.get("category"), str)}
    rules = yaml.safe_load(constraints.read_text(encoding="utf-8"))
    vocabulary = RedactionVocabulary.from_policy(constraints, normalization)
    return {
        "Incidents": len(incidents),
        "PartitionOne": len(_incident_dirs(partitions[0])),
        "PartitionTwo": len(_incident_dirs(partitions[1])),
        "SeverityCritical": severity["critical"],
        "SeverityHigh": severity["high"],
        "SeverityMedium": severity["medium"],
        "SeverityLow": severity["low"],
        "TypeMalware": incident_type["malware"],
        "TypeSuspicious": incident_type["suspicious_activity"],
        "TypeOther": len(incidents) - incident_type["malware"] - incident_type["suspicious_activity"],
        "TypeCount": len(incident_type),
        "CaseNotes": len(notes),
        "CaseNotesUnclassified": sum(_note_kind(note) == UNCLASSIFIED_NOTE_KIND for note in notes),
        "DistinctCategories": len(categories),
        "DeclaredPatterns": len(vocabulary.patterns),
        "ApplicabilityChecks": len(rules) * len(incidents),
    }


def render_macros(profile: dict[str, int]) -> str:
    return "\n".join(
        f"\\newcommand{{\\{MACRO_PREFIX}{name}}}{{{value:,}}}" for name, value in profile.items()
    )
