"""Deterministic classification of SOAR case-note text into a closed kind set.

A raw case note is operator prose written inside a customer tenant. It carries
asset names, privileged account names, vendor detection identifiers and
internal ticket identifiers that no pseudonymization pass over structured
fields can reach. This module replaces the prose with a declared kind label so
that a package can describe *what happened* without republishing the text.

Classification is a declared marker table, never an inference: a note whose
markers match no declared kind is classified ``unclassified`` and its text is
discarded rather than summarized.
"""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum


class CaseNoteKind(StrEnum):
    """The publishable shapes a SOAR case note may take."""

    detection_name = "detection_name"
    detection_action_summary = "detection_action_summary"
    assistant_threat_analysis = "assistant_threat_analysis"
    hash_reputation_analysis = "hash_reputation_analysis"
    siem_query_empty = "siem_query_empty"
    siem_query_result = "siem_query_result"
    siem_event_payload = "siem_event_payload"
    alert_status_change = "alert_status_change"
    incident_status_change = "incident_status_change"
    incident_assignment = "incident_assignment"
    deduplication_key = "deduplication_key"
    deduplication_master = "deduplication_master"
    deduplication_lookup_empty = "deduplication_lookup_empty"
    analyst_remediation_action = "analyst_remediation_action"
    automation_status = "automation_status"
    automation_rule_trace = "automation_rule_trace"
    edr_comment_added = "edr_comment_added"
    edr_alert_closed = "edr_alert_closed"
    contact_directory = "contact_directory"
    unclassified = "unclassified"


DETECTION_NAMES = (
    "adware detected",
    "browser exploit",
    "code injection",
    "command execution",
    "credential theft",
    "establish persistence",
    "general authentication failed",
    "general authentication successful",
    "host login failed",
    "machine learning",
    "misc exploit",
    "ransomware",
    "remote code execution",
    "suspicious activity",
    "access denied",
    "compliance violation",
    "cryptomining activity detected",
    "firewall deny",
    "firewall permit",
    "information",
    "mail reconnaissance",
    "misc network communication event",
    "potential worm activity",
    "process creation success",
    "session reset",
    "ssh login succeeded",
    "unauthorized access attempt",
    "user account locked",
)

# Each entry is (kind, markers) and matches when every marker is present in the
# normalized text. Order matters: the first matching entry wins, so narrower
# families are declared before broader ones.
_MARKER_TABLE: tuple[tuple[CaseNoteKind, tuple[str, ...]], ...] = (
    (CaseNoteKind.contact_directory, ("redacted operational contact note",)),
    (CaseNoteKind.contact_directory, ("contatos operacionais",)),
    (CaseNoteKind.contact_directory, ("regras operacionais",)),
    (CaseNoteKind.contact_directory, ("fora do expediente",)),
    (CaseNoteKind.contact_directory, ("central - contatos",)),
    (CaseNoteKind.contact_directory, ("agencia - contatos",)),
    (CaseNoteKind.assistant_threat_analysis, ("resumo do incidente",)),
    (CaseNoteKind.assistant_threat_analysis, ("analise de ameaca",)),
    (CaseNoteKind.assistant_threat_analysis, ("triagem inicial",)),
    (CaseNoteKind.automation_rule_trace, ("debug:", "atendida")),
    (CaseNoteKind.automation_status, ("automacao:",)),
    (CaseNoteKind.edr_alert_closed, ("alerta devidamente tratado",)),
    (CaseNoteKind.hash_reputation_analysis, ("virustotal",)),
    (CaseNoteKind.hash_reputation_analysis, ("hybrid analysis",)),
    (CaseNoteKind.hash_reputation_analysis, ("analise de hash",)),
    (CaseNoteKind.detection_action_summary, ("acao tomada",)),
    (CaseNoteKind.detection_action_summary, ("resumo da deteccao",)),
    (CaseNoteKind.siem_query_empty, ("query executou", "nao retornou eventos")),
    (CaseNoteKind.siem_event_payload, ("campos do evento",)),
    (CaseNoteKind.siem_query_result, ("qrhost",)),
    (CaseNoteKind.deduplication_master, ("dedup", "incidente master")),
    (CaseNoteKind.deduplication_lookup_empty, ("dedup", "nenhum incidente encontrado")),
    (CaseNoteKind.analyst_remediation_action, ("troca de senha",)),
    (CaseNoteKind.deduplication_key, ("deduplicacao", "chave gerada")),
    (CaseNoteKind.edr_alert_closed, ("alerta fechado", "edr")),
    (CaseNoteKind.edr_comment_added, ("edr atualizado",)),
    (CaseNoteKind.alert_status_change, ("status do alerta alterado",)),
    (CaseNoteKind.incident_status_change, ("status do incidente foi alterado",)),
    (CaseNoteKind.incident_assignment, ("foi atribuido a",)),
)

_TAG_RE = re.compile(r"<[^>]+>")


@dataclass(frozen=True)
class CaseNoteClassification:
    """The publishable description of one case note."""

    kind: CaseNoteKind
    detection_name: str | None = None

    @property
    def is_publishable(self) -> bool:
        return self.kind is not CaseNoteKind.unclassified


def normalize_note_text(text: str) -> str:
    """Fold a raw note into accent-free lowercase words for marker matching."""
    folded = html.unescape(text)
    folded = _TAG_RE.sub(" ", folded)
    folded = unicodedata.normalize("NFKD", folded)
    folded = "".join(char for char in folded if not unicodedata.combining(char))
    return " ".join(folded.lower().split())


def classify_case_note(text: str) -> CaseNoteClassification:
    """Map raw note prose onto the declared kind set, or abstain."""
    normalized = normalize_note_text(text)
    if not normalized:
        return CaseNoteClassification(CaseNoteKind.unclassified)

    if normalized in DETECTION_NAMES:
        return CaseNoteClassification(
            CaseNoteKind.detection_name,
            detection_name=normalized,
        )

    for kind, markers in _MARKER_TABLE:
        if all(marker in normalized for marker in markers):
            return CaseNoteClassification(kind)

    return CaseNoteClassification(CaseNoteKind.unclassified)


def publishable_note_text(classification: CaseNoteClassification) -> str:
    """Render the note as its declared label, with no operator prose."""
    if classification.kind is CaseNoteKind.detection_name:
        assert classification.detection_name is not None
        return f"detection_name: {classification.detection_name}"
    return f"soar_note: {classification.kind.value}"


def parse_publishable_note_text(text: str) -> CaseNoteClassification | None:
    """Accept only an exact label produced by the declared note vocabulary."""
    detection_prefix = "detection_name: "
    if text.startswith(detection_prefix):
        name = text.removeprefix(detection_prefix)
        if name in DETECTION_NAMES:
            return CaseNoteClassification(
                CaseNoteKind.detection_name, detection_name=name
            )
        return None
    note_prefix = "soar_note: "
    if text.startswith(note_prefix):
        name = text.removeprefix(note_prefix)
        kind = next(
            (candidate for candidate in CaseNoteKind if candidate.value == name), None
        )
        if kind is not None and kind is not CaseNoteKind.detection_name:
            return CaseNoteClassification(kind)
    return None
