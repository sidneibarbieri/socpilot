from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from soc_llm_policy.io import TelemetryEvent

_HIGH_CONFIDENCE_THRESHOLD = 0.8


class AttackRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rule_id: str = Field(..., min_length=1)
    technique_id: str
    technique_name: str
    tactic_id: str
    tactic_name: str
    event_type_contains: list[str]
    command_contains: list[str] = Field(default_factory=list)
    category_contains: list[str] = Field(default_factory=list)
    source_type_contains: list[str] = Field(default_factory=list)
    event_type_weight: float = Field(default=0.55, ge=0.0, le=1.0)
    command_weight: float = Field(default=0.25, ge=0.0, le=1.0)
    category_weight: float = Field(default=0.1, ge=0.0, le=1.0)
    source_type_weight: float = Field(default=0.1, ge=0.0, le=1.0)
    min_confidence: float = Field(default=0.55, ge=0.0, le=1.0)


@dataclass(frozen=True)
class AttackMatch:
    rule_id: str
    technique_id: str
    technique_name: str
    tactic_id: str
    tactic_name: str
    event_type: str
    confidence: float
    matched_signals: list[str]
    evidence: str

    def to_dict(self) -> dict[str, object]:
        return {
            "rule_id": self.rule_id,
            "technique_id": self.technique_id,
            "technique_name": self.technique_name,
            "tactic_id": self.tactic_id,
            "tactic_name": self.tactic_name,
            "event_type": self.event_type,
            "confidence": self.confidence,
            "matched_signals": self.matched_signals,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class AttackCandidate:
    rule_id: str
    technique_id: str
    technique_name: str
    tactic_id: str
    tactic_name: str
    event_type: str
    score: float
    matched_signals: list[str]
    evidence: str

    def to_dict(self) -> dict[str, object]:
        return {
            "rule_id": self.rule_id,
            "technique_id": self.technique_id,
            "technique_name": self.technique_name,
            "tactic_id": self.tactic_id,
            "tactic_name": self.tactic_name,
            "event_type": self.event_type,
            "score": self.score,
            "matched_signals": self.matched_signals,
            "evidence": self.evidence,
        }


def load_attack_rules(path: Path) -> list[AttackRule]:
    """Load and validate ATT&CK rules from a versioned YAML file."""
    if not path.exists():
        raise FileNotFoundError(f"Missing ATT&CK mapping file: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, list):
        raise ValueError(f"attack_mapping.yaml must contain a list of rules: {path}")

    return [AttackRule.model_validate(item) for item in raw]


def _contains_any(value: str, needles: list[str]) -> tuple[bool, list[str]]:
    if not needles:
        return False, []
    hits = [needle for needle in needles if needle.lower() in value]
    return bool(hits), hits


def _matches_rule(rule: AttackRule, event: TelemetryEvent) -> bool:
    event_type = event.event_type.lower()
    return any(needle.lower() in event_type for needle in rule.event_type_contains)


def _build_evidence(event: TelemetryEvent, signals: list[str]) -> str:
    command = event.details.command or ""
    signal_str = ", ".join(signals) if signals else "event_type"
    if command:
        return f"{signal_str} | {command[:120]}"
    return f"{signal_str} | {event.event_type}"


def _score_event_against_rule(
    rule: AttackRule,
    event: TelemetryEvent,
) -> tuple[float, list[str]]:
    command = (event.details.command or "").lower()
    category = (event.category or "").lower()
    source_type = (event.source_type or "").lower()

    score = rule.event_type_weight
    matched_signals = ["event_type"]

    command_match, command_hits = _contains_any(command, rule.command_contains)
    if command_match:
        score += rule.command_weight
        matched_signals.extend(f"command:{hit}" for hit in command_hits)

    category_match, category_hits = _contains_any(category, rule.category_contains)
    if category_match:
        score += rule.category_weight
        matched_signals.extend(f"category:{hit}" for hit in category_hits)

    source_match, source_hits = _contains_any(source_type, rule.source_type_contains)
    if source_match:
        score += rule.source_type_weight
        matched_signals.extend(f"source_type:{hit}" for hit in source_hits)

    final_score = min(round(score, 3), 1.0)
    return final_score, matched_signals


def map_event_to_attack(
    event: TelemetryEvent,
    rules: list[AttackRule],
) -> list[AttackMatch]:
    """Map one telemetry event to ATT&CK techniques using heuristics."""
    matches: list[AttackMatch] = []
    for rule in rules:
        if _matches_rule(rule, event):
            final_confidence, matched_signals = _score_event_against_rule(rule, event)
            if final_confidence < rule.min_confidence:
                continue

            matches.append(
                AttackMatch(
                    rule_id=rule.rule_id,
                    technique_id=rule.technique_id,
                    technique_name=rule.technique_name,
                    tactic_id=rule.tactic_id,
                    tactic_name=rule.tactic_name,
                    event_type=event.event_type,
                    confidence=final_confidence,
                    matched_signals=matched_signals,
                    evidence=_build_evidence(event, matched_signals),
                )
            )
    return matches


def map_event_to_attack_candidates(
    event: TelemetryEvent,
    rules: list[AttackRule],
) -> list[AttackCandidate]:
    """Layer A: ATT&CK candidates without minimum-confidence filtering."""
    candidates: list[AttackCandidate] = []
    for rule in rules:
        if not _matches_rule(rule, event):
            continue
        score, matched_signals = _score_event_against_rule(rule, event)
        candidates.append(
            AttackCandidate(
                rule_id=rule.rule_id,
                technique_id=rule.technique_id,
                technique_name=rule.technique_name,
                tactic_id=rule.tactic_id,
                tactic_name=rule.tactic_name,
                event_type=event.event_type,
                score=score,
                matched_signals=matched_signals,
                evidence=_build_evidence(event, matched_signals),
            )
        )
    return candidates


def summarize_attack_context(
    telemetry: list[TelemetryEvent],
    rules: list[AttackRule],
) -> dict[str, object]:
    """Summarize ATT&CK context inferred from incident telemetry."""
    matches = [
        match
        for event in telemetry
        for match in map_event_to_attack(event, rules)
    ]

    technique_counter = Counter(
        (match.technique_id, match.technique_name) for match in matches
    )
    tactic_counter = Counter((match.tactic_id, match.tactic_name) for match in matches)

    unique_techniques = sorted(
        {
            (match.technique_id, match.technique_name)
            for match in matches
        }
    )
    unique_tactics = sorted(
        {
            (match.tactic_id, match.tactic_name)
            for match in matches
        }
    )

    high_confidence_matches = [
        match
        for match in matches
        if match.confidence >= _HIGH_CONFIDENCE_THRESHOLD
    ]

    return {
        "rule_count": len(rules),
        "match_count": len(matches),
        "high_confidence_match_count": len(high_confidence_matches),
        "techniques": [
            {"technique_id": technique_id, "technique_name": technique_name}
            for technique_id, technique_name in unique_techniques
        ],
        "tactics": [
            {"tactic_id": tactic_id, "tactic_name": tactic_name}
            for tactic_id, tactic_name in unique_tactics
        ],
        "technique_counts": [
            {
                "technique_id": technique_id,
                "technique_name": technique_name,
                "count": count,
            }
            for (technique_id, technique_name), count in sorted(
                technique_counter.items(),
                key=lambda item: (-item[1], item[0][0]),
            )
        ],
        "tactic_counts": [
            {"tactic_id": tactic_id, "tactic_name": tactic_name, "count": count}
            for (tactic_id, tactic_name), count in sorted(
                tactic_counter.items(),
                key=lambda item: (-item[1], item[0][0]),
            )
        ],
        "matches": [match.to_dict() for match in matches[:10]],
    }


def summarize_attack_candidates(
    telemetry: list[TelemetryEvent],
    rules: list[AttackRule],
) -> dict[str, object]:
    """Layer A: ATT&CK candidate summary for exploratory analysis."""
    candidates = [
        candidate
        for event in telemetry
        for candidate in map_event_to_attack_candidates(event, rules)
    ]

    technique_counter = Counter(
        (candidate.technique_id, candidate.technique_name) for candidate in candidates
    )
    tactic_counter = Counter(
        (candidate.tactic_id, candidate.tactic_name) for candidate in candidates
    )
    avg_score = (
        round(sum(candidate.score for candidate in candidates) / len(candidates), 3)
        if candidates
        else 0.0
    )

    return {
        "rule_count": len(rules),
        "candidate_count": len(candidates),
        "average_score": avg_score,
        "technique_counts": [
            {
                "technique_id": technique_id,
                "technique_name": technique_name,
                "count": count,
            }
            for (technique_id, technique_name), count in sorted(
                technique_counter.items(),
                key=lambda item: (-item[1], item[0][0]),
            )
        ],
        "tactic_counts": [
            {"tactic_id": tactic_id, "tactic_name": tactic_name, "count": count}
            for (tactic_id, tactic_name), count in sorted(
                tactic_counter.items(),
                key=lambda item: (-item[1], item[0][0]),
            )
        ],
        "candidates": [candidate.to_dict() for candidate in candidates[:10]],
    }
