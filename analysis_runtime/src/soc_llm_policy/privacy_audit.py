from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from soc_llm_policy.paths import RepoPaths

_DEFAULT_FORBIDDEN_TERMS = ["banco"]
_DEFAULT_SUFFIXES = [".json", ".jsonl", ".yaml", ".yml", ".txt", ".md", ".csv"]
_REGEX_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "ipv4": re.compile(
        r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"
    ),
    "phone": re.compile(
        r"(?:\b\d{4}\s*\(\d{2}\)\s*\d{8,9}\b)|"
        r"(?:\+?\d{1,3}\s*\(?\d{2}\)?\s*\d{4,5}[-\s]?\d{4})"
    ),
}
_ANON_EMAIL_RE = re.compile(r"^user_\d+@bank\.local$")
_ANON_IP_RE = re.compile(r"^10\.255\.\d{1,3}\.\d{1,3}$")
_ANON_PHONE_RE = re.compile(r"^<phone_\d+>$")
_HEX_HASH_RE = re.compile(r"\b[a-f0-9]{32,64}\b", flags=re.IGNORECASE)
_WINDOWS_USER_PROFILE_RE = re.compile(
    r"\\+users\\+([^\\/\s\"']+)",
    flags=re.IGNORECASE,
)
_PUBLIC_WINDOWS_PROFILES = {
    "all users",
    "default",
    "default user",
    "public",
    "user_home",
}


def _phone_match_is_hash_false_positive(line: str, value: str) -> bool:
    """Ignore numeric substrings that only appear inside a hex checksum field."""
    if "sha256" not in line.lower():
        return False
    for token in _HEX_HASH_RE.findall(line):
        if value in token:
            return True
    return False


def load_anonymization_policy(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "forbidden_terms": list(_DEFAULT_FORBIDDEN_TERMS),
            "text_file_suffixes": list(_DEFAULT_SUFFIXES),
        }
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid anonymization policy (expected object): {path}")
    terms = raw.get("forbidden_terms")
    suffixes = raw.get("text_file_suffixes")
    return {
        "forbidden_terms": (
            [str(item).strip() for item in terms if str(item).strip()]
            if isinstance(terms, list)
            else list(_DEFAULT_FORBIDDEN_TERMS)
        ),
        "text_file_suffixes": (
            [str(item).strip().lower() for item in suffixes if str(item).strip()]
            if isinstance(suffixes, list)
            else list(_DEFAULT_SUFFIXES)
        ),
    }


def _iter_target_files(
    paths: RepoPaths,
    incidents: list[str],
    suffixes: set[str],
) -> list[Path]:
    out: list[Path] = [
        paths.inbox_action_catalog_path,
        paths.inbox_constraints_path,
    ]
    for incident_id in incidents:
        incident_dir = paths.inbox_incident_dir(incident_id)
        if not incident_dir.exists():
            continue
        for file_path in incident_dir.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() in suffixes:
                out.append(file_path)
    return sorted(set(out))


def _scan_line(
    line: str,
    forbidden_term_patterns: list[tuple[str, re.Pattern[str]]],
) -> list[tuple[str, str]]:
    findings: list[tuple[str, str]] = []
    for finding_type, pattern in _REGEX_PATTERNS.items():
        for match in pattern.finditer(line):
            value = match.group(0)
            if finding_type == "email" and _ANON_EMAIL_RE.match(value):
                continue
            if finding_type == "ipv4" and _ANON_IP_RE.match(value):
                continue
            if finding_type == "phone" and _ANON_PHONE_RE.match(value):
                continue
            if finding_type == "phone" and _phone_match_is_hash_false_positive(
                line, value
            ):
                continue
            findings.append((finding_type, value))
    for term, pattern in forbidden_term_patterns:
        match = pattern.search(line)
        if match:
            findings.append(("forbidden_term", term))
    for match in _WINDOWS_USER_PROFILE_RE.finditer(line):
        profile = match.group(1)
        if profile.lower() not in _PUBLIC_WINDOWS_PROFILES:
            findings.append(("windows_user_profile", profile))
    return findings


def scan_dataset_privacy(
    *,
    paths: RepoPaths,
    incidents: list[str],
) -> dict[str, Any]:
    policy_path = paths.config_dir / "anonymization_policy.yaml"
    policy = load_anonymization_policy(policy_path)
    forbidden_terms = list(policy["forbidden_terms"])
    suffixes = {item for item in policy["text_file_suffixes"]}
    forbidden_term_patterns = [
        (term, re.compile(rf"\b{re.escape(term)}\b", flags=re.IGNORECASE))
        for term in forbidden_terms
    ]
    findings: list[dict[str, Any]] = []
    for file_path in _iter_target_files(paths, incidents, suffixes):
        # A file this audit cannot read is a file it never scanned. Skipping it
        # lets the report state zero privacy issues over content nobody saw.
        lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
        for line_no, line in enumerate(lines, start=1):
            line_findings = _scan_line(
                line,
                forbidden_term_patterns,
            )
            for finding_type, value in line_findings:
                findings.append(
                    {
                        "type": finding_type,
                        "value": value,
                        "file": str(file_path),
                        "line": line_no,
                    }
                )

    return {
        "policy_path": str(policy_path),
        "forbidden_terms": forbidden_terms,
        "issue_count": len(findings),
        "issues": findings,
    }
