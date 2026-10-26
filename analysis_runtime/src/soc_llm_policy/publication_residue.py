"""Content gate for institutional residue in publishable incident packages.

The pseudonymization pass and the dataset privacy audit share one rule set, so
the audit can only ever rediscover what the pass already masks. Residue that
the pass never looks for therefore reaches a package with a clean privacy
report. This module is the independent check: it does not reapply the
pseudonymization rules, it asserts what a publishable string is allowed to look
like and names everything else.

Two rules decide a package:

1. every string key, leaf, and relative path must be free of declared residue markers;
2. every case-note text must be a declared label from
   :mod:`soc_llm_policy.case_note_kinds`, never operator prose.

Both rules are decidable, so a package is admitted or refused, never repaired.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from ipaddress import IPv4Address, IPv4Network
from pathlib import Path
from typing import Any

from soc_llm_policy.case_note_kinds import DETECTION_NAMES, CaseNoteKind

_REPORTED_FINDING_LIMIT = 20

CASE_NOTE_EVENT_TYPES = frozenset({"case_note"})


class ResidueKind(StrEnum):
    """The residue classes this gate refuses to publish."""

    vendor_detection_identifier = "vendor_detection_identifier"
    vendor_product_name = "vendor_product_name"
    internal_asset_name = "internal_asset_name"
    internal_log_source = "internal_log_source"
    internal_account_name = "internal_account_name"
    internal_case_identifier = "internal_case_identifier"
    tenant_identifier = "tenant_identifier"
    correlation_digest = "correlation_digest"
    operator_prose = "operator_prose"
    markup_entity = "markup_entity"
    public_ip_address = "public_ip_address"
    private_ip_address = "private_ip_address"
    raw_private_ip_collision = "raw_private_ip_collision"
    unmasked_email = "unmasked_email"
    taxpayer_identifier = "taxpayer_identifier"
    undeclared_case_note_text = "undeclared_case_note_text"


_PSEUDONYM_EMAIL = re.compile(r"^user_\d+@bank\.local$")
_PRIVATE_IP = re.compile(
    r"^(?:10\.|127\.|192\.168\.|169\.254\.|172\.(?:1[6-9]|2\d|3[01])\.)"
)
_IPV4_OCTET_COUNT = 4
_MAX_IPV4_OCTET = 255
_ALIAS_BUCKET_SIZE = 250


def _matches_synthetic_ip_namespace(address: str) -> bool:
    octets = address.split(".")
    return (
        len(octets) == _IPV4_OCTET_COUNT
        and octets[:2] == ["10", "255"]
        and all(octet.isdecimal() for octet in octets[2:])
        and 1 <= int(octets[2]) <= _MAX_IPV4_OCTET
        and 1 <= int(octets[3]) <= _ALIAS_BUCKET_SIZE
    )


_DETECTORS: tuple[tuple[ResidueKind, re.Pattern[str]], ...] = (
    (
        ResidueKind.vendor_detection_identifier,
        re.compile(r"[0-9a-f]{32}:ind:[0-9a-f]{32}|detect\s*id", re.IGNORECASE),
    ),
    (
        # Asset naming conventions mix at least two capitals with digits in one
        # unspaced token, a shape no declared label or enum value takes.
        ResidueKind.internal_asset_name,
        re.compile(
            r"\b(?=[A-Z0-9]{6,}\b)(?=(?:[A-Z0-9]*[A-Z]){2})(?=[A-Z0-9]*\d)[A-Z0-9]+\b"
        ),
    ),
    (
        # A sanitized corpus names a log source by a pseudonym or not at all; a
        # real segment/interface name kept after "log source" is residue. "id"
        # is excluded because that is the internal_case_identifier shape.
        ResidueKind.internal_log_source,
        re.compile(
            r"log[_\s]?source[_\s]+(?!id\b)[a-z][a-z0-9]*(?:_[a-z0-9]+)+", re.IGNORECASE
        ),
    ),
    (
        ResidueKind.internal_account_name,
        re.compile(r"\b(?:adm|admin|svc|srv|op)[._-][a-z0-9]{2,}\b", re.IGNORECASE),
    ),
    (
        ResidueKind.internal_case_identifier,
        re.compile(
            r"\b(?:alertaid|alerta_id|note_id|qid|log\s*source\s*id"
            r"|incident\s*id)\W{0,4}\d",
            re.IGNORECASE,
        ),
    ),
    (ResidueKind.markup_entity, re.compile(r"&[a-zA-Z]{2,8};")),
    (
        ResidueKind.taxpayer_identifier,
        re.compile(
            r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b|\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b"
        ),
    ),
)

_TIMESTAMP = re.compile(
    r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?$"
)
# Public identifiers that share the asset-name shape: ATT&CK object ids, CVE
# numbers and hash algorithm names.
_PUBLIC_UPPERCASE_TOKEN = re.compile(
    r"^(?:TA|T|S|G|C|M|DS)\d{4}$|^CVE\d+$|^SHA\d{1,3}$|^MD5$"
)
# A LaTeX/xcolor HTML colour such as {HTML}{4477AA} is a colour, not an asset.
_TEX_HTML_COLOUR = re.compile(r"\{HTML\}\{([0-9A-F]{6})\}")
# Free-text fields are where operators write; everywhere else a hex digest is a
# content address the artifact itself computed.
FREE_TEXT_KEYS = frozenset(
    {"command", "text", "description", "detection_name", "content"}
)
_URL = re.compile(r"^[a-z][a-z0-9+.-]*://", re.IGNORECASE)
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_BARE_DIGEST = re.compile(r"\b[0-9a-f]{32,64}\b", re.IGNORECASE)
_UUID = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)
_VENDOR_COMPOUND = re.compile(
    r"(?<![a-z0-9_])(?:crowdstrike|crowdstike|falcon|qradar|sentinelone|zabbix)"
    r"(?:[_-][a-z0-9]+)+\b",
    re.IGNORECASE,
)
_PUBLIC_VENDOR_COMPOUNDS = frozenset({"falcon_sensor"})
_SENSITIVE_IDENTIFIER_FIELDS = frozenset(
    {"cid", "aid", "deviceid", "customerid", "clientid", "tenantid"}
)
_TENANT_ASSIGNMENT = re.compile(
    r"\btenant[_\s-]?(?:id|identifier|guid)\s*[:=]\s*[a-z0-9_-]{3,}",
    re.IGNORECASE,
)
_IDENTIFIER_ASSIGNMENT = re.compile(
    r"\b(?:cid|aid|(?:device|customer|client)[_\s-]?id)\s*[:=]\s*[a-z0-9_-]{3,}",
    re.IGNORECASE,
)
_TENANT_FIELD_SUFFIXES = (
    "tenantid",
    "tenantidentifier",
    "tenantguid",
    "tenantuuid",
    "tenant",
)

# Portuguese function words do not appear in a declared label, an identifier or
# a command line, so their presence marks surviving operator prose.
_PROSE_MARKERS = re.compile(
    r"(?<![./@-])\b(?:para|foi|que|com|uma|pelo|pela|este|esta|nao|est[aá]|ser|dos|das"
    r"|usu[aá]rio|incidente|processo|arquivo|senha|chave|motivo)\b",
    re.IGNORECASE,
)

_DECLARED_NOTE_LABELS = frozenset(
    [f"soar_note: {kind.value}" for kind in CaseNoteKind]
    + [f"detection_name: {name}" for name in DETECTION_NAMES]
)


@dataclass(frozen=True)
class ResidueFinding:
    """One refusal reason, located precisely enough to fix at the source."""

    kind: ResidueKind
    value: str
    path: str
    location: str

    def as_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind.value,
            "value": self.value,
            "path": self.path,
            "location": self.location,
        }


def _first_residue_match(
    kind: ResidueKind, pattern: re.Pattern[str], text: str
) -> str | None:
    colours = {colour.span(1) for colour in _TEX_HTML_COLOUR.finditer(text)}
    for match in pattern.finditer(text):
        token = match.group(0)
        if kind is ResidueKind.internal_asset_name and (
            _PUBLIC_UPPERCASE_TOKEN.match(token) or match.span() in colours
        ):
            continue
        return token
    return None


def _is_tenant_field(name: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]", "", name.casefold())
    return normalized == "tid" or normalized.endswith(_TENANT_FIELD_SUFFIXES)


def _is_sensitive_identifier_field(name: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]", "", name.casefold())
    return normalized in _SENSITIVE_IDENTIFIER_FIELDS or _is_tenant_field(name)


def scan_string(text: str, *, key: str) -> list[tuple[ResidueKind, str]]:
    """Report every residue marker carried by one string leaf."""
    hits: list[tuple[ResidueKind, str]] = []
    if (
        _is_sensitive_identifier_field(key)
        or (not key and _is_sensitive_identifier_field(text))
        or _TENANT_ASSIGNMENT.search(text)
        or _IDENTIFIER_ASSIGNMENT.search(text)
    ):
        hits.append((ResidueKind.tenant_identifier, "[institutional identifier]"))
    if _UUID.search(text):
        hits.append((ResidueKind.tenant_identifier, "[opaque identifier]"))
    for compound in _VENDOR_COMPOUND.finditer(text):
        if compound.group(0).casefold() not in _PUBLIC_VENDOR_COMPOUNDS:
            hits.append(
                (ResidueKind.vendor_product_name, "[vendor-linked internal label]")
            )
            break
    if _TIMESTAMP.match(text.strip()):
        return hits
    for kind, pattern in _DETECTORS:
        match = _first_residue_match(kind, pattern, text)
        if match is not None:
            hits.append((kind, match))

    for email in _EMAIL.finditer(text):
        if not _PSEUDONYM_EMAIL.fullmatch(email.group(0)):
            hits.append((ResidueKind.unmasked_email, email.group(0)))
            break

    for address in _IPV4.findall(text):
        if _PRIVATE_IP.match(address):
            if (
                key == "ip" or key.endswith("_ip")
            ) and not _matches_synthetic_ip_namespace(address):
                hits.append(
                    (ResidueKind.private_ip_address, "[unapproved private address]")
                )
                break
        else:
            hits.append((ResidueKind.public_ip_address, address))
            break

    if key in FREE_TEXT_KEYS:
        digest = _BARE_DIGEST.search(text)
        if digest:
            hits.append((ResidueKind.correlation_digest, digest.group(0)))

    if not _URL.match(text.strip()):
        prose = _PROSE_MARKERS.search(text)
        if prose:
            hits.append((ResidueKind.operator_prose, prose.group(0)))

    return hits


def _walk_strings(value: Any, key: str = "") -> Iterator[tuple[str, str, str]]:
    """Yield string keys and leaves with residue-safe JSON locations."""
    if isinstance(value, dict):
        for child_key, child in value.items():
            key_text = str(child_key)
            segment = (
                _opaque_location(key_text)
                if scan_string(key_text, key="")
                else key_text.replace("~", "~0").replace("/", "~1")
            )
            yield "", key_text, f"/{segment} [key]"
            for item in _walk_strings(child, key_text):
                yield item[0], item[1], f"/{segment}{item[2]}"
    elif isinstance(value, list):
        for index, child in enumerate(value):
            for item in _walk_strings(child, key):
                yield item[0], item[1], f"/{index}{item[2]}"
    elif isinstance(value, str):
        yield key, value, ""


def _opaque_location(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _audit_relative_name(display_path: str) -> tuple[str, list[ResidueFinding]]:
    """Refuse residue in shipped names without copying it into report paths."""
    hits = scan_string(display_path, key="")
    if not hits:
        return display_path, []
    safe_path = _opaque_location(display_path)
    findings = [
        ResidueFinding(kind=kind, value=value, path=safe_path, location="relative path")
        for kind, value in hits
    ]
    return safe_path, findings


def _note_text(event: dict[str, Any]) -> str | None:
    if event.get("event_type") not in CASE_NOTE_EVENT_TYPES:
        return None
    details = event.get("details")
    if not isinstance(details, dict):
        return None
    command = details.get("command")
    return command if isinstance(command, str) else None


def audit_telemetry_file(path: Path, *, display_path: str) -> list[ResidueFinding]:
    """Apply both publication rules to one telemetry file."""
    findings: list[ResidueFinding] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        event = json.loads(line)

        note = _note_text(event)
        if note is not None and note not in _DECLARED_NOTE_LABELS:
            findings.append(
                ResidueFinding(
                    kind=ResidueKind.undeclared_case_note_text,
                    value=note[:80],
                    path=display_path,
                    location=f"line {line_number}",
                )
            )
            continue

        for key, text, pointer in _walk_strings(event):
            for kind, value in scan_string(text, key=key):
                findings.append(
                    ResidueFinding(
                        kind=kind,
                        value=value,
                        path=display_path,
                        location=f"line {line_number} {pointer}",
                    )
                )
    return findings


def audit_json_file(path: Path, *, display_path: str) -> list[ResidueFinding]:
    """Apply the residue rule to one JSON document."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    findings: list[ResidueFinding] = []
    for key, text, pointer in _walk_strings(payload):
        for kind, value in scan_string(text, key=key):
            findings.append(
                ResidueFinding(
                    kind=kind,
                    value=value,
                    path=display_path,
                    location=pointer,
                )
            )
    return findings


def audit_package(
    package_dir: Path, *, display_root: str | None = None
) -> list[ResidueFinding]:
    """Audit every telemetry and JSON file inside one incident package."""
    root = display_root or package_dir.name
    findings: list[ResidueFinding] = []
    for path in sorted(package_dir.rglob("*")):
        display_path, name_findings = _audit_relative_name(
            f"{root}/{path.relative_to(package_dir).as_posix()}"
        )
        findings.extend(name_findings)
        if not path.is_file():
            continue
        if path.suffix.lower() == ".jsonl":
            findings.extend(audit_telemetry_file(path, display_path=display_path))
        elif path.suffix.lower() == ".json":
            findings.extend(audit_json_file(path, display_path=display_path))
    return findings


def audit_scenario_bank(bank_dir: Path) -> list[ResidueFinding]:
    """Audit every package of one scenario bank."""
    packages_dir = bank_dir / "packages"
    if not packages_dir.is_dir():
        return []
    findings: list[ResidueFinding] = []
    for package_dir in sorted(path for path in packages_dir.iterdir() if path.is_dir()):
        findings.extend(
            audit_package(
                package_dir, display_root=f"{bank_dir.name}/packages/{package_dir.name}"
            )
        )
    return findings


def audit_path(target: Path) -> list[ResidueFinding]:
    """Audit a scenario bank, an incident package, or a single file."""
    if target.is_file():
        display_path, findings = _audit_relative_name(target.name)
        if target.suffix.lower() == ".jsonl":
            return findings + audit_telemetry_file(target, display_path=display_path)
        return findings + audit_json_file(target, display_path=display_path)
    if (target / "packages").is_dir():
        return audit_scenario_bank(target)
    return audit_package(target)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m soc_llm_policy.publication_residue",
        description=(
            "Refuse institutional residue in data about to be published. "
            "Point it at a scenario bank, an incident package, or one file."
        ),
    )
    parser.add_argument(
        "target", nargs="?", type=Path, help="Bank, package, or file to audit."
    )
    parser.add_argument(
        "--release-root",
        type=Path,
        help="Audit every publishable data and output tree in an artifact package.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        help="Also reject private IPs shared with the authorized raw corpus.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Write the findings as JSON to this path.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if (args.target is None) == (args.release_root is None):
        parser.error("provide either a target or --release-root")
    target = (args.release_root or args.target).resolve()
    if not target.exists():
        parser.error("audit target does not exist")
    if args.release_root and not (target / "artifact_data").is_dir():
        parser.error("release root has no artifact_data tree")
    if args.raw_dir is not None and args.release_root is None:
        parser.error("--raw-dir requires --release-root")
    findings = (
        audit_release_residue(target) if args.release_root else audit_path(target)
    )
    if args.raw_dir is not None:
        findings.extend(audit_raw_private_ip_overlap(args.raw_dir, target))
    target_display, _ = _audit_relative_name(str(target))
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(
                {
                    "target": target_display,
                    "finding_count": len(findings),
                    "findings": [
                        {
                            "kind": finding.kind.value,
                            "path": finding.path,
                            "location": finding.location,
                        }
                        for finding in findings
                    ],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    if not findings:
        print(f"No institutional residue found in {target_display}.")
        return 0

    print(
        f"Institutional residue found in {target_display}: {len(findings)} finding(s)."
    )
    for finding in findings[:_REPORTED_FINDING_LIMIT]:
        print(f"  {finding.kind.value} at {finding.path} {finding.location}")
    if len(findings) > _REPORTED_FINDING_LIMIT:
        print(f"  ... {len(findings) - _REPORTED_FINDING_LIMIT} more; see --report.")
    return 1


_TEXT_SUFFIXES = frozenset(
    {".md", ".html", ".csv", ".tsv", ".txt", ".log", ".yaml", ".yml", ".tex"}
)
_RELEASE_SCANNED_TREES = ("artifact_data", "artifact_outputs", "scenario_banks", "docs")
_RENDERED_TEXT_EXEMPT_KINDS = frozenset({ResidueKind.operator_prose})
_RFC1918_NETWORKS = tuple(
    IPv4Network(network)
    for network in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")
)


def _rfc1918_addresses(path: Path) -> set[IPv4Address]:
    addresses: set[IPv4Address] = set()
    for value in _IPV4.findall(path.read_text(encoding="utf-8")):
        try:
            address = IPv4Address(value)
        except ValueError:
            continue
        if any(address in network for network in _RFC1918_NETWORKS):
            addresses.add(address)
    return addresses


def raw_private_addresses(raw_dir: Path) -> set[IPv4Address]:
    raw_files = sorted(raw_dir.glob("*.json"))
    if not raw_files:
        raise ValueError("Raw corpus has no JSON exports for overlap audit")
    return set().union(*(_rfc1918_addresses(path) for path in raw_files))


def audit_raw_private_ip_overlap(
    raw_dir: Path, release_root: Path
) -> list[ResidueFinding]:
    """Reject released private addresses found anywhere in the raw corpus."""
    raw_addresses = raw_private_addresses(raw_dir)
    findings: list[ResidueFinding] = []
    for tree in ("artifact_data", "artifact_outputs"):
        root = release_root / tree
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in _TEXT_SUFFIXES | {
                ".json",
                ".jsonl",
            }:
                continue
            if raw_addresses.isdisjoint(_rfc1918_addresses(path)):
                continue
            findings.append(
                ResidueFinding(
                    kind=ResidueKind.raw_private_ip_collision,
                    value="[raw private address collision]",
                    path=path.relative_to(release_root).as_posix(),
                    location="content",
                )
            )
    return findings


def _audit_text_file(path: Path, *, display_path: str) -> list[ResidueFinding]:
    findings: list[ResidueFinding] = []
    is_html = path.suffix.lower() == ".html"
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
    ):
        hits = scan_string(line, key="")
        if is_html:
            hits.extend(scan_string(html.unescape(line), key=""))
        for kind, value in dict.fromkeys(hits):
            if kind in _RENDERED_TEXT_EXEMPT_KINDS or (
                kind is ResidueKind.markup_entity and is_html
            ):
                continue
            findings.append(
                ResidueFinding(
                    kind=kind,
                    value=value,
                    path=display_path,
                    location=f"line {line_number}",
                )
            )
    return findings


def audit_release_residue(release_root: Path) -> list[ResidueFinding]:
    """Check package data, outputs, documentation and root reviewer surfaces.

    Rendered text files are exempt from two markers that are ordinary there:
    Portuguese function words, since reports quote rule rationale, and HTML
    entities inside HTML. Structured files are held to every rule.
    """
    findings: list[ResidueFinding] = []
    paths = [
        path
        for path in release_root.iterdir()
        if path.is_file()
        and path.suffix.lower() in _TEXT_SUFFIXES | {".json", ".jsonl"}
    ]
    for tree in _RELEASE_SCANNED_TREES:
        root = release_root / tree
        if root.is_dir():
            paths.extend(root.rglob("*"))
    for path in sorted(paths):
        display_path, name_findings = _audit_relative_name(
            path.relative_to(release_root).as_posix()
        )
        findings.extend(name_findings)
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            findings.extend(audit_telemetry_file(path, display_path=display_path))
        elif suffix == ".json":
            findings.extend(audit_json_file(path, display_path=display_path))
        elif suffix in _TEXT_SUFFIXES:
            findings.extend(_audit_text_file(path, display_path=display_path))
    return findings


if __name__ == "__main__":
    raise SystemExit(main())
