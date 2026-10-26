"""Project source correlation keys without changing frozen evidence."""

from __future__ import annotations

import json
import os
import re
import secrets
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from ipaddress import IPv4Address
from pathlib import Path
from types import MappingProxyType
from typing import Any

from soc_llm_policy.publication_residue import raw_private_addresses

# Letters only: digit runs inside an alias would read as phone numbers or case
# numbers to the package's own privacy audit.
_ALIAS_ALPHABET = "abcdefghjkmnpqrstuvwxyz"
_ALIAS_LENGTH = 20
_SYNTHETIC_IP_PREFIX = "10.255."
_SYNTHETIC_IP_BUCKET_SIZE = 250
_SYNTHETIC_IP_CAPACITY = 255 * _SYNTHETIC_IP_BUCKET_SIZE
_SYNTHETIC_IP_TOKEN = re.compile(r"(?<![\d.])10\.255\.(?:\d{1,3}\.)\d{1,3}(?![\d.])")

PROJECTED_TEXT_SUFFIXES = frozenset(
    {".html", ".json", ".jsonl", ".md", ".tex", ".yaml"}
)


def _parse_timestamp(value: str) -> datetime:
    timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        raise ValueError("Publication timestamps must include a timezone")
    return timestamp


def _format_shifted_timestamp(source: str, offset: timedelta) -> str:
    shifted = _parse_timestamp(source) + offset
    precision = "microseconds" if "." in source else "seconds"
    published = shifted.isoformat(timespec=precision)
    return published.replace("+00:00", "Z") if source.endswith("Z") else published


@dataclass(frozen=True)
class CorrelationProjection:
    """Private source-to-public aliases for incident IDs and exact timestamps."""

    incident_ids: tuple[tuple[str, str], ...]
    timestamps: tuple[tuple[str, str], ...]
    second_prefixes: tuple[tuple[str, str], ...] = ()
    export_names: tuple[tuple[str, str], ...] = ()
    ip_aliases: tuple[tuple[str, str], ...] = ()
    _replacements: Mapping[str, str] = field(init=False, repr=False, compare=False)
    _pattern: re.Pattern[str] = field(init=False, repr=False, compare=False)
    _ip_pattern: re.Pattern[str] | None = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        pairs = (
            *self.incident_ids,
            *self.timestamps,
            *self.second_prefixes,
            *self.export_names,
        )
        replacements = dict(pairs)
        if not pairs or len(replacements) != len(pairs):
            raise ValueError("Correlation aliases must be nonempty and unique")
        sources = set(replacements)
        targets = set(replacements.values())
        if (
            any(
                not source or not target or source == target for source, target in pairs
            )
            or len(targets) != len(pairs)
            or sources & targets
            or any("/" in target or "\x00" in target for target in targets)
        ):
            raise ValueError("Correlation aliases must be distinct and disjoint")
        for source, published in self.timestamps:
            _parse_timestamp(source)
            _parse_timestamp(published)
        ip_sources = {source for source, _ in self.ip_aliases}
        ip_targets = {target for _, target in self.ip_aliases}
        if (
            len(ip_sources) != len(self.ip_aliases)
            or len(ip_targets) != len(self.ip_aliases)
            or ip_sources & ip_targets
            or any(
                not source.startswith(_SYNTHETIC_IP_PREFIX)
                or not target.startswith(_SYNTHETIC_IP_PREFIX)
                or source == target
                for source, target in self.ip_aliases
            )
        ):
            raise ValueError("Synthetic IP aliases must be distinct")
        object.__setattr__(self, "_replacements", MappingProxyType(replacements))
        pattern = re.compile(
            "|".join(
                re.escape(source) for source in sorted(sources, key=len, reverse=True)
            )
        )
        object.__setattr__(self, "_pattern", pattern)
        ip_pattern = (
            re.compile(
                r"(?<![\d.])(?:"
                + "|".join(re.escape(source) for source in sorted(ip_sources))
                + r")(?![\d.])"
            )
            if ip_sources
            else None
        )
        object.__setattr__(self, "_ip_pattern", ip_pattern)

    def project_text(self, value: str) -> str:
        if self._ip_pattern is not None:
            ip_aliases = dict(self.ip_aliases)
            value = self._ip_pattern.sub(lambda match: ip_aliases[match.group()], value)
        return self._pattern.sub(lambda match: self._replacements[match.group()], value)

    def project_path(self, relative_path: Path) -> Path:
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative_path == Path(".")
        ):
            raise ValueError("Publication path must be relative to the package")
        return Path(*(self.project_text(part) for part in relative_path.parts))

    def project_json(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.project_text(value)
        if isinstance(value, list):
            return [self.project_json(item) for item in value]
        if not isinstance(value, dict):
            return value
        projected: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError("JSON object keys must be strings")
            published_key = self.project_text(key)
            if published_key in projected:
                raise ValueError("Correlation projection merges JSON object keys")
            projected[published_key] = self.project_json(child)
        return projected

    def verify_incident_timing(self, original_times: list[str]) -> None:
        """Require a constant offset within one incident's ordered events."""
        aliases = dict(self.timestamps)
        if not original_times:
            raise ValueError("Incident has no timestamps to verify")
        if any(original not in aliases for original in original_times):
            raise ValueError("Incident has a timestamp without a publication alias")
        source_start = _parse_timestamp(original_times[0])
        published_start = _parse_timestamp(aliases[original_times[0]])
        for original in original_times:
            source_time = _parse_timestamp(original)
            published_time = _parse_timestamp(aliases[original])
            if source_time - source_start != published_time - published_start:
                raise ValueError("Publication changes within-incident timing")


@dataclass(frozen=True)
class ProjectionStats:
    files_scanned: int
    files_rewritten: int
    paths_renamed: int
    links_retargeted: int


def project_package_trees(
    candidate_root: Path,
    projection: CorrelationProjection,
    trees: tuple[str, ...] = ("artifact_data", "scenario_banks", "artifact_outputs"),
) -> ProjectionStats:
    """Project an unsealed private staging copy before evidence digests are bound."""
    sealed_paths = (
        candidate_root / "artifact_manifest.json",
        candidate_root / "artifact_outputs/publication/redaction_record.json",
    )
    if any(path.exists() for path in sealed_paths):
        raise ValueError("Correlation projection requires an unsealed staging copy")
    paths: list[Path] = []
    files: list[Path] = []
    links: list[tuple[Path, str]] = []
    projected_paths: set[Path] = set()
    package_root = candidate_root.resolve()
    for tree_name in trees:
        tree_root = candidate_root / tree_name
        if not tree_root.is_dir():
            raise ValueError("Publication candidate is missing a required tree")
        for path in tree_root.rglob("*"):
            if path.is_symlink():
                target = os.readlink(path)
                resolved_target = (path.parent / target).resolve(strict=True)
                if Path(target).is_absolute() or not resolved_target.is_relative_to(
                    package_root
                ):
                    raise ValueError("Publication candidate has an external symlink")
                links.append((path, target))
            relative_path = path.relative_to(candidate_root)
            projected_path = projection.project_path(relative_path)
            if projected_path in projected_paths:
                raise ValueError("Correlation projection merges package paths")
            projected_paths.add(projected_path)
            paths.append(path)
            if path.is_file() and not path.is_symlink():
                if path.suffix.lower() not in PROJECTED_TEXT_SUFFIXES:
                    raise ValueError("Publication candidate has an unclassified file")
                files.append(path)
    if not files:
        raise ValueError("Publication candidate has no files to project")

    rewritten = 0
    for path in files:
        original = path.read_text(encoding="utf-8")
        published = projection.project_text(original)
        if published != original:
            path.write_text(published, encoding="utf-8")
            rewritten += 1

    retargeted = 0
    for path, target in links:
        published_target = projection.project_text(target)
        if published_target != target:
            replacement = path.with_name(path.name + ".publication-tmp")
            if replacement.exists() or replacement.is_symlink():
                raise ValueError("Temporary publication link already exists")
            replacement.symlink_to(published_target)
            replacement.replace(path)
            retargeted += 1

    renamed = 0
    for path in sorted(paths, key=lambda item: len(item.parts), reverse=True):
        published_name = projection.project_text(path.name)
        if published_name != path.name:
            target = path.with_name(published_name)
            if target.exists():
                raise ValueError("Correlation projection collides with package path")
            path.rename(target)
            renamed += 1
    return ProjectionStats(
        files_scanned=len(files),
        files_rewritten=rewritten,
        paths_renamed=renamed,
        links_retargeted=retargeted,
    )


def _second_prefixes(
    timestamps: list[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    """Map the seconds-resolution rendering of each fractional source time."""
    prefixes: dict[str, str] = {}
    for source, published in timestamps:
        if "." not in source:
            continue
        source_prefix, published_prefix = source[:19], published[:19]
        if prefixes.setdefault(source_prefix, published_prefix) != published_prefix:
            raise ValueError("A source second maps to two published seconds")
    return tuple(sorted(prefixes.items()))


def project_loose_files(
    candidate_root: Path,
    projection: CorrelationProjection,
    excluded_trees: tuple[str, ...],
) -> int:
    """Rewrite text files outside the projected trees; their paths carry no IDs."""
    rewritten = 0
    for path in sorted(candidate_root.rglob("*")):
        relative = path.relative_to(candidate_root)
        if (
            relative.parts[0] in excluded_trees
            or path.is_symlink()
            or not path.is_file()
            or path.suffix.lower() not in PROJECTED_TEXT_SUFFIXES
        ):
            continue
        if projection.project_path(relative) != relative:
            raise ValueError(
                "A file outside the projected trees has a source ID in its path"
            )
        original = path.read_text(encoding="utf-8")
        published = projection.project_text(original)
        if published != original:
            path.write_text(published, encoding="utf-8")
            rewritten += 1
    return rewritten


def write_private_map(projection: CorrelationProjection, path: Path) -> None:
    """Persist the crosswalk with owner-only permissions; never inside a package."""
    payload = {
        "incident_ids": dict(projection.incident_ids),
        "timestamps": dict(projection.timestamps),
        "ip_aliases": dict(projection.ip_aliases),
    }
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True)


def collision_free_ip_aliases(
    release_root: Path, raw_dir: Path
) -> tuple[tuple[str, str], ...]:
    """Reassign synthetic source IP aliases that coincide with raw addresses."""
    raw_addresses = {str(address) for address in raw_private_addresses(raw_dir)}
    source_aliases: set[str] = set()
    for telemetry_path in sorted(
        (release_root / "artifact_data").rglob("incident_telemetry.jsonl")
    ):
        for line in telemetry_path.read_text(encoding="utf-8").splitlines():
            address = json.loads(line).get("source_ip")
            if address is None:
                continue
            if not isinstance(address, str):
                raise ValueError("Publication source IP has an invalid type")
            if not address.startswith(_SYNTHETIC_IP_PREFIX):
                if re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", address):
                    raise ValueError("Publication source IP is not a synthetic alias")
                continue
            if not re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", address):
                raise ValueError("Publication source IP is not a synthetic alias")
            IPv4Address(address)
            source_aliases.add(address)
    if not source_aliases:
        raise ValueError("Publication has no source IP aliases to audit")
    collisions = sorted(source_aliases & raw_addresses)
    reserved = source_aliases | raw_addresses
    for tree_name in ("artifact_data", "scenario_banks", "artifact_outputs"):
        tree_root = release_root / tree_name
        if not tree_root.is_dir():
            continue
        for path in tree_root.rglob("*"):
            reserved.update(_SYNTHETIC_IP_TOKEN.findall(path.name))
            if path.is_file() and path.suffix.lower() in PROJECTED_TEXT_SUFFIXES:
                reserved.update(
                    _SYNTHETIC_IP_TOKEN.findall(path.read_text(encoding="utf-8"))
                )
    replacements: list[tuple[str, str]] = []
    for source in collisions:
        for index in range(_SYNTHETIC_IP_CAPACITY):
            candidate = (
                f"10.255.{index // _SYNTHETIC_IP_BUCKET_SIZE + 1}."
                f"{index % _SYNTHETIC_IP_BUCKET_SIZE + 1}"
            )
            if candidate not in reserved:
                reserved.add(candidate)
                replacements.append((source, candidate))
                break
        else:
            raise ValueError("Synthetic IP alias namespace exhausted")
    return tuple(replacements)


def _export_name_aliases(
    identifiers: list[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    """Alias the raw-export names, which carry the case number: ``incident_<n>``."""
    names: list[tuple[str, str]] = []
    for source, alias in identifiers:
        number = re.search(r"_(\d+)$", source)
        if number is not None:
            names.append(
                (
                    f"incident_{number.group(1)}",
                    f"incident_{alias.removeprefix('INC_PUBLIC_')}",
                )
            )
    return tuple(names)


def _shared_second_groups(
    incident_times: Mapping[str, list[str]],
) -> list[list[str]]:
    """Group incidents whose fractional events share a source second.

    Prompt previews render times at second resolution, so one source second
    must map to one published second: incidents that share one share an offset.
    """
    parent = {incident_id: incident_id for incident_id in incident_times}

    def root(incident_id: str) -> str:
        while parent[incident_id] != incident_id:
            parent[incident_id] = parent[parent[incident_id]]
            incident_id = parent[incident_id]
        return incident_id

    owner_of_second: dict[str, str] = {}
    for incident_id, observed_times in sorted(incident_times.items()):
        for timestamp in observed_times:
            if "." not in timestamp:
                continue
            owner = owner_of_second.setdefault(timestamp[:19], incident_id)
            parent[root(incident_id)] = root(owner)
    groups: dict[str, list[str]] = {}
    for incident_id in sorted(incident_times):
        groups.setdefault(root(incident_id), []).append(incident_id)
    return list(groups.values())


def _order_preserving_aliases(incident_ids: list[str]) -> list[tuple[str, str]]:
    """Random opaque aliases assigned in source order.

    Manifests require incident lists sorted by ID, so aliases keep the source
    order. They disclose an incident's rank in this corpus, not its number.
    """
    aliases: set[str] = set()
    while len(aliases) < len(incident_ids):
        alias = "INC_PUBLIC_" + "".join(
            secrets.choice(_ALIAS_ALPHABET) for _ in range(_ALIAS_LENGTH)
        )
        if alias in incident_ids:
            raise ValueError("Publication alias collides with a source incident ID")
        aliases.add(alias)
    return list(zip(sorted(incident_ids), sorted(aliases), strict=True))


def _random_past_offset() -> timedelta:
    return -timedelta(
        days=365 + secrets.randbelow(3286), seconds=secrets.randbelow(86400)
    )


def new_private_projection(
    incident_times: Mapping[str, list[str]],
    *,
    ip_aliases: tuple[tuple[str, str], ...] = (),
) -> CorrelationProjection:
    """Choose opaque aliases and past offsets; retain the result privately."""
    if not incident_times:
        raise ValueError("No incidents supplied for publication projection")
    source_times: set[str] = set()
    for incident_id, observed_times in sorted(incident_times.items()):
        if not incident_id or not observed_times:
            raise ValueError("Incident ID and timestamps must be nonempty")
        if source_times & set(observed_times):
            raise ValueError("Source time belongs to multiple incidents")
        source_times.update(observed_times)
    reserved = source_times | {timestamp[:19] for timestamp in source_times}

    identifiers = _order_preserving_aliases(list(incident_times))
    timestamps: list[tuple[str, str]] = []
    for group in _shared_second_groups(incident_times):
        group_times = {t for incident_id in group for t in incident_times[incident_id]}
        for _ in range(16):
            offset = _random_past_offset()
            shifted = {t: _format_shifted_timestamp(t, offset) for t in group_times}
            published = set(shifted.values()) | {t[:19] for t in shifted.values()}
            if not published & reserved:
                break
        else:
            raise ValueError("Could not create disjoint publication timestamps")
        reserved |= published
        timestamps.extend(sorted(shifted.items()))

    projection = CorrelationProjection(
        incident_ids=tuple(identifiers),
        timestamps=tuple(sorted(timestamps)),
        second_prefixes=_second_prefixes(timestamps),
        ip_aliases=ip_aliases,
        export_names=_export_name_aliases(identifiers),
    )
    for observed_times in incident_times.values():
        projection.verify_incident_timing(observed_times)
    return projection


def read_incident_times(source_data: Path) -> dict[str, list[str]]:
    """Read exact source times for each incident without changing source files."""
    incidents: dict[str, list[str]] = {}
    for metadata_path in sorted(source_data.rglob("incident_meta.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        incident_id = metadata.get("incident_id")
        if not isinstance(incident_id, str) or not incident_id:
            raise ValueError("Source incident metadata has no incident ID")
        if incident_id in incidents:
            raise ValueError("Source incident ID is duplicated")
        times = [
            metadata.get("time_window_start"),
            metadata.get("time_window_end"),
        ]
        telemetry_path = metadata_path.with_name("incident_telemetry.jsonl")
        if not telemetry_path.is_file():
            raise ValueError("Source incident has no telemetry")
        for line in telemetry_path.read_text(encoding="utf-8").splitlines():
            if line.strip() and "timestamp" in (event := json.loads(line)):
                times.append(event["timestamp"])
        # The export timestamp trails the incident by minutes; it is an incident time.
        source_manifest = metadata_path.parent / "evidence" / "source_manifest.json"
        if source_manifest.is_file():
            exported = json.loads(source_manifest.read_text(encoding="utf-8"))
            if "generated_at_utc" in exported:
                times.append(exported["generated_at_utc"])
        if any(not isinstance(value, str) or not value for value in times):
            raise ValueError("Source incident has an invalid timestamp")
        incidents[incident_id] = times
    if not incidents:
        raise ValueError("Source corpus has no incidents")
    return incidents
