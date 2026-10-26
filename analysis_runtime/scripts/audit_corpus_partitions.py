#!/usr/bin/env python3
"""Audit the canonical derivation and held-out corpus partitions.

The public path recomputes corpus metrics from the released canonical packages.
The private preflight additionally proves a bijection to the raw exports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from soc_llm_policy.corpus_hash import canonical_corpus_sha256
from soc_llm_policy.paths import resolve_repo_root

_PRIMARY_EVENT_TYPE = "command_execution_attempt"
_FORENSICS_ONLY_SEQUENCE = ("collect_forensics",)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        rows.append(row)
    return rows


def _canonical_incident_dirs(corpus_dir: Path) -> list[Path]:
    return sorted(path for path in corpus_dir.iterdir() if path.is_dir())


def _summary_for_corpus(corpus_dir: Path) -> dict[str, Any]:
    if not corpus_dir.is_dir():
        raise FileNotFoundError(
            f"Canonical corpus directory does not exist: {corpus_dir}"
        )

    incident_ids: list[str] = []
    primary_categories: set[str] = set()
    incident_type_counts: Counter[str] = Counter()
    sequence_counts: Counter[tuple[str, ...]] = Counter()
    assignment_modes: Counter[str] = Counter()

    for incident_dir in _canonical_incident_dirs(corpus_dir):
        incident_id = incident_dir.name
        legacy_mapping_path = (
            incident_dir / "evidence" / "mapping_support_manifest.json"
        )
        if legacy_mapping_path.exists():
            raise ValueError(
                f"{incident_id} contains a retired workflow-to-action manifest"
            )
        meta = _read_json(incident_dir / "incident_meta.json")
        if meta.get("incident_id") != incident_id:
            raise ValueError(
                f"Incident metadata id does not match directory: {incident_dir}"
            )
        incident_type = meta.get("incident_type")
        if not isinstance(incident_type, str) or not incident_type.strip():
            raise ValueError(f"{incident_id} has no incident type")
        incident_type_counts[incident_type] += 1

        telemetry = _read_jsonl(incident_dir / "incident_telemetry.jsonl")
        primary_events = [
            row for row in telemetry if row.get("event_type") == _PRIMARY_EVENT_TYPE
        ]
        if len(primary_events) != 1:
            raise ValueError(
                f"{incident_id} must contain exactly one {_PRIMARY_EVENT_TYPE} event"
            )
        category = primary_events[0].get("category")
        if not isinstance(category, str) or not category.strip():
            raise ValueError(f"{incident_id} has no primary detection category")
        primary_categories.add(category)

        actions = _read_jsonl(incident_dir / "incident_human_actions.jsonl")
        ordered_actions = sorted(actions, key=lambda row: int(row.get("order", 0)))
        action_sequence = tuple(
            str(row.get("action_id", "")) for row in ordered_actions
        )
        if any(not action_id for action_id in action_sequence):
            raise ValueError(f"{incident_id} has an incomplete action sequence")
        sequence_counts[action_sequence] += 1

        source_manifest = _read_json(incident_dir / "evidence" / "source_manifest.json")
        assignment_modes[
            str(source_manifest.get("incident_id_assignment", "legacy"))
        ] += 1
        incident_ids.append(incident_id)

    if not incident_ids:
        raise ValueError(f"Canonical corpus is empty: {corpus_dir}")
    canonical_ids_payload = "\n".join(incident_ids).encode("utf-8")
    return {
        "incident_count": len(incident_ids),
        "incident_type_count": len(incident_type_counts),
        "incident_type_counts": dict(sorted(incident_type_counts.items())),
        "primary_detection_category_count": len(primary_categories),
        "primary_detection_categories": sorted(primary_categories),
        "collect_forensics_only_incident_count": sequence_counts[
            _FORENSICS_ONLY_SEQUENCE
        ],
        "empty_action_profile_count": sequence_counts[()],
        "action_sequence_counts": {
            (" -> ".join(sequence) if sequence else "(empty)"): count
            for sequence, count in sorted(sequence_counts.items())
        },
        "incident_id_list_sha256": hashlib.sha256(canonical_ids_payload).hexdigest(),
        "corpus_sha256": canonical_corpus_sha256(corpus_dir),
        "source_assignment_modes": dict(sorted(assignment_modes.items())),
    }


def primary_detection_categories(corpus_dir: Path) -> list[str]:
    """The detection-category strings a corpus window actually contains.

    This is the raw vocabulary a trigger could match before normalization. It is
    exposed so the normalization ablation anchors its counterfactual on the same
    extraction this audit reports, rather than on a second reading of the corpus.
    """
    return _summary_for_corpus(corpus_dir)["primary_detection_categories"]


def _sha256_lines(lines: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(lines)).encode("utf-8")).hexdigest()


def _logical_prefix(path: Path, repo_root: Path, external_prefix: str) -> str:
    """Return the manifest identity for a local or externally mounted input."""
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return external_prefix


def _audit_source_lineage(
    *,
    repo_root: Path,
    derivation_dir: Path,
    heldout_dir: Path,
    raw_dir: Path | None,
) -> dict[str, Any]:
    """Prove that every private export maps to one canonical incident once."""
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw export directory does not exist: {raw_dir}")
    raw_prefix = _logical_prefix(raw_dir, repo_root, "incoming/raw")
    corpus_prefixes = {
        derivation_dir: _logical_prefix(
            derivation_dir, repo_root, "dataset/incidents"
        ),
        heldout_dir: _logical_prefix(
            heldout_dir, repo_root, "dataset/incidents_heldout"
        ),
    }

    raw_sources = {
        f"{raw_prefix}/{path.name}"
        for path in raw_dir.glob("*.json")
        if path.is_file()
    }
    if not raw_sources:
        raise ValueError(f"Raw export directory is empty: {raw_dir}")

    assignments: list[tuple[str, str]] = []
    for corpus_dir, corpus_prefix in corpus_prefixes.items():
        for incident_dir in _canonical_incident_dirs(corpus_dir):
            manifest = _read_json(incident_dir / "evidence" / "source_manifest.json")
            source = manifest.get("source_raw_file")
            if not isinstance(source, str) or not source.strip():
                raise ValueError(f"{incident_dir} is missing source_raw_file")
            source_path = Path(source)
            if source_path.is_absolute() or ".." in source_path.parts:
                raise ValueError(f"{incident_dir} has an unsafe source_raw_file")
            if source_path.parent.as_posix() != raw_prefix:
                raise ValueError(
                    f"{incident_dir} source_raw_file is outside the audited "
                    "raw directory"
                )
            expected_incident_dir = f"{corpus_prefix}/{incident_dir.name}"
            if manifest.get("incident_dir") != expected_incident_dir:
                raise ValueError(
                    f"{incident_dir} source manifest does not identify its "
                    "canonical directory"
                )
            assignments.append((expected_incident_dir, source_path.as_posix()))

    source_counts = Counter(source for _, source in assignments)
    duplicate_sources = sorted(
        source for source, count in source_counts.items() if count != 1
    )
    if duplicate_sources:
        raise ValueError("A raw export maps to multiple canonical incidents")
    canonical_sources = set(source_counts)
    unmatched_raw = raw_sources - canonical_sources
    unknown_sources = canonical_sources - raw_sources
    if unmatched_raw or unknown_sources:
        raise ValueError(
            "Raw export and canonical source-manifest sets differ: "
            f"unmatched_raw={len(unmatched_raw)}, "
            f"unknown_manifest_sources={len(unknown_sources)}"
        )

    return {
        "raw_export_count": len(raw_sources),
        "canonical_source_manifest_count": len(assignments),
        "unique_mapped_raw_export_count": len(canonical_sources),
        "duplicate_source_mapping_count": 0,
        "unmapped_raw_export_count": 0,
        "manifest_source_not_in_raw_count": 0,
        "raw_source_set_sha256": _sha256_lines(list(raw_sources)),
        "canonical_source_mapping_sha256": _sha256_lines(
            [f"{incident_dir}\0{source}" for incident_dir, source in assignments]
        ),
    }


def build_report(
    *,
    repo_root: Path,
    derivation_dir: Path,
    heldout_dir: Path,
    raw_dir: Path,
    expected_derivation_count: int,
    expected_heldout_count: int,
) -> dict[str, Any]:
    derivation = _summary_for_corpus(derivation_dir)
    heldout = _summary_for_corpus(heldout_dir)
    derivation_ids = {path.name for path in derivation_dir.iterdir() if path.is_dir()}
    heldout_ids = {path.name for path in heldout_dir.iterdir() if path.is_dir()}
    overlap = sorted(derivation_ids & heldout_ids)
    if overlap:
        raise ValueError(f"Corpus partitions share canonical ids: {overlap[:5]}")

    observed_total = derivation["incident_count"] + heldout["incident_count"]
    expected_total = expected_derivation_count + expected_heldout_count
    if derivation["incident_count"] != expected_derivation_count:
        raise ValueError(
            "Derivation corpus count mismatch: "
            f"expected {expected_derivation_count}, got {derivation['incident_count']}"
        )
    if heldout["incident_count"] != expected_heldout_count:
        raise ValueError(
            "Held-out corpus count mismatch: "
            f"expected {expected_heldout_count}, got {heldout['incident_count']}"
        )

    shared_categories = sorted(
        set(derivation["primary_detection_categories"])
        & set(heldout["primary_detection_categories"])
    )
    source_lineage = None
    if raw_dir is not None:
        source_lineage = _audit_source_lineage(
            repo_root=repo_root,
            derivation_dir=derivation_dir,
            heldout_dir=heldout_dir,
            raw_dir=raw_dir,
        )

    return {
        "schema_version": 2,
        "expected": {
            "derivation_incident_count": expected_derivation_count,
            "heldout_incident_count": expected_heldout_count,
            "total_incident_count": expected_total,
        },
        "observed": {
            "derivation": derivation,
            "heldout": heldout,
            "total_incident_count": observed_total,
        },
        # The count the transfer discovery rests on: a detection vocabulary that
        # barely overlaps across windows is why a string-anchored trigger dies and
        # a behavior-anchored one lives.
        "shared_primary_detection_category_count": len(shared_categories),
        "shared_primary_detection_categories": shared_categories,
        "source_lineage": source_lineage,
        "checks": {
            "canonical_ids_disjoint": True,
            "expected_counts_match": observed_total == expected_total,
            "raw_exports_map_bijectively": (
                True if source_lineage is not None else None
            ),
        },
    }


def render_macros(report: dict[str, Any]) -> str:
    """Return report values derived directly from the partition audit."""
    derivation = report["observed"]["derivation"]
    heldout = report["observed"]["heldout"]
    values = {
        "CorpusIncidents": report["observed"]["total_incident_count"],
        "CorpusDerivationIncidents": derivation["incident_count"],
        "CorpusHeldoutIncidents": heldout["incident_count"],
        "VocabDerivationForms": derivation["primary_detection_category_count"],
        "VocabHeldoutForms": heldout["primary_detection_category_count"],
        "VocabSharedForms": report["shared_primary_detection_category_count"],
        "DerivationIncidentTypes": derivation["incident_type_count"],
        "HeldoutIncidentTypes": heldout["incident_type_count"],
        "DerivationForensicsOnly": derivation["collect_forensics_only_incident_count"],
        "HeldoutForensicsOnly": heldout["collect_forensics_only_incident_count"],
        "DerivationEmptyActionProfiles": derivation["empty_action_profile_count"],
        "HeldoutEmptyActionProfiles": heldout["empty_action_profile_count"],
    }
    return "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in values.items())


def _render_markdown(report: dict[str, Any]) -> str:
    derivation = report["observed"]["derivation"]
    heldout = report["observed"]["heldout"]
    lines = [
        "# Corpus Partition Audit",
        "",
        "Aggregate audit of the canonical policy-development partitions. "
        "It exposes no incident content or source filenames.",
        "",
        (
            "| Partition | Incidents | Primary categories | "
            "Forensics-only profiles | Empty profiles | Corpus hash |"
        ),
        "| --- | ---: | ---: | ---: | ---: | --- |",
        (
            "| Derivation | {incidents} | {categories} | {forensics_only} | "
            "{empty} | `{hash}` |"
        ).format(
            incidents=derivation["incident_count"],
            categories=derivation["primary_detection_category_count"],
            forensics_only=derivation["collect_forensics_only_incident_count"],
            empty=derivation["empty_action_profile_count"],
            hash=derivation["corpus_sha256"][:12],
        ),
        (
            "| Held-out | {incidents} | {categories} | {forensics_only} | "
            "{empty} | `{hash}` |"
        ).format(
            incidents=heldout["incident_count"],
            categories=heldout["primary_detection_category_count"],
            forensics_only=heldout["collect_forensics_only_incident_count"],
            empty=heldout["empty_action_profile_count"],
            hash=heldout["corpus_sha256"][:12],
        ),
        "",
        (
            "- Total canonical incidents: "
            f"`{report['observed']['total_incident_count']}`"
        ),
        "- Canonical identifiers are disjoint across partitions: `true`",
    ]
    lineage = report.get("source_lineage")
    if isinstance(lineage, dict):
        lines.append(
            "- Raw-to-canonical lineage: "
            f"`{lineage['raw_export_count']}` raw exports, "
            f"`{lineage['canonical_source_manifest_count']}` canonical "
            "manifests, one-to-one."
        )
    else:
        lines.append(
            "- Raw-to-canonical lineage is a private preflight check; this "
            "public recomputation audits the released canonical partitions."
        )
    lines.extend(
        [
            "- Primary category means the one command-execution-attempt event "
            "recorded per canonical incident.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--derivation-dir", default="dataset/incidents")
    parser.add_argument("--heldout-dir", default="dataset/incidents_heldout")
    parser.add_argument(
        "--raw-dir",
        default=None,
        help="Optional private raw-export directory for bijection checking.",
    )
    parser.add_argument("--expected-derivation-count", type=int, default=200)
    parser.add_argument("--expected-heldout-count", type=int, default=200)
    parser.add_argument("--macros-output")
    parser.add_argument(
        "--output-json", default="results/analysis/corpus_partition_audit.json"
    )
    parser.add_argument(
        "--output-md", default="results/analysis/corpus_partition_audit.md"
    )
    args = parser.parse_args()
    repo_root = resolve_repo_root(args.repo_root)
    report = build_report(
        repo_root=repo_root,
        derivation_dir=(repo_root / args.derivation_dir).resolve(),
        heldout_dir=(repo_root / args.heldout_dir).resolve(),
        raw_dir=(repo_root / args.raw_dir).resolve() if args.raw_dir else None,
        expected_derivation_count=args.expected_derivation_count,
        expected_heldout_count=args.expected_heldout_count,
    )
    output_json = (repo_root / args.output_json).resolve()
    output_markdown = (repo_root / args.output_md).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    output_markdown.write_text(_render_markdown(report), encoding="utf-8")
    if args.macros_output:
        macros = Path(args.macros_output)
        macros.parent.mkdir(parents=True, exist_ok=True)
        macros.write_text(
            "% Generated by audit_corpus_partitions.py\n"
            + render_macros(report)
            + "\n",
            encoding="utf-8",
        )
        print(f"Wrote macros: {macros}")
    print(f"Wrote corpus partition audit: {output_json}")


if __name__ == "__main__":
    main()
