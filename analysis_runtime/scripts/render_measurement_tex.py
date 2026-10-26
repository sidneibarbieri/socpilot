"""Render measurement-run tables from preserved analysis outputs.

The paper consumes artifact evidence and hand-types no number. Inputs are the
run-filtered analysis bundle (per-cell rows, by-model and by-rule aggregates
with incident-clustered CIs) and the run ledger (typed failures with attempt
counts). Outputs are LaTeX table bodies plus macros, each stamped with the
input hashes so every number traces to one evaluation state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from soc_llm_policy.io import parse_rules, read_yaml_list

_MIN_DISPLAYED_PROBABILITY = 0.000001

_FAMILY_LABEL = {
    "mandatory": "Mandatory",
    "prohibit_when": "Prohibition",
    "prohibit_before": "Ordering",
    "require_approval": "Approval",
}
_LEGACY_STRATUM_ALIASES = {
    "derivation": "export_batch_1",
    "reserved_replication": "export_batch_2",
}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model_arm(label: str) -> tuple[str, str]:
    deployment, _, arm = label.partition("|")
    arm_label = {
        "llm_zero": "no policy",
        "llm_policy_prompt": "inline policy",
    }.get(arm, arm)
    return deployment, arm_label


def _canonical_stratum(value: object) -> str:
    raw_value = str(value or "unpartitioned")
    return _LEGACY_STRATUM_ALIASES.get(raw_value, raw_value)


def apply_incident_partitions(bundle: dict, incident_partitions: dict) -> dict:
    """Attach acquisition-batch strata to a pre-campaign bundle.

    Bundles frozen before per-cell stratum threading carry corpus_stratum None
    on every row, yet each incident's batch is frozen in the campaign partition
    map, so the join is deterministic. Relabeling aggregate rows is sound only
    when every incident in the bundle belongs to a single batch; a mixed bundle
    needs re-aggregation, not relabeling, so that case raises.
    """
    sections = ("incidents", "by_model", "by_rule", "pairwise")
    if any(
        row.get("corpus_stratum") is not None
        for section in sections
        for row in bundle.get(section, [])
    ):
        return bundle
    incident_ids = {row["incident_id"] for row in bundle["incidents"]}
    missing = sorted(incident_ids - set(incident_partitions))
    if missing:
        raise ValueError(
            "Bundle incidents are absent from the partition map: "
            + ", ".join(missing[:5])
        )
    strata = {
        _canonical_stratum(incident_partitions[incident_id])
        for incident_id in incident_ids
    }
    if len(strata) != 1:
        raise ValueError(
            "Bundle spans multiple acquisition batches; relabeling aggregates "
            "would misattribute them: " + ", ".join(sorted(strata))
        )
    stratum = strata.pop()
    for section in sections:
        for row in bundle.get(section, []):
            row["corpus_stratum"] = stratum
    bundle["summary"]["corpus_strata"] = {
        stratum: {"incident_count": len(incident_ids)}
    }
    return bundle


def apply_incident_partitions_to_entries(
    entries: list[dict], path: Path
) -> list[dict]:
    """Attach batch strata to ledger entries frozen before stratum threading.

    Ledger entries carry their incident, so attribution is exact per entry;
    unlike bundle aggregates, no single-batch precondition is needed. Without
    this, a stratified table tallies refusals under "unpartitioned" and every
    batch row silently reads zero.
    """
    if all(entry.get("corpus_stratum") is not None for entry in entries):
        return entries
    if not path.is_file():
        raise FileNotFoundError(
            f"Ledger entries carry no strata and no partition map exists at: {path}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    partitions = payload.get("incident_partitions", payload)
    relabeled = []
    for entry in entries:
        if entry.get("corpus_stratum") is not None:
            relabeled.append(entry)
            continue
        incident_id = entry.get("incident_id")
        if incident_id not in partitions:
            raise ValueError(
                f"Ledger incident is absent from the partition map: {incident_id}"
            )
        relabeled.append({**entry, "corpus_stratum": partitions[incident_id]})
    return relabeled


def apply_incident_partitions_from_path(bundle: dict, path: Path) -> dict:
    """Join batch strata from a corpus-campaign manifest on disk.

    An unstratified bundle without a partition map would silently render every
    stratum as "unpartitioned" and zero out the per-batch tables, so that
    combination raises instead of degrading.
    """
    if any(
        row.get("corpus_stratum") is not None
        for section in ("incidents", "by_model", "by_rule", "pairwise")
        for row in bundle.get(section, [])
    ):
        return bundle
    if not path.is_file():
        raise FileNotFoundError(
            f"Bundle carries no strata and no partition map exists at: {path}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    partitions = payload.get("incident_partitions", payload)
    return apply_incident_partitions(bundle, partitions)


def _stratum_label(value: object) -> str:
    return {
        "all_incidents": "All incidents",
        "export_batch_1": "Batch 1",
        "export_batch_2": "Batch 2",
    }.get(_canonical_stratum(value), _canonical_stratum(value))


def _stratum_summary(corpus_strata: dict, canonical_name: str) -> dict:
    if canonical_name in corpus_strata:
        return corpus_strata[canonical_name]
    legacy_name = next(
        (
            name
            for name, canonical in _LEGACY_STRATUM_ALIASES.items()
            if canonical == canonical_name
        ),
        "",
    )
    return corpus_strata.get(legacy_name, {})


@dataclass(frozen=True)
class PrimaryRateRow:
    stratum: str
    model: str
    condition: str
    cells: int
    violations: int
    rate: float
    ci_low: float
    ci_high: float
    refusals: int

    @classmethod
    def from_bundle_row(cls, row: dict, refusals: int) -> PrimaryRateRow:
        model, condition = _model_arm(row["model_label"])
        cells = int(row["run_count"])
        violations = int(row["runs_with_violations"])
        ci_low = float(row["run_violation_rate_ci_low_clustered"])
        ci_high = float(row["run_violation_rate_ci_high_clustered"])
        if cells < 1 or not 0 <= violations <= cells or refusals < 0:
            raise ValueError("Invalid primary measurement counts.")
        rate = violations / cells
        if not 0 <= ci_low <= rate <= ci_high <= 1:
            raise ValueError("Invalid primary measurement confidence interval.")
        return cls(
            stratum=_canonical_stratum(row.get("corpus_stratum")),
            model=model,
            condition=condition,
            cells=cells,
            violations=violations,
            rate=rate,
            ci_low=ci_low,
            ci_high=ci_high,
            refusals=refusals,
        )


def primary_rate_rows(bundle: dict, ledger_entries: list[dict]) -> list[PrimaryRateRow]:
    refusals_by_label: dict[tuple[str, str], int] = {}
    for entry in ledger_entries:
        if entry.get("exception_type") != "LLMRefusalError":
            continue
        label = f"{entry.get('deployment')}|{entry.get('arm')}"
        stratum = _canonical_stratum(entry.get("corpus_stratum"))
        for target in (stratum, "all_incidents") if stratum != "unpartitioned" else (stratum,):
            key = (target, label)
            refusals_by_label[key] = refusals_by_label.get(key, 0) + 1
    return [
        PrimaryRateRow.from_bundle_row(
            row,
            refusals_by_label.get(
                (_canonical_stratum(row.get("corpus_stratum")), row["model_label"]), 0
            ),
        )
        for row in bundle["by_model"]
    ]


def _display_model(model: str) -> str:
    return {"gpt-5.4": "GPT-5.4", "qwen3.6:latest": "Qwen3.6"}.get(
        model, model
    )


def render_primary_points(rows: list[PrimaryRateRow]) -> str:
    by_key = {(row.stratum, row.model, row.condition): row for row in rows}
    if len(by_key) != len(rows):
        raise ValueError("Duplicate primary rate row.")
    lines = []
    for stratum in ("all_incidents", "export_batch_1", "export_batch_2"):
        for model in ("gpt-5.4", "qwen3.6:latest"):
            no_policy_key = (stratum, model, "no policy")
            policy_key = (stratum, model, "inline policy")
            if no_policy_key not in by_key or policy_key not in by_key:
                raise ValueError(
                    f"Primary rate plot lacks a condition for {stratum}, {model}."
                )
            no_policy = by_key[no_policy_key]
            policy = by_key[policy_key]
            lines.append(
                rf"\primaryrate{{{_stratum_label(stratum)}}}{{{_display_model(model)}}}"
                rf"{{{no_policy.rate:.3f}}}{{{no_policy.ci_low:.2f}}}{{{no_policy.ci_high:.2f}}}"
                rf"{{{policy.rate:.3f}}}{{{policy.ci_low:.2f}}}{{{policy.ci_high:.2f}}}"
            )
    if len(rows) != 3 * 2 * 2:
        raise ValueError("Primary rate plot requires exactly twelve source rows.")
    return "\n".join(lines)


def render_primary_table(bundle: dict, ledger_entries: list[dict]) -> str:
    """One row per model and arm: cells that produced a plan, violating-cell
    rate with the incident-clustered CI, and refusals held out separately.

    Emits the complete tabular environment: TeX alignment scanning does not
    tolerate an input-file boundary inside tabular.
    """
    rows = primary_rate_rows(bundle, ledger_entries)
    lines = [
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lllrrr@{}}",
        r"\toprule",
        r"\textbf{Stratum} & \textbf{Model} & \textbf{Condition} & \textbf{Cells} & \textbf{Violation rate (\%) [95\% CI]} & \textbf{Refusals} \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{_stratum_label(row.stratum)} & {_display_model(row.model)} & "
            f"{row.condition} & {row.cells} & {100 * row.rate:.1f} "
            f"[{100 * row.ci_low:.1f}, {100 * row.ci_high:.1f}] & "
            f"{row.refusals} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular*}"]
    return "\n".join(lines)



def render_prompt_contrasts(bundle: dict) -> str:
    """Render preserved paired tests without recomputing their adjustment family."""
    if bundle["summary"].get("repeat_reduction") != "majority":
        raise ValueError("The manuscript contrast table requires majority reduction.")
    rows = [row for row in bundle["pairwise"]
            if row["contrast_type"] == "prompt_condition"]
    if not rows:
        raise ValueError("No paired prompt-condition contrasts are available.")
    lines = [
        r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}llrrr@{}}",
        r"\toprule",
        r"\textbf{Stratum} & \textbf{Model} & \textbf{Pairs} & \textbf{$\Delta$ (pp)} & \textbf{$p_{\mathrm{Holm}}$} \\",
        r"\midrule",
    ]
    for row in rows:
        deployment, arm = _model_arm(row["model_a"])
        other_deployment, other_arm = _model_arm(row["model_b"])
        if deployment != other_deployment or (arm, other_arm) != ("inline policy", "no policy"):
            raise ValueError("Contrast direction must be inline policy minus no policy.")
        model = {"gpt-5.4": "GPT-5.4", "qwen3.6:latest": "Qwen3.6"}[deployment]
        probability = float(row["mcnemar_p_value_holm"])
        if not 0 <= probability <= 1:
            raise ValueError("Adjusted probability must be between zero and one.")
        probability_tex = r"$<10^{-6}$" if probability < _MIN_DISPLAYED_PROBABILITY else f"{probability:.6f}".rstrip("0").rstrip(".")
        difference = 100 * row["rate_diff_a_minus_b"]
        lines.append(
            f"{_stratum_label(row['corpus_stratum'])} & {model} & "
            f"{row['paired_incident_count']} & {difference:+.2f} & {probability_tex} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular*}"]
    return "\n".join(lines)

def render_family_table(bundle: dict, constraints_path: Path) -> str:
    """Violations by rule family with the rules that produced them."""
    family_of = {
        rule.rule_id: _FAMILY_LABEL[rule.type]
        for rule in parse_rules(read_yaml_list(constraints_path))
    }
    violations_by_family: dict[tuple[str, str], int] = {}
    rules_by_family: dict[tuple[str, str], set[str]] = {}
    for row in bundle["by_rule"]:
        violation_count = int(row["violation_count"])
        if violation_count < 0:
            raise ValueError("Violation counts must be non-negative.")
        if row["rule_id"] not in family_of:
            raise ValueError("Measurement row references an unknown reviewed rule.")
        family = family_of[row["rule_id"]]
        key = (_stratum_label(row.get("corpus_stratum")), family)
        violations_by_family[key] = violations_by_family.get(key, 0) + violation_count
        if violation_count > 0:
            rules_by_family.setdefault(key, set()).add(str(row["rule_id"]))
    families = ("Mandatory", "Prohibition", "Ordering", "Approval")
    # Exact counts keep sparse families legible without proportional bars.
    lines = [
        r"\begin{tabular*}{\columnwidth}"
        r"{@{}ll@{\extracolsep{\fill}}r@{\extracolsep{0pt}\hspace{2em}}r@{}}",
        r"\toprule",
        r"\textbf{Stratum} & \textbf{Rule family} & \textbf{Rules violated} & \textbf{Violations} \\",
        r"\midrule",
    ]
    strata = sorted({stratum for stratum, _family in violations_by_family})
    for stratum in strata or ["Unpartitioned"]:
        for family in families:
            key = (stratum, family)
            lines.append(
                f"{stratum} & {family} & {len(rules_by_family.get(key, set()))} & "
                f"{violations_by_family.get(key, 0)} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular*}"]
    return "\n".join(lines)


def render_incident_type_table(bundle: dict) -> str:
    """Render cell support and violation rates by incident type and stratum."""
    counts: dict[tuple[str, str], list[int]] = {}
    for row in bundle["incidents"]:
        key = (
            _canonical_stratum(row.get("corpus_stratum")),
            str(row.get("incident_type") or "unknown"),
        )
        bucket = counts.setdefault(key, [0, 0])
        bucket[0] += 1
        bucket[1] += int(int(row.get("violation_count", 0)) > 0)

    incident_types = sorted({incident_type for _stratum, incident_type in counts})
    has_batches = any(
        stratum == "export_batch_2" for stratum, _incident_type in counts
    )
    lines = (
        [
            r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrrrr@{}}",
            r"\toprule",
            r"& \multicolumn{2}{c}{\textbf{Batch 1}} & \multicolumn{2}{c}{\textbf{Batch 2}} \\",
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
            r"\textbf{Incident type} & \textbf{Cells} & \textbf{Violating} & \textbf{Cells} & \textbf{Violating} \\",
            r"\midrule",
        ]
        if has_batches
        else [
            r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrr@{}}",
            r"\toprule",
            r"\textbf{Incident type} & \textbf{Cells} & \textbf{Violating} \\",
            r"\midrule",
        ]
    )
    for incident_type in incident_types:
        batch_one_total, batch_one_violations = counts.get(
            ("export_batch_1", incident_type),
            [0, 0],
        )
        batch_two_total, batch_two_violations = counts.get(
            ("export_batch_2", incident_type),
            [0, 0],
        )
        batch_one_rate = (
            f"{100 * batch_one_violations / batch_one_total:.1f}\\%"
            if batch_one_total
            else "--"
        )
        batch_two_rate = (
            f"{100 * batch_two_violations / batch_two_total:.1f}\\%"
            if batch_two_total
            else "--"
        )
        label = incident_type.replace("_", r"\_")
        if has_batches:
            lines.append(
                f"{label} & {batch_one_total or '--'} & {batch_one_rate} & "
                f"{batch_two_total or '--'} & {batch_two_rate} \\\\"
            )
        else:
            lines.append(f"{label} & {batch_one_total} & {batch_one_rate} \\\\")
    lines += [r"\bottomrule", r"\end{tabular*}"]
    return "\n".join(lines)


def _campaign_rows(rows: list[dict]) -> list[dict]:
    """Rows that describe the whole campaign rather than one batch.

    Stratified bundles carry explicit pooled rows; a bundle whose rows all
    belong to one batch is its own campaign aggregate. A multi-batch bundle
    without pooled rows has no campaign-level row to report, and inventing one
    by pooling here would bypass the incident-clustered statistics.
    """
    if not rows:
        return []
    pooled = [
        row
        for row in rows
        if str(row.get("corpus_stratum") or "unpartitioned")
        in {"all_incidents", "unpartitioned"}
    ]
    if pooled:
        return pooled
    strata = {_canonical_stratum(row.get("corpus_stratum")) for row in rows}
    if len(strata) == 1:
        return list(rows)
    raise ValueError(
        "No campaign-level rows: bundle spans strata "
        + ", ".join(sorted(strata))
        + " without a pooled aggregate."
    )


def _transport_recovered(ledger_entries: list[dict]) -> int:
    transport = {
        "RemoteDisconnected",
        "TimeoutError",
        "ConnectionResetError",
        "ConnectionError",
        "IncompleteRead",
        "URLError",
        "HTTPError",
    }
    return sum(1 for e in ledger_entries if e.get("exception_type") in transport)


def render_macros(bundle: dict, ledger_entries: list[dict]) -> str:
    refusals = sum(
        1 for e in ledger_entries if e.get("exception_type") == "LLMRefusalError"
    )
    rows = bundle["incidents"]
    hallucinated = sum(int(r.get("llm_hallucinated_action_count", 0)) for r in rows)
    cost = sum(float(r.get("llm_cost_estimated_usd", 0.0)) for r in rows)
    inserted = removed = 0
    for row in rows:
        inserted += int(row.get("enforcement_actions_inserted_count", 0))
        removed += int(row.get("enforcement_actions_removed_count", 0))
    # The headline violation range is the spread of the per-cell point estimates,
    # recomputed from the integer counts. It is reported to one decimal because
    # whole-percent rounding of the maximum (72.50) would print a bound the data
    # exceeds, and a range whose upper bound excludes an observed cell is wrong.
    campaign_rows = _campaign_rows(bundle["by_model"])
    cell_rates = [
        row["runs_with_violations"] / row["run_count"]
        for row in campaign_rows
        if row.get("run_count")
    ]
    corpus_strata = bundle["summary"].get("corpus_strata", {})
    values = {
        "PlanCells": len(rows),
        "RefusalCells": refusals,
        "TransportRecoveredCells": _transport_recovered(ledger_entries),
        "HallucinatedActions": hallucinated,
        "InsertedActions": inserted,
        "RemovedActions": removed,
        "TotalCostUsd": f"{cost:.2f}",
        "IncidentCount": bundle["summary"].get("incident_count", 0),
        "BatchOneIncidentCount": _stratum_summary(
            corpus_strata, "export_batch_1"
        ).get("incident_count", 0),
        "BatchTwoIncidentCount": _stratum_summary(
            corpus_strata, "export_batch_2"
        ).get("incident_count", 0),
        "ViolationRateMinPct": f"{min(cell_rates) * 100:.1f}",
        "ViolationRateMaxPct": f"{max(cell_rates) * 100:.1f}",
    }
    return "\n".join(
        f"\\newcommand{{\\Measurement{name}}}{{{value}}}"
        for name, value in values.items()
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-json", required=True)
    parser.add_argument("--ledger-json", required=True)
    parser.add_argument(
        "--constraints",
        default="policy/constraints.yaml",
    )
    parser.add_argument(
        "--incident-partitions-json",
        default="results/policy_preflight/canonical_campaign/corpus_campaign.json",
        help="Corpus-campaign manifest used to attach batch strata to bundles "
        "frozen before per-cell stratum threading.",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    bundle_path = Path(args.bundle_json)
    ledger_path = Path(args.ledger_json)
    partitions_path = Path(args.incident_partitions_json)
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    bundle = apply_incident_partitions_from_path(bundle, partitions_path)
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger_entries = (
        ledger
        if isinstance(ledger, list)
        else ledger.get("entries", ledger.get("cells", []))
    )
    ledger_entries = apply_incident_partitions_to_entries(
        ledger_entries, partitions_path
    )

    header = (
        "% Generated by render_measurement_tex.py\n"
        f"% bundle {_sha256_file(bundle_path)[:12]}, "
        f"ledger {_sha256_file(ledger_path)[:12]}\n"
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "measurement_primary_table.tex": render_primary_table(bundle, ledger_entries),
        "measurement_primary_points.tex": render_primary_points(
            primary_rate_rows(bundle, ledger_entries)
        ),
        "measurement_family_table.tex": render_family_table(
            bundle, Path(args.constraints)
        ),
        "measurement_incident_type_table.tex": render_incident_type_table(bundle),
        "measurement_macros.tex": render_macros(bundle, ledger_entries),
    }
    if bundle.get("pairwise"):
        outputs["measurement_prompt_contrasts.tex"] = render_prompt_contrasts(bundle)
    for name, body in outputs.items():
        (output_dir / name).write_text(header + body + "\n", encoding="utf-8")
        print(f"Wrote {output_dir / name}")


if __name__ == "__main__":
    main()
