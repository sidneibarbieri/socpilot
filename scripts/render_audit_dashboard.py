#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
from typing import Any

import yaml

from soc_llm_policy.policy_predicates import predicate_digest
from soc_llm_policy.preflight_evidence import validate_policy_preflight_reports

_FAMILY_LABEL = {
    "mandatory": "Mandatory",
    "prohibit_when": "Restraint",
    "prohibit_before": "Ordering",
    "require_approval": "Approval",
}
_SHA256_LENGTH = 64


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must be a JSON object")
    return payload


def _read_json_optional(path: Path) -> dict[str, Any] | None:
    """Read a report that may not ship on every artifact path (H2/H3/probe)."""
    if not path.is_file():
        return None
    return _read_json(path)


def _safe_float(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool):
        raise ValueError(f"Expected a numeric value, got {value!r}.")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(f"Expected a numeric value, got {value!r}.") from exc
    raise ValueError(f"Expected a numeric value, got {value!r}.")


def _safe_int(value: Any) -> int:
    return round(_safe_float(value))


def _pct(value: Any) -> str:
    return "N/A" if value is None else f"{_safe_float(value) * 100:.1f}%"


def _num(value: Any) -> str:
    return f"{_safe_int(value):,}"


def _money(value: Any) -> str:
    return f"${_safe_float(value):.4f}"


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _load_dashboard_data(
    analysis_dir: Path,
    policy_preflight_dir: Path | None = None,
) -> dict[str, Any]:
    protocol_freeze = _read_json(analysis_dir / "protocol_freeze.json")
    runs = _read_json(analysis_dir / "official_runs_manifest.json")
    summary = _read_json(analysis_dir / "official_evaluation_summary.json")
    pairwise = _read_json(analysis_dir / "official_pairwise_tests.json")
    readiness = _read_json(analysis_dir / "corpus_readiness.json")
    assessment = _read_json(analysis_dir / "global_artifact_assessment.json")
    consistency = _read_json(analysis_dir / "official_public_consistency.json")
    stability = _read_json_optional(
        analysis_dir / "repeat_stability" / "repeat_stability_summary.json"
    )
    return {
        "protocol_freeze": protocol_freeze,
        "runs": runs,
        "summary": summary,
        "pairwise": pairwise,
        "readiness": readiness,
        "assessment": assessment,
        "consistency": consistency,
        "stability": stability,
        "separation": _read_json_optional(analysis_dir / "separation_report.json"),
        "adversarial": _read_json_optional(analysis_dir / "adversarial_report.json"),
        "fidelity": _read_json_optional(
            analysis_dir / "projection_fidelity_report.json"
        ),
        "projection": _read_json_optional(analysis_dir / "projection_diagnostics.json"),
        "normalization_ablation": (
            _read_json_optional(policy_preflight_dir / "normalization_ablation.json")
            if policy_preflight_dir
            else None
        ),
        "heldout_activation": (
            _read_json_optional(policy_preflight_dir / "heldout_activation.json")
            if policy_preflight_dir
            else None
        ),
        "partition_audit": (
            _read_json_optional(policy_preflight_dir / "corpus_partition_audit.json")
            if policy_preflight_dir
            else None
        ),
        "policy_preflight_dir": policy_preflight_dir,
    }


def _frozen_constraints_hash(data: dict[str, Any]) -> str:
    protocol_freeze = data.get("protocol_freeze")
    if not isinstance(protocol_freeze, dict):
        raise ValueError("Dashboard data is missing the protocol freeze manifest.")
    input_hashes = protocol_freeze.get("input_hashes")
    if not isinstance(input_hashes, dict):
        raise ValueError("Protocol freeze manifest is missing input hashes.")
    constraints = input_hashes.get("constraints")
    if not isinstance(constraints, dict):
        raise ValueError("Protocol freeze manifest is missing the constraints hash.")
    expected_hash = constraints.get("sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != _SHA256_LENGTH:
        raise ValueError("Protocol freeze manifest has an invalid constraints hash.")
    return expected_hash


def _validate_policy_alignment(data: dict[str, Any], constraints_path: Path) -> str:
    expected_hash = _frozen_constraints_hash(data)
    if not constraints_path.is_file():
        raise FileNotFoundError(
            f"Policy constraints file not found: {constraints_path}"
        )
    actual_hash = hashlib.sha256(constraints_path.read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError(
            "Dashboard policy hash does not match the preserved evidence. "
            "Render with the policy recorded by protocol_freeze.json or rerun "
            "the evaluation under the selected policy bundle."
        )
    return expected_hash


def _validate_preflight_evidence(data: dict[str, Any], predicate_sha256: str) -> None:
    """Reject policy-development reports that decide over other predicates.

    The file hash already pinned the policy this run consumed. These reports
    read predicates, so a corrected provenance sentence must not invalidate
    them, and a widened scope must.
    """
    activation = data.get("heldout_activation")
    partition = data.get("partition_audit")
    normalization = data.get("normalization_ablation")
    if activation is None and partition is None and normalization is None:
        return
    if (
        not isinstance(activation, dict)
        or not isinstance(partition, dict)
        or not isinstance(normalization, dict)
    ):
        raise ValueError(
            "Policy-development evidence must include held-out activation, "
            "partition audit, and normalization ablation reports."
        )
    validate_policy_preflight_reports(
        heldout_activation=activation,
        partition_audit=partition,
        normalization_ablation=normalization,
        expected_predicate_sha256=predicate_sha256,
    )


def _run_success_rate(accounting: dict[str, Any]) -> float:
    completed = _safe_float(accounting.get("llm_trajectories_completed"))
    failed = _safe_float(accounting.get("llm_trajectories_failed_total"))
    total = completed + failed
    return completed / total if total else 0.0


def _summary_metrics(data: dict[str, Any]) -> list[tuple[str, str, str]]:
    runs = data["runs"]
    summary = data["summary"]
    readiness = data["readiness"]
    consistency = data["consistency"]
    accounting = runs.get("execution_accounting", {})
    return [
        (
            "Public evaluation corpus",
            _num(readiness.get("incident_count")),
            "model-evaluation incidents",
        ),
        (
            "Completed runs",
            _num(accounting.get("llm_trajectories_completed")),
            "LLM trajectories",
        ),
        ("Run success", _pct(_run_success_rate(accounting)), "official manifest"),
        ("Privacy issues", _num(readiness.get("privacy_issue_count")), "release audit"),
        (
            "Telemetry events",
            _num(readiness.get("event_count_total")),
            "audited corpus",
        ),
        (
            "Incident violation rate",
            _pct(summary.get("incident_violation_rate")),
            "incidents with any violating cell",
        ),
        (
            "Actions inserted",
            _num(summary.get("enforcement_actions_inserted_count_total")),
            "missing mandatory",
        ),
        (
            "Actions removed",
            _num(summary.get("enforcement_actions_removed_count_total")),
            "approval gates",
        ),
        (
            "Cost",
            _money(summary.get("llm_cost_estimated_usd_total")),
            "reported execution",
        ),
        (
            "Public checks",
            f"{consistency.get('failed_check_count', 0)}/{consistency.get('check_count', 0)}",
            "failed/total",
        ),
    ]


def _render_quickstart_cards(command_prefix: str) -> str:
    cards = [
        (
            "1. Inspect",
            "Open this file first. It summarizes the corpus, evaluation runs, policy coverage, paired contrasts, and replay checks from the public evidence bundle.",
            "REVIEWER_DASHBOARD.html",
        ),
        (
            "2. Reproduce",
            "Run the zero-cost path to validate manifests, regenerate evidence "
            "reports, rebuild this dashboard, and write the reproduction report.",
            f"bash {command_prefix}run.sh reproduce-results",
        ),
        (
            "3. Audit",
            "Follow exact source files for run accounting, pairwise tests, rule coverage, and release hygiene.",
            "artifact_outputs/analysis/",
        ),
    ]
    rendered = []
    for title, body, command in cards:
        rendered.append(
            '<div class="quick-card">'
            f"<h3>{_escape(title)}</h3>"
            f"<p>{_escape(body)}</p>"
            f"<code>{_escape(command)}</code>"
            "</div>"
        )
    return "\n".join(rendered)


def _render_evidence_links(data: dict[str, Any], artifact_prefix: str) -> str:
    links = [
        ("Reproduction report", "artifact_outputs/reproduction_report.md"),
        ("Protocol manifest", "artifact_outputs/analysis/protocol_freeze.json"),
        (
            "Public evidence provenance",
            "artifact_outputs/analysis/public_evidence_provenance.json",
        ),
        (
            "Evaluation run manifest",
            "artifact_outputs/analysis/official_runs_manifest.json",
        ),
        (
            "Evaluation pairwise tests",
            "artifact_outputs/analysis/official_pairwise_tests.json",
        ),
        (
            "Evaluation consistency checks",
            "artifact_outputs/analysis/official_public_consistency.json",
        ),
        ("Corpus readiness", "artifact_outputs/analysis/corpus_readiness.json"),
        (
            "Global artifact assessment",
            "artifact_outputs/analysis/global_artifact_assessment.json",
        ),
        ("Policy surface audit", "artifact_outputs/analysis/policy_surface_audit.json"),
    ]
    optional_links = (
        ("separation", "Separation report", "separation_report.json"),
        ("adversarial", "Adversarial report", "adversarial_report.json"),
        (
            "fidelity",
            "Projection fidelity report",
            "projection_fidelity_report.json",
        ),
        (
            "projection",
            "Projection diagnostics",
            "projection_diagnostics.json",
        ),
    )
    for data_key, label, filename in optional_links:
        if data.get(data_key):
            links.append((label, f"artifact_outputs/analysis/{filename}"))
    if data.get("partition_audit"):
        preflight_dir = data.get("policy_preflight_dir")
        if not isinstance(preflight_dir, Path):
            raise ValueError("Policy preflight evidence is missing its directory.")
        artifact_root = preflight_dir.parents[3]
        preflight_rel = preflight_dir.relative_to(artifact_root).as_posix()
        links.append(
            ("Corpus partition audit", f"{preflight_rel}/corpus_partition_audit.json")
        )
    if data.get("heldout_activation"):
        preflight_dir = data.get("policy_preflight_dir")
        if not isinstance(preflight_dir, Path):
            raise ValueError("Policy preflight evidence is missing its directory.")
        artifact_root = preflight_dir.parents[3]
        preflight_rel = preflight_dir.relative_to(artifact_root).as_posix()
        links.append(
            (
                "Cross-batch activation report",
                f"{preflight_rel}/heldout_activation.json",
            )
        )
    if data.get("normalization_ablation"):
        preflight_dir = data.get("policy_preflight_dir")
        if not isinstance(preflight_dir, Path):
            raise ValueError("Policy preflight evidence is missing its directory.")
        artifact_root = preflight_dir.parents[3]
        preflight_rel = preflight_dir.relative_to(artifact_root).as_posix()
        links.append(
            (
                "Normalization ablation",
                f"{preflight_rel}/normalization_ablation.json",
            )
        )
    return "\n".join(
        f'<li><a href="{_escape(artifact_prefix + href)}">{_escape(label)}</a></li>'
        for label, href in links
    )


def _render_metric_strip(metrics: list[tuple[str, str, str]]) -> str:
    rows = []
    for label, value, detail in metrics:
        rows.append(
            '<div class="metric">'
            f'<div class="metric-label">{_escape(label)}</div>'
            f'<div class="metric-value">{_escape(value)}</div>'
            f'<div class="metric-detail">{_escape(detail)}</div>'
            "</div>"
        )
    return "\n".join(rows)


def _render_stability(stability: dict[str, Any] | None) -> str:
    if stability is None:
        return (
            '<p class="note">A separate repeat-stability report is not '
            "part of this execution state.</p>"
        )
    return (
        "<table><tbody>"
        f"<tr><th>Repeats</th><td>{_num(stability.get('repeat_count'))}</td></tr>"
        # Incident-level, not the per-cell rate in the primary report. One label for
        # two denominators is how a reviewer concludes the artifact contradicts
        # the published result.
        "<tr><th>Incident violation range</th><td>"
        f"{_pct(stability.get('incident_violation_rate', {}).get('min'))} to "
        f"{_pct(stability.get('incident_violation_rate', {}).get('max'))}</td></tr>"
        "<tr><th>Modification range</th><td>"
        f"{_pct(stability.get('enforcement_modification_rate', {}).get('min'))} to "
        f"{_pct(stability.get('enforcement_modification_rate', {}).get('max'))}</td></tr>"
        "<tr><th>Task coverage drop</th><td>"
        f"{_pct(stability.get('task_coverage_drop_rate', {}).get('max'))}</td></tr>"
        "</tbody></table>"
    )


def _arm_label(row: dict[str, Any]) -> str:
    arm = row.get("arm") or row.get("arm_internal") or ""
    return {"llm_zero": "zero", "llm_policy_prompt": "policy"}.get(arm, arm)


def _rate_from_counts(row: dict[str, Any], numerator_key: str) -> float | None:
    """Derive a rate from the integer counts, never from a stored rate.

    The bundle rounds its rates to four decimals. Formatting one of those to a
    percentage rounds a second time, and 383/597 prints as 64.1% here while the
    report prints 0.642 from the same counts. Reviewers read that as a defect,
    and they are right to.
    """
    numerator = row.get(numerator_key)
    denominator = row.get("completed_runs")
    if type(numerator) is not int or type(denominator) is not int:
        raise ValueError(
            f"Model row is missing integer counts for {numerator_key}; the "
            "dashboard will not render a rate it cannot derive."
        )
    if denominator < 0 or numerator < 0 or numerator > denominator:
        raise ValueError(f"Invalid counts for {numerator_key}: {numerator}/{denominator}")
    if denominator == 0:
        return None
    return numerator / denominator


def _render_model_rows(rows: list[dict[str, Any]]) -> str:
    out = []
    for row in sorted(rows, key=lambda item: (item.get("model", ""), _arm_label(item))):
        arm = _arm_label(row)
        rate = _rate_from_counts(row, "runs_with_violations")
        mod = _rate_from_counts(row, "enforcement_modified_run_count")
        width = rate * 100 if rate is not None else 0.0
        out.append(
            '<tr data-model="{model}" data-arm="{arm}">'
            "<td>{model}</td><td>{arm}</td><td>{runs}</td>"
            '<td><span class="bar" aria-hidden="true">'
            '<span style="width:{width:.1f}%">&nbsp;</span></span>{rate}</td>'
            "<td>{mod}</td><td>{coverage}</td><td>{jaccard}</td><td>{cost}</td>"
            "</tr>".format(
                model=_escape(row.get("model", "")),
                arm=_escape(arm),
                runs=_num(row.get("completed_runs")),
                width=width,
                rate=_pct(rate),
                mod=_pct(mod),
                coverage=_pct(row.get("task_coverage_enforced_avg")),
                jaccard=("N/A" if row.get("delta_jaccard_avg") is None
                         else f"{_safe_float(row['delta_jaccard_avg']):.4f}"),
                cost=_money(row.get("llm_cost_estimated_usd_total")),
            )
        )
    return "\n".join(out)


def _pairwise_label(row: dict[str, Any]) -> str:
    label = row.get("comparison_label")
    if label:
        return str(label)
    model_a = str(row.get("model_a", "")).replace("|", " ")
    model_b = str(row.get("model_b", "")).replace("|", " ")
    return f"{model_a} vs {model_b}" if model_a and model_b else ""


def _render_pairwise_rows(rows: list[dict[str, Any]]) -> str:
    out = []
    for row in rows:
        out.append(
            "<tr>"
            f"<td>{_escape(_pairwise_label(row))}</td>"
            f"<td>{_num(row.get('paired_incident_count'))}</td>"
            f"<td>{_safe_float(row.get('rate_diff_a_minus_b')):+.3f}</td>"
            f"<td>{_safe_float(row.get('cohens_h')):+.3f}</td>"
            f"<td>{_safe_float(row.get('mcnemar_p_value_holm')):.6f}</td>"
            "</tr>"
        )
    return "\n".join(out)


def _render_diagnostic_studies(data: dict[str, Any]) -> str:
    """H2 separation, H3 adversarial, and projection-fidelity/diagnostic panels.

    Each panel renders only when its report is present, so the zero-cost public
    path stays valid while a full run exposes every study a reviewer audits.
    """
    blocks: list[str] = []

    separation = data.get("separation")
    if separation and isinstance(separation.get("report"), dict):
        rows = separation["report"].get("baselines", [])
        label = {
            "prompt_guardrail": "Accept-all guardrail",
            "llm_judge": "LLM judge (independent)",
        }
        body = "\n".join(
            "<tr><td>{name}</td><td>{fa}</td><td>{fr}</td><td>{na} / {nr}</td></tr>".format(
                name=_escape(label.get(r["baseline_id"], r["baseline_id"])),
                fa=_pct(r["false_accept_rate"]),
                fr=_pct(r["false_reject_rate"]),
                na=_num(r["false_accept_denominator"]),
                nr=_num(r["false_reject_denominator"]),
            )
            for r in sorted(rows, key=lambda x: x["baseline_id"])
        )
        blocks.append(
            "<div><h2>Separation study (H2)</h2>"
            '<p class="note">False accept and false reject of each text-level baseline '
            "against the typed verifier's verdict over the evaluated cells.</p>"
            "<table><thead><tr><th>Baseline</th><th>False accept</th>"
            "<th>False reject</th><th>n (accept / reject)</th></tr></thead>"
            f"<tbody>{body}</tbody></table></div>"
        )

    adversarial = data.get("adversarial")
    if adversarial:
        rate = _safe_float(adversarial.get("induced_violation_rate"))
        blocks.append(
            "<div><h2>Adversarial study (H3)</h2>"
            '<p class="note">Injection-induced violations over clean/injected pairs whose '
            "clean plan is compliant; the typed decision surfaces each one.</p>"
            "<table><tbody>"
            f"<tr><th>Induced violations</th><td>{_num(adversarial.get('induced_violation_count'))} "
            f"/ {_num(adversarial.get('induced_violation_denominator'))} ({_pct(rate)})</td></tr>"
            f"<tr><th>Complete pairs</th><td>{_num(adversarial.get('pair_count'))}</td></tr>"
            f"<tr><th>Refusal pairs (excluded)</th><td>{_num(adversarial.get('incomplete_pair_count'))}</td></tr>"
            "</tbody></table></div>"
        )

    fidelity = data.get("fidelity")
    if fidelity:
        blocks.append(
            "<div><h2>Projection fidelity</h2>"
            '<p class="note">Extraction over authored free-form plans with known targets and '
            "distractors (paraphrase, indirect reference, negation).</p>"
            "<table><tbody>"
            f"<tr><th>Precision</th><td>{_pct(fidelity.get('overall_precision'))}</td></tr>"
            f"<tr><th>Recall</th><td>{_pct(fidelity.get('overall_recall'))}</td></tr>"
            f"<tr><th>Distractors rejected</th><td>{_num(fidelity.get('distractor_rejected'))} "
            f"/ {_num(fidelity.get('distractor_total'))}</td></tr>"
            "</tbody></table></div>"
        )

    projection = data.get("projection")
    if projection:
        blocks.append(
            "<div><h2>Extraction diagnostics</h2>"
            '<p class="note">Signals that bound projection under structured output over '
            f"{_num(projection.get('cell_count'))} cells.</p>"
            "<table><tbody>"
            f"<tr><th>Parse errors</th><td>{_num(projection.get('parse_error_count'))}</td></tr>"
            f"<tr><th>Out-of-catalog actions</th><td>{_num(projection.get('hallucinated_action_count'))}</td></tr>"
            f"<tr><th>Empty plans</th><td>{_num(projection.get('empty_plan_count'))}</td></tr>"
            f"<tr><th>Max completion tokens / cap</th>"
            f"<td>{_num(projection.get('max_completion_tokens'))} / {_num(projection.get('completion_token_cap'))}</td></tr>"
            "</tbody></table></div>"
        )

    if not blocks:
        return ""
    return (
        "<section><h2>Preserved studies</h2>"
        '<p class="note">Separation, adversarial, and projection studies behind the '
        "H2, H3, and fidelity claims. Each panel is loaded from its preserved report.</p>"
        f'<div class="grid2">{"".join(blocks)}</div></section>'
    )


def _load_rule_surface(constraints_path: Path) -> list[dict[str, str]]:
    """Read the declared rule set from the shipped policy bundle."""
    if not constraints_path.is_file():
        raise FileNotFoundError(
            f"Policy constraints file not found: {constraints_path}"
        )
    parsed = yaml.safe_load(constraints_path.read_text(encoding="utf-8"))
    if isinstance(parsed, list):
        rules = parsed
    elif isinstance(parsed, dict) and isinstance(parsed.get("rules"), list):
        rules = parsed["rules"]
    else:
        raise ValueError(
            f"Policy constraints must contain a YAML rule list: {constraints_path}"
        )
    surface: list[dict[str, str]] = []
    for rule in rules:
        if not isinstance(rule, dict):
            raise ValueError(f"Policy rule must be a mapping: {constraints_path}")
        rule_id = str(rule.get("rule_id", "")).strip()
        rule_type = str(rule.get("type", "")).strip()
        action = str(rule.get("action", "")).strip()
        if not rule_id or not rule_type or not action:
            raise ValueError(
                f"Policy rule must define rule_id, type, and action: {constraints_path}"
            )
        surface.append(
            {
                "rule_id": rule_id,
                "family": _FAMILY_LABEL.get(rule_type, rule_type),
                "action": action,
                "repair": str(rule.get("repair_operator", "")),
            }
        )
    if not surface:
        raise ValueError(f"Policy constraints contain no rules: {constraints_path}")
    return surface


def _render_rule_rows(summary: dict[str, Any], constraints_path: Path) -> str:
    rule_counts = summary.get("violations_by_rule", {})
    if not isinstance(rule_counts, dict):
        rule_counts = {}
    surface = _load_rule_surface(constraints_path)
    # Most-fired first, then by rule id, so the reader sees where the
    # violations concentrate at a glance.
    surface.sort(
        key=lambda rule: (
            -_safe_int(rule_counts.get(rule["rule_id"], 0)),
            rule["rule_id"],
        )
    )
    out = []
    for rule in surface:
        count = _safe_int(rule_counts.get(rule["rule_id"], 0))
        status = "active" if count > 0 else "inactive"
        out.append(
            f'<tr data-rule-status="{status}"><td>{_escape(rule["rule_id"])}</td>'
            f"<td>{_escape(rule['family'])}</td><td>{_escape(rule['action'])}</td>"
            f"<td>{_escape(rule['repair'])}</td>"
            f"<td>{count}</td><td>{status}</td></tr>"
        )
    return "\n".join(out)


def _render_identifier_detail(value: Any) -> str:
    identifiers = value if isinstance(value, list) else []
    if not identifiers:
        return "&mdash;"
    rendered = "<br>".join(
        f"<code>{_escape(identifier)}</code>" for identifier in identifiers
    )
    return (
        f"<details><summary>{len(identifiers)} identifiers</summary>"
        f'<div class="identifier-list">{rendered}</div></details>'
    )


def _render_surface_rows(assessment: dict[str, Any]) -> str:
    surface = assessment.get("policy_surface", {})
    dataset = assessment.get("dataset", {})
    rows = [
        (
            "Action catalog",
            surface.get("action_catalog_count"),
            "research action abstraction",
        ),
        (
            "Operational rules",
            len(surface.get("operational_rule_ids", [])),
            surface.get("operational_rule_ids", []),
        ),
        (
            "Specification-only rules",
            len(surface.get("specification_rule_ids", [])),
            surface.get("specification_rule_ids", []),
        ),
        (
            "Observed response actions",
            dataset.get("observed_action_count"),
            "none recorded in source exports",
        ),
        (
            "Workflow tasks",
            dataset.get("workflow_task_count"),
            "not treated as response actions",
        ),
    ]
    rendered = []
    for label, value, detail in rows:
        evidence = (
            _render_identifier_detail(detail)
            if isinstance(detail, list)
            else _escape(detail)
        )
        rendered.append(
            f"<tr><td>{_escape(label)}</td><td>{_escape(value)}</td><td>{evidence}</td></tr>"
        )
    return "\n".join(rendered)


def _render_policy_derivation(data: dict[str, Any]) -> str:
    partition = data.get("partition_audit")
    activation = data.get("heldout_activation")
    if not isinstance(partition, dict) or not isinstance(activation, dict):
        return ""
    observed = partition.get("observed")
    if not isinstance(observed, dict):
        raise ValueError("Corpus partition audit is missing observed partitions.")
    derivation = observed.get("derivation")
    heldout = observed.get("heldout")
    if not isinstance(derivation, dict) or not isinstance(heldout, dict):
        raise ValueError("Corpus partition audit is missing a normalized partition.")
    rules = activation.get("rules")
    if not isinstance(rules, list):
        raise ValueError("Cross-batch activation report is missing rule rows.")
    content_rules = [
        rule
        for rule in rules
        if isinstance(rule, dict) and rule.get("scope_kind") == "content_triggered"
    ]
    transfer_rules = [
        rule for rule in content_rules if rule.get("meets_transfer_floor") is True
    ]
    source_lineage = partition["source_lineage"]
    return (
        "<section><h2>Corpus Partition and Policy Derivation</h2>"
        '<p class="note">The model-evaluation corpus and the policy-development '
        "audit are distinct objects. The partition is deterministic and disjoint; "
        "content-triggered scope support is measured before any model response exists.</p>"
        "<table><tbody>"
        f"<tr><th>Derivation partition</th><td>{_num(derivation.get('incident_count'))} incidents</td></tr>"
        f"<tr><th>Batch 2 transfer check</th><td>{_num(heldout.get('incident_count'))} incidents</td></tr>"
        f"<tr><th>Normalized total</th><td>{_num(observed.get('total_incident_count'))} incidents</td></tr>"
        f"<tr><th>Source-to-normalized lineage</th><td>{_num(source_lineage.get('raw_export_count'))} / {_num(source_lineage.get('canonical_source_manifest_count'))} one-to-one</td></tr>"
        f"<tr><th>Shared primary categories</th><td>{_num(partition.get('shared_primary_detection_category_count'))}</td></tr>"
        f"<tr><th>Content rules meeting the transfer floor</th><td>{_num(len(transfer_rules))} / {_num(len(content_rules))}</td></tr>"
        "</tbody></table></section>"
    )


def _render_html(
    data: dict[str, Any],
    constraints_path: Path,
    *,
    artifact_prefix: str = "",
    command_prefix: str = "",
) -> str:
    _validate_policy_alignment(data, constraints_path)
    _validate_preflight_evidence(data, predicate_digest(constraints_path))
    model_rows = data["runs"].get("by_model_and_arm", [])
    pairwise_rows = data["pairwise"].get("rows", [])
    stability = data.get("stability")
    summary = data["summary"]
    assessment = data["assessment"]
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SOCpilot Audit Dashboard</title>
<style>
:root {{
  color-scheme: light;
  --bg: #f7f8fa;
  --ink: #17202a;
  --muted: #5e6a78;
  --line: #d8dee8;
  --blue: #1c5a86;
  --green: #176d55;
  --red: #a13f35;
  --surface: #ffffff;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: var(--ink);
  background: var(--bg);
}}
header {{
  padding: 28px 32px 20px;
  border-bottom: 1px solid var(--line);
  background: var(--surface);
}}
h1 {{ margin: 0 0 6px; font-size: 28px; letter-spacing: 0; }}
.subtitle {{ color: var(--muted); max-width: 980px; }}
.toolbar {{
  position: sticky;
  top: 0;
  z-index: 2;
  display: flex;
  gap: 12px;
  align-items: center;
  padding: 12px 32px;
  border-bottom: 1px solid var(--line);
  background: rgba(255,255,255,.96);
}}
select, input, button {{
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 8px 10px;
  background: white;
  min-width: 160px;
}}
button {{ color: var(--ink); cursor: pointer; min-width: auto; }}
button:hover {{ border-color: var(--blue); color: var(--blue); }}
main {{ padding: 24px 32px 40px; }}
.quickstart {{
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 14px;
  margin-bottom: 20px;
}}
.quick-card {{
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 7px;
  padding: 14px;
}}
.quick-card h3 {{ margin: 0 0 8px; font-size: 15px; }}
.quick-card p {{ margin: 0 0 10px; color: var(--muted); }}
.quick-card code {{
  display: block;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  background: #f1f5f8;
  border: 1px solid var(--line);
  border-radius: 5px;
  padding: 8px;
}}
.metrics {{
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 1px;
  border: 1px solid var(--line);
  background: var(--line);
}}
.metric {{ padding: 14px; background: var(--surface); min-height: 96px; }}
.metric-label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; }}
.metric-value {{ font-size: 25px; font-weight: 720; margin: 4px 0; }}
.metric-detail {{ color: var(--muted); font-size: 12px; }}
section {{ margin-top: 28px; }}
h2 {{ margin: 0 0 10px; font-size: 18px; }}
.note {{ color: var(--muted); margin: -4px 0 12px; }}
table {{
  width: 100%;
  border-collapse: collapse;
  background: var(--surface);
  border: 1px solid var(--line);
}}
th, td {{
  padding: 9px 10px;
  text-align: left;
  border-bottom: 1px solid var(--line);
  vertical-align: middle;
}}
th {{ font-size: 12px; color: var(--muted); text-transform: uppercase; }}
tbody tr:hover {{ background: #f1f5f8; }}
.bar {{
  display: inline-block;
  width: 96px;
  height: 8px;
  margin-right: 8px;
  background: #e8edf3;
  vertical-align: middle;
}}
.bar span {{ display: block; height: 100%; background: var(--blue); }}
.grid2 {{ display: grid; grid-template-columns: minmax(0,1fr) minmax(0,1fr); gap: 18px; }}
.evidence-list {{
  columns: 2;
  background: var(--surface);
  border: 1px solid var(--line);
  padding: 14px 18px;
}}
.evidence-list li {{ break-inside: avoid; margin: 0 0 7px; }}
.evidence-list a {{ color: var(--blue); text-decoration: none; }}
.evidence-list a:hover {{ text-decoration: underline; }}
.pill {{
  display: inline-block;
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 2px 8px;
  color: var(--muted);
  font-size: 12px;
}}
details {{ max-width: 100%; }}
summary {{ color: var(--blue); cursor: pointer; }}
.identifier-list {{ margin-top: 7px; color: var(--muted); overflow-wrap: anywhere; }}
@media (max-width: 1100px) {{ .metrics {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }} }}
@media (max-width: 680px) {{ .metrics {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
@media (max-width: 860px) {{
  header, main, .toolbar {{ padding-left: 16px; padding-right: 16px; }}
  .toolbar {{ flex-wrap: wrap; }}
  .grid2, .quickstart {{ grid-template-columns: 1fr; }}
  .evidence-list {{ columns: 1; }}
  table {{ font-size: 12px; }}
}}
@media (max-width: 480px) {{ .metrics {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<header>
  <h1>SOCpilot Audit Dashboard</h1>
  <div class="subtitle">Reviewer-facing view of the shipped public evidence: corpus readiness, run accounting, rule activation, paired contrasts, and replay consistency. All values are loaded from files in <code>artifact_outputs/analysis</code>.</div>
</header>
<div class="toolbar">
  <label>Model <select id="modelFilter"><option value="">All models</option></select></label>
  <label>Arm <select id="armFilter"><option value="">All arms</option></select></label>
  <label>Search outcomes <input id="searchBox" placeholder="model, arm, or value"></label>
  <button id="clearFilters" type="button">Clear filters</button>
  <span class="pill">zero-cost public audit</span>
</div>
<main>
  <section class="quickstart" aria-label="Reviewer quickstart">
    {_render_quickstart_cards(command_prefix)}
  </section>

  <div class="metrics">
    {_render_metric_strip(_summary_metrics(data))}
  </div>

  <section>
    <h2>Model and Arm Outcomes</h2>
    <p class="note">Violation and enforcement rates expose the provider split; task coverage shows whether deterministic enforcement removed baseline-covered work.</p>
    <table id="outcomesTable">
      <thead><tr><th>Model</th><th>Arm</th><th>Runs</th><th>Violation</th><th>Modified</th><th>Task coverage</th><th>Delta Jaccard</th><th>Cost</th></tr></thead>
      <tbody>{_render_model_rows(model_rows)}</tbody>
    </table>
  </section>

  <section class="grid2">
    <div>
      <h2>Rule Activation</h2>
      <p class="note">All four families are declared and exercised; missing mandatory actions dominate the violations, and ordering rules did not fire in this run.</p>
      <table><thead><tr><th>Rule</th><th>Family</th><th>Action</th><th>Repair</th><th>Violations</th><th>Status</th></tr></thead><tbody>{_render_rule_rows(summary, constraints_path)}</tbody></table>
    </div>
    <div>
      <h2>Protocol Surface</h2>
      <p class="note">These rows show why the claim is a declared protocol state rather than a generic SOC benchmark.</p>
      <table><thead><tr><th>Surface</th><th>Count</th><th>Evidence</th></tr></thead><tbody>{_render_surface_rows(assessment)}</tbody></table>
    </div>
  </section>

  <section>
    <h2>Paired Contrasts</h2>
    <p class="note">Each row compares the same incident set across two model/arm cells.</p>
    <table><thead><tr><th>Contrast</th><th>n</th><th>Rate diff</th><th>Cohen h</th><th>Holm p</th></tr></thead><tbody>{_render_pairwise_rows(pairwise_rows)}</tbody></table>
  </section>

  {_render_diagnostic_studies(data)}

  {_render_policy_derivation(data)}

  <section>
    <h2>Evidence Map</h2>
    <p class="note">Direct links to the files supporting the reported claims.</p>
    <ul class="evidence-list">{_render_evidence_links(data, artifact_prefix)}</ul>
  </section>

  <section class="grid2">
    <div>
      <h2>Replay Stability</h2>
      {_render_stability(stability)}
    </div>
    <div>
      <h2>Repair Accounting</h2>
      <table><tbody>
        <tr><th>Removed</th><td>{_num(summary.get("enforcement_actions_removed_count_total"))}</td></tr>
        <tr><th>Inserted</th><td>{_num(summary.get("enforcement_actions_inserted_count_total"))}</td></tr>
        <tr><th>Reordered</th><td>{_num(summary.get("enforcement_actions_reordered_count_total"))}</td></tr>
        <tr><th>Deferred</th><td>{_num(summary.get("enforcement_actions_deferred_count_total"))}</td></tr>
      </tbody></table>
    </div>
  </section>
</main>
<script>
const rows = [...document.querySelectorAll("#outcomesTable tbody tr")];
const modelFilter = document.querySelector("#modelFilter");
const armFilter = document.querySelector("#armFilter");
const searchBox = document.querySelector("#searchBox");
const clearFilters = document.querySelector("#clearFilters");
for (const value of [...new Set(rows.map(row => row.dataset.model))].sort()) {{
  modelFilter.add(new Option(value, value));
}}
for (const value of [...new Set(rows.map(row => row.dataset.arm))].sort()) {{
  armFilter.add(new Option(value, value));
}}
function applyFilters() {{
  const model = modelFilter.value;
  const arm = armFilter.value;
  const query = searchBox.value.trim().toLowerCase();
  for (const row of rows) {{
    const visible = (!model || row.dataset.model === model)
      && (!arm || row.dataset.arm === arm)
      && (!query || row.innerText.toLowerCase().includes(query));
    row.style.display = visible ? "" : "none";
  }}
}}
modelFilter.addEventListener("change", applyFilters);
armFilter.addEventListener("change", applyFilters);
searchBox.addEventListener("input", applyFilters);
clearFilters.addEventListener("click", () => {{
  modelFilter.value = "";
  armFilter.value = "";
  searchBox.value = "";
  applyFilters();
}});
</script>
</body>
</html>
"""


def _artifact_prefix(output_path: Path, artifact_root: Path) -> str:
    """Return a relative package-root prefix without leaking local paths."""
    if output_path != artifact_root and artifact_root not in output_path.parents:
        return ""
    relative = Path(os.path.relpath(artifact_root, output_path.parent)).as_posix()
    return "" if relative == "." else f"{relative}/"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a static reviewer dashboard from public analysis outputs."
    )
    parser.add_argument(
        "--analysis-dir",
        default="artifact_outputs/analysis",
    )
    parser.add_argument(
        "--output-html",
        default="artifact_outputs/dashboard/index.html",
    )
    parser.add_argument(
        "--constraints",
        default="artifact_data/global/constraints.yaml",
        help="Policy constraints file that supplies the rule set for the table.",
    )
    parser.add_argument(
        "--landing-html",
        default="REVIEWER_DASHBOARD.html",
        help="Optional reviewer-facing dashboard copy at the package root.",
    )
    parser.add_argument(
        "--policy-preflight-dir",
        default=None,
        help="Optional deterministic policy-preflight directory for this freeze.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    analysis_dir = Path(args.analysis_dir).expanduser().resolve()
    output_html = Path(args.output_html).expanduser().resolve()
    constraints_path = Path(args.constraints).expanduser()
    policy_preflight_dir = (
        Path(args.policy_preflight_dir).expanduser().resolve()
        if args.policy_preflight_dir
        else None
    )
    data = _load_dashboard_data(analysis_dir, policy_preflight_dir)
    artifact_root = analysis_dir.parents[1]
    dashboard_prefix = _artifact_prefix(output_html, artifact_root)
    html_text = _render_html(
        data,
        constraints_path,
        artifact_prefix=dashboard_prefix,
        command_prefix=dashboard_prefix,
    )
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html_text, encoding="utf-8")
    print(f"Audit dashboard written to: {output_html}")
    if args.landing_html:
        landing_html = Path(args.landing_html).expanduser()
        if not landing_html.is_absolute():
            landing_html = analysis_dir.parents[1] / landing_html
        landing_html.parent.mkdir(parents=True, exist_ok=True)
        landing_prefix = _artifact_prefix(landing_html, artifact_root)
        landing_html.write_text(
            _render_html(
                data,
                constraints_path,
                artifact_prefix=landing_prefix,
                command_prefix=landing_prefix,
            ),
            encoding="utf-8",
        )
        print(f"Reviewer dashboard entry point written to: {landing_html}")


if __name__ == "__main__":
    main()
