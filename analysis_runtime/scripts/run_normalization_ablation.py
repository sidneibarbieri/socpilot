"""Measure how sensitive rule activation is to the normalization instrument.

The policy triggers on behavior classes, not on detection strings, so a fair
reader asks how much of the result rests on the particular class boundaries and
the particular pattern subset the authors wrote down. This tool perturbs the
instrument and re-measures activation through the same audited scope matcher the
frozen report uses, so every variant is comparable to the baseline by
construction. It makes no provider calls and is deterministic.

Variants
  curated         the shipped scopes (baseline)
  class_complete  every content rule takes its class's full pattern set, which
                  removes any within-class pattern selection by the authors
  noise_folded    every content rule additionally takes the unclassified-noise
                  patterns, which tests whether excluding that bucket is
                  load-bearing rather than cosmetic
  string_anchored every content rule is rewritten against the detection-category
                  strings observed in Batch 1, which is the rule a
                  team would have written from the vocabulary it saw
  leave_one_out   each transfer-supporting rule drops one scope pattern at a time;
                  the worst case bounds how much any single detection string
                  carries

The transfer-check floor (three or more incidents) is the survival test: a
rule that still clears it under a variant does not owe its transfer status to
that modeling choice.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml

from soc_llm_policy.paths import resolve_repo_root
from soc_llm_policy.policy_predicates import predicate_digest

_NOISE_CLASS = "unclassified_noise"
_HELDOUT_FLOOR = 3
_MIN_DROPPABLE_PATTERNS = 2

Rules = list[dict[str, Any]]
Variant = dict[str, dict[str, Any]]


def _load_script(repo_root: Path, name: str) -> ModuleType:
    """Reuse a shipped instrument so every variant comes from the audited path."""
    spec = importlib.util.spec_from_file_location(
        name, repo_root / "scripts" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _class_of(patterns: list[str], classes: dict[str, list[str]]) -> str:
    """The unique behavior class whose pattern set contains this rule's scope."""
    owners = [name for name, pats in classes.items() if set(patterns) <= set(pats)]
    if len(owners) != 1:
        raise ValueError(
            f"scope {patterns} is not contained in exactly one class: {owners}"
        )
    return owners[0]


def _content_rules(rules: Rules) -> Rules:
    return [r for r in rules if (r.get("scope") or {}).get("category_contains")]


def _string_anchored_scopes(
    content: Rules, derivation_categories: list[str]
) -> dict[str, list[str]]:
    """Rewrite each content rule as the rule a team would have written against
    the detection strings it actually saw: the derivation-window category strings
    the rule's behavior class already covers, with no class union behind them.
    """
    scopes: dict[str, list[str]] = {}
    for rule in content:
        patterns = [p.lower() for p in rule["scope"]["category_contains"]]
        matched = sorted(
            category
            for category in derivation_categories
            if any(pattern in category.lower() for pattern in patterns)
        )
        if not matched:
            raise ValueError(
                f"{rule['rule_id']} matches no derivation category string; the "
                "rule could not have been derived from this window."
            )
        scopes[rule["rule_id"]] = matched
    return scopes


def _rescope(rules: Rules, new_scopes: dict[str, list[str]]) -> Rules:
    out = deepcopy(rules)
    for rule in out:
        if rule["rule_id"] in new_scopes:
            rule["scope"]["category_contains"] = sorted(new_scopes[rule["rule_id"]])
    return out


def _measure(
    vha: ModuleType,
    repo_root: Path,
    derivation_dir: Path,
    heldout_dir: Path,
    rules: Rules,
    work: Path,
    tag: str,
) -> Variant:
    """Write a perturbed constraints file and run the audited activation report."""
    path = work / f"constraints_{tag}.yaml"
    path.write_text(yaml.safe_dump(rules, sort_keys=False), encoding="utf-8")
    report = vha.build_report(
        repo_root=repo_root,
        derivation_dir=derivation_dir,
        heldout_dir=heldout_dir,
        constraints_path=path,
    )
    return {
        row.rule_id: {
            "scope_kind": row.scope_kind,
            "derivation": row.derivation_incidents,
            "heldout": row.heldout_incidents,
            "meets_transfer_floor": row.meets_transfer_floor,
        }
        for row in report.rules
    }


def _leave_one_out(
    ctx: tuple[ModuleType, Path, Path, Path, Path],
    rules: Rules,
    content: Rules,
    transfer_supporting: list[str],
) -> dict[str, Any]:
    """Drop one scope pattern at a time and keep the worst Batch 2 activation."""
    vha, repo_root, derivation_dir, heldout_dir, work = ctx
    loo: dict[str, Any] = {}
    for rid in transfer_supporting:
        rule = next(r for r in content if r["rule_id"] == rid)
        patterns = rule["scope"]["category_contains"]
        if len(patterns) < _MIN_DROPPABLE_PATTERNS:
            loo[rid] = {"droppable": False, "patterns": len(patterns)}
            continue
        worst_heldout: int | None = None
        worst_pattern: str | None = None
        for dropped in patterns:
            kept = [p for p in patterns if p != dropped]
            tag = f"loo_{rid}_{dropped}"
            row = _measure(
                vha,
                repo_root,
                derivation_dir,
                heldout_dir,
                _rescope(rules, {rid: kept}),
                work,
                tag,
            )
            heldout = row[rid]["heldout"]
            if worst_heldout is None or heldout < worst_heldout:
                worst_heldout, worst_pattern = heldout, dropped
        loo[rid] = {
            "droppable": True,
            "patterns": len(patterns),
            "worst_heldout": worst_heldout,
            "worst_dropped_pattern": worst_pattern,
            "survives_floor": worst_heldout >= _HELDOUT_FLOOR,
        }
    return loo


def _macro_values(report: dict[str, Any]) -> dict[str, int]:
    loo = report["leave_one_out"]
    droppable = [r for r, v in loo.items() if v["droppable"]]
    below = [r for r in droppable if not loo[r]["survives_floor"]]
    collapsed = [r for r in droppable if loo[r]["worst_heldout"] == 0]
    transfer_supporting = report["transfer_supporting_content_rules"]
    variants = report["variants"]
    class_heldout = [
        variants["curated"][rid]["heldout"] for rid in transfer_supporting
    ]
    anchored_heldout = [
        variants["string_anchored"][rid]["heldout"] for rid in transfer_supporting
    ]
    return {
        "NormAblationTransferRules": len(transfer_supporting),
        "NormAblationContentRules": sum(
            1
            for row in variants["curated"].values()
            if row["scope_kind"] == "content_triggered"
        ),
        "NormAblationClassCompleteSurvivors": len(
            report["survivors"]["class_complete"]
        ),
        "NormAblationNoiseFoldedSurvivors": len(report["survivors"]["noise_folded"]),
        "NormAblationClassMinHeldout": min(class_heldout),
        "NormAblationClassMaxHeldout": max(class_heldout),
        "NormAblationStringAnchoredSurvivors": len(
            report["survivors"]["string_anchored"]
        ),
        "NormAblationStringAnchoredCollapsed": sum(
            1 for heldout in anchored_heldout if heldout == 0
        ),
        "NormAblationLOODroppable": len(droppable),
        "NormAblationLOODrops": report["leave_one_out_drops"],
        "NormAblationLOOBelowFloor": len(below),
        "NormAblationLOOCollapsed": len(collapsed),
    }


def render_macros(report: dict[str, Any]) -> str:
    """Render ablation macros derived only from the report."""
    values = _macro_values(report)
    body = "\n".join(
        f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in values.items()
    )
    return "% Generated by run_normalization_ablation.py\n" + body + "\n"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_report(
    repo_root: Path,
    constraints: str,
    normalization: str,
    derivation_dir: Path,
    heldout_dir: Path,
) -> dict[str, Any]:
    """Run every variant and return the deterministic ablation record."""
    vha = _load_script(repo_root, "validate_heldout_activation")
    partitions = _load_script(repo_root, "audit_corpus_partitions")
    derivation_categories = partitions.primary_detection_categories(derivation_dir)
    rules = yaml.safe_load((repo_root / constraints).read_text(encoding="utf-8"))
    instrument = yaml.safe_load((repo_root / normalization).read_text(encoding="utf-8"))
    classes = {
        c["behavior_class"]: c["category_patterns"]
        for c in instrument["behavior_classes"]
    }
    content = _content_rules(rules)
    owner = {
        r["rule_id"]: _class_of(r["scope"]["category_contains"], classes)
        for r in content
    }

    with tempfile.TemporaryDirectory(prefix="socpilot_norm_ablation_") as scratch:
        work = Path(scratch)
        baseline = _measure(
            vha, repo_root, derivation_dir, heldout_dir, rules, work, "curated"
        )
        transfer_supporting = sorted(
            rule_id for rule_id, row in baseline.items() if row["meets_transfer_floor"]
        )

        complete = {rid: classes[owner[rid]] for rid in owner}
        class_complete = _measure(
            vha,
            repo_root,
            derivation_dir,
            heldout_dir,
            _rescope(rules, complete),
            work,
            "class_complete",
        )

        noise = classes[_NOISE_CLASS]
        folded = {rid: sorted(set(classes[owner[rid]]) | set(noise)) for rid in owner}
        noise_folded = _measure(
            vha,
            repo_root,
            derivation_dir,
            heldout_dir,
            _rescope(rules, folded),
            work,
            "noise_folded",
        )

        anchored = _string_anchored_scopes(content, derivation_categories)
        string_anchored = _measure(
            vha,
            repo_root,
            derivation_dir,
            heldout_dir,
            _rescope(rules, anchored),
            work,
            "string_anchored",
        )

        ctx = (vha, repo_root, derivation_dir, heldout_dir, work)
        loo = _leave_one_out(ctx, rules, content, transfer_supporting)

    def survivors(variant: Variant) -> list[str]:
        return sorted(
            rule_id
            for rule_id in transfer_supporting
            if variant[rule_id]["meets_transfer_floor"]
        )

    droppable = [rid for rid, v in loo.items() if v["droppable"]]
    return {
        "schema_version": 1,
        "constraints_path": constraints,
        "constraints_sha256": _sha256(repo_root / constraints),
        "constraints_predicate_sha256": predicate_digest(repo_root / constraints),
        "normalization_path": normalization,
        "normalization_sha256": _sha256(repo_root / normalization),
        "behavior_class_count": len(classes),
        "heldout_floor": _HELDOUT_FLOOR,
        "transfer_supporting_content_rules": transfer_supporting,
        "derivation_category_string_count": len(derivation_categories),
        "variants": {
            "curated": baseline,
            "class_complete": class_complete,
            "noise_folded": noise_folded,
            "string_anchored": string_anchored,
        },
        "survivors": {
            "curated": survivors(baseline),
            "class_complete": survivors(class_complete),
            "noise_folded": survivors(noise_folded),
            "string_anchored": survivors(string_anchored),
        },
        "leave_one_out": loo,
        "leave_one_out_drops": sum(loo[r]["patterns"] for r in droppable),
        "leave_one_out_min_heldout": min(loo[r]["worst_heldout"] for r in droppable),
        "leave_one_out_all_survive": all(loo[r]["survives_floor"] for r in droppable),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--derivation-dir", default="dataset/incidents")
    parser.add_argument("--heldout-dir", default="dataset/incidents_heldout")
    parser.add_argument(
        "--constraints",
        default="policy/constraints.yaml",
    )
    parser.add_argument(
        "--normalization-yaml",
        default="policy/telemetry_normalization.yaml",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--macros-output")
    args = parser.parse_args(argv)

    repo_root = resolve_repo_root(args.repo_root)

    def resolve(value: str) -> Path:
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (repo_root / path).resolve()

    report = build_report(
        repo_root,
        args.constraints,
        args.normalization_yaml,
        resolve(args.derivation_dir),
        resolve(args.heldout_dir),
    )

    out = repo_root / args.output_json
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    out.write_text(payload, encoding="utf-8")
    print(f"Wrote normalization ablation: {out}")
    total = len(report["transfer_supporting_content_rules"])
    for name in ("curated", "class_complete", "noise_folded"):
        kept = len(report["survivors"][name])
        print(f"  {name:15s} transfer survivors: {kept}/{total}")
    print(
        f"  leave_one_out    worst transfer check: {report['leave_one_out_min_heldout']} "
        f"(floor {_HELDOUT_FLOOR}); all survive: {report['leave_one_out_all_survive']}"
    )

    if args.macros_output:
        macros = repo_root / args.macros_output
        macros.parent.mkdir(parents=True, exist_ok=True)
        macros.write_text(
            render_macros(report),
            encoding="utf-8",
        )
        print(f"Wrote macros: {macros}")


if __name__ == "__main__":
    main()
