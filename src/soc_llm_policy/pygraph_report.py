from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelViolationPoint:
    model_label: str
    violation_rate: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True)
class RuleViolationPoint:
    model_label: str
    rule_id: str
    violation_count: int


def _load_bundle(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid bundle: {path}")
    return data


def _extract_model_violation_points(
    bundle: dict[str, Any],
) -> list[ModelViolationPoint]:
    rows = bundle.get("by_model", [])
    out: list[ModelViolationPoint] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        out.append(
            ModelViolationPoint(
                model_label=str(row.get("model_label", "unknown")),
                violation_rate=float(row.get("run_violation_rate", 0.0)),
                ci_low=float(row.get("run_violation_rate_ci_low", 0.0)),
                ci_high=float(row.get("run_violation_rate_ci_high", 0.0)),
            )
        )
    return out


def _extract_rule_violation_points(bundle: dict[str, Any]) -> list[RuleViolationPoint]:
    rows = bundle.get("by_rule", [])
    out: list[RuleViolationPoint] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        out.append(
            RuleViolationPoint(
                model_label=str(row.get("model_label", "unknown")),
                rule_id=str(row.get("rule_id", "unknown")),
                violation_count=int(row.get("violation_count", 0)),
            )
        )
    return out


def _extract_attack_tactic_counts(bundle: dict[str, Any]) -> dict[str, int]:
    summary = bundle.get("summary", {})
    if not isinstance(summary, dict):
        return {}
    raw = summary.get("attack_tactic_counts", {})
    if not isinstance(raw, dict):
        return {}
    out: dict[str, int] = {}
    for key, value in raw.items():
        out[str(key)] = int(value)
    return out


def _extract_token_cost(bundle: dict[str, Any]) -> tuple[int, float]:
    summary = bundle.get("summary", {})
    if not isinstance(summary, dict):
        return 0, 0.0
    tokens = int(summary.get("llm_total_tokens_total", 0))
    cost = float(summary.get("llm_cost_estimated_usd_total", 0.0))
    return tokens, cost


def _lazy_import_pyplot() -> Any:
    try:
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg")  # headless-safe backend before pyplot import
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - environment without matplotlib
        raise RuntimeError(
            "matplotlib is not installed. Run: .venv/bin/pip install matplotlib"
        ) from exc
    return plt


def _plot_violation_rate(points: list[ModelViolationPoint], outpath: Path) -> None:
    if not points:
        return
    plt = _lazy_import_pyplot()

    labels = [point.model_label for point in points]
    rates = [point.violation_rate for point in points]
    err_low = [max(point.violation_rate - point.ci_low, 0.0) for point in points]
    err_high = [max(point.ci_high - point.violation_rate, 0.0) for point in points]

    figure, axes = plt.subplots(figsize=(9, 5))
    axes.bar(labels, rates, color="#3b82f6", alpha=0.85)
    axes.errorbar(
        labels,
        rates,
        yerr=[err_low, err_high],
        fmt="none",
        ecolor="#111827",
        capsize=5,
    )
    axes.set_ylim(0, 1)
    axes.set_ylabel("Run Violation Rate")
    axes.set_title("Violation Rate by Model (95% Wilson CI)")
    axes.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(outpath, dpi=150)
    plt.close(figure)


def _plot_rule_breakdown(points: list[RuleViolationPoint], outpath: Path) -> None:
    if not points:
        return
    plt = _lazy_import_pyplot()

    rules = sorted({point.rule_id for point in points})
    models = sorted({point.model_label for point in points})
    series: dict[str, list[int]] = {
        model: [0 for _ in rules] for model in models
    }
    index_by_rule = {rule: index for index, rule in enumerate(rules)}
    for point in points:
        series[point.model_label][index_by_rule[point.rule_id]] += point.violation_count

    width = 0.8 / max(len(models), 1)
    rule_positions = list(range(len(rules)))

    figure, axes = plt.subplots(figsize=(10, 5))
    for pos, model in enumerate(models):
        offset = (pos - (len(models) - 1) / 2) * width
        shifted = [value + offset for value in rule_positions]
        axes.bar(shifted, series[model], width=width, label=model)

    axes.set_xticks(rule_positions)
    axes.set_xticklabels(rules)
    axes.set_ylabel("Violation Count")
    axes.set_title("Violations by Rule and Model")
    axes.legend()
    axes.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(outpath, dpi=150)
    plt.close(figure)


def _plot_attack_tactics(tactic_counts: dict[str, int], outpath: Path) -> None:
    if not tactic_counts:
        return
    plt = _lazy_import_pyplot()

    labels = list(tactic_counts.keys())
    values = list(tactic_counts.values())

    figure, axes = plt.subplots(figsize=(10, 5))
    axes.bar(labels, values, color="#10b981", alpha=0.85)
    axes.set_ylabel("Match Count")
    axes.set_title("ATT&CK Tactic Frequency")
    axes.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(outpath, dpi=150)
    plt.close(figure)


def _plot_budget(tokens: int, cost_usd: float, outpath: Path) -> None:
    plt = _lazy_import_pyplot()
    figure, axes = plt.subplots(figsize=(6, 4))
    labels = ["Total Tokens", "Estimated Cost (USD)"]
    values = [float(tokens), float(cost_usd)]
    axes.bar(labels, values, color=["#6366f1", "#f59e0b"])
    axes.set_title("Experiment Usage Budget")
    axes.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(outpath, dpi=150)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="soc_llm_policy.pygraph_report")
    parser.add_argument(
        "--analysis-bundle-json",
        default="results/analysis/analysis_bundle.json",
        help="Typed bundle generated by analyze.py.",
    )
    parser.add_argument(
        "--outdir",
        default="results/analysis/figures",
        help="Output directory for PNG figures.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    bundle_path = Path(args.analysis_bundle_json).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    bundle = _load_bundle(bundle_path)
    model_points = _extract_model_violation_points(bundle)
    rule_points = _extract_rule_violation_points(bundle)
    tactic_counts = _extract_attack_tactic_counts(bundle)
    tokens, cost_usd = _extract_token_cost(bundle)

    paths = {
        "violation_rate": outdir / "violation_rate_by_model.png",
        "rule_breakdown": outdir / "violations_by_rule_model.png",
        "attack_tactics": outdir / "attack_tactic_frequency.png",
        "budget": outdir / "usage_budget.png",
    }
    _plot_violation_rate(model_points, paths["violation_rate"])
    _plot_rule_breakdown(rule_points, paths["rule_breakdown"])
    _plot_attack_tactics(tactic_counts, paths["attack_tactics"])
    _plot_budget(tokens, cost_usd, paths["budget"])

    print(f"Figures saved at: {outdir}")
    for name, path in paths.items():
        if path.exists():
            print(f" - {name}: {path}")


if __name__ == "__main__":
    main()
