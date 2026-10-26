from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from soc_llm_policy.io import read_json
from soc_llm_policy.paths import RepoPaths, resolve_repo_root
from soc_llm_policy.pipeline import (
    VerifierRunOptions,
    ingest_globals,
    ingest_incident,
    list_inbox_incidents,
    run_verifier,
)
from soc_llm_policy.result_models import LLMUsage

_VALID_ARMS = {"llm_zero", "llm_policy_prompt"}


@dataclass(frozen=True)
class SavedLLMOutput:
    incident_id: str
    source_path: Path
    source_run_tag: str
    llm_deployment: str
    llm_arm: Literal["llm_zero", "llm_policy_prompt"]
    actions: list[str]
    usage: LLMUsage | None
    latency_ms: int | None
    estimated_cost_usd: float | None


def _sanitize_tag_component(value: str) -> str:
    safe = "".join(ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in value)
    return safe.strip("_") or "unknown"


def _utc_suffix() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def _parse_source_run_tag(path: Path) -> str:
    if path.name == "llm_raw_actions.json":
        return "canonical"
    prefix = "llm_raw_actions_"
    suffix = ".json"
    if path.name.startswith(prefix) and path.name.endswith(suffix):
        return path.name[len(prefix) : -len(suffix)]
    raise ValueError(f"Unexpected saved LLM output filename: {path.name}")


def _load_saved_output(path: Path, incident_id: str) -> SavedLLMOutput | None:
    payload = read_json(path)
    llm_deployment = str(payload.get("model", "")).strip()
    llm_arm = str(payload.get("arm", "")).strip()
    raw_actions = payload.get("actions", [])
    if (
        not llm_deployment
        or llm_arm not in _VALID_ARMS
        or not isinstance(raw_actions, list)
    ):
        return None

    actions = [str(item).strip() for item in raw_actions if str(item).strip()]
    if not actions:
        return None

    usage_payload = payload.get("usage")
    usage = None
    if isinstance(usage_payload, dict):
        usage = LLMUsage.model_validate(usage_payload)

    latency_raw = payload.get("latency_ms")
    latency_ms = int(latency_raw) if isinstance(latency_raw, int | float) else None

    cost_raw = payload.get("estimated_cost_usd")
    estimated_cost_usd = (
        float(cost_raw) if isinstance(cost_raw, int | float) else None
    )

    return SavedLLMOutput(
        incident_id=incident_id,
        source_path=path,
        source_run_tag=_parse_source_run_tag(path),
        llm_deployment=llm_deployment,
        llm_arm=llm_arm,
        actions=actions,
        usage=usage,
        latency_ms=latency_ms,
        estimated_cost_usd=estimated_cost_usd,
    )


def _collect_latest_saved_outputs(
    paths: RepoPaths,
    incident_ids: list[str],
) -> list[SavedLLMOutput]:
    latest: dict[tuple[str, str, str], SavedLLMOutput] = {}
    for incident_id in incident_ids:
        incident_dir = paths.outputs_incident_dir(incident_id)
        versioned = sorted(incident_dir.glob("llm_raw_actions_*.json"))
        candidates = versioned
        if not candidates:
            canonical = paths.outputs_llm_actions_path(incident_id)
            candidates = [canonical] if canonical.exists() else []
        for path in candidates:
            saved = _load_saved_output(path, incident_id)
            if saved is None:
                continue
            key = (saved.incident_id, saved.llm_deployment, saved.llm_arm)
            current = latest.get(key)
            if current is None or saved.source_run_tag > current.source_run_tag:
                latest[key] = saved
    selected = list(latest.values())
    selected.sort(
        key=lambda item: (item.incident_id, item.llm_deployment, item.llm_arm)
    )
    return selected


def _build_replay_run_tag(
    source_run_tag: str,
    approval_policy_mode: Literal["remove", "defer_to_human_approval"],
) -> str:
    return (
        f"replay_{_sanitize_tag_component(source_run_tag)}"
        f"_approval_{_sanitize_tag_component(approval_policy_mode)}"
        f"_{_utc_suffix()}"
    )


def replay_saved_outputs(
    *,
    paths: RepoPaths,
    incident_ids: list[str],
    approval_policy_mode: Literal["remove", "defer_to_human_approval"],
) -> dict[str, Any]:
    ingest_globals(paths)
    saved_outputs = _collect_latest_saved_outputs(paths, incident_ids)
    if not saved_outputs:
        raise FileNotFoundError(
            "No saved llm_raw_actions outputs were found for the selected incidents. "
            "Run a paid probe first, then replay before cleaning the workspace."
        )

    replayed_runs: list[dict[str, Any]] = []
    for incident_id in sorted({item.incident_id for item in saved_outputs}):
        ingest_incident(
            paths,
            incident_id,
            clean_target=False,
            strict_data=True,
        )

    for saved in saved_outputs:
        replay_run_tag = _build_replay_run_tag(
            saved.source_run_tag,
            approval_policy_mode=approval_policy_mode,
        )
        run_verifier(
            paths,
            saved.incident_id,
            options=VerifierRunOptions(
                run_tag=replay_run_tag,
                strict_data=True,
                approval_policy_mode=approval_policy_mode,
                write_canonical_output=False,
                llm_actions=saved.actions,
                llm_deployment=saved.llm_deployment,
                llm_arm=saved.llm_arm,
                llm_usage=saved.usage,
                llm_latency_ms=saved.latency_ms,
                llm_cost_estimated_usd=saved.estimated_cost_usd,
            ),
        )
        replayed_runs.append(
            {
                "incident_id": saved.incident_id,
                "llm_deployment": saved.llm_deployment,
                "llm_arm": saved.llm_arm,
                "source_run_tag": saved.source_run_tag,
                "source_path": str(saved.source_path),
                "approval_policy_mode": approval_policy_mode,
                "replay_run_tag": replay_run_tag,
                "action_count": len(saved.actions),
                "estimated_cost_usd_reused": saved.estimated_cost_usd,
            }
        )

    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "approval_policy_mode": approval_policy_mode,
        "incident_count": len(sorted({item["incident_id"] for item in replayed_runs})),
        "replayed_run_count": len(replayed_runs),
        "replayed_runs": replayed_runs,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soc_llm_policy.replay_saved_llm_outputs",
        description=(
            "Replay the verifier over saved llm_raw_actions outputs using a different "
            "approval handling mode, without calling the LLM again."
        ),
    )
    parser.add_argument("--repo-root", default=None)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--incidents", default=None, help="CSV incident list.")
    group.add_argument("--all", action="store_true")
    parser.add_argument(
        "--approval-policy-mode",
        default="defer_to_human_approval",
        choices=["remove", "defer_to_human_approval"],
        help="Approval handling mode to replay against the saved raw LLM actions.",
    )
    parser.add_argument(
        "--output-json",
        default="results/analysis/replay_saved_llm_outputs.json",
        help="Path to the replay summary JSON.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = resolve_repo_root(args.repo_root)
    paths = RepoPaths(repo_root=repo_root)

    if args.all:
        incident_ids = list_inbox_incidents(paths)
    else:
        incident_ids = sorted(
            {
                item.strip()
                for item in str(args.incidents).split(",")
                if item.strip()
            }
        )
        if not incident_ids:
            raise ValueError("Provide --incidents INC_001,INC_002 or use --all.")

    report = replay_saved_outputs(
        paths=paths,
        incident_ids=incident_ids,
        approval_policy_mode=args.approval_policy_mode,
    )

    output_json = Path(args.output_json)
    if not output_json.is_absolute():
        output_json = repo_root / output_json
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        "Replay completed: "
        f"{report['replayed_run_count']} saved LLM run(s) re-evaluated "
        f"with approval mode '{report['approval_policy_mode']}'."
    )
    print(f"Summary saved at: {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
