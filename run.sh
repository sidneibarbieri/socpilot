#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1

if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  PYTHON="${VIRTUAL_ENV}/bin/python"
elif [[ -x "${ROOT}/.venv/bin/python" ]]; then
  PYTHON="${ROOT}/.venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi

artifact_stage() {
  "${PYTHON}" - <<'PY'
import json
from pathlib import Path

path = Path("artifact_manifest.json")
if not path.is_file():
    print("development_workspace")
else:
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(payload.get("artifact_stage", "unknown"))
PY
}

show_help() {
  if [[ "$(artifact_stage)" == "pre_execution_source" ]]; then
    cat <<'EOF'
Usage:
  bash run.sh <command> [options]

Source-package commands:
  validate-public-artifact  Run integrity, dataset, consistency, and hygiene gates.
  artifact-verify           Verify package files against the hash manifest.
  dataset-audit             Recompute dataset quality and readiness metrics.
  release-audit             Scan the package for release-boundary violations.
  recompute-policy-evidence Recompute transfer-check evidence.
  policy-graph-validate     Validate the policy graph against the typed rules.
  rule-coverage             Exercise every declared rule without provider calls.

Use `bash provider_run.sh help` for the controlled provider evaluation. This
source package contains no model results, per-cell verdicts, or report
renderings. All commands above are zero-cost.
EOF
    return
  fi
  cat <<'EOF'
Usage:
  bash run.sh <command> [options]

Commands:
  reproduce-results         Validate the package and regenerate evidence reports.
  validate-public-artifact  Run integrity, dataset, consistency, and hygiene gates.
  artifact-verify           Verify package files against the hash manifest.
  dataset-audit             Recompute dataset quality and readiness metrics.
  release-audit             Scan the package for release-boundary violations.
  policy-surface-audit      Audit rule evidence and trace availability.
  recompute-policy-evidence Recompute transfer-check evidence.
  policy-graph-validate     Validate the policy graph against the typed rules.
  recount-aggregates        Recount every reported rate from the per-cell verdicts.

The public path is zero-cost and performs no provider calls.
EOF
}

require_reported_results() {
  local command="$1"
  if [[ "$(artifact_stage)" != "pre_execution_source" ]]; then
    return
  fi
  echo "${command} requires a reported-results package with provider evidence." >&2
  echo "This package is the pre-execution evaluation source." >&2
  exit 2
}

require_dependencies() {
  if "${PYTHON}" - <<'PY' >/dev/null 2>&1
import dotenv
import pydantic
import yaml
PY
  then
    return
  fi

  cat >&2 <<'EOF'
Required Python dependencies are unavailable.

Create the local environment:
  python3 -m venv .venv
  .venv/bin/pip install -r requirements.txt -r requirements-dev.txt
EOF
  exit 1
}

validate_artifact() {
  echo "[1/6] Verifying package manifest"
  "${PYTHON}" -m soc_llm_policy.artifact_verify --package-root .

  echo "[2/6] Recomputing Export Batch 1 audit"
  audit_dataset

  echo "[3/6] Recomputing Export Batch 2 audit"
  "${PYTHON}" -m soc_llm_policy.dataset_audit \
    --repo-root . \
    --dataset-dir artifact_data/heldout \
    --all \
    --output-json artifact_outputs/analysis/heldout_dataset_audit_recheck.json \
    --readiness-json \
      artifact_outputs/analysis/heldout_corpus_readiness_recheck.json \
    --fail-on-error \
    --fail-on-pii

  echo "[4/6] Recomputing global assessment"
  local assessment_cmd=(
    "${PYTHON}" -m soc_llm_policy.global_artifact_assessment
    --repo-root . \
    --all \
    --output-json \
      artifact_outputs/analysis/global_artifact_assessment_recheck.json \
    --output-md \
      artifact_outputs/analysis/global_artifact_assessment_recheck.md
  )
  if [[ -f artifact_outputs/analysis/official_evaluation_summary.json ]]; then
    assessment_cmd+=(
      --official-summary-json
        artifact_outputs/analysis/official_evaluation_summary.json
      --official-summary-output-json
        artifact_outputs/analysis/official_evaluation_summary_recheck.json
    )
  fi
  "${assessment_cmd[@]}"

  echo "[5/6] Checking sanitized provider evidence"
  if [[ -f artifact_outputs/analysis/official_llm_analysis_bundle.json ]]; then
    "${PYTHON}" -m soc_llm_policy.official_public_recheck \
      --repo-root . \
      --output-json artifact_outputs/analysis/official_public_consistency.json
  else
    if [[ "$(artifact_stage)" == "pre_execution_source" ]]; then
      echo "No provider evidence is present in this pre-execution package."
    else
      echo "Reported-results package is missing provider evidence." >&2
      exit 1
    fi
  fi

  echo "[6/6] Auditing release boundary"
  "${PYTHON}" -m soc_llm_policy.release_audit \
    --repo-root . \
    --allow-local-env \
    --output-json artifact_outputs/analysis/release_readiness_recheck.json

  echo "Public artifact validation passed."
}

# One definition per command. The reproduce path and the documented command call
# the same helper, so a flag the release needs cannot be present in one and
# missing from the other.
audit_dataset() {
  "${PYTHON}" -m soc_llm_policy.dataset_audit \
    --repo-root . \
    --all \
    --output-json artifact_outputs/analysis/dataset_audit_recheck.json \
    --readiness-json artifact_outputs/analysis/corpus_readiness_recheck.json \
    "$@"
}

audit_policy_surface() {
  "${PYTHON}" -m soc_llm_policy.policy_surface_audit \
    --repo-root . \
    --dataset-dir artifact_data/dataset \
    --constraints artifact_data/global/constraints.yaml \
    --official-bundle \
      artifact_outputs/analysis/official_llm_analysis_bundle.json \
    --repeat-assignments \
      artifact_outputs/analysis/repeat_stability/repeat_assignments.json \
    --output-json artifact_outputs/analysis/policy_surface_audit_recheck.json \
    --output-markdown artifact_outputs/analysis/policy_surface_audit_recheck.md \
    "$@"
}

partition_count() {
  local partition="$1"
  "${PYTHON}" - "${partition}" <<'PY'
import json
import sys
from pathlib import Path

partition = sys.argv[1]
manifest = json.loads(Path("artifact_manifest.json").read_text(encoding="utf-8"))
partitions = manifest.get("partitions", {})
entry = partitions.get(partition)
if not isinstance(entry, dict) or not isinstance(entry.get("incident_count"), int):
    raise SystemExit(f"Manifest has no incident count for partition {partition!r}.")
print(entry["incident_count"])
PY
}

recompute_policy_evidence() {
  local output_dir="artifact_outputs/recomputed/policy"
  mkdir -p "${output_dir}"
  echo "Recomputing export-batch evidence from released incident packages"
  "${PYTHON}" scripts/audit_corpus_partitions.py \
    --repo-root . \
    --derivation-dir artifact_data/dataset \
    --heldout-dir artifact_data/heldout \
    --expected-derivation-count "$(partition_count export_batch_1)" \
    --expected-heldout-count "$(partition_count export_batch_2)" \
    --output-json "${output_dir}/corpus_partition_audit.json" \
    --output-md "${output_dir}/corpus_partition_audit.md"

  echo "Recomputing Batch 2 policy-transfer activation"
  "${PYTHON}" scripts/validate_heldout_activation.py \
    --repo-root . \
    --derivation-dir artifact_data/dataset \
    --heldout-dir artifact_data/heldout \
    --constraints artifact_data/global/constraints.yaml \
    --output-json "${output_dir}/heldout_activation.json" \
    --output-md "${output_dir}/heldout_activation.md"

  echo "Recomputing normalization sensitivity"
  "${PYTHON}" scripts/run_normalization_ablation.py \
    --repo-root . \
    --derivation-dir artifact_data/dataset \
    --heldout-dir artifact_data/heldout \
    --constraints artifact_data/global/constraints.yaml \
    --normalization-yaml artifact_data/global/telemetry_normalization.yaml \
    --output-json "${output_dir}/normalization_ablation.json"
}

validate_policy_graph() {
  "${PYTHON}" scripts/policy_graph.py validate \
    --graph artifact_data/global/policy_graph.json
  echo "Policy graph matches its typed rules and action catalog."
}

recount_aggregates() {
  echo "Recounting every reported aggregate from the per-cell verdict record"
  local cells
  cells=$(find artifact_outputs/experiments -name policy_family_cells.jsonl 2>/dev/null | sort)
  if [[ -z "${cells}" ]]; then
    echo "No per-cell verdict record shipped with this release." >&2
    exit 1
  fi
  # shellcheck disable=SC2086
  "${PYTHON}" scripts/recount_aggregates_from_cells.py \
    --repo-root . \
    --bundle-json artifact_outputs/analysis/official_llm_analysis_bundle.json \
    --cells-jsonl ${cells}
}

reproduce_results() {
  validate_artifact

  echo "Auditing policy evidence"
  audit_policy_surface

  echo "Checking rule coverage (every declared rule activates)"
  "${PYTHON}" scripts/build_rule_coverage_report.py \
    --repo-root . \
    --output-json artifact_outputs/analysis/rule_coverage_report.json

  echo "Writing reproduction report"
  "${PYTHON}" -m soc_llm_policy.reproduction_report \
    --repo-root . \
    --output-json artifact_outputs/reproduction_report.json \
    --output-md artifact_outputs/reproduction_report.md

  echo "Rendering evidence dashboard"
  "${PYTHON}" scripts/render_audit_dashboard.py \
    --analysis-dir artifact_outputs/analysis \
    --output-html artifact_outputs/dashboard/index.html \
    --landing-html REVIEWER_DASHBOARD.html

  echo "Recounting aggregates from the cells that produced them"
  recount_aggregates

  echo "Zero-cost reproduction passed."
}

COMMAND="${1:-}"
case "${COMMAND}" in
  ""|-h|--help|help)
    show_help
    exit 0
    ;;
esac

require_dependencies
case "${COMMAND}" in
  recount-aggregates)
    require_reported_results "recount-aggregates"
    recount_aggregates
    ;;
  reproduce-results)
    require_reported_results "reproduce-results"
    reproduce_results
    ;;
  validate-public-artifact)
    validate_artifact
    ;;
  artifact-verify)
    shift
    exec "${PYTHON}" -m soc_llm_policy.artifact_verify --package-root . "$@"
    ;;
  dataset-audit)
    shift
    audit_dataset "$@"
    ;;
  release-audit)
    shift
    exec "${PYTHON}" -m soc_llm_policy.release_audit --repo-root . "$@"
    ;;
  policy-surface-audit)
    require_reported_results "policy-surface-audit"
    shift
    audit_policy_surface "$@"
    ;;
  recompute-policy-evidence)
    recompute_policy_evidence
    ;;
  policy-graph-validate)
    validate_policy_graph
    ;;
  rule-coverage)
    shift
    exec "${PYTHON}" scripts/build_rule_coverage_report.py --repo-root . "$@"
    ;;
  *)
    echo "Unknown command: ${COMMAND}" >&2
    show_help >&2
    exit 2
    ;;
esac
