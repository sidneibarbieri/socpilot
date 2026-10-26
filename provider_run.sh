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

show_help() {
  cat <<'EOF'
Usage:
  bash provider_run.sh prepare
  bash provider_run.sh approve --plan PATH --budget-usd AMOUNT
  bash provider_run.sh execute --plan PATH [--validate-only] [--resume] [--lease-seconds N]
  bash provider_run.sh finalize --plan PATH
  bash provider_run.sh package --plan PATH [--output-dir PATH]
  bash provider_run.sh status --plan PATH
  bash provider_run.sh dashboard --plan PATH
  bash provider_run.sh backup create --plan PATH --destination NEW_DIRECTORY
  bash provider_run.sh backup verify --backup DIRECTORY --manifest-sha256 SHA256
  bash provider_run.sh backup restore --backup DIRECTORY --manifest-sha256 SHA256 --destination NEW_DIRECTORY

prepare is zero-cost. It materializes the 400-incident corpus declared by the
published protocol manifest and derives one 4,800-cell plan. It cannot
change models, conditions, repeats, or protocol settings. execute requires an
approved plan and provider credentials supplied through the local environment
or .env. Before scientific cell 1, execute performs one bounded live request
for every declared model profile. All profiles must pass or the run stops.
EOF
}

plan_value() {
  local plan="$1"
  local key="$2"
  "${PYTHON}" - "${plan}" "${key}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
key = sys.argv[2]
if key == "incident_count":
    print(len(payload["incident_ids"]))
    raise SystemExit
value = payload
for part in key.split("."):
    value = value[part]
print(",".join(str(item) for item in value) if isinstance(value, list) else value)
PY
}

study_value() {
  local plan="$1"
  local study_id="$2"
  local key="$3"
  "${PYTHON}" - "${plan}" "${study_id}" "${key}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
matches = [
    study
    for study in payload.get("diagnostic_studies", [])
    if study.get("study_id") == sys.argv[2]
]
if len(matches) != 1:
    raise SystemExit(f"Plan must contain one diagnostic study: {sys.argv[2]}")
value = matches[0]
for part in sys.argv[3].split("."):
    value = value[part]
print(",".join(str(item) for item in value) if isinstance(value, list) else value)
PY
}

primary_max_completion_tokens() {
  local plan="$1"
  "${PYTHON}" - "${plan}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
limits = {
    int(item["max_completion_tokens_per_request"])
    for item in payload["cost_estimate"]["models"]
    if item.get("study_id") == "primary"
}
if len(limits) != 1:
    raise SystemExit("Primary model profiles must share one completion-token cap.")
print(limits.pop())
PY
}

show_campaign_size() {
  local plan="$1"
  echo "Primary scientific cells: $(plan_value "${plan}" primary_cell_count)"
  echo "Diagnostic task ceiling: $(plan_value "${plan}" diagnostic_task_ceiling)"
  echo "Scientific task ceiling: $(plan_value "${plan}" scientific_task_ceiling)"
  echo "Provider attempts per scientific task: $(plan_value "${plan}" provider_transport_attempt_limit)"
  echo "Provider-readiness requests: $(plan_value "${plan}" provider_readiness.request_count)"
  echo "Provider-readiness attempt ceiling: $(plan_value "${plan}" provider_readiness.provider_attempt_ceiling)"
  echo "Provider transport-attempt ceiling: $(plan_value "${plan}" provider_transport_attempt_ceiling)"
  echo "Expected provider cost: USD $(plan_value "${plan}" cost_estimate.estimated_total_usd)"
  echo "Planned budget ceiling: USD $(plan_value "${plan}" cost_estimate.approval_budget_with_headroom_usd)"
  echo "Approval status: $(plan_value "${plan}" approval.status)"
}

freeze_input_path() {
  local freeze="$1"
  local role="$2"
  "${PYTHON}" - "${freeze}" "${role}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
entry = payload.get("input_hashes", {}).get(sys.argv[2])
if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
    raise SystemExit(f"Protocol freeze is missing input path: {sys.argv[2]}")
print(entry["path"])
PY
}

prepare_run() {
  if [[ $# -ne 0 ]]; then
    echo "prepare accepts no options; scientific settings come from the published freeze." >&2
    exit 2
  fi
  local state_dir="artifact_outputs/provider_state/canonical_campaign"
  local freeze="artifact_outputs/analysis/protocol_freeze.json"
  local plan="${state_dir}/provider_execution_plan.json"

  bash run.sh validate-public-artifact
  if [[ ! -f "${freeze}" ]]; then
    echo "Published protocol freeze is missing: ${freeze}" >&2
    exit 1
  fi
  local campaign_manifest execution_source_manifest campaign_view
  local dataset_release_id protocol_version models arms repeats
  campaign_manifest=$(freeze_input_path "${freeze}" corpus_campaign_manifest)
  execution_source_manifest=$(freeze_input_path "${freeze}" execution_source_manifest)
  campaign_view=$(plan_value "${campaign_manifest}" view_path)
  dataset_release_id=$(plan_value "${freeze}" dataset_release_id)
  protocol_version=$(plan_value "${freeze}" eval_protocol_version)
  models=$(plan_value "${freeze}" official_models)
  arms=$(plan_value "${freeze}" arms)
  repeats=$(plan_value "${freeze}" repeats)

  "${PYTHON}" - "${freeze}" <<'PY'
import json
import sys
from pathlib import Path

freeze = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
dimensions = {
    "incidents": len(freeze.get("incident_ids", [])),
    "models": len(freeze.get("official_models", [])),
    "conditions": len(freeze.get("arms", [])),
    "repeats": freeze.get("repeats"),
}
if dimensions != {"incidents": 400, "models": 2, "conditions": 2, "repeats": 3}:
    raise SystemExit(f"Published protocol has unexpected dimensions: {dimensions}")
if freeze.get("repeat_reduction") != "majority":
    raise SystemExit("Published freeze must use majority repeat reduction.")
if 400 * 2 * 2 * 3 != 4800:
    raise SystemExit("Internal evaluation-dimension assertion failed.")
PY

  "${PYTHON}" -m soc_llm_policy.execution_source \
    --repo-root . \
    --manifest "${execution_source_manifest}" \
    --validate-only

  if [[ -f "${plan}" ]]; then
    "${PYTHON}" -m soc_llm_policy.corpus_campaign \
      --repo-root . \
      --view "${campaign_view}" \
      --manifest "${campaign_manifest}" \
      --validate-only
    validate_plan "${plan}" "${campaign_view}" false
    echo "Provider state already prepared and valid."
    echo "Plan: ${plan}"
    show_campaign_size "${plan}"
    return 0
  fi
  if [[ -e "${state_dir}" ]]; then
    echo "Partial provider state exists without a valid plan: ${state_dir}" >&2
    exit 1
  fi

  "${PYTHON}" -m soc_llm_policy.corpus_campaign \
    --repo-root . \
    --view "${campaign_view}" \
    --manifest "${campaign_manifest}" \
    --materialize-view

  if ! SOC_LLM_POLICY_INBOX_INCIDENTS_DIR="${campaign_view}" \
    "${PYTHON}" scripts/build_provider_execution_plan.py \
    --repo-root . \
    --all \
    --output-json "${plan}" \
    --protocol-freeze-json "${freeze}" \
    --dataset-release-id "${dataset_release_id}" \
    --eval-protocol-version "${protocol_version}" \
    --models "${models}" \
    --arms "${arms}" \
    --repeats "${repeats}" \
    --corpus-campaign-manifest "${campaign_manifest}"; then
    rm -rf "${state_dir}"
    return 1
  fi

  echo "Prepared provider state without provider calls."
  echo "Plan: ${plan}"
  show_campaign_size "${plan}"
}

approve_run() {
  local plan=""
  local budget=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --plan)
        plan="$2"
        shift 2
        ;;
      --budget-usd)
        budget="$2"
        shift 2
        ;;
      *)
        echo "Unknown approve option: $1" >&2
        exit 2
        ;;
    esac
  done
  if [[ -z "${plan}" || -z "${budget}" ]]; then
    echo "approve requires --plan and --budget-usd." >&2
    exit 2
  fi

  "${PYTHON}" -m soc_llm_policy.provider_plan approve \
    --plan "${plan}" \
    --budget-usd "${budget}"
}

validate_plan() {
  local plan="$1"
  local corpus_root="$2"
  local require_approval="${3:-true}"
  local output
  output="$(dirname "${plan}")/provider_execution_plan_validation.json"
  local cmd=(
    "${PYTHON}" scripts/validate_provider_execution_plan.py
    --repo-root .
    --plan-json "${plan}"
    --output-json "${output}"
  )
  if [[ "${require_approval}" == "true" ]]; then
    cmd+=(--require-approval)
  fi
  SOC_LLM_POLICY_INBOX_INCIDENTS_DIR="${corpus_root}" \
    "${cmd[@]}"
  "${PYTHON}" -m soc_llm_policy.campaign_diagnostics \
    --repo-root . \
    validate \
    --plan "${plan}"
}

execute_run() {
  local plan=""
  local resume="false"
  local validate_only="false"
  local lease_seconds="900"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --plan)
        plan="$2"
        shift 2
        ;;
      --resume)
        resume="true"
        shift
        ;;
      --validate-only)
        validate_only="true"
        shift
        ;;
      --lease-seconds)
        lease_seconds="$2"
        shift 2
        ;;
      *)
        echo "Unknown execute option: $1" >&2
        exit 2
        ;;
    esac
  done
  if [[ -z "${plan}" ]]; then
    echo "execute requires --plan." >&2
    exit 2
  fi
  if [[ "${validate_only}" != "true" ]]; then
    "${PYTHON}" -m soc_llm_policy.execution_storage "${ROOT}"
  fi
  bash run.sh validate-public-artifact

  local run_id models arms repeats repeat_reduction dataset_release_id protocol_version
  local incident_count freeze attempt_limit retry_wait corpus_root campaign
  run_id=$(plan_value "${plan}" plan_id)
  models=$(plan_value "${plan}" models)
  arms=$(plan_value "${plan}" arms)
  repeats=$(plan_value "${plan}" repeats)
  repeat_reduction=$(plan_value "${plan}" repeat_reduction)
  dataset_release_id=$(plan_value "${plan}" dataset_release_id)
  protocol_version=$(plan_value "${plan}" eval_protocol_version)
  incident_count=$(plan_value "${plan}" selection_basis.incident_count)
  freeze=$(plan_value "${plan}" frozen_inputs.protocol_freeze.path)
  corpus_root=$(plan_value "${plan}" selection_basis.source)
  campaign=$(plan_value "${plan}" frozen_inputs.corpus_campaign_manifest.path)
  attempt_limit=$(plan_value "${freeze}" retry_policy.llm_attempt_limit)
  retry_wait=$(plan_value "${freeze}" retry_policy.llm_retry_wait_seconds)
  mkdir -p "$(dirname "${plan}")/derived"
  "${PYTHON}" -m soc_llm_policy.corpus_campaign \
    --repo-root . \
    --view "${corpus_root}" \
    --manifest "${campaign}" \
    --validate-only
  if [[ "${validate_only}" == "true" ]]; then
    validate_plan "${plan}" "${corpus_root}" false
  else
    validate_plan "${plan}" "${corpus_root}" true
    "${PYTHON}" -m soc_llm_policy.provider_readiness \
      --repo-root . \
      execute \
      --plan "${plan}" \
      --output-json "$(dirname "${plan}")/derived/provider_readiness.json" \
      --attempt-log "$(dirname "${plan}")/derived/provider_readiness_attempts.jsonl"
  fi

  local cmd=(
    "${PYTHON}" -m soc_llm_policy.experiment_runner
    --repo-root .
    --all
    --models "${models}"
    --arms "${arms}"
    --repeats "${repeats}"
    --run-id "${run_id}"
    --dataset-release-id "${dataset_release_id}"
    --min-incidents "${incident_count}"
    --eval-protocol-version "${protocol_version}"
    --protocol-freeze-manifest "${freeze}"
    --execution-order paired_blocked
    --matrix-seed 0
    --repeat-reduction "${repeat_reduction}"
    --lease-seconds "${lease_seconds}"
    --fail-fast
    --summary-json "$(dirname "${plan}")/derived/primary_summary.json"
    --data-preflight-json "$(dirname "${plan}")/derived/primary_data_preflight.json"
    --data-quality-json "$(dirname "${plan}")/derived/primary_data_quality.json"
    --analysis-bundle-json "$(dirname "${plan}")/derived/primary_analysis_bundle.json"
    --failures-json "$(dirname "${plan}")/derived/primary_failures.json"
  )
  if [[ "${validate_only}" == "true" ]]; then
    cmd+=(--validate-protocol-freeze-only)
  elif [[ "${resume}" == "true" ]]; then
    cmd+=(--approved-budget-usd "$(plan_value "${plan}" cost_estimate.study_budget_ceiling_usd.primary)")
    cmd+=(--resume)
  else
    cmd+=(--approved-budget-usd "$(plan_value "${plan}" cost_estimate.study_budget_ceiling_usd.primary)")
    cmd+=(--clean-target)
  fi

  if [[ "${validate_only}" == "true" ]]; then
    echo "Validating provider plan without provider calls: ${run_id}"
  else
    echo "Executing approved provider plan: ${run_id}"
  fi
  SOC_LLM_POLICY_INBOX_INCIDENTS_DIR="${corpus_root}" \
  SOC_LLM_ATTEMPT_LIMIT="${attempt_limit}" \
  SOC_LLM_RETRY_WAIT_SECONDS="${retry_wait}" \
    "${cmd[@]}"

  execute_campaign_diagnostics \
    "${plan}" \
    "${corpus_root}" \
    "${validate_only}" \
    "${resume}" \
    "${lease_seconds}"
}

execute_diagnostic_task_study() {
  local plan="$1"
  local study_id="$2"
  local resume="$3"
  local lease_seconds="$4"
  local run_id
  run_id="$(plan_value "${plan}" plan_id)__${study_id}"
  local ledger="artifact_outputs/experiments/${run_id}/diagnostic_ledger.json"
  local cmd=(
    "${PYTHON}" -m soc_llm_policy.campaign_diagnostics
    --repo-root .
    execute
    --plan "${plan}"
    --study "${study_id}"
    --lease-seconds "${lease_seconds}"
  )
  if [[ "${resume}" == "true" && -f "${ledger}" ]]; then
    cmd+=(--resume)
  elif [[ "${resume}" == "true" && -d "$(dirname "${ledger}")" ]]; then
    echo "Partial diagnostic state has no ledger: $(dirname "${ledger}")" >&2
    exit 1
  fi
  "${cmd[@]}"
}

execute_adversarial_study() {
  local plan="$1"
  local corpus_root="$2"
  local validate_only="$3"
  local resume="$4"
  local lease_seconds="$5"
  local freeze dataset_release_id protocol_version
  local models arms repeats repeat_reduction matrix_seed scenario_manifest
  local scenario_bank_dir scenario_count run_id state_dir budget
  freeze=$(plan_value "${plan}" frozen_inputs.protocol_freeze.path)
  dataset_release_id=$(plan_value "${plan}" dataset_release_id)
  protocol_version=$(plan_value "${plan}" eval_protocol_version)
  models=$(study_value "${plan}" adversarial models)
  arms=$(study_value "${plan}" adversarial settings.arms)
  repeats=$(study_value "${plan}" adversarial settings.repeats)
  repeat_reduction=$(study_value "${plan}" adversarial settings.repeat_reduction)
  matrix_seed=$(study_value "${plan}" adversarial settings.matrix_seed)
  scenario_count=$(study_value "${plan}" adversarial settings.scenario_count)
  scenario_manifest=$(study_value "${plan}" adversarial frozen_inputs.adversarial_scenario_manifest.path)
  scenario_bank_dir=$(dirname "${scenario_manifest}")
  run_id="$(plan_value "${plan}" plan_id)__adversarial"
  state_dir="$(dirname "${plan}")/derived"
  budget=$(plan_value "${plan}" cost_estimate.study_budget_ceiling_usd.adversarial)

  local cmd=(
    "${PYTHON}" -m soc_llm_policy.experiment_runner
    --repo-root .
    --scenario-manifest "${scenario_manifest}"
    --scenario-bank-dir "${scenario_bank_dir}"
    --scenario-source-incidents "${corpus_root}"
    --protocol-study adversarial
    --models "${models}"
    --arms "${arms}"
    --repeats "${repeats}"
    --run-id "${run_id}"
    --dataset-release-id "${dataset_release_id}"
    --min-incidents "${scenario_count}"
    --eval-protocol-version "${protocol_version}"
    --protocol-freeze-manifest "${freeze}"
    --execution-order paired_blocked
    --matrix-seed "${matrix_seed}"
    --repeat-reduction "${repeat_reduction}"
    --lease-seconds "${lease_seconds}"
    --fail-fast
    --summary-json "${state_dir}/adversarial_summary.json"
    --data-preflight-json "${state_dir}/adversarial_data_preflight.json"
    --data-quality-json "${state_dir}/adversarial_data_quality.json"
    --analysis-bundle-json "${state_dir}/adversarial_analysis_bundle.json"
    --failures-json "${state_dir}/adversarial_failures.json"
  )
  if [[ "${validate_only}" == "true" ]]; then
    cmd+=(--validate-protocol-freeze-only)
  elif [[ "${resume}" == "true" && -f "artifact_outputs/experiments/${run_id}/run_ledger.json" ]]; then
    cmd+=(--approved-budget-usd "${budget}" --resume)
  elif [[ "${resume}" == "true" && -d "artifact_outputs/experiments/${run_id}" ]]; then
    echo "Partial adversarial state has no ledger: ${run_id}" >&2
    exit 1
  else
    cmd+=(--approved-budget-usd "${budget}" --clean-target)
  fi

  SOC_LLM_ATTEMPT_LIMIT="$(plan_value "${freeze}" retry_policy.llm_attempt_limit)" \
  SOC_LLM_RETRY_WAIT_SECONDS="$(plan_value "${freeze}" retry_policy.llm_retry_wait_seconds)" \
    "${cmd[@]}"
}

execute_campaign_diagnostics() {
  local plan="$1"
  local corpus_root="$2"
  local validate_only="$3"
  local resume="$4"
  local lease_seconds="$5"
  local sequence
  sequence=$(plan_value "${plan}" execution_sequence)
  IFS=',' read -r -a studies <<< "${sequence}"
  for study_id in "${studies[@]}"; do
    case "${study_id}" in
      primary)
        ;;
      independent_judge|judge_ablation|projection_fidelity)
        if [[ "${validate_only}" != "true" ]]; then
          execute_diagnostic_task_study \
            "${plan}" "${study_id}" "${resume}" "${lease_seconds}"
        fi
        ;;
      adversarial)
        execute_adversarial_study \
          "${plan}" "${corpus_root}" "${validate_only}" "${resume}" "${lease_seconds}"
        ;;
      *)
        echo "Unsupported study in approved plan: ${study_id}" >&2
        exit 1
        ;;
    esac
  done
}

show_status() {
  "${PYTHON}" -m soc_llm_policy.provider_status --repo-root . "$@"
}

render_dashboard() {
  if [[ $# -ne 2 || "${1:-}" != "--plan" ]]; then
    echo "dashboard requires --plan PATH." >&2
    exit 2
  fi
  local plan="$2"
  "${PYTHON}" scripts/render_execution_dashboard.py \
    --repo-root . --plan-json "${plan}" \
    --output-html "$(dirname "${plan}")/execution_dashboard.html"
}

finalize_run() {
  local plan=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --plan)
        plan="$2"
        shift 2
        ;;
      *)
        echo "Unknown finalize option: $1" >&2
        exit 2
        ;;
    esac
  done
  if [[ -z "${plan}" ]]; then
    echo "finalize requires --plan." >&2
    exit 2
  fi

  local run_id corpus_root output_dir dataset_release_id evidence_dir
  local primary_evidence adversarial_evidence campaign_evidence readiness_report
  local pair_manifest fidelity_probe adversarial_run
  local primary_bundle primary_ledger primary_repeats primary_token_cap
  run_id=$(plan_value "${plan}" plan_id)
  corpus_root=$(plan_value "${plan}" selection_basis.source)
  dataset_release_id=$(plan_value "${plan}" dataset_release_id)
  validate_plan "${plan}" "${corpus_root}"
  evidence_dir="artifact_outputs/evidence/${run_id}"
  primary_evidence="${evidence_dir}/primary_execution_evidence.json"
  adversarial_evidence="${evidence_dir}/adversarial_execution_evidence.json"
  campaign_evidence="${evidence_dir}/campaign_evidence.json"
  readiness_report="$(dirname "${plan}")/derived/provider_readiness.json"
  output_dir="artifact_outputs/analysis/figures"
  pair_manifest=$(study_value "${plan}" adversarial frozen_inputs.adversarial_pair_manifest.path)
  fidelity_probe=$(study_value "${plan}" projection_fidelity frozen_inputs.projection_fidelity_probe.path)
  adversarial_run="${run_id}__adversarial"
  primary_bundle="artifact_outputs/experiments/${run_id}/analysis_bundle.json"
  primary_ledger="artifact_outputs/experiments/${run_id}/run_ledger.json"
  primary_repeats=$(plan_value "${plan}" repeats)
  primary_token_cap=$(primary_max_completion_tokens "${plan}")

  "${PYTHON}" -m soc_llm_policy.campaign_diagnostics \
    --repo-root . \
    finalize \
    --plan "${plan}"

  "${PYTHON}" scripts/build_separation_report.py \
    --outputs-incidents-dir artifact_outputs/incidents \
    --analysis-bundle-json \
      "artifact_outputs/experiments/${run_id}/analysis_bundle.json" \
    --llm-judge-verdicts-json \
      artifact_outputs/analysis/llm_judge_verdicts.json \
    --run-id "${run_id}" \
    --dataset-release-id "${dataset_release_id}" \
    --output-json artifact_outputs/analysis/separation_report.json

  "${PYTHON}" scripts/build_adversarial_report.py \
    --run-matrix-json \
      "artifact_outputs/experiments/${adversarial_run}/run_matrix.json" \
    --adversarial-manifest-json "${pair_manifest}" \
    --outputs-incidents-dir artifact_outputs/incidents \
    --output-json artifact_outputs/analysis/adversarial_report.json

  local fidelity_models fidelity_inputs=()
  fidelity_models=$(study_value "${plan}" projection_fidelity models)
  IFS=',' read -r -a fidelity_model_names <<< "${fidelity_models}"
  for model_name in "${fidelity_model_names[@]}"; do
    fidelity_inputs+=(
      "artifact_outputs/analysis/projection_fidelity_extractions_${model_name}.json"
    )
  done
  "${PYTHON}" scripts/build_projection_fidelity_report.py \
    --probe-yaml "${fidelity_probe}" \
    --extractions-json "${fidelity_inputs[@]}" \
    --output-json artifact_outputs/analysis/projection_fidelity_report.json

  "${PYTHON}" scripts/build_projection_diagnostics.py \
    --outputs-incidents-dir artifact_outputs/incidents \
    --ledger-json "${primary_ledger}" \
    --run-id "${run_id}" \
    --token-cap "${primary_token_cap}" \
    --output-json artifact_outputs/analysis/projection_diagnostics.json

  "${PYTHON}" scripts/build_stability_diagnostics.py \
    --outputs-incidents-dir artifact_outputs/incidents \
    --run-id "${run_id}" \
    --expected-repeats "${primary_repeats}" \
    --output-json artifact_outputs/analysis/stability_diagnostics.json

  "${PYTHON}" scripts/run_repeat_reduction_sensitivity.py \
    --bundle-json "${primary_bundle}" \
    --output-json artifact_outputs/analysis/repeat_reduction_sensitivity.json

  "${PYTHON}" -m soc_llm_policy.pygraph_report \
    --analysis-bundle-json "${primary_bundle}" \
    --outdir "${output_dir}"
  "${PYTHON}" scripts/build_execution_evidence_state.py \
    --repo-root . \
    --run-id "${run_id}" \
    --output-json "${primary_evidence}"

  "${PYTHON}" scripts/build_execution_evidence_state.py \
    --repo-root . \
    --run-id "${adversarial_run}" \
    --output-json "${adversarial_evidence}"

  "${PYTHON}" -m soc_llm_policy.campaign_evidence \
    --repo-root . \
    build \
    --plan "${plan}" \
    --primary-evidence "${primary_evidence}" \
    --adversarial-evidence "${adversarial_evidence}" \
    --provider-readiness-report "${readiness_report}" \
    --output-json "${campaign_evidence}"

  echo "Finalized run evidence: ${campaign_evidence}"
  echo "Derived figures: ${output_dir}"
}

package_run() {
  local plan=""
  local output_dir=".local/releases"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --plan)
        plan="$2"
        shift 2
        ;;
      --output-dir)
        output_dir="$2"
        shift 2
        ;;
      *)
        echo "Unknown package option: $1" >&2
        exit 2
        ;;
    esac
  done
  if [[ -z "${plan}" ]]; then
    echo "package requires --plan." >&2
    exit 2
  fi

  local run_id dataset_release_id campaign_evidence
  run_id=$(plan_value "${plan}" plan_id)
  dataset_release_id=$(plan_value "${plan}" dataset_release_id)
  campaign_evidence="artifact_outputs/evidence/${run_id}/campaign_evidence.json"
  if [[ ! -f "${campaign_evidence}" ]]; then
    echo "Finalized run evidence is missing: ${campaign_evidence}" >&2
    exit 1
  fi

  "${PYTHON}" -m soc_llm_policy.artifact_package \
    --repo-root . \
    --all \
    --heldout-dir artifact_data/heldout \
    --dataset-release-id "${dataset_release_id}" \
    --include-analysis \
    --run-id "${run_id}" \
    --campaign-evidence-json "${campaign_evidence}" \
    --output-dir "${output_dir}"
}

case "${1:-help}" in
  help|-h|--help)
    show_help
    ;;
  prepare)
    shift
    prepare_run "$@"
    ;;
  approve)
    shift
    approve_run "$@"
    ;;
  execute)
    shift
    execute_run "$@"
    ;;
  finalize)
    shift
    finalize_run "$@"
    ;;
  package)
    shift
    package_run "$@"
    ;;
  status)
    shift
    show_status "$@"
    ;;
  dashboard)
    shift
    render_dashboard "$@"
    ;;
  backup)
    shift
    "${PYTHON}" -m soc_llm_policy.execution_backup "$@"
    ;;
  *)
    echo "Unknown command: ${1}" >&2
    show_help >&2
    exit 2
    ;;
esac
