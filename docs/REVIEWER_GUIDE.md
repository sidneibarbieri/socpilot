# Reviewer Guide

This is an evaluation tool for incident-response plans. It checks proposed action
lists against explicit policy rules; it does not execute incident-response actions.
The source package and a completed-results package serve different purposes.
Check `artifact_manifest.json` and `CLAIM_EVIDENCE_MATRIX.md` before interpreting
any output as experimental evidence.

## Environment and Resources

- Python 3.12, Bash, and a local filesystem are required. Exact Python packages
  are listed in `requirements.txt` and `requirements-dev.txt`.
- Offline commands have been exercised on macOS with Python 3.12.14. The tests
  also use a loopback TCP listener and Unix file locks. Windows is not a verified
  execution platform. No administrator access is required.
- Corpus checks, rule coverage, plan preparation, and the HTML dashboard need
  neither a GPU nor model credentials. Allow several minutes after dependency
  installation; the full model execution is a separate workload.
- The declared model execution uses a remote OpenAI profile and local Ollama
  profiles. The local profiles require approximately 39 GB of installed model
  weights in total, plus inference working memory. The intended machine has
  64 GB of unified memory. This is a reference configuration, not a measured
  minimum. Exact names, digests, and settings are in `config/models.freeze.yaml`.
- Reserve persistent disk space for requests, responses, and attempt records.
  Output size and total run time have not yet been measured for this protocol.
  Do not interpret an unmeasured duration as a service-level guarantee.

Dependency installation and model downloads require network access. Subsequent
offline checks use packaged inputs. Model execution requires a reachable local
Ollama service and Internet connectivity for the remote profile. A successful
metadata query does not establish account balance or generation availability.

## Install and Test Without Model Calls

After installing the dependencies below, start with a small library example:

```bash
PYTHONPATH=src .venv/bin/python -B -m soc_llm_policy.policy_example
```

The synthetic proposal contains reimaging and isolation. Two example rules require
evidence collection before reimaging and prohibit isolation in this example.
The JSON shows the original violations, two edits, the repaired sequence
`[collect_forensics, reimage_host]`, and a separate post-repair check. Running the
checker again leaves the repaired sequence unchanged. The command writes no files
and calls no models. These invented rules teach the interface; they are not the
research policy or evidence of operational effectiveness.

Run from the extracted package root in a persistent directory:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-dev.txt
bash run.sh validate-public-artifact
bash run.sh policy-graph-validate
bash run.sh rule-coverage
PYTHONPATH=src .venv/bin/python -m pytest -q
bash provider_run.sh prepare
PLAN=artifact_outputs/provider_state/canonical_campaign/provider_execution_plan.json
bash provider_run.sh execute --plan "$PLAN" --validate-only
bash provider_run.sh status --plan "$PLAN"
bash provider_run.sh dashboard --plan "$PLAN"
```

Expected observations:

| Step | Evidence to inspect | Interpretation |
| --- | --- | --- |
| Artifact validation | Six gates complete without an exception | Packaged files, corpus checks, and release-boundary checks agree |
| Policy graph | Validation exits with code 0 | The graph agrees with typed policy rules; this is not evidence of operational policy validity |
| Rule coverage | `results/analysis/rule_coverage_report.json` | Every declared rule is exercised by a specification scenario; scenario frequencies are not incident frequencies |
| Tests | Test summary with no failures | The packaged regression cases pass; fixtures are not model measurements |
| Prepare | The plan at `$PLAN` | The declared matrix and budget estimate are materialized without calls |
| Validate only | Binding checks exit with code 0 | Runtime inputs match the plan; live generation is still untested |
| Status | JSON with `approval_status: draft`, absent ledgers and no readiness report | No recorded execution, not a zero-violation result |
| Dashboard | `artifact_outputs/provider_state/canonical_campaign/execution_dashboard.html` | A read-only snapshot of the same operational records |

Open the HTML file in a browser. The study filter should update the visible rows
and announce an empty result when no study matches. All values remain available
with JavaScript disabled; only filtering needs JavaScript. Narrow screens use
horizontal scrolling within tables. There is no web server, remote font, analytics
script, or dashboard dependency to install.

## Operator Journey

Inspect the fixed plan, model requirements, cost estimate, and attempt allowance.
An authorized operator records a numeric ceiling using `approve --budget-usd`.
Approval is not a credit-balance check. The first normal `execute` performs bounded
live readiness checks before creating scientific experiment state. These requests
can incur cost. Do not use `execute` as an offline smoke test.

The HTML page is a snapshot, not a live worker-health monitor. Regenerate it after
checking CLI status. A recorded running state can outlive a stopped worker; check
the execution log and lease before considering recovery. Never infer that a
missing attempt log means zero cost. Estimated spend is not a provider invoice.

Resume only through `execute --resume` and the same approved plan. The runner
retains the original attempt allowance. Refusals and complete responses that fail
parsing remain observations. Another attempt is permitted only under the fixed
no-observation transport policy, not because a result is inconvenient. Do not edit
ledgers, replace cells, or increase the allowance while a run is in progress.

## Reproduce Claims

For a data-only audit, inspect `artifact_data/global/constraints.yaml`,
`action_catalog.yaml`, `telemetry_normalization.yaml`, and `review_record.yaml`.
These describe rules, action identifiers, normalization, and per-rule provenance.
Normalized incidents contain `incident_meta.json`, `incident_telemetry.jsonl`, and
`incident_human_actions.jsonl`. Their presence does not make recorded analyst tasks
a ground-truth judgment of response quality.

`CLAIM_EVIDENCE_MATRIX.md` maps each current claim to its audit command and evidence
paths. Corpus and policy checks are available in this source package. Empirical
model-result claims remain pending until all required studies pass finalization.

A completed-results package adds preserved model outputs and derived reports.
Its `bash run.sh reproduce-results` command rechecks those records without model
access. A new model execution is a new stochastic measurement, not byte-identical
reproduction. Do not substitute its results silently for the published evidence.
Expected primary-result tables, figures, durations, and tolerances must accompany
that completed package; they cannot be supplied as measured facts before the run.

The deterministic builders expose their inputs through `--help`:

```bash
.venv/bin/python scripts/build_separation_report.py --help
.venv/bin/python scripts/build_adversarial_report.py --help
.venv/bin/python scripts/build_projection_fidelity_report.py --help
.venv/bin/python scripts/build_projection_diagnostics.py --help
.venv/bin/python scripts/build_stability_diagnostics.py --help
```

The final-result audit dashboard and this source package's execution-status
snapshot are different views: one explains audited findings; the other records
operational progress. Neither changes the published observations.

## Security and Preservation

Model execution transmits the rendered incident context to each declared model
endpoint. Review the redacted corpus and the applicable data-sharing permissions
before providing credentials. Local Ollama profiles are local only when their
configured endpoint is local; a cloud alias is not a local model.

Keep credentials in the local environment or an untracked `.env` with restricted
file permissions. Never include it in an archive, screenshot, issue, or log. The
dashboard contains identifiers and counts, not raw prompts or response bodies.
Treat any generated HTML as local operational information and review it before
sharing. No public listener is needed to read it.

Preserve the downloaded archive, approved plan, complete `artifact_outputs/`
tree, and execution logs. Use a separate device or approved encrypted backup
destination for device-loss protection. A second directory on the same disk does
not provide that protection. Verify backup hashes and retain invalid observations.
Do not clean caches or delete outputs while workers or checks are using them.
Use the [backup and recovery commands](EXECUTION_BACKUP.md) to create a quiescent
snapshot and rehearse restoration without model calls. Backup verification is
distinct from final scientific evidence validation.

## Licensing

Source code is available under the [MIT license](../LICENSE). The anonymized
incident data may be used and redistributed for research and reproducibility
under [the data terms](../DATA_LICENSE.md). Preserve anonymization and do not
attempt reidentification. The purpose-restricted data terms are distinct from
the software license. Model weights and dependencies retain their own terms.
