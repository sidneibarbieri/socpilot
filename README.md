# Artifact Package

This package is the executable SOCpilot research artifact. It contains the
400-incident anonymized corpus, acquisition-batch provenance, declared policy inputs, verifier, experiment
scripts, and non-private evidence needed to validate the reported results.

Incident identifiers are opaque aliases (`INC_PUBLIC_...`) and every incident's
event times are shifted into the past by a random offset. The shift preserves the
order and spacing of events within an incident; calendar dates and the timing
between unrelated incidents carry no information. Verdicts do not read times, so
the recorded decision digests are unchanged.

The package is independent of any publication workspace or event template. It
does not include literature collections, private notes, raw exports, local
provider payloads, credentials, or publication-workspace automation.

## Environment setup
Python 3.12 or newer is required. Check `python3 --version` before creating a
virtual environment inside this package. If it reports an older version, use a
Python 3.12+ interpreter for the command below. The bundled
`reproduce.sh` selects the offline analysis runtime and auto-detects `.venv/`
when present.

```bash
python3 -m venv .venv
source .venv/bin/activate
.venv/bin/pip install -r requirements.txt
```

`requirements-dev.txt` is optional and intended for maintainers working from the
full source repository. Use `reproduce.sh` for the zero-cost checks below: it
selects the bundled `analysis_runtime`. The separate `run.sh` retains the frozen
collection source for provenance and is not the offline replay entry point.

## Recommended verification order
Open `REVIEWER_DASHBOARD.html` first when reviewing a downloaded copy of this
package. It is a static, self-contained entry point for the public evidence
surface: corpus readiness, model-execution accounting, policy coverage, paired
contrasts, replay stability, and direct links to the files behind the reported
claims.

Regenerate it with:

```bash
bash reproduce.sh reproduce-results
```

This zero-cost command validates the public artifact, recomputes corpus
partition, transfer activation, and normalization-sensitivity evidence from the
released incident packages, regenerates evidence summaries, writes a
reproduction report, audits the policy-evidence boundary, and renders both
`REVIEWER_DASHBOARD.html` and
`artifact_outputs/dashboard/index.html`. To run only the integrity and dataset
checks, use `bash reproduce.sh validate-public-artifact`.

The reviewed YAML policy is the semantic source. Its graph is a deterministic
projection for inspection, not a second policy definition. Verify the graph and
typed rules agree with:

```bash
bash reproduce.sh policy-graph-validate
```

`docs/POLICY_REVIEW_PROTOCOL.md` separates executable policy checks from the
domain judgment required to accept, revise, or reject a rule.

The validation path runs artifact integrity checks, dataset audit, global-artifact
assessment, and release-hygiene rechecks and writes fresh verification reports
under `artifact_outputs/analysis/`.
When `official_llm_analysis_bundle.json` is present, the command also rechecks
that the sanitized official row-level metrics agree with the shipped official
summary, run manifest, and paired-test rows.
Local transient files such as `.venv/` and `__pycache__/` are ignored by
the release-surface checks for the packaged public path.
It also verifies that `artifact_outputs/analysis/protocol_freeze.json` points
to packaged files whose SHA-256 digests match the published bundle.
The terminal model-execution evidence includes the anonymized per-task model inputs,
model outputs, verifier records, worker-execution histories, and provider
transport-attempt metadata needed to recount the results. It excludes
credentials, authorization headers, private raw exports, and private incident
lineage. `artifact_outputs/evidence/` contains the sole terminal evidence
manifest; convenience reports in `artifact_outputs/analysis/` are derived from
evidence bound by that manifest. The released package does not mutate this
evidence. A study with another model, policy, or corpus must use a separate copy
of the source code and a new protocol manifest.

Readers should interpret the rule set across two evidence tiers. On the real
corpus, the evaluation uses only the 15 rules marked `operational`. Four
approval rules are exercised only by fixed specification-coverage scenarios
because the exports contain no authenticated approval record. Model behavior is
reported for the complete 400-incident corpus and with both export-batch
sensitivity estimates.

## Package layout
- `artifact_data/dataset/`: first export batch; policy-derivation source
- `artifact_data/heldout/`: second acquisition batch; policy-transfer check
- `artifact_data/global/`: policy and normalization inputs used by the run
- `artifact_outputs/analysis/`: non-private analysis outputs and the official protocol manifest
- `src/`, `scripts/`, `config/`, `local_redaction/`: reproducibility and data-preparation code

Use `README.md` for the package entry point and `EVAL_PROTOCOL.md` for the
evaluation setup. The model registry used by the reported run is recorded in
`config/models.freeze.yaml`; the filename is retained for compatibility. Local
execution overrides are intentionally excluded.

## Collection source and offline reanalysis

The root `src/`, `scripts/`, `run.sh`, and `provider_run.sh` retain the manifest-bound collection implementation. Use `bash reproduce.sh` for offline review. It selects the separately hashed `analysis_runtime/` maintenance code; this code did not collect the model responses. Its dependency declarations are in `analysis_runtime/requirements.txt` and `analysis_runtime/requirements-dev.txt`. The package manifest binds both implementations. No collection inputs or observations are replaced.

## Offline decision replay

Recompute the preserved decisions without calling a provider. Expected counts cover preserved decisions, not planned cells. Missing outputs remain in failure accounting rather than being replaced. The adversarial study uses its own frozen scenario inputs, not the primary corpus. These checks do not certify publication safety or operational effects.

```bash
bash reproduce.sh offline-verdict-audit --run-id measurement_e6889d081957361f --expected-cells 4798
bash reproduce.sh offline-verdict-audit --run-id measurement_e6889d081957361f__adversarial --expected-cells 160 --incident-input-root artifact_outputs/experiments/measurement_e6889d081957361f__adversarial/scenario_input/dataset
```

## Repair stability and residual conflicts

Stable does not mean compliant. The offline audit separately counts post-repair violations and `active_requirement_prohibition_conflict_cells`: contexts where active rules both require and prohibit the same action. This count screens direct conflicts, not general policy satisfiability. Such conflicts require policy review; retrying the same repair does not resolve them. Historical decisions and failures remain part of the preserved evidence.

## Dashboard preview maintenance

Opening the shipped dashboard requires no browser automation. To regenerate its documentation preview, provide an installed headless-capable Chrome/Chromium and the reviewer Python environment:

```bash
PYTHON=.venv/bin/python CHROME_BIN=/path/to/chromium bash analysis_runtime/scripts/render_dashboard_screenshot.sh REVIEWER_DASHBOARD.html /path/to/preview.png
```

Write outside the verified package. The adjacent `.preview.json` binds the HTML and PNG hashes; a preview is not experiment evidence. The maintained verifier checks the shipped preview binding as well as manifest integrity.

## Explicit preview backend selection

The default preview engine is `chrome-cli`. If the installed browser cannot finish that capture, select the optional Playwright backend explicitly. It requires Node.js and the `playwright` module available to Node; use `NODE_PATH` for an existing installation outside this package. This command uses the browser you supply, downloads no browser, makes no provider calls, and does not silently switch engines after a failure:

```bash
PREVIEW_ENGINE=playwright PYTHON=.venv/bin/python CHROME_BIN=/path/to/chromium bash analysis_runtime/scripts/render_dashboard_screenshot.sh REVIEWER_DASHBOARD.html /path/to/preview.png
```

Write the preview outside this verified package. A missing dependency, browser error or timeout fails the capture without a success receipt. Opening the existing HTML needs neither Node.js nor Playwright.
