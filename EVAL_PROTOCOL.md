# Evaluation Protocol

## Objective
Evaluate whether LLM-generated incident-response plans violate declared SOC
policy rules, where those violations concentrate, and how deterministic
enforcement changes the resulting plans before analyst review.

## Reported configuration
- Evaluation incident count: `400`
- Export Batch 1: `200`
- Export Batch 2: `200`
- Official models: `azure_gpt54, ollama_qwen36`
- Prompt conditions: `llm_zero, llm_policy_prompt`
- Operational policy tier: `operational`

## Primary outputs
- `artifact_outputs/analysis/protocol_freeze.json`
- `artifact_outputs/analysis/summary.json`
- `artifact_outputs/analysis/analysis_bundle.json`
- `artifact_outputs/analysis/official_llm_analysis_bundle.json`
- `artifact_outputs/analysis/official_evaluation_summary.json`
- `artifact_outputs/analysis/official_pairwise_tests.json`
- `artifact_outputs/analysis/global_artifact_assessment.json`
- `artifact_outputs/analysis/global_artifact_provenance.json`
- `artifact_outputs/analysis/repeat_stability/` (when present)

## Public-bundle scope
- The package includes one terminal evidence manifest, one protocol manifest, the
  complete per-task evidence it binds, and deterministic analysis projections.
- Worker executions and provider transport attempts are operational records;
  neither changes the three scientific repeats per model condition.
- The zero-cost path validates all evidence hashes, recounts aggregates, and
  regenerates deterministic reports without contacting a provider.
- Credentials, authorization headers, private raw exports, and private incident
  lineage are excluded.
- Reviewers may supply their own provider keys and run new experiments with the
  same artifact or with their own incident packages. Such runs are new execution
  lineages unless the incident slice, policy files, model registry, prompt
  template, and verifier manifest all match the reported state.

## Rule-activation interpretation
- The real-corpus evaluation uses the 15 rules marked `operational`.
- Four approval rules are specification-only because the exports contain no authenticated approval record; fixed coverage scenarios test their implementation semantics without contributing to corpus rates.
- Rules whose behavior classes recur weakly in the transfer-check batch remain local operational rules and are also exercised by a fixed specification-coverage set; they do not support cross-batch transfer claims.

## Inspectable prompt-condition surfaces
- `src/soc_llm_policy/pipeline.py`: binds `llm_zero` to `policy_prompt_mode=none` and `llm_policy_prompt` to `policy_prompt_mode=inline_constraints`.
- `src/soc_llm_policy/llm_adapter.py`: defines the shared prompt template, requested JSON format, and inline policy-constraint block.
- `config/models.freeze.yaml`: records the model registry used in the study; the filename is retained for compatibility.

## Planned statistical contrasts
- Three repeats for each incident-model-condition group reduce by majority verdict.
- A paired contrast uses only groups in which all three repeats produced a plan; refusals and parse failures remain in terminal accounting.
- Within each analysis stratum, two contrasts compare prompt conditions within a model and two compare models within a prompt condition.
- No contrast changes model and prompt condition at the same time.
- Holm adjustment covers the complete planned family across the full corpus and both batch-specific sensitivity strata.
- First-repeat, last-repeat, and any-violation reductions are sensitivity checks; they do not replace the majority estimate.

## Reproducibility requirements
- Preserve the packaged anonymized dataset under `artifact_data/`
- Treat `artifact_outputs/analysis/protocol_freeze.json` as the official protocol manifest
- Re-run verification and dataset audit from the package root to confirm integrity before any additional analysis
