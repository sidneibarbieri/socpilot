# Projection-fidelity probe

This disclosed synthetic probe measures whether the projection recovers the
ordered catalog action list proposed in unconstrained plan prose. Each item has a
known target trace, optional distractor actions, and one primary linguistic
phenomenon. The probe does not estimate the distribution of language in
operational plans.

`probe.yaml` is bound by hash in the experiment protocol. Model extraction runs
only as the `projection_fidelity` study in the approved `provider_run.sh`
workflow. There is no independent execution entry point because an extraction
made outside that plan would not belong to the reported evidence.

Scoring is deterministic. After the run has produced one extraction file
per declared model, rebuild the report with:

```bash
python scripts/build_projection_fidelity_report.py \
  --probe-yaml protocol_extensions/projection_fidelity_probe/probe.yaml \
  --extractions-json \
    artifact_outputs/analysis/projection_fidelity_extractions_openai_gpt55.json \
    artifact_outputs/analysis/projection_fidelity_extractions_ollama_qwen36.json \
  --output-json artifact_outputs/analysis/projection_fidelity_report.json
```

The report exposes integer counts before rates and breaks precision, recall,
exact match, and distractor rejection down by phenomenon. Adding or changing a
probe item creates a new protocol digest and therefore requires a new run.
