# Architecture and Extension Guide

The system separates plan generation, deterministic policy checking, execution
accounting, and result analysis. The paper is a consumer of the resulting evidence,
not an output of the runtime.

## Components

| Responsibility | Code and data | Boundary |
| --- | --- | --- |
| Typed data and policies | `src/soc_llm_policy/interfaces.py`, `policy_ir.py`, `artifact_data/global/` | Validated catalog and rule inputs |
| Model access | `llm_adapter.py`, `models_registry.py`, `config/models.freeze.yaml` | Requests, complete responses, and typed transport failures |
| Policy checking and repair | `engine.py`, `policy_predicates.py`, `verify.py` | Action lists and trusted governance inputs; no model calls |
| Fixed evaluation design | `provider_plan.py`, `protocol_freeze.py`, `evaluation_plan.py` | Matrix identity, approval, worker ownership, and durable cell states |
| Recovery and accounting | `connectivity_recovery.py`, `connectivity_evidence.py`, `provider_cost.py` | Bounded recovery records and monetary exposure, not replacement observations |
| Operational display | `provider_status.py`, `scripts/render_execution_dashboard.py` | One typed read-only view shared by CLI and HTML |
| Analysis and audit | `analyze.py`, `execution_evidence.py`, `artifact_verify.py` | Derived reports with traceable primary records |

Module paths without a directory prefix are relative to `src/soc_llm_policy/`.
Inputs belong in `artifact_data/` and `config/`; generated execution records belong
in `artifact_outputs/`. The packaged manifest distinguishes distributed source
from local runtime state. Do not add administrative review notes to the artifact.

## Experiment Vocabulary and Compatibility

An **incident** is an input case. A **proposal** is an ordered action list to be
checked, not an observed execution. A **policy** states the supported constraints.
An **experiment protocol** fixes inputs, models, conditions, repetitions, metrics,
and failure handling. A **run** executes that protocol. A **cell** identifies one
planned observation; a transport **attempt** is not an additional repetition.

Use these terms in new documentation and interfaces. Some serialized identifiers
retain `campaign`, including `config/campaign.yaml`, `canonical_campaign`, and
`corpus_campaign_manifest`. Here they identify an experiment and its corpus, not
an adversary campaign, attack replay, or response workflow. Existing manifests
bind these names. Renaming them requires a tested format migration and a new
source identity; never rename files inside preserved evidence.

In the maintenance checkout, the entry point is
`python -m scripts.build_experiment_source`.
It delegates to the existing `scripts.build_campaign_source` implementation;
both entry points retain identical arguments and behavior. There is no second
experiment configuration.
Both source builders are excluded from the public package because they require
private ingestion inputs. In a downloaded package, use `bash provider_run.sh
prepare` to prepare the experiment from the supplied anonymized inputs.

The policy-checking library accepts caller-provided typed rules and action lists.
The published experiment builder is intentionally study-specific: it enforces
400 incidents and 4,800 primary cells. Reusing the library is distinct from
changing that study. Neither interface establishes multi-tenant scalability,
production incident-response effectiveness, or validity in another organization;
those require separate measurements and operational validation.

## Contributing Changes

Work in a separate branch or checkout. Do not edit a source directory being used
by a model execution. A patch should state the problem, changed behavior, regression
test, compatibility implications, and whether the scientific protocol changes.
The public package contains a portable test subset; passing it does not replace
the maintainer suite and a fresh-package rehearsal.

Use small functions with explicit inputs and representative names. Validate
external data with the existing typed models. Handle only errors for which there
is a documented recovery action; propagate unexpected failures with their cause.
Missing observations are not zeros. Comments should explain non-obvious decisions,
not restate the code. Avoid adding frameworks for a single local operation.

## Extension Boundaries

`policy_example.py` provides a runnable starting point. Construct an
`EnforcementRequest` from typed catalog entries, rules, context, and proposed
actions. `evaluate_cell_families` checks a proposal without repairing it;
`enforce_policy` returns the edits and repaired list. Check the repaired list
separately rather than assuming every combination of rules can be satisfied.
Change an example in a separate checkout, not in an active execution package.

- A new model profile needs exact endpoint/model identity, decoding settings,
  response validation, transport tests, cost accounting, and a new fixed plan.
  A local profile is not interchangeable with a remote one during a run.
- A new policy family needs a typed schema, predicate semantics, conflict rules,
  repair behavior, and positive and negative tests. Show how it interacts with
  existing families. Passing graph validation is not human approval of a policy.
- A new corpus needs documented acquisition, redaction permissions, disjoint
  partitions, data-quality checks, and input hashes. Do not infer missing analyst
  actions or approvals from incident outcomes.
- A new metric needs its population, denominator, missingness treatment, unit of
  analysis, and independent recount test specified before collection. Changing
  analysis based on partial results requires a separately identified study.

The graph represents typed policy entities and relationships. It is not an
observed execution workflow. A future visual editor must preserve the typed policy
semantics and pass round-trip and conflict tests; drawing an edge is not evidence
that a policy is correct or approved for operational use.

## Review Checklist

1. Run the packaged tests and the zero-cost commands in `REVIEWER_GUIDE.md`.
2. Test absent, malformed, and conflicting inputs, not only successful examples.
3. Confirm that CLI and dashboard derive counts from the same records.
4. Confirm that preparation and monitoring make no model calls or scientific edits.
5. Rebuild and validate a separate package after a source change. Preserve prior
   source identities and completed evidence; never overwrite a submitted artifact.
6. Record the evidence for each new claim before changing the manuscript.

Licensing and data permissions are described in `REVIEWER_GUIDE.md`; technical
extension guidance does not grant reuse rights.
