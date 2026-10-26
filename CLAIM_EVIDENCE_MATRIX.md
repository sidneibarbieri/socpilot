# Claim-to-Evidence Matrix

This table states what the package can support at its current stage.
A pending claim must not be reported as a result.

| Claim | Status | Audit |
| --- | --- | --- |
| The evaluation corpus contains 400 incidents in two disjoint export batches. | `supported` | `bash run.sh validate-public-artifact` |
| The reviewed bundle separates operational rules from specification-only approval gates and records the transfer check. | `supported` | `bash run.sh validate-public-artifact` |
| The shipped verifier implements deterministic typed checks and repairs over ordered action lists. | `supported` | `bash run.sh validate-public-artifact` |
| The reported model comparisons, violation rates, judge results, adversarial results, and projection-fidelity results follow from one terminal model execution. | `supported` | `bash run.sh reproduce-results` |

The machine-readable matrix and evidence paths are in
`artifact_outputs/analysis/claim_evidence_matrix.json`.
