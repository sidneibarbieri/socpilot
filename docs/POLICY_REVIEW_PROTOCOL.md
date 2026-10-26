# Policy Review Protocol

The verifier can establish that a policy is typed, consistent, executable, and
reproducible. It cannot establish that the policy is appropriate for an
organization. That judgment requires domain review.

## Review unit

Review one rule at a time together with:

- its governed action and action-catalog metadata;
- scope predicates and the telemetry fields they read;
- prerequisite, approval source, and repair operator;
- derivation and transfer-support counts;
- alternatives and limitations in `policy/review_record.yaml`; and
- the specification scenario that activates the rule.

The policy graph is an inspection aid. The YAML rules and Python verifier remain
the semantic source.

## Reviewer decisions

For each rule, record one decision:

- `accept`: suitable for the stated experimental scope;
- `revise`: the objective is valid but the predicate, action, or repair is not;
- `reject`: no adequate normative or operational basis; or
- `specification_only`: useful to exercise verifier capability but unsupported
  by the operational corpus.

Each decision must include a reason and the reviewer's role. Do not collect a
name in an anonymous review package.

## Required checks

1. **Normative basis.** The rule expresses a real operating requirement, not a
   pattern selected because it improves a result.
2. **Trigger fidelity.** The scope reads fields that support the intended
   behavior class without using outcome labels.
3. **Action fit.** The governed action is proportionate to the trigger and is
   present in the catalog.
4. **Repair fit.** Insert, reorder, remove, or defer is the least disruptive
   repair that satisfies the rule.
5. **Interaction safety.** The rule does not create an ordering cycle or a
   contradictory repair when combined with other rules.
6. **Evidence tier.** Operational and specification-only rules remain separate
   in prompts, denominators, and reported results.
7. **Portability boundary.** The record states which institutional assumptions
   must be reviewed before reuse.

## Procedure

1. Generate `policy_graph.json` from the reviewed YAML bundle.
2. Inspect each rule in the dependency view and its source files.
3. Record decisions without changing the evaluation inputs.
4. Resolve disagreements before approving a new protocol manifest.
5. Compile the accepted graph and require an exact round trip.
6. Run policy preflight, all coverage scenarios, and release gates.

Changes made after provider execution begins create a new policy digest and a
new evaluation run. Review comments never modify a completed run in place.

## Evidence boundary

The current `policy/review_record.yaml` documents research-maintainer decisions
and explicitly states that the bundle is not institutionally approved. A future
independent review may strengthen normative validity, but it must be reported as
new evidence rather than inferred from the interface or verifier tests.
