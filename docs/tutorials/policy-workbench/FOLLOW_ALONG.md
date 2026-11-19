# Check, repair and audit a SOCpilot policy case

This walkthrough uses the public artifact's local policy workbench. The examples
are synthetic. The workbench runs the packaged checker and repair engine; it
does not execute incident-response actions or call a language model.

## Start from the paper's artifact

Download and extract the anonymous ZIP linked in the paper. Open a terminal in
the extracted directory. Have Python 3.12 or newer available, then run:

```bash
bash reproduce.sh workbench
```

First use needs network access to obtain pinned dependencies. The launcher builds
and qualifies an isolated installation. Subsequent runs verify and reuse it.
Open the local URL printed in the terminal, normally `http://127.0.0.1:8767`.
No model credential is required. This recording was exercised on macOS arm64
with Python 3.12; it does not certify every operating system.

## Inspect the guided example

Select **Load violating example**, then **Check and repair**. The proposed
sequence reimages a host and isolates it. The declared policy requires forensic
collection before reimaging and prohibits isolation. Inspect the violations,
the rule-specific edits, and the repaired sequence. A policy verdict is not
evidence that these actions were executed successfully on a host.

Replace the proposal with `unknown_action` and check it. The visible error
identifies the unsupported action. Load the compliant example and check again
to recover. Editing input clears its previous verdict and disables stale export.

## Import a structured policy case

Select **Your structured policy case**. Import
`workbench/socpilot_playground/policy-case.json` using **Choose JSON file…**,
then select **Check policy case**. This example has its own action catalog and
rules. It requires memory capture before a service restart and approval for
account quarantine. The repair inserts capture and removes the unapproved action.
Approval entries are declarations in the input, not authenticated authorization.

Download both the input case and result JSON. Keep them outside the extracted
artifact directory. The result records the input hash, proposal, rules, edits,
repaired sequence, remaining violations and second-pass stability.

## Compare the actual GUI export with the CLI

Adjust the download paths if your browser uses another directory:

```bash
case_result="$(mktemp "${TMPDIR:-/tmp}/socpilot-case.XXXXXX")"
bash reproduce.sh workbench \
  --case-input "$HOME/Downloads/socpilot-policy-case.json" > "$case_result"
python3 workbench/compare_results.py \
  "$HOME/Downloads/socpilot-policy-result.json" "$case_result"
```

Expected result: `PASS: complete GUI and CLI JSON results are equal`. This checks
the complete object, including value types and action order. It uses the same
application service; it does not implement a second checker.

## Interpret a conflicting policy

Import `workbench/conflicting-case.json` and check it. The rules both require and
prohibit memory capture. The repaired sequence is stable, but `must_capture`
remains violated. Read remaining violations as well as stability. An unchanged
second repair does not certify compliance.

## Reproduce the paper and stop the workbench

In another terminal in the same artifact, run:

```bash
bash reproduce.sh all
```

This checks and recomputes preserved evidence, including the 4,798 primary and
160 adversarial decisions, without calling a model. It is separate from the
synthetic examples entered through the workbench. See the root README's evidence
map for inputs, reports and limits.

Stop the workbench with Ctrl+C in its terminal. For a cache qualification failure,
use `bash reproduce.sh workbench --rebuild`; prior logs remain available. Keep
exported cases and results you need before removing any installation cache.

The video uses synthetic English narration. Browser actions and terminal
commands were captured from the public code; explanatory pauses hold recorded
frames. No generated interface or invented result is shown.
