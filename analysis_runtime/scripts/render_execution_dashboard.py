#!/usr/bin/env python3
"""Render a read-only, offline snapshot from the same records as CLI status."""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path

from soc_llm_policy.durable_io import atomic_write_bytes
from soc_llm_policy.provider_status import (
    ExecutionStatus,
    StudyStatus,
    read_execution_status,
)

_STYLES = """
:root {color-scheme:light; --ink:#233543; --muted:#506270; --line:#cad2d6;
  --paper:#faf9f5; --accent:#355c73; --notice:#765421;}
* {box-sizing:border-box} body {margin:0;background:var(--paper);color:var(--ink);
  font:16px/1.55 'Helvetica Neue',Helvetica,sans-serif}
main {max-width:1160px;margin:auto;padding:32px 24px} a {color:var(--accent)}
h1,h2 {font-family:Charter,'Bitstream Charter',Georgia,serif;line-height:1.2}
h1 {font-size:32px;margin:8px 0} h2 {font-size:23px;margin-top:32px}
p {max-width:88ch} .muted {color:var(--muted)} nav {display:flex;gap:24px;flex-wrap:wrap}
code {font:13px/1.6 Menlo,Consolas,monospace;overflow-wrap:anywhere}
.notice {border-left:4px solid var(--notice);padding:10px 16px;background:#f3eee2}
dl {display:grid;grid-template-columns:200px 1fr;gap:8px 16px}
dt {font-weight:600} dd {margin:0;overflow-wrap:anywhere}
.scroll {overflow-x:auto} table {width:100%;border-collapse:collapse;font-size:14px}
caption {text-align:left;padding:10px 0;color:var(--muted)}
th,td {text-align:left;padding:11px 12px;border-bottom:1px solid var(--line);vertical-align:top}
th {font-weight:600;background:#eaf0f2} .number {text-align:right;font-variant-numeric:tabular-nums}
small {display:block;color:var(--muted)} label {font-weight:600;margin-right:12px}
select {font:inherit;padding:6px;border:1px solid var(--muted);background:white;color:var(--ink)}
:focus-visible {outline:3px solid var(--accent);outline-offset:3px}
@media(max-width:600px) {main {padding:20px 16px} h1 {font-size:27px}
  dl {grid-template-columns:1fr;gap:3px} dd {margin-bottom:12px}}
"""


def _value(value: int | float | None) -> str:
    if value is None:
        return "Not recorded"
    return f"{value:.6f}" if isinstance(value, float) else f"{value:,}"


def _study_row(name: str, study: StudyStatus) -> str:
    labels = {
        "no_ledger": "No ledger", "recorded": "Ledger present",
        "failure_recorded": "Failure recorded", "terminal_pending_audit": "Awaiting final audit",
    }
    counts = study.status_counts
    cells = "".join(
        f'<td class="number">{_value(counts.get(status, 0) if counts is not None else None)}</td>'
        for status in ("completed", "refused", "parse_failed", "failed", "running", "planned")
    )
    return (
        f'<tr data-state="{study.state}"><th scope="row">{escape(name.replace("_", " ").title())}'
        f'<small>{escape(labels[study.state])}</small></th>'
        f'<td class="number">{_value(study.expected_task_ceiling)}</td>'
        f'<td class="number">{_value(study.materialized_tasks)}</td>{cells}</tr>'
    )


def render_html(report: ExecutionStatus) -> str:
    rows = "".join(_study_row(name, study) for name, study in report.studies.items())
    costs = {"Readiness": report.readiness_cost, **{
        name.replace("_", " ").title(): study.provider_cost
        for name, study in report.studies.items()
    }}
    cost_rows = "".join(
        f'<tr><th scope="row">{escape(name)}</th>'
        + "".join(f'<td class="number">{_value(value)}</td>' for value in (
            cost.accrued_usd, cost.ambiguous_reserved_usd, cost.unknown_cost_count,
        )) + "</tr>"
        for name, cost in costs.items()
    )
    timestamps = "".join(
        f'<dt>{escape(name.replace("_", " ").title())}</dt>'
        f'<dd>{escape(study.ledger_updated_at_utc or "Not recorded")}; '
        f'unavailable diagnostic sources: {_value(study.unavailable_sources)}</dd>'
        for name, study in report.studies.items()
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Execution status | SOCpilot</title><style>{_STYLES}</style></head>
<body><main><nav aria-label="Page sections"><a href="#progress">Progress</a>
<a href="#cost">Cost records</a><a href="#provenance">Provenance</a></nav>
<h1>Execution status</h1>
<p class="muted">Read-only snapshot. Generated <time>{escape(report.generated_at_utc)}</time>.</p>
<p class="notice">This is not a live monitor or a scientific result. A recorded running state does not
prove that its worker is alive. Regenerate this page to inspect later progress. Missing records
are not zero. Terminal observations still require the final evidence audit.</p>
<dl><dt>Plan</dt><dd><code>{escape(report.plan_id)}</code></dd>
<dt>Approval</dt><dd>{escape(report.approval_status.title())}</dd>
<dt>Authorized ceiling</dt><dd>USD {_value(report.approved_budget_usd)}</dd>
<dt>Live readiness</dt><dd>{'Passed; bound records verified' if report.readiness_state == 'passed' else 'No report; generation not verified'}</dd>
<dt>Scientific task ceiling</dt><dd>{report.scientific_task_ceiling:,}</dd>
<dt>Attempts per task</dt><dd>{report.provider_transport_attempt_limit}, including the first request</dd></dl>
<section id="progress"><h2>Study progress</h2>
<label for="study-filter">Show studies</label><select id="study-filter">
<option value="all">All studies</option><option value="failure_recorded">Failures recorded</option>
<option value="no_ledger">No ledger</option><option value="terminal_pending_audit">Awaiting final audit</option></select>
<p id="filter-summary" role="status">{len(report.studies)} studies shown</p>
<div class="scroll" tabindex="0" role="region" aria-label="Study status table"><table id="studies">
<caption>Materialized tasks can be below a diagnostic ceiling when source observations are unavailable.
Completed, refused, and invalid-output observations are retained, not retried to obtain better answers.</caption>
<thead><tr><th scope="col">Study</th><th scope="col">Ceiling</th><th scope="col">Materialized</th>
<th scope="col">Completed</th><th scope="col">Refused</th><th scope="col">Invalid output</th>
<th scope="col">Failed</th><th scope="col">Running record</th><th scope="col">Planned</th></tr></thead><tbody>{rows}</tbody></table></div></section>
<section id="cost"><h2>Provider cost records</h2><p>USD estimates from attempt records, not an account balance or invoice.
Unknown cost entries require investigation. Missing logs cannot establish zero spend.</p>
<div class="scroll" tabindex="0" role="region" aria-label="Cost table"><table><thead><tr>
<th scope="col">Study</th><th scope="col">Accrued estimate (USD)</th>
<th scope="col">Unsettled reservation (USD)</th><th scope="col">Unknown entries</th></tr></thead>
<tbody>{cost_rows}</tbody></table></div></section>
<section id="provenance"><h2>Source and freshness</h2><p>Protocol SHA-256:<br><code>{escape(report.protocol_sha256)}</code></p>
<p>Ledger file modification times below describe local records, not verified worker heartbeats.
This snapshot checks identities and counts; finalization checks the complete evidence chain.</p><dl>{timestamps}</dl>
<p>Regenerate with <code>bash provider_run.sh dashboard --plan PATH</code>.
For the same state as JSON, use <code>bash provider_run.sh status --plan PATH</code>.</p></section>
</main><script>
const filter = document.getElementById('study-filter');
filter.addEventListener('change', () => {{
  let visible = 0;
  document.querySelectorAll('#studies tbody tr').forEach(row => {{
    row.hidden = filter.value !== 'all' && row.dataset.state !== filter.value;
    if (!row.hidden) visible += 1;
  }});
  document.getElementById('filter-summary').textContent = visible ? `${{visible}} studies shown` : 'No studies match this filter';
}});
</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--plan-json", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    args = parser.parse_args()
    report = read_execution_status(args.repo_root, args.plan_json)
    atomic_write_bytes(args.output_html, render_html(report).encode("utf-8"))
    print(f"Read-only snapshot: {args.output_html}")


if __name__ == "__main__":
    main()
