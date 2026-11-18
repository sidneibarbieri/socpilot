"use strict";

const actions = document.querySelector("#actions");
const status = document.querySelector("#status");
const results = document.querySelector("#results");
const download = document.querySelector("#download");
const mode = document.querySelector("#mode");
const caseJson = document.querySelector("#case-json");
const downloadCase = document.querySelector("#download-case");
let currentResult = null;
let syntheticCase = null;

function setBusy(busy) {
  for (const control of document.querySelectorAll("form button, form textarea, form input, #mode")) {
    control.disabled = busy;
  }
}

function invalidateResult() {
  currentResult = null;
  results.hidden = true;
  download.disabled = true;
  downloadCase.disabled = true;
  status.textContent = "Proposal changed. Check it to obtain a new verdict.";
}

function showResult(result) {
  currentResult = result;
  document.querySelector("#proposed").textContent = result.proposed_actions.join(" → ") || "No proposed actions";
  document.querySelector("#before").textContent = result.violated_rules_before.join(", ") || "None";
  document.querySelector("#repaired").textContent = result.repaired_actions.join(" → ") || "No actions remain";
  document.querySelector("#after").textContent = result.violated_rules_after.join(", ") || "None";
  document.querySelector("#stable").textContent = result.second_pass_unchanged ? "Yes" : "No";
  document.querySelector("#raw").textContent = JSON.stringify(result, null, 2);
  const edits = document.querySelector("#edits");
  edits.replaceChildren();
  for (const edit of result.edits) {
    const item = document.createElement("li");
    item.textContent = `${edit.rule_id}: ${edit.repair} ${edit.missing_before || edit.action}`;
    edits.append(item);
  }
  if (!result.edits.length) {
    const item = document.createElement("li");
    item.textContent = result.plan_changed
      ? "Sequence normalization changed the plan. Compare the proposed and repaired sequences."
      : "No rule-specific repairs were reported.";
    edits.append(item);
  }
  document.querySelector("#case-reference").textContent = result.input_sha256
    ? `Case ${result.case_id} · input SHA-256 ${result.input_sha256}` : "Guided synthetic example";
  results.hidden = false;
  download.disabled = false;
  downloadCase.hidden = !result.input_case;
  downloadCase.disabled = !result.input_case;
  status.textContent = result.violated_rules_after.length
    ? "Check completed with remaining violations. Inspect the post-repair verdict."
    : "Check completed. No violations remain against the declared rules. No host actions were executed.";
}

actions.addEventListener("input", invalidateResult);
caseJson.addEventListener("input", () => {
  document.querySelector("#case-file-name").textContent = "Edited in the text area";
  invalidateResult();
});
mode.addEventListener("change", () => {
  document.querySelector("#guided-panel").hidden = mode.value !== "guided";
  document.querySelector("#case-panel").hidden = mode.value !== "case";
  invalidateResult();
});
document.querySelector("#unsafe").addEventListener("click", () => {
  actions.value = "reimage_host\nisolate_host";
  invalidateResult();
});
document.querySelector("#safe").addEventListener("click", () => {
  actions.value = "collect_forensics\nreimage_host";
  invalidateResult();
});

async function checkInput(endpoint, payload) {
  invalidateResult();
  setBusy(true);
  status.textContent = "Checking the proposed sequence…";
  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || `Check failed: HTTP ${response.status}`);
    showResult(result);
  } catch (error) {
    status.textContent = `Check failed: ${error.message}`;
  } finally {
    setBusy(false);
  }
}

document.querySelector("#proposal-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  await checkInput("/api/check", {
    actions: actions.value.split(/\r?\n/).map(value => value.trim()).filter(Boolean),
  });
});

document.querySelector("#case-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  invalidateResult();
  try {
    if (new TextEncoder().encode(caseJson.value).length > 65536) throw new Error("Case exceeds 64 KiB");
    const payload = JSON.parse(caseJson.value);
    await checkInput("/api/cases/check", payload);
  } catch (error) {
    status.textContent = `Case could not be read: ${error.message}`;
  }
});

document.querySelector("#load-case").addEventListener("click", () => {
  caseJson.value = JSON.stringify(syntheticCase, null, 2);
  document.querySelector("#case-file-name").textContent = "Built-in synthetic case";
  invalidateResult();
});

document.querySelector("#case-file").addEventListener("change", async (event) => {
  const file = event.target.files[0];
  document.querySelector("#case-file-name").textContent = file ? file.name : "No file selected";
  if (!file) return;
  invalidateResult();
  setBusy(true);
  try {
    if (file.size > 65536) throw new Error("Case exceeds 64 KiB");
    const text = await file.text();
    JSON.parse(text);
    caseJson.value = text;
    status.textContent = "Case imported. Check it to obtain a verdict.";
  } catch (error) {
    status.textContent = `Case import failed: ${error.message}`;
  } finally {
    setBusy(false);
    event.target.value = "";
  }
});

function saveJson(value, filename) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(value, null, 2)], {type: "application/json"}));
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

download.addEventListener("click", () => {
  if (currentResult !== null) saveJson(currentResult, "socpilot-policy-result.json");
});
downloadCase.addEventListener("click", () => {
  if (currentResult?.input_case) saveJson(currentResult.input_case, "socpilot-policy-case.json");
});

async function initialize() {
  try {
    const response = await fetch("/api/example");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const example = await response.json();
    actions.value = example.actions.join("\n");
    document.querySelector("#policy").textContent = JSON.stringify(example.policy, null, 2);
    for (const rule of example.policy) {
      const item = document.createElement("li");
      item.textContent = rule.type === "prohibit_before"
        ? `${rule.rule_id}: ${rule.condition_action} must precede ${rule.action}.`
        : `${rule.rule_id}: ${rule.action} is prohibited in this example.`;
      document.querySelector("#policy-summary").append(item);
    }
    const caseResponse = await fetch("/api/case-example");
    if (!caseResponse.ok) throw new Error(`HTTP ${caseResponse.status}`);
    syntheticCase = await caseResponse.json();
    caseJson.value = JSON.stringify(syntheticCase, null, 2);
    setBusy(false);
    status.textContent = "Ready. Check the proposal to see its violations and repair.";
  } catch (error) {
    status.textContent = `Initialization failed: ${error.message}`;
  }
}
initialize();
