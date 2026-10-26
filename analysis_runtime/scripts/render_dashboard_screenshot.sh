#!/usr/bin/env bash
# Capture an already-rendered dashboard and write its SHA-bound preview receipt.
# Usage:
#   scripts/render_dashboard_screenshot.sh [DASHBOARD_HTML] [OUTPUT_PNG]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PACKAGE_ROOT="${REPO_ROOT}"
if [[ -f "${REPO_ROOT}/runtime_manifest.json" ]]; then
  PACKAGE_ROOT="$(dirname "${REPO_ROOT}")"
fi
DASHBOARD_HTML="${1:-${PACKAGE_ROOT}/REVIEWER_DASHBOARD.html}"
OUTPUT_PNG="${2:-${PACKAGE_ROOT}/docs/img/reviewer_dashboard.png}"

PY="${PYTHON:-${PACKAGE_ROOT}/.venv/bin/python}"
if [[ -z "${PYTHON:-}" && ! -x "${PY}" ]]; then
  PY="python3"
fi
export PYTHONPATH="${REPO_ROOT}/src"

CAPTURE_ARGS=(--dashboard-html "${DASHBOARD_HTML}" --output-png "${OUTPUT_PNG}")
CAPTURE_ARGS+=(--engine "${PREVIEW_ENGINE:-chrome-cli}")
if [[ -n "${CHROME_BIN:-}" ]]; then
  CAPTURE_ARGS+=(--browser "${CHROME_BIN}")
fi
exec "${PY}" "${REPO_ROOT}/scripts/capture_dashboard_preview.py" "${CAPTURE_ARGS[@]}"
