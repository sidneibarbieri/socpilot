#!/usr/bin/env bash
# Regenerate the reviewer-dashboard preview image used in the README.
#
# The dashboard HTML is the reproducible artifact; this PNG is only a preview
# for the README and the repository landing page. It is produced by rendering
# the dashboard from the current public analysis outputs and screenshotting the
# self-contained page with a headless browser. No provider calls are made.
#
# Usage:
#   scripts/render_dashboard_screenshot.sh [ANALYSIS_DIR] [CONSTRAINTS] [OUTPUT_PNG]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ANALYSIS_DIR="${1:-${REPO_ROOT}/results/analysis}"
CONSTRAINTS="${2:-${REPO_ROOT}/policy/constraints.yaml}"
OUTPUT_PNG="${3:-${REPO_ROOT}/docs/img/reviewer_dashboard.png}"

PY="${REPO_ROOT}/.venv/bin/python"
[ -x "${PY}" ] || PY="python3"

WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
DASHBOARD_HTML="${WORK}/reviewer_dashboard.html"

"${PY}" "${REPO_ROOT}/scripts/render_audit_dashboard.py" \
  --analysis-dir "${ANALYSIS_DIR}" \
  --constraints "${CONSTRAINTS}" \
  --output-html "${WORK}/dashboard.html" \
  --landing-html "${DASHBOARD_HTML}"

CHROME="${CHROME_BIN:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
if [ ! -x "${CHROME}" ]; then
  CHROME="$(command -v chromium || command -v google-chrome || true)"
fi
if [ -z "${CHROME}" ]; then
  echo "No Chrome/Chromium found; set CHROME_BIN to a headless-capable browser." >&2
  exit 1
fi

mkdir -p "$(dirname "${OUTPUT_PNG}")"
# The window height must clear the full page (all studies) so the preview is not
# truncated; the headless screenshot captures up to the window height.
"${CHROME}" --headless --disable-gpu --hide-scrollbars \
  --force-device-scale-factor=2 --window-size=1280,2900 \
  --screenshot="${OUTPUT_PNG}" "file://${DASHBOARD_HTML}"

echo "Reviewer dashboard preview written to: ${OUTPUT_PNG}"
