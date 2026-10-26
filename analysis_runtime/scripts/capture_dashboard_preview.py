"""Capture a verified local dashboard without regenerating its evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

from soc_llm_policy.dashboard_preview import PreviewReceipt

_MAX_HEIGHT = 16000
_VIEWPORT_WIDTH = 1280
_PNG_HEADER_SIZE = 24

def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _browser_path(explicit: str | None) -> str:
    candidates = [explicit] if explicit else [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        shutil.which("chromium"),
        shutil.which("google-chrome"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    raise FileNotFoundError("Chrome/Chromium unavailable; supply --browser.")


def _page_height(rendered_dom: str) -> int:
    match = re.search(r'data-preview-height="([0-9]+)"', rendered_dom)
    if match is None:
        raise ValueError("Browser did not report dashboard height.")
    height = int(match.group(1))
    if not 1 <= height <= _MAX_HEIGHT:
        raise ValueError("Dashboard height is outside the supported preview range.")
    return height + 160


def _png_dimensions(path: Path) -> tuple[int, int]:
    header = path.read_bytes()[:_PNG_HEADER_SIZE]
    if len(header) != _PNG_HEADER_SIZE or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError("Browser did not produce a PNG image.")
    return struct.unpack(">II", header[16:24])


def _playwright_capture(executable: str, dashboard: Path, screenshot: Path) -> int:
    node = shutil.which("node")
    if node is None:
        raise FileNotFoundError("Playwright capture requires Node.js and the playwright module.")
    helper = Path(__file__).with_suffix(".cjs")
    result = subprocess.run(
        [node, str(helper), executable, dashboard.as_uri(), str(screenshot)],
        capture_output=True, text=True, check=True, timeout=60,
    )
    dimensions = json.loads(result.stdout)
    return int(dimensions["height"])


def capture_preview(
    dashboard: Path, output: Path, browser: str | None = None,
    engine: str = "chrome-cli",
) -> Path:
    """Capture local HTML and bind the preview to its unchanged source bytes."""
    if engine not in ("chrome-cli", "playwright"):
        raise ValueError("Unknown capture engine.")
    dashboard = dashboard.resolve(strict=True)
    output = output.resolve()
    receipt_path = output.with_suffix(".preview.json")
    if dashboard in (output, receipt_path):
        raise ValueError("Preview must not overwrite the dashboard.")
    source_digest = _digest(dashboard)
    executable = _browser_path(browser)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="socpilot-preview-") as temporary:
        workspace = Path(temporary)
        probe = workspace / "measurement.html"
        probe.write_text(
            dashboard.read_text(encoding="utf-8")
            + '<script>window.addEventListener("load",()=>{'
            + 'document.documentElement.setAttribute("data-preview-height",'
            + 'Math.ceil(Math.max(document.body.scrollHeight,'
            + 'document.documentElement.scrollHeight)));});</script>',
            encoding="utf-8",
        )
        common = [
            executable, "--headless", "--disable-gpu", "--hide-scrollbars",
            "--no-first-run", "--no-default-browser-check",
            f"--user-data-dir={workspace / 'browser-profile'}",
            "--force-device-scale-factor=1", "--virtual-time-budget=1000",
        ]
        screenshot = workspace / "preview.png"
        if engine == "playwright":
            height = _playwright_capture(executable, dashboard, screenshot)
        else:
            measurement = subprocess.run(
                [*common, "--window-size=1280,900", "--dump-dom", probe.as_uri()],
                capture_output=True, text=True, check=True, timeout=60,
            )
            height = _page_height(measurement.stdout)
            subprocess.run(
                [*common, f"--window-size=1280,{height}",
                 f"--screenshot={screenshot}", dashboard.as_uri()],
                capture_output=True, text=True, check=True, timeout=60,
            )
        width, actual_height = _png_dimensions(screenshot)
        if width != _VIEWPORT_WIDTH or actual_height != height:
            raise ValueError("Browser capture dimensions differ from the measured viewport.")
        if _digest(dashboard) != source_digest:
            raise ValueError("Dashboard changed during preview capture.")
        receipt = PreviewReceipt(
            dashboard_sha256=source_digest, png_sha256=_digest(screenshot),
            width=width, height=actual_height,
        )
        shutil.copyfile(screenshot, output)
        receipt_path.write_text(receipt.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return receipt_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dashboard-html", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    parser.add_argument("--browser")
    parser.add_argument("--engine", choices=("chrome-cli", "playwright"), default="chrome-cli")
    arguments = parser.parse_args()
    receipt = capture_preview(
        arguments.dashboard_html, arguments.output_png, arguments.browser, arguments.engine,
    )
    print(json.dumps({"status": "ok", "receipt": str(receipt)}))


if __name__ == "__main__":
    main()
