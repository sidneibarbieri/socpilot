"""Bind a documentation preview to the HTML it depicts."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PreviewReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal[1] = 1
    purpose: Literal["dashboard_preview_not_execution_evidence"] = "dashboard_preview_not_execution_evidence"
    dashboard_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    png_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    width: int = Field(gt=0)
    height: int = Field(gt=0)


def verify_preview(dashboard: Path, image: Path, receipt_path: Path) -> None:
    """Reject missing receipts, stale HTML bindings, and changed image bytes."""
    receipt = PreviewReceipt.model_validate_json(receipt_path.read_text(encoding="utf-8"))
    for path, expected in (
        (dashboard, receipt.dashboard_sha256),
        (image, receipt.png_sha256),
    ):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Dashboard preview binding mismatch: {path.name}")
