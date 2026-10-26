from __future__ import annotations

import json
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import ClassVar

from soc_llm_policy.models_registry import ModelProfile


class ProviderCostLimitError(RuntimeError):
    """The next provider request would exceed the approved cost guardrail."""


type AttemptIdentity = tuple[str, str, int]
_TRACKERS_BY_PATH: dict[Path, ProviderCostTracker] = {}


def _nonnegative_decimal(value: object, *, label: str) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except InvalidOperation as exc:
        raise RuntimeError(f"{label} is not a valid decimal value.") from exc
    if not parsed.is_finite() or parsed < 0:
        raise RuntimeError(f"{label} must be finite and non-negative.")
    return parsed


def _attempt_identity(record: dict[str, object]) -> AttemptIdentity | None:
    cell_id = record.get("cell_id")
    execution_id = record.get("execution_id")
    attempt_index = record.get("provider_attempt_index")
    if (
        isinstance(cell_id, str)
        and cell_id
        and isinstance(execution_id, str)
        and execution_id
        and isinstance(attempt_index, int)
        and attempt_index >= 1
    ):
        return cell_id, execution_id, attempt_index
    return None


def _known_reservation(
    raw_value: object,
    *,
    billing_mode: object,
    valid_token_bounds: bool,
) -> Decimal | None:
    if billing_mode == "local":
        if raw_value not in {None, 0}:
            raise RuntimeError("Local provider reservation must be zero.")
        if not valid_token_bounds:
            raise RuntimeError("Local provider attempt has invalid token bounds.")
        return Decimal("0")
    if raw_value is None or not valid_token_bounds:
        return None
    reservation = _nonnegative_decimal(
        raw_value,
        label="Provider-attempt reservation",
    )
    return reservation if reservation > 0 else None


def _known_response_cost(raw_value: object, *, billing_mode: object) -> Decimal | None:
    if billing_mode == "local":
        if raw_value not in {None, 0}:
            raise RuntimeError("Local provider response cost must be zero.")
        return Decimal("0")
    if raw_value is None:
        return None
    return _nonnegative_decimal(raw_value, label="Provider-attempt estimated cost")


class ProviderCostTracker:
    """Account for paid responses and reserve the next bounded request."""

    _PAID_RESPONSE_OUTCOMES: ClassVar[set[str]] = {
        "response_received",
        "terminal_model_response",
    }

    def __init__(self, path: Path) -> None:
        self.path = path
        self.offset = 0
        self.accrued_usd = Decimal("0")
        self.request_count = 0
        self.response_count = 0
        self._unknown_paid_cost_count = 0
        self._attempt_reservations: dict[AttemptIdentity, Decimal] = {}
        self._unknown_reservations: set[AttemptIdentity] = set()
        self._settled_attempts: set[AttemptIdentity] = set()
        self._finished_attempts: set[AttemptIdentity] = set()

    @property
    def reserved_exposure_usd(self) -> Decimal:
        return sum(
            (
                value
                for identity, value in self._attempt_reservations.items()
                if identity not in self._settled_attempts
            ),
            start=Decimal("0"),
        )

    @property
    def budget_exposure_usd(self) -> Decimal:
        return self.accrued_usd + self.reserved_exposure_usd

    @property
    def unknown_cost_count(self) -> int:
        return self._unknown_paid_cost_count + len(self._unknown_reservations)

    def refresh(self) -> None:
        if not self.path.exists():
            return
        file_size = self.path.stat().st_size
        if file_size < self.offset:
            raise RuntimeError(f"Provider-attempt log was truncated: {self.path}")
        with self.path.open("rb") as handle:
            handle.seek(self.offset)
            for line_number, encoded_line in enumerate(handle, start=1):
                if not encoded_line.endswith(b"\n"):
                    raise RuntimeError(
                        f"Provider-attempt log has an incomplete record: {self.path}"
                    )
                try:
                    record = json.loads(encoded_line)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise RuntimeError(
                        "Provider-attempt log contains invalid JSON after byte "
                        f"{self.offset} (record {line_number})."
                    ) from exc
                if not isinstance(record, dict):
                    raise RuntimeError("Provider-attempt record must be a JSON object.")
                outcome = record.get("outcome")
                identity = _attempt_identity(record)
                if outcome == "request_started":
                    if identity is None:
                        raise RuntimeError(
                            "Provider-attempt start has no complete attempt identity."
                        )
                    if (
                        identity in self._attempt_reservations
                        or identity in self._unknown_reservations
                    ):
                        raise RuntimeError(
                            "Provider-attempt log contains a duplicate request start."
                        )
                    raw_reservation = record.get("reserved_cost_usd")
                    billing_mode = record.get("billing_mode", "metered")
                    prompt_tokens = record.get("prompt_tokens_reserved")
                    completion_tokens = record.get(
                        "max_completion_tokens_reserved"
                    )
                    valid_token_bounds = (
                        isinstance(prompt_tokens, int)
                        and not isinstance(prompt_tokens, bool)
                        and prompt_tokens >= 1
                        and isinstance(completion_tokens, int)
                        and not isinstance(completion_tokens, bool)
                        and completion_tokens >= 1
                    )
                    reservation = _known_reservation(
                        raw_reservation,
                        billing_mode=billing_mode,
                        valid_token_bounds=valid_token_bounds,
                    )
                    if reservation is None:
                        self._unknown_reservations.add(identity)
                    else:
                        self._attempt_reservations[identity] = reservation
                    self.request_count += 1
                    continue
                if identity is not None:
                    if (
                        identity not in self._attempt_reservations
                        and identity not in self._unknown_reservations
                    ):
                        raise RuntimeError(
                            "Provider-attempt outcome has no matching request start."
                        )
                    if identity in self._finished_attempts:
                        raise RuntimeError(
                            "Provider-attempt log contains a duplicate outcome."
                        )
                    self._finished_attempts.add(identity)
                if outcome not in self._PAID_RESPONSE_OUTCOMES:
                    continue
                if identity is not None:
                    self._settled_attempts.add(identity)
                self.response_count += 1
                raw_cost = record.get("estimated_cost_usd")
                billing_mode = record.get("billing_mode", "metered")
                cost = _known_response_cost(raw_cost, billing_mode=billing_mode)
                if cost is None:
                    self._unknown_paid_cost_count += 1
                    continue
                self.accrued_usd += cost
            self.offset = handle.tell()

    def require_budget_for_next_attempt(
        self,
        *,
        approved_budget_usd: float,
        prompt_tokens_reserved: int,
        max_completion_tokens: int,
        prompt_price_per_1k_usd: float,
        completion_price_per_1k_usd: float,
    ) -> float:
        """Reserve one transport attempt against realized and ambiguous spend."""
        if prompt_tokens_reserved < 1 or max_completion_tokens < 1:
            raise ProviderCostLimitError(
                "Provider token reservations must be positive integers."
            )
        self.refresh()
        if self.unknown_cost_count:
            raise ProviderCostLimitError(
                "Provider cost accounting is incomplete for "
                f"{self.unknown_cost_count} paid or potentially paid request(s); "
                "execution stopped."
            )
        prompt_price = _nonnegative_decimal(
            prompt_price_per_1k_usd,
            label="Prompt price",
        )
        completion_price = _nonnegative_decimal(
            completion_price_per_1k_usd,
            label="Completion price",
        )
        projected_cost = (
            Decimal(prompt_tokens_reserved) * prompt_price / Decimal(1000)
            + Decimal(max_completion_tokens) * completion_price / Decimal(1000)
        )
        ceiling = _nonnegative_decimal(
            approved_budget_usd,
            label="Approved provider budget",
        )
        if self.budget_exposure_usd + projected_cost > ceiling:
            raise ProviderCostLimitError(
                "The next provider attempt would exceed the approved cost guardrail: "
                f"realized USD {self.accrued_usd:.6f}, ambiguous USD "
                f"{self.reserved_exposure_usd:.6f}, reserved attempt USD "
                f"{projected_cost:.6f}, guardrail USD {ceiling:.2f}."
            )
        return float(projected_cost)

    def require_budget_for_next_request(
        self,
        profile: ModelProfile,
        approved_budget_usd: float,
        *,
        max_completion_tokens: int,
    ) -> None:
        """Reserve a conservative output bound before issuing one request."""
        if profile.planning_prompt_tokens is None:
            raise ProviderCostLimitError(
                f"Model {profile.name} has no planning prompt-token assumption."
            )
        if profile.planning_completion_tokens is None:
            raise ProviderCostLimitError(
                f"Model {profile.name} has no planning completion-token assumption."
            )
        if max_completion_tokens < profile.planning_completion_tokens:
            raise ProviderCostLimitError(
                "Maximum completion tokens cannot be below the planning assumption."
            )
        self.require_budget_for_next_attempt(
            approved_budget_usd=approved_budget_usd,
            prompt_tokens_reserved=profile.planning_prompt_tokens,
            max_completion_tokens=max_completion_tokens,
            prompt_price_per_1k_usd=profile.prompt_price_per_1k_usd,
            completion_price_per_1k_usd=profile.completion_price_per_1k_usd,
        )


def provider_cost_tracker(path: Path) -> ProviderCostTracker:
    """Reuse incremental ledger state within one sequential campaign process."""
    resolved = path.expanduser().resolve()
    tracker = _TRACKERS_BY_PATH.get(resolved)
    if tracker is None:
        tracker = ProviderCostTracker(resolved)
        _TRACKERS_BY_PATH[resolved] = tracker
    return tracker
