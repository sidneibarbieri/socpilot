"""Check real policy-case inputs against the existing engine semantics."""

import unittest

from pydantic import ValidationError
from socpilot_playground.cases import PolicyCase, evaluate_case


def ordering_case() -> dict[str, object]:
    return {
        "case_id": "local-ordering-example",
        "actions": ["restart_service"],
        "catalog": [{"action_id": "capture_memory"}, {"action_id": "restart_service"}],
        "rules": [
            {
                "rule_id": "preserve_evidence",
                "type": "prohibit_before",
                "action": "restart_service",
                "condition_action": "capture_memory",
            }
        ],
    }


class PolicyCaseTests(unittest.TestCase):
    def test_catalog_normalization_is_visible_even_without_rule_repairs(self) -> None:
        payload = ordering_case()
        payload["actions"] = ["capture_memory", "capture_memory", "restart_service"]
        result = evaluate_case(PolicyCase.model_validate(payload))
        self.assertEqual(result["edits"], [])
        self.assertEqual(
            result["repaired_actions"], ["capture_memory", "restart_service"]
        )
        self.assertTrue(result["plan_changed"])

    def test_declared_catalog_supports_actions_outside_guided_example(self) -> None:
        result = evaluate_case(PolicyCase.model_validate(ordering_case()))
        self.assertEqual(
            result["repaired_actions"], ["capture_memory", "restart_service"]
        )
        self.assertEqual(result["violated_rules_before"], ["preserve_evidence"])
        self.assertEqual(result["violated_rules_after"], [])
        self.assertEqual(result["input_case"]["case_id"], "local-ordering-example")
        self.assertFalse(result["host_actions_executed"])

    def test_unknown_action_is_not_silently_filtered(self) -> None:
        payload = ordering_case()
        payload["actions"] = ["not_in_catalog"]
        with self.assertRaises(ValidationError):
            PolicyCase.model_validate(payload)

    def test_malformed_ordering_rule_is_rejected(self) -> None:
        payload = ordering_case()
        del payload["rules"][0]["condition_action"]
        with self.assertRaises(ValidationError):
            PolicyCase.model_validate(payload)

    def test_cyclic_ordering_policy_uses_core_consistency_check(self) -> None:
        payload = ordering_case()
        payload["rules"].append(
            {
                "rule_id": "reverse",
                "type": "prohibit_before",
                "action": "capture_memory",
                "condition_action": "restart_service",
            }
        )
        with self.assertRaises(ValidationError):
            PolicyCase.model_validate(payload)

    def test_duplicate_catalog_and_rule_identifiers_are_rejected(self) -> None:
        for field in ("catalog", "rules"):
            payload = ordering_case()
            payload[field].append(dict(payload[field][0]))
            with self.subTest(field=field), self.assertRaises(ValidationError):
                PolicyCase.model_validate(payload)

    def test_rule_typo_and_unknown_scope_are_rejected(self) -> None:
        for extra in (
            {"conditon_action": "capture_memory"},
            {"scope": {"misspelled_scope": ["x"]}},
        ):
            payload = ordering_case()
            payload["rules"][0].update(extra)
            with self.subTest(extra=extra), self.assertRaises(ValidationError):
                PolicyCase.model_validate(payload)

    def test_telemetry_activates_the_declared_rule(self) -> None:
        payload = ordering_case()
        payload["actions"] = []
        payload["rules"] = [
            {
                "rule_id": "capture_on_signal",
                "type": "mandatory",
                "action": "capture_memory",
                "scope": {"event_type_contains": ["signal"]},
            }
        ]
        quiet = evaluate_case(PolicyCase.model_validate(payload))
        self.assertEqual(quiet["repaired_actions"], [])
        payload["telemetry"] = [{"event_type": "synthetic_signal"}]
        active = evaluate_case(PolicyCase.model_validate(payload))
        self.assertEqual(active["repaired_actions"], ["capture_memory"])

    def test_approval_context_changes_the_declared_rule_verdict(self) -> None:
        payload = ordering_case()
        payload["catalog"][1]["requires_approval"] = True
        payload["rules"] = [
            {
                "rule_id": "restart_approval",
                "type": "require_approval",
                "action": "restart_service",
            }
        ]
        unapproved = evaluate_case(PolicyCase.model_validate(payload))
        self.assertEqual(unapproved["repaired_actions"], [])
        payload["incident_approved_actions"] = ["restart_service"]
        approved = evaluate_case(PolicyCase.model_validate(payload))
        self.assertEqual(approved["violated_rules_before"], [])
        self.assertEqual(approved["repaired_actions"], ["restart_service"])

    def test_stable_repair_can_still_violate_a_conflicting_policy(self) -> None:
        payload = ordering_case()
        payload["actions"] = []
        payload["rules"] = [
            {
                "rule_id": "must_capture",
                "type": "mandatory",
                "action": "capture_memory",
            },
            {
                "rule_id": "must_not_capture",
                "type": "prohibit_when",
                "action": "capture_memory",
            },
        ]
        result = evaluate_case(PolicyCase.model_validate(payload))
        self.assertEqual(result["repaired_actions"], [])
        self.assertEqual(result["violated_rules_after"], ["must_capture"])
        self.assertTrue(result["second_pass_unchanged"])
        self.assertTrue(result["second_pass_edits"])


if __name__ == "__main__":
    unittest.main()
