"""Behavioral checks for the isolated synthetic policy interface."""

import ast
import json
import socket
import subprocess
import sys
import tempfile
import unittest
from http.client import HTTPConnection
from http.server import HTTPServer
from pathlib import Path
from threading import Thread

from pydantic import ValidationError
from socpilot_playground.app import PlaygroundHandler, Proposal, evaluate_proposal
from test_cases import ordering_case


class PolicyPlaygroundTests(unittest.TestCase):
    def test_busy_port_names_the_recovery_option(self) -> None:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            port = listener.getsockname()[1]
            result = subprocess.run(
                [sys.executable, "-m", "socpilot_playground", "--port", str(port)],
                cwd=Path(__file__).parent,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        self.assertEqual(result.returncode, 2)
        self.assertIn("--port", result.stderr)
        self.assertIn(str(port), result.stderr)

    def test_cli_can_check_a_case_with_its_own_catalog_and_rules(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            case_path = Path(directory) / "case.json"
            case_path.write_text(json.dumps(ordering_case()))
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "socpilot_playground",
                    "--case-input",
                    str(case_path),
                ],
                cwd=Path(__file__).parent,
                capture_output=True,
                check=False,
                text=True,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            json.loads(result.stdout)["repaired_actions"],
            ["capture_memory", "restart_service"],
        )

    def test_package_syntax_supports_declared_python_minimum(self) -> None:
        root = Path(__file__).parent
        for source in [*root.glob("*.py"), *root.glob("socpilot_playground/*.py")]:
            with self.subTest(source=source.name):
                ast.parse(
                    source.read_text(), filename=str(source), feature_version=(3, 12)
                )

    def test_module_entrypoint_checks_a_proposal_file(self) -> None:
        root = Path(__file__).parent
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "socpilot_playground",
                "--input",
                "violating-proposal.json",
            ],
            cwd=root,
            capture_output=True,
            check=False,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            json.loads(result.stdout)["repaired_actions"],
            ["collect_forensics", "reimage_host"],
        )

    def test_unsafe_proposal_exposes_both_rules_and_repairs_the_sequence(self) -> None:
        result = evaluate_proposal(Proposal(actions=["reimage_host", "isolate_host"]))
        self.assertEqual(
            result["repaired_actions"], ["collect_forensics", "reimage_host"]
        )
        self.assertEqual(
            result["violated_rules_before"],
            ["evidence_before_reimage", "no_isolation_in_example"],
        )
        self.assertEqual(result["violated_rules_after"], [])
        self.assertTrue(result["second_pass_unchanged"])
        self.assertEqual(result["evidence_kind"], "synthetic_demonstration")
        self.assertFalse(result["host_actions_executed"])

    def test_compliant_proposal_requires_no_repair(self) -> None:
        result = evaluate_proposal(
            Proposal(actions=["collect_forensics", "reimage_host"])
        )
        self.assertEqual(result["violated_rules_before"], [])
        self.assertEqual(result["edits"], [])
        self.assertEqual(
            result["repaired_actions"], ["collect_forensics", "reimage_host"]
        )

    def test_order_changes_the_verdict(self) -> None:
        result = evaluate_proposal(
            Proposal(actions=["reimage_host", "collect_forensics"])
        )
        self.assertIn("evidence_before_reimage", result["violated_rules_before"])

    def test_unknown_action_is_rejected_instead_of_silently_dropped(self) -> None:
        with self.assertRaises(ValidationError):
            Proposal(actions=["arbitrary_shell_command"])

    def test_unknown_fields_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            Proposal.model_validate(
                {"actions": ["collect_forensics"], "command": "ignored"}
            )

    def test_empty_and_unbounded_proposals_are_rejected(self) -> None:
        for actions in ([], ["collect_forensics"] * 21):
            with self.subTest(length=len(actions)), self.assertRaises(ValidationError):
                Proposal(actions=actions)


class PlaygroundHttpTests(unittest.TestCase):
    def test_favicon_is_local_and_servable(self) -> None:
        status, icon = self.request("GET", "/icon.svg")
        self.assertEqual(status, 200)
        self.assertIn(b"<svg", icon)
        status, page = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn(b'href="/icon.svg"', page)
        self.assertNotIn(b"data:image", page)

    def setUp(self) -> None:
        self.server = HTTPServer(("127.0.0.1", 0), PlaygroundHandler)
        self.worker = Thread(target=self.server.serve_forever, daemon=True)
        self.worker.start()
        self.origin = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.worker.join()

    def request(
        self,
        method: str,
        path: str,
        body: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, bytes]:
        connection = HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        try:
            connection.request(method, path, body, headers or {})
            response = connection.getresponse()
            return response.status, response.read()
        finally:
            connection.close()

    def test_http_result_matches_the_cli_service(self) -> None:
        status, body = self.request(
            "POST",
            "/api/check",
            '{"actions":["isolate_host"]}',
            {"Origin": self.origin, "Content-Type": "application/json"},
        )
        self.assertEqual(status, 200)
        result = json.loads(body)
        self.assertEqual(result["violated_rules_before"], ["no_isolation_in_example"])
        self.assertEqual(result["repaired_actions"], [])

    def test_case_endpoint_checks_a_supplied_policy(self) -> None:
        status, body = self.request(
            "POST",
            "/api/cases/check",
            json.dumps(ordering_case()),
            {"Origin": self.origin, "Content-Type": "application/json"},
        )
        self.assertEqual(status, 200)
        result = json.loads(body)
        self.assertEqual(
            result["repaired_actions"], ["capture_memory", "restart_service"]
        )
        self.assertEqual(
            result["input_case"]["catalog"][0]["action_id"], "capture_memory"
        )

    def test_case_endpoint_recovers_after_invalid_case(self) -> None:
        headers = {"Origin": self.origin, "Content-Type": "application/json"}
        status, _ = self.request(
            "POST", "/api/cases/check", '{"case_id":"incomplete"}', headers
        )
        self.assertEqual(status, 422)
        status, _ = self.request(
            "POST", "/api/cases/check", json.dumps(ordering_case()), headers
        )
        self.assertEqual(status, 200)

    def test_case_schema_and_example_are_available(self) -> None:
        for path in ("/api/case-schema", "/api/case-example"):
            with self.subTest(path=path):
                status, body = self.request("GET", path)
                self.assertEqual(status, 200)
                self.assertIsInstance(json.loads(body), dict)

    def test_case_endpoint_rejects_nonfinite_json_numbers(self) -> None:
        payload = ordering_case()
        payload["telemetry"] = [
            {"event_type": "signal", "details": {"raw": {"value": float("nan")}}}
        ]
        status, _ = self.request(
            "POST",
            "/api/cases/check",
            json.dumps(payload),
            {"Origin": self.origin, "Content-Type": "application/json"},
        )
        self.assertEqual(status, 400)

    def test_external_origin_cannot_submit_a_plan(self) -> None:
        status, _ = self.request(
            "POST",
            "/api/check",
            '{"actions":["isolate_host"]}',
            {"Origin": "https://unrelated.example"},
        )
        self.assertEqual(status, 403)

    def test_browser_missing_origin_cannot_submit_a_plan(self) -> None:
        status, _ = self.request("POST", "/api/check", "{}")
        self.assertEqual(status, 403)

    def test_unknown_actions_return_a_validation_error(self) -> None:
        status, body = self.request(
            "POST",
            "/api/check",
            '{"actions":["execute"]}',
            {"Origin": self.origin, "Content-Type": "application/json"},
        )
        self.assertEqual(status, 422)
        self.assertIn("error", json.loads(body))

    def test_malformed_json_returns_an_explicit_error(self) -> None:
        status, _ = self.request(
            "POST",
            "/api/check",
            "{",
            {"Origin": self.origin, "Content-Type": "application/json"},
        )
        self.assertEqual(status, 400)

    def test_large_requests_are_rejected_before_evaluation(self) -> None:
        status, _ = self.request(
            "POST",
            "/api/check",
            " " * 4097,
            {"Origin": self.origin, "Content-Type": "application/json"},
        )
        self.assertEqual(status, 413)

    def test_server_does_not_expose_workspace_paths(self) -> None:
        status, _ = self.request("GET", "/../../../.env")
        self.assertEqual(status, 404)


if __name__ == "__main__":
    unittest.main()
