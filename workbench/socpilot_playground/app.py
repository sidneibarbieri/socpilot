"""Local synthetic proposal editor backed by the existing SOCpilot engine."""

from __future__ import annotations

import argparse
import errno
import json
from dataclasses import replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from soc_llm_policy.policy_example import example_request

from .cases import PolicyCase, decode_json, evaluate_case
from .service import evaluate_request


class Proposal(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    actions: list[Literal["collect_forensics", "reimage_host", "isolate_host"]] = Field(
        min_length=1,
        max_length=20,
    )


def evaluate_proposal(proposal: Proposal) -> dict[str, object]:
    request = replace(example_request(), llm_actions=proposal.actions)
    return evaluate_request(request, evidence_kind="synthetic_demonstration")


class PlaygroundHandler(BaseHTTPRequestHandler):
    """Serve a fixed local UI and bounded synthetic proposals only."""

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(5)

    def trusted_request(self) -> bool:
        return self.headers.get("Host") == f"127.0.0.1:{self.server.server_port}"

    def do_GET(self) -> None:
        if not self.trusted_request():
            self.send_json(
                HTTPStatus.FORBIDDEN,
                {"error": "Use the loopback URL printed by the server"},
            )
            return
        if self.path == "/api/example":
            request = example_request()
            self.send_json(
                HTTPStatus.OK,
                {
                    "actions": request.llm_actions,
                    "policy": [
                        rule.model_dump(exclude_none=True) for rule in request.rules
                    ],
                },
            )
            return
        if self.path == "/api/case-schema":
            self.send_json(HTTPStatus.OK, PolicyCase.model_json_schema())
            return
        if self.path == "/api/case-example":
            example = Path(__file__).parent / "policy-case.json"
            self.send_json(HTTPStatus.OK, decode_json(example.read_bytes()))
            return
        static = {
            "/": ("index.html", "text/html; charset=utf-8"),
            "/app.js": ("app.js", "text/javascript; charset=utf-8"),
            "/style.css": ("style.css", "text/css; charset=utf-8"),
            "/icon.svg": ("icon.svg", "image/svg+xml"),
        }.get(self.path)
        if static is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        filename, content_type = static
        self.send_body(
            HTTPStatus.OK, (Path(__file__).parent / filename).read_bytes(), content_type
        )

    def do_POST(self) -> None:
        expected_origin = f"http://127.0.0.1:{self.server.server_port}"
        if not self.trusted_request() or self.headers.get("Origin") != expected_origin:
            self.send_json(
                HTTPStatus.FORBIDDEN, {"error": "Same-origin loopback request required"}
            )
            return
        routes = {
            "/api/check": (Proposal, evaluate_proposal, 4096),
            "/api/cases/check": (PolicyCase, evaluate_case, 65536),
        }
        route = routes.get(self.path)
        if route is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        model, evaluate, maximum_bytes = route
        if self.headers.get_content_type() != "application/json":
            self.send_json(
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "Send application/json"}
            )
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid Content-Length"})
            return
        if not 1 <= content_length <= maximum_bytes:
            self.send_json(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                {"error": f"Send 1..{maximum_bytes} bytes"},
            )
            return
        try:
            payload = decode_json(self.rfile.read(content_length))
        except (ValueError, UnicodeDecodeError, RecursionError):
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid JSON"})
            return
        try:
            proposal = model.model_validate(payload)
        except ValidationError as error:
            self.send_json(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                {
                    "error": "; ".join(
                        f"{'.'.join(map(str, issue['loc']))}: {issue['msg']}"
                        for issue in error.errors()
                    ),
                },
            )
            return
        self.send_json(HTTPStatus.OK, evaluate(proposal))

    def send_json(self, status: HTTPStatus, payload: object) -> None:
        self.send_body(
            status, json.dumps(payload).encode(), "application/json; charset=utf-8"
        )

    def send_body(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'none'; script-src 'self'; "
            "style-src 'self'; img-src 'self'; connect-src 'self'; base-uri 'none'; "
            "frame-ancestors 'none'; form-action 'none'",
        )
        self.end_headers()
        self.wfile.write(body)


MAXIMUM_PORT = 65535


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8767)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--input", type=Path, help="Evaluate a guided proposal JSON")
    inputs.add_argument(
        "--case-input", type=Path, help="Evaluate a complete policy case JSON"
    )
    args = parser.parse_args()
    if args.case_input is not None:
        case = PolicyCase.model_validate(decode_json(args.case_input.read_bytes()))
        print(json.dumps(evaluate_case(case), indent=2, allow_nan=False))
        return
    if args.input is not None:
        proposal = Proposal.model_validate_json(args.input.read_text())
        print(json.dumps(evaluate_proposal(proposal), indent=2))
        return
    if not 0 <= args.port <= MAXIMUM_PORT:
        parser.error("port must be in 0..65535")
    try:
        server = HTTPServer(("127.0.0.1", args.port), PlaygroundHandler)
    except OSError as error:
        if error.errno != errno.EADDRINUSE:
            raise
        parser.error(
            f"Port {args.port} is in use; choose another --port ({error.strerror})."
        )
    with server:
        print(
            f"SOCpilot synthetic playground: http://127.0.0.1:{server.server_port}",
            flush=True,
        )
        server.serve_forever()


if __name__ == "__main__":
    main()
