"""The decidable surface of a policy bundle, addressed by its content.

A rule file carries two kinds of field. The predicates decide: a scope, a type,
an action, a severity, a repair operator. The prose explains: a rationale, an
evidence basis naming the incidents a rule was drawn from. Correcting a corpus
rewrites the prose and leaves every predicate untouched, yet the file hash moves,
and a gate that watches the file hash then blocks a state it should admit.

So the gates split by what their consumer reads. Deterministic reports over the
policy read predicates alone and bind to this digest. Model evidence binds to the
whole file, because the policy arm's prompt carried the prose into the model's
context and the run is evidence about the text the model saw.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from soc_llm_policy.io import parse_rules, read_yaml_list

# The two fields no decision procedure reads. Everything else is a predicate,
# including fields the current engine ignores: a surface that grows must fail
# this gate rather than slip past it.
EXPLANATORY_FIELDS = frozenset({"rationale", "evidence_basis"})


def predicate_surface(rules: list[Any]) -> list[dict[str, Any]]:
    """The rules stripped to the fields a decision procedure reads."""
    surface = []
    for rule in rules:
        fields = rule.model_dump(mode="json")
        surface.append(
            {name: value for name, value in fields.items() if name not in EXPLANATORY_FIELDS}
        )
    return sorted(surface, key=lambda entry: entry["rule_id"])


def predicate_digest(constraints_path: Path) -> str:
    """A content address for the decidable surface of a rule bundle."""
    surface = predicate_surface(parse_rules(read_yaml_list(constraints_path)))
    payload = json.dumps(surface, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
