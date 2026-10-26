"""Typed policy representation and deterministic graph projection.

The verifier rule models remain the semantic source of truth. The graph is a
review and authoring projection: it is accepted only when its nodes and edges
can be regenerated exactly from the embedded typed rules and action catalog.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from soc_llm_policy.engine import validate_rule_consistency, validate_rule_scope
from soc_llm_policy.io import (
    ActionCatalogItem,
    PolicyRule,
    parse_action_catalog,
    parse_rules,
    read_yaml_list,
)
from soc_llm_policy.policy_predicates import predicate_surface

PolicyNodeKind = Literal["rule", "action", "scope", "repair", "approval"]
PolicyEdgeKind = Literal[
    "governs",
    "applies_when",
    "requires_before",
    "repairs_with",
    "requires_approval",
]


class PolicyDocument(BaseModel):
    """Canonical policy data shared by the verifier and authoring tools."""

    schema_version: Literal[1] = 1
    catalog: list[ActionCatalogItem]
    rules: list[PolicyRule]

    @model_validator(mode="after")
    def validate_semantics(self) -> PolicyDocument:
        action_ids = [item.action_id for item in self.catalog]
        if len(action_ids) != len(set(action_ids)):
            raise ValueError("Action catalog identifiers must be unique")

        rule_ids = [rule.rule_id for rule in self.rules]
        if len(rule_ids) != len(set(rule_ids)):
            raise ValueError("Policy rule identifiers must be unique")

        unknown_actions = sorted(
            {
                action_id
                for rule in self.rules
                for action_id in (rule.action, rule.condition_action)
                if action_id is not None and action_id not in action_ids
            }
        )
        if unknown_actions:
            raise ValueError(
                f"Policy rules reference unknown catalog actions: {unknown_actions}"
            )

        expected_repairs = {
            "mandatory": "insert",
            "prohibit_before": "insert_before",
            "prohibit_when": "remove",
            "require_approval": "remove",
        }
        for rule in self.rules:
            validate_rule_scope(rule.scope)
            if rule.type == "prohibit_before" and rule.condition_action is None:
                raise ValueError(
                    f"Ordering rule {rule.rule_id!r} requires condition_action"
                )
            if rule.type != "prohibit_before" and rule.condition_action is not None:
                raise ValueError(
                    f"Rule {rule.rule_id!r} cannot declare condition_action"
                )
            expected_repair = expected_repairs[rule.type]
            if (
                rule.repair_operator is not None
                and rule.repair_operator != expected_repair
            ):
                raise ValueError(
                    f"Rule {rule.rule_id!r} requires repair_operator "
                    f"{expected_repair!r}, received {rule.repair_operator!r}"
                )
            if (
                rule.alternative_repair_operator is not None
                and rule.type != "require_approval"
            ):
                raise ValueError(
                    f"Rule {rule.rule_id!r} cannot declare an alternative repair"
                )

        validate_rule_consistency(self.rules)
        return self


class PolicyGraphNode(BaseModel):
    """One deterministic node in the policy review graph."""

    node_id: str = Field(..., min_length=1)
    kind: PolicyNodeKind
    label: str = Field(..., min_length=1)
    ordinal: int = Field(..., ge=0)
    action: ActionCatalogItem | None = None
    rule: PolicyRule | None = None
    scope: dict[str, object] | None = None
    repair_operator: str | None = None
    approval_source: str | None = None

    @model_validator(mode="after")
    def validate_payload(self) -> PolicyGraphNode:
        expected_payload = {
            "action": self.action is not None,
            "rule": self.rule is not None,
            "scope": self.scope is not None,
            "repair": self.repair_operator is not None,
            "approval": self.approval_source is not None,
        }
        if not expected_payload[self.kind]:
            raise ValueError(f"Node {self.node_id!r} lacks its {self.kind} payload")
        populated = sum(expected_payload.values())
        if populated != 1:
            raise ValueError(
                f"Node {self.node_id!r} must contain exactly one typed payload"
            )
        return self


class PolicyGraphEdge(BaseModel):
    """A typed relation derived from one policy rule."""

    edge_id: str = Field(..., min_length=1)
    kind: PolicyEdgeKind
    source: str = Field(..., min_length=1)
    target: str = Field(..., min_length=1)
    rule_id: str = Field(..., min_length=1)


class PolicyGraph(BaseModel):
    """Content-addressed graph projection of a policy document."""

    schema_version: Literal[1] = 1
    policy_digest: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    nodes: list[PolicyGraphNode]
    edges: list[PolicyGraphEdge]

    @model_validator(mode="after")
    def validate_graph_identity(self) -> PolicyGraph:
        node_ids = [node.node_id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("Policy graph node identifiers must be unique")
        edge_ids = [edge.edge_id for edge in self.edges]
        if len(edge_ids) != len(set(edge_ids)):
            raise ValueError("Policy graph edge identifiers must be unique")
        known_nodes = set(node_ids)
        dangling = sorted(
            {
                endpoint
                for edge in self.edges
                for endpoint in (edge.source, edge.target)
                if endpoint not in known_nodes
            }
        )
        if dangling:
            raise ValueError(f"Policy graph has dangling node references: {dangling}")
        return self


def load_policy_document(
    action_catalog_path: Path,
    constraints_path: Path,
) -> PolicyDocument:
    """Load the verifier's canonical YAML inputs into the shared PolicyIR."""
    catalog = parse_action_catalog(read_yaml_list(action_catalog_path))
    rules = parse_rules(read_yaml_list(constraints_path))
    return PolicyDocument(catalog=catalog, rules=rules)


def policy_digest(document: PolicyDocument) -> str:
    """Address the verifier-visible policy semantics, excluding prose."""
    payload = {
        "catalog": [
            item.model_dump(mode="json")
            for item in sorted(document.catalog, key=lambda item: item.action_id)
        ],
        "rules": predicate_surface(document.rules),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _scope_label(scope: dict[str, object]) -> str:
    return " AND ".join(sorted(scope))


def project_policy_graph(document: PolicyDocument) -> PolicyGraph:
    """Project typed rules into a deterministic dependency graph."""
    nodes: dict[str, PolicyGraphNode] = {}
    edges: list[PolicyGraphEdge] = []

    for ordinal, action in enumerate(document.catalog):
        node_id = f"action:{action.action_id}"
        nodes[node_id] = PolicyGraphNode(
            node_id=node_id,
            kind="action",
            label=action.action_id,
            ordinal=ordinal,
            action=action,
        )

    for ordinal, rule in enumerate(document.rules):
        rule_node_id = f"rule:{rule.rule_id}"
        nodes[rule_node_id] = PolicyGraphNode(
            node_id=rule_node_id,
            kind="rule",
            label=rule.rule_id,
            ordinal=ordinal,
            rule=rule,
        )
        edges.append(
            PolicyGraphEdge(
                edge_id=f"{rule.rule_id}:governs",
                kind="governs",
                source=rule_node_id,
                target=f"action:{rule.action}",
                rule_id=rule.rule_id,
            )
        )

        if rule.scope:
            scope_node_id = f"scope:{rule.rule_id}"
            nodes[scope_node_id] = PolicyGraphNode(
                node_id=scope_node_id,
                kind="scope",
                label=_scope_label(rule.scope),
                ordinal=ordinal,
                scope=rule.scope,
            )
            edges.append(
                PolicyGraphEdge(
                    edge_id=f"{rule.rule_id}:applies_when",
                    kind="applies_when",
                    source=scope_node_id,
                    target=rule_node_id,
                    rule_id=rule.rule_id,
                )
            )

        if rule.condition_action:
            edges.append(
                PolicyGraphEdge(
                    edge_id=f"{rule.rule_id}:requires_before",
                    kind="requires_before",
                    source=f"action:{rule.condition_action}",
                    target=f"action:{rule.action}",
                    rule_id=rule.rule_id,
                )
            )

        if rule.repair_operator:
            repair_node_id = f"repair:{rule.rule_id}"
            nodes[repair_node_id] = PolicyGraphNode(
                node_id=repair_node_id,
                kind="repair",
                label=rule.repair_operator,
                ordinal=ordinal,
                repair_operator=rule.repair_operator,
            )
            edges.append(
                PolicyGraphEdge(
                    edge_id=f"{rule.rule_id}:repairs_with",
                    kind="repairs_with",
                    source=rule_node_id,
                    target=repair_node_id,
                    rule_id=rule.rule_id,
                )
            )

        if rule.type == "require_approval":
            evidence = rule.approval_evidence or {}
            approval_source = str(evidence.get("source", "unspecified"))
            approval_node_id = f"approval:{rule.rule_id}"
            nodes[approval_node_id] = PolicyGraphNode(
                node_id=approval_node_id,
                kind="approval",
                label=approval_source,
                ordinal=ordinal,
                approval_source=approval_source,
            )
            edges.append(
                PolicyGraphEdge(
                    edge_id=f"{rule.rule_id}:requires_approval",
                    kind="requires_approval",
                    source=approval_node_id,
                    target=rule_node_id,
                    rule_id=rule.rule_id,
                )
            )

    return PolicyGraph(
        policy_digest=policy_digest(document),
        nodes=sorted(nodes.values(), key=lambda node: node.node_id),
        edges=sorted(edges, key=lambda edge: edge.edge_id),
    )


def compile_policy_graph(graph: PolicyGraph) -> PolicyDocument:
    """Recover typed policy data and reject any inconsistent graph projection."""
    action_nodes = sorted(
        (node for node in graph.nodes if node.kind == "action"),
        key=lambda node: node.ordinal,
    )
    rule_nodes = sorted(
        (node for node in graph.nodes if node.kind == "rule"),
        key=lambda node: node.ordinal,
    )
    document = PolicyDocument(
        catalog=[node.action for node in action_nodes if node.action is not None],
        rules=[node.rule for node in rule_nodes if node.rule is not None],
    )
    expected = project_policy_graph(document)
    if graph != expected:
        raise ValueError(
            "Policy graph does not match the deterministic projection of its typed rules"
        )
    return document


def dump_policy_graph(graph: PolicyGraph, path: Path) -> None:
    """Write a stable, reviewable JSON graph."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(graph.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def dump_constraints(document: PolicyDocument, path: Path) -> None:
    """Write verifier-compatible rule YAML from a validated PolicyIR."""
    payload = [
        rule.model_dump(mode="json", exclude_none=True) for rule in document.rules
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
