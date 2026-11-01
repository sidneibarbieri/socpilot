"""Derive a graph with current metadata without changing frozen source bytes."""

from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from soc_llm_policy.policy_ir import (
    PolicyGraph,
    dump_policy_graph,
    load_policy_document,
    project_policy_graph,
)

GRAPH_PATH = "artifact_data/global/policy_graph.json"
REFRESHED_GRAPH_PATH = "artifact_outputs/analysis/policy_graph.json"


class GraphMetadataRepair(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_path: str = GRAPH_PATH
    derived_path: str = REFRESHED_GRAPH_PATH
    original_sha256: str
    repaired_sha256: str
    original_policy_digest: str
    repaired_policy_digest: str


def repair_graph_digest(package_root: Path) -> GraphMetadataRepair | None:
    """Require identical typed content; write corrected metadata as analysis."""
    graph_path = package_root / GRAPH_PATH
    original_bytes = graph_path.read_bytes()
    graph = PolicyGraph.model_validate_json(original_bytes)
    global_root = graph_path.parent
    document = load_policy_document(
        global_root / "action_catalog.yaml", global_root / "constraints.yaml"
    )
    expected = project_policy_graph(document)
    rebound = graph.model_copy(update={"policy_digest": expected.policy_digest})
    if rebound != expected:
        raise ValueError("Policy graph content differs from published typed YAML rules")
    if graph == expected:
        return None
    derived_path = package_root / REFRESHED_GRAPH_PATH
    if derived_path.exists():
        previous = PolicyGraph.model_validate_json(derived_path.read_bytes())
        if previous != expected:
            raise FileExistsError(derived_path)
    else:
        dump_policy_graph(expected, derived_path)
    return GraphMetadataRepair(
        original_sha256=hashlib.sha256(original_bytes).hexdigest(),
        repaired_sha256=hashlib.sha256(derived_path.read_bytes()).hexdigest(),
        original_policy_digest=graph.policy_digest,
        repaired_policy_digest=expected.policy_digest,
    )
