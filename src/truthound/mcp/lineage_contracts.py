"""Closed graph edits and native impact analysis, separate from remote emitters."""

from typing import Annotated, Literal

from pydantic import Field, JsonValue, model_validator

from truthound.mcp.contracts import Name, WireModel
from truthound.mcp.policy import MCPError
from truthound.mcp.session_contracts import Handle
from truthound.mcp.validator_wire import contracts, fields, record_fields


def lineage_fields(kind, value):
    return fields(record_fields(contracts()["lineage"][kind]), value)


class GraphCreate(WireModel):
    config: dict[str, JsonValue] = Field(default_factory=dict)
    nodes: Annotated[list[dict[str, JsonValue]], Field(max_length=1000)] = Field(
        default_factory=list
    )
    edges: Annotated[list[dict[str, JsonValue]], Field(max_length=2000)] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def constructors(self):
        if self.config.get("persist_path") is not None:
            raise MCPError("operator_configuration_required")
        lineage_fields("config", self.config)
        for row in self.nodes:
            lineage_fields("node", row)
        for row in self.edges:
            lineage_fields("edge", row)
        return self


class GraphInput(WireModel):
    graph_id: Handle


class GraphRender(GraphInput):
    renderer: Literal["d3", "cytoscape", "graphviz", "mermaid"]
    settings: dict[str, JsonValue] = Field(default_factory=dict)
    root_node_id: Name | None = None
    direction: Literal["upstream", "downstream", "both"] = "downstream"

    @model_validator(mode="after")
    def constructor(self):
        lineage_fields("render_config", self.settings)
        return self


class NodeAdd(WireModel):
    action: Literal["add_node"]
    node: dict[str, JsonValue]

    @model_validator(mode="after")
    def constructor(self):
        lineage_fields("node", self.node)
        return self


class NodeRemove(WireModel):
    action: Literal["remove_node"]
    node_id: Name


class EdgeAdd(WireModel):
    action: Literal["add_edge"]
    edge: dict[str, JsonValue]

    @model_validator(mode="after")
    def constructor(self):
        lineage_fields("edge", self.edge)
        return self


class EdgeRemove(WireModel):
    action: Literal["remove_edge"]
    source: Name
    target: Name


class GraphClear(WireModel):
    action: Literal["clear"]


class GraphEdit(GraphInput):
    edit: Annotated[
        NodeAdd | NodeRemove | EdgeAdd | EdgeRemove | GraphClear, Field(discriminator="action")
    ]


class QueryAll(GraphInput):
    view: Literal["graph", "nodes", "edges", "roots", "leaves", "topological"] = "graph"


class QueryNode(GraphInput):
    view: Literal["upstream", "downstream"]
    node_id: Name
    depth: Annotated[int, Field(ge=-1, le=1000)] = -1


class NodeLookup(GraphInput):
    view: Literal["node", "has_node"]
    node_id: Name


class NodeEdges(GraphInput):
    view: Literal["node_edges"]
    node_id: Name
    direction: Literal["incoming", "outgoing", "both"] = "both"


class GraphQuery(WireModel):
    query: Annotated[QueryAll | QueryNode | NodeLookup | NodeEdges, Field(discriminator="view")]


class ImpactInput(GraphInput):
    node_id: Name
    impact_rules: (
        dict[
            Literal[
                "source",
                "table",
                "file",
                "stream",
                "transformation",
                "validation",
                "model",
                "report",
                "external",
                "virtual",
            ],
            Literal["none", "low", "medium", "high", "critical"],
        ]
        | None
    ) = None


class ImpactAnalyze(ImpactInput):
    max_depth: Annotated[int, Field(ge=-1, le=1000)] = -1
    include_validations: bool = True


class SchemaImpact(ImpactInput):
    new_schema: dict[Name, Name]


class DependencyChain(GraphInput):
    source_id: Name
    target_id: Name
