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


class GraphLoad(WireModel):
    path: Annotated[str, Field(min_length=1, max_length=4096)]
    config: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def constructor(self):
        if self.config.get("persist_path") is not None:
            raise MCPError("operator_configuration_required")
        lineage_fields("config", self.config)
        return self


class TrackingMetadata(WireModel):
    description: Annotated[str, Field(max_length=4096)] | None = None
    owner: Annotated[str, Field(max_length=256)] = ""
    tags: Annotated[list[Name], Field(max_length=64)] = Field(default_factory=list)
    properties: Annotated[dict[Name, JsonValue], Field(max_length=64)] = Field(default_factory=dict)

    @model_validator(mode="after")
    def metadata_does_not_override_arguments(self):
        if set(self.properties) & (set(type(self).model_fields) | {"schema"}):
            raise MCPError("invalid_arguments")
        return self


class TrackSource(TrackingMetadata):
    action: Literal["source"]
    name: Name
    source_type: Name = "file"
    location: Annotated[str, Field(max_length=4096)] = ""
    schema_spec: dict[Name, Name] | None = None


class TrackTransformation(TrackingMetadata):
    action: Literal["transformation"]
    name: Name
    sources: Annotated[list[Name], Field(max_length=1000)]
    operation: Name = "transform"
    location: Annotated[str, Field(max_length=4096)] = ""
    schema_spec: dict[Name, Name] | None = None
    column_mapping: (
        dict[Name, list[Annotated[list[Name], Field(min_length=2, max_length=2)]]] | None
    ) = None


class TrackValidation(TrackingMetadata):
    action: Literal["validation"]
    name: Name
    sources: Annotated[list[Name], Field(max_length=1000)]
    validators: Annotated[list[Name], Field(max_length=260)] | None = None
    result_summary: dict[Name, JsonValue] | None = None


class TrackOutput(TrackingMetadata):
    action: Literal["output"]
    name: Name
    sources: Annotated[list[Name], Field(max_length=1000)]
    output_type: Name = "report"
    location: Annotated[str, Field(max_length=4096)] = ""


class TrackDependency(WireModel):
    action: Literal["dependency"]
    source_id: Name
    target_id: Name
    edge_type: Literal[
        "derived_from",
        "validated_by",
        "used_by",
        "transformed_to",
        "joined_with",
        "aggregated_to",
        "filtered_to",
        "depends_on",
    ] = "depends_on"


class GraphTrack(GraphInput):
    event: Annotated[
        TrackSource | TrackTransformation | TrackValidation | TrackOutput | TrackDependency,
        Field(discriminator="action"),
    ]


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


class TrackerLineage(GraphInput):
    view: Literal["tracked_lineage"]
    node_id: Name
    direction: Literal["upstream", "downstream", "both"] = "both"


class TrackerPath(GraphInput):
    view: Literal["tracked_path"]
    source_id: Name
    target_id: Name


class GraphQuery(WireModel):
    query: Annotated[
        QueryAll | QueryNode | NodeLookup | NodeEdges | TrackerLineage | TrackerPath,
        Field(discriminator="view"),
    ]


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
