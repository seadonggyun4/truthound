"""Exact Core graph serialization, analysis and immutable-edit failure boundaries."""

import asyncio
import base64
import json
from datetime import datetime

import pytest

pytest.importorskip("mcp")

from truthound.lineage import (
    EdgeType,
    ImpactAnalyzer,
    LineageEdge,
    LineageGraph,
    LineageNode,
    NodeType,
)
from truthound.lineage.base import ColumnLineage, LineageMetadata
from truthound.mcp.policy import MCPPolicy
from truthound.mcp.service import MCPService
from truthound.mcp.tables import decode

pytestmark = pytest.mark.contract


def invoke(adapter, name, **arguments):
    return asyncio.run(adapter.call_operation_async(name, arguments))


def call(adapter, name, **arguments):
    result = invoke(adapter, name, **arguments)
    assert result["execution_status"] == "succeeded", result
    return result["data"]


def read(adapter, identity):
    chunks, offset = [], 0
    while offset is not None:
        chunk = call(adapter, "artifacts_read", resource_id=identity, offset=offset, limit=32768)
        chunks.append(base64.b64decode(chunk["base64"]))
        offset = chunk["next_offset"]
    return decode(json.loads(b"".join(chunks)))


def raw_artifact(adapter, identity):
    chunks, offset = [], 0
    while offset is not None:
        value = call(adapter, "artifacts_read", resource_id=identity, offset=offset, limit=32768)
        chunks.append(base64.b64decode(value["base64"]))
        offset = value["next_offset"]
    return b"".join(chunks)


@pytest.mark.parametrize("renderer", ["d3", "cytoscape", "graphviz", "mermaid"])
@pytest.mark.parametrize("subgraph", [False, True])
def test_actual_native_renderer_full_settings_and_subgraph_bytes(graph, renderer, subgraph):
    from truthound.lineage.visualization import RenderConfig, get_renderer

    adapter, identity, native = graph
    settings = {
        "theme": "dark",
        "orientation": "LR",
        "width": 640,
        "height": 480,
        "include_metadata": True,
        "highlight_nodes": ["b"],
        "max_depth": 1,
        "node_colors": {"source": "#abcdef"},
        "node_sizes": {"source": 17},
        "edge_colors": {"derived_from": "#654321"},
    }
    config = RenderConfig(**settings)
    target = get_renderer(renderer)
    expected = (
        target.render_subgraph(native, "a", direction="downstream", max_depth=1, config=config)
        if subgraph
        else target.render(native, config=config)
    )
    artifact = call(
        adapter,
        "lineage_render",
        graph_id=identity,
        renderer=renderer,
        settings=settings,
        **({"root_node_id": "a"} if subgraph else {}),
    )
    assert raw_artifact(adapter, artifact["resource_id"]) == expected.encode("utf-8")
    assert artifact["media_type"] == (
        "application/json" if renderer in {"d3", "cytoscape"} else "text/plain"
    )
    assert (
        read(
            adapter,
            call(adapter, "lineage_query", query={"graph_id": identity, "view": "graph"})[
                "resource_id"
            ],
        )
        == native.to_dict()
    )
    call(adapter, "resources_release", resource_id=identity)
    assert (
        invoke(adapter, "artifacts_read", resource_id=artifact["resource_id"])["execution_status"]
        == "failed"
    )


def test_native_file_reentry_export_bytes_and_revoked_origin(graph, tmp_path):
    from dataclasses import replace

    from truthound.lineage import LineageTracker

    adapter, _, native = graph
    source = tmp_path / "lineage.json"
    native.save(source)
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["nodes"][0]["metadata"]["description"] = "계보 재진입"
    source.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    loaded = call(adapter, "lineage_load", path=str(source))["resource_id"]
    artifact = call(adapter, "lineage_export", graph_id=loaded)["resource_id"]
    tracker = LineageTracker()
    tracker.load(str(source))
    expected = tracker.export_to_json().encode("utf-8")
    assert raw_artifact(adapter, artifact) == expected
    assert json.loads(raw_artifact(adapter, artifact)) == LineageGraph.load(source).to_dict()
    # Changes after admission cannot replace the owned graph snapshot.
    source.write_text('{"nodes":[],"edges":[]}', encoding="utf-8")
    assert raw_artifact(adapter, artifact) == expected
    reentry = tmp_path / "exported.json"
    reentry.write_bytes(expected)
    again = call(adapter, "lineage_load", path=str(reentry))["resource_id"]
    assert (
        raw_artifact(adapter, call(adapter, "lineage_export", graph_id=again)["resource_id"])
        == expected
    )
    denied_root = tmp_path / "new-grant"
    denied_root.mkdir()
    adapter.policy = replace(adapter.policy, allowed_directories=(denied_root,))
    assert invoke(adapter, "lineage_export", graph_id=loaded)["execution_status"] == "failed"
    assert invoke(adapter, "artifacts_read", resource_id=artifact)["execution_status"] == "failed"


@pytest.mark.parametrize(
    "content", ["", "{broken", '{"nodes":[{"id":"bad","node_type":"unknown"}]}']
)
def test_native_file_errors_are_not_empty_success(tmp_path, content):
    source = tmp_path / "bad.json"
    source.write_text(content, encoding="utf-8")
    from truthound.lineage import LineageError

    with pytest.raises((LineageError, ValueError)):
        LineageGraph.load(source)
    adapter = MCPService(MCPPolicy((tmp_path,), profile="core", allow_sensitive_results=True))
    try:
        result = invoke(adapter, "lineage_load", path=str(source))
        assert result["execution_status"] == "failed"
        assert result["data"] is None
    finally:
        adapter.close()


def test_load_disclosure_paths_and_cardinality_do_not_bypass_policy(tmp_path):
    path = tmp_path / "graph.json"
    path.write_text('{"nodes":[],"edges":[]}', encoding="utf-8")
    adapter = MCPService(MCPPolicy((tmp_path,), profile="core"))
    try:
        assert invoke(adapter, "lineage_load", path=str(path))["issues"] == [
            {"code": "approval_required"}
        ]
    finally:
        adapter.close()
    adapter = MCPService(MCPPolicy((tmp_path,), profile="core", allow_sensitive_results=True))
    try:
        for args in (
            {"path": "https://example.invalid/graph.json"},
            {"path": str(path), "config": {"persist_path": str(tmp_path / "unapproved.json")}},
        ):
            assert invoke(adapter, "lineage_load", **args)["execution_status"] == "failed"
        path.write_text(json.dumps({"nodes": [{}] * 1001}), encoding="utf-8")
        assert invoke(adapter, "lineage_load", path=str(path))["execution_status"] == "failed"
        assert not (tmp_path / "unapproved.json").exists()
    finally:
        adapter.close()


@pytest.mark.fault
def test_renderer_denies_missing_roots_external_programs_and_unknown_fields(graph):
    adapter, identity, native = graph
    for extra in (
        {"root_node_id": "absent"},
        {"renderer": "svg"},
        {"settings": {"output_path": "/not/authorized"}},
    ):
        failed = invoke(
            adapter,
            "lineage_render",
            **{
                "graph_id": identity,
                "renderer": "mermaid",
                **extra,
            },
        )
        assert failed["execution_status"] == "failed"
    assert (
        read(
            adapter,
            call(adapter, "lineage_query", query={"graph_id": identity, "view": "graph"})[
                "resource_id"
            ],
        )
        == native.to_dict()
    )


@pytest.mark.parametrize("direction", ["upstream", "downstream", "both"])
def test_tracker_queries_preserve_native_direction_and_missing_path(graph, direction):
    from truthound.lineage import LineageTracker

    adapter, identity, native = graph
    tracker = LineageTracker(native.config)
    for node in native.nodes:
        tracker.graph.add_node(node)
    for edge in native.edges:
        tracker.graph.add_edge(edge)
    result = call(
        adapter,
        "lineage_query",
        query={
            "graph_id": identity,
            "view": "tracked_lineage",
            "node_id": "b",
            "direction": direction,
        },
    )
    assert read(adapter, result["resource_id"]) == tracker.get_lineage("b", direction)
    for source, target in [("a", "b"), ("b", "a"), ("absent", "b"), ("a", "a")]:
        path = tracker.get_path(source, target)
        artifact = call(
            adapter,
            "lineage_query",
            query={
                "graph_id": identity,
                "view": "tracked_path",
                "source_id": source,
                "target_id": target,
            },
        )
        assert read(adapter, artifact["resource_id"]) == (
            None if path is None else [node.to_dict() for node in path]
        )
    missing = invoke(
        adapter,
        "lineage_query",
        query={
            "graph_id": identity,
            "view": "tracked_lineage",
            "node_id": "absent",
        },
    )
    assert missing["execution_status"] == "failed"
    call(adapter, "resources_release", resource_id=identity)
    assert (
        invoke(adapter, "artifacts_read", resource_id=result["resource_id"])["execution_status"]
        == "failed"
    )


@pytest.mark.fault
def test_tracker_dependency_rejects_cycle_without_mutating_parent(graph):
    from truthound.lineage import CyclicDependencyError, LineageTracker, NodeNotFoundError

    adapter, identity, native = graph
    original = native.to_dict()
    tracker = LineageTracker(native.config)
    for node in native.nodes:
        tracker.graph.add_node(node)
    for edge in native.edges:
        tracker.graph.add_edge(edge)
    for source, target, exception in [
        ("b", "a", CyclicDependencyError),
        ("absent", "b", NodeNotFoundError),
    ]:
        with pytest.raises(exception):
            tracker.add_dependency(source, target)
        failed = invoke(
            adapter,
            "lineage_track",
            graph_id=identity,
            event={
                "action": "dependency",
                "source_id": source,
                "target_id": target,
            },
        )
        assert failed["execution_status"] == "failed"
    assert (
        read(
            adapter,
            call(
                adapter,
                "lineage_query",
                query={
                    "graph_id": identity,
                    "view": "graph",
                },
            )["resource_id"],
        )
        == original
    )


def test_manual_native_tracking_chain_preserves_graphs_and_metadata(graph):
    import copy

    from truthound.lineage import LineageTracker

    adapter, identity, native = graph
    original = native.to_dict()
    tracker = LineageTracker(native.config)
    for node in native.nodes:
        tracker.graph.add_node(node)
    for edge in native.edges:
        tracker.graph.add_edge(edge)
    events = [
        {
            "action": "source",
            "name": "source-2",
            "source_type": "table",
            "location": "metadata-only://no-connection",
            "schema_spec": {"x": "Int64"},
            "owner": "fixture",
            "tags": ["test"],
            "properties": {"revision": 2},
        },
        {
            "action": "transformation",
            "name": "transformed",
            "sources": ["a", "source-2", "missing"],
            "operation": "join",
            "schema_spec": {"x": "Int64"},
            "column_mapping": {"x": [["source-2", "x"]]},
        },
        {
            "action": "validation",
            "name": "checked",
            "sources": ["transformed"],
            "validators": ["not_null"],
            "result_summary": {"count": 2},
        },
        {
            "action": "output",
            "name": "output",
            "sources": ["checked"],
            "output_type": "report",
            "location": "not-written.html",
        },
        {"action": "source", "name": "a", "description": "Native replacement semantics"},
        {
            "action": "dependency",
            "source_id": "a",
            "target_id": "output",
            "edge_type": "depends_on",
        },
    ]

    def semantics(value):
        value = copy.deepcopy(value)
        # Only native-generated timestamps differ between independent runs.
        for row in value["nodes"] + value["edges"]:
            for target in (row, row["metadata"]):
                for key in ("created_at", "updated_at"):
                    if key in target:
                        assert datetime.fromisoformat(target.pop(key))
        return value

    current = identity
    for event in events:
        args = {key: value for key, value in event.items() if key != "action"}
        args.update(args.pop("properties", {}))
        if "schema_spec" in args:
            args["schema"] = args.pop("schema_spec")
        if "edge_type" in args:
            args["edge_type"] = EdgeType(args["edge_type"])
        methods = {
            "source": tracker.track_source,
            "transformation": tracker.track_transformation,
            "validation": tracker.track_validation,
            "output": tracker.track_output,
            "dependency": tracker.add_dependency,
        }
        methods[event["action"]](**args)
        current = call(adapter, "lineage_track", graph_id=current, event=event)["resource_id"]
        actual = read(
            adapter,
            call(adapter, "lineage_query", query={"graph_id": current, "view": "graph"})[
                "resource_id"
            ],
        )
        assert semantics(actual) == semantics(tracker.graph.to_dict())
    assert (
        read(
            adapter,
            call(adapter, "lineage_query", query={"graph_id": identity, "view": "graph"})[
                "resource_id"
            ],
        )
        == original
    )
    # Parent ownership is retained transitively; releasing a source revokes all descendants.
    call(adapter, "resources_release", resource_id=identity)
    assert (
        invoke(adapter, "lineage_query", query={"graph_id": current, "view": "graph"})[
            "execution_status"
        ]
        == "failed"
    )


@pytest.mark.fault
def test_manual_tracking_rejects_argument_smuggling_and_preserves_parent(graph):
    adapter, identity, native = graph
    for event in (
        {"action": "source", "name": "new", "properties": {"schema": {"x": "Int64"}}},
        {"action": "execute", "name": "new"},
        {
            "action": "transformation",
            "name": "new",
            "sources": ["a"],
            "column_mapping": {"x": [["a"]]},
        },
    ):
        assert (
            invoke(adapter, "lineage_track", graph_id=identity, event=event)["execution_status"]
            == "failed"
        )
    assert (
        read(
            adapter,
            call(adapter, "lineage_query", query={"graph_id": identity, "view": "graph"})[
                "resource_id"
            ],
        )
        == native.to_dict()
    )


@pytest.fixture
def graph(tmp_path):
    adapter = MCPService(MCPPolicy((tmp_path,), profile="core", allow_sensitive_results=True))
    stamp = datetime(2026, 1, 1)
    temporal = {"value": stamp.isoformat()}
    metadata = {
        "description": "synthetic",
        "owner": "test",
        "tags": ["sample"],
        "properties": {"key": "value"},
        "created_at": temporal,
        "updated_at": temporal,
    }
    nodes, native = [], LineageGraph()
    for identity, kind in (("a", "source"), ("b", "transformation"), ("c", "validation")):
        columns = (
            []
            if identity == "a"
            else [
                {
                    "column": "x",
                    "source_columns": [["a", "x"]],
                    "transformation": "copy",
                    "dtype": "Int64",
                }
            ]
        )
        nodes.append(
            {
                "id": identity,
                "name": identity,
                "node_type": kind,
                "schema": {"x": "Int64"},
                "metadata": metadata,
                "column_lineage": columns,
                "created_at": temporal,
                "updated_at": temporal,
            }
        )
        native.add_node(
            LineageNode(
                identity,
                identity,
                NodeType(kind),
                schema={"x": "Int64"},
                metadata=LineageMetadata(
                    description="synthetic",
                    owner="test",
                    tags=("sample",),
                    properties={"key": "value"},
                    created_at=stamp,
                    updated_at=stamp,
                ),
                column_lineage=()
                if identity == "a"
                else (ColumnLineage("x", (("a", "x"),), "copy", "Int64"),),
                created_at=stamp,
                updated_at=stamp,
            )
        )
    edges = []
    for source, target in (("a", "b"), ("b", "c")):
        edges.append(
            {
                "source": source,
                "target": target,
                "edge_type": "derived_from",
                "metadata": metadata,
                "created_at": temporal,
            }
        )
        native.add_edge(
            LineageEdge(
                source,
                target,
                EdgeType.DERIVED_FROM,
                metadata=LineageMetadata(
                    description="synthetic",
                    owner="test",
                    tags=("sample",),
                    properties={"key": "value"},
                    created_at=stamp,
                    updated_at=stamp,
                ),
                created_at=stamp,
            )
        )
    identity = call(adapter, "lineage_create", nodes=nodes, edges=edges)["resource_id"]
    yield adapter, identity, native
    adapter.close()


@pytest.mark.parametrize(
    "view",
    [
        "graph",
        "nodes",
        "edges",
        "roots",
        "leaves",
        "topological",
        "node",
        "has_node",
        "upstream",
        "downstream",
        "node_edges",
    ],
)
def test_every_graph_query_reenters_native_core_with_complete_metadata(graph, view):
    adapter, identity, native = graph
    query = {"graph_id": identity, "view": view}
    if view in {"node", "has_node", "upstream", "downstream", "node_edges"}:
        query["node_id"] = "b"
    if view == "graph":
        expected = native.to_dict()
    elif view == "nodes":
        expected = [node.to_dict() for node in native.nodes]
    elif view == "edges":
        expected = [edge.to_dict() for edge in native.edges]
    elif view == "roots":
        expected = [node.to_dict() for node in native.get_roots()]
    elif view == "leaves":
        expected = [node.to_dict() for node in native.get_leaves()]
    elif view == "topological":
        expected = [node.to_dict() for node in native.topological_sort()]
    elif view == "node":
        expected = native.get_node("b").to_dict()
    elif view == "has_node":
        expected = native.has_node("b")
    elif view == "upstream":
        expected = [node.to_dict() for node in native.get_upstream("b")]
    elif view == "downstream":
        expected = [node.to_dict() for node in native.get_downstream("b")]
    else:
        expected = [edge.to_dict() for edge in native.get_edges_for_node("b")]
    result = call(adapter, "lineage_query", query=query)
    assert read(adapter, result["resource_id"]) == expected


def test_impact_paths_schema_changes_and_dependency_chain_match_core(graph):
    adapter, identity, native = graph
    analyzer = ImpactAnalyzer(native)
    expected = analyzer.analyze_impact("a", max_depth=2, include_validations=False).to_dict()
    actual = read(
        adapter,
        call(
            adapter,
            "lineage_impact",
            graph_id=identity,
            node_id="a",
            max_depth=2,
            include_validations=False,
        )["resource_id"],
    )
    for field in ("analysis_time_ms", "analyzed_at"):
        actual.pop(field)
        expected.pop(field)
    assert actual == expected
    expected = [[node.to_dict() for node in path] for path in analyzer.find_critical_paths("a")]
    assert (
        read(
            adapter,
            call(adapter, "lineage_critical_paths", graph_id=identity, node_id="a")["resource_id"],
        )
        == expected
    )
    assert read(
        adapter,
        call(adapter, "lineage_what_if_delete", graph_id=identity, node_id="a")["resource_id"],
    ) == analyzer.what_if_delete("a")
    assert read(
        adapter,
        call(
            adapter,
            "lineage_schema_impact",
            graph_id=identity,
            node_id="a",
            new_schema={"y": "String"},
        )["resource_id"],
    ) == analyzer.compare_schemas("a", {"y": "String"})
    assert read(
        adapter,
        call(adapter, "lineage_dependency_chain", graph_id=identity, source_id="a", target_id="c")[
            "resource_id"
        ],
    ) == [node.to_dict() for node in analyzer.get_dependency_chain("a", "c")]


def test_edits_are_atomic_derivations_and_cycles_cannot_corrupt_original(graph):
    adapter, identity, native = graph
    before = call(adapter, "resources_describe", resource_id=identity)["sha256"]
    failed = invoke(
        adapter,
        "lineage_edit",
        graph_id=identity,
        edit={
            "action": "add_edge",
            "edge": {"source": "c", "target": "a", "edge_type": "derived_from"},
        },
    )
    assert failed["execution_status"] == "failed" and failed["quality_status"] == "unknown"
    assert call(adapter, "resources_describe", resource_id=identity)["sha256"] == before
    derived = call(
        adapter, "lineage_edit", graph_id=identity, edit={"action": "remove_node", "node_id": "b"}
    )["resource_id"]
    native.remove_node("b")
    assert (
        read(
            adapter,
            call(adapter, "lineage_query", query={"graph_id": derived, "view": "graph"})[
                "resource_id"
            ],
        )
        == native.to_dict()
    )
    call(adapter, "resources_release", resource_id=identity)
    assert invoke(adapter, "lineage_query", query={"graph_id": derived, "view": "graph"})[
        "issues"
    ] == [{"code": "invalid_handle"}]
    assert invoke(adapter, "lineage_create", config={"persist_path": "/etc/out.json"})[
        "issues"
    ] == [{"code": "operator_configuration_required"}]
