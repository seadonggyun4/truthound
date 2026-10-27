"""Core graph and analyzer bindings; JSON state only in owned session files."""

import json
from dataclasses import asdict
from pathlib import Path

from truthound.mcp.encoding import encode
from truthound.mcp.policy import MCPError
from truthound.mcp.resources import BoundedWriter
from truthound.mcp.validator_wire import _known_type, contracts, convert


def _tracker(graph):
    tracker = _known_type("truthound.lineage.tracker.LineageTracker")(graph.config)
    for node in graph.nodes:
        tracker.graph.add_node(node)
    for edge in graph.edges:
        tracker.graph.add_edge(edge)
    return tracker


def apply_tracking_event(tracker, event):
    values = event.model_dump(exclude={"action"}, exclude_none=True)
    values.update(values.pop("properties", {}))
    if "schema_spec" in values:
        values["schema"] = values.pop("schema_spec")
    if event.action == "source":
        tracker.track_source(**values)
    elif event.action == "transformation":
        if values.get("row_mapping") is not None:
            values["row_mapping"] = {
                item["row"]: [(source["node"], source["row"]) for source in item["sources"]]
                for item in values["row_mapping"]
            }
        tracker.track_transformation(**values)
    elif event.action == "validation":
        tracker.track_validation(**values)
    elif event.action == "output":
        tracker.track_output(**values)
    elif event.action == "dependency":
        from truthound.lineage import EdgeType

        tracker.add_dependency(event.source_id, event.target_id, EdgeType(event.edge_type))


def render_media_type(arguments):
    formats = {
        "html": "text/html",
        "markdown": "text/markdown",
        "svg": "image/svg+xml",
        "png": "image/png",
    }
    return formats.get(
        arguments.output_format,
        "application/json" if arguments.renderer in {"d3", "cytoscape"} else "text/plain",
    )


def render_artifact(graph, arguments):
    definitions = contracts()["lineage"]
    renderer = _known_type(definitions["renderers"][arguments.renderer])()
    config = convert(definitions["render_config"], arguments.settings, build=True)
    media_type = "application/json" if arguments.renderer in {"d3", "cytoscape"} else "text/plain"
    if arguments.output_format == "html":
        text = renderer.render_html(graph, config=config)
        media_type = "text/html"
    elif arguments.output_format == "markdown":
        text = renderer.render_markdown(
            graph,
            config=config,
            title=arguments.title if arguments.title is not None else "Lineage Graph",
        )
        media_type = "text/markdown"
    elif arguments.output_format in {"svg", "png"}:
        render = renderer.render_svg if arguments.output_format == "svg" else renderer.render_png
        text = render(graph, config=config)
        media_type = "image/svg+xml" if arguments.output_format == "svg" else "image/png"
    elif arguments.root_node_id is None:
        text = renderer.render(graph, config=config)
    else:
        graph.get_node(arguments.root_node_id)
        text = renderer.render_subgraph(
            graph,
            arguments.root_node_id,
            direction=arguments.direction,
            max_depth=config.max_depth,
            config=config,
        )
    return text if isinstance(text, bytes) else text.encode("utf-8"), media_type


def execute(request, arguments):
    from truthound.lineage import ImpactLevel, NodeType

    graph_type = _known_type("truthound.lineage.base.LineageGraph")
    analyzer_type = _known_type("truthound.lineage.impact_analysis.ImpactAnalyzer")
    operation = request["operation"]
    definitions = contracts()["lineage"]
    if operation == "lineage_load":
        # Bound cardinality before native graph construction. The owned input
        # snapshot already passed file size, regular-file and grant checks.
        document = json.loads(Path(request["path"]).read_text(encoding="utf-8"))
        if isinstance(document, dict) and (
            len(document.get("nodes", [])) > 1000 or len(document.get("edges", [])) > 2000
        ):
            raise MCPError("resource_exhausted")
        graph = graph_type.load(
            request["path"], convert(definitions["config"], arguments.config, build=True)
        )
    elif operation == "lineage_create":
        graph = graph_type(convert(definitions["config"], arguments.config, build=True))
        for node in arguments.nodes:
            graph.add_node(convert(definitions["node"], node, build=True))
        for edge in arguments.edges:
            graph.add_edge(convert(definitions["edge"], edge, build=True))
    else:
        saved = json.loads(Path(request["graph_path"]).read_text(encoding="utf-8"))
        graph = graph_type.from_dict(
            saved["graph"], convert(definitions["config"], saved["config"], build=True)
        )
    if operation == "lineage_edit":
        edit = arguments.edit
        if edit.action == "add_node":
            graph.add_node(convert(definitions["node"], edit.node, build=True))
        elif edit.action == "add_edge":
            graph.add_edge(convert(definitions["edge"], edit.edge, build=True))
        elif edit.action == "remove_node":
            graph.remove_node(edit.node_id)
        elif edit.action == "remove_edge":
            graph.remove_edge(edit.source, edit.target)
        elif edit.action == "clear":
            graph.clear()
    if operation == "lineage_track":
        tracker = _tracker(graph)
        apply_tracking_event(tracker, arguments.event)
        graph = tracker.graph
    if operation == "lineage_export":
        with BoundedWriter(Path(request["output_path"]), request["output_budget"]) as sink:
            sink.write(_tracker(graph).export_to_json().encode("utf-8"))
        return {
            "resource_created": True,
            "media_type": "application/json",
            "row_count": None,
            "column_count": None,
        }
    if operation == "lineage_render":
        content, media_type = render_artifact(graph, arguments)
        with BoundedWriter(Path(request["output_path"]), request["output_budget"]) as sink:
            sink.write(content)
        return {
            "resource_created": True,
            "media_type": media_type,
            "row_count": None,
            "column_count": None,
        }
    if operation in {"lineage_create", "lineage_edit", "lineage_track", "lineage_load"}:
        result = {"graph": graph.to_dict(), "config": asdict(graph.config)}
        media_type = "application/vnd.truthound.lineage-graph+json"
    else:
        if operation == "lineage_query":
            query = arguments.query
            if query.view == "graph":
                result = graph.to_dict()
            elif query.view == "nodes":
                result = [node.to_dict() for node in graph.nodes]
            elif query.view == "edges":
                result = [edge.to_dict() for edge in graph.edges]
            elif query.view == "roots":
                result = [node.to_dict() for node in graph.get_roots()]
            elif query.view == "leaves":
                result = [node.to_dict() for node in graph.get_leaves()]
            elif query.view == "topological":
                result = [node.to_dict() for node in graph.topological_sort()]
            elif query.view == "node":
                result = graph.get_node(query.node_id).to_dict()
            elif query.view == "has_node":
                result = graph.has_node(query.node_id)
            elif query.view == "tracked_lineage":
                result = _tracker(graph).get_lineage(query.node_id, query.direction)
            elif query.view == "tracked_path":
                path = _tracker(graph).get_path(query.source_id, query.target_id)
                result = None if path is None else [node.to_dict() for node in path]
            elif query.view == "upstream":
                result = [node.to_dict() for node in graph.get_upstream(query.node_id, query.depth)]
            elif query.view == "downstream":
                result = [
                    node.to_dict() for node in graph.get_downstream(query.node_id, query.depth)
                ]
            else:
                result = [
                    edge.to_dict()
                    for edge in graph.get_edges_for_node(query.node_id, query.direction)
                ]
        else:
            impact_rules = None
            if getattr(arguments, "impact_rules", None) is not None:
                impact_rules = {
                    NodeType(key): ImpactLevel(value)
                    for key, value in arguments.impact_rules.items()
                }
            analyzer = analyzer_type(graph, impact_rules=impact_rules)
            if operation == "lineage_impact":
                impact = analyzer.analyze_impact(
                    arguments.node_id, arguments.max_depth, arguments.include_validations
                )
                if arguments.view == "full":
                    result = impact.to_dict()
                elif arguments.view == "summary":
                    result = impact.summary()
                elif arguments.view == "critical":
                    result = [node.to_dict() for node in impact.get_critical_nodes()]
                elif arguments.view == "by_level":
                    result = [
                        node.to_dict() for node in impact.get_by_level(ImpactLevel(arguments.level))
                    ]
                else:
                    result = [
                        node.to_dict() for node in impact.get_by_type(NodeType(arguments.node_type))
                    ]
            elif operation == "lineage_critical_paths":
                result = [
                    [node.to_dict() for node in path]
                    for path in analyzer.find_critical_paths(arguments.node_id)
                ]
            elif operation == "lineage_what_if_delete":
                result = analyzer.what_if_delete(arguments.node_id)
            elif operation == "lineage_schema_impact":
                result = analyzer.compare_schemas(arguments.node_id, arguments.new_schema)
            elif operation == "lineage_dependency_chain":
                chain = analyzer.get_dependency_chain(arguments.source_id, arguments.target_id)
                result = [node.to_dict() for node in chain] if chain is not None else None
            else:
                raise MCPError("unknown_operation")
        result = encode(result)
        media_type = "application/vnd.truthound.lineage-result+json"
    with BoundedWriter(Path(request["output_path"]), request["output_budget"]) as sink:
        sink.write(json.dumps(result, allow_nan=False).encode())
    return {
        "resource_created": True,
        "media_type": media_type,
        "row_count": None,
        "column_count": None,
    }
