"""Native lineage artifacts must preserve graph identities and treat labels as data."""

from html.parser import HTMLParser

import pytest

from truthound.lineage import LineageGraph, LineageNode, NodeType
from truthound.lineage.visualization import CytoscapeRenderer, D3Renderer, MermaidRenderer


class HTMLStructure(HTMLParser):
    def __init__(self):
        super().__init__()
        self.elements = []

    def handle_starttag(self, tag, attrs):
        self.elements.append((tag, dict(attrs)))


@pytest.mark.parametrize("renderer", [D3Renderer, CytoscapeRenderer, MermaidRenderer])
def test_native_html_cannot_turn_graph_labels_into_active_elements(renderer):
    payload = '</script><img src=x onerror="synthetic_untrusted()"><script>'
    graph = LineageGraph()
    graph.add_node(LineageNode(id="synthetic", name=payload, node_type=NodeType.TABLE))
    document = renderer().render_html(graph)
    parsed = HTMLStructure()
    parsed.feed(document)
    assert not any(
        tag == "img" or any(key.startswith("on") for key in attrs) for tag, attrs in parsed.elements
    )


@pytest.mark.parametrize("renderer_name", ["graphviz", "mermaid"])
def test_native_diagrams_do_not_merge_distinct_node_identities(renderer_name):
    from truthound.lineage.visualization import get_renderer

    graph = LineageGraph()
    for identifier in ("a-b", "a_b", "end", "graph", "truthound_id_612d62"):
        graph.add_node(LineageNode(id=identifier, name="same", node_type=NodeType.TABLE))
    output = get_renderer(renderer_name).render(graph)
    import re

    if renderer_name == "graphviz":
        identifiers = re.findall(r"^\s*(\S+)\s+\[label=", output, re.MULTILINE)
    else:
        identifiers = re.findall(r"^\s*(\S+)\[same\]", output, re.MULTILINE)
    assert len(identifiers) == 5, output
    assert len(set(identifiers)) == 5, output


@pytest.mark.parametrize(
    "settings",
    [
        {"orientation": 'TB; injected [image="unapproved.png"]'},
        {"width": 0},
        {"width": True},
        {"include_metadata": 1},
        {"node_sizes": {"table": -1}},
        {"filter_node_types": "table"},
    ],
)
def test_native_render_configuration_rejects_malformed_controls(settings):
    from truthound.lineage.visualization import RenderConfig

    with pytest.raises(ValueError):
        RenderConfig(**settings)


@pytest.mark.parametrize("renderer_name", ["d3", "cytoscape", "graphviz", "mermaid"])
def test_native_renderer_does_not_mutate_the_callers_configuration(renderer_name):
    from truthound.lineage.visualization import RenderConfig, get_renderer

    config = RenderConfig(theme="light")
    get_renderer(renderer_name, theme="dark").render(LineageGraph(), config)
    assert config.theme == "light"


def test_graphviz_declared_image_dimensions_change_the_native_drawing():
    from truthound.lineage.visualization import GraphvizRenderer, RenderConfig

    graph = LineageGraph()
    graph.add_node(LineageNode(id="a", name="same", node_type=NodeType.TABLE))
    assert GraphvizRenderer().render(graph, RenderConfig(width=64, height=64)) != (
        GraphvizRenderer().render(graph, RenderConfig(width=640, height=480))
    )


@pytest.mark.parametrize("layout", ["grid", "circular", "hierarchical"])
def test_d3_layouts_provide_distinct_positions_inside_the_requested_canvas(layout):
    import json

    from truthound.lineage import EdgeType, LineageEdge
    from truthound.lineage.visualization import RenderConfig

    graph = LineageGraph()
    for identifier in ("a", "b", "c"):
        graph.add_node(LineageNode(id=identifier, name=identifier, node_type=NodeType.TABLE))
    graph.add_edge(LineageEdge(source="a", target="b", edge_type=EdgeType.DERIVED_FROM))
    graph.add_edge(LineageEdge(source="b", target="c", edge_type=EdgeType.DERIVED_FROM))
    nodes = json.loads(
        D3Renderer().render(
            graph, RenderConfig(layout=layout, width=640, height=480, orientation="LR")
        )
    )["nodes"]
    assert len({(node["fx"], node["fy"]) for node in nodes}) == 3
    assert all(0 <= node["fx"] <= 640 and 0 <= node["fy"] <= 480 for node in nodes)
    if layout == "hierarchical":
        assert nodes[0]["fx"] < nodes[1]["fx"] < nodes[2]["fx"]


def test_cytoscape_uses_selected_node_sizes_instead_of_a_fixed_size():
    import json

    from truthound.lineage.visualization import RenderConfig

    graph = LineageGraph()
    graph.add_node(LineageNode(id="a", name="a", node_type=NodeType.TABLE))
    data = json.loads(CytoscapeRenderer().render(graph, RenderConfig(node_sizes={"table": 42})))
    assert data["elements"][0]["data"]["size"] == 42
    assert data["style"][0]["style"]["width"] == "data(size)"


@pytest.mark.parametrize("renderer_name", ["d3", "cytoscape", "graphviz", "mermaid"])
def test_native_subgraph_uses_configured_depth_unless_explicitly_overridden(renderer_name):
    import json
    import re

    from truthound.lineage import EdgeType, LineageEdge
    from truthound.lineage.visualization import RenderConfig, get_renderer

    graph = LineageGraph()
    for identifier in ("a", "b", "c"):
        graph.add_node(LineageNode(id=identifier, name=identifier, node_type=NodeType.TABLE))
    for source, target in [("a", "b"), ("b", "c")]:
        graph.add_edge(LineageEdge(source=source, target=target, edge_type=EdgeType.DERIVED_FROM))
    renderer = get_renderer(renderer_name)
    limited = renderer.render_subgraph(graph, "a", config=RenderConfig(max_depth=0))
    complete = renderer.render_subgraph(graph, "a", config=RenderConfig(max_depth=0), max_depth=-1)

    def identities(text):
        if renderer_name == "d3":
            return {node["id"] for node in json.loads(text)["nodes"]}
        if renderer_name == "cytoscape":
            return {
                element["data"]["id"]
                for element in json.loads(text)["elements"]
                if element["group"] == "nodes"
            }
        if renderer_name == "graphviz":
            return set(re.findall(r"^\s*(\S+)\s+\[label=", text, re.MULTILINE))
        return set(re.findall(r"^\s*(\S+)\[[abc]\]", text, re.MULTILINE))

    assert identities(limited) == {"a"}
    assert identities(complete) == {"a", "b", "c"}
