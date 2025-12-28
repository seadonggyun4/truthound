"""Tests for dependency graph."""

import pytest

from truthound.plugins.dependencies.graph import (
    DependencyGraph,
    DependencyNode,
    DependencyType,
)


class TestDependencyNode:
    """Tests for DependencyNode."""

    def test_create_node(self):
        """Test creating a dependency node."""
        node = DependencyNode(
            plugin_id="my-plugin",
            version="1.0.0",
        )

        assert node.plugin_id == "my-plugin"
        assert node.version == "1.0.0"
        assert node.dependencies == {}
        assert node.reverse_dependencies == set()

    def test_add_dependency(self):
        """Test adding a dependency."""
        node = DependencyNode(plugin_id="plugin-a", version="1.0.0")

        node.add_dependency("plugin-b", DependencyType.REQUIRED)

        assert "plugin-b" in node.dependencies
        assert node.dependencies["plugin-b"] == DependencyType.REQUIRED

    def test_add_reverse_dependency(self):
        """Test adding a reverse dependency."""
        node = DependencyNode(plugin_id="plugin-a", version="1.0.0")

        node.add_reverse_dependency("plugin-c")

        assert "plugin-c" in node.reverse_dependencies

    def test_required_dependencies_property(self):
        """Test required_dependencies property."""
        node = DependencyNode(
            plugin_id="plugin-a",
            version="1.0.0",
            dependencies={
                "plugin-b": DependencyType.REQUIRED,
                "plugin-c": DependencyType.OPTIONAL,
                "plugin-d": DependencyType.REQUIRED,
            },
        )

        required = node.required_dependencies

        assert "plugin-b" in required
        assert "plugin-d" in required
        assert "plugin-c" not in required

    def test_optional_dependencies_property(self):
        """Test optional_dependencies property."""
        node = DependencyNode(
            plugin_id="plugin-a",
            version="1.0.0",
            dependencies={
                "plugin-b": DependencyType.REQUIRED,
                "plugin-c": DependencyType.OPTIONAL,
            },
        )

        optional = node.optional_dependencies

        assert "plugin-c" in optional
        assert "plugin-b" not in optional


class TestDependencyGraph:
    """Tests for DependencyGraph."""

    def test_create_empty_graph(self):
        """Test creating empty graph."""
        graph = DependencyGraph()

        assert len(graph) == 0

    def test_add_node(self):
        """Test adding a node."""
        graph = DependencyGraph()

        node = graph.add_node("plugin-a", "1.0.0")

        assert node is not None
        assert len(graph) == 1
        assert "plugin-a" in graph

    def test_add_node_with_dependencies(self):
        """Test adding node with dependencies."""
        graph = DependencyGraph()

        graph.add_node("plugin-b", "1.0.0")
        graph.add_node(
            "plugin-a",
            "1.0.0",
            dependencies={"plugin-b": DependencyType.REQUIRED},
        )

        node_a = graph.get_node("plugin-a")
        node_b = graph.get_node("plugin-b")

        assert "plugin-b" in node_a.dependencies
        assert "plugin-a" in node_b.reverse_dependencies

    def test_remove_node(self):
        """Test removing a node."""
        graph = DependencyGraph()
        graph.add_node("plugin-a", "1.0.0")

        result = graph.remove_node("plugin-a")

        assert result is True
        assert "plugin-a" not in graph

    def test_remove_nonexistent_node(self):
        """Test removing nonexistent node."""
        graph = DependencyGraph()

        result = graph.remove_node("nonexistent")

        assert result is False

    def test_get_node(self):
        """Test getting a node."""
        graph = DependencyGraph()
        graph.add_node("plugin-a", "1.0.0")

        node = graph.get_node("plugin-a")

        assert node is not None
        assert node.plugin_id == "plugin-a"

    def test_get_nonexistent_node(self):
        """Test getting nonexistent node."""
        graph = DependencyGraph()

        node = graph.get_node("nonexistent")

        assert node is None

    def test_has_node(self):
        """Test checking node existence."""
        graph = DependencyGraph()
        graph.add_node("plugin-a", "1.0.0")

        assert graph.has_node("plugin-a") is True
        assert graph.has_node("nonexistent") is False


class TestDependencyGraphQueries:
    """Tests for dependency graph queries."""

    def test_get_dependencies_direct(self):
        """Test getting direct dependencies."""
        graph = DependencyGraph()
        graph.add_node("plugin-b", "1.0.0")
        graph.add_node("plugin-c", "1.0.0")
        graph.add_node(
            "plugin-a",
            "1.0.0",
            dependencies={
                "plugin-b": DependencyType.REQUIRED,
                "plugin-c": DependencyType.REQUIRED,
            },
        )

        deps = graph.get_dependencies("plugin-a")

        assert "plugin-b" in deps
        assert "plugin-c" in deps

    def test_get_dependencies_recursive(self):
        """Test getting transitive dependencies."""
        graph = DependencyGraph()
        graph.add_node("plugin-c", "1.0.0")
        graph.add_node(
            "plugin-b",
            "1.0.0",
            dependencies={"plugin-c": DependencyType.REQUIRED},
        )
        graph.add_node(
            "plugin-a",
            "1.0.0",
            dependencies={"plugin-b": DependencyType.REQUIRED},
        )

        deps = graph.get_dependencies("plugin-a", recursive=True)

        assert "plugin-b" in deps
        assert "plugin-c" in deps  # Transitive

    def test_get_dependencies_excludes_optional(self):
        """Test optional deps are excluded by default."""
        graph = DependencyGraph()
        graph.add_node("plugin-b", "1.0.0")
        graph.add_node("plugin-c", "1.0.0")
        graph.add_node(
            "plugin-a",
            "1.0.0",
            dependencies={
                "plugin-b": DependencyType.REQUIRED,
                "plugin-c": DependencyType.OPTIONAL,
            },
        )

        deps = graph.get_dependencies("plugin-a")

        assert "plugin-b" in deps
        assert "plugin-c" not in deps

    def test_get_dependencies_includes_optional(self):
        """Test including optional dependencies."""
        graph = DependencyGraph()
        graph.add_node("plugin-b", "1.0.0")
        graph.add_node("plugin-c", "1.0.0")
        graph.add_node(
            "plugin-a",
            "1.0.0",
            dependencies={
                "plugin-b": DependencyType.REQUIRED,
                "plugin-c": DependencyType.OPTIONAL,
            },
        )

        deps = graph.get_dependencies("plugin-a", include_optional=True)

        assert "plugin-b" in deps
        assert "plugin-c" in deps

    def test_get_dependents(self):
        """Test getting dependents."""
        graph = DependencyGraph()
        graph.add_node("plugin-a", "1.0.0")
        graph.add_node(
            "plugin-b",
            "1.0.0",
            dependencies={"plugin-a": DependencyType.REQUIRED},
        )
        graph.add_node(
            "plugin-c",
            "1.0.0",
            dependencies={"plugin-a": DependencyType.REQUIRED},
        )

        dependents = graph.get_dependents("plugin-a")

        assert "plugin-b" in dependents
        assert "plugin-c" in dependents


class TestDependencyGraphCycles:
    """Tests for cycle detection."""

    def test_no_cycles(self):
        """Test graph with no cycles."""
        graph = DependencyGraph()
        graph.add_node("plugin-c", "1.0.0")
        graph.add_node(
            "plugin-b",
            "1.0.0",
            dependencies={"plugin-c": DependencyType.REQUIRED},
        )
        graph.add_node(
            "plugin-a",
            "1.0.0",
            dependencies={"plugin-b": DependencyType.REQUIRED},
        )

        cycles = graph.detect_cycles()

        assert len(cycles) == 0

    def test_simple_cycle(self):
        """Test detecting simple cycle."""
        graph = DependencyGraph()
        graph.add_node(
            "plugin-a",
            "1.0.0",
            dependencies={"plugin-b": DependencyType.REQUIRED},
        )
        graph.add_node(
            "plugin-b",
            "1.0.0",
            dependencies={"plugin-a": DependencyType.REQUIRED},
        )

        cycles = graph.detect_cycles()

        assert len(cycles) > 0


class TestDependencyGraphOrdering:
    """Tests for load/unload ordering."""

    def test_load_order_simple(self):
        """Test simple load order."""
        graph = DependencyGraph()
        graph.add_node("plugin-c", "1.0.0")
        graph.add_node(
            "plugin-b",
            "1.0.0",
            dependencies={"plugin-c": DependencyType.REQUIRED},
        )
        graph.add_node(
            "plugin-a",
            "1.0.0",
            dependencies={"plugin-b": DependencyType.REQUIRED},
        )

        order = graph.get_load_order()

        # C must come before B, B must come before A
        assert order.index("plugin-c") < order.index("plugin-b")
        assert order.index("plugin-b") < order.index("plugin-a")

    def test_load_order_with_cycle_raises(self):
        """Test load order with cycle raises error."""
        graph = DependencyGraph()
        graph.add_node(
            "plugin-a",
            "1.0.0",
            dependencies={"plugin-b": DependencyType.REQUIRED},
        )
        graph.add_node(
            "plugin-b",
            "1.0.0",
            dependencies={"plugin-a": DependencyType.REQUIRED},
        )

        with pytest.raises(ValueError, match="Circular dependency"):
            graph.get_load_order()

    def test_unload_order(self):
        """Test unload order is reverse of load order."""
        graph = DependencyGraph()
        graph.add_node("plugin-b", "1.0.0")
        graph.add_node(
            "plugin-a",
            "1.0.0",
            dependencies={"plugin-b": DependencyType.REQUIRED},
        )

        load_order = graph.get_load_order()
        unload_order = graph.get_unload_order()

        assert unload_order == list(reversed(load_order))


class TestDependencyGraphSerialization:
    """Tests for graph serialization."""

    def test_to_dict(self):
        """Test converting graph to dictionary."""
        graph = DependencyGraph()
        graph.add_node("plugin-b", "1.0.0")
        graph.add_node(
            "plugin-a",
            "1.0.0",
            dependencies={"plugin-b": DependencyType.REQUIRED},
            metadata={"author": "test"},
        )

        data = graph.to_dict()

        assert "nodes" in data
        assert "plugin-a" in data["nodes"]
        assert "plugin-b" in data["nodes"]
        assert data["nodes"]["plugin-a"]["dependencies"]["plugin-b"] == "required"

    def test_from_dict(self):
        """Test creating graph from dictionary."""
        data = {
            "nodes": {
                "plugin-a": {
                    "version": "1.0.0",
                    "dependencies": {"plugin-b": "required"},
                    "metadata": {},
                },
                "plugin-b": {
                    "version": "1.0.0",
                    "dependencies": {},
                    "metadata": {},
                },
            },
        }

        graph = DependencyGraph.from_dict(data)

        assert len(graph) == 2
        assert "plugin-a" in graph
        assert "plugin-b" in graph

    def test_validate_missing_dependency(self):
        """Test validation detects missing dependency."""
        graph = DependencyGraph()
        graph.add_node(
            "plugin-a",
            "1.0.0",
            dependencies={"plugin-missing": DependencyType.REQUIRED},
        )

        errors = graph.validate()

        assert len(errors) > 0
        assert any("missing" in e.lower() for e in errors)
