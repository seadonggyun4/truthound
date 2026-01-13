"""Tests for engine registry module."""

import pytest
from truthound.datadocs.engine.registry import (
    ComponentRegistry,
    register_transformer,
    register_renderer,
    register_theme,
    register_exporter,
)


class TestComponentRegistry:
    """Tests for ComponentRegistry class."""

    def test_register_transformer(self):
        """Test registering a transformer."""
        registry = ComponentRegistry()

        class MyTransformer:
            pass

        registry.register_transformer("my_transformer", MyTransformer)
        # get_transformer returns an instance, not the class
        result = registry.get_transformer("my_transformer")
        assert isinstance(result, MyTransformer)

    def test_register_renderer(self):
        """Test registering a renderer."""
        registry = ComponentRegistry()

        class MyRenderer:
            pass

        registry.register_renderer("my_renderer", MyRenderer)
        # get_renderer returns an instance, not the class
        result = registry.get_renderer("my_renderer")
        assert isinstance(result, MyRenderer)

    def test_register_theme(self):
        """Test registering a theme."""
        registry = ComponentRegistry()

        class MyTheme:
            pass

        registry.register_theme("my_theme", MyTheme)
        # get_theme returns an instance, not the class
        result = registry.get_theme("my_theme")
        assert isinstance(result, MyTheme)

    def test_register_exporter(self):
        """Test registering an exporter."""
        registry = ComponentRegistry()

        class MyExporter:
            pass

        registry.register_exporter("my_exporter", MyExporter)
        # get_exporter returns an instance, not the class
        result = registry.get_exporter("my_exporter")
        assert isinstance(result, MyExporter)

    def test_get_nonexistent_raises(self):
        """Test that getting nonexistent component raises."""
        registry = ComponentRegistry()
        with pytest.raises(KeyError):
            registry.get_transformer("nonexistent")

    def test_list_transformers(self):
        """Test listing transformers."""
        registry = ComponentRegistry()
        registry.register_transformer("a", object)
        registry.register_transformer("b", object)
        names = registry.list_transformers()
        assert "a" in names
        assert "b" in names

    def test_list_renderers(self):
        """Test listing renderers."""
        registry = ComponentRegistry()
        registry.register_renderer("html", object)
        names = registry.list_renderers()
        assert "html" in names

    def test_list_themes(self):
        """Test listing themes."""
        registry = ComponentRegistry()
        registry.register_theme("dark", object)
        names = registry.list_themes()
        assert "dark" in names

    def test_list_exporters(self):
        """Test listing exporters."""
        registry = ComponentRegistry()
        registry.register_exporter("pdf", object)
        names = registry.list_exporters()
        assert "pdf" in names

    def test_has_methods(self):
        """Test has_* methods."""
        registry = ComponentRegistry()
        registry.register_transformer("test", object)
        assert registry.has_transformer("test")
        assert not registry.has_transformer("nonexistent")


class TestRegistryDecorators:
    """Tests for registry decorators."""

    def test_register_transformer_decorator(self):
        """Test transformer registration decorator."""
        registry = ComponentRegistry()

        @register_transformer("test_transformer", registry)
        class TestTransformer:
            pass

        assert registry.has_transformer("test_transformer")
        # get_transformer returns an instance
        result = registry.get_transformer("test_transformer")
        assert isinstance(result, TestTransformer)

    def test_register_renderer_decorator(self):
        """Test renderer registration decorator."""
        registry = ComponentRegistry()

        @register_renderer("test_renderer", registry)
        class TestRenderer:
            pass

        assert registry.has_renderer("test_renderer")

    def test_register_theme_decorator(self):
        """Test theme registration decorator."""
        registry = ComponentRegistry()

        @register_theme("test_theme", registry)
        class TestTheme:
            pass

        assert registry.has_theme("test_theme")

    def test_register_exporter_decorator(self):
        """Test exporter registration decorator."""
        registry = ComponentRegistry()

        @register_exporter("test_exporter", registry)
        class TestExporter:
            pass

        assert registry.has_exporter("test_exporter")
