"""Tests for lazy loading utilities."""

from __future__ import annotations

import pytest

from truthound.profiler._lazy import (
    LazyModule,
    LazyModuleLoader,
    ImportMetrics,
    create_lazy_loader,
    PROFILER_IMPORT_MAP,
)


class TestImportMetrics:
    """Tests for ImportMetrics."""

    def test_record_load(self) -> None:
        """Test recording a load."""
        metrics = ImportMetrics()
        metrics.record_load("DataProfiler", 0.1, lazy=True)

        assert metrics.total_lazy_loads == 1
        assert metrics.total_eager_loads == 0
        assert "DataProfiler" in metrics.load_times
        assert metrics.load_times["DataProfiler"] == 0.1

    def test_record_eager_load(self) -> None:
        """Test recording an eager load."""
        metrics = ImportMetrics()
        metrics.record_load("DataProfiler", 0.1, lazy=False)

        assert metrics.total_eager_loads == 1
        assert metrics.total_lazy_loads == 0

    def test_record_access(self) -> None:
        """Test recording access."""
        metrics = ImportMetrics()
        metrics.record_access("DataProfiler")
        metrics.record_access("DataProfiler")
        metrics.record_access("StreamingProfiler")

        assert metrics.access_counts["DataProfiler"] == 2
        assert metrics.access_counts["StreamingProfiler"] == 1

    def test_record_failure(self) -> None:
        """Test recording failure."""
        metrics = ImportMetrics()
        metrics.record_failure("NonExistent")

        assert "NonExistent" in metrics.failed_loads

    def test_get_summary(self) -> None:
        """Test getting summary."""
        metrics = ImportMetrics()
        metrics.record_load("DataProfiler", 0.1)
        metrics.record_access("DataProfiler")

        summary = metrics.get_summary()

        assert summary["total_lazy_loads"] == 1
        assert summary["total_load_time_ms"] == 100.0
        assert len(summary["most_accessed"]) > 0


class TestLazyModule:
    """Tests for LazyModule."""

    def test_lazy_module_not_loaded_initially(self) -> None:
        """Test that lazy module is not loaded on creation."""
        module = LazyModule("truthound.profiler.base")
        assert not object.__getattribute__(module, "_loaded")

    def test_lazy_module_loads_on_access(self) -> None:
        """Test that module loads on first attribute access."""
        module = LazyModule("truthound.profiler.base")

        # Access an attribute
        _ = module.DataType

        # Now it should be loaded
        assert object.__getattribute__(module, "_loaded")

    def test_lazy_module_repr(self) -> None:
        """Test lazy module string representation."""
        module = LazyModule("truthound.profiler.base")

        repr_str = repr(module)
        assert "not loaded" in repr_str

        # Load it
        _ = module.DataType
        repr_str = repr(module)
        assert "loaded" in repr_str


class TestLazyModuleLoader:
    """Tests for LazyModuleLoader."""

    @pytest.fixture
    def simple_import_map(self) -> dict[str, str]:
        """Create a simple import map for testing."""
        return {
            "DataType": "truthound.profiler.base",
            "Strictness": "truthound.profiler.base",
        }

    def test_load_attribute(self, simple_import_map: dict[str, str]) -> None:
        """Test loading an attribute."""
        loader = LazyModuleLoader(simple_import_map)

        result = loader.load("DataType")

        # Should return the actual enum
        from truthound.profiler.base import DataType
        assert result is DataType

    def test_load_caches_result(self, simple_import_map: dict[str, str]) -> None:
        """Test that loaded attributes are cached."""
        loader = LazyModuleLoader(simple_import_map)

        # Load twice
        result1 = loader.load("DataType")
        result2 = loader.load("DataType")

        assert result1 is result2

    def test_load_unknown_raises_error(self, simple_import_map: dict[str, str]) -> None:
        """Test that loading unknown attribute raises AttributeError."""
        loader = LazyModuleLoader(simple_import_map)

        with pytest.raises(AttributeError):
            loader.load("NonExistent")

    def test_is_available(self, simple_import_map: dict[str, str]) -> None:
        """Test checking if attribute is available."""
        loader = LazyModuleLoader(simple_import_map)

        assert loader.is_available("DataType")
        assert not loader.is_available("NonExistent")

    def test_get_available_names(self, simple_import_map: dict[str, str]) -> None:
        """Test getting available names."""
        loader = LazyModuleLoader(simple_import_map)

        names = loader.get_available_names()

        assert "DataType" in names
        assert "Strictness" in names

    def test_get_loaded_names(self, simple_import_map: dict[str, str]) -> None:
        """Test getting loaded names."""
        loader = LazyModuleLoader(simple_import_map)

        # Nothing loaded yet
        assert loader.get_loaded_names() == []

        # Load one
        loader.load("DataType")

        assert "DataType" in loader.get_loaded_names()

    def test_preload(self, simple_import_map: dict[str, str]) -> None:
        """Test preloading specific attributes."""
        loader = LazyModuleLoader(simple_import_map)

        loader.preload("DataType")

        assert "DataType" in loader.get_loaded_names()
        assert "Strictness" not in loader.get_loaded_names()

    def test_metrics_tracking(self, simple_import_map: dict[str, str]) -> None:
        """Test that metrics are tracked."""
        loader = LazyModuleLoader(simple_import_map)

        loader.load("DataType")

        metrics = loader.get_metrics()
        assert metrics.total_lazy_loads == 1


class TestCreateLazyLoader:
    """Tests for create_lazy_loader factory."""

    def test_creates_loader(self) -> None:
        """Test creating a loader."""
        import_map = {"DataType": "truthound.profiler.base"}
        loader = create_lazy_loader(import_map)

        assert isinstance(loader, LazyModuleLoader)

    def test_with_metrics_tracking(self) -> None:
        """Test creating loader with metrics tracking."""
        import_map = {"DataType": "truthound.profiler.base"}
        loader = create_lazy_loader(import_map, track_metrics=True)

        loader.load("DataType")

        assert loader.get_metrics() is not None

    def test_without_metrics_tracking(self) -> None:
        """Test creating loader without metrics tracking."""
        import_map = {"DataType": "truthound.profiler.base"}
        loader = create_lazy_loader(import_map, track_metrics=False)

        # Should work but metrics will be None
        loader.load("DataType")


class TestProfilerImportMap:
    """Tests for the profiler import map."""

    def test_import_map_has_essential_exports(self) -> None:
        """Test that import map has essential exports."""
        assert "DataProfiler" in PROFILER_IMPORT_MAP
        assert "DataType" in PROFILER_IMPORT_MAP
        assert "ValidationSuite" in PROFILER_IMPORT_MAP

    def test_import_map_modules_exist(self) -> None:
        """Test that mapped modules can be imported."""
        import importlib

        # Sample a few entries
        samples = ["DataType", "DataProfiler", "RuleGenerator"]

        for name in samples:
            if name in PROFILER_IMPORT_MAP:
                module_path = PROFILER_IMPORT_MAP[name]
                # Just check the module can be imported
                module = importlib.import_module(module_path)
                assert hasattr(module, name)
