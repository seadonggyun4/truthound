"""Lazy loading utilities for the profiler module.

This module implements PEP 562 lazy loading to improve import performance
by deferring the loading of heavy submodules until they are actually used.

Key Features:
- Module-level __getattr__ for lazy attribute access
- LazyModule wrapper for deferred imports
- Import mapping registry for module discovery
- Performance metrics tracking

Example:
    # In __init__.py
    from truthound.profiler._lazy import LazyModuleLoader

    _loader = LazyModuleLoader({
        "DataProfiler": "truthound.profiler.table_profiler",
        "StreamingProfiler": "truthound.profiler.streaming",
        # ... more mappings
    })

    def __getattr__(name: str):
        return _loader.load(name)
"""

from __future__ import annotations

import importlib
import logging
import sys
import time
from dataclasses import dataclass, field
from functools import lru_cache
from types import ModuleType
from typing import Any, Callable, Dict, List, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ImportMetrics:
    """Metrics for tracking import performance."""

    total_lazy_loads: int = 0
    total_eager_loads: int = 0
    load_times: Dict[str, float] = field(default_factory=dict)
    access_counts: Dict[str, int] = field(default_factory=dict)
    failed_loads: List[str] = field(default_factory=list)

    def record_load(self, name: str, duration: float, lazy: bool = True) -> None:
        """Record a module load."""
        self.load_times[name] = duration
        if lazy:
            self.total_lazy_loads += 1
        else:
            self.total_eager_loads += 1

    def record_access(self, name: str) -> None:
        """Record an attribute access."""
        self.access_counts[name] = self.access_counts.get(name, 0) + 1

    def record_failure(self, name: str) -> None:
        """Record a failed load."""
        self.failed_loads.append(name)

    def get_summary(self) -> dict[str, Any]:
        """Get metrics summary."""
        total_load_time = sum(self.load_times.values())
        return {
            "total_lazy_loads": self.total_lazy_loads,
            "total_eager_loads": self.total_eager_loads,
            "total_load_time_ms": total_load_time * 1000,
            "slowest_loads": sorted(
                self.load_times.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:10],
            "most_accessed": sorted(
                self.access_counts.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:10],
            "failed_loads": self.failed_loads,
        }


class LazyModule:
    """Lazy module wrapper that defers import until first access.

    Example:
        lazy_mod = LazyModule("truthound.profiler.streaming")
        # Module is not loaded yet

        lazy_mod.StreamingProfiler  # Now the module is loaded
    """

    __slots__ = ("_module_path", "_module", "_loaded")

    def __init__(self, module_path: str):
        """Initialize lazy module.

        Args:
            module_path: Full module path to import.
        """
        object.__setattr__(self, "_module_path", module_path)
        object.__setattr__(self, "_module", None)
        object.__setattr__(self, "_loaded", False)

    def _load(self) -> ModuleType:
        """Load the module if not already loaded."""
        if not object.__getattribute__(self, "_loaded"):
            module_path = object.__getattribute__(self, "_module_path")
            module = importlib.import_module(module_path)
            object.__setattr__(self, "_module", module)
            object.__setattr__(self, "_loaded", True)
        return object.__getattribute__(self, "_module")

    def __getattr__(self, name: str) -> Any:
        """Get attribute from the loaded module."""
        module = self._load()
        return getattr(module, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Set attribute on the loaded module."""
        module = self._load()
        setattr(module, name, value)

    def __repr__(self) -> str:
        """String representation."""
        loaded = object.__getattribute__(self, "_loaded")
        module_path = object.__getattribute__(self, "_module_path")
        status = "loaded" if loaded else "not loaded"
        return f"<LazyModule({module_path!r}) [{status}]>"


class LazyModuleLoader:
    """Loader for lazy module imports using name-to-module mapping.

    This loader maintains a mapping of attribute names to their source modules,
    enabling lazy loading of specific classes and functions.

    Example:
        loader = LazyModuleLoader({
            "DataProfiler": "truthound.profiler.table_profiler",
            "StreamingProfiler": "truthound.profiler.streaming",
        })

        # In module __getattr__:
        def __getattr__(name: str):
            return loader.load(name)
    """

    def __init__(
        self,
        import_map: dict[str, str],
        metrics: ImportMetrics | None = None,
    ):
        """Initialize loader.

        Args:
            import_map: Mapping of attribute names to module paths.
            metrics: Optional metrics tracker.
        """
        self._import_map = import_map
        self._cache: dict[str, Any] = {}
        self._metrics = metrics or ImportMetrics()
        self._loaded_modules: dict[str, ModuleType] = {}

    def load(self, name: str) -> Any:
        """Load and return an attribute by name.

        Args:
            name: Attribute name to load.

        Returns:
            The requested attribute.

        Raises:
            AttributeError: If the attribute is not in the import map.
        """
        self._metrics.record_access(name)

        # Check cache first
        if name in self._cache:
            return self._cache[name]

        # Check if we have a mapping
        if name not in self._import_map:
            raise AttributeError(f"module has no attribute '{name}'")

        module_path = self._import_map[name]

        try:
            start_time = time.perf_counter()

            # Load the module if not cached
            if module_path not in self._loaded_modules:
                self._loaded_modules[module_path] = importlib.import_module(module_path)

            module = self._loaded_modules[module_path]
            attr = getattr(module, name)

            # Cache the result
            self._cache[name] = attr

            duration = time.perf_counter() - start_time
            self._metrics.record_load(name, duration)

            logger.debug(
                f"Lazy loaded '{name}' from '{module_path}' in {duration*1000:.2f}ms"
            )

            return attr

        except (ImportError, AttributeError) as e:
            self._metrics.record_failure(name)
            logger.warning(f"Failed to lazy load '{name}' from '{module_path}': {e}")
            raise AttributeError(f"cannot load '{name}' from '{module_path}': {e}") from e

    def is_available(self, name: str) -> bool:
        """Check if an attribute is available for loading."""
        return name in self._import_map

    def get_available_names(self) -> list[str]:
        """Get list of all available attribute names."""
        return list(self._import_map.keys())

    def get_loaded_names(self) -> list[str]:
        """Get list of already loaded attribute names."""
        return list(self._cache.keys())

    def get_metrics(self) -> ImportMetrics:
        """Get import metrics."""
        return self._metrics

    def preload(self, *names: str) -> None:
        """Preload specific attributes.

        Args:
            names: Attribute names to preload.
        """
        for name in names:
            if name in self._import_map and name not in self._cache:
                try:
                    self.load(name)
                except AttributeError:
                    pass

    def preload_all(self) -> None:
        """Preload all attributes (converts to eager loading)."""
        for name in self._import_map:
            try:
                self.load(name)
            except AttributeError:
                pass


def create_lazy_loader(
    import_map: dict[str, str],
    track_metrics: bool = True,
) -> LazyModuleLoader:
    """Factory function for creating lazy loaders.

    Args:
        import_map: Mapping of attribute names to module paths.
        track_metrics: Whether to track import metrics.

    Returns:
        Configured lazy loader.
    """
    metrics = ImportMetrics() if track_metrics else None
    return LazyModuleLoader(import_map, metrics)


# =============================================================================
# Profiler Import Map
# =============================================================================

# This defines all lazy-loadable attributes for the profiler module
PROFILER_IMPORT_MAP: dict[str, str] = {
    # === Base module ===
    "DataType": "truthound.profiler.base",
    "Strictness": "truthound.profiler.base",
    "ProfileCategory": "truthound.profiler.base",
    "PatternMatch": "truthound.profiler.base",
    "DistributionStats": "truthound.profiler.base",
    "ValueFrequency": "truthound.profiler.base",
    "ColumnProfile": "truthound.profiler.base",
    "TableProfile": "truthound.profiler.base",
    "Profiler": "truthound.profiler.base",
    "ProfilerProtocol": "truthound.profiler.base",
    "TypeInferrer": "truthound.profiler.base",
    "ProfilerConfig": "truthound.profiler.base",
    "ProfilerRegistry": "truthound.profiler.base",
    "profiler_registry": "truthound.profiler.base",
    "register_profiler": "truthound.profiler.base",
    "register_type_inferrer": "truthound.profiler.base",

    # === Column profiler ===
    "ColumnAnalyzer": "truthound.profiler.column_profiler",
    "BasicStatsAnalyzer": "truthound.profiler.column_profiler",
    "NumericAnalyzer": "truthound.profiler.column_profiler",
    "StringAnalyzer": "truthound.profiler.column_profiler",
    "DatetimeAnalyzer": "truthound.profiler.column_profiler",
    "ValueFrequencyAnalyzer": "truthound.profiler.column_profiler",
    "PatternAnalyzer": "truthound.profiler.column_profiler",
    "PhysicalTypeInferrer": "truthound.profiler.column_profiler",
    "PatternBasedTypeInferrer": "truthound.profiler.column_profiler",
    "CardinalityTypeInferrer": "truthound.profiler.column_profiler",
    "ColumnProfiler": "truthound.profiler.column_profiler",
    "PatternDefinition": "truthound.profiler.column_profiler",
    "BUILTIN_PATTERNS": "truthound.profiler.column_profiler",

    # === Table profiler ===
    "TableAnalyzer": "truthound.profiler.table_profiler",
    "DuplicateRowAnalyzer": "truthound.profiler.table_profiler",
    "MemoryEstimator": "truthound.profiler.table_profiler",
    "CorrelationAnalyzer": "truthound.profiler.table_profiler",
    "DataProfiler": "truthound.profiler.table_profiler",
    "profile_dataframe": "truthound.profiler.table_profiler",
    "profile_file": "truthound.profiler.table_profiler",
    "save_profile": "truthound.profiler.table_profiler",
    "load_profile": "truthound.profiler.table_profiler",

    # === Generators ===
    "RuleGenerator": "truthound.profiler.generators",
    "GeneratedRule": "truthound.profiler.generators",
    "RuleGeneratorRegistry": "truthound.profiler.generators",
    "rule_generator_registry": "truthound.profiler.generators",
    "register_generator": "truthound.profiler.generators",
    "SchemaRuleGenerator": "truthound.profiler.generators",
    "StatsRuleGenerator": "truthound.profiler.generators",
    "PatternRuleGenerator": "truthound.profiler.generators",
    "MLRuleGenerator": "truthound.profiler.generators",
    "ValidationSuiteGenerator": "truthound.profiler.generators",
    "generate_suite": "truthound.profiler.generators",
    "RuleCategory": "truthound.profiler.generators.base",
    "RuleConfidence": "truthound.profiler.generators.base",
    "RuleBuilder": "truthound.profiler.generators.base",
    "StrictnessThresholds": "truthound.profiler.generators.base",
    "DEFAULT_THRESHOLDS": "truthound.profiler.generators.base",
    "ValidationSuite": "truthound.profiler.generators.suite_generator",
    "save_suite": "truthound.profiler.generators.suite_generator",
    "load_suite": "truthound.profiler.generators.suite_generator",

    # === Error handling ===
    "ErrorSeverity": "truthound.profiler.errors",
    "ErrorCategory": "truthound.profiler.errors",
    "ProfilerError": "truthound.profiler.errors",
    "AnalysisError": "truthound.profiler.errors",
    "PatternError": "truthound.profiler.errors",
    "TypeInferenceError": "truthound.profiler.errors",
    "ValidationError": "truthound.profiler.errors",
    "ErrorRecord": "truthound.profiler.errors",
    "ErrorCollector": "truthound.profiler.errors",
    "ErrorCatcher": "truthound.profiler.errors",
    "with_error_handling": "truthound.profiler.errors",

    # === Streaming ===
    "IncrementalStats": "truthound.profiler.streaming",
    "FileChunkIterator": "truthound.profiler.streaming",
    "DataFrameChunkIterator": "truthound.profiler.streaming",
    "StreamingProgress": "truthound.profiler.streaming",
    "ProgressCallback": "truthound.profiler.streaming",
    "StreamingProfiler": "truthound.profiler.streaming",
    "stream_profile_file": "truthound.profiler.streaming",
    "stream_profile_dataframe": "truthound.profiler.streaming",

    # === Incremental ===
    "ChangeReason": "truthound.profiler.incremental",
    "ColumnFingerprint": "truthound.profiler.incremental",
    "ChangeDetectionResult": "truthound.profiler.incremental",
    "FingerprintCalculator": "truthound.profiler.incremental",
    "IncrementalConfig": "truthound.profiler.incremental",
    "IncrementalProfiler": "truthound.profiler.incremental",
    "ProfileMerger": "truthound.profiler.incremental",
    "profile_incrementally": "truthound.profiler.incremental",

    # === Comparison ===
    "DriftType": "truthound.profiler.comparison",
    "DriftSeverity": "truthound.profiler.comparison",
    "ChangeDirection": "truthound.profiler.comparison",
    "DriftResult": "truthound.profiler.comparison",
    "ColumnComparison": "truthound.profiler.comparison",
    "ProfileComparison": "truthound.profiler.comparison",
    "DriftThresholds": "truthound.profiler.comparison",
    "DriftDetector": "truthound.profiler.comparison",
    "ProfileComparator": "truthound.profiler.comparison",
    "compare_profiles": "truthound.profiler.comparison",
    "detect_drift": "truthound.profiler.comparison",

    # === Distributed ===
    "BackendType": "truthound.profiler.distributed",
    "PartitionStrategy": "truthound.profiler.distributed",
    "DistributedBackend": "truthound.profiler.distributed",
    "DistributedProfiler": "truthound.profiler.distributed",
    "create_distributed_profiler": "truthound.profiler.distributed",
    "profile_distributed": "truthound.profiler.distributed",

    # === Sampling ===
    "SamplingMethod": "truthound.profiler.sampling",
    "SamplingConfig": "truthound.profiler.sampling",
    "Sampler": "truthound.profiler.sampling",
    "sample_data": "truthound.profiler.sampling",

    # === Caching ===
    "CacheBackend": "truthound.profiler.caching",
    "MemoryCacheBackend": "truthound.profiler.caching",
    "ProfileCache": "truthound.profiler.caching",
    "cached_profile": "truthound.profiler.caching",

    # === Visualization ===
    "ChartType": "truthound.profiler.visualization",
    "HTMLReportGenerator": "truthound.profiler.visualization",
    "generate_report": "truthound.profiler.visualization",

    # === ML Inference ===
    "MLTypeInferrer": "truthound.profiler.ml_inference",
    "infer_column_type_ml": "truthound.profiler.ml_inference",

    # === Auto Threshold ===
    "ThresholdTuner": "truthound.profiler.auto_threshold",
    "tune_thresholds": "truthound.profiler.auto_threshold",
}


# Global loader instance for the profiler module
_profiler_loader: LazyModuleLoader | None = None


def get_profiler_loader() -> LazyModuleLoader:
    """Get or create the global profiler loader."""
    global _profiler_loader
    if _profiler_loader is None:
        _profiler_loader = create_lazy_loader(PROFILER_IMPORT_MAP)
    return _profiler_loader


def profiler_getattr(name: str) -> Any:
    """Module-level __getattr__ implementation for profiler.

    This function should be assigned to __getattr__ in the profiler __init__.py
    to enable lazy loading.

    Example:
        # In profiler/__init__.py
        from truthound.profiler._lazy import profiler_getattr

        def __getattr__(name: str):
            return profiler_getattr(name)
    """
    return get_profiler_loader().load(name)
