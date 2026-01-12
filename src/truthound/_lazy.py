"""Lazy loading utilities for the truthound package.

This module implements PEP 562 lazy loading to improve import performance
by deferring the loading of heavy submodules until they are actually used.

Key Features:
- Module-level __getattr__ for lazy attribute access
- Submodule and attribute mapping for on-demand loading
- Performance metrics tracking

Example:
    # In __init__.py
    from truthound._lazy import truthound_getattr

    def __getattr__(name: str):
        return truthound_getattr(name)
"""

from __future__ import annotations

import importlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PackageImportMetrics:
    """Metrics for tracking package import performance."""

    total_lazy_loads: int = 0
    load_times: dict[str, float] = field(default_factory=dict)
    access_counts: dict[str, int] = field(default_factory=dict)

    def record_load(self, name: str, duration: float) -> None:
        """Record a module load."""
        self.load_times[name] = duration
        self.total_lazy_loads += 1

    def record_access(self, name: str) -> None:
        """Record an attribute access."""
        self.access_counts[name] = self.access_counts.get(name, 0) + 1

    def get_summary(self) -> dict[str, Any]:
        """Get metrics summary."""
        total_load_time = sum(self.load_times.values())
        return {
            "total_lazy_loads": self.total_lazy_loads,
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
        }


# =============================================================================
# Lazy Import Map
# =============================================================================

# Format: "attribute_name": ("module_path", "attribute_name_in_module" or None for module itself)
TRUTHOUND_IMPORT_MAP: dict[str, tuple[str, str | None]] = {
    # === Phase 5: Data sources ===
    "datasources": ("truthound.datasources", None),
    "execution": ("truthound.execution", None),
    "get_datasource": ("truthound.datasources", "get_datasource"),
    "get_sql_datasource": ("truthound.datasources", "get_sql_datasource"),

    # === Phase 6: Checkpoint & CI/CD ===
    "checkpoint": ("truthound.checkpoint", None),

    # === Phase 7: Auto-profiling & Rule Generation ===
    "profiler": ("truthound.profiler", None),
    "DataProfiler": ("truthound.profiler", "DataProfiler"),
    "profile_file": ("truthound.profiler", "profile_file"),
    "profile_dataframe": ("truthound.profiler", "profile_dataframe"),
    "generate_suite": ("truthound.profiler", "generate_suite"),
    "ValidationSuite": ("truthound.profiler", "ValidationSuite"),
    "TableProfile": ("truthound.profiler", "TableProfile"),
    "ColumnProfile": ("truthound.profiler", "ColumnProfile"),

    # === Phase 8: Data Docs ===
    "datadocs": ("truthound.datadocs", None),
    "HTMLReportBuilder": ("truthound.datadocs", "HTMLReportBuilder"),
    "generate_html_report": ("truthound.datadocs", "generate_html_report"),
    "generate_report_from_file": ("truthound.datadocs", "generate_report_from_file"),
    "ReportConfig": ("truthound.datadocs", "ReportConfig"),
    "ReportTheme": ("truthound.datadocs", "ReportTheme"),
    "ChartLibrary": ("truthound.datadocs", "ChartLibrary"),

    # === Phase 10: ML ===
    "ml": ("truthound.ml", None),
    "ModelRegistry": ("truthound.ml", "ModelRegistry"),
    "AnomalyDetector": ("truthound.ml", "AnomalyDetector"),
    "MLDriftDetector": ("truthound.ml", "MLDriftDetector"),
    "RuleLearner": ("truthound.ml", "RuleLearner"),
    "ModelType": ("truthound.ml", "ModelType"),
    "ModelState": ("truthound.ml", "ModelState"),
    "ZScoreAnomalyDetector": ("truthound.ml.anomaly_models", "ZScoreAnomalyDetector"),
    "IQRAnomalyDetector": ("truthound.ml.anomaly_models", "IQRAnomalyDetector"),
    "IsolationForestDetector": ("truthound.ml.anomaly_models", "IsolationForestDetector"),
    "EnsembleAnomalyDetector": ("truthound.ml.anomaly_models", "EnsembleAnomalyDetector"),
    "DistributionDriftDetector": ("truthound.ml.drift_detection", "DistributionDriftDetector"),
    "FeatureDriftDetector": ("truthound.ml.drift_detection", "FeatureDriftDetector"),

    # === Phase 10: Lineage ===
    "lineage": ("truthound.lineage", None),
    "LineageGraph": ("truthound.lineage", "LineageGraph"),
    "LineageNode": ("truthound.lineage", "LineageNode"),
    "LineageEdge": ("truthound.lineage", "LineageEdge"),
    "LineageTracker": ("truthound.lineage", "LineageTracker"),
    "ImpactAnalyzer": ("truthound.lineage", "ImpactAnalyzer"),
    "NodeType": ("truthound.lineage", "NodeType"),
    "EdgeType": ("truthound.lineage", "EdgeType"),

    # === Phase 10: Realtime ===
    "realtime": ("truthound.realtime", None),
    "StreamingValidator": ("truthound.realtime", "StreamingValidator"),
    "IncrementalValidator": ("truthound.realtime", "IncrementalValidator"),
    "StreamingConfig": ("truthound.realtime", "StreamingConfig"),
    "CheckpointManager": ("truthound.realtime", "CheckpointManager"),
    "MemoryStateStore": ("truthound.realtime", "MemoryStateStore"),
    "BatchResult": ("truthound.realtime", "BatchResult"),
    "StreamingMode": ("truthound.realtime", "StreamingMode"),

    # === Drift ===
    "compare": ("truthound.drift", "compare"),

    # === Report ===
    "Report": ("truthound.report", "Report"),
}


class LazyPackageLoader:
    """Loader for lazy package imports."""

    def __init__(
        self,
        import_map: dict[str, tuple[str, str | None]],
        metrics: PackageImportMetrics | None = None,
    ):
        self._import_map = import_map
        self._cache: dict[str, Any] = {}
        self._metrics = metrics or PackageImportMetrics()
        self._loaded_modules: dict[str, Any] = {}

    def load(self, name: str) -> Any:
        """Load and return an attribute by name."""
        self._metrics.record_access(name)

        # Check cache first
        if name in self._cache:
            return self._cache[name]

        # Check if we have a mapping
        if name not in self._import_map:
            raise AttributeError(f"module 'truthound' has no attribute '{name}'")

        module_path, attr_name = self._import_map[name]

        try:
            start_time = time.perf_counter()

            # Load the module if not cached
            if module_path not in self._loaded_modules:
                self._loaded_modules[module_path] = importlib.import_module(module_path)

            module = self._loaded_modules[module_path]

            # Get the attribute (or return the module itself)
            if attr_name is None:
                result = module
            else:
                result = getattr(module, attr_name)

            # Cache the result
            self._cache[name] = result

            duration = time.perf_counter() - start_time
            self._metrics.record_load(name, duration)

            logger.debug(
                f"Lazy loaded '{name}' from '{module_path}' in {duration*1000:.2f}ms"
            )

            return result

        except (ImportError, AttributeError) as e:
            logger.warning(f"Failed to lazy load '{name}' from '{module_path}': {e}")
            raise AttributeError(
                f"cannot load '{name}' from '{module_path}': {e}"
            ) from e

    def is_available(self, name: str) -> bool:
        """Check if an attribute is available for loading."""
        return name in self._import_map

    def get_available_names(self) -> list[str]:
        """Get list of all available attribute names."""
        return list(self._import_map.keys())

    def get_metrics(self) -> PackageImportMetrics:
        """Get import metrics."""
        return self._metrics


# Global loader instance
_package_loader: LazyPackageLoader | None = None


def get_package_loader() -> LazyPackageLoader:
    """Get or create the global package loader."""
    global _package_loader
    if _package_loader is None:
        _package_loader = LazyPackageLoader(TRUTHOUND_IMPORT_MAP)
    return _package_loader


def truthound_getattr(name: str) -> Any:
    """Module-level __getattr__ implementation for truthound.

    This function should be assigned to __getattr__ in the truthound __init__.py
    to enable lazy loading.
    """
    return get_package_loader().load(name)


def get_truthound_import_metrics() -> dict[str, Any]:
    """Get truthound package import metrics summary."""
    return get_package_loader().get_metrics().get_summary()
