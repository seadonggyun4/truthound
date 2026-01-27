"""Truthound - Zero-Configuration Data Quality Framework Powered by Polars.

This module uses lazy loading to improve import performance. Only the core API
functions (check, scan, mask, profile, learn) are loaded eagerly. All other
submodules and classes are loaded on-demand when first accessed.

Usage:
    # Core API (eagerly loaded, fast)
    from truthound import check, scan, mask, profile, learn

    # Advanced features (lazy loaded on first access)
    from truthound import profiler, ml, lineage, realtime

    # Or access directly
    import truthound
    truthound.DataProfiler  # Loaded on first access
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

# =============================================================================
# Core API - Eagerly loaded (most commonly used)
# =============================================================================
from truthound.api import check, mask, profile, read, scan
from truthound.decorators import validator
from truthound.schema import Schema, learn

# Lazy loading infrastructure
from truthound._lazy import (
    TRUTHOUND_IMPORT_MAP,
    truthound_getattr,
    get_truthound_import_metrics,
)

# Version: Single source of truth from pyproject.toml
try:
    from importlib.metadata import version, PackageNotFoundError

    __version__ = version("truthound")
except PackageNotFoundError:
    # Package not installed (development mode)
    __version__ = "0.0.0.dev"


# =============================================================================
# Lazy Loading via __getattr__ (PEP 562)
# =============================================================================

def __getattr__(name: str) -> Any:
    """Lazy load submodules and attributes on demand.

    This function is called when an attribute is not found in the module.
    It enables lazy loading of heavy submodules like profiler, ml, lineage, etc.
    """
    if name in TRUTHOUND_IMPORT_MAP:
        return truthound_getattr(name)

    raise AttributeError(f"module 'truthound' has no attribute '{name}'")


def __dir__() -> list[str]:
    """Return list of available attributes for tab completion."""
    # Eagerly loaded attributes
    eager = [
        "check", "scan", "mask", "profile", "read", "learn",
        "validator", "Schema", "__version__",
        "get_truthound_import_metrics",
    ]
    # Lazy loaded attributes
    lazy = list(TRUTHOUND_IMPORT_MAP.keys())
    return sorted(set(eager + lazy))


# =============================================================================
# Type hints for IDE support (not actually imported at runtime)
# =============================================================================

if TYPE_CHECKING:
    # Phase 5: Data sources
    from truthound import datasources as datasources
    from truthound import execution as execution
    from truthound.datasources import get_datasource as get_datasource
    from truthound.datasources import get_sql_datasource as get_sql_datasource

    # Phase 6: Checkpoint
    from truthound import checkpoint as checkpoint

    # Phase 7: Profiler
    from truthound import profiler as profiler
    from truthound.profiler import (
        DataProfiler as DataProfiler,
        profile_file as profile_file,
        profile_dataframe as profile_dataframe,
        generate_suite as generate_suite,
        ValidationSuite as ValidationSuite,
        TableProfile as TableProfile,
        ColumnProfile as ColumnProfile,
    )

    # Phase 8: Data Docs
    from truthound import datadocs as datadocs
    from truthound.datadocs import (
        HTMLReportBuilder as HTMLReportBuilder,
        generate_html_report as generate_html_report,
        generate_report_from_file as generate_report_from_file,
        ReportConfig as ReportConfig,
        ReportTheme as ReportTheme,
        ChartLibrary as ChartLibrary,
    )

    # Phase 10: ML
    from truthound import ml as ml
    from truthound.ml import (
        ModelRegistry as ModelRegistry,
        AnomalyDetector as AnomalyDetector,
        MLDriftDetector as MLDriftDetector,
        RuleLearner as RuleLearner,
        ModelType as ModelType,
        ModelState as ModelState,
    )
    from truthound.ml.anomaly_models import (
        ZScoreAnomalyDetector as ZScoreAnomalyDetector,
        IQRAnomalyDetector as IQRAnomalyDetector,
        IsolationForestDetector as IsolationForestDetector,
        EnsembleAnomalyDetector as EnsembleAnomalyDetector,
    )
    from truthound.ml.drift_detection import (
        DistributionDriftDetector as DistributionDriftDetector,
        FeatureDriftDetector as FeatureDriftDetector,
    )

    # Phase 10: Lineage
    from truthound import lineage as lineage
    from truthound.lineage import (
        LineageGraph as LineageGraph,
        LineageNode as LineageNode,
        LineageEdge as LineageEdge,
        LineageTracker as LineageTracker,
        ImpactAnalyzer as ImpactAnalyzer,
        NodeType as NodeType,
        EdgeType as EdgeType,
    )

    # Phase 10: Realtime
    from truthound import realtime as realtime
    from truthound.realtime import (
        StreamingValidator as StreamingValidator,
        IncrementalValidator as IncrementalValidator,
        StreamingConfig as StreamingConfig,
        CheckpointManager as CheckpointManager,
        MemoryStateStore as MemoryStateStore,
        BatchResult as BatchResult,
        StreamingMode as StreamingMode,
    )

    # Drift
    from truthound.drift import compare as compare

    # Report
    from truthound.report import Report as Report


__all__ = [
    # Core API (eagerly loaded)
    "check",
    "scan",
    "mask",
    "profile",
    "read",
    "learn",
    "validator",
    "Schema",
    # Version
    "__version__",
    # Metrics
    "get_truthound_import_metrics",
    # Phase 5: Data sources (lazy)
    "datasources",
    "execution",
    "get_datasource",
    "get_sql_datasource",
    # Phase 6: Checkpoint & CI/CD (lazy)
    "checkpoint",
    # Phase 7: Auto-profiling & Rule Generation (lazy)
    "profiler",
    "DataProfiler",
    "profile_file",
    "profile_dataframe",
    "generate_suite",
    "ValidationSuite",
    "TableProfile",
    "ColumnProfile",
    # Phase 8: Data Docs (lazy)
    "datadocs",
    "HTMLReportBuilder",
    "generate_html_report",
    "generate_report_from_file",
    "ReportConfig",
    "ReportTheme",
    "ChartLibrary",
    # Phase 10: ML (lazy)
    "ml",
    "ModelRegistry",
    "AnomalyDetector",
    "MLDriftDetector",
    "RuleLearner",
    "ModelType",
    "ModelState",
    "ZScoreAnomalyDetector",
    "IQRAnomalyDetector",
    "IsolationForestDetector",
    "EnsembleAnomalyDetector",
    "DistributionDriftDetector",
    "FeatureDriftDetector",
    # Phase 10: Lineage (lazy)
    "lineage",
    "LineageGraph",
    "LineageNode",
    "LineageEdge",
    "LineageTracker",
    "ImpactAnalyzer",
    "NodeType",
    "EdgeType",
    # Phase 10: Realtime (lazy)
    "realtime",
    "StreamingValidator",
    "IncrementalValidator",
    "StreamingConfig",
    "CheckpointManager",
    "MemoryStateStore",
    "BatchResult",
    "StreamingMode",
    # Drift (lazy)
    "compare",
    # Report (lazy)
    "Report",
]
