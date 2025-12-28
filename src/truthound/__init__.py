"""Truthound - Zero-Configuration Data Quality Framework Powered by Polars."""

from truthound.api import check, mask, profile, scan
from truthound.decorators import validator
from truthound.drift import compare
from truthound.report import Report
from truthound.schema import Schema, learn

# Data sources and execution engines (Phase 5)
from truthound import datasources
from truthound import execution
from truthound.datasources import get_datasource, get_sql_datasource

# Checkpoint and CI/CD integration (Phase 6)
from truthound import checkpoint

# Auto-profiling and rule generation (Phase 7)
from truthound import profiler
from truthound.profiler import (
    DataProfiler,
    profile_file,
    profile_dataframe,
    generate_suite,
    ValidationSuite,
    TableProfile,
    ColumnProfile,
)

# Data Docs - HTML Reports and Dashboard (Phase 8)
from truthound import datadocs
from truthound.datadocs import (
    HTMLReportBuilder,
    generate_html_report,
    generate_report_from_file,
    ReportConfig,
    ReportTheme,
    ChartLibrary,
)

# Advanced Features - ML, Lineage, Realtime (Phase 10)
from truthound import ml
from truthound import lineage
from truthound import realtime

# ML exports
from truthound.ml import (
    ModelRegistry,
    AnomalyDetector,
    MLDriftDetector,
    RuleLearner,
    ModelType,
    ModelState,
)
from truthound.ml.anomaly_models import (
    ZScoreAnomalyDetector,
    IQRAnomalyDetector,
    IsolationForestDetector,
    EnsembleAnomalyDetector,
)
from truthound.ml.drift_detection import (
    DistributionDriftDetector,
    FeatureDriftDetector,
)

# Lineage exports
from truthound.lineage import (
    LineageGraph,
    LineageNode,
    LineageEdge,
    LineageTracker,
    ImpactAnalyzer,
    NodeType,
    EdgeType,
)

# Realtime exports
from truthound.realtime import (
    StreamingValidator,
    IncrementalValidator,
    StreamingConfig,
    CheckpointManager,
    MemoryStateStore,
    BatchResult,
    StreamingMode,
)

# Version: Single source of truth from pyproject.toml
try:
    from importlib.metadata import version, PackageNotFoundError

    __version__ = version("truthound")
except PackageNotFoundError:
    # Package not installed (development mode)
    __version__ = "0.0.0.dev"
__all__ = [
    # Core API
    "check",
    "scan",
    "mask",
    "profile",
    "learn",
    "compare",
    "validator",
    "Report",
    "Schema",
    # Phase 5: Data sources
    "datasources",
    "execution",
    "get_datasource",
    "get_sql_datasource",
    # Phase 6: Checkpoint & CI/CD
    "checkpoint",
    # Phase 7: Auto-profiling & Rule Generation
    "profiler",
    "DataProfiler",
    "profile_file",
    "profile_dataframe",
    "generate_suite",
    "ValidationSuite",
    "TableProfile",
    "ColumnProfile",
    # Phase 8: Data Docs (HTML Reports & Dashboard)
    "datadocs",
    "HTMLReportBuilder",
    "generate_html_report",
    "generate_report_from_file",
    "ReportConfig",
    "ReportTheme",
    "ChartLibrary",
    # Phase 10: Advanced Features - ML
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
    # Phase 10: Advanced Features - Lineage
    "lineage",
    "LineageGraph",
    "LineageNode",
    "LineageEdge",
    "LineageTracker",
    "ImpactAnalyzer",
    "NodeType",
    "EdgeType",
    # Phase 10: Advanced Features - Realtime
    "realtime",
    "StreamingValidator",
    "IncrementalValidator",
    "StreamingConfig",
    "CheckpointManager",
    "MemoryStateStore",
    "BatchResult",
    "StreamingMode",
]
