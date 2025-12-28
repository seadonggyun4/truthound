"""Pipeline Engine for Data Docs report generation.

This module provides the core orchestration layer for the report generation pipeline.

Components:
- ReportContext: Immutable context passed through the pipeline
- ReportPipeline: Composable pipeline orchestrator
- ComponentRegistry: Registry for transformers, renderers, themes, and exporters
"""

from truthound.datadocs.engine.context import (
    ReportContext,
    TranslatableString,
    ReportData,
)
from truthound.datadocs.engine.pipeline import (
    ReportPipeline,
    PipelineBuilder,
    PipelineResult,
)
from truthound.datadocs.engine.registry import (
    ComponentRegistry,
    component_registry,
    register_transformer,
    register_renderer,
    register_theme,
    register_exporter,
)

__all__ = [
    # Context
    "ReportContext",
    "TranslatableString",
    "ReportData",
    # Pipeline
    "ReportPipeline",
    "PipelineBuilder",
    "PipelineResult",
    # Registry
    "ComponentRegistry",
    "component_registry",
    "register_transformer",
    "register_renderer",
    "register_theme",
    "register_exporter",
]
