"""Generate reviewed, closed constructor contracts from the installed Core source.

Run in the developer environment, never at server startup. Generated metadata is
product API metadata, not an execution/import capability or a parity claim.
"""

from __future__ import annotations

import ast
import collections.abc
import dataclasses
import datetime
import enum
import inspect
import json
import textwrap
import types
import typing
from pathlib import Path

import polars as pl

from truthound.mcp.source_identity import class_identity
from truthound.validators.base import ValidatorConfig
from truthound.validators.registry import registry

RECORDS = {}
RECORD_SOURCES = {}


def reference(cls):
    return f"{cls.__module__}.{cls.__qualname__}"


def identity(cls):
    return class_identity(cls)


def record_source(cls):
    for parent in cls.__mro__:
        if parent.__module__.startswith("truthound."):
            RECORD_SOURCES[reference(parent)] = identity(parent)


def describe(annotation):
    if annotation is inspect.Parameter.empty:
        return {"kind": "blocked", "reason": "missing_annotation"}
    if annotation is typing.Any:
        return {"kind": "json"}
    if annotation in (str, int, float, bool, type(None)):
        return {
            "kind": {str: "str", int: "int", float: "float", bool: "bool", type(None): "null"}[
                annotation
            ]
        }
    origin, args = typing.get_origin(annotation), typing.get_args(annotation)
    if origin in (typing.Union, types.UnionType):
        return {"kind": "union", "items": [describe(a) for a in args]}
    if origin is typing.Literal:
        return {"kind": "literal", "values": list(args)}
    if origin is collections.abc.Callable:
        return {
            "kind": "callback",
            "parameters": [describe(item) for item in args[0]]
            if args and args[0] is not Ellipsis
            else [],
            "returns": describe(args[1]) if len(args) > 1 else {"kind": "json"},
        }
    if origin in (list, set, tuple, collections.abc.Sequence):
        if origin is tuple and args and args[-1] is not Ellipsis:
            return {"kind": "tuple", "items": [describe(a) for a in args]}
        return {
            "kind": "set" if origin is set else "tuple_list" if origin is tuple else "list",
            "item": describe(args[0] if args else typing.Any),
        }
    if origin is dict or annotation is dict:
        key, value = args or (str, typing.Any)
        return {"kind": "map", "key": describe(key), "value": describe(value)}
    if annotation in (pl.DataFrame, pl.LazyFrame):
        return {"kind": "frame", "lazy": annotation is pl.LazyFrame}
    if annotation is pl.Expr:
        return {"kind": "expression"}
    if annotation in (datetime.datetime, datetime.date, datetime.timedelta):
        return {"kind": "temporal", "type": annotation.__name__}
    if inspect.isclass(annotation) and issubclass(annotation, enum.Enum):
        record_source(annotation)
        return {
            "kind": "enum",
            "target": reference(annotation),
            "values": [e.value for e in annotation],
        }
    if inspect.isclass(annotation) and dataclasses.is_dataclass(annotation):
        target = reference(annotation)
        if target not in RECORDS:
            record_source(annotation)
            hints = typing.get_type_hints(annotation)
            RECORDS[target] = {}
            RECORDS[target] = {
                f.name: {
                    "type": describe(hints.get(f.name, f.type)),
                    "required": f.default is dataclasses.MISSING
                    and f.default_factory is dataclasses.MISSING,
                }
                for f in dataclasses.fields(annotation)
                if f.init
            }
        return {"kind": "record", "target": target}
    if annotation is pl.DataType or origin is type or annotation is type:
        return {"kind": "dtype"}
    if annotation is Path:
        # Python 3.13 moved the implementation into pathlib._local. Keep the
        # public unsupported-type label stable; this does not authorize paths.
        return {"kind": "blocked", "reason": "<class 'pathlib.Path'>"}
    return {"kind": "blocked", "reason": str(annotation)}


def constructor(cls, *, validator_options=True):
    fields = {}
    supplied_to_parent = set()
    sources = {
        reference(parent): identity(parent)
        for parent in cls.__mro__
        if parent.__module__.startswith("truthound.")
    }
    for parent in cls.__mro__:
        if parent is object:
            continue
        method = parent.__dict__.get("__init__")
        if method is None:
            continue
        signature = inspect.signature(method)
        try:
            hints = typing.get_type_hints(method)
        except (NameError, TypeError):
            # Preserve unresolved annotations as an explicit unsupported binding.
            hints = inspect.get_annotations(method)
        for parameter in signature.parameters.values():
            if (
                parameter.name == "self"
                or parameter.name in supplied_to_parent
                or parameter.kind
                in (
                    parameter.VAR_POSITIONAL,
                    parameter.VAR_KEYWORD,
                )
            ):
                continue
            fields.setdefault(
                parameter.name,
                {
                    "type": describe(hints.get(parameter.name, parameter.annotation)),
                    "required": parameter.default is parameter.empty,
                },
            )
        # Unconditional super().__init__(field=...) supplies a parent field
        # internally. Exposing it as a caller requirement is false; accepting it
        # through **kwargs instead causes a duplicate-keyword TypeError.
        # Only inspect direct statements, never infer conditional control flow.
        tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
        for statement in tree.body[0].body:
            if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
                continue
            call = statement.value
            function = call.func
            if (
                isinstance(function, ast.Attribute)
                and function.attr == "__init__"
                and isinstance(function.value, ast.Call)
                and isinstance(function.value.func, ast.Name)
                and function.value.func.id == "super"
            ):
                supplied_to_parent.update(key.arg for key in call.keywords if key.arg is not None)
        if not any(p.kind == p.VAR_KEYWORD for p in signature.parameters.values()):
            break
    # Validator.__init__ accepts these forwarded kwargs; the Core otherwise
    # silently discards unknown fields. The wire contract never does that.
    if validator_options:
        for name, row in RECORDS[describe(ValidatorConfig)["target"]].items():
            if name not in supplied_to_parent:
                fields.setdefault(name, {**row, "required": False})
    for name in supplied_to_parent - fields.keys():
        fields[name] = {
            "type": {"kind": "blocked", "reason": "internally supplied parent argument"},
            "required": False,
        }
    if reference(cls) == "truthound.validators.multi_column.consistency.ColumnConsistencyValidator":
        fields["rules"]["type"] = {"kind": "list", "item": {"kind": "consistency_rule"}}
    return fields, sources


def generate():
    RECORDS.clear()
    RECORD_SOURCES.clear()
    import truthound.drift as drift
    import truthound.ml as ml
    import truthound.profiler as profiler
    from truthound.core.suite import (
        CheckSpec,
        EvidencePolicy,
        SchemaSpec,
        SeverityPolicy,
        ValidationSuite,
    )
    from truthound.drift import DriftDetector
    from truthound.lineage.base import LineageConfig, LineageEdge, LineageGraph, LineageNode
    from truthound.lineage.impact_analysis import ImpactAnalyzer
    from truthound.lineage.tracker import LineageTracker
    from truthound.lineage.visualization import get_renderer
    from truthound.lineage.visualization.protocols import RenderConfig
    from truthound.ml.base import MLModel
    from truthound.reporters.console_reporter import ConsoleReporterConfig
    from truthound.reporters.html_reporter import HTMLReporterConfig
    from truthound.reporters.json_reporter import JSONReporterConfig
    from truthound.reporters.markdown_reporter import MarkdownReporterConfig

    rows, abstract, source_types = {}, [], {}
    common_fields = {
        name: {**row, "required": False}
        for name, row in RECORDS[describe(ValidatorConfig)["target"]].items()
    }
    models = {}
    for name in ml.__all__:
        cls = getattr(ml, name)
        if inspect.isclass(cls) and issubclass(cls, MLModel) and not inspect.isabstract(cls):
            instance = cls()
            sources = {
                reference(parent): identity(parent)
                for parent in cls.__mro__
                if parent.__module__.startswith("truthound.")
            }
            source_types.update(sources)
            models[name] = {
                "target": reference(cls),
                "config": describe(type(instance.config)),
                "source_types": list(sources),
                "supports_online_learning": instance.info.supports_online_learning,
                "native_methods": sorted(
                    key
                    for key, value in inspect.getmembers(cls, inspect.isfunction)
                    if not key.startswith("_")
                ),
            }
    for name, cls in sorted(registry.list_all().items()):
        if inspect.isabstract(cls):
            abstract.append(name)
            continue
        fields, sources = constructor(cls)
        source_types.update(sources)
        rows[name] = {
            "target": reference(cls),
            "category": cls.category,
            "fields": {
                key: value for key, value in fields.items() if common_fields.get(key) != value
            },
            "source_types": list(sources),
        }
    lineage = {
        "config": describe(LineageConfig),
        "node": describe(LineageNode),
        "edge": describe(LineageEdge),
        "render_config": describe(RenderConfig),
        "renderers": {},
    }
    for name in ("d3", "cytoscape", "graphviz", "mermaid"):
        cls = type(get_renderer(name))
        lineage["renderers"][name] = reference(cls)
        for parent in cls.__mro__:
            if parent.__module__.startswith("truthound."):
                source_types[reference(parent)] = identity(parent)
    for cls in (LineageGraph, ImpactAnalyzer, LineageTracker):
        source_types[reference(cls)] = identity(cls)
    detectors = {}
    for name in drift.__all__:
        cls = getattr(drift, name)
        if inspect.isclass(cls) and issubclass(cls, DriftDetector) and not inspect.isabstract(cls):
            options, sources = constructor(cls, validator_options=False)
            source_types.update(sources)
            detectors[name] = {
                "target": reference(cls),
                "fields": options,
                "source_types": list(sources),
            }
    profiling = {"config": describe(profiler.ProfilerConfig), "components": {}}
    for group, names in {
        "column_analyzers": (
            "BasicStatsAnalyzer",
            "NumericAnalyzer",
            "StringAnalyzer",
            "DatetimeAnalyzer",
            "ValueFrequencyAnalyzer",
            "PatternAnalyzer",
        ),
        "table_analyzers": ("DuplicateRowAnalyzer", "MemoryEstimator", "CorrelationAnalyzer"),
        "type_inferrers": (
            "PhysicalTypeInferrer",
            "PatternBasedTypeInferrer",
            "CardinalityTypeInferrer",
        ),
        "profiler": ("DataProfiler",),
        "rule_generators": (
            "SchemaRuleGenerator",
            "StatsRuleGenerator",
            "PatternRuleGenerator",
            "MLRuleGenerator",
        ),
    }.items():
        profiling["components"][group] = {}
        for name in names:
            cls = getattr(profiler, name)
            options, sources = constructor(cls, validator_options=False)
            source_types.update(sources)
            profiling["components"][group][name] = {"target": reference(cls), "fields": options}
    record_source(profiler.ColumnProfiler)
    record_source(profiler.GeneratedRule)
    record_source(profiler.TableProfile)
    from truthound.profiler.evolution.changes import SchemaChange, SchemaChangeSummary
    from truthound.profiler.evolution.detector import SchemaEvolutionDetector

    for cls in (SchemaEvolutionDetector, SchemaChange, SchemaChangeSummary):
        record_source(cls)
    from truthound.profiler.evolution import compatibility, rename_detector

    for name in (
        "SchemaCompatibilityAnalyzer",
        "CompatibilityResult",
        "TypeCompatibilityChecker",
        "NullabilityCompatibilityChecker",
        "ColumnPresenceChecker",
    ):
        record_source(getattr(compatibility, name))

    for name in (
        "ColumnRenameDetector",
        "RenameCandidate",
        "RenameDetectionResult",
        "LevenshteinSimilarity",
        "TokenSimilarity",
        "JaroWinklerSimilarity",
        "NgramSimilarity",
        "CompositeSimilarity",
    ):
        record_source(getattr(rename_detector, name))
    suite = {
        "evidence_policy": describe(EvidencePolicy),
        "severity_policy": describe(SeverityPolicy),
    }
    for cls in (CheckSpec, ValidationSuite, SchemaSpec):
        record_source(cls)
    return {
        "contract_version": 1,
        "validators": rows,
        "abstract_aliases": abstract,
        "reporters": {
            key: describe(cls)
            for key, cls in {
                "json": JSONReporterConfig,
                "markdown": MarkdownReporterConfig,
                "console": ConsoleReporterConfig,
                "html": HTMLReporterConfig,
            }.items()
        },
        "record_types": RECORDS,
        "common_fields": common_fields,
        "source_ast_sha256": {**source_types, **RECORD_SOURCES},
        "models": models,
        "lineage": lineage,
        "drift_detectors": detectors,
        "profiling": profiling,
        "suite": suite,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    target = Path(__file__).resolve().parents[1] / "src/truthound/mcp/validator_contracts.json"
    content = json.dumps(generate(), sort_keys=True, ensure_ascii=False, indent=2) + "\n"
    if args.check:
        if target.read_text(encoding="utf-8") != content:
            raise SystemExit("MCP constructor contracts drifted; regenerate and review")
    else:
        target.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
