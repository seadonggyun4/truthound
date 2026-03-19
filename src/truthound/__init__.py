"""Truthound public package surface.

Truthound 2.0 keeps the familiar top-level validation entry points while
routing execution through the new ``truthound.core`` kernel. Advanced
subsystems remain importable via their namespaces and are lazy-loaded on
first access.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

from truthound.api import check, mask, profile, read, scan
from truthound.core import CheckResult, CheckSpec, SchemaSpec, ValidationRunResult, ValidationSuite
from truthound.decorators import validator
from truthound.schema import Schema, learn
from truthound._lazy import (
    TRUTHOUND_IMPORT_MAP,
    truthound_getattr,
    get_truthound_import_metrics,
)

try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version('truthound')
except PackageNotFoundError:
    __version__ = '0.0.0.dev'


_DEPRECATED_ROOT_EXPORTS = {
    'datasources',
    'execution',
    'checkpoint',
    'profiler',
    'datadocs',
    'ml',
    'lineage',
    'realtime',
    'compare',
    'Report',
    'ResultFormat',
    'ResultFormatConfig',
    'ValidationIssue',
}


def __getattr__(name: str) -> Any:
    if name in TRUTHOUND_IMPORT_MAP:
        if name in _DEPRECATED_ROOT_EXPORTS:
            warnings.warn(
                f"'truthound.{name}' top-level access is deprecated. "
                f"Import from 'truthound.{name}' or the subsystem namespace directly.",
                DeprecationWarning,
                stacklevel=2,
            )
        return truthound_getattr(name)
    raise AttributeError(f"module 'truthound' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted([
        'check',
        'scan',
        'mask',
        'profile',
        'read',
        'learn',
        'validator',
        'Schema',
        'CheckSpec',
        'SchemaSpec',
        'ValidationSuite',
        'ValidationRunResult',
        'CheckResult',
        '__version__',
        'get_truthound_import_metrics',
    ])


if TYPE_CHECKING:
    from truthound import checkpoint as checkpoint
    from truthound import datadocs as datadocs
    from truthound import datasources as datasources
    from truthound import execution as execution
    from truthound import lineage as lineage
    from truthound import ml as ml
    from truthound import profiler as profiler
    from truthound import realtime as realtime
    from truthound.drift import compare as compare
    from truthound.report import Report as Report
    from truthound.types import ResultFormat as ResultFormat
    from truthound.types import ResultFormatConfig as ResultFormatConfig
    from truthound.validators.base import ValidationIssue as ValidationIssue


__all__ = [
    'check',
    'scan',
    'mask',
    'profile',
    'read',
    'learn',
    'validator',
    'Schema',
    'CheckSpec',
    'SchemaSpec',
    'ValidationSuite',
    'ValidationRunResult',
    'CheckResult',
    '__version__',
    'get_truthound_import_metrics',
]
