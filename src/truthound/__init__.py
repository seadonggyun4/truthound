"""Truthound public package surface.

Truthound keeps the root package intentionally small: the validation facade
and the core result/context types live here, while advanced subsystems are
imported through their namespaces.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from truthound._ai_support import AISupportStatus, get_ai_support_status, has_ai_support
from truthound._lazy import (
    TRUTHOUND_IMPORT_MAP,
    get_truthound_import_metrics,
    truthound_getattr,
)
from truthound._version import resolve_truthound_version
from truthound.api import check, mask, profile, read, scan
from truthound.context import TruthoundContext, get_context
from truthound.core import CheckResult, CheckSpec, SchemaSpec, ValidationRunResult, ValidationSuite
from truthound.decorators import validator
from truthound.schema import Schema, learn

__version__ = resolve_truthound_version()


def __getattr__(name: str) -> Any:
    if name in TRUTHOUND_IMPORT_MAP:
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
        'TruthoundContext',
        'get_context',
        'CheckSpec',
        'SchemaSpec',
        'ValidationSuite',
        'ValidationRunResult',
        'CheckResult',
        '__version__',
        'get_truthound_import_metrics',
        'has_ai_support',
        'get_ai_support_status',
        'AISupportStatus',
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


__all__ = [
    'check',
    'scan',
    'mask',
    'profile',
    'read',
    'learn',
    'validator',
    'Schema',
    'TruthoundContext',
    'get_context',
    'CheckSpec',
    'SchemaSpec',
    'ValidationSuite',
    'ValidationRunResult',
    'CheckResult',
    '__version__',
    'get_truthound_import_metrics',
    'has_ai_support',
    'get_ai_support_status',
    'AISupportStatus',
]
