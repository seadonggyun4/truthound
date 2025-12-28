"""Plugin architecture for Truthound.

Phase 9: Plugin Architecture

This module provides a comprehensive plugin system that allows extending
Truthound's functionality through external packages.

Plugin Types:
    - validators: Custom data validation rules
    - reporters: Output format handlers
    - datasources: Data connection backends
    - profilers: Data profiling analyzers
    - hooks: Event-based extension points

Enterprise Features:
    - Security: Sandbox execution, plugin signing, trust store
    - Versioning: Semantic versioning, version constraints
    - Dependencies: Dependency resolution, cycle detection
    - Lifecycle: Hot reload, graceful shutdown
    - Documentation: Auto-generated docs from plugins

Example:
    >>> from truthound.plugins import PluginManager
    >>> manager = PluginManager()
    >>> manager.discover_plugins()
    >>> manager.load_plugin("my-custom-plugin")

Enterprise Example:
    >>> from truthound.plugins import create_enterprise_manager
    >>> manager = create_enterprise_manager()
    >>> await manager.load("my-plugin", verify_signature=True)
"""

from __future__ import annotations

from truthound.plugins.base import (
    Plugin,
    PluginConfig,
    PluginInfo,
    PluginType,
    PluginState,
    PluginError,
    PluginLoadError,
    PluginNotFoundError,
    PluginDependencyError,
    PluginCompatibilityError,
    ValidatorPlugin,
    ReporterPlugin,
    DataSourcePlugin,
    HookPlugin,
)
from truthound.plugins.manager import PluginManager
from truthound.plugins.registry import PluginRegistry
from truthound.plugins.hooks import (
    HookManager,
    Hook,
    HookType,
    hook,
    before_validation,
    after_validation,
    before_profile,
    after_profile,
    on_report_generate,
)
from truthound.plugins.discovery import PluginDiscovery
from truthound.plugins.manager import get_plugin_manager

# Enterprise security components
from truthound.plugins.security.protocols import (
    IsolationLevel,
    TrustLevel,
    ResourceLimits,
    SecurityPolicy,
    SignatureInfo,
    VerificationResult,
)
from truthound.plugins.security.policies import (
    SecurityPolicyPresets,
    create_policy,
    get_preset,
    list_presets,
)
from truthound.plugins.security.sandbox.factory import SandboxFactory
from truthound.plugins.security.signing.service import SigningServiceImpl, SignatureAlgorithm
from truthound.plugins.security.signing.trust_store import TrustStoreImpl
from truthound.plugins.security.signing.verifier import (
    create_verification_chain,
    VerificationChainBuilder,
)

# Versioning components
from truthound.plugins.versioning.constraints import VersionConstraint, parse_constraint

# Dependency management
from truthound.plugins.dependencies.graph import DependencyGraph, DependencyNode, DependencyType
from truthound.plugins.dependencies.resolver import DependencyResolver, ResolutionResult

# Lifecycle management
from truthound.plugins.lifecycle.manager import LifecycleManager, LifecycleState, LifecycleEvent
from truthound.plugins.lifecycle.hot_reload import HotReloadManager, ReloadStrategy

# Documentation
from truthound.plugins.docs.extractor import DocumentationExtractor, PluginDocumentation
from truthound.plugins.docs.renderer import (
    MarkdownRenderer,
    HtmlRenderer,
    JsonRenderer,
    render_documentation,
)

# Enterprise manager facade
from truthound.plugins.enterprise_manager import (
    EnterprisePluginManager,
    EnterprisePluginManagerConfig,
    create_enterprise_manager,
)

__all__ = [
    # Core classes
    "Plugin",
    "PluginConfig",
    "PluginInfo",
    "PluginType",
    "PluginState",
    # Specialized plugin base classes
    "ValidatorPlugin",
    "ReporterPlugin",
    "DataSourcePlugin",
    "HookPlugin",
    # Manager
    "PluginManager",
    "PluginRegistry",
    "PluginDiscovery",
    "get_plugin_manager",
    # Hooks
    "HookManager",
    "Hook",
    "HookType",
    "hook",
    "before_validation",
    "after_validation",
    "before_profile",
    "after_profile",
    "on_report_generate",
    # Errors
    "PluginError",
    "PluginLoadError",
    "PluginNotFoundError",
    "PluginDependencyError",
    "PluginCompatibilityError",
    # === Enterprise Features ===
    # Security protocols
    "IsolationLevel",
    "TrustLevel",
    "ResourceLimits",
    "SecurityPolicy",
    "SignatureInfo",
    "VerificationResult",
    # Security policies
    "SecurityPolicyPresets",
    "create_policy",
    "get_preset",
    "list_presets",
    # Sandbox
    "SandboxFactory",
    # Signing
    "SigningServiceImpl",
    "SignatureAlgorithm",
    "TrustStoreImpl",
    "create_verification_chain",
    "VerificationChainBuilder",
    # Versioning
    "VersionConstraint",
    "parse_constraint",
    # Dependencies
    "DependencyGraph",
    "DependencyNode",
    "DependencyType",
    "DependencyResolver",
    "ResolutionResult",
    # Lifecycle
    "LifecycleManager",
    "LifecycleState",
    "LifecycleEvent",
    "HotReloadManager",
    "ReloadStrategy",
    # Documentation
    "DocumentationExtractor",
    "PluginDocumentation",
    "MarkdownRenderer",
    "HtmlRenderer",
    "JsonRenderer",
    "render_documentation",
    # Enterprise manager
    "EnterprisePluginManager",
    "EnterprisePluginManagerConfig",
    "create_enterprise_manager",
]
