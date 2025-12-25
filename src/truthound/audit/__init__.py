"""Audit logging module for Truthound.

This module provides a comprehensive audit logging system for tracking
security-relevant events, data access, and system operations.

Features:
    - Multiple event types (authentication, data access, system events)
    - Pluggable storage backends (memory, file, SQLite, SQL databases)
    - Multiple output formats (JSON, CEF, LEEF, Syslog)
    - Filtering and processing pipelines
    - Privacy controls (data masking, IP anonymization)
    - Middleware for web frameworks (ASGI, WSGI)
    - Decorators for function auditing

Architecture:
    The audit system follows a pipeline design:

    Event Creation
        │
        ├── AuditEventBuilder
        │
        v
    AuditLogger
        │
        ├── Filters (decide what to log)
        │
        ├── Processors (transform/enrich)
        │
        ├── Formatter (serialize)
        │
        v
    Storage (persist)

Usage:
    >>> from truthound.audit import (
    ...     AuditLogger, AuditConfig,
    ...     AuditEventType, AuditActor, AuditResource,
    ...     audited,
    ... )
    >>>
    >>> # Simple logging
    >>> logger = AuditLogger(config=AuditConfig.production("my-service"))
    >>> logger.log(
    ...     event_type=AuditEventType.UPDATE,
    ...     action="update_user",
    ...     actor=AuditActor(id="user:123"),
    ...     resource=AuditResource(id="user:456", type="user"),
    ... )
    >>>
    >>> # Using decorator
    >>> @audited(action="create_report")
    ... def create_report(data):
    ...     return Report.create(data)
    >>>
    >>> # Query audit log
    >>> events = logger.query(
    ...     event_types=[AuditEventType.UPDATE],
    ...     actor_id="user:123",
    ...     limit=10,
    ... )
"""

# Core types and configuration
from truthound.audit.core import (
    # Enums
    AuditEventType,
    AuditSeverity,
    AuditOutcome,
    AuditCategory,
    # Data types
    AuditActor,
    AuditResource,
    AuditContext,
    AuditEvent,
    # Configuration
    AuditConfig,
    # Exceptions
    AuditError,
    AuditStorageError,
    AuditValidationError,
    AuditConfigError,
    # Interfaces
    AuditStorage,
    AuditFormatter,
    AuditFilter,
    AuditProcessor,
    # Builder
    AuditEventBuilder,
    # Utilities
    current_timestamp,
    generate_event_id,
    mask_sensitive_value,
    anonymize_ip_address,
)

# Storage backends
from truthound.audit.storage import (
    MemoryAuditStorage,
    FileAuditStorage,
    FileStorageConfig,
    SQLiteAuditStorage,
    AsyncBufferedStorage,
    CompositeAuditStorage,
    create_storage,
)

# Formatters
from truthound.audit.formatters import (
    JSONFormatter,
    CEFFormatter,
    LEEFFormatter,
    SyslogFormatter,
    HumanFormatter,
    create_formatter,
)

# Filters and processors
from truthound.audit.filters import (
    # Filters
    SeverityFilter,
    EventTypeFilter,
    CategoryFilter,
    ActorFilter,
    ActionFilter,
    OutcomeFilter,
    SamplingFilter,
    RateLimitFilter,
    CompositeFilter,
    CallableFilter,
    # Processors
    PrivacyProcessor,
    EnrichmentProcessor,
    ChecksumProcessor,
    TaggingProcessor,
    CompositeProcessor,
    # Factory functions
    create_filter_from_config,
    create_processor_from_config,
)

# Logger
from truthound.audit.logger import (
    AuditLogger,
    AuditLoggerRegistry,
    get_audit_logger,
    configure_audit,
    audit_context,
    audit_operation,
    audited,
    audited_async,
)

# Middleware
from truthound.audit.middleware import (
    AuditMiddleware,
    ASGIAuditMiddleware,
    WSGIAuditMiddleware,
    DatabaseAuditConfig,
    DatabaseAuditHook,
    CheckpointAuditHook,
    create_asgi_middleware,
    create_wsgi_middleware,
)


__all__ = [
    # Enums
    "AuditEventType",
    "AuditSeverity",
    "AuditOutcome",
    "AuditCategory",
    # Data types
    "AuditActor",
    "AuditResource",
    "AuditContext",
    "AuditEvent",
    # Configuration
    "AuditConfig",
    # Exceptions
    "AuditError",
    "AuditStorageError",
    "AuditValidationError",
    "AuditConfigError",
    # Interfaces
    "AuditStorage",
    "AuditFormatter",
    "AuditFilter",
    "AuditProcessor",
    # Builder
    "AuditEventBuilder",
    # Utilities
    "current_timestamp",
    "generate_event_id",
    "mask_sensitive_value",
    "anonymize_ip_address",
    # Storage
    "MemoryAuditStorage",
    "FileAuditStorage",
    "FileStorageConfig",
    "SQLiteAuditStorage",
    "AsyncBufferedStorage",
    "CompositeAuditStorage",
    "create_storage",
    # Formatters
    "JSONFormatter",
    "CEFFormatter",
    "LEEFFormatter",
    "SyslogFormatter",
    "HumanFormatter",
    "create_formatter",
    # Filters
    "SeverityFilter",
    "EventTypeFilter",
    "CategoryFilter",
    "ActorFilter",
    "ActionFilter",
    "OutcomeFilter",
    "SamplingFilter",
    "RateLimitFilter",
    "CompositeFilter",
    "CallableFilter",
    # Processors
    "PrivacyProcessor",
    "EnrichmentProcessor",
    "ChecksumProcessor",
    "TaggingProcessor",
    "CompositeProcessor",
    "create_filter_from_config",
    "create_processor_from_config",
    # Logger
    "AuditLogger",
    "AuditLoggerRegistry",
    "get_audit_logger",
    "configure_audit",
    "audit_context",
    "audit_operation",
    "audited",
    "audited_async",
    # Middleware
    "AuditMiddleware",
    "ASGIAuditMiddleware",
    "WSGIAuditMiddleware",
    "DatabaseAuditConfig",
    "DatabaseAuditHook",
    "CheckpointAuditHook",
    "create_asgi_middleware",
    "create_wsgi_middleware",
]
