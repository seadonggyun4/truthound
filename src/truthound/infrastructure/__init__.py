"""Enterprise infrastructure module for Truthound.

This module provides unified access to all infrastructure components:
- Structured logging with correlation IDs
- Prometheus metrics with validator instrumentation
- Environment-based configuration management
- Comprehensive audit logging
- Data encryption at rest

Usage:
    >>> from truthound.infrastructure import (
    ...     get_logger, configure_logging,
    ...     get_metrics, configure_metrics,
    ...     get_config, Environment,
    ...     get_audit_logger,
    ...     get_encryptor,
    ... )
    >>>
    >>> # Configure for production
    >>> configure_logging(environment="production", format="json")
    >>> configure_metrics(enable_http=True, port=9090)
    >>>
    >>> # Use with correlation
    >>> with correlation_context(request_id="req-123"):
    ...     logger.info("Processing request")
    ...     metrics.validation_count.inc()
"""

# Logging
from truthound.infrastructure.logging import (
    # Core
    EnterpriseLogger,
    LogConfig,
    LogLevel,
    # Correlation
    CorrelationContext,
    correlation_context,
    get_correlation_id,
    set_correlation_id,
    generate_correlation_id,
    # Sinks
    LogSink,
    ConsoleSink,
    FileSink,
    JsonFileSink,
    ElasticsearchSink,
    LokiSink,
    FluentdSink,
    # Factory
    get_logger,
    configure_logging,
    reset_logging,
)

# Metrics
from truthound.infrastructure.metrics import (
    # Core
    MetricsManager,
    MetricsConfig,
    # Validator metrics
    ValidatorMetrics,
    CheckpointMetrics,
    DataSourceMetrics,
    # HTTP server
    MetricsServer,
    # Factory
    get_metrics,
    configure_metrics,
    reset_metrics,
)

# Configuration
from truthound.infrastructure.config import (
    # Core
    ConfigManager,
    ConfigProfile,
    Environment,
    # Sources
    ConfigSource,
    EnvConfigSource,
    FileConfigSource,
    VaultConfigSource,
    AwsSecretsSource,
    # Validation
    ConfigSchema,
    ConfigValidator,
    # Factory
    get_config,
    load_config,
    reload_config,
)

# Audit (re-export from existing module with extensions)
from truthound.infrastructure.audit import (
    # Extended logger
    EnterpriseAuditLogger,
    AuditConfig as EnterpriseAuditConfig,
    # Additional storage
    ElasticsearchAuditStorage,
    S3AuditStorage,
    KafkaAuditStorage,
    # Compliance
    ComplianceReporter,
    RetentionPolicy,
    # Factory
    get_audit_logger,
    configure_audit,
)

# Encryption (re-export from existing module with extensions)
from truthound.infrastructure.encryption import (
    # Extended encryption
    AtRestEncryption,
    FieldLevelEncryption,
    # Key management
    VaultKeyProvider,
    AwsKmsProvider,
    GcpKmsProvider,
    AzureKeyVaultProvider,
    # Factory
    get_encryptor,
    configure_encryption,
)


__all__ = [
    # Logging
    "EnterpriseLogger",
    "LogConfig",
    "LogLevel",
    "CorrelationContext",
    "correlation_context",
    "get_correlation_id",
    "set_correlation_id",
    "generate_correlation_id",
    "LogSink",
    "ConsoleSink",
    "FileSink",
    "JsonFileSink",
    "ElasticsearchSink",
    "LokiSink",
    "FluentdSink",
    "get_logger",
    "configure_logging",
    "reset_logging",
    # Metrics
    "MetricsManager",
    "MetricsConfig",
    "ValidatorMetrics",
    "CheckpointMetrics",
    "DataSourceMetrics",
    "MetricsServer",
    "get_metrics",
    "configure_metrics",
    "reset_metrics",
    # Configuration
    "ConfigManager",
    "ConfigProfile",
    "Environment",
    "ConfigSource",
    "EnvConfigSource",
    "FileConfigSource",
    "VaultConfigSource",
    "AwsSecretsSource",
    "ConfigSchema",
    "ConfigValidator",
    "get_config",
    "load_config",
    "reload_config",
    # Audit
    "EnterpriseAuditLogger",
    "EnterpriseAuditConfig",
    "ElasticsearchAuditStorage",
    "S3AuditStorage",
    "KafkaAuditStorage",
    "ComplianceReporter",
    "RetentionPolicy",
    "get_audit_logger",
    "configure_audit",
    # Encryption
    "AtRestEncryption",
    "FieldLevelEncryption",
    "VaultKeyProvider",
    "AwsKmsProvider",
    "GcpKmsProvider",
    "AzureKeyVaultProvider",
    "get_encryptor",
    "configure_encryption",
]
