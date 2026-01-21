# Audit Logging Configuration

Truthound provides enterprise audit logging for compliance and operational tracking.

## Quick Start

```python
from truthound.infrastructure.audit import (
    configure_audit,
    get_audit_logger,
)

# Configure audit logging
# Note: Additional options like include_data_preview, redact_sensitive
# are passed via **kwargs to EnterpriseAuditConfig
audit = configure_audit(
    service_name="my-service",
    environment="production",
    storage_backends=["memory"],
)

# Or get the global audit logger
audit = get_audit_logger()

# Log audit events
audit.log_validation_start(checkpoint_name="daily_check", user="system")
```

## EnterpriseAuditConfig

```python
from truthound.infrastructure.audit import EnterpriseAuditConfig, RetentionPolicy

config = EnterpriseAuditConfig(
    enabled=True,
    include_data_preview=False,        # Include data samples in logs
    redact_sensitive=True,             # Redact sensitive fields
    sensitive_fields=[
        "password", "ssn", "credit_card",
        "api_key", "secret", "token",
    ],
    retention=RetentionPolicy(name="default", retention_days=365),

    # Storage backends
    storage_backends=["memory"],
    # Options: memory, sqlite, elasticsearch, s3, kafka

    # Elasticsearch configuration
    elasticsearch_url="",
    elasticsearch_index_prefix="truthound-audit",
    elasticsearch_username="",
    elasticsearch_password="",

    # S3 configuration
    s3_bucket="",
    s3_prefix="audit/",
    s3_region="",

    # Kafka configuration
    kafka_bootstrap_servers="",
    kafka_topic="truthound-audit",

    # Compliance
    compliance_standards=[],           # SOC2, GDPR, HIPAA
    require_checksum=True,
    require_signing=False,

    # Retention
    retention_policy="default",
    archive_to_cold_storage=False,
    cold_storage_after_days=90,

    # Performance
    async_write=True,
    batch_size=100,
    flush_interval=5.0,
)
```

### Presets

```python
# Production configuration
config = EnterpriseAuditConfig.production("truthound")

# Compliance configuration (7-year retention)
config = EnterpriseAuditConfig.compliance("truthound", ["SOC2"])
```

## Retention Policies

```python
from truthound.infrastructure.audit import RetentionPolicy

# Default: 1 year
policy = RetentionPolicy.default()

# SOC 2: 7 years
policy = RetentionPolicy.compliance_soc2()

# GDPR: 1 year (data minimization)
policy = RetentionPolicy.compliance_gdpr()

# HIPAA: 6 years
policy = RetentionPolicy.compliance_hipaa()
```

## Storage Backends

### Elasticsearch

```python
# Configure Elasticsearch storage via configure_audit()
configure_audit(
    service_name="my-service",
    storage_backends=["elasticsearch"],
    elasticsearch_url="https://es.example.com:9200",
)
```

### S3 (Long-term Archival)

```python
# Configure S3 storage via configure_audit()
configure_audit(
    service_name="my-service",
    storage_backends=["s3"],
    s3_bucket="audit-logs-bucket",
)
```

### Kafka (Real-time Streaming)

```python
# Configure Kafka storage via configure_audit()
configure_audit(
    service_name="my-service",
    storage_backends=["kafka"],
    kafka_bootstrap_servers="kafka:9092",
)
```

## Audit Events

### Validation Events

```python
audit = get_audit_logger()

# Validation start
audit.log_validation_start(
    checkpoint_name="daily_check",
    user="system",
    data_source="s3://bucket/data.parquet",
)

# Validation complete
audit.log_validation_complete(
    result=result,
    duration_ms=1500,
    issues_found=15,
)

# Validation failed
audit.log_validation_failed(
    checkpoint_name="daily_check",
    error="Connection timeout",
)
```

### Action Events

```python
# Action executed
audit.log_action_executed(
    action_type="slack",
    status="success",
    target="#data-quality",
)

# Action failed
audit.log_action_failed(
    action_type="email",
    error="SMTP connection refused",
)
```

### Data Access Events

```python
# Data read
audit.log_data_access(
    operation="read",
    data_source="users_table",
    rows_accessed=10000,
    user="analyst@company.com",
)

# Data export
audit.log_data_export(
    format="csv",
    destination="s3://exports/report.csv",
    rows_exported=5000,
)
```

## Compliance Reporting

```python
from truthound.infrastructure.audit import ComplianceReporter

reporter = ComplianceReporter(audit_logger=audit)

# SOC 2 report
soc2_report = reporter.generate_soc2_report(
    date_range=("2024-01-01", "2024-12-31"),
)

# GDPR report
gdpr_report = reporter.generate_gdpr_report(
    data_subject="user@example.com",
)

# HIPAA report
hipaa_report = reporter.generate_hipaa_report(
    date_range=("2024-01-01", "2024-12-31"),
)
```

## Store Observability Audit

Additional audit configuration for storage operations:

```python
from truthound.stores.observability import AuditConfig, AuditLogLevel

config = AuditConfig(
    enabled=True,
    level=AuditLogLevel.STANDARD,  # minimal, standard, verbose, debug
    backend="json",                # memory, file, json, elasticsearch, kafka
    file_path="./audit.log",
    include_data_preview=False,
    max_data_preview_size=1024,
    redact_sensitive=True,
    sensitive_fields=[
        "password", "secret", "token",
        "api_key", "ssn", "credit_card",
    ],
    retention_days=90,
    batch_size=100,
    flush_interval_seconds=5.0,
)
```

### Audit Log Levels

| Level | Description |
|-------|-------------|
| `MINIMAL` | Only errors and critical events |
| `STANDARD` | Normal operations and errors |
| `VERBOSE` | Detailed operation logging |
| `DEBUG` | Full debug information |

## Best Practices

### 1. Configure Appropriate Retention

```python
# Production: SOC 2 compliance
config = EnterpriseAuditConfig(
    retention_policy="soc2",
    archive_to_cold_storage=True,
    cold_storage_after_days=90,
)
```

### 2. Redact Sensitive Data

```python
config = EnterpriseAuditConfig(
    redact_sensitive=True,
    sensitive_fields=[
        "password", "ssn", "credit_card",
        "api_key", "secret", "token",
        "email", "phone",  # Add PII fields
    ],
)
```

### 3. Use Multiple Storage Backends

```python
# Real-time to Elasticsearch + archival to S3
configure_audit(
    config,
    storage=[
        ElasticsearchAuditStorage(url="http://elk:9200"),
        S3AuditStorage(bucket="audit-archive"),
    ],
)
```

### 4. Enable Checksums for Compliance

```python
config = EnterpriseAuditConfig(
    require_checksum=True,
    require_signing=True,  # For strict compliance
)
```
