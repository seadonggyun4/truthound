"""Tests for enterprise audit logging system."""

import json
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from truthound.infrastructure.audit import (
    # Configuration
    EnterpriseAuditConfig,
    # Storage backends
    ElasticsearchAuditStorage,
    S3AuditStorage,
    KafkaAuditStorage,
    # Compliance
    ComplianceReporter,
    RetentionPolicy,
    # Logger
    EnterpriseAuditLogger,
    # Factory
    get_audit_logger,
    configure_audit,
)

from truthound.audit import (
    AuditEventType,
    AuditSeverity,
    AuditOutcome,
    AuditCategory,
    AuditActor,
    AuditResource,
    AuditEvent,
    AuditEventBuilder,
    MemoryAuditStorage,
)


class TestEnterpriseAuditConfig:
    """Tests for EnterpriseAuditConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = EnterpriseAuditConfig()

        assert config.enabled is True
        assert "memory" in config.storage_backends
        assert config.require_checksum is True

    def test_production_config(self):
        """Test production configuration."""
        config = EnterpriseAuditConfig.production("my-service")

        assert config.service_name == "my-service"
        assert config.environment == "production"
        assert "sqlite" in config.storage_backends
        assert config.retention_days == 365

    def test_compliance_config(self):
        """Test compliance configuration."""
        config = EnterpriseAuditConfig.compliance(
            "my-service",
            standards=["SOC2", "GDPR"],
        )

        assert config.compliance_standards == ["SOC2", "GDPR"]
        assert config.require_signing is True
        assert config.retention_days == 2555  # 7 years
        assert config.archive_to_cold_storage is True


class TestRetentionPolicy:
    """Tests for RetentionPolicy."""

    def test_default_policy(self):
        """Test default retention policy."""
        policy = RetentionPolicy.default()

        assert policy.name == "default"
        assert policy.retention_days == 365

    def test_soc2_policy(self):
        """Test SOC 2 compliance policy."""
        policy = RetentionPolicy.compliance_soc2()

        assert policy.name == "soc2"
        assert policy.retention_days == 2555  # 7 years
        assert policy.archive_after_days == 90
        assert policy.archive_storage == "s3"

    def test_gdpr_policy(self):
        """Test GDPR compliance policy."""
        policy = RetentionPolicy.compliance_gdpr()

        assert policy.name == "gdpr"
        assert policy.retention_days == 365  # Minimal retention
        assert policy.archive_after_days == 30

    def test_hipaa_policy(self):
        """Test HIPAA compliance policy."""
        policy = RetentionPolicy.compliance_hipaa()

        assert policy.name == "hipaa"
        assert policy.retention_days == 2190  # 6 years


class TestEnterpriseAuditLogger:
    """Tests for EnterpriseAuditLogger."""

    def setup_method(self):
        """Create fresh logger for each test."""
        config = EnterpriseAuditConfig(
            enabled=True,
            storage_backends=["memory"],
            mask_sensitive_data=False,
            require_checksum=True,
        )
        self.storage = MemoryAuditStorage()
        self.logger = EnterpriseAuditLogger(config, storages=[self.storage])

    def test_log_event(self):
        """Test logging an event."""
        event = (
            AuditEventBuilder()
            .set_type(AuditEventType.CREATE)
            .set_action("create_user")
            .set_outcome(AuditOutcome.SUCCESS)
            .set_resource(id="user:123", type="user")
            .build()
        )

        self.logger.log(event)

        # Query the event
        events = self.storage.query(limit=10)
        assert len(events) == 1
        assert events[0].event_type == AuditEventType.CREATE
        assert events[0].action == "create_user"

    def test_log_operation(self):
        """Test logging an operation."""
        self.logger.log_operation(
            operation="validate_data",
            resource="dataset:users",
            outcome="success",
            details={"rows": 10000, "issues": 5},
            duration_ms=1500.0,
        )

        events = self.storage.query(limit=10)
        assert len(events) == 1
        assert events[0].action == "validate_data"
        assert events[0].data["rows"] == 10000

    def test_log_validation(self):
        """Test logging a validation."""
        self.logger.log_validation(
            dataset="users",
            success=True,
            rows=5000,
            issues=3,
            duration_ms=2000.0,
            validators=["not_null", "unique"],
        )

        events = self.storage.query(limit=10)
        assert len(events) == 1
        assert events[0].event_type == AuditEventType.VALIDATION_COMPLETE
        assert events[0].data["rows"] == 5000
        assert events[0].data["issues"] == 3

    def test_log_validation_failure(self):
        """Test logging a validation failure."""
        self.logger.log_validation(
            dataset="orders",
            success=False,
            rows=1000,
            issues=50,
            duration_ms=500.0,
        )

        events = self.storage.query(limit=10)
        assert len(events) == 1
        assert events[0].event_type == AuditEventType.VALIDATION_FAILED
        assert events[0].outcome == AuditOutcome.FAILURE

    def test_log_checkpoint(self):
        """Test logging a checkpoint."""
        self.logger.log_checkpoint(
            checkpoint="daily_check",
            success=True,
            validators_run=10,
            issues=5,
            duration_ms=30000.0,
        )

        events = self.storage.query(limit=10)
        assert len(events) == 1
        assert events[0].event_type == AuditEventType.CHECKPOINT_RUN
        assert events[0].data["validators_run"] == 10

    def test_operation_context(self):
        """Test operation context manager."""
        with self.logger.operation_context("process_data", "dataset:test"):
            time.sleep(0.01)

        events = self.storage.query(limit=10)
        assert len(events) == 1
        assert events[0].action == "process_data"
        assert events[0].outcome == AuditOutcome.SUCCESS

    def test_operation_context_failure(self):
        """Test operation context manager with failure."""
        with pytest.raises(ValueError):
            with self.logger.operation_context("failing_operation", "resource:test"):
                raise ValueError("Test error")

        events = self.storage.query(limit=10)
        assert len(events) == 1
        assert events[0].outcome == AuditOutcome.FAILURE
        assert "error" in events[0].data

    def test_correlation_propagation(self):
        """Test correlation ID propagation."""
        from truthound.infrastructure.logging import correlation_context

        with correlation_context(request_id="req-123", trace_id="trace-456"):
            self.logger.log_operation(
                operation="test_op",
                resource="test:resource",
            )

        events = self.storage.query(limit=10)
        assert len(events) == 1
        assert events[0].context.correlation_id == "req-123"

    def test_checksum_generation(self):
        """Test checksum is generated."""
        self.logger.log_operation(
            operation="test",
            resource="test:resource",
        )

        events = self.storage.query(limit=10)
        assert len(events) == 1
        assert len(events[0].checksum) > 0

    def test_query_events(self):
        """Test querying events."""
        # Log multiple events
        for i in range(5):
            self.logger.log_operation(
                operation=f"operation_{i}",
                resource="test:resource",
            )

        events = self.logger.query(limit=3)
        assert len(events) == 3

    def test_disabled_logger(self):
        """Test disabled logger doesn't log."""
        config = EnterpriseAuditConfig(enabled=False)
        storage = MemoryAuditStorage()
        logger = EnterpriseAuditLogger(config, storages=[storage])

        logger.log_operation("test", "test:resource")

        events = storage.query(limit=10)
        assert len(events) == 0


class TestComplianceReporter:
    """Tests for ComplianceReporter."""

    def setup_method(self):
        """Create fresh storage and reporter for each test."""
        self.storage = MemoryAuditStorage()
        self.reporter = ComplianceReporter(self.storage)

    def _create_events(self):
        """Create test events."""
        now = datetime.now(timezone.utc)

        events = [
            AuditEvent(
                event_type=AuditEventType.LOGIN,
                category=AuditCategory.AUTHENTICATION,
                action="login",
                outcome=AuditOutcome.SUCCESS,
                actor=AuditActor(id="user:1"),
            ),
            AuditEvent(
                event_type=AuditEventType.LOGIN_FAILED,
                category=AuditCategory.AUTHENTICATION,
                action="login",
                outcome=AuditOutcome.FAILURE,
                actor=AuditActor(id="user:2"),
            ),
            AuditEvent(
                event_type=AuditEventType.ACCESS_DENIED,
                category=AuditCategory.AUTHORIZATION,
                action="access",
                outcome=AuditOutcome.FAILURE,
                actor=AuditActor(id="user:3"),
            ),
            AuditEvent(
                event_type=AuditEventType.READ,
                category=AuditCategory.DATA_ACCESS,
                action="read",
                outcome=AuditOutcome.SUCCESS,
                actor=AuditActor(id="user:1"),
                resource=AuditResource(id="doc:1", type="document"),
            ),
            AuditEvent(
                event_type=AuditEventType.UPDATE,
                category=AuditCategory.DATA_MODIFICATION,
                action="update",
                outcome=AuditOutcome.SUCCESS,
                actor=AuditActor(id="user:1"),
                resource=AuditResource(id="doc:1", type="document"),
            ),
        ]

        for event in events:
            self.storage.write(event)

    def test_generate_report(self):
        """Test generating a report."""
        self._create_events()

        start_date = datetime.now(timezone.utc) - timedelta(hours=1)
        end_date = datetime.now(timezone.utc) + timedelta(hours=1)

        report = self.reporter.generate_report(
            start_date=start_date,
            end_date=end_date,
        )

        assert "metadata" in report
        assert "summary" in report
        assert "breakdown" in report
        assert report["summary"]["total_events"] == 5
        assert report["summary"]["security_events"] == 2  # login_failed + access_denied

    def test_soc2_report(self):
        """Test SOC 2 compliance report."""
        self._create_events()

        start_date = datetime.now(timezone.utc) - timedelta(hours=1)
        end_date = datetime.now(timezone.utc) + timedelta(hours=1)

        report = self.reporter.generate_report(
            start_date=start_date,
            end_date=end_date,
            standard="SOC2",
        )

        assert report["compliance"]["standard"] == "SOC2"
        assert "controls" in report["compliance"]
        assert "CC6.1" in report["compliance"]["controls"]
        assert "CC6.2" in report["compliance"]["controls"]

    def test_gdpr_report(self):
        """Test GDPR compliance report."""
        self._create_events()

        start_date = datetime.now(timezone.utc) - timedelta(hours=1)
        end_date = datetime.now(timezone.utc) + timedelta(hours=1)

        report = self.reporter.generate_report(
            start_date=start_date,
            end_date=end_date,
            standard="GDPR",
        )

        assert report["compliance"]["standard"] == "GDPR"
        assert "data_access" in report["compliance"]["controls"]

    def test_include_details(self):
        """Test including event details."""
        self._create_events()

        start_date = datetime.now(timezone.utc) - timedelta(hours=1)
        end_date = datetime.now(timezone.utc) + timedelta(hours=1)

        report = self.reporter.generate_report(
            start_date=start_date,
            end_date=end_date,
            include_details=True,
        )

        assert "events" in report
        assert len(report["events"]) == 5


class TestGlobalAuditLogger:
    """Tests for global audit logger functions."""

    def test_get_audit_logger(self):
        """Test getting global audit logger."""
        logger1 = get_audit_logger()
        logger2 = get_audit_logger()

        assert logger1 is logger2

    def test_configure_audit(self):
        """Test configuring global audit logger."""
        logger = configure_audit(
            service_name="test-service",
            environment="testing",
            storage_backends=["memory"],
        )

        assert logger._config.service_name == "test-service"
        assert logger._config.environment == "testing"


class TestElasticsearchAuditStorage:
    """Tests for ElasticsearchAuditStorage."""

    def test_create_storage(self):
        """Test creating Elasticsearch storage."""
        storage = ElasticsearchAuditStorage(
            url="http://localhost:9200",
            index_prefix="test-audit",
        )

        assert storage._url == "http://localhost:9200"
        assert storage._index_prefix == "test-audit"

        storage.close()

    def test_get_index_name(self):
        """Test index name generation."""
        storage = ElasticsearchAuditStorage(
            url="http://localhost:9200",
            index_prefix="audit",
        )

        timestamp = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        index_name = storage._get_index_name(timestamp)

        assert index_name == "audit-2024.01.15"

        storage.close()


class TestS3AuditStorage:
    """Tests for S3AuditStorage."""

    def test_create_storage(self):
        """Test creating S3 storage."""
        storage = S3AuditStorage(
            bucket="my-bucket",
            prefix="audit/",
            region="us-east-1",
        )

        assert storage._bucket == "my-bucket"
        assert storage._prefix == "audit/"

        storage.close()
