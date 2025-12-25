"""Comprehensive tests for the audit logging module."""

import json
import os
import tempfile
import threading
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch

import pytest

from truthound.audit import (
    # Core types
    AuditEventType,
    AuditSeverity,
    AuditOutcome,
    AuditCategory,
    AuditActor,
    AuditResource,
    AuditContext,
    AuditEvent,
    AuditConfig,
    AuditEventBuilder,
    # Storage
    MemoryAuditStorage,
    FileAuditStorage,
    FileStorageConfig,
    SQLiteAuditStorage,
    CompositeAuditStorage,
    create_storage,
    # Formatters
    JSONFormatter,
    CEFFormatter,
    LEEFFormatter,
    SyslogFormatter,
    HumanFormatter,
    create_formatter,
    # Filters
    SeverityFilter,
    EventTypeFilter,
    CategoryFilter,
    ActorFilter,
    ActionFilter,
    OutcomeFilter,
    SamplingFilter,
    CompositeFilter,
    CallableFilter,
    # Processors
    PrivacyProcessor,
    EnrichmentProcessor,
    ChecksumProcessor,
    TaggingProcessor,
    CompositeProcessor,
    # Logger
    AuditLogger,
    AuditLoggerRegistry,
    get_audit_logger,
    configure_audit,
    audit_context,
    audited,
    # Utilities
    mask_sensitive_value,
    anonymize_ip_address,
)


# =============================================================================
# Test Audit Event
# =============================================================================


class TestAuditEvent:
    """Tests for AuditEvent."""

    def test_event_creation(self):
        """Test creating an audit event."""
        event = AuditEvent(
            event_type=AuditEventType.LOGIN,
            action="login",
            outcome=AuditOutcome.SUCCESS,
        )
        assert event.event_type == AuditEventType.LOGIN
        assert event.action == "login"
        assert event.outcome == AuditOutcome.SUCCESS
        assert event.id is not None

    def test_event_with_actor(self):
        """Test event with actor."""
        actor = AuditActor(id="user:123", name="John", email="john@example.com")
        event = AuditEvent(
            event_type=AuditEventType.UPDATE,
            actor=actor,
        )
        assert event.actor.id == "user:123"
        assert event.actor.name == "John"

    def test_event_with_resource(self):
        """Test event with resource."""
        resource = AuditResource(id="doc:456", type="document", name="report.pdf")
        event = AuditEvent(
            event_type=AuditEventType.READ,
            resource=resource,
        )
        assert event.resource.id == "doc:456"
        assert event.resource.type == "document"

    def test_event_to_dict(self):
        """Test converting event to dictionary."""
        event = AuditEvent(
            event_type=AuditEventType.CREATE,
            action="create_user",
            outcome=AuditOutcome.SUCCESS,
        )
        data = event.to_dict()
        assert data["event_type"] == "create"
        assert data["action"] == "create_user"
        assert data["outcome"] == "success"
        assert "timestamp" in data

    def test_event_from_dict(self):
        """Test creating event from dictionary."""
        data = {
            "id": "event-123",
            "event_type": "update",
            "action": "update_config",
            "outcome": "success",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        event = AuditEvent.from_dict(data)
        assert event.id == "event-123"
        assert event.event_type == AuditEventType.UPDATE
        assert event.action == "update_config"

    def test_event_checksum(self):
        """Test event checksum computation."""
        event = AuditEvent(
            event_type=AuditEventType.LOGIN,
            action="login",
        )
        checksum1 = event.compute_checksum()
        checksum2 = event.compute_checksum()
        assert checksum1 == checksum2
        assert len(checksum1) == 64  # SHA256 hex

    def test_timestamp_properties(self):
        """Test timestamp properties."""
        event = AuditEvent(event_type=AuditEventType.CUSTOM)
        assert isinstance(event.timestamp_iso, str)
        assert isinstance(event.timestamp_unix, float)


# =============================================================================
# Test Audit Actor
# =============================================================================


class TestAuditActor:
    """Tests for AuditActor."""

    def test_actor_creation(self):
        """Test creating an actor."""
        actor = AuditActor(
            id="user:123",
            type="user",
            name="John Doe",
            email="john@example.com",
            ip_address="192.168.1.100",
        )
        assert actor.id == "user:123"
        assert actor.type == "user"
        assert actor.name == "John Doe"

    def test_system_actor(self):
        """Test creating system actor."""
        actor = AuditActor.system()
        assert actor.id == "system"
        assert actor.type == "system"

    def test_anonymous_actor(self):
        """Test creating anonymous actor."""
        actor = AuditActor.anonymous(ip_address="10.0.0.1")
        assert actor.id == "anonymous"
        assert actor.type == "anonymous"
        assert actor.ip_address == "10.0.0.1"

    def test_actor_to_dict(self):
        """Test converting actor to dictionary."""
        actor = AuditActor(id="user:123", roles=["admin", "user"])
        data = actor.to_dict()
        assert data["id"] == "user:123"
        assert "admin" in data["roles"]


# =============================================================================
# Test Audit Config
# =============================================================================


class TestAuditConfig:
    """Tests for AuditConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = AuditConfig()
        assert config.enabled is True
        assert config.min_severity == AuditSeverity.INFO
        assert config.retention_days == 90

    def test_development_config(self):
        """Test development configuration."""
        config = AuditConfig.development()
        assert config.include_debug_events is True
        assert config.async_write is False

    def test_production_config(self):
        """Test production configuration."""
        config = AuditConfig.production("my-service")
        assert config.service_name == "my-service"
        assert config.compute_checksums is True
        assert config.async_write is True


# =============================================================================
# Test Event Builder
# =============================================================================


class TestAuditEventBuilder:
    """Tests for AuditEventBuilder."""

    def test_builder_basic(self):
        """Test basic builder usage."""
        event = (
            AuditEventBuilder()
            .set_type(AuditEventType.CREATE)
            .set_action("create_user")
            .set_outcome(AuditOutcome.SUCCESS)
            .build()
        )
        assert event.event_type == AuditEventType.CREATE
        assert event.action == "create_user"
        assert event.outcome == AuditOutcome.SUCCESS

    def test_builder_with_actor(self):
        """Test builder with actor."""
        event = (
            AuditEventBuilder()
            .set_type(AuditEventType.LOGIN)
            .set_actor(id="user:123", name="john")
            .build()
        )
        assert event.actor.id == "user:123"
        assert event.actor.name == "john"

    def test_builder_with_resource(self):
        """Test builder with resource."""
        event = (
            AuditEventBuilder()
            .set_type(AuditEventType.UPDATE)
            .set_resource(id="doc:456", type="document")
            .build()
        )
        assert event.resource.id == "doc:456"
        assert event.resource.type == "document"

    def test_builder_with_changes(self):
        """Test builder with change data."""
        event = (
            AuditEventBuilder()
            .set_type(AuditEventType.UPDATE)
            .set_changes(
                old_value={"name": "old"},
                new_value={"name": "new"},
            )
            .build()
        )
        assert event.old_value == {"name": "old"}
        assert event.new_value == {"name": "new"}

    def test_builder_with_tags(self):
        """Test builder with tags."""
        event = (
            AuditEventBuilder()
            .set_type(AuditEventType.CUSTOM)
            .add_tag("important")
            .add_tag("security")
            .build()
        )
        assert "important" in event.tags
        assert "security" in event.tags


# =============================================================================
# Test Memory Storage
# =============================================================================


class TestMemoryAuditStorage:
    """Tests for MemoryAuditStorage."""

    def test_write_and_read(self):
        """Test writing and reading events."""
        storage = MemoryAuditStorage()
        event = AuditEvent(event_type=AuditEventType.LOGIN)
        storage.write(event)

        result = storage.read(event.id)
        assert result is not None
        assert result.id == event.id

    def test_write_batch(self):
        """Test batch writing."""
        storage = MemoryAuditStorage()
        events = [
            AuditEvent(event_type=AuditEventType.LOGIN),
            AuditEvent(event_type=AuditEventType.LOGOUT),
        ]
        storage.write_batch(events)

        assert storage.count() == 2

    def test_query_by_time(self):
        """Test querying by time range."""
        storage = MemoryAuditStorage()

        # Create events with different times
        old_event = AuditEvent(
            event_type=AuditEventType.LOGIN,
            timestamp=datetime.now(timezone.utc) - timedelta(hours=2),
        )
        new_event = AuditEvent(
            event_type=AuditEventType.LOGIN,
            timestamp=datetime.now(timezone.utc),
        )
        storage.write_batch([old_event, new_event])

        # Query recent events
        results = storage.query(
            start_time=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        assert len(results) == 1
        assert results[0].id == new_event.id

    def test_query_by_event_type(self):
        """Test querying by event type."""
        storage = MemoryAuditStorage()
        storage.write(AuditEvent(event_type=AuditEventType.LOGIN))
        storage.write(AuditEvent(event_type=AuditEventType.LOGOUT))
        storage.write(AuditEvent(event_type=AuditEventType.LOGIN))

        results = storage.query(event_types=[AuditEventType.LOGIN])
        assert len(results) == 2

    def test_query_by_actor(self):
        """Test querying by actor."""
        storage = MemoryAuditStorage()
        actor1 = AuditActor(id="user:123")
        actor2 = AuditActor(id="user:456")

        storage.write(AuditEvent(event_type=AuditEventType.LOGIN, actor=actor1))
        storage.write(AuditEvent(event_type=AuditEventType.LOGIN, actor=actor2))

        results = storage.query(actor_id="user:123")
        assert len(results) == 1

    def test_max_events_limit(self):
        """Test max events limit."""
        storage = MemoryAuditStorage(max_events=5)

        for i in range(10):
            storage.write(AuditEvent(event_type=AuditEventType.CUSTOM))

        assert storage.count() == 5

    def test_delete_before(self):
        """Test deleting old events."""
        storage = MemoryAuditStorage()

        old_event = AuditEvent(
            event_type=AuditEventType.LOGIN,
            timestamp=datetime.now(timezone.utc) - timedelta(days=10),
        )
        new_event = AuditEvent(
            event_type=AuditEventType.LOGIN,
            timestamp=datetime.now(timezone.utc),
        )
        storage.write_batch([old_event, new_event])

        deleted = storage.delete_before(datetime.now(timezone.utc) - timedelta(days=5))
        assert deleted == 1
        assert storage.count() == 1


# =============================================================================
# Test File Storage
# =============================================================================


class TestFileAuditStorage:
    """Tests for FileAuditStorage."""

    def test_write_and_query(self):
        """Test writing and querying file storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = FileStorageConfig(path=tmpdir)
            storage = FileAuditStorage(config)

            event = AuditEvent(
                event_type=AuditEventType.LOGIN,
                actor=AuditActor(id="user:123"),
            )
            storage.write(event)
            storage.flush()

            results = storage.query(limit=10)
            assert len(results) == 1
            assert results[0].actor.id == "user:123"

            storage.close()


# =============================================================================
# Test SQLite Storage
# =============================================================================


class TestSQLiteAuditStorage:
    """Tests for SQLiteAuditStorage."""

    def test_write_and_read(self):
        """Test writing and reading SQLite storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "audit.db")
            storage = SQLiteAuditStorage(db_path)

            event = AuditEvent(
                event_type=AuditEventType.UPDATE,
                action="update_config",
                actor=AuditActor(id="admin:1"),
                resource=AuditResource(id="config:main", type="config"),
            )
            storage.write(event)

            result = storage.read(event.id)
            assert result is not None
            assert result.action == "update_config"

            storage.close()

    def test_query_with_filters(self):
        """Test querying with various filters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "audit.db")
            storage = SQLiteAuditStorage(db_path)

            # Write events
            storage.write(AuditEvent(
                event_type=AuditEventType.CREATE,
                outcome=AuditOutcome.SUCCESS,
            ))
            storage.write(AuditEvent(
                event_type=AuditEventType.CREATE,
                outcome=AuditOutcome.FAILURE,
            ))

            # Query by outcome
            results = storage.query(outcome=AuditOutcome.SUCCESS)
            assert len(results) == 1

            storage.close()


# =============================================================================
# Test Formatters
# =============================================================================


class TestJSONFormatter:
    """Tests for JSONFormatter."""

    def test_format_basic(self):
        """Test basic JSON formatting."""
        formatter = JSONFormatter()
        event = AuditEvent(
            event_type=AuditEventType.LOGIN,
            action="login",
        )
        result = formatter.format(event)

        data = json.loads(result)
        assert data["event_type"] == "login"
        assert data["action"] == "login"

    def test_format_pretty(self):
        """Test pretty printing."""
        formatter = JSONFormatter(pretty=True)
        event = AuditEvent(event_type=AuditEventType.LOGIN)
        result = formatter.format(event)

        assert "\n" in result  # Pretty print has newlines

    def test_parse_json(self):
        """Test parsing JSON back to event."""
        formatter = JSONFormatter()
        event = AuditEvent(
            event_type=AuditEventType.UPDATE,
            action="update_user",
        )
        json_str = formatter.format(event)
        parsed = formatter.parse(json_str)

        assert parsed.event_type == AuditEventType.UPDATE
        assert parsed.action == "update_user"


class TestCEFFormatter:
    """Tests for CEFFormatter."""

    def test_format_basic(self):
        """Test basic CEF formatting."""
        formatter = CEFFormatter()
        event = AuditEvent(
            event_type=AuditEventType.LOGIN,
            severity=AuditSeverity.INFO,
            actor=AuditActor(id="user:123", ip_address="192.168.1.1"),
        )
        result = formatter.format(event)

        assert result.startswith("CEF:0|")
        assert "suser=user:123" in result
        assert "src=192.168.1.1" in result


class TestHumanFormatter:
    """Tests for HumanFormatter."""

    def test_format_basic(self):
        """Test basic human-readable formatting."""
        formatter = HumanFormatter()
        event = AuditEvent(
            event_type=AuditEventType.LOGIN,
            action="login",
            outcome=AuditOutcome.SUCCESS,
        )
        result = formatter.format(event)

        assert "login" in result
        assert "success" in result


# =============================================================================
# Test Filters
# =============================================================================


class TestFilters:
    """Tests for audit filters."""

    def test_severity_filter(self):
        """Test severity filter."""
        filter = SeverityFilter(min_severity=AuditSeverity.WARNING)

        info_event = AuditEvent(severity=AuditSeverity.INFO)
        warning_event = AuditEvent(severity=AuditSeverity.WARNING)
        error_event = AuditEvent(severity=AuditSeverity.ERROR)

        assert filter.should_log(info_event) is False
        assert filter.should_log(warning_event) is True
        assert filter.should_log(error_event) is True

    def test_event_type_filter(self):
        """Test event type filter."""
        filter = EventTypeFilter(
            include=[AuditEventType.LOGIN, AuditEventType.LOGOUT],
        )

        login_event = AuditEvent(event_type=AuditEventType.LOGIN)
        update_event = AuditEvent(event_type=AuditEventType.UPDATE)

        assert filter.should_log(login_event) is True
        assert filter.should_log(update_event) is False

    def test_category_filter(self):
        """Test category filter."""
        filter = CategoryFilter(
            include=[AuditCategory.SECURITY],
        )

        security_event = AuditEvent(category=AuditCategory.SECURITY)
        data_event = AuditEvent(category=AuditCategory.DATA_ACCESS)

        assert filter.should_log(security_event) is True
        assert filter.should_log(data_event) is False

    def test_actor_filter(self):
        """Test actor filter."""
        filter = ActorFilter(exclude_ids=["system", "healthcheck"])

        user_event = AuditEvent(actor=AuditActor(id="user:123"))
        system_event = AuditEvent(actor=AuditActor(id="system"))

        assert filter.should_log(user_event) is True
        assert filter.should_log(system_event) is False

    def test_action_filter(self):
        """Test action filter with patterns."""
        filter = ActionFilter(exclude_patterns=["health*", "internal_*"])

        normal_event = AuditEvent(action="create_user")
        health_event = AuditEvent(action="healthcheck")
        internal_event = AuditEvent(action="internal_sync")

        assert filter.should_log(normal_event) is True
        assert filter.should_log(health_event) is False
        assert filter.should_log(internal_event) is False

    def test_outcome_filter(self):
        """Test outcome filter."""
        filter = OutcomeFilter(include=[AuditOutcome.FAILURE])

        success_event = AuditEvent(outcome=AuditOutcome.SUCCESS)
        failure_event = AuditEvent(outcome=AuditOutcome.FAILURE)

        assert filter.should_log(success_event) is False
        assert filter.should_log(failure_event) is True

    def test_composite_filter_all(self):
        """Test composite filter with AND logic."""
        filter = CompositeFilter(
            filters=[
                SeverityFilter(min_severity=AuditSeverity.WARNING),
                EventTypeFilter(include=[AuditEventType.SECURITY_ALERT]),
            ],
            mode="all",
        )

        # Must match both
        matching_event = AuditEvent(
            event_type=AuditEventType.SECURITY_ALERT,
            severity=AuditSeverity.WARNING,
        )
        non_matching_event = AuditEvent(
            event_type=AuditEventType.SECURITY_ALERT,
            severity=AuditSeverity.INFO,
        )

        assert filter.should_log(matching_event) is True
        assert filter.should_log(non_matching_event) is False

    def test_callable_filter(self):
        """Test callable filter."""
        filter = CallableFilter(lambda e: e.actor and e.actor.type == "user")

        user_event = AuditEvent(actor=AuditActor(id="123", type="user"))
        service_event = AuditEvent(actor=AuditActor(id="svc", type="service"))

        assert filter.should_log(user_event) is True
        assert filter.should_log(service_event) is False


# =============================================================================
# Test Processors
# =============================================================================


class TestProcessors:
    """Tests for audit processors."""

    def test_privacy_processor_mask_fields(self):
        """Test masking sensitive fields."""
        processor = PrivacyProcessor(mask_fields=["password", "token"])

        event = AuditEvent(
            data={"username": "john", "password": "secret123"},
        )
        processed = processor.process(event)

        assert processed.data["username"] == "john"
        assert "***" in processed.data["password"]
        assert "secret123" not in processed.data["password"]

    def test_privacy_processor_anonymize_ip(self):
        """Test IP anonymization."""
        processor = PrivacyProcessor(anonymize_ip=True)

        event = AuditEvent(
            actor=AuditActor(id="user:123", ip_address="192.168.1.100"),
        )
        processed = processor.process(event)

        assert processed.actor.ip_address == "192.168.1.0"

    def test_enrichment_processor(self):
        """Test enrichment processor."""
        processor = EnrichmentProcessor(
            add_hostname=True,
            add_environment=True,
            environment="production",
            service_name="my-service",
            add_service_info=True,
        )

        event = AuditEvent()
        processed = processor.process(event)

        assert processed.context.environment == "production"
        assert processed.context.service_name == "my-service"
        assert processed.context.host is not None

    def test_checksum_processor(self):
        """Test checksum processor."""
        processor = ChecksumProcessor()

        event = AuditEvent(action="test")
        processed = processor.process(event)

        assert processed.checksum is not None
        assert len(processed.checksum) == 64

    def test_tagging_processor(self):
        """Test tagging processor."""
        processor = TaggingProcessor(
            rules=[
                (lambda e: e.severity == AuditSeverity.CRITICAL, "critical"),
                (lambda e: e.category == AuditCategory.SECURITY, "security"),
            ],
            static_tags=["env:prod"],
        )

        event = AuditEvent(
            severity=AuditSeverity.CRITICAL,
            category=AuditCategory.SECURITY,
        )
        processed = processor.process(event)

        assert "critical" in processed.tags
        assert "security" in processed.tags
        assert "env:prod" in processed.tags

    def test_composite_processor(self):
        """Test composite processor chain."""
        processor = CompositeProcessor([
            PrivacyProcessor(),
            EnrichmentProcessor(add_hostname=True),
            ChecksumProcessor(),
        ])

        event = AuditEvent(data={"password": "secret"})
        processed = processor.process(event)

        # "secret" (6 chars) becomes "se" + "**" + "et" = "se**et"
        assert "*" in processed.data["password"]
        assert processed.context.host is not None
        assert processed.checksum is not None


# =============================================================================
# Test Audit Logger
# =============================================================================


class TestAuditLogger:
    """Tests for AuditLogger."""

    def test_basic_logging(self):
        """Test basic event logging."""
        storage = MemoryAuditStorage()
        logger = AuditLogger(storage=storage)

        event = logger.log(
            event_type=AuditEventType.LOGIN,
            action="login",
            actor=AuditActor(id="user:123"),
            outcome=AuditOutcome.SUCCESS,
        )

        assert event is not None
        assert storage.count() == 1

    def test_log_with_resource(self):
        """Test logging with resource."""
        storage = MemoryAuditStorage()
        logger = AuditLogger(storage=storage)

        event = logger.log(
            event_type=AuditEventType.UPDATE,
            action="update_document",
            resource=AuditResource(id="doc:456", type="document"),
        )

        assert event.resource.id == "doc:456"

    def test_convenience_methods(self):
        """Test convenience logging methods."""
        storage = MemoryAuditStorage()
        logger = AuditLogger(storage=storage)

        actor = AuditActor(id="user:123")
        resource = AuditResource(id="doc:456", type="document")

        logger.log_login(actor)
        logger.log_logout(actor)
        logger.log_access(resource, actor=actor)
        logger.log_create(resource, actor=actor)
        logger.log_update(resource, actor=actor)
        logger.log_delete(resource, actor=actor)

        assert storage.count() == 6

    def test_filtering(self):
        """Test that filters are applied."""
        storage = MemoryAuditStorage()
        config = AuditConfig(min_severity=AuditSeverity.WARNING)
        logger = AuditLogger(config=config, storage=storage)

        # INFO should be filtered
        logger.log(
            event_type=AuditEventType.READ,
            severity=AuditSeverity.INFO,
        )

        # WARNING should pass
        logger.log(
            event_type=AuditEventType.SECURITY_ALERT,
            severity=AuditSeverity.WARNING,
        )

        assert storage.count() == 1

    def test_disabled_logging(self):
        """Test disabled logging."""
        storage = MemoryAuditStorage()
        config = AuditConfig(enabled=False)
        logger = AuditLogger(config=config, storage=storage)

        event = logger.log(event_type=AuditEventType.LOGIN)

        assert event is None
        assert storage.count() == 0

    def test_query(self):
        """Test querying through logger."""
        storage = MemoryAuditStorage()
        logger = AuditLogger(storage=storage)

        actor = AuditActor(id="user:123")
        logger.log(event_type=AuditEventType.LOGIN, actor=actor)
        logger.log(event_type=AuditEventType.UPDATE, actor=actor)

        results = logger.query(actor_id="user:123")
        assert len(results) == 2


# =============================================================================
# Test Logger Registry
# =============================================================================


class TestAuditLoggerRegistry:
    """Tests for AuditLoggerRegistry."""

    def setup_method(self):
        """Reset registry before each test."""
        AuditLoggerRegistry.reset_instance()

    def test_register_and_get(self):
        """Test registering and getting loggers."""
        registry = AuditLoggerRegistry()
        logger = AuditLogger(name="test")

        registry.register("test", logger)
        assert registry.get("test") is logger

    def test_default_logger(self):
        """Test default logger."""
        registry = AuditLoggerRegistry()
        logger = AuditLogger(name="default")

        registry.register("default", logger, set_default=True)
        assert registry.get() is logger

    def test_singleton(self):
        """Test singleton pattern."""
        registry1 = AuditLoggerRegistry()
        registry2 = AuditLoggerRegistry()

        logger = AuditLogger(name="test")
        registry1.register("test", logger)

        assert registry2.get("test") is logger


# =============================================================================
# Test Context Manager
# =============================================================================


class TestAuditContext:
    """Tests for audit context manager."""

    def test_audit_context(self):
        """Test audit context manager."""
        storage = MemoryAuditStorage()
        logger = AuditLogger(storage=storage)

        actor = AuditActor(id="user:123")

        with audit_context(actor=actor, request_id="req-456"):
            # Log without explicit actor
            logger.log(event_type=AuditEventType.READ)

        events = storage.query()
        assert events[0].actor.id == "user:123"


# =============================================================================
# Test Decorator
# =============================================================================


class TestAuditedDecorator:
    """Tests for audited decorator."""

    def setup_method(self):
        """Reset registry before each test."""
        AuditLoggerRegistry.reset_instance()

    def test_audited_success(self):
        """Test audited decorator on successful call."""
        storage = MemoryAuditStorage()
        logger = AuditLogger(storage=storage, name="test")
        AuditLoggerRegistry().register("test", logger, set_default=True)

        @audited(action="test_action", logger="test")
        def my_function():
            return "result"

        result = my_function()
        assert result == "result"

        events = storage.query()
        assert len(events) == 1
        assert events[0].action == "test_action"
        assert events[0].outcome == AuditOutcome.SUCCESS

    def test_audited_failure(self):
        """Test audited decorator on failed call."""
        storage = MemoryAuditStorage()
        logger = AuditLogger(storage=storage, name="test")
        AuditLoggerRegistry().register("test", logger, set_default=True)

        @audited(action="failing_action", logger="test")
        def failing_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            failing_function()

        events = storage.query()
        assert len(events) == 1
        assert events[0].outcome == AuditOutcome.FAILURE
        assert "Test error" in events[0].reason


# =============================================================================
# Test Utility Functions
# =============================================================================


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_mask_sensitive_value(self):
        """Test masking sensitive values."""
        assert mask_sensitive_value("secret123") == "se*****23"
        assert mask_sensitive_value("ab") == "**"
        assert mask_sensitive_value(None) == ""

    def test_anonymize_ip_address(self):
        """Test IP anonymization."""
        assert anonymize_ip_address("192.168.1.100") == "192.168.1.0"
        assert anonymize_ip_address("10.0.0.5") == "10.0.0.0"
        assert anonymize_ip_address("") == ""


# =============================================================================
# Test Thread Safety
# =============================================================================


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_writes(self):
        """Test concurrent event writing."""
        storage = MemoryAuditStorage()
        logger = AuditLogger(storage=storage)
        errors = []

        def worker():
            try:
                for _ in range(100):
                    logger.log(
                        event_type=AuditEventType.CUSTOM,
                        action="test",
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert storage.count() == 1000


# =============================================================================
# Test Storage Factory
# =============================================================================


class TestStorageFactory:
    """Tests for storage factory."""

    def test_create_memory_storage(self):
        """Test creating memory storage."""
        storage = create_storage("memory")
        assert isinstance(storage, MemoryAuditStorage)

    def test_create_unknown_storage(self):
        """Test creating unknown storage type."""
        with pytest.raises(ValueError):
            create_storage("unknown")


# =============================================================================
# Test Formatter Factory
# =============================================================================


class TestFormatterFactory:
    """Tests for formatter factory."""

    def test_create_json_formatter(self):
        """Test creating JSON formatter."""
        formatter = create_formatter("json")
        assert isinstance(formatter, JSONFormatter)

    def test_create_cef_formatter(self):
        """Test creating CEF formatter."""
        formatter = create_formatter("cef")
        assert isinstance(formatter, CEFFormatter)

    def test_create_human_formatter(self):
        """Test creating human formatter."""
        formatter = create_formatter("human")
        assert isinstance(formatter, HumanFormatter)

    def test_create_unknown_formatter(self):
        """Test creating unknown formatter type."""
        with pytest.raises(ValueError):
            create_formatter("unknown")
