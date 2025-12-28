"""Tests for audit logging."""

from __future__ import annotations

import json
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from truthound.stores.observability.audit import (
    AuditEvent,
    AuditEventType,
    AuditLogger,
    AuditStatus,
    AsyncAuditBackend,
    CompositeAuditBackend,
    DataRedactor,
    FileAuditBackend,
    InMemoryAuditBackend,
    JsonAuditBackend,
)
from truthound.stores.observability.config import AuditConfig, AuditLogLevel
from truthound.stores.observability.protocols import ObservabilityContext


class TestAuditEvent:
    """Tests for AuditEvent dataclass."""

    def test_create_default_event(self) -> None:
        event = AuditEvent()
        assert event.event_id is not None
        assert event.event_type == AuditEventType.READ
        assert event.status == AuditStatus.SUCCESS
        assert event.timestamp is not None

    def test_create_event_with_values(self) -> None:
        event = AuditEvent(
            event_type=AuditEventType.CREATE,
            status=AuditStatus.SUCCESS,
            store_type="filesystem",
            operation="save",
            resource_id="item-123",
            user_id="user-456",
        )
        assert event.event_type == AuditEventType.CREATE
        assert event.store_type == "filesystem"
        assert event.resource_id == "item-123"
        assert event.user_id == "user-456"

    def test_to_dict(self) -> None:
        event = AuditEvent(
            event_type=AuditEventType.DELETE,
            status=AuditStatus.FAILURE,
            store_type="s3",
            operation="delete",
        )
        data = event.to_dict()
        assert data["event_type"] == "delete"
        assert data["status"] == "failure"
        assert data["store_type"] == "s3"
        assert "timestamp" in data

    def test_from_dict(self) -> None:
        data = {
            "event_id": "test-id",
            "event_type": "create",
            "status": "success",
            "timestamp": "2025-01-01T00:00:00",
            "store_type": "memory",
            "store_id": "store-1",
            "operation": "save",
            "resource_type": "result",
            "resource_id": "res-1",
            "user_id": None,
            "tenant_id": None,
            "correlation_id": None,
            "trace_id": None,
            "span_id": None,
            "source_ip": None,
            "user_agent": None,
            "duration_ms": 10.5,
            "input_summary": None,
            "output_summary": None,
            "error_message": None,
            "error_type": None,
            "metadata": {},
        }
        event = AuditEvent.from_dict(data)
        assert event.event_id == "test-id"
        assert event.event_type == AuditEventType.CREATE
        assert event.status == AuditStatus.SUCCESS
        assert event.duration_ms == 10.5

    def test_from_context(self) -> None:
        context = ObservabilityContext(
            correlation_id="corr-123",
            trace_id="trace-456",
            span_id="span-789",
            user_id="user-1",
            tenant_id="tenant-1",
        )
        event = AuditEvent.from_context(
            context,
            AuditEventType.CREATE,
            "save",
            resource_id="item-1",
        )
        assert event.correlation_id == "corr-123"
        assert event.trace_id == "trace-456"
        assert event.user_id == "user-1"
        assert event.tenant_id == "tenant-1"


class TestDataRedactor:
    """Tests for DataRedactor."""

    def test_redact_sensitive_fields(self) -> None:
        redactor = DataRedactor(["password", "secret"])
        data = {
            "username": "john",
            "password": "secret123",
            "api_secret": "key456",
        }
        result = redactor.redact(data)
        assert result["username"] == "john"
        assert result["password"] == "[REDACTED]"
        assert result["api_secret"] == "[REDACTED]"

    def test_redact_nested_data(self) -> None:
        redactor = DataRedactor(["password"])
        data = {
            "user": {
                "name": "john",
                "password": "secret",
            }
        }
        result = redactor.redact(data)
        assert result["user"]["name"] == "john"
        assert result["user"]["password"] == "[REDACTED]"

    def test_redact_email_pattern(self) -> None:
        redactor = DataRedactor([])
        text = "Contact us at admin@example.com for support"
        result = redactor._redact_string(text)
        assert "[REDACTED]" in result
        assert "admin@example.com" not in result

    def test_redact_ssn_pattern(self) -> None:
        redactor = DataRedactor([])
        text = "SSN: 123-45-6789"
        result = redactor._redact_string(text)
        assert "[REDACTED]" in result
        assert "123-45-6789" not in result

    def test_redact_list(self) -> None:
        redactor = DataRedactor(["secret"])
        data = [
            {"name": "item1", "secret": "value1"},
            {"name": "item2", "secret": "value2"},
        ]
        result = redactor.redact(data)
        assert result[0]["name"] == "item1"
        assert result[0]["secret"] == "[REDACTED]"
        assert result[1]["secret"] == "[REDACTED]"


class TestInMemoryAuditBackend:
    """Tests for InMemoryAuditBackend."""

    def test_log_event(self) -> None:
        backend = InMemoryAuditBackend()
        event = AuditEvent(
            event_type=AuditEventType.CREATE,
            operation="save",
        )
        backend.log(event)
        assert len(backend.events) == 1
        assert backend.events[0].event_type == AuditEventType.CREATE

    def test_query_by_time(self) -> None:
        backend = InMemoryAuditBackend()

        now = datetime.utcnow()
        old_event = AuditEvent(
            timestamp=now - timedelta(hours=2),
            event_type=AuditEventType.CREATE,
        )
        new_event = AuditEvent(
            timestamp=now,
            event_type=AuditEventType.READ,
        )

        backend.log(old_event)
        backend.log(new_event)

        results = backend.query(start_time=now - timedelta(hours=1))
        assert len(results) == 1
        assert results[0].event_type == AuditEventType.READ

    def test_query_by_event_type(self) -> None:
        backend = InMemoryAuditBackend()
        backend.log(AuditEvent(event_type=AuditEventType.CREATE))
        backend.log(AuditEvent(event_type=AuditEventType.READ))
        backend.log(AuditEvent(event_type=AuditEventType.CREATE))

        results = backend.query(event_type="create")
        assert len(results) == 2

    def test_query_by_user_id(self) -> None:
        backend = InMemoryAuditBackend()
        backend.log(AuditEvent(user_id="user-1"))
        backend.log(AuditEvent(user_id="user-2"))
        backend.log(AuditEvent(user_id="user-1"))

        results = backend.query(user_id="user-1")
        assert len(results) == 2

    def test_query_with_limit(self) -> None:
        backend = InMemoryAuditBackend()
        for i in range(10):
            backend.log(AuditEvent(resource_id=f"item-{i}"))

        results = backend.query(limit=5)
        assert len(results) == 5

    def test_clear(self) -> None:
        backend = InMemoryAuditBackend()
        backend.log(AuditEvent())
        backend.log(AuditEvent())
        assert len(backend.events) == 2

        backend.clear()
        assert len(backend.events) == 0

    def test_log_level_filtering(self) -> None:
        config = AuditConfig(level=AuditLogLevel.MINIMAL)
        backend = InMemoryAuditBackend(config)

        # Minimal level should only log writes
        backend.log(AuditEvent(event_type=AuditEventType.READ))
        backend.log(AuditEvent(event_type=AuditEventType.CREATE))
        backend.log(AuditEvent(event_type=AuditEventType.QUERY))

        assert len(backend.events) == 1
        assert backend.events[0].event_type == AuditEventType.CREATE


class TestJsonAuditBackend:
    """Tests for JsonAuditBackend."""

    def test_log_and_query(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AuditConfig(file_path=Path(tmpdir) / "audit.jsonl")
            backend = JsonAuditBackend(config)

            event = AuditEvent(
                event_type=AuditEventType.CREATE,
                resource_id="item-1",
                user_id="user-1",
            )
            backend.log(event)
            backend.flush()

            results = backend.query(user_id="user-1")
            assert len(results) == 1
            assert results[0].resource_id == "item-1"

            backend.close()

    def test_batch_flush(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AuditConfig(
                file_path=Path(tmpdir) / "audit.jsonl",
                batch_size=5,
            )
            backend = JsonAuditBackend(config)

            # Log fewer than batch size
            for i in range(3):
                backend.log(AuditEvent(resource_id=f"item-{i}"))

            # Should still be in buffer
            assert not (Path(tmpdir) / "audit.jsonl").exists() or (Path(tmpdir) / "audit.jsonl").stat().st_size == 0

            # Force flush
            backend.flush()

            # Now file should have content
            with open(Path(tmpdir) / "audit.jsonl") as f:
                lines = f.readlines()
            assert len(lines) == 3

            backend.close()


class TestFileAuditBackend:
    """Tests for FileAuditBackend."""

    def test_log_to_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AuditConfig(file_path=Path(tmpdir) / "audit.log")
            backend = FileAuditBackend(config)

            backend.log(AuditEvent(
                event_type=AuditEventType.CREATE,
                operation="save",
                resource_id="item-1",
            ))
            backend.flush()
            backend.close()

            with open(Path(tmpdir) / "audit.log") as f:
                content = f.read()

            assert "CREATE" in content
            assert "save" in content
            assert "item-1" in content


class TestCompositeAuditBackend:
    """Tests for CompositeAuditBackend."""

    def test_log_to_multiple_backends(self) -> None:
        backend1 = InMemoryAuditBackend()
        backend2 = InMemoryAuditBackend()
        composite = CompositeAuditBackend([backend1, backend2])

        composite.log(AuditEvent(event_type=AuditEventType.CREATE))

        assert len(backend1.events) == 1
        assert len(backend2.events) == 1

    def test_query_from_first_backend(self) -> None:
        backend1 = InMemoryAuditBackend()
        backend2 = InMemoryAuditBackend()
        composite = CompositeAuditBackend([backend1, backend2])

        # Log to composite
        composite.log(AuditEvent(user_id="user-1"))

        # Query should return from first backend
        results = composite.query(user_id="user-1")
        assert len(results) == 1


class TestAsyncAuditBackend:
    """Tests for AsyncAuditBackend."""

    def test_async_logging(self) -> None:
        inner_backend = InMemoryAuditBackend()
        async_backend = AsyncAuditBackend(inner_backend)

        # Log events
        for i in range(10):
            async_backend.log(AuditEvent(resource_id=f"item-{i}"))

        # Wait for processing
        async_backend.flush()
        time.sleep(0.1)

        assert len(inner_backend.events) == 10

        async_backend.close()


class TestAuditLogger:
    """Tests for AuditLogger."""

    def test_operation_context_manager(self) -> None:
        backend = InMemoryAuditBackend()
        logger = AuditLogger(backend, store_type="filesystem", store_id="test-store")

        with logger.operation(
            AuditEventType.CREATE,
            "save",
            resource_id="item-1",
        ) as event:
            event.input_summary = "test data"

        assert len(backend.events) == 1
        assert backend.events[0].status == AuditStatus.SUCCESS
        assert backend.events[0].duration_ms is not None
        assert backend.events[0].duration_ms > 0

    def test_operation_with_error(self) -> None:
        backend = InMemoryAuditBackend()
        logger = AuditLogger(backend, store_type="filesystem")

        with pytest.raises(ValueError):
            with logger.operation(
                AuditEventType.CREATE,
                "save",
                resource_id="item-1",
            ):
                raise ValueError("Test error")

        assert len(backend.events) == 1
        assert backend.events[0].status == AuditStatus.FAILURE
        assert backend.events[0].error_type == "ValueError"
        assert "Test error" in backend.events[0].error_message

    def test_log_event_direct(self) -> None:
        backend = InMemoryAuditBackend()
        logger = AuditLogger(backend, store_type="s3")

        logger.log_event(
            AuditEventType.READ,
            "get",
            resource_id="item-1",
        )

        assert len(backend.events) == 1
        assert backend.events[0].store_type == "s3"
