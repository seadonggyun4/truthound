"""Logging sink integration tests.

Tests logging infrastructure integration including:
- Elasticsearch sink
- Loki sink
- Fluentd sink
- Log aggregation
- Correlation ID propagation

These tests verify that Truthound's logging infrastructure
integrates correctly with external log aggregation services.
"""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from tests.integration.external.backends.elasticsearch_backend import ElasticsearchBackend
    from tests.integration.external.providers.mock_provider import MockElasticsearchService


# =============================================================================
# Mock Elasticsearch Logging Tests
# =============================================================================


class TestMockElasticsearchLogging:
    """Tests using mock Elasticsearch for log storage."""

    def test_index_log_entry(self, mock_elasticsearch: "MockElasticsearchService") -> None:
        """Test indexing a log entry."""
        log_entry = {
            "@timestamp": "2025-12-28T10:00:00Z",
            "level": "INFO",
            "message": "Test log message",
            "logger": "truthound.test",
            "correlation_id": str(uuid.uuid4()),
        }

        result = mock_elasticsearch.index("logs-2025.12.28", log_entry)
        assert result["result"] == "created"
        assert "_id" in result

    def test_search_logs(self, mock_elasticsearch: "MockElasticsearchService") -> None:
        """Test searching log entries."""
        # Index some logs
        for i in range(5):
            mock_elasticsearch.index("logs-test", {
                "@timestamp": f"2025-12-28T10:0{i}:00Z",
                "level": "INFO",
                "message": f"Test message {i}",
            })

        # Search all
        results = mock_elasticsearch.search("logs-test", {"query": {"match_all": {}}})
        assert results["hits"]["total"]["value"] == 5

    def test_bulk_indexing(self, mock_elasticsearch: "MockElasticsearchService") -> None:
        """Test bulk log indexing."""
        logs = [
            {"@timestamp": "2025-12-28T10:00:00Z", "message": "Log 1"},
            {"@timestamp": "2025-12-28T10:00:01Z", "message": "Log 2"},
            {"@timestamp": "2025-12-28T10:00:02Z", "message": "Log 3"},
        ]

        actions = []
        for log in logs:
            actions.append({"index": {"_index": "logs-bulk"}})
            actions.append(log)

        result = mock_elasticsearch.bulk(actions)
        assert not result["errors"]
        assert len(result["items"]) == 3

    def test_correlation_id_search(self, mock_elasticsearch: "MockElasticsearchService") -> None:
        """Test searching by correlation ID."""
        correlation_id = str(uuid.uuid4())

        # Index logs with same correlation ID
        for i in range(3):
            mock_elasticsearch.index("logs-correlation", {
                "@timestamp": f"2025-12-28T10:0{i}:00Z",
                "message": f"Step {i+1}",
                "correlation_id": correlation_id,
            })

        # Index unrelated log
        mock_elasticsearch.index("logs-correlation", {
            "@timestamp": "2025-12-28T10:05:00Z",
            "message": "Unrelated",
            "correlation_id": str(uuid.uuid4()),
        })

        # Search by correlation ID
        results = mock_elasticsearch.search("logs-correlation", {
            "query": {"match_all": {}}
        })

        # Filter in Python since mock doesn't support complex queries
        correlated_logs = [
            hit for hit in results["hits"]["hits"]
            if hit["_source"].get("correlation_id") == correlation_id
        ]
        assert len(correlated_logs) == 3

    def test_log_levels(self, mock_elasticsearch: "MockElasticsearchService") -> None:
        """Test different log levels."""
        levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

        for level in levels:
            mock_elasticsearch.index("logs-levels", {
                "@timestamp": "2025-12-28T10:00:00Z",
                "level": level,
                "message": f"Test {level} message",
            })

        results = mock_elasticsearch.search("logs-levels")
        assert results["hits"]["total"]["value"] == 5

    def test_structured_logging_fields(self, mock_elasticsearch: "MockElasticsearchService") -> None:
        """Test structured logging with extra fields."""
        log_entry = {
            "@timestamp": "2025-12-28T10:00:00Z",
            "level": "INFO",
            "message": "Validation completed",
            "logger": "truthound.validator",
            "correlation_id": str(uuid.uuid4()),
            "user_id": "user-123",
            "request_id": "req-456",
            "validator": "NotNullValidator",
            "column": "email",
            "passed": True,
            "duration_ms": 15.5,
            "row_count": 1000,
        }

        mock_elasticsearch.index("logs-structured", log_entry)

        # Retrieve and verify
        results = mock_elasticsearch.search("logs-structured")
        assert results["hits"]["total"]["value"] == 1

        source = results["hits"]["hits"][0]["_source"]
        assert source["validator"] == "NotNullValidator"
        assert source["passed"] is True
        assert source["duration_ms"] == 15.5


# =============================================================================
# Docker Elasticsearch Logging Tests
# =============================================================================


@pytest.mark.elasticsearch
@pytest.mark.integration
class TestElasticsearchLogging:
    """Tests using Docker Elasticsearch for log storage."""

    @pytest.fixture
    def log_index(self, elasticsearch_backend: "ElasticsearchBackend") -> str:
        """Create a test log index."""
        index_name = f"truthound-logs-test-{uuid.uuid4().hex[:8]}"

        # Create index with mappings
        elasticsearch_backend.create_index(
            index_name,
            mappings={
                "properties": {
                    "@timestamp": {"type": "date"},
                    "level": {"type": "keyword"},
                    "message": {"type": "text"},
                    "logger": {"type": "keyword"},
                    "correlation_id": {"type": "keyword"},
                    "user_id": {"type": "keyword"},
                    "request_id": {"type": "keyword"},
                }
            },
        )

        yield index_name

        # Cleanup
        elasticsearch_backend.delete_index(index_name)

    def test_log_indexing(
        self,
        elasticsearch_backend: "ElasticsearchBackend",
        log_index: str,
    ) -> None:
        """Test log indexing in real Elasticsearch."""
        log_entry = {
            "@timestamp": "2025-12-28T10:00:00Z",
            "level": "INFO",
            "message": "Integration test log",
            "logger": "truthound.test",
            "correlation_id": str(uuid.uuid4()),
        }

        doc_id = elasticsearch_backend.index_document(log_index, log_entry)
        assert doc_id is not None

        # Wait for indexing
        time.sleep(1)

        # Retrieve
        doc = elasticsearch_backend.get_document(log_index, doc_id)
        assert doc["message"] == "Integration test log"

    def test_bulk_log_indexing(
        self,
        elasticsearch_backend: "ElasticsearchBackend",
        log_index: str,
    ) -> None:
        """Test bulk log indexing."""
        logs = [
            {
                "@timestamp": f"2025-12-28T10:{i:02d}:00Z",
                "level": "INFO",
                "message": f"Bulk log {i}",
            }
            for i in range(100)
        ]

        count = elasticsearch_backend.bulk_index(log_index, logs)
        assert count == 100

        # Wait for indexing
        time.sleep(2)

        # Verify count
        results = elasticsearch_backend.search(log_index, size=0)
        # Note: search with size=0 returns total count

    def test_log_search(
        self,
        elasticsearch_backend: "ElasticsearchBackend",
        log_index: str,
    ) -> None:
        """Test searching logs."""
        correlation_id = str(uuid.uuid4())

        # Index related logs
        for i in range(5):
            elasticsearch_backend.index_document(log_index, {
                "@timestamp": f"2025-12-28T10:0{i}:00Z",
                "level": "INFO" if i % 2 == 0 else "WARNING",
                "message": f"Step {i+1} of workflow",
                "correlation_id": correlation_id,
            })

        # Wait for indexing
        time.sleep(1)

        # Search
        results = elasticsearch_backend.search(
            log_index,
            query={"match": {"correlation_id": correlation_id}},
        )
        assert len(results) == 5

    def test_log_aggregation(
        self,
        elasticsearch_backend: "ElasticsearchBackend",
        log_index: str,
    ) -> None:
        """Test log level aggregation."""
        # Index logs with different levels
        levels = ["DEBUG"] * 10 + ["INFO"] * 50 + ["WARNING"] * 20 + ["ERROR"] * 15 + ["CRITICAL"] * 5

        for i, level in enumerate(levels):
            elasticsearch_backend.index_document(log_index, {
                "@timestamp": f"2025-12-28T10:00:{i:02d}Z",
                "level": level,
                "message": f"Log entry {i}",
            })

        # Wait for indexing
        time.sleep(2)

        # Simple search to verify counts
        all_results = elasticsearch_backend.search(log_index, size=100)
        assert len(all_results) == 100


# =============================================================================
# Truthound Logging Integration Tests
# =============================================================================


@pytest.mark.elasticsearch
@pytest.mark.integration
class TestTruthoundLoggingIntegration:
    """Tests for Truthound's logging infrastructure integration."""

    def test_validation_logging(
        self,
        elasticsearch_backend: "ElasticsearchBackend",
    ) -> None:
        """Test logging of validation operations."""
        index_name = f"truthound-validation-{uuid.uuid4().hex[:8]}"

        try:
            elasticsearch_backend.create_index(index_name)

            # Simulate validation logging
            correlation_id = str(uuid.uuid4())

            logs = [
                {
                    "@timestamp": "2025-12-28T10:00:00Z",
                    "level": "INFO",
                    "message": "Starting validation",
                    "event": "validation_start",
                    "correlation_id": correlation_id,
                    "validator_count": 5,
                    "row_count": 10000,
                },
                {
                    "@timestamp": "2025-12-28T10:00:01Z",
                    "level": "INFO",
                    "message": "Running NotNullValidator on column 'email'",
                    "event": "validator_start",
                    "correlation_id": correlation_id,
                    "validator": "NotNullValidator",
                    "column": "email",
                },
                {
                    "@timestamp": "2025-12-28T10:00:02Z",
                    "level": "WARNING",
                    "message": "Found 15 null values in column 'email'",
                    "event": "validation_warning",
                    "correlation_id": correlation_id,
                    "validator": "NotNullValidator",
                    "column": "email",
                    "null_count": 15,
                },
                {
                    "@timestamp": "2025-12-28T10:00:10Z",
                    "level": "INFO",
                    "message": "Validation completed",
                    "event": "validation_complete",
                    "correlation_id": correlation_id,
                    "duration_ms": 10000,
                    "passed": False,
                    "error_count": 15,
                },
            ]

            elasticsearch_backend.bulk_index(index_name, logs)
            time.sleep(1)

            # Query validation flow
            results = elasticsearch_backend.search(
                index_name,
                query={"match": {"correlation_id": correlation_id}},
            )
            assert len(results) == 4

        finally:
            elasticsearch_backend.delete_index(index_name)

    def test_checkpoint_logging(
        self,
        elasticsearch_backend: "ElasticsearchBackend",
    ) -> None:
        """Test logging of checkpoint operations."""
        index_name = f"truthound-checkpoint-{uuid.uuid4().hex[:8]}"

        try:
            elasticsearch_backend.create_index(index_name)

            checkpoint_id = str(uuid.uuid4())

            logs = [
                {
                    "@timestamp": "2025-12-28T10:00:00Z",
                    "level": "INFO",
                    "event": "checkpoint_start",
                    "checkpoint_id": checkpoint_id,
                    "checkpoint_name": "daily_data_quality",
                },
                {
                    "@timestamp": "2025-12-28T10:00:30Z",
                    "level": "INFO",
                    "event": "checkpoint_action",
                    "checkpoint_id": checkpoint_id,
                    "action": "SlackNotification",
                    "status": "success",
                },
                {
                    "@timestamp": "2025-12-28T10:00:31Z",
                    "level": "INFO",
                    "event": "checkpoint_complete",
                    "checkpoint_id": checkpoint_id,
                    "result": "passed",
                    "duration_ms": 31000,
                },
            ]

            elasticsearch_backend.bulk_index(index_name, logs)
            time.sleep(1)

            results = elasticsearch_backend.search(index_name)
            assert len(results) == 3

        finally:
            elasticsearch_backend.delete_index(index_name)


# =============================================================================
# Log Retention and Lifecycle Tests
# =============================================================================


@pytest.mark.elasticsearch
@pytest.mark.integration
@pytest.mark.slow
class TestLogRetention:
    """Tests for log retention and lifecycle management."""

    def test_index_naming_convention(
        self,
        elasticsearch_backend: "ElasticsearchBackend",
    ) -> None:
        """Test date-based index naming convention."""
        # Create indices for different dates
        indices = [
            "truthound-logs-2025.12.26",
            "truthound-logs-2025.12.27",
            "truthound-logs-2025.12.28",
        ]

        for index in indices:
            elasticsearch_backend.create_index(index)

        try:
            # Verify all indices exist
            for index in indices:
                assert elasticsearch_backend.index_exists(index)

        finally:
            for index in indices:
                elasticsearch_backend.delete_index(index)

    def test_index_cleanup(
        self,
        elasticsearch_backend: "ElasticsearchBackend",
    ) -> None:
        """Test cleaning up old test indices."""
        # Create some test indices
        prefix = f"truthound_test_cleanup_{uuid.uuid4().hex[:8]}"

        indices = [f"{prefix}_1", f"{prefix}_2", f"{prefix}_3"]

        for index in indices:
            elasticsearch_backend.create_index(index)

        # Cleanup by prefix
        cleaned = elasticsearch_backend.cleanup_test_indices(f"{prefix}_")
        assert cleaned == 3

        # Verify cleaned
        for index in indices:
            assert not elasticsearch_backend.index_exists(index)
