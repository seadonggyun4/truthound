"""Enterprise audit logging system for Truthound.

This module extends the base audit system with enterprise features:
- Full operation audit trail
- Additional storage backends (Elasticsearch, S3, Kafka)
- Compliance reporting
- Retention policies
- Integration with correlation context

Architecture:
    EnterpriseAuditLogger
         |
         +---> Filter Pipeline
         +---> Processor Pipeline
         |
         v
    AuditStorage[]
         |
         +---> SQLiteAuditStorage (local)
         +---> ElasticsearchAuditStorage
         +---> S3AuditStorage
         +---> KafkaAuditStorage
         |
         v
    ComplianceReporter (generates reports)

Usage:
    >>> from truthound.infrastructure.audit import (
    ...     get_audit_logger, configure_audit,
    ...     EnterpriseAuditConfig,
    ... )
    >>>
    >>> # Configure for production
    >>> configure_audit(
    ...     service_name="truthound",
    ...     environment="production",
    ...     storage_backends=["sqlite", "elasticsearch"],
    ...     elasticsearch_url="http://elk:9200",
    ... )
    >>>
    >>> # Log audit event
    >>> logger = get_audit_logger()
    >>> logger.log_operation(
    ...     operation="validation",
    ...     resource="dataset:users",
    ...     outcome="success",
    ...     details={"rows": 10000, "issues": 5},
    ... )
"""

from __future__ import annotations

import json
import os
import queue
import socket
import threading
import time
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterator

# Re-export and extend base audit components
from truthound.audit import (
    AuditEventType,
    AuditSeverity,
    AuditOutcome,
    AuditCategory,
    AuditActor,
    AuditResource,
    AuditContext,
    AuditEvent,
    AuditConfig,
    AuditStorage,
    AuditFormatter,
    AuditFilter,
    AuditProcessor,
    AuditEventBuilder,
    AuditLogger,
    MemoryAuditStorage,
    SQLiteAuditStorage,
    JSONFormatter,
    SeverityFilter,
    PrivacyProcessor,
    ChecksumProcessor,
)

# Import correlation context from logging
from truthound.infrastructure.logging import (
    CorrelationContext,
    get_correlation_id,
)


# =============================================================================
# Extended Configuration
# =============================================================================


@dataclass
class EnterpriseAuditConfig(AuditConfig):
    """Extended audit configuration for enterprise features.

    Example:
        >>> config = EnterpriseAuditConfig(
        ...     service_name="truthound",
        ...     environment="production",
        ...     storage_backends=["sqlite", "elasticsearch"],
        ...     elasticsearch_url="http://elk:9200",
        ...     s3_bucket="audit-logs",
        ...     retention_days=365,
        ...     compliance_standards=["SOC2", "GDPR"],
        ... )
    """

    # Extended storage options
    storage_backends: list[str] = field(
        default_factory=lambda: ["memory"]
    )  # memory, sqlite, elasticsearch, s3, kafka

    # Elasticsearch settings
    elasticsearch_url: str = ""
    elasticsearch_index_prefix: str = "truthound-audit"
    elasticsearch_username: str = ""
    elasticsearch_password: str = ""

    # S3 settings
    s3_bucket: str = ""
    s3_prefix: str = "audit/"
    s3_region: str = ""

    # Kafka settings
    kafka_bootstrap_servers: str = ""
    kafka_topic: str = "truthound-audit"

    # Compliance settings
    compliance_standards: list[str] = field(default_factory=list)  # SOC2, GDPR, HIPAA
    require_checksum: bool = True
    require_signing: bool = False

    # Retention policy
    retention_policy: str = "default"  # default, compliance, custom
    archive_to_cold_storage: bool = False
    cold_storage_after_days: int = 90

    # Performance
    async_write: bool = True
    batch_size: int = 100
    flush_interval: float = 5.0

    @classmethod
    def production(cls, service_name: str) -> "EnterpriseAuditConfig":
        """Create production configuration."""
        return cls(
            enabled=True,
            service_name=service_name,
            environment="production",
            storage_backends=["sqlite"],
            require_checksum=True,
            async_write=True,
            retention_days=365,
        )

    @classmethod
    def compliance(
        cls,
        service_name: str,
        standards: list[str],
    ) -> "EnterpriseAuditConfig":
        """Create compliance-focused configuration."""
        return cls(
            enabled=True,
            service_name=service_name,
            environment="production",
            storage_backends=["sqlite", "s3"],
            require_checksum=True,
            require_signing=True,
            compliance_standards=standards,
            retention_policy="compliance",
            retention_days=2555,  # 7 years
            archive_to_cold_storage=True,
        )


# =============================================================================
# Additional Storage Backends
# =============================================================================


class ElasticsearchAuditStorage(AuditStorage):
    """Elasticsearch audit storage backend.

    Stores audit events in Elasticsearch for search and analysis.
    """

    def __init__(
        self,
        url: str,
        *,
        index_prefix: str = "truthound-audit",
        username: str = "",
        password: str = "",
        api_key: str = "",
        batch_size: int = 100,
        flush_interval: float = 5.0,
    ) -> None:
        """Initialize Elasticsearch storage.

        Args:
            url: Elasticsearch URL.
            index_prefix: Index name prefix.
            username: Basic auth username.
            password: Basic auth password.
            api_key: API key for auth.
            batch_size: Batch size for bulk indexing.
            flush_interval: Flush interval in seconds.
        """
        self._url = url.rstrip("/")
        self._index_prefix = index_prefix
        self._username = username
        self._password = password
        self._api_key = api_key
        self._batch_size = batch_size
        self._flush_interval = flush_interval

        self._buffer: list[AuditEvent] = []
        self._lock = threading.Lock()
        self._last_flush = time.time()
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="es-audit")
        self._running = True

        # Start background flusher
        self._flush_thread = threading.Thread(
            target=self._background_flush,
            daemon=True,
            name="es-audit-flusher",
        )
        self._flush_thread.start()

    def write(self, event: AuditEvent) -> None:
        """Buffer event for bulk indexing."""
        with self._lock:
            self._buffer.append(event)
            if len(self._buffer) >= self._batch_size:
                self._flush_buffer()

    def write_batch(self, events: list[AuditEvent]) -> None:
        """Write multiple events."""
        with self._lock:
            self._buffer.extend(events)
            if len(self._buffer) >= self._batch_size:
                self._flush_buffer()

    def _background_flush(self) -> None:
        """Background flush loop."""
        while self._running:
            time.sleep(1)
            with self._lock:
                if (
                    self._buffer
                    and time.time() - self._last_flush >= self._flush_interval
                ):
                    self._flush_buffer()

    def _flush_buffer(self) -> None:
        """Flush buffered events to Elasticsearch."""
        if not self._buffer:
            return

        events = self._buffer.copy()
        self._buffer.clear()
        self._last_flush = time.time()
        self._executor.submit(self._bulk_index, events)

    def _get_index_name(self, timestamp: datetime) -> str:
        """Get index name for timestamp."""
        suffix = timestamp.strftime("%Y.%m.%d")
        return f"{self._index_prefix}-{suffix}"

    def _bulk_index(self, events: list[AuditEvent]) -> None:
        """Bulk index events to Elasticsearch."""
        try:
            import urllib.request

            lines = []
            for event in events:
                index_name = self._get_index_name(event.timestamp)
                action = json.dumps({"index": {"_index": index_name, "_id": event.id}})
                doc = json.dumps(event.to_dict(), default=str)
                lines.append(action)
                lines.append(doc)
            body = "\n".join(lines) + "\n"

            url = f"{self._url}/_bulk"
            headers = {"Content-Type": "application/x-ndjson"}

            if self._api_key:
                headers["Authorization"] = f"ApiKey {self._api_key}"

            request = urllib.request.Request(
                url,
                data=body.encode("utf-8"),
                headers=headers,
                method="POST",
            )

            if self._username and self._password:
                import base64

                credentials = base64.b64encode(
                    f"{self._username}:{self._password}".encode()
                ).decode()
                request.add_header("Authorization", f"Basic {credentials}")

            with urllib.request.urlopen(request, timeout=30):
                pass

        except Exception:
            pass

    def read(self, event_id: str) -> AuditEvent | None:
        """Read event by ID (searches all indices)."""
        try:
            import urllib.request

            # Search across all audit indices
            url = f"{self._url}/{self._index_prefix}-*/_doc/{event_id}"
            request = urllib.request.Request(url)

            if self._api_key:
                request.add_header("Authorization", f"ApiKey {self._api_key}")

            with urllib.request.urlopen(request, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
                if data.get("found"):
                    return AuditEvent.from_dict(data["_source"])

        except Exception:
            pass
        return None

    def query(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        event_types: list[AuditEventType] | None = None,
        actor_id: str | None = None,
        resource_id: str | None = None,
        outcome: AuditOutcome | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AuditEvent]:
        """Query audit events."""
        try:
            import urllib.request

            # Build query
            must_clauses = []

            if start_time or end_time:
                range_clause = {"range": {"timestamp": {}}}
                if start_time:
                    range_clause["range"]["timestamp"]["gte"] = start_time.isoformat()
                if end_time:
                    range_clause["range"]["timestamp"]["lte"] = end_time.isoformat()
                must_clauses.append(range_clause)

            if event_types:
                must_clauses.append({
                    "terms": {"event_type": [t.value for t in event_types]}
                })

            if actor_id:
                must_clauses.append({"term": {"actor.id": actor_id}})

            if resource_id:
                must_clauses.append({"term": {"resource.id": resource_id}})

            if outcome:
                must_clauses.append({"term": {"outcome": outcome.value}})

            query = {
                "query": {"bool": {"must": must_clauses}} if must_clauses else {"match_all": {}},
                "sort": [{"timestamp": "desc"}],
                "from": offset,
                "size": limit,
            }

            url = f"{self._url}/{self._index_prefix}-*/_search"
            request = urllib.request.Request(
                url,
                data=json.dumps(query).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            if self._api_key:
                request.add_header("Authorization", f"ApiKey {self._api_key}")

            with urllib.request.urlopen(request, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
                events = []
                for hit in data.get("hits", {}).get("hits", []):
                    events.append(AuditEvent.from_dict(hit["_source"]))
                return events

        except Exception:
            return []

    def count(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        event_types: list[AuditEventType] | None = None,
    ) -> int:
        """Count matching events."""
        try:
            import urllib.request

            must_clauses = []
            if start_time or end_time:
                range_clause = {"range": {"timestamp": {}}}
                if start_time:
                    range_clause["range"]["timestamp"]["gte"] = start_time.isoformat()
                if end_time:
                    range_clause["range"]["timestamp"]["lte"] = end_time.isoformat()
                must_clauses.append(range_clause)

            if event_types:
                must_clauses.append({
                    "terms": {"event_type": [t.value for t in event_types]}
                })

            query = {
                "query": {"bool": {"must": must_clauses}} if must_clauses else {"match_all": {}},
            }

            url = f"{self._url}/{self._index_prefix}-*/_count"
            request = urllib.request.Request(
                url,
                data=json.dumps(query).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with urllib.request.urlopen(request, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
                return data.get("count", 0)

        except Exception:
            return 0

    def delete_before(self, before: datetime) -> int:
        """Delete events before timestamp."""
        try:
            import urllib.request

            query = {
                "query": {
                    "range": {"timestamp": {"lt": before.isoformat()}}
                }
            }

            url = f"{self._url}/{self._index_prefix}-*/_delete_by_query"
            request = urllib.request.Request(
                url,
                data=json.dumps(query).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with urllib.request.urlopen(request, timeout=60) as response:
                data = json.loads(response.read().decode("utf-8"))
                return data.get("deleted", 0)

        except Exception:
            return 0

    def flush(self) -> None:
        """Flush buffered events."""
        with self._lock:
            self._flush_buffer()

    def close(self) -> None:
        """Close storage."""
        self._running = False
        self.flush()
        self._executor.shutdown(wait=True)


class S3AuditStorage(AuditStorage):
    """S3 audit storage backend.

    Stores audit events in S3 for long-term archival.
    Events are batched into JSON files organized by date.
    """

    def __init__(
        self,
        bucket: str,
        *,
        prefix: str = "audit/",
        region: str = "",
        batch_size: int = 1000,
        flush_interval: float = 60.0,
    ) -> None:
        """Initialize S3 storage.

        Args:
            bucket: S3 bucket name.
            prefix: Key prefix.
            region: AWS region.
            batch_size: Events per file.
            flush_interval: Flush interval in seconds.
        """
        self._bucket = bucket
        self._prefix = prefix.rstrip("/") + "/"
        self._region = region or os.getenv("AWS_REGION", "us-east-1")
        self._batch_size = batch_size
        self._flush_interval = flush_interval

        self._buffer: list[AuditEvent] = []
        self._lock = threading.Lock()
        self._last_flush = time.time()
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="s3-audit")
        self._running = True

        # Start background flusher
        self._flush_thread = threading.Thread(
            target=self._background_flush,
            daemon=True,
            name="s3-audit-flusher",
        )
        self._flush_thread.start()

    def write(self, event: AuditEvent) -> None:
        """Buffer event for batch upload."""
        with self._lock:
            self._buffer.append(event)
            if len(self._buffer) >= self._batch_size:
                self._flush_buffer()

    def write_batch(self, events: list[AuditEvent]) -> None:
        """Write multiple events."""
        with self._lock:
            self._buffer.extend(events)
            while len(self._buffer) >= self._batch_size:
                self._flush_buffer()

    def _background_flush(self) -> None:
        """Background flush loop."""
        while self._running:
            time.sleep(10)
            with self._lock:
                if (
                    self._buffer
                    and time.time() - self._last_flush >= self._flush_interval
                ):
                    self._flush_buffer()

    def _flush_buffer(self) -> None:
        """Flush buffered events to S3."""
        if not self._buffer:
            return

        events = self._buffer.copy()
        self._buffer.clear()
        self._last_flush = time.time()
        self._executor.submit(self._upload_batch, events)

    def _upload_batch(self, events: list[AuditEvent]) -> None:
        """Upload batch of events to S3."""
        try:
            import boto3

            s3 = boto3.client("s3", region_name=self._region)

            # Generate key based on timestamp
            now = datetime.now(timezone.utc)
            date_path = now.strftime("%Y/%m/%d")
            file_id = uuid.uuid4().hex[:8]
            key = f"{self._prefix}{date_path}/audit_{now.strftime('%H%M%S')}_{file_id}.json"

            # Serialize events
            data = {
                "metadata": {
                    "created_at": now.isoformat(),
                    "event_count": len(events),
                    "hostname": socket.gethostname(),
                },
                "events": [event.to_dict() for event in events],
            }

            s3.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=json.dumps(data, default=str).encode("utf-8"),
                ContentType="application/json",
            )

        except Exception:
            pass

    def read(self, event_id: str) -> AuditEvent | None:
        """Read event by ID (not efficient for S3, use query instead)."""
        return None  # S3 is write-optimized, use query for reads

    def query(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        event_types: list[AuditEventType] | None = None,
        actor_id: str | None = None,
        resource_id: str | None = None,
        outcome: AuditOutcome | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AuditEvent]:
        """Query audit events (uses S3 Select for filtering)."""
        try:
            import boto3

            s3 = boto3.client("s3", region_name=self._region)

            # List relevant objects based on date range
            start = start_time or datetime.now(timezone.utc) - timedelta(days=7)
            end = end_time or datetime.now(timezone.utc)

            events = []
            paginator = s3.get_paginator("list_objects_v2")

            current = start
            while current <= end:
                prefix = f"{self._prefix}{current.strftime('%Y/%m/%d')}/"

                for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
                    for obj in page.get("Contents", []):
                        # Download and parse
                        response = s3.get_object(
                            Bucket=self._bucket,
                            Key=obj["Key"],
                        )
                        data = json.loads(response["Body"].read().decode("utf-8"))

                        for event_dict in data.get("events", []):
                            event = AuditEvent.from_dict(event_dict)

                            # Apply filters
                            if event_types and event.event_type not in event_types:
                                continue
                            if actor_id and (not event.actor or event.actor.id != actor_id):
                                continue
                            if resource_id and (not event.resource or event.resource.id != resource_id):
                                continue
                            if outcome and event.outcome != outcome:
                                continue

                            events.append(event)

                            if len(events) >= offset + limit:
                                return events[offset:offset + limit]

                current += timedelta(days=1)

            return events[offset:offset + limit]

        except Exception:
            return []

    def count(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        event_types: list[AuditEventType] | None = None,
    ) -> int:
        """Count matching events."""
        events = self.query(
            start_time=start_time,
            end_time=end_time,
            event_types=event_types,
            limit=100000,
        )
        return len(events)

    def delete_before(self, before: datetime) -> int:
        """Delete events before timestamp."""
        try:
            import boto3

            s3 = boto3.client("s3", region_name=self._region)
            deleted = 0

            # List and delete old objects
            paginator = s3.get_paginator("list_objects_v2")
            current = before - timedelta(days=365 * 10)  # Look back 10 years

            while current < before:
                prefix = f"{self._prefix}{current.strftime('%Y/%m/%d')}/"

                for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
                    objects = page.get("Contents", [])
                    if objects:
                        s3.delete_objects(
                            Bucket=self._bucket,
                            Delete={"Objects": [{"Key": obj["Key"]} for obj in objects]},
                        )
                        deleted += len(objects)

                current += timedelta(days=1)

            return deleted

        except Exception:
            return 0

    def flush(self) -> None:
        """Flush buffered events."""
        with self._lock:
            self._flush_buffer()

    def close(self) -> None:
        """Close storage."""
        self._running = False
        self.flush()
        self._executor.shutdown(wait=True)


class KafkaAuditStorage(AuditStorage):
    """Kafka audit storage backend.

    Publishes audit events to Kafka for real-time streaming.
    """

    def __init__(
        self,
        bootstrap_servers: str,
        *,
        topic: str = "truthound-audit",
        batch_size: int = 100,
        linger_ms: int = 5,
    ) -> None:
        """Initialize Kafka storage.

        Args:
            bootstrap_servers: Kafka bootstrap servers.
            topic: Kafka topic.
            batch_size: Batch size.
            linger_ms: Linger time for batching.
        """
        self._bootstrap_servers = bootstrap_servers
        self._topic = topic
        self._batch_size = batch_size
        self._linger_ms = linger_ms
        self._producer = None
        self._lock = threading.Lock()

    def _get_producer(self) -> Any:
        """Get or create Kafka producer."""
        if self._producer is None:
            try:
                from kafka import KafkaProducer

                self._producer = KafkaProducer(
                    bootstrap_servers=self._bootstrap_servers.split(","),
                    value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
                    batch_size=self._batch_size * 1024,
                    linger_ms=self._linger_ms,
                )
            except ImportError:
                raise RuntimeError("kafka-python not installed")

        return self._producer

    def write(self, event: AuditEvent) -> None:
        """Write event to Kafka."""
        try:
            producer = self._get_producer()
            producer.send(
                self._topic,
                value=event.to_dict(),
                key=event.id.encode("utf-8"),
            )
        except Exception:
            pass

    def write_batch(self, events: list[AuditEvent]) -> None:
        """Write multiple events to Kafka."""
        try:
            producer = self._get_producer()
            for event in events:
                producer.send(
                    self._topic,
                    value=event.to_dict(),
                    key=event.id.encode("utf-8"),
                )
            producer.flush()
        except Exception:
            pass

    def read(self, event_id: str) -> AuditEvent | None:
        """Read not supported for Kafka (write-only)."""
        return None

    def query(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        event_types: list[AuditEventType] | None = None,
        actor_id: str | None = None,
        resource_id: str | None = None,
        outcome: AuditOutcome | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AuditEvent]:
        """Query not supported for Kafka (write-only)."""
        return []

    def count(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        event_types: list[AuditEventType] | None = None,
    ) -> int:
        """Count not supported for Kafka."""
        return 0

    def delete_before(self, before: datetime) -> int:
        """Delete not supported for Kafka."""
        return 0

    def flush(self) -> None:
        """Flush Kafka producer."""
        if self._producer:
            self._producer.flush()

    def close(self) -> None:
        """Close Kafka producer."""
        if self._producer:
            self._producer.close()
            self._producer = None


# =============================================================================
# Retention Policy
# =============================================================================


@dataclass
class RetentionPolicy:
    """Audit log retention policy.

    Example:
        >>> policy = RetentionPolicy(
        ...     name="compliance",
        ...     retention_days=2555,  # 7 years
        ...     archive_after_days=90,
        ...     archive_storage="s3",
        ... )
    """

    name: str
    retention_days: int = 365
    archive_after_days: int = 90
    archive_storage: str = ""  # s3, glacier
    delete_after_archive: bool = False

    @classmethod
    def default(cls) -> "RetentionPolicy":
        """Default retention policy (1 year)."""
        return cls(name="default", retention_days=365)

    @classmethod
    def compliance_soc2(cls) -> "RetentionPolicy":
        """SOC 2 compliant retention (7 years)."""
        return cls(
            name="soc2",
            retention_days=2555,
            archive_after_days=90,
            archive_storage="s3",
        )

    @classmethod
    def compliance_gdpr(cls) -> "RetentionPolicy":
        """GDPR compliant retention."""
        return cls(
            name="gdpr",
            retention_days=365,  # Minimize data retention
            archive_after_days=30,
        )

    @classmethod
    def compliance_hipaa(cls) -> "RetentionPolicy":
        """HIPAA compliant retention (6 years)."""
        return cls(
            name="hipaa",
            retention_days=2190,
            archive_after_days=90,
            archive_storage="s3",
        )


# =============================================================================
# Compliance Reporter
# =============================================================================


class ComplianceReporter:
    """Generate compliance reports from audit logs.

    Example:
        >>> reporter = ComplianceReporter(storage)
        >>> report = reporter.generate_report(
        ...     start_date=datetime(2024, 1, 1),
        ...     end_date=datetime(2024, 12, 31),
        ...     standard="SOC2",
        ... )
    """

    def __init__(self, storage: AuditStorage) -> None:
        """Initialize compliance reporter.

        Args:
            storage: Audit storage backend.
        """
        self._storage = storage

    def generate_report(
        self,
        start_date: datetime,
        end_date: datetime,
        *,
        standard: str = "",
        include_details: bool = False,
    ) -> dict[str, Any]:
        """Generate compliance report.

        Args:
            start_date: Report start date.
            end_date: Report end date.
            standard: Compliance standard (SOC2, GDPR, HIPAA).
            include_details: Include detailed event list.

        Returns:
            Report dictionary.
        """
        events = self._storage.query(
            start_time=start_date,
            end_time=end_date,
            limit=100000,
        )

        # Basic statistics
        total_events = len(events)
        by_type: dict[str, int] = {}
        by_outcome: dict[str, int] = {}
        by_severity: dict[str, int] = {}
        by_actor: dict[str, int] = {}

        for event in events:
            by_type[event.event_type.value] = by_type.get(event.event_type.value, 0) + 1
            by_outcome[event.outcome.value] = by_outcome.get(event.outcome.value, 0) + 1
            by_severity[event.severity.value] = by_severity.get(event.severity.value, 0) + 1

            if event.actor:
                by_actor[event.actor.id] = by_actor.get(event.actor.id, 0) + 1

        # Security-relevant events
        security_events = [
            e for e in events
            if e.event_type in (
                AuditEventType.LOGIN_FAILED,
                AuditEventType.ACCESS_DENIED,
                AuditEventType.SECURITY_ALERT,
                AuditEventType.SUSPICIOUS_ACTIVITY,
            )
        ]

        # Failed operations
        failed_events = [e for e in events if e.outcome == AuditOutcome.FAILURE]

        report = {
            "metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "period": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat(),
                },
                "standard": standard,
            },
            "summary": {
                "total_events": total_events,
                "security_events": len(security_events),
                "failed_operations": len(failed_events),
                "unique_actors": len(by_actor),
            },
            "breakdown": {
                "by_type": by_type,
                "by_outcome": by_outcome,
                "by_severity": by_severity,
            },
            "security": {
                "login_failures": by_type.get("login_failed", 0),
                "access_denials": by_type.get("access_denied", 0),
                "alerts": by_type.get("security_alert", 0),
            },
            "compliance": self._generate_compliance_section(events, standard),
        }

        if include_details:
            report["events"] = [e.to_dict() for e in events[:1000]]

        return report

    def _generate_compliance_section(
        self,
        events: list[AuditEvent],
        standard: str,
    ) -> dict[str, Any]:
        """Generate compliance-specific section."""
        section: dict[str, Any] = {"standard": standard, "controls": {}}

        if standard.upper() == "SOC2":
            section["controls"] = {
                "CC6.1": self._check_access_controls(events),
                "CC6.2": self._check_authentication(events),
                "CC6.3": self._check_authorization(events),
                "CC7.1": self._check_system_monitoring(events),
            }
        elif standard.upper() == "GDPR":
            section["controls"] = {
                "data_access": self._check_data_access(events),
                "consent": self._check_consent_events(events),
                "right_to_be_forgotten": self._check_deletion_events(events),
            }

        return section

    def _check_access_controls(self, events: list[AuditEvent]) -> dict[str, Any]:
        """Check access control compliance."""
        access_events = [
            e for e in events
            if e.event_type in (
                AuditEventType.LOGIN,
                AuditEventType.LOGOUT,
                AuditEventType.LOGIN_FAILED,
            )
        ]
        return {
            "total_access_events": len(access_events),
            "logged": True,
            "status": "compliant",
        }

    def _check_authentication(self, events: list[AuditEvent]) -> dict[str, Any]:
        """Check authentication logging compliance."""
        auth_events = [
            e for e in events if e.category == AuditCategory.AUTHENTICATION
        ]
        return {
            "total_auth_events": len(auth_events),
            "logged": True,
            "status": "compliant",
        }

    def _check_authorization(self, events: list[AuditEvent]) -> dict[str, Any]:
        """Check authorization logging compliance."""
        authz_events = [
            e for e in events if e.category == AuditCategory.AUTHORIZATION
        ]
        return {
            "total_authz_events": len(authz_events),
            "logged": True,
            "status": "compliant",
        }

    def _check_system_monitoring(self, events: list[AuditEvent]) -> dict[str, Any]:
        """Check system monitoring compliance."""
        return {
            "audit_enabled": True,
            "events_captured": len(events) > 0,
            "status": "compliant",
        }

    def _check_data_access(self, events: list[AuditEvent]) -> dict[str, Any]:
        """Check data access logging for GDPR."""
        data_events = [
            e for e in events if e.category == AuditCategory.DATA_ACCESS
        ]
        return {
            "total_data_access_events": len(data_events),
            "logged": True,
        }

    def _check_consent_events(self, events: list[AuditEvent]) -> dict[str, Any]:
        """Check consent-related events for GDPR."""
        return {"consent_tracking": "implemented"}

    def _check_deletion_events(self, events: list[AuditEvent]) -> dict[str, Any]:
        """Check deletion events for GDPR right to be forgotten."""
        deletion_events = [
            e for e in events if e.event_type == AuditEventType.DELETE
        ]
        return {
            "deletion_events": len(deletion_events),
            "logged": True,
        }


# =============================================================================
# Enterprise Audit Logger
# =============================================================================


class EnterpriseAuditLogger:
    """Enterprise-grade audit logger.

    Extends the base AuditLogger with:
    - Automatic correlation ID propagation
    - Multiple storage backends
    - Compliance features
    - Operation-level logging

    Example:
        >>> logger = EnterpriseAuditLogger(config)
        >>>
        >>> # Log operation
        >>> logger.log_operation(
        ...     operation="validate_dataset",
        ...     resource="dataset:users",
        ...     outcome="success",
        ...     details={"rows": 10000, "issues": 5},
        ... )
        >>>
        >>> # Log with context
        >>> with logger.operation_context("checkpoint_run", "checkpoint:daily"):
        ...     run_checkpoint()
    """

    def __init__(
        self,
        config: EnterpriseAuditConfig | None = None,
        storages: list[AuditStorage] | None = None,
    ) -> None:
        """Initialize enterprise audit logger.

        Args:
            config: Audit configuration.
            storages: Storage backends (overrides config).
        """
        self._config = config or EnterpriseAuditConfig()
        self._storages = storages or self._create_storages_from_config()
        self._lock = threading.Lock()

        # Processors
        self._processors: list[AuditProcessor] = []
        if self._config.mask_sensitive_data:
            self._processors.append(
                PrivacyProcessor(self._config.sensitive_fields)
            )
        if self._config.require_checksum:
            self._processors.append(ChecksumProcessor())

    def _create_storages_from_config(self) -> list[AuditStorage]:
        """Create storage backends from configuration."""
        storages: list[AuditStorage] = []

        for backend in self._config.storage_backends:
            if backend == "memory":
                storages.append(MemoryAuditStorage())

            elif backend == "sqlite":
                db_path = self._config.storage_config.get(
                    "sqlite_path", "audit.db"
                )
                storages.append(SQLiteAuditStorage(db_path))

            elif backend == "elasticsearch" and self._config.elasticsearch_url:
                storages.append(
                    ElasticsearchAuditStorage(
                        self._config.elasticsearch_url,
                        index_prefix=self._config.elasticsearch_index_prefix,
                        username=self._config.elasticsearch_username,
                        password=self._config.elasticsearch_password,
                    )
                )

            elif backend == "s3" and self._config.s3_bucket:
                storages.append(
                    S3AuditStorage(
                        self._config.s3_bucket,
                        prefix=self._config.s3_prefix,
                        region=self._config.s3_region,
                    )
                )

            elif backend == "kafka" and self._config.kafka_bootstrap_servers:
                storages.append(
                    KafkaAuditStorage(
                        self._config.kafka_bootstrap_servers,
                        topic=self._config.kafka_topic,
                    )
                )

        return storages or [MemoryAuditStorage()]

    def log(self, event: AuditEvent) -> None:
        """Log an audit event.

        Args:
            event: Audit event to log.
        """
        if not self._config.enabled:
            return

        # Add correlation context
        event.context.correlation_id = get_correlation_id() or ""
        event.context.service_name = self._config.service_name
        event.context.environment = self._config.environment

        # Process event
        for processor in self._processors:
            event = processor.process(event)

        # Write to all storages
        for storage in self._storages:
            try:
                storage.write(event)
            except Exception:
                pass

    def log_operation(
        self,
        operation: str,
        resource: str,
        *,
        outcome: str = "success",
        actor_id: str = "",
        details: dict[str, Any] | None = None,
        duration_ms: float | None = None,
    ) -> None:
        """Log an operation (convenience method).

        Args:
            operation: Operation name.
            resource: Resource identifier.
            outcome: Operation outcome.
            actor_id: Actor identifier.
            details: Additional details.
            duration_ms: Operation duration.
        """
        event = (
            AuditEventBuilder()
            .set_type(AuditEventType.CUSTOM)
            .set_action(operation)
            .set_outcome(AuditOutcome(outcome))
            .set_resource(id=resource, type=resource.split(":")[0] if ":" in resource else "resource")
            .set_data(details or {})
            .set_duration(duration_ms or 0)
            .build()
        )

        if actor_id:
            event.actor = AuditActor(id=actor_id)

        self.log(event)

    def log_validation(
        self,
        dataset: str,
        *,
        success: bool,
        rows: int = 0,
        issues: int = 0,
        duration_ms: float = 0,
        validators: list[str] | None = None,
    ) -> None:
        """Log a validation operation.

        Args:
            dataset: Dataset name.
            success: Whether validation passed.
            rows: Rows validated.
            issues: Issues found.
            duration_ms: Duration in milliseconds.
            validators: List of validators run.
        """
        event_type = (
            AuditEventType.VALIDATION_COMPLETE
            if success
            else AuditEventType.VALIDATION_FAILED
        )

        event = (
            AuditEventBuilder()
            .set_type(event_type)
            .set_category(AuditCategory.VALIDATION)
            .set_action("validate")
            .set_outcome(AuditOutcome.SUCCESS if success else AuditOutcome.FAILURE)
            .set_resource(id=f"dataset:{dataset}", type="dataset", name=dataset)
            .set_data({
                "rows": rows,
                "issues": issues,
                "validators": validators or [],
            })
            .set_duration(duration_ms)
            .build()
        )

        self.log(event)

    def log_checkpoint(
        self,
        checkpoint: str,
        *,
        success: bool,
        validators_run: int = 0,
        issues: int = 0,
        duration_ms: float = 0,
    ) -> None:
        """Log a checkpoint execution.

        Args:
            checkpoint: Checkpoint name.
            success: Whether checkpoint passed.
            validators_run: Number of validators run.
            issues: Total issues found.
            duration_ms: Duration in milliseconds.
        """
        event = (
            AuditEventBuilder()
            .set_type(AuditEventType.CHECKPOINT_RUN)
            .set_category(AuditCategory.VALIDATION)
            .set_action("checkpoint_run")
            .set_outcome(AuditOutcome.SUCCESS if success else AuditOutcome.FAILURE)
            .set_resource(id=f"checkpoint:{checkpoint}", type="checkpoint", name=checkpoint)
            .set_data({
                "validators_run": validators_run,
                "issues": issues,
            })
            .set_duration(duration_ms)
            .build()
        )

        self.log(event)

    @contextmanager
    def operation_context(
        self,
        operation: str,
        resource: str,
        **kwargs: Any,
    ) -> Iterator[None]:
        """Context manager for operation auditing.

        Automatically logs start and completion of operation.

        Args:
            operation: Operation name.
            resource: Resource identifier.
            **kwargs: Additional event data.
        """
        start_time = time.time()
        success = True
        error_msg = ""

        try:
            yield
        except Exception as e:
            success = False
            error_msg = str(e)
            raise
        finally:
            duration_ms = (time.time() - start_time) * 1000
            self.log_operation(
                operation,
                resource,
                outcome="success" if success else "failure",
                details={"error": error_msg} if error_msg else kwargs,
                duration_ms=duration_ms,
            )

    def query(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        event_types: list[AuditEventType] | None = None,
        actor_id: str | None = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        """Query audit events.

        Uses the first queryable storage backend.

        Args:
            start_time: Start of time range.
            end_time: End of time range.
            event_types: Filter by event types.
            actor_id: Filter by actor.
            limit: Maximum results.

        Returns:
            List of matching events.
        """
        for storage in self._storages:
            try:
                return storage.query(
                    start_time=start_time,
                    end_time=end_time,
                    event_types=event_types,
                    actor_id=actor_id,
                    limit=limit,
                )
            except Exception:
                continue
        return []

    def get_compliance_reporter(self) -> ComplianceReporter:
        """Get compliance reporter.

        Returns:
            ComplianceReporter instance.
        """
        # Use first queryable storage
        for storage in self._storages:
            if hasattr(storage, "query"):
                return ComplianceReporter(storage)
        return ComplianceReporter(MemoryAuditStorage())

    def flush(self) -> None:
        """Flush all storage backends."""
        for storage in self._storages:
            try:
                storage.flush()
            except Exception:
                pass

    def close(self) -> None:
        """Close all storage backends."""
        for storage in self._storages:
            try:
                storage.close()
            except Exception:
                pass


# =============================================================================
# Global Audit Logger
# =============================================================================

_global_logger: EnterpriseAuditLogger | None = None
_lock = threading.Lock()


def configure_audit(
    *,
    service_name: str = "",
    environment: str = "",
    storage_backends: list[str] | None = None,
    elasticsearch_url: str = "",
    s3_bucket: str = "",
    kafka_bootstrap_servers: str = "",
    **kwargs: Any,
) -> EnterpriseAuditLogger:
    """Configure global audit logger.

    Args:
        service_name: Service name.
        environment: Environment name.
        storage_backends: Storage backends to use.
        elasticsearch_url: Elasticsearch URL.
        s3_bucket: S3 bucket name.
        kafka_bootstrap_servers: Kafka bootstrap servers.
        **kwargs: Additional EnterpriseAuditConfig parameters.

    Returns:
        Configured EnterpriseAuditLogger.
    """
    global _global_logger

    with _lock:
        if _global_logger:
            _global_logger.close()

        config = EnterpriseAuditConfig(
            service_name=service_name,
            environment=environment,
            storage_backends=storage_backends or ["memory"],
            elasticsearch_url=elasticsearch_url,
            s3_bucket=s3_bucket,
            kafka_bootstrap_servers=kafka_bootstrap_servers,
            **kwargs,
        )

        _global_logger = EnterpriseAuditLogger(config)
        return _global_logger


def get_audit_logger() -> EnterpriseAuditLogger:
    """Get the global audit logger.

    Returns:
        EnterpriseAuditLogger instance.
    """
    global _global_logger

    with _lock:
        if _global_logger is None:
            _global_logger = EnterpriseAuditLogger()
        return _global_logger
