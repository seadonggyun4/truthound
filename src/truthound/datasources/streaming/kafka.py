"""Apache Kafka data source implementation.

This module provides async Kafka data source with aiokafka driver support.
This is a BOUNDED data source for batch processing, not real-time streaming.

For real-time streaming validation, see truthound.realtime.streaming.KafkaSource.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable

from truthound.datasources._protocols import ColumnType, DataSourceCapability
from truthound.datasources.streaming.base import (
    BaseStreamingDataSource,
    StreamingDataSourceConfig,
    StreamingDataSourceError,
    DeserializationError,
)

if TYPE_CHECKING:
    import polars as pl


# =============================================================================
# Exceptions
# =============================================================================


class KafkaDataSourceError(StreamingDataSourceError):
    """Kafka-specific error."""

    pass


class KafkaConnectionError(KafkaDataSourceError):
    """Kafka connection error."""

    def __init__(self, message: str, bootstrap_servers: str | None = None) -> None:
        self.bootstrap_servers = bootstrap_servers
        super().__init__(f"Kafka connection failed: {message}")


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class KafkaDataSourceConfig(StreamingDataSourceConfig):
    """Configuration for Kafka data source.

    Attributes:
        bootstrap_servers: Kafka broker addresses.
        topic: Topic name to consume from.
        group_id: Consumer group ID.
        security_protocol: Security protocol (PLAINTEXT, SSL, SASL_SSL, etc.).
        sasl_mechanism: SASL mechanism (PLAIN, SCRAM-SHA-256, etc.).
        sasl_username: SASL username.
        sasl_password: SASL password.
        ssl_cafile: Path to CA certificate file.
        ssl_certfile: Path to client certificate file.
        ssl_keyfile: Path to client key file.
        partition: Specific partition to read from (None for all).
        start_offset: Starting offset (None for auto_offset_reset).
        end_offset: Ending offset (None for no limit).
        key_deserializer: Key format (json, string, bytes).
        include_key: Whether to include message key.
        include_headers: Whether to include message headers.
    """

    bootstrap_servers: str = "localhost:9092"
    topic: str = ""
    group_id: str = "truthound-datasource"
    security_protocol: str = "PLAINTEXT"
    sasl_mechanism: str | None = None
    sasl_username: str | None = None
    sasl_password: str | None = None
    ssl_cafile: str | None = None
    ssl_certfile: str | None = None
    ssl_keyfile: str | None = None
    partition: int | None = None
    start_offset: int | None = None
    end_offset: int | None = None
    key_deserializer: str = "string"
    include_key: bool = True
    include_headers: bool = False


# =============================================================================
# Kafka Data Source
# =============================================================================


class KafkaDataSource(BaseStreamingDataSource):
    """Async Kafka data source using aiokafka.

    Provides async access to Kafka topics with automatic schema inference
    and Polars LazyFrame conversion. This is a BOUNDED data source that
    consumes up to max_messages messages.

    Example:
        >>> # Basic usage
        >>> config = KafkaDataSourceConfig(
        ...     bootstrap_servers="localhost:9092",
        ...     topic="my-topic",
        ...     max_messages=10000,
        ... )
        >>> source = KafkaDataSource(config)
        >>>
        >>> async with source:
        ...     schema = await source.get_schema_async()
        ...     lf = await source.to_polars_lazyframe_async()

        >>> # From connection string
        >>> source = KafkaDataSource.from_connection_string(
        ...     "kafka://localhost:9092/my-topic",
        ...     group_id="my-group",
        ... )

        >>> # With SASL authentication
        >>> source = KafkaDataSource(KafkaDataSourceConfig(
        ...     bootstrap_servers="kafka.example.com:9093",
        ...     topic="secure-topic",
        ...     security_protocol="SASL_SSL",
        ...     sasl_mechanism="PLAIN",
        ...     sasl_username="user",
        ...     sasl_password="pass",
        ... ))
    """

    source_type = "kafka"

    def __init__(self, config: KafkaDataSourceConfig) -> None:
        """Initialize Kafka data source.

        Args:
            config: Kafka configuration.

        Raises:
            KafkaDataSourceError: If topic not specified.
        """
        if not config.topic:
            raise KafkaDataSourceError("Topic name is required")

        super().__init__(config)
        self._consumer: Any = None

    @property
    def config(self) -> KafkaDataSourceConfig:
        """Get typed configuration."""
        return self._config  # type: ignore

    @property
    def name(self) -> str:
        """Get data source name."""
        if self._config.name:
            return self._config.name
        return f"kafka://{self.config.topic}"

    @property
    def topic(self) -> str:
        """Get topic name."""
        return self.config.topic

    @property
    def capabilities(self) -> set[DataSourceCapability]:
        """Get data source capabilities."""
        return {
            DataSourceCapability.SCHEMA_INFERENCE,
            DataSourceCapability.STREAMING,
            DataSourceCapability.SAMPLING,
        }

    # -------------------------------------------------------------------------
    # Factory Methods
    # -------------------------------------------------------------------------

    @classmethod
    def from_connection_string(
        cls,
        connection_string: str,
        **kwargs: Any,
    ) -> "KafkaDataSource":
        """Create data source from connection string.

        Supports formats:
        - kafka://host:port/topic
        - host:port/topic
        - host:port,host2:port2/topic

        Args:
            connection_string: Kafka connection string.
            **kwargs: Additional configuration options.

        Returns:
            KafkaDataSource instance.

        Example:
            >>> source = KafkaDataSource.from_connection_string(
            ...     "kafka://localhost:9092/my-topic",
            ...     max_messages=5000,
            ... )
        """
        # Parse connection string
        conn = connection_string
        if conn.startswith("kafka://"):
            conn = conn[8:]

        if "/" in conn:
            servers, topic = conn.rsplit("/", 1)
        else:
            raise KafkaDataSourceError(
                f"Invalid connection string: {connection_string}. "
                "Expected format: kafka://host:port/topic"
            )

        config = KafkaDataSourceConfig(
            bootstrap_servers=servers,
            topic=topic,
            **kwargs,
        )
        return cls(config)

    @classmethod
    def from_confluent(
        cls,
        bootstrap_servers: str,
        topic: str,
        api_key: str,
        api_secret: str,
        **kwargs: Any,
    ) -> "KafkaDataSource":
        """Create data source for Confluent Cloud.

        Args:
            bootstrap_servers: Confluent Cloud bootstrap servers.
            topic: Topic name.
            api_key: Confluent API key.
            api_secret: Confluent API secret.
            **kwargs: Additional configuration options.

        Returns:
            KafkaDataSource instance.

        Example:
            >>> source = KafkaDataSource.from_confluent(
            ...     bootstrap_servers="pkc-xxxxx.us-east-1.aws.confluent.cloud:9092",
            ...     topic="my-topic",
            ...     api_key="ABCDEFGHIJKLMNOP",
            ...     api_secret="secret123",
            ... )
        """
        config = KafkaDataSourceConfig(
            bootstrap_servers=bootstrap_servers,
            topic=topic,
            security_protocol="SASL_SSL",
            sasl_mechanism="PLAIN",
            sasl_username=api_key,
            sasl_password=api_secret,
            **kwargs,
        )
        return cls(config)

    # -------------------------------------------------------------------------
    # Connection Management
    # -------------------------------------------------------------------------

    async def _create_connection_factory(self) -> Callable:
        """Create a connection factory."""

        async def factory():
            try:
                from aiokafka import AIOKafkaConsumer
            except ImportError:
                raise ImportError(
                    "aiokafka is required for Kafka support. "
                    "Install it with: pip install aiokafka"
                )

            consumer_kwargs = {
                "bootstrap_servers": self.config.bootstrap_servers,
                "group_id": self.config.group_id,
                "auto_offset_reset": self.config.auto_offset_reset,
                "enable_auto_commit": False,  # Manual offset management
                "security_protocol": self.config.security_protocol,
            }

            if self.config.sasl_mechanism:
                consumer_kwargs["sasl_mechanism"] = self.config.sasl_mechanism
            if self.config.sasl_username:
                consumer_kwargs["sasl_plain_username"] = self.config.sasl_username
            if self.config.sasl_password:
                consumer_kwargs["sasl_plain_password"] = self.config.sasl_password
            if self.config.ssl_cafile:
                consumer_kwargs["ssl_cafile"] = self.config.ssl_cafile
            if self.config.ssl_certfile:
                consumer_kwargs["ssl_certfile"] = self.config.ssl_certfile
            if self.config.ssl_keyfile:
                consumer_kwargs["ssl_keyfile"] = self.config.ssl_keyfile

            consumer = AIOKafkaConsumer(self.config.topic, **consumer_kwargs)
            return consumer

        return factory

    async def connect_async(self) -> None:
        """Establish connection to Kafka."""
        if self._is_connected:
            return

        async with self._lock:
            if self._is_connected:
                return

            try:
                from aiokafka import AIOKafkaConsumer
            except ImportError:
                raise ImportError(
                    "aiokafka is required for Kafka support. "
                    "Install it with: pip install aiokafka"
                )

            try:
                consumer_kwargs: dict[str, Any] = {
                    "bootstrap_servers": self.config.bootstrap_servers,
                    "group_id": self.config.group_id,
                    "auto_offset_reset": self.config.auto_offset_reset,
                    "enable_auto_commit": False,
                    "security_protocol": self.config.security_protocol,
                }

                if self.config.sasl_mechanism:
                    consumer_kwargs["sasl_mechanism"] = self.config.sasl_mechanism
                if self.config.sasl_username:
                    consumer_kwargs["sasl_plain_username"] = self.config.sasl_username
                if self.config.sasl_password:
                    consumer_kwargs["sasl_plain_password"] = self.config.sasl_password
                if self.config.ssl_cafile:
                    consumer_kwargs["ssl_cafile"] = self.config.ssl_cafile
                if self.config.ssl_certfile:
                    consumer_kwargs["ssl_certfile"] = self.config.ssl_certfile
                if self.config.ssl_keyfile:
                    consumer_kwargs["ssl_keyfile"] = self.config.ssl_keyfile

                self._consumer = AIOKafkaConsumer(
                    self.config.topic, **consumer_kwargs
                )
                await self._consumer.start()

                # Seek to specific offset if configured
                if self.config.start_offset is not None:
                    await self._seek_to_offset(self.config.start_offset)

                self._is_connected = True

            except Exception as e:
                raise KafkaConnectionError(
                    str(e), bootstrap_servers=self.config.bootstrap_servers
                )

    async def _seek_to_offset(self, offset: int) -> None:
        """Seek consumer to specific offset.

        Args:
            offset: Target offset.
        """
        partitions = self._consumer.assignment()
        for tp in partitions:
            self._consumer.seek(tp, offset)

    async def disconnect_async(self) -> None:
        """Close Kafka connection."""
        if not self._is_connected:
            return

        async with self._lock:
            if not self._is_connected:
                return

            if self._consumer:
                await self._consumer.stop()
                self._consumer = None

            self._is_connected = False

    async def validate_connection_async(self) -> bool:
        """Validate Kafka connection.

        Returns:
            True if connection is healthy.
        """
        try:
            if not self._is_connected:
                await self.connect_async()
            # Check if we can get topic info
            await self._get_topic_info()
            return True
        except Exception:
            return False

    # -------------------------------------------------------------------------
    # Message Consumption
    # -------------------------------------------------------------------------

    async def _consume_messages(
        self,
        max_messages: int,
        timeout: float | None = None,
    ) -> list[dict[str, Any]]:
        """Consume messages from Kafka topic.

        Args:
            max_messages: Maximum messages to consume.
            timeout: Timeout in seconds.

        Returns:
            List of deserialized messages.
        """
        if not self._is_connected:
            await self.connect_async()

        timeout = timeout or self.config.consume_timeout
        timeout_ms = int(timeout * 1000)
        messages: list[dict[str, Any]] = []
        consumed = 0
        empty_polls = 0
        max_empty_polls = 3

        while consumed < max_messages:
            # Fetch batch
            batch = await self._consumer.getmany(
                timeout_ms=min(timeout_ms, 5000),
                max_records=min(max_messages - consumed, 1000),
            )

            if not batch:
                empty_polls += 1
                if empty_polls >= max_empty_polls:
                    break
                continue

            empty_polls = 0

            for tp, records in batch.items():
                for record in records:
                    # Check end offset limit
                    if (
                        self.config.end_offset is not None
                        and record.offset >= self.config.end_offset
                    ):
                        return messages

                    try:
                        message = self._deserialize_message(record)
                        messages.append(message)
                        consumed += 1

                        if consumed >= max_messages:
                            return messages
                    except DeserializationError:
                        # Skip malformed messages
                        continue

        return messages

    def _deserialize_message(self, record: Any) -> dict[str, Any]:
        """Deserialize a Kafka message record.

        Args:
            record: Kafka consumer record.

        Returns:
            Deserialized message dict.
        """
        # Deserialize value
        if record.value is not None:
            message = self._deserializer.deserialize(record.value)
        else:
            message = {}

        # Add key if configured
        if self.config.include_key and record.key is not None:
            key = record.key
            if isinstance(key, bytes):
                try:
                    key = key.decode("utf-8")
                except UnicodeDecodeError:
                    key = key.hex()
            message[f"{self.config.metadata_prefix}key"] = key

        # Add metadata if configured
        if self.config.include_metadata:
            prefix = self.config.metadata_prefix
            message[f"{prefix}topic"] = record.topic
            message[f"{prefix}partition"] = record.partition
            message[f"{prefix}offset"] = record.offset
            message[f"{prefix}timestamp"] = record.timestamp

            if self.config.include_headers and record.headers:
                headers = {
                    k: v.decode("utf-8") if isinstance(v, bytes) else v
                    for k, v in record.headers
                }
                message[f"{prefix}headers"] = headers

        return message

    # -------------------------------------------------------------------------
    # Topic Information
    # -------------------------------------------------------------------------

    async def _get_topic_info(self) -> dict[str, Any]:
        """Get topic metadata.

        Returns:
            Topic metadata including partitions.
        """
        if not self._is_connected:
            await self.connect_async()

        partitions = self._consumer.partitions_for_topic(self.config.topic)

        return {
            "topic": self.config.topic,
            "partitions": list(partitions) if partitions else [],
            "partition_count": len(partitions) if partitions else 0,
        }

    async def get_topic_offsets_async(self) -> dict[str, dict[int, tuple[int, int]]]:
        """Get topic partition offsets.

        Returns:
            Dict with beginning and end offsets per partition.
        """
        if not self._is_connected:
            await self.connect_async()

        from aiokafka import TopicPartition

        partitions = self._consumer.partitions_for_topic(self.config.topic)
        if not partitions:
            return {"offsets": {}}

        tps = [TopicPartition(self.config.topic, p) for p in partitions]

        beginning = await self._consumer.beginning_offsets(tps)
        end = await self._consumer.end_offsets(tps)

        offsets = {}
        for tp in tps:
            offsets[tp.partition] = (beginning[tp], end[tp])

        return {"offsets": offsets}

    # -------------------------------------------------------------------------
    # Sampling
    # -------------------------------------------------------------------------

    async def sample_async(
        self, n: int = 1000, seed: int | None = None
    ) -> "KafkaDataSource":
        """Create a sampled data source.

        Note: Kafka doesn't support random sampling. This returns
        a new data source configured to consume fewer messages.

        Args:
            n: Number of messages to consume.
            seed: Ignored (Kafka doesn't support random access).

        Returns:
            New KafkaDataSource with reduced message limit.
        """
        config = KafkaDataSourceConfig(
            bootstrap_servers=self.config.bootstrap_servers,
            topic=self.config.topic,
            group_id=f"{self.config.group_id}_sample",
            security_protocol=self.config.security_protocol,
            sasl_mechanism=self.config.sasl_mechanism,
            sasl_username=self.config.sasl_username,
            sasl_password=self.config.sasl_password,
            name=f"{self.name}_sample",
            max_messages=n,
            auto_offset_reset="earliest",
            deserializer_type=self.config.deserializer_type,
        )
        return KafkaDataSource(config)

    # -------------------------------------------------------------------------
    # Iteration
    # -------------------------------------------------------------------------

    async def iter_messages_async(
        self,
        batch_size: int = 100,
        max_messages: int | None = None,
    ):
        """Iterate over messages in batches.

        Args:
            batch_size: Messages per batch.
            max_messages: Maximum total messages.

        Yields:
            Batches of messages.

        Example:
            >>> async for batch in source.iter_messages_async(batch_size=100):
            ...     for msg in batch:
            ...         process(msg)
        """
        if not self._is_connected:
            await self.connect_async()

        max_messages = max_messages or self.config.max_messages
        consumed = 0

        while consumed < max_messages:
            batch = await self._consume_messages(
                min(batch_size, max_messages - consumed),
                timeout=5.0,
            )

            if not batch:
                break

            yield batch
            consumed += len(batch)

    # -------------------------------------------------------------------------
    # Consumer Group Operations
    # -------------------------------------------------------------------------

    async def commit_offsets_async(self) -> None:
        """Commit current offsets.

        Call this after processing messages to save progress.
        """
        if not self._is_connected:
            await self.connect_async()

        await self._consumer.commit()

    async def get_committed_offsets_async(self) -> dict[int, int]:
        """Get committed offsets for topic partitions.

        Returns:
            Dict mapping partition to committed offset.
        """
        if not self._is_connected:
            await self.connect_async()

        from aiokafka import TopicPartition

        partitions = self._consumer.partitions_for_topic(self.config.topic)
        if not partitions:
            return {}

        tps = [TopicPartition(self.config.topic, p) for p in partitions]
        committed = {}

        for tp in tps:
            offset_meta = await self._consumer.committed(tp)
            if offset_meta is not None:
                committed[tp.partition] = offset_meta

        return committed
