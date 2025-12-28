"""Apache Kafka adapter using aiokafka.

Provides async Kafka integration with:
- Consumer group management
- Partition assignment
- Offset management
- SASL/SSL authentication
- Message serialization/deserialization
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator
import json
import logging

from truthound.realtime.protocols import (
    StreamMessage,
    MessageHeader,
    DeserializationFormat,
    OffsetReset,
    AckMode,
    SerializationError,
)
from truthound.realtime.adapters.base import (
    BaseStreamAdapter,
    AdapterConfig,
)


logger = logging.getLogger(__name__)


@dataclass
class KafkaAdapterConfig(AdapterConfig):
    """Configuration for Kafka adapter.

    Attributes:
        bootstrap_servers: Comma-separated list of Kafka brokers
        topic: Topic to consume from
        topics: Multiple topics to consume from
        consumer_group: Consumer group ID
        auto_offset_reset: Offset reset policy
        enable_auto_commit: Enable automatic offset commits
        security_protocol: Security protocol (PLAINTEXT, SSL, SASL_SSL, SASL_PLAINTEXT)
        sasl_mechanism: SASL mechanism (PLAIN, SCRAM-SHA-256, SCRAM-SHA-512)
        sasl_username: SASL username
        sasl_password: SASL password
        ssl_context: SSL context for SSL connections
        session_timeout_ms: Session timeout
        heartbeat_interval_ms: Heartbeat interval
    """

    # Connection
    bootstrap_servers: str = "localhost:9092"
    topic: str = ""
    topics: list[str] = field(default_factory=list)

    # Consumer
    consumer_group: str = "truthound-consumer"
    auto_offset_reset: OffsetReset = OffsetReset.LATEST
    enable_auto_commit: bool = False
    ack_mode: AckMode = AckMode.MANUAL

    # Timeouts
    session_timeout_ms: int = 30000
    heartbeat_interval_ms: int = 3000
    max_poll_interval_ms: int = 300000

    # Security
    security_protocol: str = "PLAINTEXT"
    sasl_mechanism: str | None = None
    sasl_username: str | None = None
    sasl_password: str | None = None
    ssl_cafile: str | None = None
    ssl_certfile: str | None = None
    ssl_keyfile: str | None = None

    # Serialization
    key_deserializer: DeserializationFormat = DeserializationFormat.STRING
    value_deserializer: DeserializationFormat = DeserializationFormat.JSON

    # Producer settings (when used as sink)
    acks: str = "all"
    compression_type: str | None = None  # gzip, snappy, lz4, zstd
    max_request_size: int = 1048576  # 1MB


class KafkaAdapter(BaseStreamAdapter[dict[str, Any], KafkaAdapterConfig]):
    """Apache Kafka streaming adapter using aiokafka.

    Provides async Kafka consumer and producer capabilities
    with comprehensive configuration options.

    Example:
        >>> config = KafkaAdapterConfig(
        ...     bootstrap_servers="localhost:9092",
        ...     topic="events",
        ...     consumer_group="validator",
        ... )
        >>> async with KafkaAdapter(config) as adapter:
        ...     async for message in adapter.consume():
        ...         print(message.value)
        ...         await adapter.commit(message)

    Requires:
        pip install aiokafka
    """

    adapter_type = "kafka"

    def __init__(
        self,
        config: KafkaAdapterConfig | None = None,
        **kwargs: Any,
    ):
        config = config or KafkaAdapterConfig()
        super().__init__(config, **kwargs)

        self._consumer: Any = None
        self._producer: Any = None
        self._topics: list[str] = []

    @property
    def source_name(self) -> str:
        """Get Kafka source identifier."""
        topics = ",".join(self._topics) if self._topics else self._config.topic
        return f"kafka://{self._config.bootstrap_servers}/{topics}"

    async def _do_connect(self) -> None:
        """Connect to Kafka cluster."""
        try:
            from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
        except ImportError:
            raise ImportError(
                "aiokafka is required for Kafka adapter. "
                "Install with: pip install aiokafka"
            )

        # Determine topics
        self._topics = self._config.topics or (
            [self._config.topic] if self._config.topic else []
        )

        if not self._topics:
            raise ValueError("At least one topic is required")

        # Build consumer config
        consumer_config = {
            "bootstrap_servers": self._config.bootstrap_servers,
            "group_id": self._config.consumer_group,
            "auto_offset_reset": self._config.auto_offset_reset.value,
            "enable_auto_commit": self._config.enable_auto_commit,
            "session_timeout_ms": self._config.session_timeout_ms,
            "heartbeat_interval_ms": self._config.heartbeat_interval_ms,
            "max_poll_interval_ms": self._config.max_poll_interval_ms,
            "security_protocol": self._config.security_protocol,
        }

        # SASL config
        if self._config.sasl_mechanism:
            consumer_config["sasl_mechanism"] = self._config.sasl_mechanism
            consumer_config["sasl_plain_username"] = self._config.sasl_username
            consumer_config["sasl_plain_password"] = self._config.sasl_password

        # SSL config
        if self._config.ssl_cafile:
            consumer_config["ssl_cafile"] = self._config.ssl_cafile
        if self._config.ssl_certfile:
            consumer_config["ssl_certfile"] = self._config.ssl_certfile
        if self._config.ssl_keyfile:
            consumer_config["ssl_keyfile"] = self._config.ssl_keyfile

        # Deserializers
        consumer_config["key_deserializer"] = self._get_deserializer(
            self._config.key_deserializer
        )
        consumer_config["value_deserializer"] = self._get_deserializer(
            self._config.value_deserializer
        )

        # Create consumer
        self._consumer = AIOKafkaConsumer(*self._topics, **consumer_config)
        await self._consumer.start()

        # Create producer for sink operations
        producer_config = {
            "bootstrap_servers": self._config.bootstrap_servers,
            "acks": self._config.acks,
            "security_protocol": self._config.security_protocol,
        }

        if self._config.compression_type:
            producer_config["compression_type"] = self._config.compression_type

        if self._config.sasl_mechanism:
            producer_config["sasl_mechanism"] = self._config.sasl_mechanism
            producer_config["sasl_plain_username"] = self._config.sasl_username
            producer_config["sasl_plain_password"] = self._config.sasl_password

        # Serializers
        producer_config["key_serializer"] = self._get_serializer(
            self._config.key_deserializer
        )
        producer_config["value_serializer"] = self._get_serializer(
            self._config.value_deserializer
        )

        self._producer = AIOKafkaProducer(**producer_config)
        await self._producer.start()

        logger.info(f"Connected to Kafka: {self._config.bootstrap_servers}")

    async def _do_disconnect(self) -> None:
        """Disconnect from Kafka cluster."""
        if self._consumer:
            await self._consumer.stop()
            self._consumer = None

        if self._producer:
            await self._producer.stop()
            self._producer = None

        logger.info("Disconnected from Kafka")

    async def _do_consume(self) -> AsyncIterator[StreamMessage[dict[str, Any]]]:
        """Consume messages from Kafka."""
        if not self._consumer:
            raise RuntimeError("Consumer not initialized")

        async for record in self._consumer:
            # Extract headers
            headers = tuple(
                MessageHeader(key=h[0], value=h[1])
                for h in (record.headers or [])
            )

            # Build message
            message = StreamMessage(
                key=record.key,
                value=record.value if isinstance(record.value, dict) else {"value": record.value},
                partition=record.partition,
                offset=record.offset,
                timestamp=datetime.fromtimestamp(
                    record.timestamp / 1000, tz=timezone.utc
                ),
                headers=headers,
                topic=record.topic,
                metadata={
                    "serialized_key_size": record.serialized_key_size,
                    "serialized_value_size": record.serialized_value_size,
                },
            )

            yield message

    async def _do_produce(self, message: StreamMessage[dict[str, Any]]) -> None:
        """Produce message to Kafka."""
        if not self._producer:
            raise RuntimeError("Producer not initialized")

        # Convert headers
        headers = [(h.key, h.value) for h in message.headers] if message.headers else None

        await self._producer.send_and_wait(
            topic=message.topic or self._config.topic,
            key=message.key,
            value=message.value,
            partition=message.partition if message.partition >= 0 else None,
            headers=headers,
        )

    async def _do_produce_batch(self, messages: list[StreamMessage[dict[str, Any]]]) -> None:
        """Produce batch of messages to Kafka."""
        if not self._producer:
            raise RuntimeError("Producer not initialized")

        batch = self._producer.create_batch()

        for message in messages:
            headers = [(h.key, h.value) for h in message.headers] if message.headers else None

            # Try to append to batch
            metadata = batch.append(
                key=message.key.encode() if message.key else None,
                value=json.dumps(message.value).encode(),
                timestamp=int(message.timestamp.timestamp() * 1000),
                headers=headers,
            )

            if metadata is None:
                # Batch is full, send and create new one
                await self._producer.send_batch(
                    batch,
                    message.topic or self._config.topic,
                    partition=message.partition if message.partition >= 0 else None,
                )
                batch = self._producer.create_batch()
                batch.append(
                    key=message.key.encode() if message.key else None,
                    value=json.dumps(message.value).encode(),
                    timestamp=int(message.timestamp.timestamp() * 1000),
                    headers=headers,
                )

        # Send remaining batch
        if len(batch) > 0:
            topic = messages[-1].topic or self._config.topic
            await self._producer.send_batch(batch, topic, partition=None)

    async def _do_flush(self) -> None:
        """Flush producer buffer."""
        if self._producer:
            await self._producer.flush()

    async def _do_commit(self, message: StreamMessage[dict[str, Any]]) -> None:
        """Commit offset for message."""
        if not self._consumer:
            return

        from aiokafka import TopicPartition

        tp = TopicPartition(message.topic, message.partition)
        await self._consumer.commit({tp: message.offset + 1})

    async def _do_commit_batch(self, messages: list[StreamMessage[dict[str, Any]]]) -> None:
        """Commit offsets for batch of messages."""
        if not self._consumer or not messages:
            return

        from aiokafka import TopicPartition

        # Build offset map (latest offset per partition)
        offsets: dict[Any, int] = {}
        for msg in messages:
            tp = TopicPartition(msg.topic, msg.partition)
            current = offsets.get(tp, -1)
            if msg.offset > current:
                offsets[tp] = msg.offset + 1

        await self._consumer.commit(offsets)

    async def _do_seek(self, partition: int, offset: int) -> None:
        """Seek to specific offset."""
        if not self._consumer:
            return

        from aiokafka import TopicPartition

        for topic in self._topics:
            tp = TopicPartition(topic, partition)
            self._consumer.seek(tp, offset)

    # -------------------------------------------------------------------------
    # Serialization Helpers
    # -------------------------------------------------------------------------

    def _get_deserializer(self, format: DeserializationFormat):
        """Get deserializer function for format."""
        if format == DeserializationFormat.JSON:
            return lambda m: json.loads(m.decode("utf-8")) if m else None
        elif format == DeserializationFormat.STRING:
            return lambda m: m.decode("utf-8") if m else None
        elif format == DeserializationFormat.BYTES:
            return lambda m: m
        elif format == DeserializationFormat.MSGPACK:
            try:
                import msgpack
                return lambda m: msgpack.unpackb(m) if m else None
            except ImportError:
                raise ImportError("msgpack is required for MSGPACK deserialization")
        else:
            return lambda m: m.decode("utf-8") if m else None

    def _get_serializer(self, format: DeserializationFormat):
        """Get serializer function for format."""
        if format == DeserializationFormat.JSON:
            return lambda m: json.dumps(m).encode("utf-8") if m else None
        elif format == DeserializationFormat.STRING:
            return lambda m: str(m).encode("utf-8") if m else None
        elif format == DeserializationFormat.BYTES:
            return lambda m: m if isinstance(m, bytes) else str(m).encode("utf-8")
        elif format == DeserializationFormat.MSGPACK:
            try:
                import msgpack
                return lambda m: msgpack.packb(m) if m else None
            except ImportError:
                raise ImportError("msgpack is required for MSGPACK serialization")
        else:
            return lambda m: str(m).encode("utf-8") if m else None

    # -------------------------------------------------------------------------
    # Kafka-Specific Methods
    # -------------------------------------------------------------------------

    async def get_partition_info(self) -> dict[str, list[int]]:
        """Get partition information for subscribed topics.

        Returns:
            Dict mapping topic names to partition lists
        """
        if not self._consumer:
            return {}

        result = {}
        for topic in self._topics:
            partitions = self._consumer.partitions_for_topic(topic)
            result[topic] = list(partitions) if partitions else []

        return result

    async def get_committed_offsets(self) -> dict[str, dict[int, int]]:
        """Get committed offsets for all partitions.

        Returns:
            Dict mapping topic to partition->offset mapping
        """
        if not self._consumer:
            return {}

        from aiokafka import TopicPartition

        result: dict[str, dict[int, int]] = {}

        for topic in self._topics:
            partitions = self._consumer.partitions_for_topic(topic)
            if not partitions:
                continue

            result[topic] = {}
            for partition in partitions:
                tp = TopicPartition(topic, partition)
                offset = await self._consumer.committed(tp)
                if offset is not None:
                    result[topic][partition] = offset

        return result

    async def get_end_offsets(self) -> dict[str, dict[int, int]]:
        """Get end offsets for all partitions.

        Returns:
            Dict mapping topic to partition->offset mapping
        """
        if not self._consumer:
            return {}

        from aiokafka import TopicPartition

        result: dict[str, dict[int, int]] = {}

        for topic in self._topics:
            partitions = self._consumer.partitions_for_topic(topic)
            if not partitions:
                continue

            tps = [TopicPartition(topic, p) for p in partitions]
            offsets = await self._consumer.end_offsets(tps)

            result[topic] = {tp.partition: offset for tp, offset in offsets.items()}

        return result

    async def get_consumer_lag(self) -> dict[str, dict[int, int]]:
        """Calculate consumer lag for all partitions.

        Returns:
            Dict mapping topic to partition->lag mapping
        """
        committed = await self.get_committed_offsets()
        end_offsets = await self.get_end_offsets()

        result: dict[str, dict[int, int]] = {}

        for topic in self._topics:
            result[topic] = {}
            topic_committed = committed.get(topic, {})
            topic_end = end_offsets.get(topic, {})

            for partition in topic_end:
                end = topic_end.get(partition, 0)
                committed_offset = topic_committed.get(partition, 0)
                result[topic][partition] = end - committed_offset

        return result
