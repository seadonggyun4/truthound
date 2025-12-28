"""Streaming data source implementations.

This package provides data sources for streaming platforms including:
- Apache Kafka (message streaming)

Note: These are **bounded** data sources for batch processing.
For real-time streaming validation, see truthound.realtime.streaming.

The key difference:
- Phase 5 DataSources: Read a bounded set of messages for validation
- Phase 10 Realtime: Continuous stream validation with windowing

Example:
    >>> from truthound.datasources.streaming import KafkaDataSource
    >>>
    >>> async with KafkaDataSource(KafkaDataSourceConfig(
    ...     bootstrap_servers="localhost:9092",
    ...     topic="my-topic",
    ...     max_messages=10000,
    ... )) as source:
    ...     schema = await source.get_schema_async()
    ...     lf = await source.to_polars_lazyframe_async()
"""

from truthound.datasources.streaming.base import (
    BaseStreamingDataSource,
    StreamingDataSourceConfig,
    MessageDeserializer,
    JSONDeserializer,
    AvroDeserializer,
)
from truthound.datasources.streaming.kafka import (
    KafkaDataSource,
    KafkaDataSourceConfig,
)

__all__ = [
    # Base
    "BaseStreamingDataSource",
    "StreamingDataSourceConfig",
    "MessageDeserializer",
    "JSONDeserializer",
    "AvroDeserializer",
    # Kafka
    "KafkaDataSource",
    "KafkaDataSourceConfig",
]
