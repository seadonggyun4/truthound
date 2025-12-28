"""Base classes for streaming data sources.

This module provides common functionality for streaming platform data sources.
These are bounded data sources that consume a fixed number of messages.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Protocol

from truthound.datasources._protocols import ColumnType, DataSourceCapability
from truthound.datasources.async_base import (
    AsyncBaseDataSource,
    AsyncDataSourceConfig,
    AsyncDataSourceError,
)
from truthound.datasources.nosql.base import DocumentSchemaInferrer

if TYPE_CHECKING:
    import polars as pl


# =============================================================================
# Exceptions
# =============================================================================


class StreamingDataSourceError(AsyncDataSourceError):
    """Base exception for streaming data source errors."""

    pass


class DeserializationError(StreamingDataSourceError):
    """Error during message deserialization."""

    def __init__(self, message: str, offset: int | None = None) -> None:
        self.offset = offset
        super().__init__(f"Deserialization failed: {message}")


# =============================================================================
# Message Deserializers
# =============================================================================


class MessageDeserializer(Protocol):
    """Protocol for message deserializers."""

    def deserialize(self, data: bytes) -> dict[str, Any]:
        """Deserialize bytes to dictionary.

        Args:
            data: Raw message bytes.

        Returns:
            Deserialized message as dict.
        """
        ...


class JSONDeserializer:
    """JSON message deserializer."""

    def __init__(self, encoding: str = "utf-8") -> None:
        """Initialize deserializer.

        Args:
            encoding: Character encoding.
        """
        self._encoding = encoding

    def deserialize(self, data: bytes) -> dict[str, Any]:
        """Deserialize JSON bytes.

        Args:
            data: JSON bytes.

        Returns:
            Parsed dictionary.

        Raises:
            DeserializationError: If JSON parsing fails.
        """
        import json

        try:
            return json.loads(data.decode(self._encoding))
        except Exception as e:
            raise DeserializationError(str(e))


class AvroDeserializer:
    """Avro message deserializer.

    Requires fastavro library.
    """

    def __init__(
        self,
        schema: dict[str, Any] | None = None,
        schema_registry_url: str | None = None,
    ) -> None:
        """Initialize Avro deserializer.

        Args:
            schema: Avro schema dict.
            schema_registry_url: Optional schema registry URL.
        """
        self._schema = schema
        self._registry_url = schema_registry_url
        self._parsed_schema: Any = None

    def deserialize(self, data: bytes) -> dict[str, Any]:
        """Deserialize Avro bytes.

        Args:
            data: Avro bytes.

        Returns:
            Parsed dictionary.

        Raises:
            DeserializationError: If Avro parsing fails.
        """
        try:
            import fastavro
            from io import BytesIO
        except ImportError:
            raise ImportError(
                "fastavro is required for Avro deserialization. "
                "Install it with: pip install fastavro"
            )

        try:
            if self._parsed_schema is None and self._schema:
                self._parsed_schema = fastavro.parse_schema(self._schema)

            reader = fastavro.reader(BytesIO(data), self._parsed_schema)
            return next(reader)
        except Exception as e:
            raise DeserializationError(str(e))


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class StreamingDataSourceConfig(AsyncDataSourceConfig):
    """Base configuration for streaming data sources.

    Attributes:
        max_messages: Maximum messages to consume (bounded).
        consume_timeout: Timeout waiting for messages (seconds).
        auto_offset_reset: Where to start reading (earliest, latest).
        deserializer_type: Message format (json, avro, string).
        include_metadata: Whether to include message metadata.
        metadata_prefix: Prefix for metadata fields.
        schema_sample_size: Messages to sample for schema inference.
    """

    max_messages: int = 10000
    consume_timeout: float = 30.0
    auto_offset_reset: str = "earliest"
    deserializer_type: str = "json"
    include_metadata: bool = False
    metadata_prefix: str = "_kafka_"
    schema_sample_size: int = 100


# =============================================================================
# Abstract Streaming Base Data Source
# =============================================================================


class BaseStreamingDataSource(AsyncBaseDataSource[StreamingDataSourceConfig]):
    """Abstract base class for streaming data sources.

    Provides common functionality for consuming bounded message sets
    from streaming platforms.
    """

    source_type: str = "streaming"

    def __init__(self, config: StreamingDataSourceConfig) -> None:
        """Initialize streaming data source.

        Args:
            config: Streaming configuration.
        """
        super().__init__(config)
        self._deserializer = self._create_deserializer()
        self._schema_inferrer = DocumentSchemaInferrer(
            flatten_nested=False,
            infer_nested_types=True,
        )
        self._consumed_messages: list[dict[str, Any]] = []

    @classmethod
    def _default_config(cls) -> StreamingDataSourceConfig:
        """Create default configuration."""
        return StreamingDataSourceConfig()

    @property
    def capabilities(self) -> set[DataSourceCapability]:
        """Get data source capabilities."""
        return {
            DataSourceCapability.SCHEMA_INFERENCE,
            DataSourceCapability.STREAMING,
        }

    def _create_deserializer(self) -> MessageDeserializer:
        """Create message deserializer based on config.

        Returns:
            Deserializer instance.
        """
        deserializer_type = self.config.deserializer_type.lower()

        if deserializer_type == "json":
            return JSONDeserializer()
        elif deserializer_type == "avro":
            return AvroDeserializer()
        elif deserializer_type == "string":
            return JSONDeserializer()  # Treat as JSON
        else:
            raise StreamingDataSourceError(
                f"Unknown deserializer type: {deserializer_type}"
            )

    # -------------------------------------------------------------------------
    # Abstract Methods
    # -------------------------------------------------------------------------

    @abstractmethod
    async def _consume_messages(
        self, max_messages: int, timeout: float | None = None
    ) -> list[dict[str, Any]]:
        """Consume messages from the stream.

        Args:
            max_messages: Maximum messages to consume.
            timeout: Optional timeout in seconds.

        Returns:
            List of deserialized messages.
        """
        pass

    @abstractmethod
    async def _get_topic_info(self) -> dict[str, Any]:
        """Get topic/stream information.

        Returns:
            Topic metadata.
        """
        pass

    # -------------------------------------------------------------------------
    # Schema Inference
    # -------------------------------------------------------------------------

    async def get_schema_async(self) -> dict[str, ColumnType]:
        """Infer schema from sample messages.

        Returns:
            Column name to type mapping.
        """
        if self._cached_schema is not None:
            return self._cached_schema

        # Consume sample messages
        sample_messages = await self._consume_messages(
            self.config.schema_sample_size,
            timeout=self.config.consume_timeout,
        )

        if not sample_messages:
            raise StreamingDataSourceError(
                "No messages available for schema inference"
            )

        # Store for later use
        self._consumed_messages = sample_messages

        # Infer schema
        self._cached_schema = self._schema_inferrer.infer_from_documents(
            sample_messages
        )
        return self._cached_schema

    # -------------------------------------------------------------------------
    # Polars Conversion
    # -------------------------------------------------------------------------

    async def to_polars_lazyframe_async(self) -> "pl.LazyFrame":
        """Convert consumed messages to Polars LazyFrame.

        Returns:
            Polars LazyFrame.
        """
        import polars as pl

        # Consume if not already done
        if not self._consumed_messages:
            self._consumed_messages = await self._consume_messages(
                self.config.max_messages,
                timeout=self.config.consume_timeout,
            )

        if not self._consumed_messages:
            # Return empty DataFrame with inferred schema
            schema = await self.get_schema_async()
            return pl.DataFrame({col: [] for col in schema.keys()}).lazy()

        # Normalize messages (ensure all have same keys)
        all_keys = set()
        for msg in self._consumed_messages:
            all_keys.update(msg.keys())

        normalized = []
        for msg in self._consumed_messages:
            normalized_msg = {key: msg.get(key) for key in all_keys}
            normalized.append(normalized_msg)

        return pl.DataFrame(normalized).lazy()

    # -------------------------------------------------------------------------
    # Row Count
    # -------------------------------------------------------------------------

    async def get_row_count_async(self) -> int | None:
        """Get consumed message count.

        Returns:
            Number of consumed messages.
        """
        if self._consumed_messages:
            return len(self._consumed_messages)
        return None

    @property
    def row_count(self) -> int | None:
        """Get cached message count."""
        return len(self._consumed_messages) if self._consumed_messages else None
