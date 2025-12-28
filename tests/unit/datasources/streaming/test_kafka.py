"""Tests for Kafka data source."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from truthound.datasources._protocols import ColumnType, DataSourceCapability
from truthound.datasources.streaming.kafka import (
    KafkaDataSource,
    KafkaDataSourceConfig,
    KafkaDataSourceError,
    KafkaConnectionError,
)
from truthound.datasources.streaming.base import (
    JSONDeserializer,
    AvroDeserializer,
    DeserializationError,
)


class TestKafkaDataSourceConfig:
    """Tests for KafkaDataSourceConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = KafkaDataSourceConfig(topic="test-topic")

        assert config.bootstrap_servers == "localhost:9092"
        assert config.topic == "test-topic"
        assert config.group_id == "truthound-datasource"
        assert config.security_protocol == "PLAINTEXT"
        assert config.max_messages == 10000
        assert config.auto_offset_reset == "earliest"

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = KafkaDataSourceConfig(
            bootstrap_servers="kafka1:9092,kafka2:9092",
            topic="my-topic",
            group_id="my-group",
            max_messages=5000,
            security_protocol="SASL_SSL",
        )

        assert config.bootstrap_servers == "kafka1:9092,kafka2:9092"
        assert config.topic == "my-topic"
        assert config.group_id == "my-group"
        assert config.max_messages == 5000
        assert config.security_protocol == "SASL_SSL"


class TestKafkaDataSource:
    """Tests for KafkaDataSource."""

    def test_requires_topic(self) -> None:
        """Test that topic is required."""
        with pytest.raises(KafkaDataSourceError, match="Topic name is required"):
            KafkaDataSource(KafkaDataSourceConfig())

    def test_creation(self) -> None:
        """Test source creation."""
        config = KafkaDataSourceConfig(topic="test-topic")
        source = KafkaDataSource(config)

        assert source.source_type == "kafka"
        assert source.topic == "test-topic"

    def test_name_property(self) -> None:
        """Test name property."""
        config = KafkaDataSourceConfig(topic="my-topic")
        source = KafkaDataSource(config)

        assert source.name == "kafka://my-topic"

    def test_name_property_custom(self) -> None:
        """Test custom name."""
        config = KafkaDataSourceConfig(
            name="custom_name",
            topic="topic",
        )
        source = KafkaDataSource(config)

        assert source.name == "custom_name"

    def test_capabilities(self) -> None:
        """Test capabilities."""
        config = KafkaDataSourceConfig(topic="topic")
        source = KafkaDataSource(config)
        caps = source.capabilities

        assert DataSourceCapability.SCHEMA_INFERENCE in caps
        assert DataSourceCapability.STREAMING in caps
        assert DataSourceCapability.SAMPLING in caps

    def test_from_connection_string(self) -> None:
        """Test factory method from connection string."""
        source = KafkaDataSource.from_connection_string(
            "kafka://localhost:9092/my-topic",
            max_messages=5000,
        )

        assert source.topic == "my-topic"
        assert "localhost:9092" in source.config.bootstrap_servers
        assert source.config.max_messages == 5000

    def test_from_connection_string_multiple_brokers(self) -> None:
        """Test connection string with multiple brokers."""
        source = KafkaDataSource.from_connection_string(
            "kafka://broker1:9092,broker2:9092/my-topic",
        )

        assert source.topic == "my-topic"
        assert "broker1:9092" in source.config.bootstrap_servers
        assert "broker2:9092" in source.config.bootstrap_servers

    def test_from_connection_string_invalid(self) -> None:
        """Test invalid connection string raises error."""
        with pytest.raises(KafkaDataSourceError, match="Invalid connection string"):
            KafkaDataSource.from_connection_string("localhost:9092")

    def test_from_confluent(self) -> None:
        """Test factory method for Confluent Cloud."""
        source = KafkaDataSource.from_confluent(
            bootstrap_servers="pkc-xxxxx.us-east-1.aws.confluent.cloud:9092",
            topic="my-topic",
            api_key="ABCDEF",
            api_secret="secret123",
        )

        assert source.topic == "my-topic"
        assert source.config.security_protocol == "SASL_SSL"
        assert source.config.sasl_mechanism == "PLAIN"
        assert source.config.sasl_username == "ABCDEF"
        assert source.config.sasl_password == "secret123"

    @pytest.mark.asyncio
    async def test_sample_async(self) -> None:
        """Test async sampling creates new config."""
        config = KafkaDataSourceConfig(
            bootstrap_servers="localhost:9092",
            topic="test-topic",
            max_messages=10000,
        )
        source = KafkaDataSource(config)

        sampled = await source.sample_async(100)

        assert sampled.config.max_messages == 100
        assert "sample" in sampled.name


class TestDeserializers:
    """Tests for message deserializers."""

    def test_json_deserializer_success(self) -> None:
        """Test successful JSON deserialization."""
        deserializer = JSONDeserializer()
        data = b'{"name": "Alice", "age": 30}'

        result = deserializer.deserialize(data)

        assert result == {"name": "Alice", "age": 30}

    def test_json_deserializer_invalid(self) -> None:
        """Test JSON deserialization with invalid data."""
        deserializer = JSONDeserializer()
        data = b"invalid json"

        with pytest.raises(DeserializationError):
            deserializer.deserialize(data)

    def test_json_deserializer_custom_encoding(self) -> None:
        """Test JSON deserializer with custom encoding."""
        deserializer = JSONDeserializer(encoding="utf-8")
        data = '{"name": "日本語"}'.encode("utf-8")

        result = deserializer.deserialize(data)
        assert result["name"] == "日本語"

    def test_avro_deserializer_creation(self) -> None:
        """Test Avro deserializer creation."""
        schema = {
            "type": "record",
            "name": "User",
            "fields": [
                {"name": "name", "type": "string"},
            ],
        }

        deserializer = AvroDeserializer(schema=schema)
        assert deserializer._schema == schema


class TestKafkaExceptions:
    """Tests for Kafka exceptions."""

    def test_kafka_datasource_error(self) -> None:
        """Test KafkaDataSourceError."""
        error = KafkaDataSourceError("Test error")
        assert str(error) == "Test error"

    def test_kafka_connection_error(self) -> None:
        """Test KafkaConnectionError."""
        error = KafkaConnectionError(
            "Connection refused",
            bootstrap_servers="localhost:9092",
        )
        assert error.bootstrap_servers == "localhost:9092"
        assert "Kafka connection failed" in str(error)
        assert "Connection refused" in str(error)

    def test_deserialization_error(self) -> None:
        """Test DeserializationError."""
        error = DeserializationError("Invalid format", offset=123)
        assert error.offset == 123
        assert "Deserialization failed" in str(error)
