"""Tests for Elasticsearch data source."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from truthound.datasources._protocols import ColumnType, DataSourceCapability
from truthound.datasources.nosql.elasticsearch import (
    ElasticsearchDataSource,
    ElasticsearchConfig,
    ElasticsearchError,
    ElasticsearchConnectionError,
)


class TestElasticsearchConfig:
    """Tests for ElasticsearchConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = ElasticsearchConfig(index="test-index")

        assert config.hosts == ["http://localhost:9200"]
        assert config.index == "test-index"
        assert config.cloud_id is None
        assert config.verify_certs is True
        assert config.scroll_timeout == "5m"
        assert config.scroll_size == 1000

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = ElasticsearchConfig(
            hosts=["http://node1:9200", "http://node2:9200"],
            index="my-index-*",
            scroll_size=500,
            request_timeout=60,
        )

        assert len(config.hosts) == 2
        assert config.index == "my-index-*"
        assert config.scroll_size == 500
        assert config.request_timeout == 60


class TestElasticsearchDataSource:
    """Tests for ElasticsearchDataSource."""

    def test_requires_index(self) -> None:
        """Test that index is required."""
        with pytest.raises(ElasticsearchError, match="Index name is required"):
            ElasticsearchDataSource(ElasticsearchConfig())

    def test_creation(self) -> None:
        """Test source creation."""
        config = ElasticsearchConfig(index="test-index")
        source = ElasticsearchDataSource(config)

        assert source.source_type == "elasticsearch"
        assert source.index == "test-index"

    def test_name_property(self) -> None:
        """Test name property."""
        config = ElasticsearchConfig(index="my-index")
        source = ElasticsearchDataSource(config)

        assert source.name == "es://my-index"

    def test_name_property_custom(self) -> None:
        """Test custom name."""
        config = ElasticsearchConfig(
            name="custom_name",
            index="idx",
        )
        source = ElasticsearchDataSource(config)

        assert source.name == "custom_name"

    def test_capabilities(self) -> None:
        """Test capabilities include expected values."""
        config = ElasticsearchConfig(index="idx")
        source = ElasticsearchDataSource(config)
        caps = source.capabilities

        assert DataSourceCapability.SCHEMA_INFERENCE in caps
        assert DataSourceCapability.SAMPLING in caps
        assert DataSourceCapability.STREAMING in caps
        assert DataSourceCapability.ROW_COUNT in caps

    def test_from_cloud(self) -> None:
        """Test factory method for Elastic Cloud."""
        source = ElasticsearchDataSource.from_cloud(
            cloud_id="my-deployment:base64string",
            api_key="my-api-key",
            index="logs-*",
        )

        assert source.index == "logs-*"
        assert source.config.cloud_id == "my-deployment:base64string"
        assert source.config.api_key == "my-api-key"

    def test_from_hosts(self) -> None:
        """Test factory method from host list."""
        source = ElasticsearchDataSource.from_hosts(
            hosts=["http://node1:9200", "http://node2:9200"],
            index="my-index",
            username="elastic",
            password="changeme",
        )

        assert source.index == "my-index"
        assert len(source.config.hosts) == 2
        assert source.config.username == "elastic"

    def test_es_type_mapping(self) -> None:
        """Test Elasticsearch type to ColumnType mapping."""
        mapping = ElasticsearchDataSource.ES_TYPE_MAPPING

        assert mapping["text"] == ColumnType.STRING
        assert mapping["keyword"] == ColumnType.STRING
        assert mapping["long"] == ColumnType.INTEGER
        assert mapping["integer"] == ColumnType.INTEGER
        assert mapping["double"] == ColumnType.FLOAT
        assert mapping["boolean"] == ColumnType.BOOLEAN
        assert mapping["date"] == ColumnType.DATETIME
        assert mapping["binary"] == ColumnType.BINARY
        assert mapping["object"] == ColumnType.STRUCT
        assert mapping["nested"] == ColumnType.STRUCT

    def test_parse_mapping_properties(self) -> None:
        """Test parsing Elasticsearch mappings."""
        config = ElasticsearchConfig(index="test")
        source = ElasticsearchDataSource(config)

        properties = {
            "title": {"type": "text"},
            "count": {"type": "integer"},
            "price": {"type": "double"},
            "active": {"type": "boolean"},
            "created": {"type": "date"},
        }

        schema: dict[str, ColumnType] = {}
        source._parse_mapping_properties(properties, "", schema)

        assert schema["title"] == ColumnType.STRING
        assert schema["count"] == ColumnType.INTEGER
        assert schema["price"] == ColumnType.FLOAT
        assert schema["active"] == ColumnType.BOOLEAN
        assert schema["created"] == ColumnType.DATETIME

    def test_parse_nested_mapping(self) -> None:
        """Test parsing nested Elasticsearch mappings."""
        config = ElasticsearchConfig(index="test", flatten_nested=True)
        source = ElasticsearchDataSource(config)

        properties = {
            "user": {
                "properties": {
                    "name": {"type": "keyword"},
                    "age": {"type": "integer"},
                }
            },
        }

        schema: dict[str, ColumnType] = {}
        source._parse_mapping_properties(properties, "", schema)

        assert "user.name" in schema
        assert "user.age" in schema
        assert schema["user.name"] == ColumnType.STRING
        assert schema["user.age"] == ColumnType.INTEGER

    def test_parse_nested_mapping_no_flatten(self) -> None:
        """Test parsing nested mappings without flattening."""
        config = ElasticsearchConfig(index="test", flatten_nested=False)
        source = ElasticsearchDataSource(config)

        properties = {
            "user": {
                "properties": {
                    "name": {"type": "keyword"},
                }
            },
        }

        schema: dict[str, ColumnType] = {}
        source._parse_mapping_properties(properties, "", schema)

        assert "user" in schema
        assert schema["user"] == ColumnType.STRUCT

    @pytest.mark.asyncio
    async def test_sample_async(self) -> None:
        """Test async sampling creates new config."""
        config = ElasticsearchConfig(
            index="test-index",
            max_documents=10000,
        )
        source = ElasticsearchDataSource(config)

        sampled = await source.sample_async(100)

        assert sampled.config.max_documents == 100
        assert "sample" in sampled.name


class TestElasticsearchExceptions:
    """Tests for Elasticsearch exceptions."""

    def test_elasticsearch_error(self) -> None:
        """Test ElasticsearchError."""
        error = ElasticsearchError("Test error")
        assert str(error) == "Test error"

    def test_elasticsearch_connection_error(self) -> None:
        """Test ElasticsearchConnectionError."""
        error = ElasticsearchConnectionError(
            "Connection refused",
            hosts=["http://localhost:9200"],
        )
        assert error.hosts == ["http://localhost:9200"]
        assert "Elasticsearch connection failed" in str(error)
        assert "Connection refused" in str(error)
