"""Tests for MongoDB data source."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from truthound.datasources._protocols import ColumnType, DataSourceCapability
from truthound.datasources.nosql.mongodb import (
    MongoDBDataSource,
    MongoDBConfig,
    MongoDBError,
    MongoDBConnectionError,
)
from truthound.datasources.nosql.base import DocumentSchemaInferrer


class TestMongoDBConfig:
    """Tests for MongoDBConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = MongoDBConfig(database="testdb", collection="testcol")

        assert config.host == "localhost"
        assert config.port == 27017
        assert config.database == "testdb"
        assert config.collection == "testcol"
        assert config.tls is False
        assert config.read_preference == "primary"

    def test_get_connection_string_basic(self) -> None:
        """Test basic connection string generation."""
        config = MongoDBConfig(
            host="localhost",
            port=27017,
            database="testdb",
            collection="testcol",
        )

        uri = config.get_connection_string()

        assert "mongodb://localhost:27017" in uri
        assert "readPreference=primary" in uri

    def test_get_connection_string_with_auth(self) -> None:
        """Test connection string with authentication."""
        config = MongoDBConfig(
            host="localhost",
            username="user",
            password="pass",
            database="testdb",
            collection="testcol",
        )

        uri = config.get_connection_string()

        assert "mongodb://user:pass@localhost" in uri
        assert "authSource=admin" in uri

    def test_get_connection_string_explicit(self) -> None:
        """Test explicit connection string takes precedence."""
        config = MongoDBConfig(
            connection_string="mongodb://custom:27017",
            database="testdb",
            collection="testcol",
        )

        uri = config.get_connection_string()
        assert uri == "mongodb://custom:27017"

    def test_get_connection_string_with_replica_set(self) -> None:
        """Test connection string with replica set."""
        config = MongoDBConfig(
            host="localhost",
            replica_set="rs0",
            database="testdb",
            collection="testcol",
        )

        uri = config.get_connection_string()
        assert "replicaSet=rs0" in uri


class TestMongoDBDataSource:
    """Tests for MongoDBDataSource."""

    def test_requires_database(self) -> None:
        """Test that database is required."""
        with pytest.raises(MongoDBError, match="Database name is required"):
            MongoDBDataSource(MongoDBConfig(collection="col"))

    def test_requires_collection(self) -> None:
        """Test that collection is required."""
        with pytest.raises(MongoDBError, match="Collection name is required"):
            MongoDBDataSource(MongoDBConfig(database="db"))

    def test_creation(self) -> None:
        """Test source creation."""
        config = MongoDBConfig(database="testdb", collection="testcol")
        source = MongoDBDataSource(config)

        assert source.source_type == "mongodb"
        assert source.database == "testdb"
        assert source.collection_name == "testcol"

    def test_name_property(self) -> None:
        """Test name property."""
        config = MongoDBConfig(database="mydb", collection="mycol")
        source = MongoDBDataSource(config)

        assert source.name == "mongodb://mydb.mycol"

    def test_name_property_custom(self) -> None:
        """Test custom name."""
        config = MongoDBConfig(
            name="custom_name",
            database="db",
            collection="col",
        )
        source = MongoDBDataSource(config)

        assert source.name == "custom_name"

    def test_capabilities(self) -> None:
        """Test capabilities include expected values."""
        config = MongoDBConfig(database="db", collection="col")
        source = MongoDBDataSource(config)
        caps = source.capabilities

        assert DataSourceCapability.SCHEMA_INFERENCE in caps
        assert DataSourceCapability.SAMPLING in caps
        assert DataSourceCapability.STREAMING in caps
        assert DataSourceCapability.ROW_COUNT in caps

    def test_from_connection_string(self) -> None:
        """Test factory method from connection string."""
        source = MongoDBDataSource.from_connection_string(
            uri="mongodb://localhost:27017",
            database="testdb",
            collection="testcol",
        )

        assert source.database == "testdb"
        assert source.collection_name == "testcol"
        assert "localhost" in source.config.get_connection_string()

    def test_from_atlas(self) -> None:
        """Test factory method for Atlas."""
        source = MongoDBDataSource.from_atlas(
            cluster_url="cluster0.xxxxx.mongodb.net",
            database="testdb",
            collection="testcol",
            username="user",
            password="pass",
        )

        assert source.database == "testdb"
        assert source.config.tls is True
        assert "mongodb+srv://" in source.config.get_connection_string()

    @pytest.mark.asyncio
    async def test_normalize_document(self) -> None:
        """Test document normalization."""
        config = MongoDBConfig(database="db", collection="col")
        source = MongoDBDataSource(config)

        doc = {
            "_id": MagicMock(__class__=type("ObjectId", (), {"__name__": "ObjectId"})),
            "name": "test",
            "nested": {"key": "value"},
        }
        doc["_id"].__str__ = lambda self: "507f1f77bcf86cd799439011"

        normalized = source._normalize_document(doc)

        assert isinstance(normalized["_id"], str)
        assert normalized["name"] == "test"
        assert normalized["nested"]["key"] == "value"

    @pytest.mark.asyncio
    async def test_sample_async(self) -> None:
        """Test async sampling creates new config."""
        config = MongoDBConfig(
            database="db",
            collection="col",
            max_documents=10000,
        )
        source = MongoDBDataSource(config)

        sampled = await source.sample_async(100)

        assert sampled.config.max_documents == 100
        assert "sample" in sampled.name


class TestDocumentSchemaInferrer:
    """Tests for DocumentSchemaInferrer."""

    def test_infer_from_simple_documents(self) -> None:
        """Test schema inference from simple documents."""
        inferrer = DocumentSchemaInferrer()
        docs = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
        ]

        schema = inferrer.infer_from_documents(docs)

        assert schema["name"] == ColumnType.STRING
        assert schema["age"] == ColumnType.INTEGER

    def test_infer_with_null_values(self) -> None:
        """Test schema inference handles null values."""
        inferrer = DocumentSchemaInferrer()
        docs = [
            {"name": "Alice", "age": 30},
            {"name": None, "age": 25},
        ]

        schema = inferrer.infer_from_documents(docs)

        # Should prefer non-null type
        assert schema["name"] == ColumnType.STRING

    def test_infer_nested_documents(self) -> None:
        """Test schema inference with nested documents."""
        inferrer = DocumentSchemaInferrer(flatten_nested=False)
        docs = [
            {"name": "Alice", "address": {"city": "NYC"}},
        ]

        schema = inferrer.infer_from_documents(docs)

        assert schema["name"] == ColumnType.STRING
        assert schema["address"] == ColumnType.STRUCT

    def test_infer_flattened_nested(self) -> None:
        """Test schema inference with flattening."""
        inferrer = DocumentSchemaInferrer(flatten_nested=True)
        docs = [
            {"name": "Alice", "address": {"city": "NYC", "zip": 10001}},
        ]

        schema = inferrer.infer_from_documents(docs)

        assert "address.city" in schema
        assert "address.zip" in schema
        assert schema["address.city"] == ColumnType.STRING
        assert schema["address.zip"] == ColumnType.INTEGER

    def test_infer_datetime(self) -> None:
        """Test datetime type inference."""
        inferrer = DocumentSchemaInferrer()
        docs = [
            {"created": datetime.now()},
        ]

        schema = inferrer.infer_from_documents(docs)
        assert schema["created"] == ColumnType.DATETIME

    def test_infer_boolean(self) -> None:
        """Test boolean type inference."""
        inferrer = DocumentSchemaInferrer()
        docs = [
            {"active": True},
            {"active": False},
        ]

        schema = inferrer.infer_from_documents(docs)
        assert schema["active"] == ColumnType.BOOLEAN

    def test_infer_list(self) -> None:
        """Test list type inference."""
        inferrer = DocumentSchemaInferrer()
        docs = [
            {"tags": ["a", "b", "c"]},
        ]

        schema = inferrer.infer_from_documents(docs)
        assert schema["tags"] == ColumnType.LIST

    def test_infer_mixed_types(self) -> None:
        """Test mixed type inference prefers non-null."""
        inferrer = DocumentSchemaInferrer()
        docs = [
            {"value": 1},
            {"value": 1.5},
        ]

        schema = inferrer.infer_from_documents(docs)
        # Integer + Float = Float
        assert schema["value"] == ColumnType.FLOAT

    def test_empty_documents_raises(self) -> None:
        """Test empty document list raises error."""
        inferrer = DocumentSchemaInferrer()

        with pytest.raises(Exception):
            inferrer.infer_from_documents([])

    def test_flatten_document(self) -> None:
        """Test document flattening."""
        inferrer = DocumentSchemaInferrer()
        doc = {
            "name": "Alice",
            "address": {
                "city": "NYC",
                "location": {
                    "lat": 40.7,
                    "lng": -74.0,
                },
            },
        }

        flat = inferrer.flatten_document(doc)

        assert flat["name"] == "Alice"
        assert flat["address.city"] == "NYC"
        assert flat["address.location.lat"] == 40.7
        assert flat["address.location.lng"] == -74.0


class TestMongoDBExceptions:
    """Tests for MongoDB exceptions."""

    def test_mongodb_error(self) -> None:
        """Test MongoDBError."""
        error = MongoDBError("Test error")
        assert str(error) == "Test error"

    def test_mongodb_connection_error(self) -> None:
        """Test MongoDBConnectionError."""
        error = MongoDBConnectionError("Connection refused", host="localhost")
        assert error.host == "localhost"
        assert "MongoDB connection failed" in str(error)
        assert "Connection refused" in str(error)
