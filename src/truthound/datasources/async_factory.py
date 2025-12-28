"""Async factory functions for creating async data sources.

This module provides factory functions for creating async-capable data sources,
with auto-detection and convenient helper functions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TYPE_CHECKING

from truthound.datasources._async_protocols import AsyncDataSourceProtocol
from truthound.datasources.adapters import SyncToAsyncAdapter, adapt_to_async
from truthound.datasources.base import DataSourceError

if TYPE_CHECKING:
    import polars as pl
    from truthound.datasources.nosql.mongodb import MongoDBDataSource
    from truthound.datasources.nosql.elasticsearch import ElasticsearchDataSource
    from truthound.datasources.streaming.kafka import KafkaDataSource


# =============================================================================
# Connection String Detection
# =============================================================================


def _is_mongodb_uri(uri: str) -> bool:
    """Check if string is a MongoDB connection URI."""
    return uri.startswith(("mongodb://", "mongodb+srv://"))


def _is_elasticsearch_uri(uri: str) -> bool:
    """Check if string looks like an Elasticsearch connection."""
    return (
        "elasticsearch" in uri.lower()
        or uri.startswith(("http://localhost:9200", "https://"))
        and ":9200" in uri
    )


def _is_kafka_uri(uri: str) -> bool:
    """Check if string looks like a Kafka connection."""
    return (
        uri.startswith("kafka://")
        or ":9092" in uri
        or "confluent.cloud" in uri
    )


# =============================================================================
# Async Factory Function
# =============================================================================


async def get_async_datasource(
    data: Any,
    name: str | None = None,
    **kwargs: Any,
) -> AsyncDataSourceProtocol | SyncToAsyncAdapter:
    """Create an async-capable data source.

    This function auto-detects the type of input data and returns
    the appropriate async data source implementation. Sync data sources
    are wrapped in SyncToAsyncAdapter.

    Supported input types:
    - MongoDB connection string (mongodb://, mongodb+srv://)
    - Elasticsearch URL
    - Kafka connection string (kafka://)
    - All sync data sources (wrapped in adapter)

    Args:
        data: Input data in any supported format.
        name: Optional name for the data source.
        **kwargs: Additional arguments passed to the data source constructor.

    Returns:
        Async-compatible data source.

    Raises:
        DataSourceError: If the input type is not supported.

    Example:
        >>> # MongoDB
        >>> source = await get_async_datasource(
        ...     "mongodb://localhost:27017",
        ...     database="mydb",
        ...     collection="users",
        ... )
        >>>
        >>> async with source:
        ...     schema = await source.get_schema_async()
        >>>
        >>> # Existing sync source
        >>> polars_source = PolarsDataSource(df)
        >>> async_source = await get_async_datasource(polars_source)
    """
    # Check for native async sources first

    # MongoDB connection string
    if isinstance(data, str) and _is_mongodb_uri(data):
        database = kwargs.pop("database", None)
        collection = kwargs.pop("collection", None)

        if not database or not collection:
            raise DataSourceError(
                "MongoDB connection requires 'database' and 'collection' parameters"
            )

        return await from_mongodb(
            uri=data,
            database=database,
            collection=collection,
            name=name,
            **kwargs,
        )

    # Elasticsearch connection
    if isinstance(data, str) and _is_elasticsearch_uri(data):
        index = kwargs.pop("index", None)
        if not index:
            raise DataSourceError(
                "Elasticsearch connection requires 'index' parameter"
            )

        return await from_elasticsearch(
            hosts=[data],
            index=index,
            name=name,
            **kwargs,
        )

    # Kafka connection string
    if isinstance(data, str) and _is_kafka_uri(data):
        return await from_kafka(
            connection_string=data,
            name=name,
            **kwargs,
        )

    # Check if already async
    if isinstance(data, AsyncDataSourceProtocol):
        return data

    # Check if it's a sync adapter
    if isinstance(data, SyncToAsyncAdapter):
        return data

    # Fallback: create sync source and wrap
    from truthound.datasources.factory import get_datasource

    sync_source = get_datasource(data, name=name, **kwargs)
    return adapt_to_async(sync_source)


# =============================================================================
# Async Convenience Functions
# =============================================================================


async def from_mongodb(
    uri: str | None = None,
    host: str = "localhost",
    port: int = 27017,
    database: str = "",
    collection: str = "",
    name: str | None = None,
    **kwargs: Any,
) -> "MongoDBDataSource":
    """Create a MongoDB async data source.

    Args:
        uri: Optional MongoDB connection URI.
        host: MongoDB host (if not using URI).
        port: MongoDB port (if not using URI).
        database: Database name.
        collection: Collection name.
        name: Optional data source name.
        **kwargs: Additional configuration options.

    Returns:
        MongoDBDataSource instance.

    Example:
        >>> # From URI
        >>> source = await from_mongodb(
        ...     uri="mongodb://localhost:27017",
        ...     database="mydb",
        ...     collection="users",
        ... )
        >>>
        >>> # From parameters
        >>> source = await from_mongodb(
        ...     host="localhost",
        ...     database="mydb",
        ...     collection="users",
        ... )
        >>>
        >>> async with source:
        ...     lf = await source.to_polars_lazyframe_async()
    """
    from truthound.datasources.nosql.mongodb import (
        MongoDBDataSource,
        MongoDBConfig,
    )

    if uri:
        source = MongoDBDataSource.from_connection_string(
            uri=uri,
            database=database,
            collection=collection,
            name=name,
            **kwargs,
        )
    else:
        config = MongoDBConfig(
            host=host,
            port=port,
            database=database,
            collection=collection,
            name=name,
            **kwargs,
        )
        source = MongoDBDataSource(config)

    return source


async def from_elasticsearch(
    hosts: list[str] | None = None,
    cloud_id: str | None = None,
    api_key: str | tuple[str, str] | None = None,
    index: str = "",
    name: str | None = None,
    **kwargs: Any,
) -> "ElasticsearchDataSource":
    """Create an Elasticsearch async data source.

    Args:
        hosts: List of Elasticsearch hosts.
        cloud_id: Elastic Cloud deployment ID.
        api_key: API key for authentication.
        index: Index name or pattern.
        name: Optional data source name.
        **kwargs: Additional configuration options.

    Returns:
        ElasticsearchDataSource instance.

    Example:
        >>> # From hosts
        >>> source = await from_elasticsearch(
        ...     hosts=["http://localhost:9200"],
        ...     index="my-index",
        ... )
        >>>
        >>> # From Elastic Cloud
        >>> source = await from_elasticsearch(
        ...     cloud_id="my-deployment:...",
        ...     api_key="my-api-key",
        ...     index="logs-*",
        ... )
        >>>
        >>> async with source:
        ...     schema = await source.get_schema_async()
    """
    from truthound.datasources.nosql.elasticsearch import (
        ElasticsearchDataSource,
        ElasticsearchConfig,
    )

    if cloud_id:
        source = ElasticsearchDataSource.from_cloud(
            cloud_id=cloud_id,
            api_key=api_key or "",
            index=index,
            name=name,
            **kwargs,
        )
    elif hosts:
        source = ElasticsearchDataSource.from_hosts(
            hosts=hosts,
            index=index,
            name=name,
            **kwargs,
        )
    else:
        config = ElasticsearchConfig(
            index=index,
            name=name,
            **kwargs,
        )
        source = ElasticsearchDataSource(config)

    return source


async def from_kafka(
    connection_string: str | None = None,
    bootstrap_servers: str = "localhost:9092",
    topic: str = "",
    name: str | None = None,
    **kwargs: Any,
) -> "KafkaDataSource":
    """Create a Kafka async data source.

    Note: This is a BOUNDED data source that consumes up to max_messages.
    For real-time streaming validation, see truthound.realtime.streaming.

    Args:
        connection_string: Kafka connection string (kafka://host:port/topic).
        bootstrap_servers: Kafka broker addresses.
        topic: Topic name (if not in connection string).
        name: Optional data source name.
        **kwargs: Additional configuration options.

    Returns:
        KafkaDataSource instance.

    Example:
        >>> # From connection string
        >>> source = await from_kafka(
        ...     connection_string="kafka://localhost:9092/my-topic",
        ...     max_messages=5000,
        ... )
        >>>
        >>> # From parameters
        >>> source = await from_kafka(
        ...     bootstrap_servers="localhost:9092",
        ...     topic="my-topic",
        ...     max_messages=10000,
        ... )
        >>>
        >>> async with source:
        ...     lf = await source.to_polars_lazyframe_async()
    """
    from truthound.datasources.streaming.kafka import (
        KafkaDataSource,
        KafkaDataSourceConfig,
    )

    if connection_string:
        source = KafkaDataSource.from_connection_string(
            connection_string,
            name=name,
            **kwargs,
        )
    else:
        config = KafkaDataSourceConfig(
            bootstrap_servers=bootstrap_servers,
            topic=topic,
            name=name,
            **kwargs,
        )
        source = KafkaDataSource(config)

    return source


async def from_confluent(
    bootstrap_servers: str,
    topic: str,
    api_key: str,
    api_secret: str,
    name: str | None = None,
    **kwargs: Any,
) -> "KafkaDataSource":
    """Create a Kafka data source for Confluent Cloud.

    Args:
        bootstrap_servers: Confluent Cloud bootstrap servers.
        topic: Topic name.
        api_key: Confluent API key.
        api_secret: Confluent API secret.
        name: Optional data source name.
        **kwargs: Additional configuration options.

    Returns:
        KafkaDataSource configured for Confluent Cloud.

    Example:
        >>> source = await from_confluent(
        ...     bootstrap_servers="pkc-xxxxx.us-east-1.aws.confluent.cloud:9092",
        ...     topic="my-topic",
        ...     api_key="ABCDEFGHIJKLMNOP",
        ...     api_secret="secret123",
        ... )
    """
    from truthound.datasources.streaming.kafka import KafkaDataSource

    return KafkaDataSource.from_confluent(
        bootstrap_servers=bootstrap_servers,
        topic=topic,
        api_key=api_key,
        api_secret=api_secret,
        name=name,
        **kwargs,
    )


async def from_atlas(
    cluster_url: str,
    database: str,
    collection: str,
    username: str,
    password: str,
    name: str | None = None,
    **kwargs: Any,
) -> "MongoDBDataSource":
    """Create a MongoDB data source for MongoDB Atlas.

    Args:
        cluster_url: Atlas cluster URL (e.g., "cluster0.xxxxx.mongodb.net").
        database: Database name.
        collection: Collection name.
        username: Atlas username.
        password: Atlas password.
        name: Optional data source name.
        **kwargs: Additional configuration options.

    Returns:
        MongoDBDataSource configured for Atlas.

    Example:
        >>> source = await from_atlas(
        ...     cluster_url="cluster0.xxxxx.mongodb.net",
        ...     database="mydb",
        ...     collection="users",
        ...     username="user",
        ...     password="pass",
        ... )
    """
    from truthound.datasources.nosql.mongodb import MongoDBDataSource

    return MongoDBDataSource.from_atlas(
        cluster_url=cluster_url,
        database=database,
        collection=collection,
        username=username,
        password=password,
        name=name,
        **kwargs,
    )


# =============================================================================
# Async Source Detection
# =============================================================================


def detect_async_datasource_type(data: Any) -> str:
    """Detect the type of async data source for given data.

    Args:
        data: Input data.

    Returns:
        Data source type string.

    Example:
        >>> detect_async_datasource_type("mongodb://localhost:27017")
        'mongodb'
    """
    if isinstance(data, str):
        if _is_mongodb_uri(data):
            return "mongodb"
        if _is_elasticsearch_uri(data):
            return "elasticsearch"
        if _is_kafka_uri(data):
            return "kafka"

    if isinstance(data, AsyncDataSourceProtocol):
        return data.source_type

    if isinstance(data, SyncToAsyncAdapter):
        return f"async_{data._wrapped.source_type}"

    # Fallback to sync detection
    from truthound.datasources.factory import detect_datasource_type

    sync_type = detect_datasource_type(data)
    return f"async_{sync_type}"


def is_native_async_source(data: Any) -> bool:
    """Check if data represents a native async data source.

    Args:
        data: Input data.

    Returns:
        True if the data source is natively async.

    Example:
        >>> is_native_async_source("mongodb://localhost:27017")
        True
        >>> is_native_async_source(polars_df)
        False
    """
    if isinstance(data, AsyncDataSourceProtocol):
        return True

    if isinstance(data, str):
        return (
            _is_mongodb_uri(data)
            or _is_elasticsearch_uri(data)
            or _is_kafka_uri(data)
        )

    return False
