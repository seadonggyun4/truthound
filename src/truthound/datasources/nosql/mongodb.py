"""MongoDB data source implementation.

This module provides async MongoDB data source with Motor driver support.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable
from urllib.parse import urlparse

from truthound.datasources._protocols import ColumnType, DataSourceCapability
from truthound.datasources.async_base import (
    AsyncConnectionPoolError,
    AsyncTimeoutError,
)
from truthound.datasources.nosql.base import (
    BaseNoSQLDataSource,
    NoSQLDataSourceConfig,
    NoSQLDataSourceError,
)

if TYPE_CHECKING:
    import polars as pl


# =============================================================================
# Exceptions
# =============================================================================


class MongoDBError(NoSQLDataSourceError):
    """MongoDB-specific error."""

    pass


class MongoDBConnectionError(MongoDBError):
    """MongoDB connection error."""

    def __init__(self, message: str, host: str | None = None) -> None:
        self.host = host
        super().__init__(f"MongoDB connection failed: {message}")


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class MongoDBConfig(NoSQLDataSourceConfig):
    """Configuration for MongoDB data source.

    Supports connection via individual parameters or connection string.

    Attributes:
        host: MongoDB host.
        port: MongoDB port.
        database: Database name.
        collection: Collection name.
        username: Optional username.
        password: Optional password.
        auth_source: Authentication database.
        replica_set: Optional replica set name.
        tls: Whether to use TLS.
        tls_ca_file: Path to CA certificate file.
        tls_cert_key_file: Path to client certificate file.
        read_preference: Read preference (primary, primaryPreferred, etc.).
        write_concern: Write concern (0, 1, majority).
        app_name: Application name for connection.
    """

    host: str = "localhost"
    port: int = 27017
    database: str = ""
    collection: str = ""
    username: str | None = None
    password: str | None = None
    auth_source: str = "admin"
    replica_set: str | None = None
    tls: bool = False
    tls_ca_file: str | None = None
    tls_cert_key_file: str | None = None
    read_preference: str = "primary"
    write_concern: str = "1"
    app_name: str = "truthound"

    # Connection string (alternative to individual params)
    connection_string: str | None = None

    def get_connection_string(self) -> str:
        """Build MongoDB connection string.

        Returns:
            MongoDB URI string.
        """
        if self.connection_string:
            return self.connection_string

        # Build URI from parameters
        if self.username and self.password:
            auth = f"{self.username}:{self.password}@"
        else:
            auth = ""

        uri = f"mongodb://{auth}{self.host}:{self.port}"

        # Add options
        options = []
        if self.replica_set:
            options.append(f"replicaSet={self.replica_set}")
        if self.tls:
            options.append("tls=true")
        if self.auth_source and self.username:
            options.append(f"authSource={self.auth_source}")
        options.append(f"readPreference={self.read_preference}")
        options.append(f"appName={self.app_name}")

        if options:
            uri += "/?" + "&".join(options)

        return uri


# =============================================================================
# MongoDB Data Source
# =============================================================================


class MongoDBDataSource(BaseNoSQLDataSource):
    """Async MongoDB data source using Motor driver.

    Provides async access to MongoDB collections with automatic schema
    inference and Polars LazyFrame conversion.

    Example:
        >>> # From connection string
        >>> source = MongoDBDataSource.from_connection_string(
        ...     uri="mongodb://localhost:27017",
        ...     database="mydb",
        ...     collection="users",
        ... )
        >>>
        >>> async with source:
        ...     schema = await source.get_schema_async()
        ...     lf = await source.to_polars_lazyframe_async()
        ...     print(lf.collect())

        >>> # From config
        >>> config = MongoDBConfig(
        ...     host="localhost",
        ...     database="mydb",
        ...     collection="users",
        ... )
        >>> source = MongoDBDataSource(config)
    """

    source_type = "mongodb"

    def __init__(self, config: MongoDBConfig) -> None:
        """Initialize MongoDB data source.

        Args:
            config: MongoDB configuration.

        Raises:
            MongoDBError: If database or collection not specified.
        """
        if not config.database:
            raise MongoDBError("Database name is required")
        if not config.collection:
            raise MongoDBError("Collection name is required")

        super().__init__(config)
        self._client: Any = None
        self._collection: Any = None

    @property
    def config(self) -> MongoDBConfig:
        """Get typed configuration."""
        return self._config  # type: ignore

    @property
    def name(self) -> str:
        """Get data source name."""
        if self._config.name:
            return self._config.name
        return f"mongodb://{self.config.database}.{self.config.collection}"

    @property
    def database(self) -> str:
        """Get database name."""
        return self.config.database

    @property
    def collection_name(self) -> str:
        """Get collection name."""
        return self.config.collection

    @property
    def capabilities(self) -> set[DataSourceCapability]:
        """Get data source capabilities."""
        return {
            DataSourceCapability.SCHEMA_INFERENCE,
            DataSourceCapability.SAMPLING,
            DataSourceCapability.STREAMING,
            DataSourceCapability.ROW_COUNT,
        }

    # -------------------------------------------------------------------------
    # Factory Methods
    # -------------------------------------------------------------------------

    @classmethod
    def from_connection_string(
        cls,
        uri: str,
        database: str,
        collection: str,
        **kwargs: Any,
    ) -> "MongoDBDataSource":
        """Create data source from connection string.

        Args:
            uri: MongoDB connection URI.
            database: Database name.
            collection: Collection name.
            **kwargs: Additional configuration options.

        Returns:
            MongoDBDataSource instance.

        Example:
            >>> source = MongoDBDataSource.from_connection_string(
            ...     uri="mongodb://user:pass@host:27017",
            ...     database="mydb",
            ...     collection="users",
            ...     max_documents=10000,
            ... )
        """
        # Parse URI to extract host info
        parsed = urlparse(uri)
        host = parsed.hostname or "localhost"
        port = parsed.port or 27017

        config = MongoDBConfig(
            connection_string=uri,
            host=host,
            port=port,
            database=database,
            collection=collection,
            **kwargs,
        )
        return cls(config)

    @classmethod
    def from_atlas(
        cls,
        cluster_url: str,
        database: str,
        collection: str,
        username: str,
        password: str,
        **kwargs: Any,
    ) -> "MongoDBDataSource":
        """Create data source for MongoDB Atlas.

        Args:
            cluster_url: Atlas cluster URL (e.g., "cluster0.xxxxx.mongodb.net").
            database: Database name.
            collection: Collection name.
            username: Atlas username.
            password: Atlas password.
            **kwargs: Additional configuration options.

        Returns:
            MongoDBDataSource instance.

        Example:
            >>> source = MongoDBDataSource.from_atlas(
            ...     cluster_url="cluster0.xxxxx.mongodb.net",
            ...     database="mydb",
            ...     collection="users",
            ...     username="user",
            ...     password="pass",
            ... )
        """
        uri = (
            f"mongodb+srv://{username}:{password}@{cluster_url}"
            f"/?retryWrites=true&w=majority"
        )
        config = MongoDBConfig(
            connection_string=uri,
            database=database,
            collection=collection,
            tls=True,
            **kwargs,
        )
        return cls(config)

    # -------------------------------------------------------------------------
    # Connection Management
    # -------------------------------------------------------------------------

    async def _create_connection_factory(self) -> Callable:
        """Create a connection factory for the pool."""

        async def factory():
            try:
                from motor.motor_asyncio import AsyncIOMotorClient
            except ImportError:
                raise ImportError(
                    "Motor is required for MongoDB support. "
                    "Install it with: pip install motor"
                )

            client = AsyncIOMotorClient(
                self.config.get_connection_string(),
                maxPoolSize=self.config.pool_size,
                serverSelectionTimeoutMS=int(self.config.connection_timeout * 1000),
            )
            # Test connection
            await client.admin.command("ping")
            return client

        return factory

    async def connect_async(self) -> None:
        """Establish connection to MongoDB."""
        if self._is_connected:
            return

        async with self._lock:
            if self._is_connected:
                return

            try:
                from motor.motor_asyncio import AsyncIOMotorClient
            except ImportError:
                raise ImportError(
                    "Motor is required for MongoDB support. "
                    "Install it with: pip install motor"
                )

            try:
                self._client = AsyncIOMotorClient(
                    self.config.get_connection_string(),
                    maxPoolSize=self.config.pool_size,
                    serverSelectionTimeoutMS=int(
                        self.config.connection_timeout * 1000
                    ),
                )

                # Test connection
                await self._client.admin.command("ping")

                # Get database and collection
                db = self._client[self.config.database]
                self._collection = db[self.config.collection]

                # Pre-fetch schema
                self._cached_schema = await self.get_schema_async()

                self._is_connected = True

            except Exception as e:
                raise MongoDBConnectionError(
                    str(e), host=self.config.host
                )

    async def disconnect_async(self) -> None:
        """Close MongoDB connection."""
        if not self._is_connected:
            return

        async with self._lock:
            if not self._is_connected:
                return

            if self._client:
                self._client.close()
                self._client = None
                self._collection = None

            self._is_connected = False

    async def validate_connection_async(self) -> bool:
        """Validate MongoDB connection.

        Returns:
            True if connection is healthy.
        """
        try:
            if not self._is_connected:
                await self.connect_async()

            await self._client.admin.command("ping")
            return True
        except Exception:
            return False

    # -------------------------------------------------------------------------
    # Document Operations
    # -------------------------------------------------------------------------

    async def _fetch_sample_documents(
        self, n: int
    ) -> list[dict[str, Any]]:
        """Fetch sample documents for schema inference.

        Uses $sample aggregation for random sampling.

        Args:
            n: Number of documents to sample.

        Returns:
            List of sample documents.
        """
        if not self._is_connected:
            await self.connect_async()

        pipeline = [{"$sample": {"size": n}}]
        cursor = self._collection.aggregate(pipeline)
        documents = await cursor.to_list(length=n)

        # Convert ObjectId to string for JSON compatibility
        return [self._normalize_document(doc) for doc in documents]

    async def _fetch_documents(
        self,
        filter: dict[str, Any] | None = None,
        limit: int | None = None,
        skip: int | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch documents from collection.

        Args:
            filter: MongoDB query filter.
            limit: Maximum documents to return.
            skip: Number of documents to skip.

        Returns:
            List of documents.
        """
        if not self._is_connected:
            await self.connect_async()

        cursor = self._collection.find(filter or {})

        if skip:
            cursor = cursor.skip(skip)
        if limit:
            cursor = cursor.limit(limit)

        documents = await cursor.to_list(length=limit or self.config.max_documents)
        return [self._normalize_document(doc) for doc in documents]

    async def _count_documents(
        self, filter: dict[str, Any] | None = None
    ) -> int:
        """Count documents in collection.

        Args:
            filter: Optional filter criteria.

        Returns:
            Document count.
        """
        if not self._is_connected:
            await self.connect_async()

        return await self._collection.count_documents(filter or {})

    def _normalize_document(self, doc: dict[str, Any]) -> dict[str, Any]:
        """Normalize document for Polars compatibility.

        Converts ObjectId, Binary, etc. to Python native types.

        Args:
            doc: MongoDB document.

        Returns:
            Normalized document.
        """
        normalized = {}

        for key, value in doc.items():
            # Skip internal MongoDB fields if needed
            if key == "_id":
                normalized[key] = str(value)
            elif hasattr(value, "__class__"):
                type_name = value.__class__.__name__
                if type_name == "ObjectId":
                    normalized[key] = str(value)
                elif type_name == "Binary":
                    normalized[key] = bytes(value)
                elif type_name == "Decimal128":
                    normalized[key] = float(value.to_decimal())
                elif type_name == "datetime":
                    normalized[key] = value
                elif isinstance(value, dict):
                    normalized[key] = self._normalize_document(value)
                elif isinstance(value, list):
                    normalized[key] = [
                        self._normalize_document(v) if isinstance(v, dict) else v
                        for v in value
                    ]
                else:
                    normalized[key] = value
            else:
                normalized[key] = value

        return normalized

    # -------------------------------------------------------------------------
    # Aggregation
    # -------------------------------------------------------------------------

    async def aggregate_async(
        self,
        pipeline: list[dict[str, Any]],
        allow_disk_use: bool = True,
    ) -> list[dict[str, Any]]:
        """Execute aggregation pipeline.

        Args:
            pipeline: MongoDB aggregation pipeline stages.
            allow_disk_use: Allow disk use for large operations.

        Returns:
            Aggregation results.

        Example:
            >>> pipeline = [
            ...     {"$match": {"status": "active"}},
            ...     {"$group": {"_id": "$category", "count": {"$sum": 1}}},
            ... ]
            >>> results = await source.aggregate_async(pipeline)
        """
        if not self._is_connected:
            await self.connect_async()

        cursor = self._collection.aggregate(
            pipeline,
            allowDiskUse=allow_disk_use,
        )
        documents = await cursor.to_list(length=self.config.max_documents)
        return [self._normalize_document(doc) for doc in documents]

    # -------------------------------------------------------------------------
    # Sampling
    # -------------------------------------------------------------------------

    async def sample_async(
        self, n: int = 1000, seed: int | None = None
    ) -> "MongoDBDataSource":
        """Create a sampled data source.

        Note: MongoDB's $sample doesn't support seeds, so results
        may vary between calls.

        Args:
            n: Number of documents to sample.
            seed: Ignored (MongoDB $sample doesn't support seeds).

        Returns:
            New MongoDBDataSource with sampled configuration.
        """
        # Create new config with reduced max_documents
        config = MongoDBConfig(
            connection_string=self.config.connection_string,
            host=self.config.host,
            port=self.config.port,
            database=self.config.database,
            collection=self.config.collection,
            username=self.config.username,
            password=self.config.password,
            name=f"{self.name}_sample",
            max_documents=n,
            schema_sample_size=min(n, self.config.schema_sample_size),
        )
        return MongoDBDataSource(config)

    # -------------------------------------------------------------------------
    # Additional Query Methods
    # -------------------------------------------------------------------------

    async def find_one_async(
        self,
        filter: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Find a single document.

        Args:
            filter: Query filter.

        Returns:
            Document or None if not found.
        """
        if not self._is_connected:
            await self.connect_async()

        doc = await self._collection.find_one(filter or {})
        return self._normalize_document(doc) if doc else None

    async def distinct_async(
        self,
        field: str,
        filter: dict[str, Any] | None = None,
    ) -> list[Any]:
        """Get distinct values for a field.

        Args:
            field: Field name.
            filter: Optional filter.

        Returns:
            List of distinct values.
        """
        if not self._is_connected:
            await self.connect_async()

        return await self._collection.distinct(field, filter or {})

    async def get_indexes_async(self) -> list[dict[str, Any]]:
        """Get collection indexes.

        Returns:
            List of index definitions.
        """
        if not self._is_connected:
            await self.connect_async()

        cursor = self._collection.list_indexes()
        return await cursor.to_list(length=100)
