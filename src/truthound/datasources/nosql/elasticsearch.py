"""Elasticsearch data source implementation.

This module provides async Elasticsearch data source with official
elasticsearch-py async client support.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from truthound.datasources._protocols import ColumnType, DataSourceCapability
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


class ElasticsearchError(NoSQLDataSourceError):
    """Elasticsearch-specific error."""

    pass


class ElasticsearchConnectionError(ElasticsearchError):
    """Elasticsearch connection error."""

    def __init__(self, message: str, hosts: list[str] | None = None) -> None:
        self.hosts = hosts
        super().__init__(f"Elasticsearch connection failed: {message}")


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class ElasticsearchConfig(NoSQLDataSourceConfig):
    """Configuration for Elasticsearch data source.

    Supports connection via hosts list, cloud ID, or custom configuration.

    Attributes:
        hosts: List of Elasticsearch hosts (e.g., ["http://localhost:9200"]).
        index: Index name or pattern (supports wildcards like "logs-*").
        cloud_id: Elastic Cloud ID for cloud deployments.
        api_key: API key for authentication (string or tuple).
        username: Basic auth username.
        password: Basic auth password.
        bearer_token: Bearer token for authentication.
        ca_certs: Path to CA certificate file.
        verify_certs: Whether to verify SSL certificates.
        scroll_timeout: Scroll context timeout.
        scroll_size: Number of documents per scroll page.
        request_timeout: Request timeout in seconds.
        track_total_hits: Track exact total hits count.
    """

    hosts: list[str] = field(default_factory=lambda: ["http://localhost:9200"])
    index: str = ""
    cloud_id: str | None = None
    api_key: str | tuple[str, str] | None = None
    username: str | None = None
    password: str | None = None
    bearer_token: str | None = None
    ca_certs: str | None = None
    verify_certs: bool = True
    scroll_timeout: str = "5m"
    scroll_size: int = 1000
    request_timeout: int = 30
    track_total_hits: bool = True


# =============================================================================
# Elasticsearch Data Source
# =============================================================================


class ElasticsearchDataSource(BaseNoSQLDataSource):
    """Async Elasticsearch data source.

    Provides async access to Elasticsearch indices with automatic schema
    inference from index mappings and Polars LazyFrame conversion.

    Example:
        >>> # Basic usage
        >>> source = ElasticsearchDataSource(ElasticsearchConfig(
        ...     hosts=["http://localhost:9200"],
        ...     index="my-index",
        ... ))
        >>>
        >>> async with source:
        ...     schema = await source.get_schema_async()
        ...     lf = await source.to_polars_lazyframe_async()

        >>> # Elastic Cloud
        >>> source = ElasticsearchDataSource.from_cloud(
        ...     cloud_id="my-deployment:dXMtY2VudHJhbC0x...",
        ...     api_key="your-api-key",
        ...     index="logs-*",
        ... )
    """

    source_type = "elasticsearch"

    # Elasticsearch type to ColumnType mapping
    ES_TYPE_MAPPING = {
        # Core types
        "text": ColumnType.STRING,
        "keyword": ColumnType.STRING,
        "long": ColumnType.INTEGER,
        "integer": ColumnType.INTEGER,
        "short": ColumnType.INTEGER,
        "byte": ColumnType.INTEGER,
        "double": ColumnType.FLOAT,
        "float": ColumnType.FLOAT,
        "half_float": ColumnType.FLOAT,
        "scaled_float": ColumnType.FLOAT,
        "boolean": ColumnType.BOOLEAN,
        "date": ColumnType.DATETIME,
        "binary": ColumnType.BINARY,
        # Numeric
        "unsigned_long": ColumnType.INTEGER,
        # Complex types
        "object": ColumnType.STRUCT,
        "nested": ColumnType.STRUCT,
        "flattened": ColumnType.STRUCT,
        # IP and geo
        "ip": ColumnType.STRING,
        "geo_point": ColumnType.STRUCT,
        "geo_shape": ColumnType.STRUCT,
        # Range types
        "integer_range": ColumnType.STRUCT,
        "float_range": ColumnType.STRUCT,
        "long_range": ColumnType.STRUCT,
        "double_range": ColumnType.STRUCT,
        "date_range": ColumnType.STRUCT,
        "ip_range": ColumnType.STRUCT,
        # Other
        "completion": ColumnType.STRING,
        "search_as_you_type": ColumnType.STRING,
        "alias": ColumnType.UNKNOWN,
        "dense_vector": ColumnType.LIST,
        "sparse_vector": ColumnType.STRUCT,
    }

    def __init__(self, config: ElasticsearchConfig) -> None:
        """Initialize Elasticsearch data source.

        Args:
            config: Elasticsearch configuration.

        Raises:
            ElasticsearchError: If index not specified.
        """
        if not config.index:
            raise ElasticsearchError("Index name is required")

        super().__init__(config)
        self._client: Any = None

    @property
    def config(self) -> ElasticsearchConfig:
        """Get typed configuration."""
        return self._config  # type: ignore

    @property
    def name(self) -> str:
        """Get data source name."""
        if self._config.name:
            return self._config.name
        return f"es://{self.config.index}"

    @property
    def index(self) -> str:
        """Get index name or pattern."""
        return self.config.index

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
    def from_cloud(
        cls,
        cloud_id: str,
        api_key: str | tuple[str, str],
        index: str,
        **kwargs: Any,
    ) -> "ElasticsearchDataSource":
        """Create data source for Elastic Cloud.

        Args:
            cloud_id: Elastic Cloud deployment ID.
            api_key: API key (string or (id, key) tuple).
            index: Index name or pattern.
            **kwargs: Additional configuration options.

        Returns:
            ElasticsearchDataSource instance.

        Example:
            >>> source = ElasticsearchDataSource.from_cloud(
            ...     cloud_id="my-deployment:base64string",
            ...     api_key="my-api-key",
            ...     index="logs-*",
            ... )
        """
        config = ElasticsearchConfig(
            cloud_id=cloud_id,
            api_key=api_key,
            index=index,
            verify_certs=True,
            **kwargs,
        )
        return cls(config)

    @classmethod
    def from_hosts(
        cls,
        hosts: list[str],
        index: str,
        username: str | None = None,
        password: str | None = None,
        **kwargs: Any,
    ) -> "ElasticsearchDataSource":
        """Create data source from host list.

        Args:
            hosts: List of Elasticsearch hosts.
            index: Index name or pattern.
            username: Optional basic auth username.
            password: Optional basic auth password.
            **kwargs: Additional configuration options.

        Returns:
            ElasticsearchDataSource instance.

        Example:
            >>> source = ElasticsearchDataSource.from_hosts(
            ...     hosts=["http://node1:9200", "http://node2:9200"],
            ...     index="my-index",
            ...     username="elastic",
            ...     password="changeme",
            ... )
        """
        config = ElasticsearchConfig(
            hosts=hosts,
            index=index,
            username=username,
            password=password,
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
                from elasticsearch import AsyncElasticsearch
            except ImportError:
                raise ImportError(
                    "elasticsearch[async] is required for Elasticsearch support. "
                    "Install it with: pip install elasticsearch[async]"
                )

            client_kwargs: dict[str, Any] = {
                "request_timeout": self.config.request_timeout,
            }

            if self.config.cloud_id:
                client_kwargs["cloud_id"] = self.config.cloud_id
            else:
                client_kwargs["hosts"] = self.config.hosts

            if self.config.api_key:
                client_kwargs["api_key"] = self.config.api_key
            elif self.config.username and self.config.password:
                client_kwargs["basic_auth"] = (
                    self.config.username,
                    self.config.password,
                )
            elif self.config.bearer_token:
                client_kwargs["bearer_auth"] = self.config.bearer_token

            if self.config.ca_certs:
                client_kwargs["ca_certs"] = self.config.ca_certs
            client_kwargs["verify_certs"] = self.config.verify_certs

            return AsyncElasticsearch(**client_kwargs)

        return factory

    async def connect_async(self) -> None:
        """Establish connection to Elasticsearch."""
        if self._is_connected:
            return

        async with self._lock:
            if self._is_connected:
                return

            try:
                from elasticsearch import AsyncElasticsearch
            except ImportError:
                raise ImportError(
                    "elasticsearch[async] is required for Elasticsearch support. "
                    "Install it with: pip install elasticsearch[async]"
                )

            try:
                client_kwargs: dict[str, Any] = {
                    "request_timeout": self.config.request_timeout,
                }

                if self.config.cloud_id:
                    client_kwargs["cloud_id"] = self.config.cloud_id
                else:
                    client_kwargs["hosts"] = self.config.hosts

                if self.config.api_key:
                    client_kwargs["api_key"] = self.config.api_key
                elif self.config.username and self.config.password:
                    client_kwargs["basic_auth"] = (
                        self.config.username,
                        self.config.password,
                    )
                elif self.config.bearer_token:
                    client_kwargs["bearer_auth"] = self.config.bearer_token

                if self.config.ca_certs:
                    client_kwargs["ca_certs"] = self.config.ca_certs
                client_kwargs["verify_certs"] = self.config.verify_certs

                self._client = AsyncElasticsearch(**client_kwargs)

                # Test connection
                await self._client.info()

                # Pre-fetch schema
                self._cached_schema = await self.get_schema_async()

                self._is_connected = True

            except Exception as e:
                raise ElasticsearchConnectionError(
                    str(e), hosts=self.config.hosts
                )

    async def disconnect_async(self) -> None:
        """Close Elasticsearch connection."""
        if not self._is_connected:
            return

        async with self._lock:
            if not self._is_connected:
                return

            if self._client:
                await self._client.close()
                self._client = None

            self._is_connected = False

    async def validate_connection_async(self) -> bool:
        """Validate Elasticsearch connection.

        Returns:
            True if connection is healthy.
        """
        try:
            if not self._is_connected:
                await self.connect_async()

            result = await self._client.ping()
            return result
        except Exception:
            return False

    # -------------------------------------------------------------------------
    # Schema from Mapping
    # -------------------------------------------------------------------------

    async def get_schema_async(self) -> dict[str, ColumnType]:
        """Get schema from index mapping.

        Returns:
            Column name to type mapping.
        """
        if self._cached_schema is not None:
            return self._cached_schema

        if not self._is_connected:
            await self.connect_async()

        # Get index mapping
        mapping_response = await self._client.indices.get_mapping(
            index=self.config.index
        )

        # Parse mapping to schema
        schema: dict[str, ColumnType] = {}

        for index_name, index_data in mapping_response.items():
            mappings = index_data.get("mappings", {})
            properties = mappings.get("properties", {})
            self._parse_mapping_properties(properties, "", schema)

        self._cached_schema = schema
        return schema

    def _parse_mapping_properties(
        self,
        properties: dict[str, Any],
        prefix: str,
        schema: dict[str, ColumnType],
    ) -> None:
        """Parse mapping properties recursively.

        Args:
            properties: Mapping properties.
            prefix: Current field prefix.
            schema: Schema dict to populate.
        """
        for field_name, field_def in properties.items():
            full_name = f"{prefix}.{field_name}" if prefix else field_name
            field_type = field_def.get("type", "object")

            # Handle nested properties
            if "properties" in field_def:
                if self.config.flatten_nested:
                    self._parse_mapping_properties(
                        field_def["properties"], full_name, schema
                    )
                else:
                    schema[full_name] = ColumnType.STRUCT
            else:
                # Map ES type to ColumnType
                schema[full_name] = self.ES_TYPE_MAPPING.get(
                    field_type, ColumnType.UNKNOWN
                )

    # -------------------------------------------------------------------------
    # Document Operations
    # -------------------------------------------------------------------------

    async def _fetch_sample_documents(
        self, n: int
    ) -> list[dict[str, Any]]:
        """Fetch sample documents for schema inference.

        Args:
            n: Number of documents to sample.

        Returns:
            List of sample documents.
        """
        if not self._is_connected:
            await self.connect_async()

        response = await self._client.search(
            index=self.config.index,
            size=n,
            query={"match_all": {}},
        )

        hits = response.get("hits", {}).get("hits", [])
        return [hit["_source"] for hit in hits]

    async def _fetch_documents(
        self,
        filter: dict[str, Any] | None = None,
        limit: int | None = None,
        skip: int | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch documents using search or scroll.

        Args:
            filter: Elasticsearch query.
            limit: Maximum documents to return.
            skip: Number of documents to skip.

        Returns:
            List of documents.
        """
        if not self._is_connected:
            await self.connect_async()

        limit = limit or self.config.max_documents
        query = filter or {"match_all": {}}

        # For small result sets, use regular search
        if limit <= 10000 and not skip:
            response = await self._client.search(
                index=self.config.index,
                size=limit,
                query=query,
                track_total_hits=self.config.track_total_hits,
            )
            hits = response.get("hits", {}).get("hits", [])
            return [self._normalize_document(hit["_source"]) for hit in hits]

        # For large result sets, use scroll
        return await self._scroll_documents(query, limit, skip)

    async def _scroll_documents(
        self,
        query: dict[str, Any],
        limit: int,
        skip: int | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch documents using scroll API.

        Args:
            query: Elasticsearch query.
            limit: Maximum documents.
            skip: Documents to skip.

        Returns:
            List of documents.
        """
        documents: list[dict[str, Any]] = []
        scroll_id = None
        fetched = 0
        skipped = 0

        try:
            # Initial search
            response = await self._client.search(
                index=self.config.index,
                scroll=self.config.scroll_timeout,
                size=self.config.scroll_size,
                query=query,
                track_total_hits=self.config.track_total_hits,
            )

            while True:
                hits = response.get("hits", {}).get("hits", [])
                if not hits:
                    break

                scroll_id = response.get("_scroll_id")

                for hit in hits:
                    if skip and skipped < skip:
                        skipped += 1
                        continue

                    documents.append(
                        self._normalize_document(hit["_source"])
                    )
                    fetched += 1

                    if fetched >= limit:
                        break

                if fetched >= limit:
                    break

                # Continue scrolling
                response = await self._client.scroll(
                    scroll_id=scroll_id,
                    scroll=self.config.scroll_timeout,
                )

        finally:
            # Clear scroll context
            if scroll_id:
                try:
                    await self._client.clear_scroll(scroll_id=scroll_id)
                except Exception:
                    pass  # Best effort cleanup

        return documents

    async def _count_documents(
        self, filter: dict[str, Any] | None = None
    ) -> int:
        """Count documents in index.

        Args:
            filter: Optional query filter.

        Returns:
            Document count.
        """
        if not self._is_connected:
            await self.connect_async()

        query = filter or {"match_all": {}}
        response = await self._client.count(
            index=self.config.index,
            query=query,
        )
        return response.get("count", 0)

    def _normalize_document(self, doc: dict[str, Any]) -> dict[str, Any]:
        """Normalize document for Polars compatibility.

        Args:
            doc: Elasticsearch document.

        Returns:
            Normalized document.
        """
        # ES documents are generally JSON-compatible
        # Flatten if configured
        if self.config.flatten_nested:
            return self._schema_inferrer.flatten_document(doc)
        return doc

    # -------------------------------------------------------------------------
    # Search and Aggregation
    # -------------------------------------------------------------------------

    async def search_async(
        self,
        query: dict[str, Any] | None = None,
        size: int = 10,
        sort: list[dict[str, Any]] | None = None,
        source: list[str] | bool | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Execute search query.

        Args:
            query: Elasticsearch query DSL.
            size: Number of results.
            sort: Sort specification.
            source: Fields to include/exclude.
            **kwargs: Additional search parameters.

        Returns:
            List of matching documents.

        Example:
            >>> results = await source.search_async(
            ...     query={"match": {"title": "python"}},
            ...     size=20,
            ...     sort=[{"date": "desc"}],
            ... )
        """
        if not self._is_connected:
            await self.connect_async()

        search_kwargs: dict[str, Any] = {
            "index": self.config.index,
            "size": size,
            "query": query or {"match_all": {}},
        }

        if sort:
            search_kwargs["sort"] = sort
        if source is not None:
            search_kwargs["source"] = source
        search_kwargs.update(kwargs)

        response = await self._client.search(**search_kwargs)
        hits = response.get("hits", {}).get("hits", [])
        return [hit["_source"] for hit in hits]

    async def aggregate_async(
        self,
        aggs: dict[str, Any],
        query: dict[str, Any] | None = None,
        size: int = 0,
    ) -> dict[str, Any]:
        """Execute aggregation.

        Args:
            aggs: Aggregation specification.
            query: Optional query filter.
            size: Number of hits to return (0 for aggs only).

        Returns:
            Aggregation results.

        Example:
            >>> aggs_result = await source.aggregate_async(
            ...     aggs={
            ...         "status_counts": {
            ...             "terms": {"field": "status"}
            ...         },
            ...         "avg_price": {
            ...             "avg": {"field": "price"}
            ...         },
            ...     },
            ...     query={"range": {"date": {"gte": "2024-01-01"}}},
            ... )
        """
        if not self._is_connected:
            await self.connect_async()

        response = await self._client.search(
            index=self.config.index,
            size=size,
            query=query or {"match_all": {}},
            aggs=aggs,
        )
        return response.get("aggregations", {})

    # -------------------------------------------------------------------------
    # Sampling
    # -------------------------------------------------------------------------

    async def sample_async(
        self, n: int = 1000, seed: int | None = None
    ) -> "ElasticsearchDataSource":
        """Create a sampled data source.

        Uses Elasticsearch's random_score for sampling.

        Args:
            n: Number of documents to sample.
            seed: Random seed for reproducibility.

        Returns:
            New ElasticsearchDataSource with sampled configuration.
        """
        # Create new config with reduced max_documents
        config = ElasticsearchConfig(
            hosts=self.config.hosts,
            cloud_id=self.config.cloud_id,
            api_key=self.config.api_key,
            username=self.config.username,
            password=self.config.password,
            index=self.config.index,
            name=f"{self.name}_sample",
            max_documents=n,
            schema_sample_size=min(n, self.config.schema_sample_size),
        )
        return ElasticsearchDataSource(config)

    # -------------------------------------------------------------------------
    # Index Information
    # -------------------------------------------------------------------------

    async def get_index_info_async(self) -> dict[str, Any]:
        """Get index information.

        Returns:
            Index settings and stats.
        """
        if not self._is_connected:
            await self.connect_async()

        settings = await self._client.indices.get_settings(
            index=self.config.index
        )
        stats = await self._client.indices.stats(
            index=self.config.index
        )

        return {
            "settings": settings,
            "stats": stats,
        }

    async def get_field_caps_async(
        self, fields: list[str] | None = None
    ) -> dict[str, Any]:
        """Get field capabilities.

        Args:
            fields: Fields to get capabilities for.

        Returns:
            Field capabilities.
        """
        if not self._is_connected:
            await self.connect_async()

        return await self._client.field_caps(
            index=self.config.index,
            fields=fields or ["*"],
        )
