"""NoSQL data source implementations.

This package provides data sources for NoSQL databases including:
- MongoDB (document database)
- Elasticsearch (search engine)

All NoSQL data sources support async operations and provide efficient
schema inference from document samples.

Example:
    >>> from truthound.datasources.nosql import MongoDBDataSource
    >>>
    >>> async with MongoDBDataSource.from_connection_string(
    ...     uri="mongodb://localhost:27017",
    ...     database="mydb",
    ...     collection="users",
    ... ) as source:
    ...     schema = await source.get_schema_async()
    ...     lf = await source.to_polars_lazyframe_async()
"""

from truthound.datasources.nosql.base import (
    BaseNoSQLDataSource,
    NoSQLDataSourceConfig,
    DocumentSchemaInferrer,
)
from truthound.datasources.nosql.mongodb import (
    MongoDBDataSource,
    MongoDBConfig,
)
from truthound.datasources.nosql.elasticsearch import (
    ElasticsearchDataSource,
    ElasticsearchConfig,
)

__all__ = [
    # Base
    "BaseNoSQLDataSource",
    "NoSQLDataSourceConfig",
    "DocumentSchemaInferrer",
    # MongoDB
    "MongoDBDataSource",
    "MongoDBConfig",
    # Elasticsearch
    "ElasticsearchDataSource",
    "ElasticsearchConfig",
]
