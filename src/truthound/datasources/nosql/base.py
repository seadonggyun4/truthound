"""Base classes for NoSQL data sources.

This module provides common functionality for NoSQL database data sources,
including document schema inference and type mapping.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Any, Iterator

from truthound.datasources._protocols import ColumnType, DataSourceCapability
from truthound.datasources.async_base import (
    AsyncBaseDataSource,
    AsyncDataSourceConfig,
    AsyncDataSourceError,
)


# =============================================================================
# Exceptions
# =============================================================================


class NoSQLDataSourceError(AsyncDataSourceError):
    """Base exception for NoSQL data source errors."""

    pass


class SchemaInferenceError(NoSQLDataSourceError):
    """Error during schema inference."""

    def __init__(self, message: str, document_count: int | None = None) -> None:
        self.document_count = document_count
        super().__init__(message)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class NoSQLDataSourceConfig(AsyncDataSourceConfig):
    """Base configuration for NoSQL data sources.

    Attributes:
        schema_sample_size: Number of documents to sample for schema inference.
        infer_nested_types: Whether to infer types for nested documents.
        flatten_nested: Whether to flatten nested documents in output.
        flatten_separator: Separator for flattened field names.
        max_documents: Maximum documents to retrieve.
        batch_size: Batch size for iteration.
    """

    schema_sample_size: int = 100
    infer_nested_types: bool = True
    flatten_nested: bool = False
    flatten_separator: str = "."
    max_documents: int = 100_000
    batch_size: int = 1000


# =============================================================================
# Document Schema Inferrer
# =============================================================================


class DocumentSchemaInferrer:
    """Infer schema from sample documents.

    This class analyzes document structures to determine column types
    and handles nested documents appropriately.

    Example:
        >>> inferrer = DocumentSchemaInferrer(flatten_nested=True)
        >>> schema = inferrer.infer_from_documents([
        ...     {"name": "Alice", "age": 30, "address": {"city": "NYC"}},
        ...     {"name": "Bob", "age": 25, "address": {"city": "LA"}},
        ... ])
        >>> print(schema)
        {'name': ColumnType.STRING, 'age': ColumnType.INTEGER, 'address.city': ColumnType.STRING}
    """

    # Python type to ColumnType mapping
    TYPE_MAPPING = {
        str: ColumnType.STRING,
        int: ColumnType.INTEGER,
        float: ColumnType.FLOAT,
        bool: ColumnType.BOOLEAN,
        bytes: ColumnType.BINARY,
        datetime: ColumnType.DATETIME,
        date: ColumnType.DATE,
        list: ColumnType.LIST,
        dict: ColumnType.STRUCT,
        type(None): ColumnType.NULL,
    }

    def __init__(
        self,
        flatten_nested: bool = False,
        flatten_separator: str = ".",
        infer_nested_types: bool = True,
    ) -> None:
        """Initialize the schema inferrer.

        Args:
            flatten_nested: Whether to flatten nested documents.
            flatten_separator: Separator for flattened field names.
            infer_nested_types: Whether to recurse into nested documents.
        """
        self._flatten_nested = flatten_nested
        self._separator = flatten_separator
        self._infer_nested = infer_nested_types

    def infer_from_documents(
        self,
        documents: list[dict[str, Any]],
    ) -> dict[str, ColumnType]:
        """Infer schema from a list of documents.

        Analyzes all documents and merges field types, preferring
        non-null types when a field has mixed null/non-null values.

        Args:
            documents: List of documents to analyze.

        Returns:
            Column name to type mapping.

        Raises:
            SchemaInferenceError: If no documents provided.
        """
        if not documents:
            raise SchemaInferenceError("No documents provided for schema inference")

        # Collect all field types from all documents
        field_types: dict[str, set[ColumnType]] = {}

        for doc in documents:
            self._extract_field_types(doc, "", field_types)

        # Merge types, preferring non-null
        schema: dict[str, ColumnType] = {}
        for field_name, types in field_types.items():
            schema[field_name] = self._merge_types(types)

        return schema

    def _extract_field_types(
        self,
        doc: dict[str, Any],
        prefix: str,
        field_types: dict[str, set[ColumnType]],
    ) -> None:
        """Extract field types from a document.

        Args:
            doc: Document to analyze.
            prefix: Current field name prefix.
            field_types: Dict to accumulate field types.
        """
        for key, value in doc.items():
            field_name = f"{prefix}{self._separator}{key}" if prefix else key

            # Handle nested documents
            if isinstance(value, dict) and self._infer_nested:
                if self._flatten_nested:
                    self._extract_field_types(value, field_name, field_types)
                else:
                    # Store as STRUCT type
                    if field_name not in field_types:
                        field_types[field_name] = set()
                    field_types[field_name].add(ColumnType.STRUCT)
            else:
                # Get type for value
                col_type = self._infer_type(value)
                if field_name not in field_types:
                    field_types[field_name] = set()
                field_types[field_name].add(col_type)

    def _infer_type(self, value: Any) -> ColumnType:
        """Infer ColumnType from a Python value.

        Args:
            value: Python value.

        Returns:
            Corresponding ColumnType.
        """
        if value is None:
            return ColumnType.NULL

        # Check exact type matches
        value_type = type(value)
        if value_type in self.TYPE_MAPPING:
            return self.TYPE_MAPPING[value_type]

        # Handle special cases
        if isinstance(value, (list, tuple)):
            return ColumnType.LIST
        if isinstance(value, dict):
            return ColumnType.STRUCT

        # Check for ObjectId (MongoDB)
        type_name = value_type.__name__
        if type_name == "ObjectId":
            return ColumnType.STRING
        if type_name == "Decimal128":
            return ColumnType.DECIMAL
        if type_name == "Binary":
            return ColumnType.BINARY
        if type_name == "UUID":
            return ColumnType.STRING

        return ColumnType.UNKNOWN

    def _merge_types(self, types: set[ColumnType]) -> ColumnType:
        """Merge multiple types into a single type.

        Prefers non-null types and more specific types.

        Args:
            types: Set of types observed for a field.

        Returns:
            Merged type.
        """
        # Remove NULL if other types present
        non_null_types = types - {ColumnType.NULL}

        if not non_null_types:
            return ColumnType.NULL

        if len(non_null_types) == 1:
            return non_null_types.pop()

        # Multiple types - check for compatible promotions
        if non_null_types == {ColumnType.INTEGER, ColumnType.FLOAT}:
            return ColumnType.FLOAT
        if ColumnType.STRING in non_null_types:
            return ColumnType.STRING  # String is most general

        # Default to UNKNOWN for incompatible types
        return ColumnType.UNKNOWN

    def flatten_document(
        self,
        doc: dict[str, Any],
        prefix: str = "",
    ) -> dict[str, Any]:
        """Flatten a nested document.

        Args:
            doc: Document to flatten.
            prefix: Current field name prefix.

        Returns:
            Flattened document.
        """
        result: dict[str, Any] = {}

        for key, value in doc.items():
            field_name = f"{prefix}{self._separator}{key}" if prefix else key

            if isinstance(value, dict):
                # Recursively flatten
                nested = self.flatten_document(value, field_name)
                result.update(nested)
            else:
                result[field_name] = value

        return result


# =============================================================================
# Abstract NoSQL Base Data Source
# =============================================================================


class BaseNoSQLDataSource(AsyncBaseDataSource[NoSQLDataSourceConfig]):
    """Abstract base class for NoSQL data sources.

    Provides common functionality for document-based databases including
    schema inference, document iteration, and Polars conversion.
    """

    source_type: str = "nosql"

    def __init__(self, config: NoSQLDataSourceConfig | None = None) -> None:
        """Initialize the NoSQL data source.

        Args:
            config: Optional configuration.
        """
        super().__init__(config)
        self._schema_inferrer = DocumentSchemaInferrer(
            flatten_nested=self._config.flatten_nested,
            flatten_separator=self._config.flatten_separator,
            infer_nested_types=self._config.infer_nested_types,
        )

    @classmethod
    def _default_config(cls) -> NoSQLDataSourceConfig:
        """Create default configuration."""
        return NoSQLDataSourceConfig()

    @property
    def capabilities(self) -> set[DataSourceCapability]:
        """Get data source capabilities."""
        return {
            DataSourceCapability.SCHEMA_INFERENCE,
            DataSourceCapability.SAMPLING,
            DataSourceCapability.STREAMING,
        }

    # -------------------------------------------------------------------------
    # Abstract Methods (to be implemented by subclasses)
    # -------------------------------------------------------------------------

    @abstractmethod
    async def _fetch_sample_documents(
        self, n: int
    ) -> list[dict[str, Any]]:
        """Fetch sample documents for schema inference.

        Args:
            n: Number of documents to fetch.

        Returns:
            List of documents.
        """
        pass

    @abstractmethod
    async def _fetch_documents(
        self,
        filter: dict[str, Any] | None = None,
        limit: int | None = None,
        skip: int | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch documents from the data source.

        Args:
            filter: Optional filter criteria.
            limit: Maximum documents to return.
            skip: Number of documents to skip.

        Returns:
            List of documents.
        """
        pass

    @abstractmethod
    async def _count_documents(
        self, filter: dict[str, Any] | None = None
    ) -> int:
        """Count documents in the data source.

        Args:
            filter: Optional filter criteria.

        Returns:
            Document count.
        """
        pass

    # -------------------------------------------------------------------------
    # Schema Inference
    # -------------------------------------------------------------------------

    async def get_schema_async(self) -> dict[str, ColumnType]:
        """Asynchronously infer and return the schema.

        Returns:
            Column name to type mapping.
        """
        if self._cached_schema is not None:
            return self._cached_schema

        # Fetch sample documents
        sample_docs = await self._fetch_sample_documents(
            self._config.schema_sample_size
        )

        # Infer schema
        self._cached_schema = self._schema_inferrer.infer_from_documents(sample_docs)
        return self._cached_schema

    # -------------------------------------------------------------------------
    # Row Count
    # -------------------------------------------------------------------------

    async def get_row_count_async(self) -> int | None:
        """Asynchronously get document count.

        Returns:
            Document count.
        """
        if self._cached_row_count is not None:
            return self._cached_row_count

        self._cached_row_count = await self._count_documents()
        return self._cached_row_count

    # -------------------------------------------------------------------------
    # Polars Conversion
    # -------------------------------------------------------------------------

    async def to_polars_lazyframe_async(self) -> "pl.LazyFrame":
        """Asynchronously convert to Polars LazyFrame.

        Returns:
            Polars LazyFrame containing the data.
        """
        import polars as pl

        # Fetch documents (respecting max_documents limit)
        documents = await self._fetch_documents(limit=self._config.max_documents)

        # Flatten if configured
        if self._config.flatten_nested:
            documents = [
                self._schema_inferrer.flatten_document(doc)
                for doc in documents
            ]

        # Convert to DataFrame
        if not documents:
            # Return empty DataFrame with inferred schema
            schema = await self.get_schema_async()
            return pl.DataFrame(
                {col: [] for col in schema.keys()}
            ).lazy()

        # Normalize documents (ensure all have same keys)
        all_keys = set()
        for doc in documents:
            all_keys.update(doc.keys())

        normalized = []
        for doc in documents:
            normalized_doc = {key: doc.get(key) for key in all_keys}
            normalized.append(normalized_doc)

        return pl.DataFrame(normalized).lazy()

    # -------------------------------------------------------------------------
    # Iteration
    # -------------------------------------------------------------------------

    async def iter_documents_async(
        self,
        filter: dict[str, Any] | None = None,
        batch_size: int | None = None,
    ):
        """Asynchronously iterate over documents in batches.

        Args:
            filter: Optional filter criteria.
            batch_size: Number of documents per batch.

        Yields:
            Batches of documents.
        """
        batch_size = batch_size or self._config.batch_size
        skip = 0

        while True:
            batch = await self._fetch_documents(
                filter=filter,
                limit=batch_size,
                skip=skip,
            )

            if not batch:
                break

            yield batch

            if len(batch) < batch_size:
                break

            skip += batch_size
