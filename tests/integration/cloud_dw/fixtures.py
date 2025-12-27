"""Test fixtures and data generators for Cloud DW integration tests.

This module provides:
    - Standard test data schemas
    - Test data generators
    - Pytest fixtures
    - Data validation helpers

The test data is designed to cover various data quality scenarios
that Truthound validators need to handle.
"""

from __future__ import annotations

import random
import string
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Iterator, TypeVar

from tests.integration.cloud_dw.base import TestDataType


# =============================================================================
# Type Mapping
# =============================================================================


class SQLDialect(Enum):
    """SQL dialect for type mapping."""

    BIGQUERY = "bigquery"
    SNOWFLAKE = "snowflake"
    REDSHIFT = "redshift"
    DATABRICKS = "databricks"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"


# Standard type mappings for each dialect
TYPE_MAPPINGS: dict[SQLDialect, dict[str, str]] = {
    SQLDialect.BIGQUERY: {
        "int": "INT64",
        "bigint": "INT64",
        "float": "FLOAT64",
        "double": "FLOAT64",
        "decimal": "NUMERIC",
        "string": "STRING",
        "text": "STRING",
        "bool": "BOOL",
        "date": "DATE",
        "datetime": "DATETIME",
        "timestamp": "TIMESTAMP",
        "json": "JSON",
        "array_int": "ARRAY<INT64>",
        "array_string": "ARRAY<STRING>",
    },
    SQLDialect.SNOWFLAKE: {
        "int": "INTEGER",
        "bigint": "BIGINT",
        "float": "FLOAT",
        "double": "DOUBLE",
        "decimal": "DECIMAL(18,4)",
        "string": "VARCHAR",
        "text": "TEXT",
        "bool": "BOOLEAN",
        "date": "DATE",
        "datetime": "TIMESTAMP_NTZ",
        "timestamp": "TIMESTAMP_TZ",
        "json": "VARIANT",
        "array_int": "ARRAY",
        "array_string": "ARRAY",
    },
    SQLDialect.REDSHIFT: {
        "int": "INTEGER",
        "bigint": "BIGINT",
        "float": "REAL",
        "double": "DOUBLE PRECISION",
        "decimal": "DECIMAL(18,4)",
        "string": "VARCHAR(256)",
        "text": "VARCHAR(65535)",
        "bool": "BOOLEAN",
        "date": "DATE",
        "datetime": "TIMESTAMP",
        "timestamp": "TIMESTAMPTZ",
        "json": "SUPER",
        "array_int": "SUPER",
        "array_string": "SUPER",
    },
    SQLDialect.DATABRICKS: {
        "int": "INT",
        "bigint": "BIGINT",
        "float": "FLOAT",
        "double": "DOUBLE",
        "decimal": "DECIMAL(18,4)",
        "string": "STRING",
        "text": "STRING",
        "bool": "BOOLEAN",
        "date": "DATE",
        "datetime": "TIMESTAMP",
        "timestamp": "TIMESTAMP",
        "json": "STRING",  # JSON stored as string in Databricks
        "array_int": "ARRAY<INT>",
        "array_string": "ARRAY<STRING>",
    },
}


def get_type(base_type: str, dialect: SQLDialect) -> str:
    """Get the dialect-specific SQL type.

    Args:
        base_type: Base type name (e.g., "int", "string").
        dialect: SQL dialect.

    Returns:
        Dialect-specific SQL type string.
    """
    mappings = TYPE_MAPPINGS.get(dialect, TYPE_MAPPINGS[SQLDialect.BIGQUERY])
    return mappings.get(base_type, base_type.upper())


# =============================================================================
# Standard Test Data Schemas
# =============================================================================


@dataclass
class StandardTestData:
    """Standard test data schemas and generators.

    This class provides pre-defined test data schemas that cover
    common validation scenarios.
    """

    @staticmethod
    def users_schema(dialect: SQLDialect) -> dict[str, str]:
        """Basic users table schema."""
        return {
            "id": get_type("int", dialect),
            "email": get_type("string", dialect),
            "name": get_type("string", dialect),
            "age": get_type("int", dialect),
            "salary": get_type("decimal", dialect),
            "is_active": get_type("bool", dialect),
            "created_at": get_type("timestamp", dialect),
        }

    @staticmethod
    def users_data(n: int = 100) -> list[dict[str, Any]]:
        """Generate users test data."""
        data = []
        domains = ["gmail.com", "yahoo.com", "example.com", "test.org"]
        names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry"]

        for i in range(1, n + 1):
            name = random.choice(names)
            domain = random.choice(domains)
            data.append({
                "id": i,
                "email": f"{name.lower()}{i}@{domain}",
                "name": name,
                "age": random.randint(18, 80),
                "salary": round(random.uniform(30000, 200000), 2),
                "is_active": random.choice([True, False]),
                "created_at": datetime.now() - timedelta(days=random.randint(0, 365)),
            })

        return data

    @staticmethod
    def products_schema(dialect: SQLDialect) -> dict[str, str]:
        """Products table schema for e-commerce scenarios."""
        return {
            "product_id": get_type("string", dialect),
            "name": get_type("string", dialect),
            "category": get_type("string", dialect),
            "price": get_type("decimal", dialect),
            "quantity": get_type("int", dialect),
            "description": get_type("text", dialect),
            "created_date": get_type("date", dialect),
        }

    @staticmethod
    def products_data(n: int = 50) -> list[dict[str, Any]]:
        """Generate products test data."""
        categories = ["Electronics", "Clothing", "Books", "Home", "Sports"]
        adjectives = ["Premium", "Basic", "Deluxe", "Pro", "Lite"]
        nouns = ["Widget", "Gadget", "Tool", "Device", "Item"]

        data = []
        for i in range(n):
            name = f"{random.choice(adjectives)} {random.choice(nouns)} {i+1}"
            data.append({
                "product_id": f"PROD-{uuid.uuid4().hex[:8].upper()}",
                "name": name,
                "category": random.choice(categories),
                "price": round(random.uniform(9.99, 999.99), 2),
                "quantity": random.randint(0, 1000),
                "description": f"Description for {name}",
                "created_date": date.today() - timedelta(days=random.randint(0, 730)),
            })

        return data

    @staticmethod
    def transactions_schema(dialect: SQLDialect) -> dict[str, str]:
        """Transactions table schema for financial scenarios."""
        return {
            "transaction_id": get_type("string", dialect),
            "user_id": get_type("int", dialect),
            "amount": get_type("decimal", dialect),
            "currency": get_type("string", dialect),
            "status": get_type("string", dialect),
            "transaction_date": get_type("timestamp", dialect),
        }

    @staticmethod
    def transactions_data(n: int = 200, user_ids: list[int] | None = None) -> list[dict[str, Any]]:
        """Generate transactions test data."""
        currencies = ["USD", "EUR", "GBP", "JPY"]
        statuses = ["completed", "pending", "failed", "refunded"]

        if user_ids is None:
            user_ids = list(range(1, 101))

        data = []
        for _ in range(n):
            data.append({
                "transaction_id": f"TXN-{uuid.uuid4().hex[:12].upper()}",
                "user_id": random.choice(user_ids),
                "amount": round(random.uniform(1.00, 5000.00), 2),
                "currency": random.choice(currencies),
                "status": random.choice(statuses),
                "transaction_date": datetime.now() - timedelta(
                    seconds=random.randint(0, 86400 * 30)
                ),
            })

        return data

    @staticmethod
    def nulls_schema(dialect: SQLDialect) -> dict[str, str]:
        """Schema for testing null handling."""
        return {
            "id": get_type("int", dialect),
            "required_field": get_type("string", dialect),
            "optional_field": get_type("string", dialect),
            "numeric_field": get_type("float", dialect),
            "date_field": get_type("date", dialect),
        }

    @staticmethod
    def nulls_data(n: int = 100, null_ratio: float = 0.2) -> list[dict[str, Any]]:
        """Generate data with controlled null patterns."""
        data = []
        for i in range(1, n + 1):
            row = {
                "id": i,
                "required_field": f"value_{i}",
            }

            # Add optional fields with null probability
            row["optional_field"] = (
                None if random.random() < null_ratio else f"optional_{i}"
            )
            row["numeric_field"] = (
                None if random.random() < null_ratio else random.uniform(0, 100)
            )
            row["date_field"] = (
                None if random.random() < null_ratio
                else date.today() - timedelta(days=random.randint(0, 365))
            )

            data.append(row)

        return data

    @staticmethod
    def edge_cases_schema(dialect: SQLDialect) -> dict[str, str]:
        """Schema for testing edge cases."""
        return {
            "id": get_type("int", dialect),
            "empty_string": get_type("string", dialect),
            "whitespace_string": get_type("string", dialect),
            "min_int": get_type("bigint", dialect),
            "max_int": get_type("bigint", dialect),
            "zero_value": get_type("float", dialect),
            "negative_value": get_type("float", dialect),
        }

    @staticmethod
    def edge_cases_data() -> list[dict[str, Any]]:
        """Generate edge case test data."""
        return [
            {
                "id": 1,
                "empty_string": "",
                "whitespace_string": "   ",
                "min_int": -(2**62),
                "max_int": 2**62 - 1,
                "zero_value": 0.0,
                "negative_value": -999999.99,
            },
            {
                "id": 2,
                "empty_string": "normal",
                "whitespace_string": "normal",
                "min_int": 0,
                "max_int": 0,
                "zero_value": 1.0,
                "negative_value": 1.0,
            },
            {
                "id": 3,
                "empty_string": None,
                "whitespace_string": "\t\n",
                "min_int": 1,
                "max_int": -1,
                "zero_value": float("inf") if False else 0.0,  # Most DBs don't support inf
                "negative_value": 0.0,
            },
        ]

    @staticmethod
    def unicode_schema(dialect: SQLDialect) -> dict[str, str]:
        """Schema for testing Unicode handling."""
        return {
            "id": get_type("int", dialect),
            "latin": get_type("string", dialect),
            "chinese": get_type("string", dialect),
            "japanese": get_type("string", dialect),
            "korean": get_type("string", dialect),
            "arabic": get_type("string", dialect),
            "emoji": get_type("string", dialect),
        }

    @staticmethod
    def unicode_data() -> list[dict[str, Any]]:
        """Generate Unicode test data."""
        return [
            {
                "id": 1,
                "latin": "Hello World",
                "chinese": "你好世界",
                "japanese": "こんにちは世界",
                "korean": "안녕하세요 세계",
                "arabic": "مرحبا بالعالم",
                "emoji": "Hello 👋 World 🌍",
            },
            {
                "id": 2,
                "latin": "Café résumé naïve",
                "chinese": "数据质量",
                "japanese": "データ品質",
                "korean": "데이터 품질",
                "arabic": "جودة البيانات",
                "emoji": "Data 📊 Quality ✅",
            },
        ]

    @staticmethod
    def duplicates_schema(dialect: SQLDialect) -> dict[str, str]:
        """Schema for testing duplicate detection."""
        return {
            "id": get_type("int", dialect),
            "unique_key": get_type("string", dialect),
            "category": get_type("string", dialect),
            "value": get_type("float", dialect),
        }

    @staticmethod
    def duplicates_data(n: int = 50, duplicate_ratio: float = 0.1) -> list[dict[str, Any]]:
        """Generate data with controlled duplicates."""
        categories = ["A", "B", "C"]
        data = []

        for i in range(1, n + 1):
            is_duplicate = random.random() < duplicate_ratio and i > 1
            data.append({
                "id": i,
                "unique_key": data[random.randint(0, len(data)-1)]["unique_key"]
                if is_duplicate else f"KEY-{uuid.uuid4().hex[:8]}",
                "category": random.choice(categories),
                "value": random.uniform(0, 100),
            })

        return data


# =============================================================================
# Test Data Generator
# =============================================================================


@dataclass
class GeneratorConfig:
    """Configuration for test data generation.

    Attributes:
        row_count: Number of rows to generate.
        null_ratio: Ratio of null values (0.0 to 1.0).
        duplicate_ratio: Ratio of duplicate values (0.0 to 1.0).
        random_seed: Random seed for reproducibility.
    """

    row_count: int = 100
    null_ratio: float = 0.1
    duplicate_ratio: float = 0.05
    random_seed: int | None = None


class TestDataGenerator:
    """Configurable test data generator.

    This class provides methods to generate test data for various
    validation scenarios with controllable properties.

    Example:
        >>> generator = TestDataGenerator(dialect=SQLDialect.BIGQUERY)
        >>> schema = generator.get_schema("users")
        >>> data = generator.generate("users", row_count=1000)
    """

    def __init__(
        self,
        dialect: SQLDialect,
        config: GeneratorConfig | None = None,
    ) -> None:
        """Initialize the generator.

        Args:
            dialect: SQL dialect for type mapping.
            config: Generator configuration.
        """
        self.dialect = dialect
        self.config = config or GeneratorConfig()

        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)

        # Register standard schemas
        self._schemas: dict[str, Callable[[SQLDialect], dict[str, str]]] = {
            "users": StandardTestData.users_schema,
            "products": StandardTestData.products_schema,
            "transactions": StandardTestData.transactions_schema,
            "nulls": StandardTestData.nulls_schema,
            "edge_cases": StandardTestData.edge_cases_schema,
            "unicode": StandardTestData.unicode_schema,
            "duplicates": StandardTestData.duplicates_schema,
        }

        # Register standard generators
        self._generators: dict[str, Callable[..., list[dict[str, Any]]]] = {
            "users": StandardTestData.users_data,
            "products": StandardTestData.products_data,
            "transactions": StandardTestData.transactions_data,
            "nulls": StandardTestData.nulls_data,
            "edge_cases": StandardTestData.edge_cases_data,
            "unicode": StandardTestData.unicode_data,
            "duplicates": StandardTestData.duplicates_data,
        }

    def get_schema(self, name: str) -> dict[str, str]:
        """Get schema for a named dataset.

        Args:
            name: Dataset name (e.g., "users", "products").

        Returns:
            Schema dictionary mapping column names to types.

        Raises:
            KeyError: If dataset name not found.
        """
        if name not in self._schemas:
            raise KeyError(f"Unknown dataset: {name}. Available: {list(self._schemas.keys())}")
        return self._schemas[name](self.dialect)

    def generate(
        self,
        name: str,
        row_count: int | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Generate test data for a named dataset.

        Args:
            name: Dataset name.
            row_count: Number of rows (overrides config).
            **kwargs: Additional generator arguments.

        Returns:
            List of data rows.
        """
        if name not in self._generators:
            raise KeyError(f"Unknown dataset: {name}. Available: {list(self._generators.keys())}")

        generator = self._generators[name]
        count = row_count or self.config.row_count

        # Some generators don't take row count
        try:
            return generator(n=count, **kwargs)
        except TypeError:
            return generator(**kwargs)

    def register_schema(
        self,
        name: str,
        schema_fn: Callable[[SQLDialect], dict[str, str]],
    ) -> None:
        """Register a custom schema.

        Args:
            name: Dataset name.
            schema_fn: Function that takes dialect and returns schema.
        """
        self._schemas[name] = schema_fn

    def register_generator(
        self,
        name: str,
        generator_fn: Callable[..., list[dict[str, Any]]],
    ) -> None:
        """Register a custom data generator.

        Args:
            name: Dataset name.
            generator_fn: Function that generates data.
        """
        self._generators[name] = generator_fn

    def list_datasets(self) -> list[str]:
        """List available dataset names."""
        return list(self._schemas.keys())

    def generate_all(
        self,
        row_count: int | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Generate data for all registered datasets.

        Args:
            row_count: Number of rows per dataset.

        Returns:
            Dictionary mapping dataset names to data.
        """
        return {
            name: self.generate(name, row_count)
            for name in self._schemas.keys()
        }


# =============================================================================
# Validation Helpers
# =============================================================================


def validate_schema_match(
    expected: dict[str, str],
    actual: dict[str, str],
    strict: bool = False,
) -> tuple[bool, list[str]]:
    """Validate that actual schema matches expected.

    Args:
        expected: Expected schema.
        actual: Actual schema from database.
        strict: If True, types must match exactly.

    Returns:
        Tuple of (success, list of error messages).
    """
    errors = []

    # Check for missing columns
    for col in expected:
        if col.lower() not in {c.lower() for c in actual}:
            errors.append(f"Missing column: {col}")

    # Check for extra columns
    for col in actual:
        if col.lower() not in {c.lower() for c in expected}:
            errors.append(f"Unexpected column: {col}")

    # Check types if strict
    if strict:
        for col, expected_type in expected.items():
            actual_col = next((c for c in actual if c.lower() == col.lower()), None)
            if actual_col:
                actual_type = actual[actual_col]
                if expected_type.upper() != actual_type.upper():
                    errors.append(
                        f"Type mismatch for {col}: expected {expected_type}, got {actual_type}"
                    )

    return len(errors) == 0, errors


def validate_row_count(
    expected: int,
    actual: int,
    tolerance: float = 0.0,
) -> tuple[bool, str | None]:
    """Validate row count with optional tolerance.

    Args:
        expected: Expected row count.
        actual: Actual row count.
        tolerance: Allowed deviation ratio (0.0 to 1.0).

    Returns:
        Tuple of (success, error message or None).
    """
    if tolerance == 0.0:
        if expected != actual:
            return False, f"Row count mismatch: expected {expected}, got {actual}"
    else:
        min_count = int(expected * (1 - tolerance))
        max_count = int(expected * (1 + tolerance))
        if not (min_count <= actual <= max_count):
            return False, (
                f"Row count {actual} outside tolerance: "
                f"expected {expected} ± {tolerance*100:.0f}%"
            )

    return True, None


def validate_data_types(
    data: list[dict[str, Any]],
    schema: dict[str, str],
) -> tuple[bool, list[str]]:
    """Validate that data matches expected types.

    Args:
        data: Data rows.
        schema: Expected schema.

    Returns:
        Tuple of (success, list of error messages).
    """
    errors = []

    type_checks = {
        "int": (int,),
        "bigint": (int,),
        "float": (int, float),
        "double": (int, float),
        "decimal": (int, float, Decimal),
        "string": (str,),
        "text": (str,),
        "bool": (bool,),
        "date": (date, datetime),
        "datetime": (datetime,),
        "timestamp": (datetime,),
    }

    for i, row in enumerate(data[:100]):  # Check first 100 rows
        for col, dtype in schema.items():
            if col not in row:
                continue

            value = row[col]
            if value is None:
                continue

            # Get base type for checking
            base_type = dtype.lower().split("(")[0].strip()
            allowed_types = type_checks.get(base_type)

            if allowed_types and not isinstance(value, allowed_types):
                errors.append(
                    f"Row {i}, column {col}: expected {base_type}, got {type(value).__name__}"
                )

    return len(errors) == 0, errors
