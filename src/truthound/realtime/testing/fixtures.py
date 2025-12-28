"""Test fixtures for streaming validation tests.

Provides reusable test data generators and fixtures for
both unit and integration testing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Iterator
import json
import random
import string
import uuid


# =============================================================================
# Test Message Generator
# =============================================================================


@dataclass
class SchemaField:
    """Schema field definition.

    Attributes:
        name: Field name
        dtype: Data type (int, float, str, bool, datetime, email, uuid)
        nullable: Whether field can be null
        min_value: Minimum value for numeric types
        max_value: Maximum value for numeric types
        pattern: Pattern for string types
        choices: List of valid choices
    """

    name: str
    dtype: str = "str"
    nullable: bool = False
    min_value: float | None = None
    max_value: float | None = None
    pattern: str | None = None
    choices: list[Any] | None = None


@dataclass
class TestMessageGeneratorConfig:
    """Configuration for test message generation.

    Attributes:
        schema: List of schema fields
        error_rate: Probability of generating invalid data [0.0-1.0]
        null_rate: Probability of generating null values [0.0-1.0]
        seed: Random seed for reproducibility
    """

    schema: list[SchemaField] = field(default_factory=list)
    error_rate: float = 0.0
    null_rate: float = 0.0
    seed: int = 42


class TestMessageGenerator:
    """Generator for test messages with configurable schemas.

    Example:
        >>> schema = [
        ...     SchemaField("id", "int", min_value=1, max_value=10000),
        ...     SchemaField("name", "str"),
        ...     SchemaField("email", "email"),
        ...     SchemaField("amount", "float", min_value=0, max_value=1000),
        ... ]
        >>> gen = TestMessageGenerator(TestMessageGeneratorConfig(
        ...     schema=schema,
        ...     error_rate=0.1,
        ... ))
        >>> messages = list(gen.generate(100))
    """

    def __init__(self, config: TestMessageGeneratorConfig | None = None):
        self._config = config or TestMessageGeneratorConfig()
        self._rng = random.Random(self._config.seed)
        self._counter = 0

    def generate(self, count: int) -> Iterator[dict[str, Any]]:
        """Generate test messages.

        Args:
            count: Number of messages to generate

        Yields:
            Generated messages
        """
        for _ in range(count):
            yield self._generate_message()

    def generate_batch(self, count: int) -> list[dict[str, Any]]:
        """Generate a batch of test messages.

        Args:
            count: Number of messages

        Returns:
            List of messages
        """
        return list(self.generate(count))

    def _generate_message(self) -> dict[str, Any]:
        """Generate a single message."""
        self._counter += 1
        message: dict[str, Any] = {}
        inject_error = self._rng.random() < self._config.error_rate

        for field in self._config.schema:
            # Check for null injection
            if field.nullable and self._rng.random() < self._config.null_rate:
                message[field.name] = None
                continue

            # Generate value
            message[field.name] = self._generate_field(field, inject_error)

        return message

    def _generate_field(self, field: SchemaField, inject_error: bool) -> Any:
        """Generate a value for a field."""
        # Use choices if available
        if field.choices and not inject_error:
            return self._rng.choice(field.choices)

        dtype = field.dtype

        if dtype == "int":
            if inject_error and self._rng.random() < 0.5:
                return "not_an_int"  # Type error
            min_val = int(field.min_value or 0)
            max_val = int(field.max_value or 10000)
            return self._rng.randint(min_val, max_val)

        elif dtype == "float":
            if inject_error and self._rng.random() < 0.5:
                return float("nan")  # Invalid value
            min_val = field.min_value or 0.0
            max_val = field.max_value or 1000.0
            return round(self._rng.uniform(min_val, max_val), 2)

        elif dtype == "str":
            if inject_error and self._rng.random() < 0.5:
                return ""  # Empty string
            length = self._rng.randint(5, 20)
            return "".join(self._rng.choices(string.ascii_letters, k=length))

        elif dtype == "bool":
            if inject_error and self._rng.random() < 0.5:
                return "maybe"  # Invalid boolean
            return self._rng.choice([True, False])

        elif dtype == "datetime":
            if inject_error and self._rng.random() < 0.5:
                return "invalid-date"  # Invalid datetime
            base = datetime.now(timezone.utc)
            offset = timedelta(days=self._rng.randint(-365, 365))
            return (base + offset).isoformat()

        elif dtype == "email":
            if inject_error and self._rng.random() < 0.5:
                return "not-an-email"  # Invalid email
            domains = ["example.com", "test.org", "mock.io"]
            return f"user{self._counter}@{self._rng.choice(domains)}"

        elif dtype == "uuid":
            if inject_error and self._rng.random() < 0.5:
                return "not-a-uuid"  # Invalid UUID
            return str(uuid.uuid4())

        elif dtype == "json":
            if inject_error and self._rng.random() < 0.5:
                return "{invalid json}"  # Invalid JSON
            return {"nested": {"value": self._rng.randint(1, 100)}}

        else:
            return None

    def reset(self) -> None:
        """Reset generator state."""
        self._rng = random.Random(self._config.seed)
        self._counter = 0


# =============================================================================
# Predefined Schemas
# =============================================================================


def create_test_schema(schema_type: str = "default") -> list[SchemaField]:
    """Create a predefined test schema.

    Args:
        schema_type: Schema type (default, user, order, event, metrics)

    Returns:
        List of schema fields
    """
    schemas = {
        "default": [
            SchemaField("id", "int", min_value=1, max_value=100000),
            SchemaField("value", "float", min_value=0, max_value=1000),
            SchemaField("name", "str"),
            SchemaField("timestamp", "datetime"),
        ],
        "user": [
            SchemaField("user_id", "uuid"),
            SchemaField("username", "str"),
            SchemaField("email", "email"),
            SchemaField("age", "int", nullable=True, min_value=18, max_value=99),
            SchemaField("is_active", "bool"),
            SchemaField("created_at", "datetime"),
        ],
        "order": [
            SchemaField("order_id", "uuid"),
            SchemaField("customer_id", "uuid"),
            SchemaField("product_id", "int", min_value=1, max_value=10000),
            SchemaField("quantity", "int", min_value=1, max_value=100),
            SchemaField("unit_price", "float", min_value=0.01, max_value=9999.99),
            SchemaField("total_amount", "float", min_value=0.01, max_value=999999.99),
            SchemaField(
                "status",
                "str",
                choices=["pending", "processing", "shipped", "delivered", "cancelled"],
            ),
            SchemaField("order_date", "datetime"),
        ],
        "event": [
            SchemaField("event_id", "uuid"),
            SchemaField("event_type", "str", choices=["click", "view", "purchase", "signup"]),
            SchemaField("user_id", "uuid"),
            SchemaField("session_id", "uuid"),
            SchemaField("timestamp", "datetime"),
            SchemaField("properties", "json"),
        ],
        "metrics": [
            SchemaField("metric_id", "uuid"),
            SchemaField(
                "metric_name",
                "str",
                choices=["cpu_usage", "memory_usage", "disk_io", "network_io"],
            ),
            SchemaField("value", "float", min_value=0, max_value=100),
            SchemaField("host", "str"),
            SchemaField("timestamp", "datetime"),
            SchemaField("tags", "json"),
        ],
    }

    if schema_type not in schemas:
        raise ValueError(f"Unknown schema type: {schema_type}. Available: {list(schemas.keys())}")

    return schemas[schema_type]


def create_test_messages(
    count: int,
    schema_type: str = "default",
    error_rate: float = 0.0,
    null_rate: float = 0.0,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Create test messages with a predefined schema.

    Args:
        count: Number of messages
        schema_type: Schema type
        error_rate: Error injection rate
        null_rate: Null injection rate
        seed: Random seed

    Returns:
        List of test messages
    """
    schema = create_test_schema(schema_type)
    config = TestMessageGeneratorConfig(
        schema=schema,
        error_rate=error_rate,
        null_rate=null_rate,
        seed=seed,
    )
    generator = TestMessageGenerator(config)
    return generator.generate_batch(count)


# =============================================================================
# Test Fixtures Class
# =============================================================================


class TestFixtures:
    """Collection of test fixtures for streaming tests.

    Example:
        >>> fixtures = TestFixtures()
        >>> messages = fixtures.user_messages(100)
        >>> orders = fixtures.order_messages(50, error_rate=0.1)
    """

    def __init__(self, seed: int = 42):
        self._seed = seed

    def default_messages(
        self,
        count: int,
        error_rate: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Generate default test messages."""
        return create_test_messages(
            count,
            schema_type="default",
            error_rate=error_rate,
            seed=self._seed,
        )

    def user_messages(
        self,
        count: int,
        error_rate: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Generate user test messages."""
        return create_test_messages(
            count,
            schema_type="user",
            error_rate=error_rate,
            seed=self._seed,
        )

    def order_messages(
        self,
        count: int,
        error_rate: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Generate order test messages."""
        return create_test_messages(
            count,
            schema_type="order",
            error_rate=error_rate,
            seed=self._seed,
        )

    def event_messages(
        self,
        count: int,
        error_rate: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Generate event test messages."""
        return create_test_messages(
            count,
            schema_type="event",
            error_rate=error_rate,
            seed=self._seed,
        )

    def metrics_messages(
        self,
        count: int,
        error_rate: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Generate metrics test messages."""
        return create_test_messages(
            count,
            schema_type="metrics",
            error_rate=error_rate,
            seed=self._seed,
        )

    def mixed_messages(
        self,
        count_per_type: int,
        error_rate: float = 0.0,
    ) -> dict[str, list[dict[str, Any]]]:
        """Generate messages of all types.

        Args:
            count_per_type: Number of messages per type
            error_rate: Error injection rate

        Returns:
            Dict mapping type name to message list
        """
        return {
            "default": self.default_messages(count_per_type, error_rate),
            "user": self.user_messages(count_per_type, error_rate),
            "order": self.order_messages(count_per_type, error_rate),
            "event": self.event_messages(count_per_type, error_rate),
            "metrics": self.metrics_messages(count_per_type, error_rate),
        }

    def validation_test_cases(self) -> list[dict[str, Any]]:
        """Generate test cases for validation testing.

        Returns a mix of valid and invalid data points with
        known validation issues.
        """
        return [
            # Valid cases
            {"id": 1, "value": 100.0, "name": "valid", "is_valid": True},
            {"id": 2, "value": 0.0, "name": "zero_value", "is_valid": True},
            {"id": 3, "value": 999.99, "name": "high_value", "is_valid": True},
            # Null cases
            {"id": None, "value": 100.0, "name": "null_id", "is_valid": False},
            {"id": 4, "value": None, "name": "null_value", "is_valid": False},
            {"id": 5, "value": 100.0, "name": None, "is_valid": False},
            # Type error cases
            {"id": "not_int", "value": 100.0, "name": "type_error_id", "is_valid": False},
            {"id": 6, "value": "not_float", "name": "type_error_value", "is_valid": False},
            {"id": 7, "value": 100.0, "name": 12345, "is_valid": False},
            # Range error cases
            {"id": -1, "value": 100.0, "name": "negative_id", "is_valid": False},
            {"id": 8, "value": -100.0, "name": "negative_value", "is_valid": False},
            {"id": 9, "value": float("inf"), "name": "infinite_value", "is_valid": False},
            {"id": 10, "value": float("nan"), "name": "nan_value", "is_valid": False},
            # Empty string case
            {"id": 11, "value": 100.0, "name": "", "is_valid": False},
        ]

    def drift_test_data(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Generate reference and current data for drift testing.

        Returns:
            Tuple of (reference_data, current_data_with_drift)
        """
        rng = random.Random(self._seed)

        # Reference data: normal distribution centered at 50
        reference = [
            {"value": rng.gauss(50, 10), "category": rng.choice(["A", "B", "C"])}
            for _ in range(1000)
        ]

        # Current data: distribution shifted to 70 (drift!)
        current = [
            {"value": rng.gauss(70, 15), "category": rng.choice(["A", "B", "D"])}  # "D" replaces some "C"
            for _ in range(1000)
        ]

        return reference, current


# =============================================================================
# Pytest Fixtures (for integration with pytest)
# =============================================================================


def pytest_fixtures():
    """Get pytest fixture definitions.

    Import and use in conftest.py:
        from truthound.realtime.testing.fixtures import pytest_fixtures
        pytest_fixtures()
    """
    try:
        import pytest
    except ImportError:
        return

    @pytest.fixture
    def test_messages():
        """Fixture for default test messages."""
        return create_test_messages(100)

    @pytest.fixture
    def test_fixtures():
        """Fixture for TestFixtures instance."""
        return TestFixtures()

    @pytest.fixture
    def message_generator():
        """Fixture for TestMessageGenerator."""
        schema = create_test_schema("default")
        config = TestMessageGeneratorConfig(schema=schema)
        return TestMessageGenerator(config)

    return {
        "test_messages": test_messages,
        "test_fixtures": test_fixtures,
        "message_generator": message_generator,
    }
