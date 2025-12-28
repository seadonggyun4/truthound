"""Mock adapter for testing streaming pipelines.

Provides a fully controllable mock adapter that generates
synthetic data for testing streaming validation without
requiring external infrastructure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator
import asyncio
import random

from truthound.realtime.protocols import (
    StreamMessage,
    MessageHeader,
    DeserializationFormat,
)
from truthound.realtime.adapters.base import (
    BaseStreamAdapter,
    AdapterConfig,
)


@dataclass
class MockAdapterConfig(AdapterConfig):
    """Configuration for mock adapter.

    Attributes:
        schema: Schema for generated data {column: type}
        num_messages: Total messages to generate (0 for infinite)
        messages_per_batch: Messages per consume call
        error_rate: Probability of generating bad data [0.0-1.0]
        delay_ms: Delay between messages in milliseconds
        seed: Random seed for reproducibility
        topic: Mock topic name
        partitions: Number of partitions
    """

    schema: dict[str, str] = field(default_factory=lambda: {
        "id": "int",
        "value": "float",
        "name": "str",
        "timestamp": "datetime",
    })
    num_messages: int = 1000
    messages_per_batch: int = 100
    error_rate: float = 0.1
    delay_ms: int = 0
    seed: int = 42
    topic: str = "mock-topic"
    partitions: int = 1
    value_deserializer: DeserializationFormat = DeserializationFormat.JSON


class MockAdapter(BaseStreamAdapter[dict[str, Any], MockAdapterConfig]):
    """Mock streaming adapter for testing.

    Generates synthetic data with configurable error rates
    for testing streaming validation pipelines without
    requiring external infrastructure.

    Example:
        >>> config = MockAdapterConfig(
        ...     schema={"id": "int", "value": "float"},
        ...     num_messages=1000,
        ...     error_rate=0.1,
        ... )
        >>> async with MockAdapter(config) as adapter:
        ...     async for message in adapter.consume():
        ...         print(message.value)
    """

    adapter_type = "mock"

    def __init__(
        self,
        config: MockAdapterConfig | None = None,
        **kwargs: Any,
    ):
        config = config or MockAdapterConfig()
        super().__init__(config, **kwargs)

        self._rng = random.Random()
        self._message_count = 0
        self._current_offset = 0
        self._committed_offsets: dict[int, int] = {}
        self._message_buffer: list[dict[str, Any]] = []

    async def _do_connect(self) -> None:
        """Initialize mock connection."""
        self._rng = random.Random(self._config.seed)
        self._message_count = 0
        self._current_offset = 0
        self._committed_offsets.clear()
        self._message_buffer.clear()

    async def _do_disconnect(self) -> None:
        """Disconnect mock adapter."""
        self._message_buffer.clear()

    async def _do_consume(self) -> AsyncIterator[StreamMessage[dict[str, Any]]]:
        """Generate mock messages."""
        while True:
            # Check if we've generated all messages
            if 0 < self._config.num_messages <= self._message_count:
                break

            # Generate batch of messages
            batch_size = min(
                self._config.messages_per_batch,
                self._config.num_messages - self._message_count
                if self._config.num_messages > 0
                else self._config.messages_per_batch,
            )

            for _ in range(batch_size):
                if 0 < self._config.num_messages <= self._message_count:
                    break

                # Apply delay if configured
                if self._config.delay_ms > 0:
                    await asyncio.sleep(self._config.delay_ms / 1000)

                message = self._generate_message()
                self._message_count += 1
                yield message

    async def _do_produce(self, message: StreamMessage[dict[str, Any]]) -> None:
        """Store produced message in buffer."""
        self._message_buffer.append(message.value)

    async def _do_commit(self, message: StreamMessage[dict[str, Any]]) -> None:
        """Commit offset for partition."""
        self._committed_offsets[message.partition] = message.offset

    async def _do_seek(self, partition: int, offset: int) -> None:
        """Seek to offset (affects next generated offset)."""
        self._current_offset = offset

    def _generate_message(self) -> StreamMessage[dict[str, Any]]:
        """Generate a single mock message."""
        partition = self._rng.randint(0, self._config.partitions - 1)
        offset = self._current_offset
        self._current_offset += 1

        is_error = self._rng.random() < self._config.error_rate
        value = self._generate_value(is_error)

        return StreamMessage(
            key=f"key-{offset}",
            value=value,
            partition=partition,
            offset=offset,
            timestamp=datetime.now(timezone.utc),
            headers=(MessageHeader("source", b"mock"),),
            topic=self._config.topic,
            metadata={"is_synthetic": True, "error_injected": is_error},
        )

    def _generate_value(self, is_error: bool) -> dict[str, Any]:
        """Generate message value based on schema."""
        value: dict[str, Any] = {}

        for col, dtype in self._config.schema.items():
            value[col] = self._generate_field(col, dtype, is_error)

        return value

    def _generate_field(self, col: str, dtype: str, is_error: bool) -> Any:
        """Generate a single field value."""
        # Randomly inject null as error
        if is_error and self._rng.random() < 0.5:
            return None

        if dtype == "int":
            if is_error and self._rng.random() < 0.3:
                return -999  # Invalid value
            return self._rng.randint(1, 10000)

        elif dtype == "float":
            if is_error and self._rng.random() < 0.3:
                return float("nan")
            return round(self._rng.uniform(0, 1000), 2)

        elif dtype == "str":
            if is_error and self._rng.random() < 0.3:
                return ""  # Empty string
            names = ["Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace"]
            return self._rng.choice(names)

        elif dtype == "datetime":
            return datetime.now(timezone.utc).isoformat()

        elif dtype == "bool":
            if is_error and self._rng.random() < 0.3:
                return None
            return self._rng.choice([True, False])

        elif dtype == "email":
            if is_error and self._rng.random() < 0.3:
                return "invalid-email"
            domains = ["example.com", "test.org", "mock.io"]
            return f"user{self._rng.randint(1, 100)}@{self._rng.choice(domains)}"

        elif dtype == "uuid":
            if is_error and self._rng.random() < 0.3:
                return "not-a-uuid"
            import uuid
            return str(uuid.uuid4())

        else:
            return None

    # -------------------------------------------------------------------------
    # Test Helpers
    # -------------------------------------------------------------------------

    def get_committed_offsets(self) -> dict[int, int]:
        """Get committed offsets (for testing)."""
        return dict(self._committed_offsets)

    def get_produced_messages(self) -> list[dict[str, Any]]:
        """Get produced messages (for testing)."""
        return list(self._message_buffer)

    def reset(self) -> None:
        """Reset adapter state (for testing)."""
        self._message_count = 0
        self._current_offset = 0
        self._committed_offsets.clear()
        self._message_buffer.clear()
        self._rng = random.Random(self._config.seed)

    def inject_messages(self, messages: list[dict[str, Any]]) -> None:
        """Inject messages for consumption (for testing).

        Args:
            messages: Messages to inject
        """
        self._message_buffer = list(messages)
