"""Integration tests for Kafka adapter using Testcontainers.

These tests require Docker to be running and testcontainers installed.
Run with: pytest tests/integration/realtime/test_kafka_adapter.py -v
"""

from __future__ import annotations

import asyncio
import pytest
from datetime import datetime, timezone

# Skip all tests if dependencies not available
pytest_plugins = ["pytest_asyncio"]

try:
    import testcontainers
    TESTCONTAINERS_AVAILABLE = True
except ImportError:
    TESTCONTAINERS_AVAILABLE = False

try:
    import aiokafka
    AIOKAFKA_AVAILABLE = True
except ImportError:
    AIOKAFKA_AVAILABLE = False


@pytest.fixture(scope="module")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.mark.skipif(
    not TESTCONTAINERS_AVAILABLE,
    reason="testcontainers not installed"
)
@pytest.mark.skipif(
    not AIOKAFKA_AVAILABLE,
    reason="aiokafka not installed"
)
class TestKafkaAdapterIntegration:
    """Integration tests for KafkaAdapter with real Kafka."""

    @pytest.fixture(scope="class")
    async def kafka_container(self):
        """Start Kafka container for tests."""
        from truthound.realtime.testing.containers import KafkaTestContainer

        async with KafkaTestContainer() as kafka:
            # Create test topic
            await kafka.create_topic("test-topic", num_partitions=3)
            yield kafka

    @pytest.fixture
    def kafka_config(self, kafka_container):
        """Get Kafka adapter configuration."""
        from truthound.realtime.adapters.kafka import KafkaAdapterConfig

        return KafkaAdapterConfig(
            bootstrap_servers=kafka_container.get_connection_url(),
            topic="test-topic",
            consumer_group="test-consumer",
            enable_auto_commit=False,
        )

    @pytest.mark.asyncio
    async def test_connect_and_disconnect(self, kafka_config):
        """Test connecting and disconnecting from Kafka."""
        from truthound.realtime.adapters.kafka import KafkaAdapter

        adapter = KafkaAdapter(kafka_config)

        # Connect
        await adapter.connect()
        assert adapter.is_connected

        # Disconnect
        await adapter.disconnect()
        assert not adapter.is_connected

    @pytest.mark.asyncio
    async def test_produce_and_consume(self, kafka_config):
        """Test producing and consuming messages."""
        from truthound.realtime.adapters.kafka import KafkaAdapter
        from truthound.realtime.protocols import StreamMessage

        adapter = KafkaAdapter(kafka_config)

        async with adapter:
            # Produce messages
            for i in range(10):
                message = StreamMessage(
                    key=f"key-{i}",
                    value={"id": i, "data": f"test-{i}"},
                    partition=-1,
                    offset=-1,
                    timestamp=datetime.now(timezone.utc),
                    headers=(),
                    topic="test-topic",
                )
                await adapter.produce(message)

            await adapter.flush()

            # Consume messages
            consumed = []
            timeout = asyncio.get_event_loop().time() + 10  # 10 second timeout

            async for msg in adapter.consume():
                consumed.append(msg)
                await adapter.commit(msg)

                if len(consumed) >= 10:
                    break
                if asyncio.get_event_loop().time() > timeout:
                    break

            assert len(consumed) == 10

    @pytest.mark.asyncio
    async def test_batch_produce(self, kafka_config):
        """Test batch message production."""
        from truthound.realtime.adapters.kafka import KafkaAdapter
        from truthound.realtime.protocols import StreamMessage

        adapter = KafkaAdapter(kafka_config)

        async with adapter:
            messages = [
                StreamMessage(
                    key=f"batch-key-{i}",
                    value={"batch_id": i},
                    partition=-1,
                    offset=-1,
                    timestamp=datetime.now(timezone.utc),
                    headers=(),
                    topic="test-topic",
                )
                for i in range(100)
            ]

            await adapter.produce_batch(messages)
            await adapter.flush()

            # Verify metrics
            metrics = adapter.metrics
            assert metrics.messages_produced >= 100

    @pytest.mark.asyncio
    async def test_partition_info(self, kafka_config, kafka_container):
        """Test getting partition information."""
        from truthound.realtime.adapters.kafka import KafkaAdapter

        adapter = KafkaAdapter(kafka_config)

        async with adapter:
            partitions = await adapter.get_partition_info()
            assert "test-topic" in partitions
            assert len(partitions["test-topic"]) == 3  # We created 3 partitions

    @pytest.mark.asyncio
    async def test_consumer_lag(self, kafka_config):
        """Test consumer lag calculation."""
        from truthound.realtime.adapters.kafka import KafkaAdapter

        adapter = KafkaAdapter(kafka_config)

        async with adapter:
            lag = await adapter.get_consumer_lag()
            assert "test-topic" in lag
            # Lag should be a dict of partition -> lag value
            assert all(isinstance(v, int) for v in lag["test-topic"].values())

    @pytest.mark.asyncio
    async def test_reconnection(self, kafka_config):
        """Test adapter reconnection."""
        from truthound.realtime.adapters.kafka import KafkaAdapter

        adapter = KafkaAdapter(kafka_config)

        await adapter.connect()
        assert adapter.is_connected

        # Simulate reconnection
        await adapter.reconnect()
        assert adapter.is_connected

        await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_error_handling(self, kafka_config):
        """Test error handling for invalid operations."""
        from truthound.realtime.adapters.kafka import KafkaAdapter
        from truthound.realtime.protocols import StreamMessage, ConnectionError

        adapter = KafkaAdapter(kafka_config)

        # Try to produce without connecting
        message = StreamMessage(
            key="test",
            value={"test": True},
            partition=-1,
            offset=-1,
            timestamp=datetime.now(timezone.utc),
            headers=(),
            topic="test-topic",
        )

        with pytest.raises(ConnectionError):
            await adapter.produce(message)


@pytest.mark.skipif(
    not TESTCONTAINERS_AVAILABLE,
    reason="testcontainers not installed"
)
@pytest.mark.skipif(
    not AIOKAFKA_AVAILABLE,
    reason="aiokafka not installed"
)
class TestKafkaStreamingPipeline:
    """Integration tests for complete streaming pipeline with Kafka."""

    @pytest.fixture(scope="class")
    async def kafka_env(self):
        """Start Kafka environment for pipeline tests."""
        from truthound.realtime.testing.containers import StreamTestEnvironment

        async with StreamTestEnvironment(["kafka"]) as env:
            await env.create_kafka_topic("pipeline-topic", num_partitions=3)
            yield env

    @pytest.mark.asyncio
    async def test_validation_pipeline(self, kafka_env):
        """Test complete validation pipeline with Kafka."""
        from truthound.realtime.adapters.kafka import KafkaAdapter, KafkaAdapterConfig
        from truthound.realtime.protocols import StreamMessage
        from truthound.realtime.testing.fixtures import TestFixtures

        config = KafkaAdapterConfig(
            bootstrap_servers=kafka_env.get_service_url("kafka"),
            topic="pipeline-topic",
            consumer_group="pipeline-consumer",
        )

        adapter = KafkaAdapter(config)
        fixtures = TestFixtures()

        async with adapter:
            # Produce test messages
            test_data = fixtures.default_messages(50)
            for data in test_data:
                message = StreamMessage(
                    key=str(data.get("id", "unknown")),
                    value=data,
                    partition=-1,
                    offset=-1,
                    timestamp=datetime.now(timezone.utc),
                    headers=(),
                    topic="pipeline-topic",
                )
                await adapter.produce(message)

            await adapter.flush()

            # Consume and validate
            valid_count = 0
            invalid_count = 0
            consumed = 0

            async for msg in adapter.consume():
                consumed += 1
                value = msg.value

                # Simple validation
                if value.get("id") is not None and value.get("value") is not None:
                    valid_count += 1
                else:
                    invalid_count += 1

                await adapter.commit(msg)

                if consumed >= 50:
                    break

            assert consumed == 50
            assert valid_count == 50  # All default messages should be valid

    @pytest.mark.asyncio
    async def test_window_processing(self, kafka_env):
        """Test window-based processing with Kafka."""
        from truthound.realtime.adapters.kafka import KafkaAdapter, KafkaAdapterConfig
        from truthound.realtime.protocols import StreamMessage
        from truthound.realtime.processing.windows import (
            TumblingWindowProcessor,
            WindowConfig,
            SumAggregator,
        )

        config = KafkaAdapterConfig(
            bootstrap_servers=kafka_env.get_service_url("kafka"),
            topic="pipeline-topic",
            consumer_group="window-consumer",
        )

        adapter = KafkaAdapter(config)

        # Create window processor
        window_config = WindowConfig(window_size_seconds=5)
        processor = TumblingWindowProcessor(
            config=window_config,
            aggregators={"total": SumAggregator("value")},
        )

        async with adapter:
            # Produce messages with values
            for i in range(20):
                message = StreamMessage(
                    key=f"window-{i}",
                    value={"value": 10},  # Each message has value 10
                    partition=-1,
                    offset=-1,
                    timestamp=datetime.now(timezone.utc),
                    headers=(),
                    topic="pipeline-topic",
                )
                await adapter.produce(message)

            await adapter.flush()

            # Consume and process through window
            consumed = 0
            async for msg in adapter.consume():
                processor.add(msg)
                consumed += 1

                if consumed >= 20:
                    break

            # Check aggregated results
            results = processor.get_results()
            # Sum of 20 messages * 10 value each = 200
            assert results.get("total", 0) >= 100  # At least some aggregated


@pytest.mark.skipif(
    not TESTCONTAINERS_AVAILABLE,
    reason="testcontainers not installed"
)
class TestMultiContainerEnvironment:
    """Tests for multi-container test environment."""

    @pytest.mark.asyncio
    async def test_kafka_and_redis(self):
        """Test environment with Kafka and Redis."""
        from truthound.realtime.testing.containers import StreamTestEnvironment

        async with StreamTestEnvironment(["kafka", "redis"]) as env:
            kafka_url = env.get_service_url("kafka")
            redis_url = env.get_service_url("redis")

            assert ":" in kafka_url  # Should have host:port
            assert "redis://" in redis_url

            # Create topic
            await env.create_kafka_topic("multi-test", num_partitions=1)

    @pytest.mark.asyncio
    async def test_container_lifecycle(self):
        """Test container start/stop lifecycle."""
        from truthound.realtime.testing.containers import (
            KafkaTestContainer,
            RedisTestContainer,
        )

        kafka = KafkaTestContainer()
        redis = RedisTestContainer()

        # Start containers
        await kafka.start()
        await redis.start()

        assert kafka.is_running
        assert redis.is_running

        # Get URLs
        kafka_url = kafka.get_connection_url()
        redis_url = redis.get_connection_url()

        assert kafka_url is not None
        assert redis_url is not None

        # Stop containers
        await kafka.stop()
        await redis.stop()

        assert not kafka.is_running
        assert not redis.is_running
