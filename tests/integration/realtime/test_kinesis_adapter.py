"""Integration tests for Kinesis adapter using LocalStack.

These tests require Docker to be running and testcontainers installed.
Run with: pytest tests/integration/realtime/test_kinesis_adapter.py -v
"""

from __future__ import annotations

import asyncio
import pytest
from datetime import datetime, timezone

pytest_plugins = ["pytest_asyncio"]

try:
    import testcontainers
    TESTCONTAINERS_AVAILABLE = True
except ImportError:
    TESTCONTAINERS_AVAILABLE = False

try:
    import aiobotocore
    AIOBOTOCORE_AVAILABLE = True
except ImportError:
    AIOBOTOCORE_AVAILABLE = False


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
    not AIOBOTOCORE_AVAILABLE,
    reason="aiobotocore not installed"
)
class TestKinesisAdapterIntegration:
    """Integration tests for KinesisAdapter with LocalStack."""

    @pytest.fixture(scope="class")
    async def localstack_container(self):
        """Start LocalStack container for tests."""
        from truthound.realtime.testing.containers import LocalStackTestContainer

        async with LocalStackTestContainer() as localstack:
            # Create test stream
            await localstack.create_kinesis_stream("test-stream", shard_count=2)
            yield localstack

    @pytest.fixture
    def kinesis_config(self, localstack_container):
        """Get Kinesis adapter configuration."""
        from truthound.realtime.adapters.kinesis import KinesisAdapterConfig

        return KinesisAdapterConfig(
            stream_name="test-stream",
            region_name="us-east-1",
            endpoint_url=localstack_container.get_connection_url(),
            aws_access_key_id="test",
            aws_secret_access_key="test",
        )

    @pytest.mark.asyncio
    async def test_connect_and_disconnect(self, kinesis_config):
        """Test connecting and disconnecting from Kinesis."""
        from truthound.realtime.adapters.kinesis import KinesisAdapter

        adapter = KinesisAdapter(kinesis_config)

        # Connect
        await adapter.connect()
        assert adapter.is_connected

        # Disconnect
        await adapter.disconnect()
        assert not adapter.is_connected

    @pytest.mark.asyncio
    async def test_produce_and_consume(self, kinesis_config):
        """Test producing and consuming messages."""
        from truthound.realtime.adapters.kinesis import KinesisAdapter
        from truthound.realtime.protocols import StreamMessage

        adapter = KinesisAdapter(kinesis_config)

        async with adapter:
            # Produce messages
            for i in range(10):
                message = StreamMessage(
                    key=f"partition-key-{i % 3}",  # Distribute across shards
                    value={"id": i, "data": f"kinesis-test-{i}"},
                    partition=-1,
                    offset=-1,
                    timestamp=datetime.now(timezone.utc),
                    headers=(),
                    topic="test-stream",
                )
                await adapter.produce(message)

            await adapter.flush()

            # Wait for messages to be available
            await asyncio.sleep(2)

            # Consume messages
            consumed = []
            timeout = asyncio.get_event_loop().time() + 30  # 30 second timeout

            async for msg in adapter.consume():
                consumed.append(msg)

                if len(consumed) >= 10:
                    break
                if asyncio.get_event_loop().time() > timeout:
                    break

            assert len(consumed) >= 1  # At least some messages consumed

    @pytest.mark.asyncio
    async def test_batch_produce(self, kinesis_config):
        """Test batch message production."""
        from truthound.realtime.adapters.kinesis import KinesisAdapter
        from truthound.realtime.protocols import StreamMessage

        adapter = KinesisAdapter(kinesis_config)

        async with adapter:
            messages = [
                StreamMessage(
                    key=f"batch-pk-{i % 5}",
                    value={"batch_id": i, "data": f"batch-{i}"},
                    partition=-1,
                    offset=-1,
                    timestamp=datetime.now(timezone.utc),
                    headers=(),
                    topic="test-stream",
                )
                for i in range(50)
            ]

            await adapter.produce_batch(messages)
            await adapter.flush()

            # Verify metrics
            metrics = adapter.metrics
            assert metrics.messages_produced >= 50

    @pytest.mark.asyncio
    async def test_shard_info(self, kinesis_config):
        """Test getting shard information."""
        from truthound.realtime.adapters.kinesis import KinesisAdapter

        adapter = KinesisAdapter(kinesis_config)

        async with adapter:
            shards = await adapter.get_shard_info()
            assert len(shards) == 2  # We created 2 shards

    @pytest.mark.asyncio
    async def test_stream_description(self, kinesis_config):
        """Test getting stream description."""
        from truthound.realtime.adapters.kinesis import KinesisAdapter

        adapter = KinesisAdapter(kinesis_config)

        async with adapter:
            description = await adapter.get_stream_description()
            assert description["StreamName"] == "test-stream"
            assert description["StreamStatus"] in ["ACTIVE", "CREATING"]

    @pytest.mark.asyncio
    async def test_reconnection(self, kinesis_config):
        """Test adapter reconnection."""
        from truthound.realtime.adapters.kinesis import KinesisAdapter

        adapter = KinesisAdapter(kinesis_config)

        await adapter.connect()
        assert adapter.is_connected

        # Simulate reconnection
        await adapter.reconnect()
        assert adapter.is_connected

        await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_error_handling(self, kinesis_config):
        """Test error handling for invalid operations."""
        from truthound.realtime.adapters.kinesis import KinesisAdapter
        from truthound.realtime.protocols import StreamMessage, ConnectionError

        adapter = KinesisAdapter(kinesis_config)

        # Try to produce without connecting
        message = StreamMessage(
            key="test-pk",
            value={"test": True},
            partition=-1,
            offset=-1,
            timestamp=datetime.now(timezone.utc),
            headers=(),
            topic="test-stream",
        )

        with pytest.raises(ConnectionError):
            await adapter.produce(message)


@pytest.mark.skipif(
    not TESTCONTAINERS_AVAILABLE,
    reason="testcontainers not installed"
)
@pytest.mark.skipif(
    not AIOBOTOCORE_AVAILABLE,
    reason="aiobotocore not installed"
)
class TestKinesisStreamingPipeline:
    """Integration tests for complete streaming pipeline with Kinesis."""

    @pytest.fixture(scope="class")
    async def kinesis_env(self):
        """Start Kinesis environment for pipeline tests."""
        from truthound.realtime.testing.containers import StreamTestEnvironment

        async with StreamTestEnvironment(["kinesis"]) as env:
            await env.create_kinesis_stream("pipeline-stream", shard_count=2)
            yield env

    @pytest.mark.asyncio
    async def test_validation_pipeline(self, kinesis_env):
        """Test complete validation pipeline with Kinesis."""
        from truthound.realtime.adapters.kinesis import KinesisAdapter, KinesisAdapterConfig
        from truthound.realtime.protocols import StreamMessage
        from truthound.realtime.testing.fixtures import TestFixtures

        config = KinesisAdapterConfig(
            stream_name="pipeline-stream",
            region_name="us-east-1",
            endpoint_url=kinesis_env.get_service_url("kinesis"),
            aws_access_key_id="test",
            aws_secret_access_key="test",
        )

        adapter = KinesisAdapter(config)
        fixtures = TestFixtures()

        async with adapter:
            # Produce test messages
            test_data = fixtures.default_messages(20)
            for i, data in enumerate(test_data):
                message = StreamMessage(
                    key=f"pk-{i % 5}",
                    value=data,
                    partition=-1,
                    offset=-1,
                    timestamp=datetime.now(timezone.utc),
                    headers=(),
                    topic="pipeline-stream",
                )
                await adapter.produce(message)

            await adapter.flush()

            # Wait for messages
            await asyncio.sleep(3)

            # Consume and validate
            consumed = 0
            timeout = asyncio.get_event_loop().time() + 30

            async for msg in adapter.consume():
                consumed += 1
                value = msg.value

                # Basic validation check
                assert "id" in value or "value" in value

                if consumed >= 20 or asyncio.get_event_loop().time() > timeout:
                    break

            assert consumed >= 1  # At least some messages processed


@pytest.mark.skipif(
    not TESTCONTAINERS_AVAILABLE,
    reason="testcontainers not installed"
)
class TestLocalStackContainer:
    """Tests for LocalStack container management."""

    @pytest.mark.asyncio
    async def test_container_lifecycle(self):
        """Test LocalStack container start/stop."""
        from truthound.realtime.testing.containers import LocalStackTestContainer

        container = LocalStackTestContainer()

        # Start
        await container.start()
        assert container.is_running

        # Get URL
        url = container.get_connection_url()
        assert "http://" in url

        # Stop
        await container.stop()
        assert not container.is_running

    @pytest.mark.asyncio
    async def test_create_multiple_streams(self):
        """Test creating multiple Kinesis streams."""
        from truthound.realtime.testing.containers import LocalStackTestContainer

        async with LocalStackTestContainer() as container:
            # Create multiple streams
            await container.create_kinesis_stream("stream-1", shard_count=1)
            await container.create_kinesis_stream("stream-2", shard_count=2)
            await container.create_kinesis_stream("stream-3", shard_count=3)

            # All streams should be created successfully
            # (no exceptions thrown)
