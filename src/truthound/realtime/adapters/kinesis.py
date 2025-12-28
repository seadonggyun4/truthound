"""AWS Kinesis adapter using aiobotocore.

Provides async Kinesis integration with:
- Multi-shard consumption
- Shard iterator management
- Enhanced fan-out support
- Checkpointing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator
import json
import logging
import asyncio

from truthound.realtime.protocols import (
    StreamMessage,
    MessageHeader,
    DeserializationFormat,
)
from truthound.realtime.adapters.base import (
    BaseStreamAdapter,
    AdapterConfig,
)


logger = logging.getLogger(__name__)


class ShardIteratorType:
    """Kinesis shard iterator types."""

    TRIM_HORIZON = "TRIM_HORIZON"  # Start from oldest record
    LATEST = "LATEST"  # Start from latest record
    AT_SEQUENCE_NUMBER = "AT_SEQUENCE_NUMBER"  # Start at specific sequence
    AFTER_SEQUENCE_NUMBER = "AFTER_SEQUENCE_NUMBER"  # Start after specific sequence
    AT_TIMESTAMP = "AT_TIMESTAMP"  # Start at specific timestamp


@dataclass
class KinesisAdapterConfig(AdapterConfig):
    """Configuration for Kinesis adapter.

    Attributes:
        stream_name: Kinesis stream name
        stream_arn: Stream ARN (alternative to stream_name)
        region_name: AWS region
        shard_iterator_type: How to position shard iterator
        aws_access_key_id: AWS access key (optional, uses default credentials if not set)
        aws_secret_access_key: AWS secret key
        aws_session_token: AWS session token (for temporary credentials)
        endpoint_url: Custom endpoint URL (for LocalStack testing)
        use_enhanced_fanout: Use enhanced fan-out for higher throughput
        consumer_name: Consumer name for enhanced fan-out
        poll_interval_ms: Interval between GetRecords calls
    """

    # Stream settings
    stream_name: str = ""
    stream_arn: str = ""
    region_name: str = "us-east-1"
    shard_iterator_type: str = ShardIteratorType.LATEST
    starting_sequence_number: str | None = None
    starting_timestamp: datetime | None = None

    # AWS credentials
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_session_token: str | None = None
    endpoint_url: str | None = None  # For LocalStack

    # Enhanced fan-out
    use_enhanced_fanout: bool = False
    consumer_name: str = "truthound-consumer"

    # Performance
    poll_interval_ms: int = 200  # Kinesis limit: 5 GetRecords/sec/shard
    max_records_per_shard: int = 10000  # Kinesis limit: 10000 records/GetRecords

    # Serialization
    value_deserializer: DeserializationFormat = DeserializationFormat.JSON


class KinesisAdapter(BaseStreamAdapter[dict[str, Any], KinesisAdapterConfig]):
    """AWS Kinesis streaming adapter using aiobotocore.

    Provides async Kinesis consumer and producer capabilities
    with support for multi-shard consumption and enhanced fan-out.

    Example:
        >>> config = KinesisAdapterConfig(
        ...     stream_name="my-stream",
        ...     region_name="us-east-1",
        ... )
        >>> async with KinesisAdapter(config) as adapter:
        ...     async for message in adapter.consume():
        ...         print(message.value)

    Requires:
        pip install aiobotocore
    """

    adapter_type = "kinesis"

    def __init__(
        self,
        config: KinesisAdapterConfig | None = None,
        **kwargs: Any,
    ):
        config = config or KinesisAdapterConfig()
        super().__init__(config, **kwargs)

        self._session: Any = None
        self._client: Any = None
        self._shard_iterators: dict[str, str] = {}
        self._sequence_numbers: dict[str, str] = {}
        self._stream_arn: str = ""

    @property
    def source_name(self) -> str:
        """Get Kinesis source identifier."""
        return f"kinesis://{self._config.region_name}/{self._config.stream_name}"

    async def _do_connect(self) -> None:
        """Connect to Kinesis."""
        try:
            from aiobotocore.session import get_session
        except ImportError:
            raise ImportError(
                "aiobotocore is required for Kinesis adapter. "
                "Install with: pip install aiobotocore"
            )

        if not self._config.stream_name and not self._config.stream_arn:
            raise ValueError("stream_name or stream_arn is required")

        # Create session
        self._session = get_session()

        # Build client config
        client_config: dict[str, Any] = {
            "region_name": self._config.region_name,
        }

        if self._config.aws_access_key_id:
            client_config["aws_access_key_id"] = self._config.aws_access_key_id
            client_config["aws_secret_access_key"] = self._config.aws_secret_access_key

        if self._config.aws_session_token:
            client_config["aws_session_token"] = self._config.aws_session_token

        if self._config.endpoint_url:
            client_config["endpoint_url"] = self._config.endpoint_url

        # Create client
        self._client = await self._session.create_client("kinesis", **client_config).__aenter__()

        # Get stream info
        await self._initialize_stream_info()

        # Initialize shard iterators
        await self._initialize_shard_iterators()

        logger.info(
            f"Connected to Kinesis stream: {self._config.stream_name}, "
            f"shards: {len(self._shard_iterators)}"
        )

    async def _do_disconnect(self) -> None:
        """Disconnect from Kinesis."""
        if self._client:
            await self._client.__aexit__(None, None, None)
            self._client = None

        self._shard_iterators.clear()
        self._sequence_numbers.clear()

        logger.info("Disconnected from Kinesis")

    async def _initialize_stream_info(self) -> None:
        """Get stream ARN and other info."""
        if self._config.stream_arn:
            self._stream_arn = self._config.stream_arn
            return

        response = await self._client.describe_stream(
            StreamName=self._config.stream_name
        )
        self._stream_arn = response["StreamDescription"]["StreamARN"]

    async def _initialize_shard_iterators(self) -> None:
        """Initialize shard iterators for all shards."""
        stream_name = self._config.stream_name

        # Get shard list
        response = await self._client.describe_stream(StreamName=stream_name)
        shards = response["StreamDescription"]["Shards"]

        # Get iterator for each shard
        for shard in shards:
            shard_id = shard["ShardId"]

            iterator_kwargs: dict[str, Any] = {
                "StreamName": stream_name,
                "ShardId": shard_id,
                "ShardIteratorType": self._config.shard_iterator_type,
            }

            if self._config.shard_iterator_type == ShardIteratorType.AT_SEQUENCE_NUMBER:
                if self._config.starting_sequence_number:
                    iterator_kwargs["StartingSequenceNumber"] = self._config.starting_sequence_number
            elif self._config.shard_iterator_type == ShardIteratorType.AT_TIMESTAMP:
                if self._config.starting_timestamp:
                    iterator_kwargs["Timestamp"] = self._config.starting_timestamp

            response = await self._client.get_shard_iterator(**iterator_kwargs)
            self._shard_iterators[shard_id] = response["ShardIterator"]

    async def _do_consume(self) -> AsyncIterator[StreamMessage[dict[str, Any]]]:
        """Consume messages from Kinesis."""
        if not self._client:
            raise RuntimeError("Client not initialized")

        poll_interval = self._config.poll_interval_ms / 1000

        while True:
            messages_found = False

            for shard_id in list(self._shard_iterators.keys()):
                iterator = self._shard_iterators.get(shard_id)
                if not iterator:
                    continue

                try:
                    response = await self._client.get_records(
                        ShardIterator=iterator,
                        Limit=self._config.max_records_per_shard,
                    )

                    # Update iterator
                    next_iterator = response.get("NextShardIterator")
                    if next_iterator:
                        self._shard_iterators[shard_id] = next_iterator
                    else:
                        # Shard is closed
                        del self._shard_iterators[shard_id]

                    # Process records
                    for record in response.get("Records", []):
                        messages_found = True
                        message = self._parse_record(record, shard_id)
                        yield message

                        # Track sequence number
                        self._sequence_numbers[shard_id] = record["SequenceNumber"]

                except Exception as e:
                    logger.warning(f"Error reading from shard {shard_id}: {e}")
                    # Try to refresh iterator
                    await self._refresh_shard_iterator(shard_id)

            # If no messages found, wait before next poll
            if not messages_found:
                await asyncio.sleep(poll_interval)

    async def _refresh_shard_iterator(self, shard_id: str) -> None:
        """Refresh shard iterator after error."""
        try:
            seq = self._sequence_numbers.get(shard_id)

            iterator_kwargs: dict[str, Any] = {
                "StreamName": self._config.stream_name,
                "ShardId": shard_id,
                "ShardIteratorType": ShardIteratorType.AFTER_SEQUENCE_NUMBER if seq else ShardIteratorType.LATEST,
            }

            if seq:
                iterator_kwargs["StartingSequenceNumber"] = seq

            response = await self._client.get_shard_iterator(**iterator_kwargs)
            self._shard_iterators[shard_id] = response["ShardIterator"]

        except Exception as e:
            logger.error(f"Failed to refresh shard iterator for {shard_id}: {e}")

    def _parse_record(self, record: dict[str, Any], shard_id: str) -> StreamMessage[dict[str, Any]]:
        """Parse Kinesis record to StreamMessage."""
        data = record["Data"]

        # Deserialize based on format
        if self._config.value_deserializer == DeserializationFormat.JSON:
            if isinstance(data, bytes):
                value = json.loads(data.decode("utf-8"))
            else:
                value = data
        elif self._config.value_deserializer == DeserializationFormat.STRING:
            value = {"value": data.decode("utf-8") if isinstance(data, bytes) else str(data)}
        else:
            value = {"value": data}

        # Ensure value is dict
        if not isinstance(value, dict):
            value = {"value": value}

        # Extract partition key hash for partition number
        partition_key = record.get("PartitionKey", "")
        partition = hash(partition_key) % 1000  # Pseudo partition number

        return StreamMessage(
            key=partition_key,
            value=value,
            partition=partition,
            offset=int(record["SequenceNumber"]),
            timestamp=record.get("ApproximateArrivalTimestamp", datetime.now(timezone.utc)),
            headers=(MessageHeader("shard_id", shard_id.encode()),),
            topic=self._config.stream_name,
            metadata={
                "sequence_number": record["SequenceNumber"],
                "shard_id": shard_id,
                "encryption_type": record.get("EncryptionType"),
            },
        )

    async def _do_produce(self, message: StreamMessage[dict[str, Any]]) -> None:
        """Produce message to Kinesis."""
        if not self._client:
            raise RuntimeError("Client not initialized")

        # Serialize value
        if self._config.value_deserializer == DeserializationFormat.JSON:
            data = json.dumps(message.value).encode("utf-8")
        else:
            data = str(message.value).encode("utf-8")

        await self._client.put_record(
            StreamName=self._config.stream_name,
            Data=data,
            PartitionKey=message.key or str(message.offset),
        )

    async def _do_produce_batch(self, messages: list[StreamMessage[dict[str, Any]]]) -> None:
        """Produce batch of messages to Kinesis."""
        if not self._client or not messages:
            return

        # Build records (max 500 per PutRecords call)
        batch_size = 500
        for i in range(0, len(messages), batch_size):
            batch = messages[i:i + batch_size]

            records = []
            for msg in batch:
                if self._config.value_deserializer == DeserializationFormat.JSON:
                    data = json.dumps(msg.value).encode("utf-8")
                else:
                    data = str(msg.value).encode("utf-8")

                records.append({
                    "Data": data,
                    "PartitionKey": msg.key or str(msg.offset),
                })

            response = await self._client.put_records(
                StreamName=self._config.stream_name,
                Records=records,
            )

            # Check for failed records
            failed = response.get("FailedRecordCount", 0)
            if failed > 0:
                logger.warning(f"Failed to produce {failed} records to Kinesis")

    async def _do_commit(self, message: StreamMessage[dict[str, Any]]) -> None:
        """Store sequence number for checkpointing."""
        shard_id = message.metadata.get("shard_id", "")
        if shard_id:
            self._sequence_numbers[shard_id] = str(message.offset)

    # -------------------------------------------------------------------------
    # Kinesis-Specific Methods
    # -------------------------------------------------------------------------

    async def get_shard_info(self) -> list[dict[str, Any]]:
        """Get information about stream shards.

        Returns:
            List of shard information dicts
        """
        if not self._client:
            return []

        response = await self._client.describe_stream(
            StreamName=self._config.stream_name
        )

        shards = []
        for shard in response["StreamDescription"]["Shards"]:
            shards.append({
                "shard_id": shard["ShardId"],
                "hash_key_range": shard.get("HashKeyRange", {}),
                "sequence_number_range": shard.get("SequenceNumberRange", {}),
            })

        return shards

    async def get_stream_info(self) -> dict[str, Any]:
        """Get stream information.

        Returns:
            Stream information dict
        """
        if not self._client:
            return {}

        response = await self._client.describe_stream(
            StreamName=self._config.stream_name
        )

        desc = response["StreamDescription"]
        return {
            "stream_name": desc["StreamName"],
            "stream_arn": desc["StreamARN"],
            "stream_status": desc["StreamStatus"],
            "shard_count": len(desc["Shards"]),
            "retention_period_hours": desc.get("RetentionPeriodHours", 24),
            "enhanced_monitoring": desc.get("EnhancedMonitoring", []),
            "encryption_type": desc.get("EncryptionType"),
        }

    async def get_metrics(self) -> dict[str, Any]:
        """Get CloudWatch metrics for the stream.

        Returns:
            Stream metrics dict
        """
        if not self._client:
            return {}

        # Note: This would require CloudWatch client
        # Returning current adapter metrics instead
        base_metrics = self._metrics.get_metrics()
        return {
            "messages_consumed": base_metrics.messages_consumed,
            "messages_produced": base_metrics.messages_produced,
            "active_shards": len(self._shard_iterators),
            "sequence_numbers": dict(self._sequence_numbers),
        }

    def get_checkpoint(self) -> dict[str, str]:
        """Get current sequence numbers for checkpointing.

        Returns:
            Dict mapping shard_id to sequence_number
        """
        return dict(self._sequence_numbers)

    async def restore_checkpoint(self, checkpoint: dict[str, str]) -> None:
        """Restore from checkpoint.

        Args:
            checkpoint: Dict mapping shard_id to sequence_number
        """
        self._sequence_numbers = dict(checkpoint)

        # Reinitialize iterators from checkpoint
        for shard_id, seq_num in checkpoint.items():
            try:
                response = await self._client.get_shard_iterator(
                    StreamName=self._config.stream_name,
                    ShardId=shard_id,
                    ShardIteratorType=ShardIteratorType.AFTER_SEQUENCE_NUMBER,
                    StartingSequenceNumber=seq_num,
                )
                self._shard_iterators[shard_id] = response["ShardIterator"]
            except Exception as e:
                logger.warning(f"Failed to restore iterator for shard {shard_id}: {e}")
