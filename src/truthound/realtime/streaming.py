"""Streaming data source implementations.

Provides concrete implementations for various streaming platforms:
- Kafka
- Kinesis
- Google Pub/Sub
- Mock source for testing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator
import json
import threading
import random
import time
from datetime import datetime

import polars as pl

from truthound.realtime.base import (
    StreamingSource,
    StreamingConfig,
    StreamingError,
    ConnectionError,
)


# =============================================================================
# Mock Source for Testing
# =============================================================================


@dataclass
class MockStreamingConfig(StreamingConfig):
    """Configuration for mock streaming source.

    Attributes:
        schema: Schema for generated data
        num_batches: Number of batches to generate
        records_per_batch: Records per batch
        error_rate: Probability of generating bad data
        delay_ms: Delay between records
    """

    schema: dict[str, str] = field(default_factory=lambda: {
        "id": "int",
        "value": "float",
        "name": "str",
        "timestamp": "datetime",
    })
    num_batches: int = 10
    records_per_batch: int = 100
    error_rate: float = 0.1  # 10% bad records
    delay_ms: int = 0
    seed: int = 42


class MockStreamingSource(StreamingSource[MockStreamingConfig]):
    """Mock streaming source for testing.

    Generates synthetic data with configurable error rate
    for testing streaming validation pipelines.

    Example:
        >>> source = MockStreamingSource(
        ...     records_per_batch=100,
        ...     error_rate=0.1,
        ... )
        >>> with source:
        ...     for batch in source.read_batches(max_batches=5):
        ...         print(f"Batch: {len(batch)} records")
    """

    source_type = "mock"

    def __init__(self, config: MockStreamingConfig | None = None, **kwargs: Any):
        super().__init__(config, **kwargs)
        self._batch_num = 0
        self._rng = random.Random()

    def _default_config(self) -> MockStreamingConfig:
        return MockStreamingConfig()

    def connect(self) -> None:
        """Initialize the mock source."""
        self._rng = random.Random(self.config.seed)
        self._batch_num = 0
        self._connected = True

    def disconnect(self) -> None:
        """Disconnect from mock source."""
        self._connected = False

    def read_batch(self, max_records: int | None = None) -> pl.DataFrame:
        """Generate a batch of mock data.

        Args:
            max_records: Maximum records to generate

        Returns:
            DataFrame with generated data
        """
        if not self._connected:
            raise ConnectionError("Not connected to mock source")

        if self._batch_num >= self.config.num_batches:
            return pl.DataFrame()

        self._batch_num += 1
        num_records = max_records or self.config.records_per_batch

        # Apply delay if configured
        if self.config.delay_ms > 0:
            time.sleep(self.config.delay_ms / 1000)

        return self._generate_batch(num_records)

    def _generate_batch(self, num_records: int) -> pl.DataFrame:
        """Generate a batch of random data."""
        data: dict[str, list] = {col: [] for col in self.config.schema}

        for i in range(num_records):
            is_error = self._rng.random() < self.config.error_rate

            for col, dtype in self.config.schema.items():
                value = self._generate_value(col, dtype, is_error)
                data[col].append(value)

        return pl.DataFrame(data)

    def _generate_value(self, col: str, dtype: str, is_error: bool) -> Any:
        """Generate a single value."""
        if is_error and self._rng.random() < 0.5:
            return None  # Generate null as error

        if dtype == "int":
            if is_error and self._rng.random() < 0.3:
                return -999  # Invalid value
            return self._rng.randint(1, 1000)

        elif dtype == "float":
            if is_error and self._rng.random() < 0.3:
                return float("nan")
            return round(self._rng.uniform(0, 100), 2)

        elif dtype == "str":
            if is_error and self._rng.random() < 0.3:
                return ""  # Empty string
            names = ["Alice", "Bob", "Charlie", "David", "Eve"]
            return self._rng.choice(names)

        elif dtype == "datetime":
            return datetime.now()

        else:
            return None


# =============================================================================
# Kafka Source
# =============================================================================


@dataclass
class KafkaConfig(StreamingConfig):
    """Configuration for Kafka source.

    Attributes:
        bootstrap_servers: Kafka bootstrap servers
        topic: Topic to consume from
        group_id: Consumer group ID
        auto_offset_reset: Offset reset policy
        enable_auto_commit: Enable auto commit
    """

    bootstrap_servers: str = "localhost:9092"
    topic: str = ""
    group_id: str = "truthound-consumer"
    auto_offset_reset: str = "earliest"
    enable_auto_commit: bool = True
    security_protocol: str = "PLAINTEXT"
    sasl_mechanism: str | None = None
    sasl_username: str | None = None
    sasl_password: str | None = None
    ssl_cafile: str | None = None
    value_deserializer: str = "json"  # json, avro, string


class KafkaSource(StreamingSource[KafkaConfig]):
    """Kafka streaming source.

    Connects to Apache Kafka and reads messages as batches.
    Requires kafka-python package to be installed.

    Example:
        >>> source = KafkaSource(
        ...     bootstrap_servers="localhost:9092",
        ...     topic="events",
        ...     group_id="validator",
        ... )
        >>> with source:
        ...     for batch in source.read_batches():
        ...         process(batch)
    """

    source_type = "kafka"

    def __init__(self, config: KafkaConfig | None = None, **kwargs: Any):
        super().__init__(config, **kwargs)
        self._consumer = None

    def _default_config(self) -> KafkaConfig:
        return KafkaConfig()

    def connect(self) -> None:
        """Connect to Kafka broker."""
        try:
            from kafka import KafkaConsumer
        except ImportError:
            raise ImportError(
                "kafka-python is required for Kafka source. "
                "Install with: pip install kafka-python"
            )

        if not self.config.topic:
            raise ValueError("Kafka topic is required")

        kafka_config = {
            "bootstrap_servers": self.config.bootstrap_servers,
            "group_id": self.config.group_id,
            "auto_offset_reset": self.config.auto_offset_reset,
            "enable_auto_commit": self.config.enable_auto_commit,
            "security_protocol": self.config.security_protocol,
            "consumer_timeout_ms": self.config.batch_timeout_ms,
        }

        if self.config.sasl_mechanism:
            kafka_config["sasl_mechanism"] = self.config.sasl_mechanism
            kafka_config["sasl_plain_username"] = self.config.sasl_username
            kafka_config["sasl_plain_password"] = self.config.sasl_password

        if self.config.ssl_cafile:
            kafka_config["ssl_cafile"] = self.config.ssl_cafile

        # Value deserializer
        if self.config.value_deserializer == "json":
            kafka_config["value_deserializer"] = lambda m: json.loads(m.decode("utf-8"))
        else:
            kafka_config["value_deserializer"] = lambda m: m.decode("utf-8")

        try:
            self._consumer = KafkaConsumer(self.config.topic, **kafka_config)
            self._connected = True
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Kafka: {e}")

    def disconnect(self) -> None:
        """Disconnect from Kafka."""
        if self._consumer:
            self._consumer.close()
            self._consumer = None
        self._connected = False

    def read_batch(self, max_records: int | None = None) -> pl.DataFrame:
        """Read a batch of messages from Kafka.

        Args:
            max_records: Maximum records to read

        Returns:
            DataFrame with message data
        """
        if not self._connected or not self._consumer:
            raise ConnectionError("Not connected to Kafka")

        max_records = max_records or self.config.batch_size
        records = []

        try:
            # Poll for messages
            message_batch = self._consumer.poll(
                timeout_ms=self.config.batch_timeout_ms,
                max_records=max_records,
            )

            for topic_partition, messages in message_batch.items():
                for message in messages:
                    if isinstance(message.value, dict):
                        records.append(message.value)
                    else:
                        records.append({"value": message.value})

        except Exception as e:
            if self.config.error_handling == "fail":
                raise StreamingError(f"Error reading from Kafka: {e}")
            # Skip error

        if not records:
            return pl.DataFrame()

        return pl.DataFrame(records)

    def commit(self) -> None:
        """Commit current offsets."""
        if self._consumer:
            self._consumer.commit()

    def seek(self, position: Any) -> None:
        """Seek to a specific offset.

        Args:
            position: Dict of {partition: offset}
        """
        if not self._consumer:
            return

        from kafka import TopicPartition

        for partition, offset in position.items():
            tp = TopicPartition(self.config.topic, partition)
            self._consumer.seek(tp, offset)


# =============================================================================
# AWS Kinesis Source
# =============================================================================


@dataclass
class KinesisConfig(StreamingConfig):
    """Configuration for AWS Kinesis source.

    Attributes:
        stream_name: Kinesis stream name
        region_name: AWS region
        shard_iterator_type: Shard iterator type
    """

    stream_name: str = ""
    region_name: str = "us-east-1"
    shard_iterator_type: str = "LATEST"
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None


class KinesisSource(StreamingSource[KinesisConfig]):
    """AWS Kinesis streaming source.

    Connects to AWS Kinesis and reads records.
    Requires boto3 package to be installed.

    Example:
        >>> source = KinesisSource(
        ...     stream_name="my-stream",
        ...     region_name="us-west-2",
        ... )
        >>> with source:
        ...     for batch in source.read_batches():
        ...         process(batch)
    """

    source_type = "kinesis"

    def __init__(self, config: KinesisConfig | None = None, **kwargs: Any):
        super().__init__(config, **kwargs)
        self._client = None
        self._shard_iterators: dict[str, str] = {}

    def _default_config(self) -> KinesisConfig:
        return KinesisConfig()

    def connect(self) -> None:
        """Connect to Kinesis."""
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "boto3 is required for Kinesis source. "
                "Install with: pip install boto3"
            )

        if not self.config.stream_name:
            raise ValueError("Kinesis stream_name is required")

        boto_config = {
            "region_name": self.config.region_name,
        }

        if self.config.aws_access_key_id:
            boto_config["aws_access_key_id"] = self.config.aws_access_key_id
            boto_config["aws_secret_access_key"] = self.config.aws_secret_access_key

        try:
            self._client = boto3.client("kinesis", **boto_config)

            # Get shard iterators
            response = self._client.describe_stream(
                StreamName=self.config.stream_name
            )
            shards = response["StreamDescription"]["Shards"]

            for shard in shards:
                shard_id = shard["ShardId"]
                iter_response = self._client.get_shard_iterator(
                    StreamName=self.config.stream_name,
                    ShardId=shard_id,
                    ShardIteratorType=self.config.shard_iterator_type,
                )
                self._shard_iterators[shard_id] = iter_response["ShardIterator"]

            self._connected = True

        except Exception as e:
            raise ConnectionError(f"Failed to connect to Kinesis: {e}")

    def disconnect(self) -> None:
        """Disconnect from Kinesis."""
        self._client = None
        self._shard_iterators.clear()
        self._connected = False

    def read_batch(self, max_records: int | None = None) -> pl.DataFrame:
        """Read a batch of records from Kinesis.

        Args:
            max_records: Maximum records to read

        Returns:
            DataFrame with record data
        """
        if not self._connected or not self._client:
            raise ConnectionError("Not connected to Kinesis")

        max_records = max_records or self.config.batch_size
        all_records = []

        for shard_id, iterator in list(self._shard_iterators.items()):
            try:
                response = self._client.get_records(
                    ShardIterator=iterator,
                    Limit=max_records,
                )

                records = response.get("Records", [])
                for record in records:
                    data = record["Data"]
                    if isinstance(data, bytes):
                        try:
                            data = json.loads(data.decode("utf-8"))
                        except Exception:
                            data = {"value": data.decode("utf-8")}
                    all_records.append(data)

                # Update iterator
                next_iterator = response.get("NextShardIterator")
                if next_iterator:
                    self._shard_iterators[shard_id] = next_iterator

            except Exception as e:
                if self.config.error_handling == "fail":
                    raise StreamingError(f"Error reading from Kinesis: {e}")

        if not all_records:
            return pl.DataFrame()

        return pl.DataFrame(all_records)


# =============================================================================
# Google Pub/Sub Source
# =============================================================================


@dataclass
class PubSubConfig(StreamingConfig):
    """Configuration for Google Pub/Sub source.

    Attributes:
        project_id: GCP project ID
        subscription_id: Pub/Sub subscription ID
        credentials_path: Path to service account credentials
    """

    project_id: str = ""
    subscription_id: str = ""
    credentials_path: str | None = None
    ack_deadline_seconds: int = 60


class PubSubSource(StreamingSource[PubSubConfig]):
    """Google Cloud Pub/Sub streaming source.

    Connects to GCP Pub/Sub and reads messages.
    Requires google-cloud-pubsub package.

    Example:
        >>> source = PubSubSource(
        ...     project_id="my-project",
        ...     subscription_id="my-subscription",
        ... )
        >>> with source:
        ...     for batch in source.read_batches():
        ...         process(batch)
    """

    source_type = "pubsub"

    def __init__(self, config: PubSubConfig | None = None, **kwargs: Any):
        super().__init__(config, **kwargs)
        self._subscriber = None
        self._subscription_path = None

    def _default_config(self) -> PubSubConfig:
        return PubSubConfig()

    def connect(self) -> None:
        """Connect to Pub/Sub."""
        try:
            from google.cloud import pubsub_v1
            from google.oauth2 import service_account
        except ImportError:
            raise ImportError(
                "google-cloud-pubsub is required for Pub/Sub source. "
                "Install with: pip install google-cloud-pubsub"
            )

        if not self.config.project_id or not self.config.subscription_id:
            raise ValueError("project_id and subscription_id are required")

        try:
            if self.config.credentials_path:
                credentials = service_account.Credentials.from_service_account_file(
                    self.config.credentials_path
                )
                self._subscriber = pubsub_v1.SubscriberClient(credentials=credentials)
            else:
                self._subscriber = pubsub_v1.SubscriberClient()

            self._subscription_path = self._subscriber.subscription_path(
                self.config.project_id,
                self.config.subscription_id,
            )

            self._connected = True

        except Exception as e:
            raise ConnectionError(f"Failed to connect to Pub/Sub: {e}")

    def disconnect(self) -> None:
        """Disconnect from Pub/Sub."""
        if self._subscriber:
            self._subscriber.close()
            self._subscriber = None
        self._connected = False

    def read_batch(self, max_records: int | None = None) -> pl.DataFrame:
        """Read a batch of messages from Pub/Sub.

        Args:
            max_records: Maximum records to read

        Returns:
            DataFrame with message data
        """
        if not self._connected or not self._subscriber:
            raise ConnectionError("Not connected to Pub/Sub")

        max_records = max_records or self.config.batch_size
        records = []
        ack_ids = []

        try:
            response = self._subscriber.pull(
                subscription=self._subscription_path,
                max_messages=max_records,
                timeout=self.config.batch_timeout_ms / 1000,
            )

            for msg in response.received_messages:
                data = msg.message.data
                if isinstance(data, bytes):
                    try:
                        data = json.loads(data.decode("utf-8"))
                    except Exception:
                        data = {"value": data.decode("utf-8")}
                records.append(data)
                ack_ids.append(msg.ack_id)

            # Acknowledge messages
            if ack_ids:
                self._subscriber.acknowledge(
                    subscription=self._subscription_path,
                    ack_ids=ack_ids,
                )

        except Exception as e:
            if self.config.error_handling == "fail":
                raise StreamingError(f"Error reading from Pub/Sub: {e}")

        if not records:
            return pl.DataFrame()

        return pl.DataFrame(records)
