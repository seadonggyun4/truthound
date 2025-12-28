"""Stream adapter factory with dependency injection.

Provides runtime adapter selection and configuration with
support for dependency injection of state stores and metrics collectors.
"""

from __future__ import annotations

from typing import Any, TypeVar, Generic
import logging

from truthound.realtime.protocols import (
    IStreamSource,
    IStreamSink,
    IStateStore,
    IMetricsCollector,
    StreamSourceConfig,
)
from truthound.realtime.adapters.base import (
    BaseStreamAdapter,
    AdapterConfig,
    InMemoryStateStore,
    DefaultMetricsCollector,
    NoOpMetricsCollector,
)


logger = logging.getLogger(__name__)

T = TypeVar("T")


class StreamAdapterFactory:
    """Factory for creating stream adapters.

    Provides runtime adapter selection based on adapter type
    with support for dependency injection.

    Example:
        >>> # Register custom adapter
        >>> @StreamAdapterFactory.register("custom")
        ... class CustomAdapter(BaseStreamAdapter):
        ...     pass
        >>>
        >>> # Create adapter
        >>> adapter = StreamAdapterFactory.create(
        ...     "kafka",
        ...     config=KafkaAdapterConfig(topic="events"),
        ...     state_store=RedisStateStore(...),
        ... )
    """

    _registry: dict[str, type[BaseStreamAdapter]] = {}

    @classmethod
    def register(cls, name: str):
        """Register an adapter class.

        Args:
            name: Adapter type name

        Returns:
            Decorator function

        Example:
            >>> @StreamAdapterFactory.register("pulsar")
            ... class PulsarAdapter(BaseStreamAdapter):
            ...     pass
        """
        def decorator(adapter_cls: type[BaseStreamAdapter]) -> type[BaseStreamAdapter]:
            cls._registry[name.lower()] = adapter_cls
            logger.debug(f"Registered stream adapter: {name}")
            return adapter_cls
        return decorator

    @classmethod
    def unregister(cls, name: str) -> bool:
        """Unregister an adapter.

        Args:
            name: Adapter type name

        Returns:
            True if unregistered, False if not found
        """
        name = name.lower()
        if name in cls._registry:
            del cls._registry[name]
            return True
        return False

    @classmethod
    def create(
        cls,
        adapter_type: str,
        config: AdapterConfig | dict[str, Any] | None = None,
        *,
        state_store: IStateStore | None = None,
        metrics_collector: IMetricsCollector | None = None,
        enable_metrics: bool = True,
        **kwargs: Any,
    ) -> BaseStreamAdapter:
        """Create a stream adapter.

        Args:
            adapter_type: Type of adapter (kafka, kinesis, mock, etc.)
            config: Adapter configuration (dict or config object)
            state_store: Optional state store for stateful processing
            metrics_collector: Optional metrics collector
            enable_metrics: Enable metrics collection (default True)
            **kwargs: Additional configuration parameters

        Returns:
            Configured stream adapter

        Raises:
            ValueError: If adapter type is unknown
        """
        adapter_type = adapter_type.lower()

        # Auto-register built-in adapters
        cls._ensure_builtin_adapters()

        if adapter_type not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys()))
            raise ValueError(
                f"Unknown adapter type: {adapter_type}. "
                f"Available types: {available}"
            )

        adapter_cls = cls._registry[adapter_type]

        # Build config
        if config is None:
            config = cls._get_default_config(adapter_type)
        elif isinstance(config, dict):
            config = cls._build_config(adapter_type, config)

        # Apply kwargs to config
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        # Setup state store
        if state_store is None:
            state_store = InMemoryStateStore()

        # Setup metrics collector
        if metrics_collector is None:
            if enable_metrics:
                metrics_collector = DefaultMetricsCollector()
            else:
                metrics_collector = NoOpMetricsCollector()

        return adapter_cls(
            config=config,
            state_store=state_store,
            metrics_collector=metrics_collector,
        )

    @classmethod
    def create_from_url(
        cls,
        url: str,
        *,
        state_store: IStateStore | None = None,
        metrics_collector: IMetricsCollector | None = None,
        **kwargs: Any,
    ) -> BaseStreamAdapter:
        """Create adapter from connection URL.

        Args:
            url: Connection URL (e.g., kafka://localhost:9092/topic)
            state_store: Optional state store
            metrics_collector: Optional metrics collector
            **kwargs: Additional configuration

        Returns:
            Configured stream adapter

        Example:
            >>> adapter = StreamAdapterFactory.create_from_url(
            ...     "kafka://broker1:9092,broker2:9092/events?group_id=mygroup"
            ... )
        """
        from urllib.parse import urlparse, parse_qs

        parsed = urlparse(url)
        adapter_type = parsed.scheme

        # Build config from URL components
        config_dict: dict[str, Any] = {}

        if adapter_type == "kafka":
            config_dict["bootstrap_servers"] = parsed.netloc
            if parsed.path and parsed.path != "/":
                config_dict["topic"] = parsed.path.lstrip("/")

        elif adapter_type == "kinesis":
            config_dict["region_name"] = parsed.netloc
            if parsed.path and parsed.path != "/":
                config_dict["stream_name"] = parsed.path.lstrip("/")

        elif adapter_type == "mock":
            # Parse mock-specific settings from query
            pass

        # Parse query parameters
        query_params = parse_qs(parsed.query)
        for key, values in query_params.items():
            config_dict[key] = values[0] if len(values) == 1 else values

        # Merge with kwargs
        config_dict.update(kwargs)

        return cls.create(
            adapter_type,
            config=config_dict,
            state_store=state_store,
            metrics_collector=metrics_collector,
        )

    @classmethod
    def list_adapters(cls) -> list[str]:
        """List all registered adapter types.

        Returns:
            List of adapter type names
        """
        cls._ensure_builtin_adapters()
        return sorted(cls._registry.keys())

    @classmethod
    def get_adapter_class(cls, adapter_type: str) -> type[BaseStreamAdapter] | None:
        """Get adapter class by type.

        Args:
            adapter_type: Adapter type name

        Returns:
            Adapter class or None if not found
        """
        cls._ensure_builtin_adapters()
        return cls._registry.get(adapter_type.lower())

    @classmethod
    def _ensure_builtin_adapters(cls) -> None:
        """Ensure built-in adapters are registered."""
        if cls._registry:
            return

        # Import and register built-in adapters
        from truthound.realtime.adapters.mock import MockAdapter
        from truthound.realtime.adapters.kafka import KafkaAdapter
        from truthound.realtime.adapters.kinesis import KinesisAdapter

        cls._registry["mock"] = MockAdapter
        cls._registry["kafka"] = KafkaAdapter
        cls._registry["kinesis"] = KinesisAdapter

    @classmethod
    def _get_default_config(cls, adapter_type: str) -> AdapterConfig:
        """Get default config for adapter type."""
        from truthound.realtime.adapters.mock import MockAdapterConfig
        from truthound.realtime.adapters.kafka import KafkaAdapterConfig
        from truthound.realtime.adapters.kinesis import KinesisAdapterConfig

        config_map = {
            "mock": MockAdapterConfig,
            "kafka": KafkaAdapterConfig,
            "kinesis": KinesisAdapterConfig,
        }

        config_cls = config_map.get(adapter_type, AdapterConfig)
        return config_cls()

    @classmethod
    def _build_config(
        cls,
        adapter_type: str,
        config_dict: dict[str, Any],
    ) -> AdapterConfig:
        """Build config object from dict."""
        from truthound.realtime.adapters.mock import MockAdapterConfig
        from truthound.realtime.adapters.kafka import KafkaAdapterConfig
        from truthound.realtime.adapters.kinesis import KinesisAdapterConfig

        config_map = {
            "mock": MockAdapterConfig,
            "kafka": KafkaAdapterConfig,
            "kinesis": KinesisAdapterConfig,
        }

        config_cls = config_map.get(adapter_type, AdapterConfig)

        # Filter to valid fields
        valid_fields = set(config_cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in config_dict.items() if k in valid_fields}

        return config_cls(**filtered)


# =============================================================================
# Convenience Functions
# =============================================================================


def create_kafka_adapter(
    bootstrap_servers: str = "localhost:9092",
    topic: str = "",
    consumer_group: str = "truthound-consumer",
    **kwargs: Any,
) -> BaseStreamAdapter:
    """Create a Kafka adapter with common settings.

    Args:
        bootstrap_servers: Kafka broker addresses
        topic: Topic to consume from
        consumer_group: Consumer group ID
        **kwargs: Additional configuration

    Returns:
        Configured Kafka adapter
    """
    return StreamAdapterFactory.create(
        "kafka",
        bootstrap_servers=bootstrap_servers,
        topic=topic,
        consumer_group=consumer_group,
        **kwargs,
    )


def create_kinesis_adapter(
    stream_name: str,
    region_name: str = "us-east-1",
    **kwargs: Any,
) -> BaseStreamAdapter:
    """Create a Kinesis adapter with common settings.

    Args:
        stream_name: Kinesis stream name
        region_name: AWS region
        **kwargs: Additional configuration

    Returns:
        Configured Kinesis adapter
    """
    return StreamAdapterFactory.create(
        "kinesis",
        stream_name=stream_name,
        region_name=region_name,
        **kwargs,
    )


def create_mock_adapter(
    num_messages: int = 1000,
    error_rate: float = 0.1,
    schema: dict[str, str] | None = None,
    **kwargs: Any,
) -> BaseStreamAdapter:
    """Create a mock adapter for testing.

    Args:
        num_messages: Number of messages to generate
        error_rate: Error injection rate [0.0-1.0]
        schema: Message schema
        **kwargs: Additional configuration

    Returns:
        Configured mock adapter
    """
    config_kwargs: dict[str, Any] = {
        "num_messages": num_messages,
        "error_rate": error_rate,
    }
    if schema:
        config_kwargs["schema"] = schema
    config_kwargs.update(kwargs)

    return StreamAdapterFactory.create("mock", **config_kwargs)
