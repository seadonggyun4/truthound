"""Factory and registry for distributed monitoring.

This module provides factory and registry patterns for creating
and managing monitor instances.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Type

from truthound.profiler.distributed.monitoring.callbacks import (
    CallbackChain,
    ConsoleMonitorCallback,
    ConsoleStyle,
    FileMonitorCallback,
    LoggingMonitorCallback,
    MonitorCallbackAdapter,
    ProgressBarCallback,
    WebhookMonitorCallback,
)
from truthound.profiler.distributed.monitoring.config import MonitorConfig
from truthound.profiler.distributed.monitoring.monitor import DistributedMonitor
from truthound.profiler.distributed.monitoring.protocols import (
    AggregatedProgress,
    EventSeverity,
)


class MonitorRegistry:
    """Registry for callback types and monitor presets.

    Provides discovery and factory pattern for callbacks and monitors.

    Example:
        registry = MonitorRegistry()

        # Register custom callback
        @registry.register_callback("custom")
        class CustomCallback(MonitorCallbackAdapter):
            ...

        # Create callback by name
        callback = registry.create_callback("console", bar_width=50)

        # Create monitor with preset
        monitor = registry.create_monitor("production")
    """

    _instance: "MonitorRegistry | None" = None
    _lock = threading.Lock()

    def __new__(cls) -> "MonitorRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:  # type: ignore
            return

        self._callbacks: dict[str, Type[MonitorCallbackAdapter]] = {}
        self._callback_factories: dict[str, Callable[..., MonitorCallbackAdapter]] = {}
        self._config_presets: dict[str, MonitorConfig] = {}

        # Register built-in callbacks
        self._register_builtin_callbacks()
        self._register_builtin_presets()

        self._initialized = True

    def _register_builtin_callbacks(self) -> None:
        """Register built-in callback types."""
        self._callbacks["console"] = ConsoleMonitorCallback
        self._callbacks["progress_bar"] = ProgressBarCallback
        self._callbacks["logging"] = LoggingMonitorCallback
        self._callbacks["file"] = FileMonitorCallback
        self._callbacks["webhook"] = WebhookMonitorCallback

    def _register_builtin_presets(self) -> None:
        """Register built-in configuration presets."""
        self._config_presets["minimal"] = MonitorConfig.minimal()
        self._config_presets["standard"] = MonitorConfig.standard()
        self._config_presets["full"] = MonitorConfig.full()
        self._config_presets["production"] = MonitorConfig.production()

    def register_callback(
        self,
        name: str,
    ) -> Callable[[Type[MonitorCallbackAdapter]], Type[MonitorCallbackAdapter]]:
        """Decorator to register a callback type.

        Args:
            name: Registration name

        Returns:
            Decorator function
        """

        def decorator(
            cls: Type[MonitorCallbackAdapter],
        ) -> Type[MonitorCallbackAdapter]:
            self._callbacks[name] = cls
            return cls

        return decorator

    def register_callback_factory(
        self,
        name: str,
        factory: Callable[..., MonitorCallbackAdapter],
    ) -> None:
        """Register a callback factory function.

        Args:
            name: Registration name
            factory: Factory function
        """
        self._callback_factories[name] = factory

    def register_preset(self, name: str, config: MonitorConfig) -> None:
        """Register a configuration preset.

        Args:
            name: Preset name
            config: Configuration
        """
        self._config_presets[name] = config

    def create_callback(self, name: str, **kwargs: Any) -> MonitorCallbackAdapter:
        """Create callback by name.

        Args:
            name: Registered name
            **kwargs: Callback arguments

        Returns:
            Created callback

        Raises:
            KeyError: If name not registered
        """
        if name in self._callback_factories:
            return self._callback_factories[name](**kwargs)

        if name in self._callbacks:
            return self._callbacks[name](**kwargs)

        raise KeyError(
            f"Unknown callback: {name}. Available: {self.list_callbacks()}"
        )

    def get_preset(self, name: str) -> MonitorConfig:
        """Get configuration preset.

        Args:
            name: Preset name

        Returns:
            Configuration

        Raises:
            KeyError: If preset not found
        """
        if name not in self._config_presets:
            raise KeyError(
                f"Unknown preset: {name}. Available: {list(self._config_presets.keys())}"
            )
        return self._config_presets[name]

    def list_callbacks(self) -> list[str]:
        """List registered callback names."""
        return sorted(set(self._callbacks.keys()) | set(self._callback_factories.keys()))

    def list_presets(self) -> list[str]:
        """List registered preset names."""
        return sorted(self._config_presets.keys())


class MonitorFactory:
    """Factory for creating DistributedMonitor instances.

    Provides convenient factory methods for common configurations.

    Example:
        # Create with preset
        monitor = MonitorFactory.create(preset="production")

        # Create with console output
        monitor = MonitorFactory.with_console()

        # Create with full observability
        monitor = MonitorFactory.with_full_observability(log_file="events.jsonl")
    """

    @staticmethod
    def create(
        *,
        config: MonitorConfig | None = None,
        preset: str | None = None,
        callbacks: list[MonitorCallbackAdapter] | None = None,
        on_progress: Callable[[AggregatedProgress], None] | None = None,
    ) -> DistributedMonitor:
        """Create a DistributedMonitor.

        Args:
            config: Monitor configuration
            preset: Configuration preset name
            callbacks: Callbacks to add
            on_progress: Progress callback

        Returns:
            Configured DistributedMonitor
        """
        # Get configuration
        if config is None:
            if preset:
                registry = MonitorRegistry()
                config = registry.get_preset(preset)
            else:
                config = MonitorConfig()

        # Create monitor
        monitor = DistributedMonitor(config=config, on_progress=on_progress)

        # Add callbacks
        if callbacks:
            for callback in callbacks:
                monitor.add_callback(callback)

        return monitor

    @staticmethod
    def minimal() -> DistributedMonitor:
        """Create minimal monitor with no callbacks.

        Returns:
            DistributedMonitor with minimal configuration
        """
        return MonitorFactory.create(preset="minimal")

    @staticmethod
    def with_console(
        *,
        color: bool = True,
        show_progress_bar: bool = True,
        min_severity: EventSeverity = EventSeverity.INFO,
    ) -> DistributedMonitor:
        """Create monitor with console output.

        Args:
            color: Enable color output
            show_progress_bar: Show progress bar
            min_severity: Minimum severity

        Returns:
            DistributedMonitor with console callbacks
        """
        callbacks: list[MonitorCallbackAdapter] = [
            ConsoleMonitorCallback(
                style=ConsoleStyle(color_enabled=color),
                min_severity=min_severity,
            )
        ]

        if show_progress_bar:
            callbacks.append(ProgressBarCallback())

        return MonitorFactory.create(
            preset="standard",
            callbacks=callbacks,
        )

    @staticmethod
    def with_logging(
        *,
        logger_name: str = "truthound.distributed",
        min_severity: EventSeverity = EventSeverity.INFO,
        include_console: bool = False,
    ) -> DistributedMonitor:
        """Create monitor with logging output.

        Args:
            logger_name: Logger name
            min_severity: Minimum severity
            include_console: Also output to console

        Returns:
            DistributedMonitor with logging callback
        """
        callbacks: list[MonitorCallbackAdapter] = [
            LoggingMonitorCallback(
                logger_name=logger_name,
                min_severity=min_severity,
            )
        ]

        if include_console:
            callbacks.append(ConsoleMonitorCallback())

        return MonitorFactory.create(
            preset="standard",
            callbacks=callbacks,
        )

    @staticmethod
    def with_file(
        path: str,
        *,
        min_severity: EventSeverity = EventSeverity.INFO,
        include_console: bool = False,
    ) -> DistributedMonitor:
        """Create monitor with file output.

        Args:
            path: Output file path
            min_severity: Minimum severity
            include_console: Also output to console

        Returns:
            DistributedMonitor with file callback
        """
        callbacks: list[MonitorCallbackAdapter] = [
            FileMonitorCallback(path, min_severity=min_severity)
        ]

        if include_console:
            callbacks.append(ConsoleMonitorCallback())

        return MonitorFactory.create(
            preset="standard",
            callbacks=callbacks,
        )

    @staticmethod
    def with_webhook(
        url: str,
        *,
        headers: dict[str, str] | None = None,
        batch_size: int = 10,
        min_severity: EventSeverity = EventSeverity.WARNING,
        include_console: bool = True,
    ) -> DistributedMonitor:
        """Create monitor with webhook output.

        Args:
            url: Webhook URL
            headers: HTTP headers
            batch_size: Events per batch
            min_severity: Minimum severity for webhook
            include_console: Also output to console

        Returns:
            DistributedMonitor with webhook callback
        """
        callbacks: list[MonitorCallbackAdapter] = [
            WebhookMonitorCallback(
                url,
                headers=headers,
                batch_size=batch_size,
                min_severity=min_severity,
            )
        ]

        if include_console:
            callbacks.append(ConsoleMonitorCallback())

        return MonitorFactory.create(
            preset="production",
            callbacks=callbacks,
        )

    @staticmethod
    def with_full_observability(
        *,
        log_file: str | None = None,
        logger_name: str = "truthound.distributed",
        webhook_url: str | None = None,
        webhook_headers: dict[str, str] | None = None,
    ) -> DistributedMonitor:
        """Create monitor with full observability.

        Args:
            log_file: Optional file path for event logging
            logger_name: Logger name
            webhook_url: Optional webhook URL
            webhook_headers: Webhook headers

        Returns:
            DistributedMonitor with full observability
        """
        callbacks: list[MonitorCallbackAdapter] = [
            ConsoleMonitorCallback(),
            ProgressBarCallback(),
            LoggingMonitorCallback(logger_name=logger_name),
        ]

        if log_file:
            callbacks.append(FileMonitorCallback(log_file))

        if webhook_url:
            callbacks.append(
                WebhookMonitorCallback(
                    webhook_url,
                    headers=webhook_headers,
                    min_severity=EventSeverity.WARNING,
                )
            )

        return MonitorFactory.create(
            preset="full",
            callbacks=callbacks,
        )

    @staticmethod
    def production(
        *,
        logger_name: str = "truthound.distributed",
        log_file: str | None = None,
        alert_webhook: str | None = None,
    ) -> DistributedMonitor:
        """Create production-ready monitor.

        Args:
            logger_name: Logger name
            log_file: Optional log file path
            alert_webhook: Optional webhook for alerts

        Returns:
            Production-configured DistributedMonitor
        """
        callbacks: list[MonitorCallbackAdapter] = [
            LoggingMonitorCallback(
                logger_name=logger_name,
                min_severity=EventSeverity.INFO,
            )
        ]

        if log_file:
            callbacks.append(
                FileMonitorCallback(
                    log_file,
                    min_severity=EventSeverity.INFO,
                    rotate_size_mb=100,
                )
            )

        if alert_webhook:
            callbacks.append(
                WebhookMonitorCallback(
                    alert_webhook,
                    min_severity=EventSeverity.ERROR,
                    batch_size=1,  # Send alerts immediately
                )
            )

        return MonitorFactory.create(
            preset="production",
            callbacks=callbacks,
        )


# =============================================================================
# Convenience Functions
# =============================================================================


def create_monitor(
    *,
    preset: str = "standard",
    callbacks: list[MonitorCallbackAdapter] | None = None,
    on_progress: Callable[[AggregatedProgress], None] | None = None,
) -> DistributedMonitor:
    """Create a DistributedMonitor.

    Args:
        preset: Configuration preset
        callbacks: Callbacks to add
        on_progress: Progress callback

    Returns:
        Configured DistributedMonitor
    """
    return MonitorFactory.create(
        preset=preset,
        callbacks=callbacks,
        on_progress=on_progress,
    )


def create_console_monitor() -> DistributedMonitor:
    """Create a monitor with console output.

    Returns:
        DistributedMonitor with console callback
    """
    return MonitorFactory.with_console()


def create_production_monitor(
    *,
    logger_name: str = "truthound.distributed",
    log_file: str | None = None,
) -> DistributedMonitor:
    """Create a production-ready monitor.

    Args:
        logger_name: Logger name
        log_file: Optional log file

    Returns:
        Production-configured DistributedMonitor
    """
    return MonitorFactory.production(
        logger_name=logger_name,
        log_file=log_file,
    )
