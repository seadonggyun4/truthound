"""Installed, local-only monitor factory for real Core lifecycle tests."""

from contextlib import asynccontextmanager


@asynccontextmanager
async def factory():
    from truthound.mcp.ml_services import MLRuntime, MonitoringModel
    from truthound.ml.monitoring.alerting.rules import ThresholdConfig, ThresholdRule
    from truthound.ml.monitoring.monitor import ModelMonitor, MonitorConfig
    from truthound.ml.monitoring.stores.memory import InMemoryMetricStore

    from . import MARKER, note

    class Store(InMemoryMetricStore):
        async def store(self, metrics):
            note("ml-store")
            await super().store(metrics)
            if MARKER.with_name("ml-commit-failed").exists():
                raise OSError("synthetic-private-metric-commit")

    class Handler:
        name = "synthetic-private-handler"

        async def handle(self, alert):
            note("ml-alert")
            return not MARKER.with_name("ml-alert-failed").exists()

        async def resolve(self, alert):
            note("ml-resolved")
            return True

    monitor = ModelMonitor(Store())
    config = MonitorConfig(
        batch_size=10, collect_interval_seconds=0.05, alert_evaluation_interval_seconds=0.05
    )
    model = MonitoringModel(
        "synthetic-private-native-model",
        config,
        {"feature": "x"},
        "prediction",
        "actual",
        "latency",
    )
    note("ml-open")
    try:
        yield MLRuntime(
            monitor,
            {"local": model},
            rules={
                "latency": ThresholdRule(
                    "synthetic-private-rule", ThresholdConfig("latency_ms", 0.5)
                )
            },
            handlers={"local": Handler()},
        )
    finally:
        await monitor.stop()
        note("ml-close")
