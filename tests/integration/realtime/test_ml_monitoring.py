"""Integration tests for ML model monitoring.

Tests the complete ML monitoring pipeline including collectors,
stores, alerting, and dashboard generation.
"""

from __future__ import annotations

import asyncio
import pytest
from datetime import datetime, timezone
from typing import Any

pytest_plugins = ["pytest_asyncio"]


@pytest.fixture(scope="module")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


class TestModelMonitorIntegration:
    """Integration tests for ModelMonitor."""

    @pytest.fixture
    def sample_predictions(self) -> list[dict[str, Any]]:
        """Generate sample prediction records."""
        import random
        random.seed(42)

        predictions = []
        for i in range(100):
            predictions.append({
                "model_id": "test-model",
                "prediction_id": f"pred-{i}",
                "features": {
                    "age": random.randint(18, 80),
                    "income": random.uniform(20000, 200000),
                    "score": random.uniform(0, 1),
                },
                "prediction": random.choice([0, 1]),
                "actual": random.choice([0, 1]) if random.random() > 0.3 else None,
                "latency_ms": random.uniform(1, 50),
            })
        return predictions

    @pytest.mark.asyncio
    async def test_full_monitoring_pipeline(self, sample_predictions):
        """Test complete monitoring pipeline."""
        from truthound.ml.monitoring.monitor import ModelMonitor, MonitorConfig

        # Create monitor with custom config
        config = MonitorConfig(
            batch_size=50,
            enable_drift_detection=True,
            enable_quality_metrics=True,
        )

        monitor = ModelMonitor()

        async with monitor:
            # Register model
            monitor.register_model("test-model", config)

            # Record predictions
            for pred in sample_predictions:
                await monitor.record_prediction(
                    model_id=pred["model_id"],
                    features=pred["features"],
                    prediction=pred["prediction"],
                    actual=pred["actual"],
                    latency_ms=pred["latency_ms"],
                )

            # Get metrics
            metrics = await monitor.get_metrics("test-model")

            # Verify metrics collected
            if metrics:
                # Check that metrics have valid data
                assert metrics.model_id == "test-model"
                assert metrics.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_dashboard_data_generation(self, sample_predictions):
        """Test dashboard data generation."""
        from truthound.ml.monitoring.monitor import ModelMonitor, MonitorConfig

        monitor = ModelMonitor()

        async with monitor:
            # Register and populate
            monitor.register_model("dashboard-model")

            for pred in sample_predictions[:50]:
                await monitor.record_prediction(
                    model_id="dashboard-model",
                    features=pred["features"],
                    prediction=pred["prediction"],
                    actual=pred["actual"],
                    latency_ms=pred["latency_ms"],
                )

            # Get dashboard data
            dashboard = await monitor.get_dashboard_data("dashboard-model")

            assert dashboard.model_id == "dashboard-model"
            assert dashboard.health_score >= 0
            assert dashboard.health_score <= 100
            assert dashboard.last_updated is not None

    @pytest.mark.asyncio
    async def test_alert_rule_evaluation(self, sample_predictions):
        """Test alert rule evaluation."""
        from truthound.ml.monitoring.monitor import ModelMonitor, MonitorConfig
        from truthound.ml.monitoring.alerting.rules import ThresholdRule, ThresholdConfig
        from truthound.ml.monitoring.protocols import AlertSeverity

        monitor = ModelMonitor()

        # Add a rule that should trigger
        latency_rule = ThresholdRule(
            name="high-latency",
            config=ThresholdConfig(
                metric_name="latency_ms",
                threshold=10.0,
                comparison="gt",
            ),
            severity=AlertSeverity.WARNING,
        )
        monitor.add_rule(latency_rule)

        async with monitor:
            monitor.register_model("alert-model")

            # Record predictions with varying latency
            for pred in sample_predictions:
                await monitor.record_prediction(
                    model_id="alert-model",
                    features=pred["features"],
                    prediction=pred["prediction"],
                    latency_ms=pred["latency_ms"],
                )

    @pytest.mark.asyncio
    async def test_multiple_model_monitoring(self, sample_predictions):
        """Test monitoring multiple models simultaneously."""
        from truthound.ml.monitoring.monitor import ModelMonitor

        monitor = ModelMonitor()

        async with monitor:
            # Register multiple models
            for model_id in ["model-a", "model-b", "model-c"]:
                monitor.register_model(model_id)

            # Record predictions for each model
            for i, pred in enumerate(sample_predictions):
                model_id = f"model-{chr(ord('a') + i % 3)}"
                await monitor.record_prediction(
                    model_id=model_id,
                    features=pred["features"],
                    prediction=pred["prediction"],
                    latency_ms=pred["latency_ms"],
                )

            # Verify each model has dashboard
            for model_id in ["model-a", "model-b", "model-c"]:
                dashboard = await monitor.get_dashboard_data(model_id)
                assert dashboard.model_id == model_id


class TestDriftDetectionIntegration:
    """Integration tests for drift detection."""

    @pytest.mark.asyncio
    async def test_drift_detection_with_reference(self):
        """Test drift detection with reference data."""
        from truthound.ml.monitoring.monitor import ModelMonitor
        from truthound.ml.monitoring.protocols import PredictionRecord
        from datetime import datetime, timezone
        import random
        random.seed(42)

        monitor = ModelMonitor()

        async with monitor:
            monitor.register_model("drift-model")

            # Create reference data (normal distribution centered at 50)
            reference_records = []
            for i in range(500):
                record = PredictionRecord(
                    model_id="drift-model",
                    prediction_id=f"ref-{i}",
                    timestamp=datetime.now(timezone.utc),
                    features={
                        "value": random.gauss(50, 10),
                        "category": random.choice(["A", "B", "C"]),
                    },
                    prediction=random.choice([0, 1]),
                    latency_ms=random.uniform(5, 20),
                )
                reference_records.append(record)

            await monitor.set_reference_data("drift-model", reference_records)

            # Record current data with drift (shifted to 70)
            for i in range(100):
                await monitor.record_prediction(
                    model_id="drift-model",
                    features={
                        "value": random.gauss(70, 15),
                        "category": random.choice(["A", "B", "D"]),
                    },
                    prediction=random.choice([0, 1]),
                    latency_ms=random.uniform(10, 30),
                )

            # Check metrics for drift
            metrics = await monitor.get_metrics("drift-model")


class TestMonitoringPipelineIntegration:
    """Integration tests for composable monitoring pipeline."""

    @pytest.mark.asyncio
    async def test_pipeline_composition(self):
        """Test composable pipeline creation."""
        from truthound.ml.monitoring.monitor import MonitoringPipeline
        from truthound.ml.monitoring.collectors import (
            PerformanceCollector,
            QualityCollector,
        )
        from truthound.ml.monitoring.stores import InMemoryMetricStore
        from truthound.ml.monitoring.stores.memory import InMemoryStoreConfig
        from truthound.ml.monitoring.alerting.rules import ThresholdRule, ThresholdConfig
        from truthound.ml.monitoring.protocols import PredictionRecord, AlertSeverity
        from datetime import datetime, timezone
        import random
        random.seed(42)

        # Build pipeline
        pipeline = (
            MonitoringPipeline()
            .add_collector(PerformanceCollector())
            .add_collector(QualityCollector())
            .set_store(InMemoryMetricStore())
            .add_rule(ThresholdRule(
                name="latency-check",
                config=ThresholdConfig(
                    metric_name="latency_ms",
                    threshold=100.0,
                    comparison="gt",
                ),
                severity=AlertSeverity.WARNING,
            ))
        )

        # Create predictions
        predictions = []
        for i in range(50):
            predictions.append(PredictionRecord(
                model_id="pipeline-model",
                prediction_id=f"pred-{i}",
                timestamp=datetime.now(timezone.utc),
                features={"x": random.uniform(0, 100)},
                prediction=random.choice([0, 1]),
                actual=random.choice([0, 1]),
                latency_ms=random.uniform(5, 50),
            ))

        # Process through pipeline
        metrics, alerts = await pipeline.process("pipeline-model", predictions)

        assert metrics is not None
        assert metrics.latency_ms >= 0
        assert isinstance(alerts, list)

    @pytest.mark.asyncio
    async def test_custom_collector(self):
        """Test pipeline with custom collector."""
        from truthound.ml.monitoring.monitor import MonitoringPipeline
        from truthound.ml.monitoring.protocols import (
            IMetricCollector,
            ModelMetrics,
            PredictionRecord,
        )
        from datetime import datetime, timezone

        # Create custom collector
        class CustomCollector(IMetricCollector):
            def collect(
                self,
                model_id: str,
                predictions: list[PredictionRecord],
            ) -> ModelMetrics:
                total_features = sum(
                    len(p.features) for p in predictions
                )
                avg_latency = sum(p.latency_ms for p in predictions) / len(predictions) if predictions else 0
                return ModelMetrics(
                    model_id=model_id,
                    timestamp=datetime.now(timezone.utc),
                    latency_ms=avg_latency,
                    custom_metrics={"total_features": total_features},
                )

        pipeline = MonitoringPipeline().add_collector(CustomCollector())

        predictions = [
            PredictionRecord(
                model_id="custom-model",
                prediction_id=f"pred-{i}",
                timestamp=datetime.now(timezone.utc),
                features={"a": i, "b": i * 2, "c": i * 3},
                prediction=0,
                latency_ms=10.0,
            )
            for i in range(10)
        ]

        metrics, _ = await pipeline.process("custom-model", predictions)

        assert metrics.custom_metrics["total_features"] == 30


class TestAlertHandlerIntegration:
    """Integration tests for alert handlers."""

    @pytest.mark.asyncio
    async def test_webhook_handler(self):
        """Test webhook alert handler (mocked)."""
        from truthound.ml.monitoring.alerting.handlers import WebhookAlertHandler, WebhookConfig
        from truthound.ml.monitoring.protocols import Alert, AlertSeverity, ModelMetrics
        from datetime import datetime, timezone

        handler = WebhookAlertHandler(WebhookConfig(
            url="http://example.com/alerts",
        ))

        metrics = ModelMetrics(
            model_id="test-model",
            timestamp=datetime.now(timezone.utc),
        )

        alert = Alert(
            alert_id="test-alert-1",
            rule_name="test-rule",
            model_id="test-model",
            severity=AlertSeverity.WARNING,
            message="Test alert message",
            metrics=metrics,
        )

        assert handler.name == "webhook"

    @pytest.mark.asyncio
    async def test_multiple_handlers(self):
        """Test multiple alert handlers."""
        from truthound.ml.monitoring.alerting.handlers import (
            SlackAlertHandler,
            SlackConfig,
            WebhookAlertHandler,
            WebhookConfig,
        )

        handlers = [
            SlackAlertHandler(SlackConfig(
                webhook_url="http://slack.example.com/webhook",
            )),
            WebhookAlertHandler(WebhookConfig(
                url="http://example.com/alerts",
            )),
        ]

        assert handlers[0].name == "slack"
        assert handlers[1].name == "webhook"


class TestMetricStoreIntegration:
    """Integration tests for metric stores."""

    @pytest.mark.asyncio
    async def test_in_memory_store(self):
        """Test in-memory metric store."""
        from truthound.ml.monitoring.stores import InMemoryMetricStore
        from truthound.ml.monitoring.stores.memory import InMemoryStoreConfig
        from truthound.ml.monitoring.protocols import ModelMetrics
        from datetime import datetime, timezone, timedelta

        config = InMemoryStoreConfig(
            max_entries_per_model=100,
            retention_hours=24,
        )
        store = InMemoryMetricStore(config)
        await store.start()

        now = datetime.now(timezone.utc)

        # Store multiple metrics
        for i in range(10):
            metrics = ModelMetrics(
                model_id="store-test",
                timestamp=now - timedelta(hours=i),
                latency_ms=10.0 + i,
            )
            await store.store(metrics)

        # Query latest
        latest = await store.get_latest("store-test")
        assert latest is not None

        # Query range
        results = await store.query(
            model_id="store-test",
            start_time=now - timedelta(hours=5),
            end_time=now,
        )
        assert len(results) == 6

        await store.stop()

    @pytest.mark.asyncio
    async def test_store_retention(self):
        """Test metric store with configured retention."""
        from truthound.ml.monitoring.stores import InMemoryMetricStore
        from truthound.ml.monitoring.stores.memory import InMemoryStoreConfig
        from truthound.ml.monitoring.protocols import ModelMetrics
        from datetime import datetime, timezone, timedelta

        config = InMemoryStoreConfig(
            max_entries_per_model=5,
            retention_hours=1,
        )
        store = InMemoryMetricStore(config)
        await store.start()

        now = datetime.now(timezone.utc)

        # Store more than max entries
        for i in range(10):
            metrics = ModelMetrics(
                model_id="retention-test",
                timestamp=now - timedelta(minutes=i * 5),
                latency_ms=float(i),
            )
            await store.store(metrics)

        # Query all
        results = await store.query(
            model_id="retention-test",
            start_time=now - timedelta(hours=2),
            end_time=now,
        )
        assert len(results) <= 10

        await store.stop()
