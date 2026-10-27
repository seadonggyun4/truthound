"""Native model monitoring must not lose batches or acknowledge failed delivery."""

import asyncio
from datetime import UTC, datetime

import pytest

from truthound.ml.monitoring.monitor import ModelMonitor, MonitorConfig
from truthound.ml.monitoring.protocols import ModelMetrics
from truthound.ml.monitoring.stores.memory import InMemoryMetricStore

pytestmark = pytest.mark.fault


def configuration(**values):
    return MonitorConfig(enable_drift_detection=False, enable_quality_metrics=False, **values)


def test_started_monitor_collects_incomplete_batch_at_configured_interval():
    async def scenario():
        ready = asyncio.Event()

        class Store(InMemoryMetricStore):
            async def store(self, metrics):
                await super().store(metrics)
                ready.set()

        monitor = ModelMonitor(Store())
        monitor.register_model(
            "synthetic", configuration(batch_size=10, collect_interval_seconds=0.02)
        )
        try:
            await monitor.start()
            await monitor.record_prediction("synthetic", {"x": 1}, 1, latency_ms=2)
            await asyncio.wait_for(ready.wait(), timeout=1)
            assert (await monitor.get_metrics("synthetic")).latency_ms == 2
        finally:
            await monitor.stop()
        assert not monitor._tasks

    asyncio.run(scenario())


def test_prediction_identity_supports_delayed_feedback_and_missing_identity_fails():
    async def scenario():
        monitor = ModelMonitor()
        monitor.register_model("synthetic", configuration())
        try:
            identity = await monitor.record_prediction("synthetic", {"x": 1}, 1)
            assert isinstance(identity, str) and identity
            await monitor.record_actual("synthetic", identity, 0)
            assert monitor._prediction_buffers["synthetic"][0].actual == 0
            with pytest.raises(KeyError):
                await monitor.record_actual("synthetic", "missing", 1)
        finally:
            await monitor.stop()

    asyncio.run(scenario())


def test_concurrent_prediction_during_store_wait_is_not_collected_twice():
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        batches = []

        class Store(InMemoryMetricStore):
            async def store(self, metrics):
                entered.set()
                await release.wait()
                await super().store(metrics)

        class Collector:
            def collect(self, model_id, predictions):
                batches.append([item.prediction_id for item in predictions])
                return ModelMetrics(model_id, datetime.now(UTC))

        monitor = ModelMonitor(Store())
        monitor.register_model("synthetic", configuration(batch_size=1))
        monitor._collectors["synthetic"] = Collector()
        first = asyncio.create_task(monitor.record_prediction("synthetic", {"x": 1}, 1))
        second = None
        try:
            await asyncio.wait_for(entered.wait(), 1)
            second = asyncio.create_task(monitor.record_prediction("synthetic", {"x": 2}, 2))
            await asyncio.sleep(0)
            release.set()
            await asyncio.wait_for(asyncio.gather(first, second), 1)
            assert [len(batch) for batch in batches] == [1, 1]
            assert batches[0][0] != batches[1][0]
        finally:
            release.set()
            await asyncio.gather(
                *(task for task in (first, second) if task is not None), return_exceptions=True
            )
            await monitor.stop()

    asyncio.run(scenario())


def test_unknown_metric_store_commit_is_not_automatically_replayed():
    async def scenario():
        class Store(InMemoryMetricStore):
            calls = 0

            async def store(self, metrics):
                self.calls += 1
                await super().store(metrics)
                raise OSError("synthetic-private-commit-receipt")

        store = Store()
        monitor = ModelMonitor(store)
        monitor.register_model("synthetic", configuration(batch_size=1))
        try:
            with pytest.raises(OSError):
                await monitor.record_prediction("synthetic", {"x": 1}, 1)
            with pytest.raises(RuntimeError, match="uncertain"):
                await monitor.record_prediction("synthetic", {"x": 2}, 2)
            assert store.calls == 1
            assert len(monitor._prediction_buffers["synthetic"]) == 1
        finally:
            await monitor.stop()

    asyncio.run(scenario())


def test_false_alert_delivery_is_uncertain_and_never_automatically_resent():
    async def scenario():
        from truthound.ml.monitoring.alerting.rules import ThresholdConfig, ThresholdRule

        class Handler:
            name = "synthetic-local"
            calls = 0

            async def handle(self, alert):
                self.calls += 1
                return False

        handler = Handler()
        monitor = ModelMonitor(handlers=[handler])
        monitor.register_model("synthetic", configuration(batch_size=1))
        monitor.add_rule(
            ThresholdRule("latency", ThresholdConfig(metric_name="latency_ms", threshold=0))
        )
        try:
            with pytest.raises(RuntimeError, match="uncertain"):
                await monitor.record_prediction("synthetic", {"x": 1}, 1, latency_ms=1)
            with pytest.raises(RuntimeError, match="uncertain"):
                await monitor.record_prediction("synthetic", {"x": 2}, 2, latency_ms=1)
            assert handler.calls == 1
            assert monitor.get_status()["models"]["synthetic"]["uncertain"]
        finally:
            await monitor.stop()

    asyncio.run(scenario())


def test_alert_interval_observes_new_store_snapshot_and_dashboard_keeps_history():
    async def scenario():
        from truthound.ml.monitoring.alerting.rules import ThresholdConfig, ThresholdRule

        received = asyncio.Event()

        class Handler:
            name = "synthetic-local"
            calls = 0

            async def handle(self, alert):
                self.calls += 1
                received.set()
                return True

        handler, store = Handler(), InMemoryMetricStore()
        monitor = ModelMonitor(store, [handler])
        monitor.register_model(
            "synthetic",
            configuration(collect_interval_seconds=60, alert_evaluation_interval_seconds=0.02),
        )
        monitor.add_rule(
            ThresholdRule("latency", ThresholdConfig(metric_name="latency_ms", threshold=0))
        )
        try:
            await monitor.start()
            await store.store(ModelMetrics("synthetic", datetime.now(UTC), latency_ms=1))
            await asyncio.wait_for(received.wait(), 1)
            dashboard = await monitor.get_dashboard_data("synthetic")
            assert len(dashboard.alert_history) == 1 and handler.calls == 1
        finally:
            await monitor.stop()

    asyncio.run(scenario())


@pytest.mark.parametrize("verdict", [False, "delivered", None])
def test_pipeline_rejects_unconfirmed_delivery_and_latches_model(verdict):
    async def scenario():
        from truthound.ml.monitoring.alerting.rules import ThresholdConfig, ThresholdRule
        from truthound.ml.monitoring.collectors import PerformanceCollector
        from truthound.ml.monitoring.monitor import MonitoringPipeline
        from truthound.ml.monitoring.protocols import PredictionRecord

        class Handler:
            calls = 0

            async def handle(self, alert):
                self.calls += 1
                return verdict

        handler = Handler()
        pipeline = (
            MonitoringPipeline()
            .add_collector(PerformanceCollector())
            .set_store(InMemoryMetricStore())
        )
        pipeline.add_rule(ThresholdRule("latency", ThresholdConfig("latency_ms", 0))).add_handler(
            handler
        )
        record = PredictionRecord(
            "synthetic", "local-prediction", datetime.now(UTC), {"x": 1}, 1, latency_ms=1
        )
        with pytest.raises(RuntimeError, match="uncertain"):
            await pipeline.process("synthetic", [record])
        with pytest.raises(RuntimeError, match="uncertain"):
            await pipeline.process("synthetic", [record])
        assert handler.calls == 1

    asyncio.run(scenario())


def test_rule_engine_rejects_truthy_verdict_and_preserves_resolved_history():
    from truthound.ml.monitoring.alerting.rules import RuleEngine, ThresholdConfig, ThresholdRule

    class UntypedRule(ThresholdRule):
        def evaluate(self, metrics):
            return "false"

    metrics = ModelMetrics("synthetic", datetime.now(UTC), latency_ms=1)
    engine = RuleEngine()
    engine.add_rule(UntypedRule("untyped", ThresholdConfig("latency_ms", 0)))
    with pytest.raises(TypeError, match="boolean"):
        engine.evaluate(metrics)
    assert not engine.get_active_alerts()
    engine.remove_rule("untyped")
    engine.add_rule(ThresholdRule("latency", ThresholdConfig("latency_ms", 0)))
    engine.evaluate(metrics)
    engine.evaluate(ModelMetrics("synthetic", datetime.now(UTC)))
    assert not engine.get_active_alerts()
    assert len(engine.get_alert_history()) == 1 and engine.get_alert_history()[0].is_resolved


def test_stop_failure_is_visible_and_cannot_restart_until_cleanup_confirmed():
    async def scenario():
        class Store(InMemoryMetricStore):
            failed = True

            async def stop(self):
                await super().stop()
                return False if self.failed else None

        store = Store()
        monitor = ModelMonitor(store)
        monitor.register_model("synthetic", configuration())
        await monitor.start()
        await monitor.record_prediction("synthetic", {"x": 1}, 1)
        with pytest.raises(RuntimeError, match="cleanup"):
            await monitor.stop()
        with pytest.raises(RuntimeError, match="cleanup"):
            await monitor.start()
        assert monitor.get_status()["store_active"] and not monitor._tasks
        assert monitor.get_status()["models"]["synthetic"]["buffered"] == 1
        store.failed = False
        await monitor.stop()

    asyncio.run(scenario())


def test_quality_window_includes_previous_batches_and_eviction_changes_accuracy():
    from truthound.ml.monitoring.collectors.quality import QualityCollector, QualityConfig
    from truthound.ml.monitoring.protocols import PredictionRecord

    collector = QualityCollector(QualityConfig(window_size=2))

    def record(identity, prediction, actual):
        return PredictionRecord("synthetic", identity, datetime.now(UTC), {}, prediction, actual)

    collector.collect("synthetic", [record("first", 1, 1)])
    metrics = collector.collect("synthetic", [record("second", 1, 0)])
    assert metrics.accuracy == 0.5
    metrics = collector.collect("synthetic", [record("third", 0, 0)])
    assert metrics.accuracy == 0.5 and metrics.custom_metrics["total_predictions"] == 2


def test_drift_reference_and_detection_windows_apply_configured_bounds():
    from truthound.ml.monitoring.collectors.drift import DriftCollector, DriftConfig
    from truthound.ml.monitoring.protocols import PredictionRecord

    collector = DriftCollector(DriftConfig(reference_window_size=2, detection_window_size=2))

    def record(value):
        return PredictionRecord("synthetic", str(value), datetime.now(UTC), {"x": value}, value)

    collector.set_reference("synthetic", [record(100), record(0), record(0)])
    collector.collect("synthetic", [record(0)])
    same = collector.collect("synthetic", [record(0)])
    assert collector._reference_predictions["synthetic"] == [0, 0]
    assert collector._current_predictions["synthetic"] == [0, 0]
    assert same.prediction_drift == 0


@pytest.mark.parametrize(
    "predictions,actuals,expected",
    [
        ([0.9, 0.1], [1, 0], 1.0),
        (["cat", "dog"], ["dog", "cat"], 0.0),
    ],
)
def test_quality_auto_classification_does_not_misread_probabilities_or_string_labels(
    predictions, actuals, expected
):
    from truthound.ml.monitoring.collectors.quality import QualityCollector
    from truthound.ml.monitoring.protocols import PredictionRecord

    records = [
        PredictionRecord("synthetic", str(index), datetime.now(UTC), {}, predicted, actual)
        for index, (predicted, actual) in enumerate(zip(predictions, actuals, strict=True))
    ]
    metrics = QualityCollector().collect("synthetic", records)
    assert metrics.accuracy == expected


def test_stop_waits_for_inflight_store_and_does_not_send_followup_alerts():
    async def scenario():
        from truthound.ml.monitoring.alerting.rules import ThresholdConfig, ThresholdRule

        entered, release = asyncio.Event(), asyncio.Event()
        effects = []

        class Store(InMemoryMetricStore):
            async def store(self, metrics):
                entered.set()
                await release.wait()
                effects.append("store")
                await super().store(metrics)

            async def stop(self):
                effects.append("stop")
                await super().stop()

        class Handler:
            name = "local"

            async def handle(self, alert):
                effects.append("alert")
                return True

        monitor = ModelMonitor(Store(), [Handler()])
        monitor.register_model("synthetic", configuration(batch_size=1))
        monitor.add_rule(ThresholdRule("latency", ThresholdConfig("latency_ms", 0)))
        record = asyncio.create_task(monitor.record_prediction("synthetic", {}, 1, latency_ms=1))
        stopping = None
        try:
            await asyncio.wait_for(entered.wait(), 1)
            stopping = asyncio.create_task(monitor.stop())
            await asyncio.sleep(0)
            assert not stopping.done() and "stop" not in effects
            release.set()
            await asyncio.wait_for(asyncio.gather(record, stopping, return_exceptions=True), 1)
            assert effects == ["store", "stop"]
            assert monitor.get_status()["models"]["synthetic"]["uncertain"]
        finally:
            release.set()
            await asyncio.gather(
                *(task for task in (record, stopping) if task is not None), return_exceptions=True
            )
            await monitor.stop()

    asyncio.run(scenario())


def test_removing_handler_during_delivery_does_not_skip_another_selected_handler():
    async def scenario():
        from truthound.ml.monitoring.alerting.rules import ThresholdConfig, ThresholdRule

        entered, release = asyncio.Event(), asyncio.Event()
        delivered = []

        class First:
            name = "first"

            async def handle(self, alert):
                entered.set()
                await release.wait()
                delivered.append("first")
                return True

        class Second:
            name = "second"

            async def handle(self, alert):
                delivered.append("second")
                return True

        monitor = ModelMonitor(handlers=[First(), Second()])
        monitor.register_model("synthetic", configuration(batch_size=1))
        monitor.add_rule(ThresholdRule("latency", ThresholdConfig("latency_ms", 0)))
        recording = asyncio.create_task(monitor.record_prediction("synthetic", {}, 1, latency_ms=1))
        try:
            await asyncio.wait_for(entered.wait(), 1)
            assert monitor.remove_handler("first")
            release.set()
            await asyncio.wait_for(recording, 1)
            assert delivered == ["first", "second"]
        finally:
            release.set()
            await asyncio.gather(recording, return_exceptions=True)
            await monitor.stop()

    asyncio.run(scenario())
