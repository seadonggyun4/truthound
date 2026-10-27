# Model monitoring

`truthound.ml.monitoring.ModelMonitor` collects performance, prediction/feature
drift and labelled quality metrics. It owns a metric store, registered models,
alert rules and selected notification handlers. This advanced subsystem is
separate from the `truthound.core` validation kernel.

## Native lifecycle

```python
import asyncio

from truthound.ml.monitoring import ModelMonitor, MonitorConfig
from truthound.ml.monitoring.alerting.rules import ThresholdConfig, ThresholdRule
from truthound.ml.monitoring.collectors.quality import QualityConfig


async def main():
    monitor = ModelMonitor()  # native in-memory metric store; no external delivery
    monitor.register_model("classifier", MonitorConfig(
        batch_size=10,
        collect_interval_seconds=60,
        alert_evaluation_interval_seconds=30,
        quality=QualityConfig(window_size=1000, task_type="binary"),
    ))
    monitor.add_rule(ThresholdRule("latency", ThresholdConfig("latency_ms", 100)))
    async with monitor:
        identity = await monitor.record_prediction(
            "classifier", {"age": 25}, prediction=1, latency_ms=5,
        )
        await monitor.record_actual("classifier", identity, actual=1)
        collected = await monitor.collect_now("classifier")
        metrics = await monitor.get_metrics("classifier")
        assert collected == 1 and metrics.accuracy == 1.0


asyncio.run(main())
```

`record_prediction` returns a native prediction identity. Feedback is accepted
only while that prediction is buffered; missing or already collected identities
raise `KeyError`. `collect_now` returns the number of confirmed buffered records
collected. Model IDs must be unique. Registering a second time does not replace
pending state. `unregister_model` explicitly discards that model's pending records
and stops its timers; stored metrics are retained.

`start()` starts the selected store and one collection timer and one evaluation
timer for each model. Models registered while running receive both timers.
Collection also runs when `batch_size` is reached, including before `start()`.
Each model serializes recording, feedback, reference changes and collection so
records appended during a store await cannot enter two batches.

| Configuration | Behavior |
| --- | --- |
| `batch_size` | Positive integer threshold for immediate collection; default 100. |
| `collect_interval_seconds` | Positive finite interval for collecting an incomplete batch while started; default 60. |
| `alert_evaluation_interval_seconds` | Positive finite interval for observing the latest stored snapshot and applying per-model retention; default 30. |
| `retention_hours` | Positive finite model history window and deletion cutoff; default 24. The selected store may impose additional retention or entry limits. |
| `enable_drift_detection`, `enable_quality_metrics` | Actual booleans selecting native collectors; both default to true. |
| `performance`, `drift`, `quality` | Optional native `PerformanceConfig`, `DriftConfig`, `QualityConfig`. Registration captures independent configuration copies. |

Performance latency uses `window_size` and configured percentiles. Drift uses the
last `reference_window_size` reference records and a rolling
`detection_window_size` detection window. `set_reference_data` requires nonempty
native `PredictionRecord` objects belonging to that registered model and enabled
drift collection. Quality uses a rolling window of labelled records across
batches. `QualityConfig.task_type` selects `binary`, `multiclass`, `regression`
or the default `auto` heuristic; select a task explicitly when probability-like
regression values or numeric class labels would make inference ambiguous.

## Alerts, failure and cleanup

Rules must return actual boolean verdicts. A new stored snapshot is evaluated
once; periodic reads of the same snapshot do not add anomaly/trend observations
or redeliver an alert. Duration thresholds require subsequent fresh observations
above the threshold after the configured elapsed duration. Native rule names and
handler names must be unique. `remove_rule` and `remove_handler` remove explicitly
selected components. Removing a rule resolves its active state locally without
sending a resolution notification.

Each selected handler's `handle` or `resolve` must return `True`. `False`, missing
receipts and exceptions are failures. An alert is recorded when triggered;
history is not proof that every handler received it. The rule engine keeps at
most 10,000 triggers in memory by default, including later resolutions. Dashboard
data includes current metrics, metric history, active alerts and that real alert
history. Backend and handler persistence belong to their selected implementations.

A collection/evaluation error or cancellation latches that model's outcome as
uncertain. Pending predictions remain available for inspection by count and ID;
new collection and recording are blocked, and timers do not resend unknown
effects. `get_status()` reports the latch, buffer count and store lifecycle.
After investigation, `discard_uncertain(model_id)` explicitly discards pending
records, resets collectors and reference state, and marks the current stored
snapshot observed without sending it. It does not undo stored metrics or
notifications. Set reference data again before further drift detection.

`stop()` cancels and awaits owned timers, then stops the selected store. It does
not flush pending records or send new alerts. Failed cleanup remains visible;
`start()` rejects restart until store cleanup is confirmed. The native
`MonitoringPipeline` also propagates failed storage/delivery and blocks further
processing of the affected model until explicit `discard_uncertain(model_id)`.
It does not automatically retry unknown effects.

The default prediction buffer, metric store, rule history and failure latch are
in memory. They do not provide recovery after process loss or exactly-once
external notification delivery. Configure and verify external stores/handlers
under their own operational contracts.

## Approved MCP bindings

The unreleased Core profile exposes `ml_services_open` and `ml_services_execute`.
An approved installed factory yields `MLRuntime(monitor, models, rules, handlers)`.
Each model alias selects `MonitoringModel(native_id, config, features,
prediction_column, actual_column=None, latency_column=None)`. `features` maps
public feature aliases to fixed asset columns. Rules and handlers are factory
objects selected by aliases; caller code, native identities, connection settings
and endpoints are not wire inputs. The monitor must initially be stopped and
have no registered models, rules or handlers. Map the binding alias to its
installed capability under private `ml_services` bindings.

Commands are `inspect`, `status`, `register`, `unregister`, `start`, `stop`,
`shutdown`, `record`, `feedback`, `reference`, `collect`, `evaluate`,
`discard_uncertain`, `metrics`, `dashboard`, `rule_add`, `rule_remove`,
`handler_add` and `handler_remove`. Every command has exact approval and sensitive
disclosure requirements. Starting approves future periodic collection,
evaluation, retention deletion and selected handler delivery until stop, release
or expiry. The operator remains responsible for what the installed factory does
while entering and exiting its lifetime.

`record` and `reference` read pinned asset handles with bounded `offset`/`limit`;
`feedback` reads one row's fixed actual column at `offset` and requires an
owner-local prediction reference returned by `record`. Returned references state
whether the corresponding prediction remains pending after batching.
Releasing any bound input revokes the monitor and its derived artifacts. Cleanup
stops timers without flushing records. Metric output projects finite standard
fields, built-in aggregate metrics (including regression errors and configured
latency percentiles), configured feature aliases and optional factory-selected
`MonitoringModel.metric_aliases` mappings; arbitrary custom metric names, alert
messages, native IDs, handler metadata and endpoints are omitted.

An unconfirmed native operation returns a failed `native_monitor_incomplete`
artifact with current state. Reusing its approval key replays that result and
does not repeat collection/delivery. Inspect state before explicitly discarding
unknown pending work. A successful `shutdown` is acknowledged only after monitor
and factory cleanup finish. The same key may replay that result; further native
commands are rejected. Resource release alone is not proof that a slow factory
finalizer completed. These source features are not part of the published 3.2.0
package. See the [MCP guide](mcp.md#core-profile-unreleased-source-checkout).


Stopping also waits for in-flight foreground model operations before stopping the
store. New model work is blocked while cleanup is in progress; an already started
write is not rolled back, and later alert handlers are not started after stop.
Calling stop from inside that model's own operation is rejected to avoid waiting
on itself. `unregister_model` rejects a busy model. Handler edits affect subsequent
alerts; delivery iterates a snapshot so removing one handler during its await does
not silently skip another selected handler. Status includes `stopping`.
