"""One native monitor owner. No wire-selected models, code, paths or endpoints."""

import math
from datetime import UTC, datetime
from numbers import Real
from uuid import uuid4

from pydantic import TypeAdapter

from truthound.mcp.broker_worker import NativeServiceResult
from truthound.mcp.checkpoint_worker import binding_guard
from truthound.mcp.contracts import Name
from truthound.mcp.governance_worker import selected
from truthound.mcp.ml_services import MLRuntime, MonitoringModel
from truthound.mcp.policy import MCPError


class MLOwner:
    def __init__(self, runtime, request):
        from truthound.ml.monitoring.alerting.rules import AlertRule
        from truthound.ml.monitoring.monitor import ModelMonitor, MonitorConfig
        from truthound.ml.monitoring.protocols import IAlertHandler

        if request["native_group"] != "ml_services" or not isinstance(runtime, MLRuntime):
            raise MCPError("invalid_binding")
        if not isinstance(runtime.monitor, ModelMonitor) or not runtime.models:
            raise MCPError("invalid_binding")
        alias = TypeAdapter(Name)
        for mapping in (runtime.models, runtime.rules, runtime.handlers):
            for key in mapping:
                alias.validate_python(key)
        models = list(runtime.models.values())
        if any(
            not isinstance(model, MonitoringModel)
            or not isinstance(model.config, MonitorConfig)
            or not isinstance(model.native_id, str)
            or not model.native_id
            or not isinstance(model.prediction_column, str)
            or not model.prediction_column
            or not model.features
            or any(not isinstance(value, str) or not value for value in model.features.values())
            or any(
                value is not None and (not isinstance(value, str) or not value)
                for value in (model.actual_column, model.latency_column)
            )
            for model in models
        ):
            raise MCPError("invalid_binding")
        if len({model.native_id for model in models}) != len(models):
            raise MCPError("invalid_binding")
        for model in models:
            for key in (*model.features, *model.metric_aliases):
                alias.validate_python(key)
            if any(
                not isinstance(value, str) or not value for value in model.metric_aliases.values()
            ):
                raise MCPError("invalid_binding")
        if (
            any(not isinstance(rule, AlertRule) for rule in runtime.rules.values())
            or any(not isinstance(handler, IAlertHandler) for handler in runtime.handlers.values())
            or len({rule.name for rule in runtime.rules.values()}) != len(runtime.rules)
            or len({handler.name for handler in runtime.handlers.values()}) != len(runtime.handlers)
        ):
            raise MCPError("invalid_binding")
        state = runtime.monitor.get_status()
        # Components are activated through explicit commands; constructor effects
        # and cleanup still belong to the approved factory.
        components = runtime.monitor.get_components()
        if (
            state["running"]
            or state["store_active"]
            or state["models"]
            or components["handlers"]
            or components["rules"]
        ):
            raise MCPError("invalid_binding")
        self.runtime = runtime
        self.monitor = runtime.monitor
        self.guard = binding_guard(request)
        self.monitor.set_execution_guard(self.guard)
        self.predictions = {}
        self.active_rules = set()
        self.active_handlers = set()
        self.shutdown_requested = False

    def state(self):
        native = self.monitor.get_status()
        return {
            "running": native["running"],
            "store_active": native["store_active"],
            "stopping": native["stopping"],
            "models": {
                alias: {
                    **native["models"][model.native_id],
                    "error_type": "native_monitor_error"
                    if native["models"][model.native_id]["error_type"]
                    else None,
                }
                for alias, model in self.runtime.models.items()
                if model.native_id in native["models"]
            },
            "active_rules": sorted(self.active_rules),
            "active_handlers": sorted(self.active_handlers),
        }

    def metric_view(self, value, model):
        if value is None:
            return None
        if value.model_id != model.native_id:
            raise MCPError("invalid_binding")

        def numeric(number):
            if number is None:
                return None
            if (
                isinstance(number, bool)
                or not isinstance(number, Real)
                or not math.isfinite(number)
            ):
                raise MCPError("unsupported_value")
            return float(number)

        fields = (
            "latency_ms",
            "throughput_rps",
            "prediction_drift",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
        )
        known = {
            "mse",
            "mae",
            "rmse",
            "total_predictions",
            "num_classes",
            "true_positives",
            "true_negatives",
            "false_positives",
            "false_negatives",
            "specificity",
            "balanced_accuracy",
            "latency_min",
            "latency_max",
            "latency_std",
            "error_count",
            "error_rate",
            "request_count",
            "max_feature_drift",
            "mean_feature_drift",
            "drifted_feature_count",
        }
        performance = model.config.performance
        known.update(
            "latency_p" + str(int(value * 100))
            for value in (performance.percentiles if performance else [0.5, 0.95, 0.99])
        )
        custom = {
            key: numeric(value.custom_metrics[key])
            for key in sorted(known)
            if key in value.custom_metrics
        }
        custom.update(
            {
                key: numeric(value.custom_metrics[column])
                for key, column in model.metric_aliases.items()
                if column in value.custom_metrics
            }
        )
        return {
            "timestamp": value.timestamp.isoformat(),
            **{name: numeric(getattr(value, name)) for name in fields},
            "feature_drift": {
                key: numeric(value.feature_drift[key])
                for key in model.features
                if key in value.feature_drift
            },
            "custom_metrics": custom,
        }

    def alert_view(self, alert, model):
        rule_alias = next(
            (key for key, rule in self.runtime.rules.items() if rule.name == alert.rule_name), None
        )
        if rule_alias is None:
            raise MCPError("invalid_binding")
        return {
            "rule_id": rule_alias,
            "severity": alert.severity.value,
            "triggered_at": alert.triggered_at.isoformat(),
            "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at is not None else None,
            "metrics": self.metric_view(alert.metrics, model),
        }

    def prune_predictions(self, alias, model):
        pending = set(self.monitor.get_pending_prediction_ids(model.native_id))
        self.predictions = {
            key: value
            for key, value in self.predictions.items()
            if value[0] != alias or value[1] in pending
        }

    async def execute(self, args, budget, *, inputs=None):
        import polars as pl

        from truthound.ml.monitoring.protocols import PredictionRecord

        self.guard()
        command = args["command"]
        limit = args.get("limit", 100)
        if command == "inspect":
            return {
                "models": sorted(self.runtime.models)[:limit],
                "rules": sorted(self.runtime.rules)[:limit],
                "handlers": sorted(self.runtime.handlers)[:limit],
                "state": self.state(),
            }
        if command == "status":
            return self.state()
        if command == "shutdown":
            self.shutdown_requested = True
            return {"shutdown": True}
        model_alias = args.get("model_id")
        model = selected(self.runtime.models, model_alias) if model_alias is not None else None
        if command in {"record", "reference", "feedback"}:
            if not inputs or set(inputs) != {"asset_path"}:
                raise MCPError("invalid_handle")
            frame = pl.scan_ipc(inputs["asset_path"], memory_map=False)
            columns = (
                [model.actual_column]
                if command == "feedback"
                else list(model.features.values()) + [model.prediction_column]
            )
            if command != "feedback":
                columns += [
                    column
                    for column in (model.actual_column, model.latency_column)
                    if column is not None
                ]
            if any(column is None or column not in frame.collect_schema() for column in columns):
                raise MCPError("invalid_arguments")
            rows = (
                frame.select(list(dict.fromkeys(columns)))
                .slice(args.get("offset", 0), 1 if command == "feedback" else limit)
                .collect()
                .to_dicts()
            )
            if not rows:
                raise MCPError("invalid_arguments")
            if command == "record":
                self.prune_predictions(model_alias, model)
                # Retained identities are bounded separately from the artifact budget.
                if len(self.predictions) + len(rows) > min(10000, budget // 256):
                    raise MCPError("resource_exhausted")
        try:
            if command == "start":
                await self.monitor.start()
            elif command == "stop":
                await self.monitor.stop()
            elif command == "register":
                self.monitor.register_model(model.native_id, model.config)
            elif command == "unregister":
                self.monitor.unregister_model(model.native_id)
                self.predictions = {
                    key: value for key, value in self.predictions.items() if value[0] != model_alias
                }
            elif command == "rule_add":
                self.monitor.add_rule(selected(self.runtime.rules, args["rule_id"]))
                self.active_rules.add(args["rule_id"])
            elif command == "rule_remove":
                removed = self.monitor.remove_rule(
                    selected(self.runtime.rules, args["rule_id"]).name
                )
                if not removed:
                    raise MCPError("invalid_arguments")
                self.active_rules.discard(args["rule_id"])
            elif command in {"handler_add", "handler_remove"}:
                handler = selected(self.runtime.handlers, args["handler_id"])
                if command == "handler_add":
                    self.monitor.add_handler(handler)
                    self.active_handlers.add(args["handler_id"])
                else:
                    if not self.monitor.remove_handler(handler.name):
                        raise MCPError("invalid_arguments")
                    self.active_handlers.discard(args["handler_id"])
            elif command == "record":
                result = []
                for index, row in enumerate(rows):
                    self.guard()
                    identity = await self.monitor.record_prediction(
                        model.native_id,
                        {key: row[column] for key, column in model.features.items()},
                        row[model.prediction_column],
                        actual=row[model.actual_column] if model.actual_column else None,
                        latency_ms=row[model.latency_column] if model.latency_column else 0.0,
                    )
                    reference = "prediction_" + uuid4().hex
                    self.predictions[reference] = (model_alias, identity)
                    result.append(
                        {"prediction_id": reference, "input_index": args.get("offset", 0) + index}
                    )
                self.prune_predictions(model_alias, model)
                for row in result:
                    row["pending"] = row["prediction_id"] in self.predictions
                return {"recorded": len(result), "predictions": result, "state": self.state()}
            elif command == "reference":
                records = [
                    PredictionRecord(
                        model.native_id,
                        uuid4().hex,
                        datetime.now(UTC),
                        {key: row[column] for key, column in model.features.items()},
                        row[model.prediction_column],
                    )
                    for row in rows
                ]
                await self.monitor.set_reference_data(model.native_id, records)
                return {"reference_records": len(records), "state": self.state()}
            elif command == "feedback":
                retained = selected(self.predictions, args["prediction_id"])
                if retained[0] != model_alias:
                    raise MCPError("invalid_handle")
                await self.monitor.record_actual(
                    model.native_id, retained[1], rows[0][model.actual_column]
                )
            elif command == "collect":
                count = await self.monitor.collect_now(model.native_id)
                self.prune_predictions(model_alias, model)
                return {"collected": count, "state": self.state()}
            elif command == "evaluate":
                deleted = await self.monitor.evaluate_now(model.native_id)
                return {"retention_deleted": deleted, "state": self.state()}
            elif command == "discard_uncertain":
                count = await self.monitor.discard_uncertain(model.native_id)
                self.prune_predictions(model_alias, model)
                return {"discarded": count, "state": self.state()}
            elif command == "metrics":
                value = await self.monitor.get_metrics(model.native_id)
                return {
                    "model_id": model_alias,
                    "metrics": self.metric_view(value, model),
                    "state": self.state(),
                }
            elif command == "dashboard":
                value = await self.monitor.get_dashboard_data(model.native_id)
                return {
                    "model_id": model_alias,
                    "metrics": self.metric_view(value.current_metrics, model),
                    "metrics_history": [
                        self.metric_view(metric, model) for metric in value.metrics_history[-limit:]
                    ],
                    "active_alerts": [
                        self.alert_view(alert, model) for alert in value.active_alerts[:limit]
                    ],
                    "alert_history": [
                        self.alert_view(alert, model) for alert in value.alert_history[-limit:]
                    ],
                    "truncated": any(
                        len(items) > limit
                        for items in (
                            value.metrics_history,
                            value.active_alerts,
                            value.alert_history,
                        )
                    ),
                    "health_score": value.health_score,
                    "state": self.state(),
                }
            else:
                raise MCPError("invalid_arguments")
        except MCPError:
            raise
        except Exception:
            # Unknown native storage/delivery effects remain visible and latched;
            # no arbitrary provider error, record or endpoint enters the reply.
            self.guard()
            return NativeServiceResult({"state": self.state()}, "native_monitor_incomplete")
        self.guard()
        return self.state()

    async def close(self):
        # Revocation/close stops timers without flushing or sending new alerts.
        await self.monitor.stop()


if __name__ == "__main__":
    from truthound.mcp.broker_worker import main

    main(MLOwner)
