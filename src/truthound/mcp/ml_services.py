"""Approved native model monitoring with fixed input mappings and alert components."""

from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from truthound.mcp.approved_sources import SourceBindingOpen
from truthound.mcp.bound_services import BoundServiceSession
from truthound.mcp.checkpoint_services import ServiceProcess
from truthound.mcp.contracts import Name
from truthound.mcp.session_contracts import Handle


@dataclass
class MonitoringModel:
    """Factory-selected native identity, config and asset column mappings.

    Feature keys are public aliases passed to Core; values are physical asset
    columns. Prediction, actual and latency columns never come from wire code.
    """

    native_id: str
    config: Any
    features: dict[str, str]
    prediction_column: str
    actual_column: str | None = None
    latency_column: str | None = None
    metric_aliases: dict[str, str] = field(default_factory=dict)


@dataclass
class MLRuntime:
    monitor: Any
    models: dict[str, MonitoringModel]
    rules: dict[str, Any] = field(default_factory=dict)
    handlers: dict[str, Any] = field(default_factory=dict)


class MLCommand(SourceBindingOpen):
    service_id: Handle
    command: Literal[
        "inspect",
        "status",
        "register",
        "unregister",
        "start",
        "stop",
        "shutdown",
        "record",
        "feedback",
        "reference",
        "collect",
        "evaluate",
        "discard_uncertain",
        "metrics",
        "dashboard",
        "rule_add",
        "rule_remove",
        "handler_add",
        "handler_remove",
    ]
    model_id: Name | None = None
    asset_id: Handle | None = None
    prediction_id: Name | None = None
    rule_id: Name | None = None
    handler_id: Name | None = None
    offset: Annotated[int, Field(ge=0, le=1_000_000)] = 0
    limit: Annotated[int, Field(ge=1, le=1000)] = 100

    @model_validator(mode="after")
    def shape(self):
        required = {
            "model_id": {
                "register",
                "unregister",
                "record",
                "feedback",
                "reference",
                "collect",
                "evaluate",
                "discard_uncertain",
                "metrics",
                "dashboard",
            },
            "asset_id": {"record", "feedback", "reference"},
            "prediction_id": {"feedback"},
            "rule_id": {"rule_add", "rule_remove"},
            "handler_id": {"handler_add", "handler_remove"},
        }
        for name, commands in required.items():
            if (self.command in commands) != (getattr(self, name) is not None):
                raise ValueError("Field does not match command")
        if "offset" in self.model_fields_set and self.command not in {
            "record",
            "feedback",
            "reference",
        }:
            raise ValueError("Field does not match command")
        if "limit" in self.model_fields_set and self.command not in {
            "inspect",
            "record",
            "reference",
            "dashboard",
        }:
            raise ValueError("Field does not match command")
        return self


class MLProcess(ServiceProcess):
    worker_module = "truthound.mcp.ml_worker"

    def validate_reply(self, result):
        if result == {"ok": True, "native_failed": "native_monitor_incomplete"}:
            return result
        return super().validate_reply(result)


class MLServices(BoundServiceSession):
    group = "ml_services"
    process_type = MLProcess
    limit_commands = {"inspect", "record", "reference", "dashboard"}
    effects = (
        "installed_extension_execute",
        "operator_selected_native_model_monitor",
        "operator_configured_prediction_and_reference_asset_read",
        "native_metric_store_read_write_retention_delete",
        "native_periodic_collection_and_fresh_snapshot_alert_evaluation_until_stop_or_release",
        "operator_selected_alert_rule_and_notification_handler_execute",
        "explicit_pending_prediction_discard_without_external_effect_rollback",
        "sensitive_data_artifact",
    )

    def input_resources(self, arguments):
        if arguments.command in {"record", "feedback", "reference"}:
            return {"asset_path": self.session.resources.get(arguments.asset_id, "asset")}
        return {}

    def reply_error(self, arguments, reply):
        return reply.get("native_failed")
