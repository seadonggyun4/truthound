"""Approved loopback native dashboard lifecycle; all configuration is factory-owned."""

from typing import Annotated, Literal

from pydantic import Field, model_validator

from truthound.mcp.approved_sources import SourceBindingOpen
from truthound.mcp.bound_services import BoundServiceSession
from truthound.mcp.checkpoint_services import ServiceProcess
from truthound.mcp.session_contracts import Handle


class DashboardCommand(SourceBindingOpen):
    service_id: Handle
    command: Literal["start", "stop", "status"]
    timeout_seconds: Annotated[float, Field(gt=0, le=30, allow_inf_nan=False)] = 10

    @model_validator(mode="after")
    def shape(self):
        if self.command != "start" and "timeout_seconds" in self.model_fields_set:
            raise ValueError("timeout_seconds is only supported by start")
        return self


class DashboardProcess(ServiceProcess):
    worker_module = "truthound.mcp.dashboard_worker"


class DashboardServices(BoundServiceSession):
    group = "dashboards"
    process_type = DashboardProcess
    effects = (
        "installed_extension_execute",
        "native_loopback_dashboard_server",
        "operator_configured_dashboard_data_access",
    )
