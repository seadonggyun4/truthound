"""Operator-bound rate budgets and secret metadata; never return secret material."""

from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from truthound.mcp.approved_sources import SourceBindingOpen
from truthound.mcp.bound_services import BoundServiceSession
from truthound.mcp.checkpoint_services import ServiceProcess
from truthound.mcp.contracts import Name
from truthound.mcp.session_contracts import Handle


@dataclass
class RateLimitRuntime:
    limiter: Any
    subjects: dict[str, str]
    contexts: dict[str, Any] = field(default_factory=dict)
    policies: dict[str, Any] = field(default_factory=dict)
    quotas: dict[str, Any] = field(default_factory=dict)
    policy_registry: Any = None
    quota_manager: Any = None


@dataclass
class SecretProviderSlot:
    provider: Any
    name: str
    priority: int = 100


@dataclass
class SecretRuntime:
    manager: Any
    references: dict[str, Any]
    providers: dict[str, SecretProviderSlot] = field(default_factory=dict)


class RateLimitCommand(SourceBindingOpen):
    service_id: Handle
    command: Literal[
        "catalog",
        "config",
        "acquire",
        "peek",
        "reset",
        "quota_register",
        "quota_check",
        "quota_consume",
        "quota_usage",
        "quota_reset",
        "policy_register",
        "policy_unregister",
        "policy_resolve",
    ]
    subject_id: Name | None = None
    quota_id: Name | None = None
    policy_id: Name | None = None
    context_id: Name | None = None
    tokens: Annotated[int, Field(ge=1, le=1_000_000)] = 1
    wait: bool = False
    timeout_seconds: Annotated[float, Field(gt=0, le=10, allow_inf_nan=False)] | None = None
    limit: Annotated[int, Field(ge=1, le=1000)] = 100
    offset: Annotated[int, Field(ge=0)] = 0

    @model_validator(mode="after")
    def shape(self):
        subjects = {
            "acquire",
            "peek",
            "reset",
            "quota_check",
            "quota_consume",
            "quota_usage",
            "quota_reset",
        }
        quotas = {"quota_register", "quota_check", "quota_consume", "quota_usage", "quota_reset"}
        if (self.command in subjects) != (self.subject_id is not None):
            raise ValueError("subject_id does not match command")
        if (self.command in quotas) != (self.quota_id is not None):
            raise ValueError("quota_id does not match command")
        if (self.command in {"policy_register", "policy_unregister"}) != (
            self.policy_id is not None
        ):
            raise ValueError("policy_id does not match command")
        if self.command == "policy_resolve" and self.context_id is None:
            raise ValueError("policy_resolve requires an operator context")
        for key, commands in {
            "context_id": {"acquire", "peek", "policy_resolve"},
            "tokens": {"acquire", "quota_check", "quota_consume"},
            "wait": {"acquire"},
            "timeout_seconds": {"acquire"},
            "limit": {"catalog"},
            "offset": {"catalog"},
        }.items():
            if key in self.model_fields_set and self.command not in commands:
                raise ValueError("Field does not match command")
        if self.wait != (self.timeout_seconds is not None):
            raise ValueError("Waiting requires an explicit bounded timeout")
        return self


class SecretCommand(SourceBindingOpen):
    service_id: Handle
    command: Literal["catalog", "resolve", "cache_clear", "provider_add", "provider_remove"]
    reference_id: Name | None = None
    provider_id: Name | None = None
    limit: Annotated[int, Field(ge=1, le=1000)] = 100
    offset: Annotated[int, Field(ge=0)] = 0

    @model_validator(mode="after")
    def shape(self):
        if (self.command == "resolve") != (self.reference_id is not None):
            raise ValueError("reference_id is only required by resolve")
        if self.command in {"provider_add", "provider_remove"} and self.provider_id is None:
            raise ValueError("Provider selection is required")
        if self.provider_id is not None and self.command not in {
            "provider_add",
            "provider_remove",
            "cache_clear",
        }:
            raise ValueError("provider_id does not match command")
        if self.command != "catalog" and self.model_fields_set & {"limit", "offset"}:
            raise ValueError("Pagination is only supported by catalog")
        return self


class GovernanceProcess(ServiceProcess):
    worker_module = "truthound.mcp.governance_worker"


class RateLimitServices(BoundServiceSession):
    group = "rate_limit_services"
    process_type = GovernanceProcess
    limit_commands = {"catalog"}
    effects = (
        "installed_extension_execute",
        "native_rate_budget_read_consume_reset",
        "operator_selected_policy_quota_storage",
    )


class SecretServices(BoundServiceSession):
    group = "secret_services"
    process_type = GovernanceProcess
    limit_commands = {"catalog"}
    effects = (
        "installed_extension_execute",
        "operator_selected_secret_provider_read",
        "native_secret_cache_provider_lifecycle",
    )
