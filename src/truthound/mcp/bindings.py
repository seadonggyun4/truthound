"""Private operator-selected constructor templates, never model-selected imports."""

import hashlib
import json

from pydantic import Field, JsonValue

from truthound.mcp.approval import private_read
from truthound.mcp.contracts import Name, WireModel
from truthound.mcp.plugin_bindings import PluginBinding
from truthound.mcp.policy import MCPError
from truthound.mcp.schema_audit import TermBinding
from truthound.mcp.validator_wire import callback_handles, settings, spec


class ValidatorBinding(WireModel):
    validator: Name
    settings: dict[str, JsonValue] = Field(default_factory=dict)
    object_parameters: dict[Name, Name] = Field(default_factory=dict)


class Bindings(WireModel):
    validators: dict[Name, ValidatorBinding] = Field(default_factory=dict)
    plugins: dict[Name, PluginBinding] = Field(default_factory=dict)
    schema_audit: dict[Name, list[TermBinding]] = Field(default_factory=dict)
    sources: dict[Name, Name] = Field(default_factory=dict)


def load(policy):
    if policy.control_directory is None:
        raise MCPError("operator_configuration_required")
    try:
        raw = private_read(policy.control_directory / "bindings.json", 1024 * 1024)
        document = Bindings.model_validate(json.loads(raw))
    except (FileNotFoundError, ValueError, OSError):
        raise MCPError("operator_configuration_required") from None
    return document, hashlib.sha256(raw).hexdigest()


def resolve(policy, checks):
    if not any(check.binding_id for check in checks or ()):
        return (
            [check.settings for check in checks or ()],
            set(),
            None,
            {},
            [{} for _ in checks or ()],
        )
    document, digest = load(policy)
    resolved, trusted, objects = [], set(), []
    capabilities = {}
    for plugin in document.plugins.values():
        for exposed in plugin.capabilities:
            if exposed in capabilities:
                raise MCPError("invalid_control_state")
            capabilities[exposed] = True
    needed = set()
    for index, check in enumerate(checks or ()):
        if check.binding_id is None:
            resolved.append(check.settings)
            objects.append({})
            continue
        bound = document.validators.get(check.binding_id)
        if bound is None or bound.validator != check.validator:
            raise MCPError("invalid_binding")
        if (set(bound.settings) | set(bound.object_parameters)) & set(check.settings):
            raise MCPError("binding_override_denied")
        shapes = spec(check.validator)["fields"]
        if set(bound.object_parameters) - set(shapes) or set(bound.settings) & set(
            bound.object_parameters
        ):
            raise MCPError("invalid_binding")
        if any(key not in capabilities for key in bound.object_parameters.values()):
            raise MCPError("invalid_binding")
        combined = {**bound.settings, **check.settings}
        if any(
            row["required"] and key not in combined and key not in bound.object_parameters
            for key, row in shapes.items()
        ):
            raise MCPError("invalid_binding")
        settings(
            check.validator,
            combined,
            trusted=True,
            callbacks=capabilities,
            partial=bool(bound.object_parameters),
        )
        needed |= callback_handles(
            check.validator,
            combined,
            trusted=True,
            callbacks=capabilities,
            partial=bool(bound.object_parameters),
        )
        needed |= set(bound.object_parameters.values())
        resolved.append(combined)
        objects.append(bound.object_parameters)
        trusted.add(index)
    plugins = {
        name: {
            **plugin.model_dump(),
            "capabilities": {
                key: value for key, value in plugin.capabilities.items() if key in needed
            },
        }
        for name, plugin in document.plugins.items()
        if set(plugin.capabilities) & needed
    }
    return resolved, trusted, digest, plugins, objects
