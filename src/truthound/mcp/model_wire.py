"""Closed Core ML constructors and schema validation, without pickle bridges."""

from truthound.mcp.policy import MCPError
from truthound.mcp.source_identity import class_identity
from truthound.mcp.validator_wire import _json, _known_type, contracts, fields, record_fields


def validate_spec(value, *, depth=0):
    if depth > 4:
        raise MCPError("resource_exhausted")
    row = contracts()["models"].get(value["algorithm"])
    if row is None:
        raise MCPError("unknown_model")
    _json(value["config"])
    fields(record_fields(row["config"]), value["config"])
    children = value.get("detectors")
    if children is not None:
        if value["algorithm"] != "EnsembleAnomalyDetector" or len(children) > 8:
            raise MCPError("invalid_arguments")
        for child in children:
            validate_spec(child, depth=depth + 1)


def build(value):
    validate_spec(value)
    row = contracts()["models"][value["algorithm"]]
    cls = _known_type(row["target"])
    for parent in cls.__mro__:
        target = f"{parent.__module__}.{parent.__qualname__}"
        if target in row["source_types"]:
            digest = class_identity(parent)
            if digest != contracts()["source_ast_sha256"][target]:
                raise MCPError("contract_drift")
    config_type = _known_type(row["config"]["target"])
    config = config_type(**fields(record_fields(row["config"]), value["config"], build=True))
    options = {}
    if value.get("detectors") is not None:
        options["detectors"] = [build(child) for child in value["detectors"]]
    return cls(config=config, **options)
