"""Closed native profiling composition; results remain owned full artifacts."""

from typing import Annotated, Literal

from pydantic import Field, JsonValue, model_validator

from truthound.mcp.contracts import Name, WireModel
from truthound.mcp.policy import MCPError
from truthound.mcp.session_contracts import AssetInput
from truthound.mcp.validator_wire import _known_type, contracts, fields, record_fields


class Component(WireModel):
    component: Name
    settings: dict[str, JsonValue] = Field(default_factory=dict)


def component(group, value, *, build=False):
    row = contracts()["profiling"]["components"][group].get(value.component)
    if row is None:
        raise MCPError("unknown_component")
    # Custom patterns execute regex. Native reviewed extension objects require
    # an explicit trust binding, not a dictionary smuggled into discovery.
    if value.settings.get("patterns") is not None:
        raise MCPError("operator_configuration_required")
    settings = fields(row["fields"], value.settings, build=build)
    return _known_type(row["target"])(**settings) if build else None


class Profile(AssetInput):
    config: dict[str, JsonValue] = Field(default_factory=dict)
    options: dict[str, JsonValue] = Field(default_factory=dict)
    column_analyzers: Annotated[list[Component], Field(min_length=1, max_length=20)] | None = None
    table_analyzers: Annotated[list[Component], Field(min_length=1, max_length=20)] | None = None
    type_inferrers: Annotated[list[Component], Field(min_length=1, max_length=20)] | None = None
    name: str = ""
    source: str = ""

    @model_validator(mode="after")
    def configuration(self):
        row = contracts()["profiling"]
        fields(record_fields(row["config"]), self.config)
        options = row["components"]["profiler"]["DataProfiler"]["fields"]
        forbidden = {"config", "column_profiler", "table_analyzers"}
        if forbidden.intersection(self.options):
            raise MCPError("invalid_arguments")
        fields({key: value for key, value in options.items() if key not in forbidden}, self.options)
        for group in ("column_analyzers", "table_analyzers", "type_inferrers"):
            for value in getattr(self, group) or ():
                component(group, value)
        return self


class GenerateRules(Profile):
    generator: Component
    strictness: Literal["loose", "medium", "strict"] = "medium"
    column: Name | None = None
    representation: Literal["rules", "validator_configs"] = "rules"

    @model_validator(mode="after")
    def generator_configuration(self):
        component("rule_generators", self.generator)
        return self


def generate_rules(arguments, frame):
    from truthound.profiler import Strictness

    profile = execute(arguments, frame)
    generator = component("rule_generators", arguments.generator, build=True)
    _known_type("truthound.profiler.generators.base.GeneratedRule")
    _known_type("truthound.profiler.base.TableProfile")
    # Call one explicit generator: the native suite aggregator suppresses
    # generator exceptions, which would hide an incomplete result here.
    strictness = Strictness(arguments.strictness)
    rules = (
        generator.generate(profile, strictness)
        if arguments.column is None
        else generator.generate_for_column(profile[arguments.column], strictness)
    )
    if arguments.representation == "validator_configs":
        return [
            {"validator_class": rule.validator_class, "config": rule.to_validator_config()}
            for rule in rules
        ]
    return [rule.to_dict() for rule in rules]


def execute(arguments, frame):
    from truthound.profiler import ColumnProfiler, DataProfiler

    row = contracts()["profiling"]
    config_type = _known_type(row["config"]["target"])
    config = config_type(**fields(record_fields(row["config"]), arguments.config, build=True))
    groups = {
        group: None
        if getattr(arguments, group) is None
        else [component(group, value, build=True) for value in getattr(arguments, group)]
        for group in ("column_analyzers", "table_analyzers", "type_inferrers")
    }
    # Verify source pins for the native composition classes, not only leaf analyzers.
    _known_type("truthound.profiler.column_profiler.ColumnProfiler")
    cls = _known_type("truthound.profiler.table_profiler.DataProfiler")
    assert cls is DataProfiler
    columns = ColumnProfiler(
        analyzers=groups["column_analyzers"], type_inferrers=groups["type_inferrers"], config=config
    )
    options = fields(
        row["components"]["profiler"]["DataProfiler"]["fields"], arguments.options, build=True
    )
    return cls(
        config=config, column_profiler=columns, table_analyzers=groups["table_analyzers"], **options
    ).profile(frame.lazy(), name=arguments.name, source=arguments.source)
