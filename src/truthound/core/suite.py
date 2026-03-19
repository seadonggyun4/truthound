from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from truthound.types import ResultFormat, ResultFormatConfig, Severity

if TYPE_CHECKING:
    from truthound.context import TruthoundContext
    from truthound.datasources.base import BaseDataSource
    from truthound.schema import Schema
    from truthound.validators.base import Validator


ValidatorFactory = Callable[[], 'Validator']


@dataclass(frozen=True)
class EvidencePolicy:
    result_format: ResultFormatConfig = field(default_factory=ResultFormatConfig)

    @classmethod
    def from_any(
        cls,
        value: str | ResultFormat | ResultFormatConfig,
    ) -> 'EvidencePolicy':
        return cls(result_format=ResultFormatConfig.from_any(value))


@dataclass(frozen=True)
class SeverityPolicy:
    min_severity: Severity | None = None


@dataclass(frozen=True)
class CheckSpec:
    id: str
    name: str
    category: str
    factory: ValidatorFactory
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def build_validator(self) -> 'Validator':
        return self.factory()


@dataclass(frozen=True)
class SchemaSpec:
    schema: 'Schema | None' = None
    auto_learned: bool = False
    source: str | None = None

    def to_check_spec(self, *, evidence_policy: EvidencePolicy) -> CheckSpec | None:
        if self.schema is None:
            return None

        def factory() -> 'Validator':
            from truthound.validators.schema_validator import SchemaValidator

            validator = SchemaValidator(self.schema)
            if hasattr(validator, 'config') and hasattr(validator.config, 'replace'):
                validator.config = validator.config.replace(
                    result_format=evidence_policy.result_format,
                )
            return validator

        return CheckSpec(
            id='schema',
            name='schema',
            category='schema',
            factory=factory,
            tags=('schema',),
            metadata={'auto_learned': self.auto_learned},
        )


@dataclass(frozen=True)
class ValidationSuite:
    name: str = 'default'
    checks: tuple[CheckSpec, ...] = ()
    evidence_policy: EvidencePolicy = field(default_factory=EvidencePolicy)
    severity_policy: SeverityPolicy = field(default_factory=SeverityPolicy)
    schema_spec: SchemaSpec | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_legacy(
        cls,
        *,
        context: "TruthoundContext | None" = None,
        validators: list[str | 'Validator'] | None = None,
        validator_config: dict[str, dict[str, Any]] | None = None,
        schema: str | Path | 'Schema' | None = None,
        auto_schema: bool = False,
        data: Any = None,
        source: 'BaseDataSource | None' = None,
        catch_exceptions: bool = True,
        max_retries: int = 0,
        exclude_columns: list[str] | None = None,
        result_format: str | ResultFormat | ResultFormatConfig = ResultFormat.SUMMARY,
        min_severity: str | Severity | None = None,
    ) -> 'ValidationSuite':
        evidence_policy = EvidencePolicy.from_any(result_format)
        severity_policy = SeverityPolicy(min_severity=_parse_min_severity(min_severity))
        schema_spec = resolve_schema_spec(
            schema=schema,
            auto_schema=auto_schema,
            data=data,
            source=source,
        )
        validator_config = validator_config or {}
        exclude_columns_tuple = tuple(exclude_columns or ())

        specs: list[CheckSpec] = []
        if validators is None:
            auto_specs, schema_spec = AutoSuiteBuilder(
                context=context,
                data=data,
                source=source,
                evidence_policy=evidence_policy,
                catch_exceptions=catch_exceptions,
                max_retries=max_retries,
                exclude_columns=exclude_columns_tuple,
                validator_config=validator_config,
            ).build(schema_spec=schema_spec)
            specs.extend(auto_specs)
        else:
            if schema_spec is not None:
                schema_check = schema_spec.to_check_spec(evidence_policy=evidence_policy)
                if schema_check is not None:
                    specs.append(schema_check)
            for index, validator in enumerate(validators):
                if isinstance(validator, str):
                    from truthound.validators import get_validator

                    validator_cls = get_validator(validator)
                    config = validator_config.get(validator, {})
                    specs.append(
                        _build_check_spec(
                            name=validator,
                            validator_factory=lambda cls=validator_cls, cfg=dict(config): cls(**cfg) if cfg else cls(),
                            category=getattr(validator_cls, 'category', 'general'),
                            evidence_policy=evidence_policy,
                            catch_exceptions=catch_exceptions,
                            max_retries=max_retries,
                            exclude_columns=exclude_columns_tuple,
                            metadata={'config': config},
                        )
                    )
                else:
                    validator_name = getattr(validator, 'name', validator.__class__.__name__.lower())
                    specs.append(
                        _build_check_spec(
                            name=validator_name,
                            validator_factory=lambda instance=validator: _clone_validator(instance),
                            category=getattr(validator, 'category', 'general'),
                            evidence_policy=evidence_policy,
                            catch_exceptions=catch_exceptions,
                            max_retries=max_retries,
                            exclude_columns=exclude_columns_tuple,
                            metadata={'legacy_instance': True, 'index': index},
                        )
                    )

        return cls(
            name='truthound-auto-suite' if validators is None else 'truthound-default-suite',
            checks=tuple(specs),
            evidence_policy=evidence_policy,
            severity_policy=severity_policy,
            schema_spec=schema_spec,
            metadata={
                'compatibility_mode': validators is not None,
                'auto_suite': validators is None,
                'context_root': str(getattr(context, 'root_dir', '')) if context is not None else '',
            },
        )


def resolve_schema_spec(
    *,
    schema: str | Path | 'Schema' | None = None,
    auto_schema: bool = False,
    data: Any = None,
    source: 'BaseDataSource | None' = None,
) -> SchemaSpec | None:
    if schema is None and not auto_schema:
        return None

    from truthound.schema import Schema as SchemaClass, learn

    if schema is not None:
        if isinstance(schema, (str, Path)):
            schema_obj = SchemaClass.load(schema)
        else:
            schema_obj = schema
        return SchemaSpec(schema=schema_obj, auto_learned=False, source='explicit')

    if data is not None:
        try:
            from truthound.cache import get_or_learn_schema

            schema_obj, _ = get_or_learn_schema(data)
        except Exception:
            schema_obj = learn(data=data)
        return SchemaSpec(schema=schema_obj, auto_learned=True, source='cache')

    schema_obj = learn(source=source)
    return SchemaSpec(schema=schema_obj, auto_learned=True, source='source')


def _parse_min_severity(value: str | Severity | None) -> Severity | None:
    if value is None:
        return None
    if isinstance(value, Severity):
        return value
    return Severity(value.lower())


def _clone_validator(validator: 'Validator') -> 'Validator':
    try:
        return deepcopy(validator)
    except Exception:
        return validator


def _configure_validator(
    validator: 'Validator',
    *,
    evidence_policy: EvidencePolicy,
    catch_exceptions: bool,
    max_retries: int,
    exclude_columns: tuple[str, ...],
) -> 'Validator':
    if hasattr(validator, 'config') and hasattr(validator.config, 'replace'):
        existing = tuple(getattr(validator.config, 'exclude_columns', ()) or ())
        merged = tuple(dict.fromkeys(existing + exclude_columns))
        validator.config = validator.config.replace(
            result_format=evidence_policy.result_format,
            catch_exceptions=catch_exceptions,
            max_retries=max_retries,
            exclude_columns=merged,
        )
    return validator


def _build_check_spec(
    *,
    name: str,
    validator_factory: ValidatorFactory,
    category: str,
    evidence_policy: EvidencePolicy,
    catch_exceptions: bool,
    max_retries: int,
    exclude_columns: tuple[str, ...],
    metadata: dict[str, Any],
) -> CheckSpec:
    def factory() -> 'Validator':
        validator = validator_factory()
        return _configure_validator(
            validator,
            evidence_policy=evidence_policy,
            catch_exceptions=catch_exceptions,
            max_retries=max_retries,
            exclude_columns=exclude_columns,
        )

    return CheckSpec(
        id=name,
        name=name,
        category=category,
        factory=factory,
        tags=(category,),
        metadata=metadata,
    )


@dataclass(frozen=True)
class AutoSuiteDecision:
    name: str
    columns: tuple[str, ...] = ()
    reason: str = ''


class AutoSuiteBuilder:
    """Deterministic zero-config suite synthesis for Truthound 3.0."""

    _KEY_PATTERNS = ("id", "key", "uuid", "guid", "email", "code", "slug")
    _NUMERIC_TYPES = (
        "Int8", "Int16", "Int32", "Int64",
        "UInt8", "UInt16", "UInt32", "UInt64",
        "Float32", "Float64", "Decimal",
    )
    _STRING_TYPES = ("String", "Utf8")

    def __init__(
        self,
        *,
        context: "TruthoundContext | None",
        data: Any,
        source: "BaseDataSource | None",
        evidence_policy: EvidencePolicy,
        catch_exceptions: bool,
        max_retries: int,
        exclude_columns: tuple[str, ...],
        validator_config: dict[str, dict[str, Any]],
    ) -> None:
        self._context = context
        self._data = data
        self._source = source
        self._evidence_policy = evidence_policy
        self._catch_exceptions = catch_exceptions
        self._max_retries = max_retries
        self._exclude_columns = exclude_columns
        self._validator_config = validator_config

    def build(
        self,
        *,
        schema_spec: SchemaSpec | None,
    ) -> tuple[list[CheckSpec], SchemaSpec | None]:
        if schema_spec is None:
            schema_spec = self._resolve_baseline_schema_spec()

        specs: list[CheckSpec] = []
        if schema_spec is not None:
            schema_check = schema_spec.to_check_spec(evidence_policy=self._evidence_policy)
            if schema_check is not None:
                specs.append(schema_check)

        decisions = self._build_decisions(schema_spec)
        for decision in decisions:
            specs.append(
                self._make_spec(
                    decision.name,
                    columns=decision.columns,
                    metadata={"auto_reason": decision.reason},
                )
            )

        return specs, schema_spec

    def _resolve_baseline_schema_spec(self) -> SchemaSpec | None:
        if self._context is None:
            return resolve_schema_spec(
                schema=None,
                auto_schema=True,
                data=self._data,
                source=self._source,
            )

        schema, created = self._context.get_or_create_schema_baseline(
            data=self._data,
            source=self._source,
        )
        source_name = "truthound-context:auto-created" if created else "truthound-context:baseline"
        return SchemaSpec(schema=schema, auto_learned=created, source=source_name)

    def _build_decisions(self, schema_spec: SchemaSpec | None) -> list[AutoSuiteDecision]:
        decisions: list[AutoSuiteDecision] = [
            AutoSuiteDecision(name="null", reason="Always validate completeness across discovered columns."),
        ]

        schema = schema_spec.schema if schema_spec is not None else None
        if schema is None:
            decisions.extend([
                AutoSuiteDecision(name="type", reason="String-like columns may encode mixed semantic types."),
                AutoSuiteDecision(name="range", reason="Known numeric column names receive deterministic range checks."),
            ])
            return decisions

        string_columns = tuple(
            name for name, col in schema.columns.items()
            if any(dtype in col.dtype for dtype in self._STRING_TYPES)
        )
        if string_columns:
            decisions.append(
                AutoSuiteDecision(
                    name="type",
                    columns=string_columns,
                    reason="String columns are checked for mixed numeric/string typing anomalies.",
                )
            )

        numeric_columns = tuple(
            name for name, col in schema.columns.items()
            if any(dtype in col.dtype for dtype in self._NUMERIC_TYPES)
        )
        if numeric_columns:
            decisions.append(
                AutoSuiteDecision(
                    name="range",
                    columns=numeric_columns,
                    reason="Numeric columns with known semantic names receive bounded range validation.",
                )
            )

        key_like_columns = tuple(self._infer_key_like_columns(schema))
        if key_like_columns:
            decisions.append(
                AutoSuiteDecision(
                    name="unique",
                    columns=key_like_columns,
                    reason="Key-like columns inferred from schema/profile heuristics are checked for uniqueness.",
                )
            )

        return decisions

    def _infer_key_like_columns(self, schema: "Schema") -> list[str]:
        candidates: list[str] = []
        for name, col in schema.columns.items():
            lowered = name.lower()
            if lowered in self._exclude_columns:
                continue
            if col.unique:
                candidates.append(name)
                continue
            if col.unique_ratio is not None and col.unique_ratio >= 0.98:
                if any(pattern in lowered for pattern in self._KEY_PATTERNS):
                    candidates.append(name)
                    continue
            if any(lowered == pattern or lowered.endswith(f"_{pattern}") for pattern in self._KEY_PATTERNS):
                candidates.append(name)
        return list(dict.fromkeys(candidates))

    def _make_spec(
        self,
        name: str,
        *,
        columns: tuple[str, ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> CheckSpec:
        from truthound.validators import get_validator

        validator_cls = get_validator(name)
        config = dict(self._validator_config.get(name, {}))
        if columns:
            config["columns"] = list(columns)

        return _build_check_spec(
            name=name,
            validator_factory=lambda cls=validator_cls, cfg=dict(config): cls(**cfg) if cfg else cls(),
            category=getattr(validator_cls, 'category', 'general'),
            evidence_policy=self._evidence_policy,
            catch_exceptions=self._catch_exceptions,
            max_retries=self._max_retries,
            exclude_columns=self._exclude_columns,
            metadata={
                "config": config,
                "auto_suite": True,
                **(metadata or {}),
            },
        )
