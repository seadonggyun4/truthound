from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from truthound.types import ResultFormat, ResultFormatConfig, Severity

if TYPE_CHECKING:
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
        if schema_spec is not None:
            schema_check = schema_spec.to_check_spec(evidence_policy=evidence_policy)
            if schema_check is not None:
                specs.append(schema_check)

        if validators is None:
            from truthound.validators import BUILTIN_VALIDATORS

            for name, validator_cls in BUILTIN_VALIDATORS.items():
                config = validator_config.get(name, {})
                specs.append(
                    _build_check_spec(
                        name=name,
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
            name='truthound-default-suite',
            checks=tuple(specs),
            evidence_policy=evidence_policy,
            severity_policy=severity_policy,
            schema_spec=schema_spec,
            metadata={'compatibility_mode': True},
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
