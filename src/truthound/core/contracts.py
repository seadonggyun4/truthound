from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
import warnings

import polars as pl

from truthound.adapters import to_lazyframe

if TYPE_CHECKING:
    from truthound.datasources.base import BaseDataSource
    from truthound.datasources.sql.base import BaseSQLDataSource
    from truthound.core.suite import CheckSpec


@runtime_checkable
class DataAsset(Protocol):
    name: str
    row_count: int
    column_count: int
    backend_name: str
    metadata: dict[str, Any]
    capabilities: BackendCapabilities

    def to_lazyframe(self) -> pl.LazyFrame:
        ...


@runtime_checkable
class ExecutionBackend(Protocol):
    name: str
    supports_parallel: bool
    supports_pushdown: bool


@runtime_checkable
class MetricRepository(Protocol):
    def get(self, key: str) -> Any | None:
        ...

    def put(self, key: str, value: Any) -> None:
        ...


@runtime_checkable
class ArtifactStore(Protocol):
    def write(self, name: str, payload: Any) -> None:
        ...


@runtime_checkable
class PluginCapability(Protocol):
    name: str

    def attach(self, manager: Any) -> None:
        ...


@runtime_checkable
class CheckSpecFactory(Protocol):
    name: str

    def build_check_specs(self) -> list['CheckSpec']:
        ...


@runtime_checkable
class DataAssetProvider(Protocol):
    name: str

    def get_asset(self) -> DataAsset:
        ...


@dataclass(frozen=True)
class BackendCapabilities:
    parallel: bool = True
    pushdown: bool = False


@dataclass
class LazyFrameDataAsset:
    name: str
    row_count: int
    column_count: int
    backend_name: str = 'polars'
    metadata: dict[str, Any] = field(default_factory=dict)
    lazyframe: pl.LazyFrame | None = None
    source: 'BaseDataSource | None' = None
    sql_source: 'BaseSQLDataSource | None' = None
    capabilities: BackendCapabilities = field(default_factory=BackendCapabilities)

    def to_lazyframe(self) -> pl.LazyFrame:
        if self.lazyframe is None:
            raise RuntimeError('This data asset does not expose an in-memory LazyFrame path.')
        return self.lazyframe


def build_validation_asset(
    data: Any = None,
    source: 'BaseDataSource | None' = None,
    *,
    pushdown: bool | None = None,
) -> LazyFrameDataAsset:
    use_pushdown = False
    sql_source: 'BaseSQLDataSource | None' = None

    if source is not None:
        from truthound.datasources._protocols import DataSourceCapability
        from truthound.datasources.base import BaseDataSource

        if not isinstance(source, BaseDataSource):
            raise ValueError(
                f"source must be a DataSource instance, got {type(source).__name__}"
            )

        if pushdown is True:
            use_pushdown = True
        elif pushdown is None:
            use_pushdown = DataSourceCapability.SQL_PUSHDOWN in source.capabilities

        if use_pushdown:
            try:
                from truthound.datasources.sql.base import BaseSQLDataSource

                if isinstance(source, BaseSQLDataSource):
                    sql_source = source
                else:
                    use_pushdown = False
            except ImportError:
                use_pushdown = False

        if not use_pushdown and source.needs_sampling():
            warnings.warn(
                (
                    f"Data source '{source.name}' has {source.row_count:,} rows, "
                    f"which exceeds the limit of {source.config.max_rows:,}. "
                    'Consider using source.sample() for better performance.'
                ),
                UserWarning,
            )

        source_name = source.name
    else:
        if data is None:
            raise ValueError("Either 'data' or 'source' must be provided")
        source_name = str(data) if isinstance(data, str) else type(data).__name__

    if use_pushdown and sql_source is not None:
        return LazyFrameDataAsset(
            name=source_name,
            row_count=sql_source.row_count or 0,
            column_count=len(sql_source.columns),
            backend_name='sql',
            metadata={'pushdown_enabled': True},
            lazyframe=None,
            source=source,
            sql_source=sql_source,
            capabilities=BackendCapabilities(parallel=True, pushdown=True),
        )

    lf = source.to_polars_lazyframe() if source is not None else to_lazyframe(data)
    polars_schema = lf.collect_schema()
    row_count = lf.select(pl.len()).collect().item()
    column_count = len(polars_schema)

    return LazyFrameDataAsset(
        name=source_name,
        row_count=row_count,
        column_count=column_count,
        backend_name='polars',
        metadata={'pushdown_enabled': False},
        lazyframe=lf,
        source=source,
        sql_source=sql_source,
        capabilities=BackendCapabilities(parallel=True, pushdown=False),
    )
