"""Repo-tracked workload manifests for benchmark parity and release gating."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import polars as pl

from truthound.schema import Schema


class WorkloadBackend(str, Enum):
    """Canonical backends for benchmark workloads."""

    LOCAL = "local"
    SQLITE = "sqlite"
    DUCKDB_SHADOW = "duckdb-shadow"


class WorkloadClass(str, Enum):
    """Benchmark classes used by GA gates."""

    LOCAL_EXACT = "local_exact"
    SQL_EXACT = "sql_exact"
    SQL_SHADOW = "sql_shadow"


@dataclass(frozen=True)
class LoaderSpec:
    """Deterministic dataset loading instructions."""

    format: str = "csv"
    null_values: tuple[str, ...] = ("", "NULL")

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> LoaderSpec:
        if not data:
            return cls()
        return cls(
            format=str(data.get("format", "csv")),
            null_values=tuple(str(value) for value in data.get("null_values", ("", "NULL"))),
        )


@dataclass(frozen=True)
class WorkloadExpectation:
    """Expected verdict and issue count for a workload."""

    success: bool
    issue_count: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkloadExpectation:
        return cls(
            success=bool(data.get("success", False)),
            issue_count=int(data.get("issue_count", 0)),
        )


@dataclass(frozen=True)
class TruthoundWorkloadSpec:
    """Truthound-specific execution mapping."""

    validators: tuple[str, ...] = ()
    validator_config: dict[str, dict[str, Any]] = field(default_factory=dict)
    schema: dict[str, Any] | None = None
    pushdown: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> TruthoundWorkloadSpec:
        payload = dict(data or {})
        validator_config = payload.get("validator_config", {})
        return cls(
            validators=tuple(str(value) for value in payload.get("validators", ())),
            validator_config={
                str(name): dict(config)
                for name, config in dict(validator_config).items()
            },
            schema=dict(payload["schema"]) if isinstance(payload.get("schema"), dict) else None,
            pushdown=bool(payload.get("pushdown", False)),
        )

    def build_schema(self) -> Schema | None:
        if self.schema is None:
            return None
        return Schema.from_dict(self.schema)


@dataclass(frozen=True)
class ExpectationSpec:
    """Comparable expectation used by the GX adapter."""

    type: str
    column: str | None = None
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExpectationSpec:
        params = dict(data)
        expectation_type = str(params.pop("type"))
        column = params.pop("column", None)
        return cls(
            type=expectation_type,
            column=str(column) if column is not None else None,
            params=params,
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"type": self.type}
        if self.column is not None:
            payload["column"] = self.column
        payload.update(self.params)
        return payload


@dataclass(frozen=True)
class GXWorkloadSpec:
    """GX-specific execution mapping."""

    expectations: tuple[ExpectationSpec, ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> GXWorkloadSpec:
        payload = dict(data or {})
        expectations = tuple(
            ExpectationSpec.from_dict(item)
            for item in payload.get("expectations", ())
        )
        return cls(expectations=expectations)


@dataclass(frozen=True)
class ParityWorkload:
    """Immutable repo-tracked parity workload."""

    id: str
    name: str
    dataset: str
    backend: WorkloadBackend
    benchmark_class: WorkloadClass
    exactness: str
    suites: tuple[str, ...]
    expected: WorkloadExpectation
    truthound: TruthoundWorkloadSpec
    gx: GXWorkloadSpec
    loader: LoaderSpec = field(default_factory=LoaderSpec)
    table_name: str | None = None
    supported_frameworks: tuple[str, ...] = ("truthound", "gx")
    metadata: dict[str, Any] = field(default_factory=dict)
    manifest_path: Path | None = None

    @property
    def dataset_path(self) -> Path:
        if self.manifest_path is None:
            raise RuntimeError("Workload manifest path is not set.")
        return (self.manifest_path.parent / self.dataset).resolve()

    @property
    def dataset_fingerprint(self) -> str:
        hasher = hashlib.sha256()
        hasher.update(self.dataset_path.read_bytes())
        hasher.update(json.dumps(self.truthound.validator_config, sort_keys=True).encode("utf-8"))
        if self.truthound.schema is not None:
            hasher.update(json.dumps(self.truthound.schema, sort_keys=True).encode("utf-8"))
        return hasher.hexdigest()[:16]

    @property
    def row_count(self) -> int:
        return self.load_polars().height

    def supports_framework(self, framework: str) -> bool:
        return framework in self.supported_frameworks

    def load_polars(self) -> pl.DataFrame:
        if self.loader.format != "csv":
            raise ValueError(f"Unsupported workload format: {self.loader.format}")
        return pl.read_csv(
            self.dataset_path,
            null_values=list(self.loader.null_values),
        )

    def load_pandas(self) -> Any:
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - exercised via optional extras
            raise RuntimeError(
                "pandas is required to load benchmark workloads for GX parity."
            ) from exc
        return pd.read_csv(
            self.dataset_path,
            keep_default_na=True,
            na_values=list(self.loader.null_values),
        )

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        manifest_path: Path,
    ) -> ParityWorkload:
        return cls(
            id=str(data["id"]),
            name=str(data.get("name", data["id"])),
            dataset=str(data["dataset"]),
            backend=WorkloadBackend(str(data["backend"])),
            benchmark_class=WorkloadClass(str(data["benchmark_class"])),
            exactness=str(data.get("exactness", "exact")),
            suites=tuple(str(value) for value in data.get("suites", ())),
            expected=WorkloadExpectation.from_dict(dict(data.get("expected", {}))),
            truthound=TruthoundWorkloadSpec.from_dict(data.get("truthound")),
            gx=GXWorkloadSpec.from_dict(data.get("gx")),
            loader=LoaderSpec.from_dict(data.get("loader")),
            table_name=str(data["table_name"]) if data.get("table_name") is not None else None,
            supported_frameworks=tuple(
                str(value) for value in data.get("supported_frameworks", ("truthound", "gx"))
            ),
            metadata=dict(data.get("metadata", {})),
            manifest_path=manifest_path.resolve(),
        )


PARITY_SUITES: tuple[str, ...] = (
    "pr-fast",
    "nightly-core",
    "nightly-sql",
    "release-ga",
)


def workload_root() -> Path:
    """Return the repo-tracked workload root."""
    override = os.environ.get("TRUTHOUND_BENCHMARK_WORKLOAD_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()

    candidates: list[Path] = []
    cwd = Path.cwd().resolve()
    candidates.append(cwd / "benchmarks" / "workloads")
    candidates.extend(parent / "benchmarks" / "workloads" for parent in cwd.parents)
    module_path = Path(__file__).resolve()
    candidates.extend(parent / "benchmarks" / "workloads" for parent in module_path.parents)

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_dir():
            return resolved

    return candidates[0]


def load_workload(path: str | Path) -> ParityWorkload:
    """Load a single workload manifest."""

    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Workload manifest must contain an object: {manifest_path}")
    return ParityWorkload.from_dict(payload, manifest_path=manifest_path)


def discover_workloads(root: str | Path | None = None) -> list[ParityWorkload]:
    """Discover all repo-tracked parity workloads."""

    base = Path(root) if root is not None else workload_root()
    manifests = sorted(path for path in base.glob("*.json") if path.is_file())
    return [load_workload(path) for path in manifests]


def load_suite_workloads(
    suite: str,
    *,
    backend: str | None = None,
    root: str | Path | None = None,
) -> list[ParityWorkload]:
    """Load workloads for a named parity suite."""

    if suite not in PARITY_SUITES:
        raise ValueError(
            f"Unknown parity suite '{suite}'. Expected one of: {', '.join(PARITY_SUITES)}."
        )

    selected: list[ParityWorkload] = []
    for workload in discover_workloads(root):
        if suite not in workload.suites:
            continue
        if backend is not None and workload.backend.value != backend:
            continue
        selected.append(workload)
    return selected
