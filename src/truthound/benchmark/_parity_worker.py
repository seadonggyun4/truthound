"""Child-process worker for framework-specific parity observations."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sqlite3
import statistics
import sys
import threading
import time
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any
from uuid import uuid4

from truthound.benchmark.parity import (
    FrameworkObservation,
    GreatExpectationsAdapter,
    TruthoundAdapter,
)
from truthound.benchmark.workloads import ParityWorkload, WorkloadBackend, load_workload


def _sum_truthound_issue_count(run_result: Any) -> int:
    total = 0
    for issue in getattr(run_result, "issues", ()):
        issue_count = getattr(issue, "count", 0)
        total += int(issue_count) if int(issue_count) > 0 else 1
    return total


class _PeakRSSMonitor:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak_rss = 0

    @property
    def peak_rss(self) -> int:
        return self._peak_rss

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._peak_rss = max(self._peak_rss, self._rss_bytes())
            time.sleep(0.01)

    def _rss_bytes(self) -> int:
        try:
            import psutil

            return psutil.Process().memory_info().rss
        except Exception:
            try:
                import resource

                usage = resource.getrusage(resource.RUSAGE_SELF)
                max_rss = int(usage.ru_maxrss)
                if sys.platform == "darwin":
                    return max_rss
                return max_rss * 1024
            except Exception:
                return 0


def _framework_version(distribution_name: str, fallback: str) -> str:
    try:
        return importlib_metadata.version(distribution_name)
    except importlib_metadata.PackageNotFoundError:
        return fallback


def _materialize_sql_table(workload: ParityWorkload, artifact_dir: Path) -> Path:
    table_name = workload.table_name or "truthound_benchmark"
    data = workload.load_polars()

    if workload.backend == WorkloadBackend.SQLITE:
        database_path = artifact_dir / "workload.sqlite"
        conn = sqlite3.connect(database_path)
        try:
            columns = []
            for name, dtype in data.schema.items():
                dtype_name = str(dtype)
                if "Int" in dtype_name or "UInt" in dtype_name:
                    sql_type = "INTEGER"
                elif "Float" in dtype_name or "Decimal" in dtype_name:
                    sql_type = "REAL"
                elif "Boolean" in dtype_name:
                    sql_type = "BOOLEAN"
                else:
                    sql_type = "TEXT"
                columns.append(f"{name} {sql_type}")
            conn.execute(f"CREATE TABLE {table_name} ({', '.join(columns)})")
            placeholders = ", ".join(["?"] * len(data.columns))
            rows = [
                tuple(row.get(column) for column in data.columns)
                for row in data.to_dicts()
            ]
            conn.executemany(
                f"INSERT INTO {table_name} VALUES ({placeholders})",
                rows,
            )
            conn.commit()
        finally:
            conn.close()
        return database_path

    if workload.backend == WorkloadBackend.DUCKDB_SHADOW:
        try:
            import duckdb
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "duckdb is not installed. Install with truthound[benchmarks]."
            ) from exc
        database_path = artifact_dir / "workload.duckdb"
        conn = duckdb.connect(str(database_path))
        try:
            columns = []
            for name, dtype in data.schema.items():
                dtype_name = str(dtype)
                if "Int" in dtype_name or "UInt" in dtype_name:
                    sql_type = "BIGINT"
                elif "Float" in dtype_name or "Decimal" in dtype_name:
                    sql_type = "DOUBLE"
                elif "Boolean" in dtype_name:
                    sql_type = "BOOLEAN"
                else:
                    sql_type = "VARCHAR"
                columns.append(f"{name} {sql_type}")
            conn.execute(f"CREATE TABLE {table_name} ({', '.join(columns)})")
            placeholders = ", ".join(["?"] * len(data.columns))
            rows = [
                tuple(row.get(column) for column in data.columns)
                for row in data.to_dicts()
            ]
            conn.executemany(
                f"INSERT INTO {table_name} VALUES ({placeholders})",
                rows,
            )
        finally:
            conn.close()
        return database_path

    raise ValueError(f"Unsupported SQL backend: {workload.backend.value}")


def _truthound_execute(
    workload: ParityWorkload,
    *,
    project_root: Path,
    warm_iterations: int,
    artifact_dir: Path,
) -> FrameworkObservation:
    import truthound as th
    from truthound.datasources import get_sql_datasource

    project_root.mkdir(parents=True, exist_ok=True)
    schema = workload.truthound.build_schema()

    sql_database_path: Path | None = None
    sql_source: Any | None = None
    if workload.backend != WorkloadBackend.LOCAL:
        sql_database_path = _materialize_sql_table(workload, artifact_dir)
        if workload.backend == WorkloadBackend.SQLITE:
            connection_string = f"sqlite:///{sql_database_path}"
        elif workload.backend == WorkloadBackend.DUCKDB_SHADOW:
            connection_string = f"duckdb:///{sql_database_path}"
        else:  # pragma: no cover - defensive guard for future backends
            raise ValueError(f"Unsupported SQL backend: {workload.backend.value}")
        sql_source = get_sql_datasource(connection_string, table=workload.table_name)

    cold_duration = 0.0
    warm_durations: list[float] = []
    final_result: Any | None = None
    cwd = Path.cwd()
    monitor = _PeakRSSMonitor()
    monitor.start()
    try:
        os.chdir(project_root)
        for iteration in range(warm_iterations + 1):
            started = time.perf_counter()
            if workload.backend == WorkloadBackend.LOCAL:
                payload = workload.load_polars()
                final_result = th.check(
                    payload,
                    validators=list(workload.truthound.validators),
                    validator_config=workload.truthound.validator_config or None,
                    schema=schema,
                )
            else:
                final_result = th.check(
                    source=sql_source,
                    validators=list(workload.truthound.validators),
                    validator_config=workload.truthound.validator_config or None,
                    schema=schema,
                    pushdown=workload.truthound.pushdown,
                )
            elapsed = time.perf_counter() - started
            if iteration == 0:
                cold_duration = elapsed
            else:
                warm_durations.append(elapsed)
    finally:
        os.chdir(cwd)
        monitor.stop()

    if final_result is None:
        raise RuntimeError("Truthound benchmark produced no validation result.")

    observed_issue_count = _sum_truthound_issue_count(final_result)
    warm_median = statistics.median(warm_durations) if warm_durations else cold_duration
    return FrameworkObservation(
        framework="truthound",
        framework_version=_framework_version("truthound", TruthoundAdapter().framework_version()),
        workload_id=workload.id,
        dataset_fingerprint=workload.dataset_fingerprint,
        backend=workload.backend.value,
        exactness=workload.exactness,
        cold_start_seconds=cold_duration,
        warm_median_seconds=warm_median,
        peak_rss_bytes=monitor.peak_rss,
        correctness_passed=(
            final_result.success == workload.expected.success
            and observed_issue_count == workload.expected.issue_count
        ),
        expected_issue_count=workload.expected.issue_count,
        observed_issue_count=observed_issue_count,
        artifact_paths={
            "project_root": str(project_root),
            "workspace_root": str(project_root / ".truthound"),
            "run_artifact": str(final_result.metadata.get("context_run_artifact", "")),
            "docs_artifact": str(final_result.metadata.get("context_docs_artifact", "")),
            "database_path": str(sql_database_path) if sql_database_path is not None else "",
        },
        metadata={
            "row_count": workload.row_count,
            "workload_class": workload.benchmark_class.value,
            "success": final_result.success,
            "execution_mode": getattr(final_result, "execution_mode", "unknown"),
            "warm_iterations": warm_iterations,
            "run_id": getattr(final_result, "run_id", ""),
        },
    )


def _gx_get_context(gx: Any) -> Any:
    try:
        return gx.get_context(mode="ephemeral")
    except TypeError:
        return gx.get_context()


def _gx_add_pandas_datasource(context: Any, *, name: str) -> Any:
    if hasattr(context, "data_sources") and hasattr(context.data_sources, "add_pandas"):
        return context.data_sources.add_pandas(name=name)
    if hasattr(context, "sources") and hasattr(context.sources, "add_pandas"):
        return context.sources.add_pandas(name=name)
    raise RuntimeError("Current Great Expectations version does not expose add_pandas().")


def _gx_add_sqlite_datasource(context: Any, *, name: str, connection_string: str) -> Any:
    if hasattr(context, "data_sources") and hasattr(context.data_sources, "add_sqlite"):
        return context.data_sources.add_sqlite(name=name, connection_string=connection_string)
    if hasattr(context, "sources") and hasattr(context.sources, "add_sqlite"):
        return context.sources.add_sqlite(name=name, connection_string=connection_string)
    if hasattr(context, "data_sources") and hasattr(context.data_sources, "add_sql"):
        return context.data_sources.add_sql(name=name, connection_string=connection_string)
    if hasattr(context, "sources") and hasattr(context.sources, "add_sql"):
        return context.sources.add_sql(name=name, connection_string=connection_string)
    raise RuntimeError("Current Great Expectations version does not expose add_sqlite()/add_sql().")


def _gx_local_batch(datasource: Any, *, asset_name: str, dataframe: Any) -> Any:
    asset = datasource.add_dataframe_asset(name=asset_name)
    if hasattr(asset, "add_batch_definition_whole_dataframe"):
        batch_definition = asset.add_batch_definition_whole_dataframe("whole_dataframe")
        return batch_definition.get_batch(batch_parameters={"dataframe": dataframe})
    if hasattr(asset, "get_batch"):
        return asset.get_batch(batch_parameters={"dataframe": dataframe})
    raise RuntimeError("Current Great Expectations dataframe asset does not expose a batch API.")


def _gx_sql_batch(datasource: Any, *, asset_name: str, table_name: str) -> Any:
    asset = datasource.add_table_asset(name=asset_name, table_name=table_name)
    if hasattr(asset, "add_batch_definition_whole_table"):
        batch_definition = asset.add_batch_definition_whole_table("whole_table")
        return batch_definition.get_batch()
    if hasattr(asset, "get_batch"):
        return asset.get_batch()
    raise RuntimeError("Current Great Expectations table asset does not expose a batch API.")


def _gx_expectation_object(gx_expectations: Any, spec: Any) -> Any:
    mapping = {
        "not_null": "ExpectColumnValuesToNotBeNull",
        "unique": "ExpectColumnValuesToBeUnique",
        "between": "ExpectColumnValuesToBeBetween",
        "column_exists": "ExpectColumnToExist",
        "column_type": "ExpectColumnValuesToBeOfType",
    }
    if spec.type not in mapping:
        raise ValueError(f"Unsupported GX expectation mapping: {spec.type}")
    cls = getattr(gx_expectations, mapping[spec.type])
    kwargs = dict(spec.params)
    if spec.column is not None:
        kwargs["column"] = spec.column
    if spec.type == "column_type":
        kwargs["type_"] = str(kwargs.pop("dtype"))
    return cls(**kwargs)


def _gx_extract_validation(result: Any) -> tuple[bool, int, int]:
    if hasattr(result, "to_json_dict"):
        payload = result.to_json_dict()
    elif isinstance(result, dict):
        payload = result
    else:
        payload = {
            "success": getattr(result, "success", False),
            "result": getattr(result, "result", {}),
        }

    success = bool(payload.get("success", False))
    result_payload = payload.get("result", {}) or {}
    unexpected_count = result_payload.get("unexpected_count")
    if unexpected_count is None:
        partial = result_payload.get("partial_unexpected_list")
        if isinstance(partial, list):
            unexpected_count = len(partial)
    raw_observed = int(unexpected_count) if unexpected_count is not None else (0 if success else 1)
    # Truthound's canonical issue count for release parity is check-level, not row-level.
    observed = 0 if success else 1
    return success, observed, raw_observed


def _gx_execute(
    workload: ParityWorkload,
    *,
    project_root: Path,
    warm_iterations: int,
    artifact_dir: Path,
) -> FrameworkObservation:
    adapter = GreatExpectationsAdapter()
    available, reason = adapter.is_available()
    if not available:
        return FrameworkObservation.unavailable(
            framework="gx",
            workload=workload,
            reason=reason or "great-expectations is unavailable.",
        )

    import great_expectations as gx

    try:
        gx_expectations = importlib.import_module("great_expectations.expectations")
    except ImportError as exc:  # pragma: no cover - optional dependency API mismatch
        raise RuntimeError("Unable to import great_expectations.expectations.") from exc

    project_root.mkdir(parents=True, exist_ok=True)

    sql_database_path: Path | None = None
    if workload.backend != WorkloadBackend.LOCAL:
        sql_database_path = _materialize_sql_table(workload, artifact_dir)

    cold_duration = 0.0
    warm_durations: list[float] = []
    final_success = False
    final_issue_count = 0
    final_raw_issue_count = 0
    cwd = Path.cwd()
    monitor = _PeakRSSMonitor()
    monitor.start()
    try:
        os.chdir(project_root)
        for iteration in range(warm_iterations + 1):
            started = time.perf_counter()
            context = _gx_get_context(gx)
            run_id = uuid4().hex[:8]
            datasource_name = f"truthound_parity_{workload.id}_{iteration}_{run_id}"
            asset_name = f"asset_{workload.id}_{iteration}_{run_id}"

            if workload.backend == WorkloadBackend.LOCAL:
                dataframe = workload.load_pandas()
                datasource = _gx_add_pandas_datasource(context, name=datasource_name)
                batch = _gx_local_batch(
                    datasource,
                    asset_name=asset_name,
                    dataframe=dataframe,
                )
            elif workload.backend == WorkloadBackend.SQLITE:
                datasource = _gx_add_sqlite_datasource(
                    name=datasource_name,
                    context=context,
                    connection_string=f"sqlite:///{sql_database_path}",
                )
                batch = _gx_sql_batch(
                    datasource,
                    asset_name=asset_name,
                    table_name=workload.table_name or "truthound_benchmark",
                )
            else:
                return FrameworkObservation.unavailable(
                    framework="gx",
                    workload=workload,
                    reason="DuckDB shadow workloads are Truthound-only advisory benchmarks.",
                )

            issue_total = 0
            raw_issue_total = 0
            successes: list[bool] = []
            for expectation_spec in workload.gx.expectations:
                expectation = _gx_expectation_object(gx_expectations, expectation_spec)
                validation = batch.validate(expectation)
                expectation_success, observed, raw_observed = _gx_extract_validation(validation)
                successes.append(expectation_success)
                issue_total += observed
                raw_issue_total += raw_observed

            final_success = all(successes)
            final_issue_count = issue_total
            final_raw_issue_count = raw_issue_total
            elapsed = time.perf_counter() - started
            if iteration == 0:
                cold_duration = elapsed
            else:
                warm_durations.append(elapsed)
    finally:
        os.chdir(cwd)
        monitor.stop()

    warm_median = statistics.median(warm_durations) if warm_durations else cold_duration
    return FrameworkObservation(
        framework="gx",
        framework_version=_framework_version("great-expectations", adapter.framework_version()),
        workload_id=workload.id,
        dataset_fingerprint=workload.dataset_fingerprint,
        backend=workload.backend.value,
        exactness=workload.exactness,
        cold_start_seconds=cold_duration,
        warm_median_seconds=warm_median,
        peak_rss_bytes=monitor.peak_rss,
        correctness_passed=(
            final_success == workload.expected.success
            and final_issue_count == workload.expected.issue_count
        ),
        expected_issue_count=workload.expected.issue_count,
        observed_issue_count=final_issue_count,
        artifact_paths={
            "project_root": str(project_root),
            "database_path": str(sql_database_path) if sql_database_path is not None else "",
        },
        metadata={
            "row_count": workload.row_count,
            "workload_class": workload.benchmark_class.value,
            "success": final_success,
            "warm_iterations": warm_iterations,
            "raw_unexpected_count": final_raw_issue_count,
        },
    )


def execute_framework_observation(
    workload: ParityWorkload,
    *,
    framework: str,
    artifact_dir: str | Path,
    warm_iterations: int,
) -> FrameworkObservation:
    """Execute one framework/workload observation in-process."""

    root = Path(artifact_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    project_root = root / "workspace"

    if not workload.supports_framework(framework):
        return FrameworkObservation.unavailable(
            framework=framework,
            workload=workload,
            reason=f"Framework '{framework}' is not supported for workload '{workload.id}'.",
        )

    if framework == "truthound":
        return _truthound_execute(
            workload,
            project_root=project_root,
            warm_iterations=warm_iterations,
            artifact_dir=root,
        )
    if framework == "gx":
        return _gx_execute(
            workload,
            project_root=project_root,
            warm_iterations=warm_iterations,
            artifact_dir=root,
        )
    return FrameworkObservation.unavailable(
        framework=framework,
        workload=workload,
        reason=f"Unknown framework '{framework}'.",
    )


@dataclass(frozen=True)
class WorkerArgs:
    manifest: Path
    framework: str
    artifact_dir: Path
    output: Path
    warm_iterations: int


def _parse_args() -> WorkerArgs:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--framework", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--warm-iterations", required=True, type=int)
    args = parser.parse_args()
    return WorkerArgs(
        manifest=Path(args.manifest).resolve(),
        framework=str(args.framework),
        artifact_dir=Path(args.artifact_dir).resolve(),
        output=Path(args.output).resolve(),
        warm_iterations=int(args.warm_iterations),
    )


def main() -> None:
    args = _parse_args()
    workload = load_workload(args.manifest)
    observation = execute_framework_observation(
        workload,
        framework=args.framework,
        artifact_dir=args.artifact_dir,
        warm_iterations=args.warm_iterations,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(observation.to_dict(), indent=2), encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess CLI
    main()
