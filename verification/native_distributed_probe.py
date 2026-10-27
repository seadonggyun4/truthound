"""Probe actual local distributed engines over a closed, synthetic-data protocol.

SDK prerequisites belong to the caller. This helper never discovers or connects
to an external cluster; the caller owns its process and diagnostic directory.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import json
import logging
import os
import sys
import tempfile
import traceback
from pathlib import Path


def failure_diagnostic(error):
    """Return structural SDK diagnostics without messages or full paths."""
    unavailable = False

    def message_of(value):
        nonlocal unavailable
        try:
            return str(value)[:32768]
        except Exception:
            # A Py4J exception may outlive the JVM needed by its __str__.
            unavailable = True
            return ""

    frames = [
        {"file": Path(frame.filename).name, "function": frame.name, "line": frame.lineno}
        for frame in traceback.extract_tb(error.__traceback__)
    ]
    messages = [message_of(error)]
    chain = []
    current = error.__cause__ or error.__context__
    seen = {id(error)}
    while current is not None and id(current) not in seen and len(chain) < 4:
        seen.add(id(current))
        messages.append(message_of(current))
        chain.append({"error_type": type(current).__name__, "frames": [
            {"file": Path(frame.filename).name, "function": frame.name, "line": frame.lineno}
            for frame in traceback.extract_tb(current.__traceback__)
        ]})
        current = current.__cause__ or current.__context__
    message = "\n".join(messages)
    return {"error_type": type(error).__name__, "frames": frames,
            "known_causes": {
                "windows_hadoop_native_missing": any(token in message for token in ("winutils.exe", "HADOOP_HOME", "NativeIO$Windows")),
                "native_jni_missing": "UnsatisfiedLinkError" in message,
                "gcs_start_failed": "Failed to start GCS" in message,
                "home_unresolved": "Could not determine home directory" in message,
            }, "exception_chain": chain, "message_unavailable": unavailable,
            "errno": getattr(error, "errno", None),
            "winerror": getattr(error, "winerror", None)}


@contextlib.contextmanager
def runtime_directory(prefix):
    if os.name == "nt":
        # Ray's driver can retain native log handles after ray.shutdown().
        # The enclosing owner removes this workspace after the driver exits.
        yield tempfile.mkdtemp(prefix=prefix, dir=Path.cwd())
    else:
        with tempfile.TemporaryDirectory(prefix=prefix, dir="/tmp") as directory:
            yield directory


@contextlib.contextmanager
def engine(backend, scratch):
    if backend == "ray":
        print("native-probe:ray-import", flush=True)
        os.environ["RAY_USAGE_STATS_ENABLED"] = "0"
        import ray

        from truthound.execution.distributed.ray_engine import RayEngineConfig, RayExecutionEngine

        print("native-probe:ray-imported", flush=True)

        if ray.is_initialized():
            raise RuntimeError("The probe requires its own local Ray lifetime")
        try:
            print("native-probe:ray-start", flush=True)
            ray.init(
                address="local",
                num_cpus=4,
                num_gpus=0,
                include_dashboard=False,
                object_store_memory=80 * 1024**2,
                _node_ip_address="127.0.0.1",
                _temp_dir=str(scratch),
                logging_level=logging.ERROR,
                log_to_driver=False,
            )
            print("native-probe:ray-started", flush=True)
            dataset = ray.data.from_items(
                [{"x": 1.0, "g": "a"}, {"x": None, "g": "b"}, {"x": 3.0, "g": "a"}]
            )
            print("native-probe:ray-dataset", flush=True)
            with RayExecutionEngine(
                dataset, RayEngineConfig(ray_address="local", num_cpus=4, num_gpus=0)
            ) as native:
                print("native-probe:ray-ready", flush=True)
                yield native
        finally:
            ray.shutdown()
    else:
        print("native-probe:spark-import", flush=True)
        os.environ["PYSPARK_PYTHON"] = sys.executable
        from pyspark import SparkContext
        from pyspark.sql import SparkSession
        from pyspark.sql.types import DoubleType, StringType, StructField, StructType

        from truthound.execution.distributed.spark_engine import (
            SparkEngineConfig,
            SparkExecutionEngine,
        )

        print("native-probe:spark-imported", flush=True)
        if SparkContext._active_spark_context is not None:
            raise RuntimeError("The probe requires its own local Spark lifetime")
        previous_gateway = SparkContext._gateway
        spark = None
        try:
            print("native-probe:spark-start", flush=True)
            spark = (
                SparkSession.builder.master("local[1]")
                .appName("truthound-native-probe")
                .config("spark.driver.bindAddress", "127.0.0.1")
                .config("spark.driver.host", "127.0.0.1")
                .config("spark.ui.enabled", "false")
                .config("spark.sql.shuffle.partitions", "1")
                .config("spark.local.dir", str(scratch / "local"))
                .config("spark.sql.warehouse.dir", (scratch / "warehouse").as_uri())
                .config("spark.pyspark.python", sys.executable)
                .getOrCreate()
            )
            print("native-probe:spark-started", flush=True)
            spark.sparkContext.setLogLevel("ERROR")
            frame = spark.createDataFrame(
                [(1.0, "a"), (None, "b"), (3.0, "a")],
                StructType([StructField("x", DoubleType(), True), StructField("g", StringType(), True)]),
            )
            print("native-probe:spark-frame", flush=True)
            with SparkExecutionEngine(frame, SparkEngineConfig(shuffle_partitions=1)) as native:
                print("native-probe:spark-ready", flush=True)
                yield native
        except BaseException:
            # Py4J exception text needs its still-live gateway. Preserve SDK
            # diagnostics before stopping the JVM owned by this probe.
            traceback.print_exc()
            raise
        finally:
            if spark is not None:
                spark.stop()
            if previous_gateway is None and SparkContext._gateway is not None:
                SparkContext._gateway.shutdown()


def execute(native, command):
    from truthound.execution._protocols import AggregationType

    if command == "arrow_bridge_policy":
        from truthound.execution.distributed.arrow_bridge import ArrowBridge, ArrowBridgeConfig
        data = native.dataset if native.engine_type == "ray" else native.spark_dataframe
        bridge = ArrowBridge(ArrowBridgeConfig(batch_size=1))
        before = None
        if native.engine_type == "spark":
            before = data.sparkSession.conf.get("spark.sql.execution.arrow.pyspark.enabled", "false")
        frame = bridge.to_polars(data)
        chunks = list((bridge.ray_to_polars_chunked if native.engine_type == "ray" else bridge.spark_to_polars_chunked)(data, 1))
        rejected = False
        try:
            ArrowBridge(ArrowBridgeConfig(null_handling="error")).to_polars(data)
        except ValueError:
            rejected = True
        after = before
        if native.engine_type == "spark":
            after = data.sparkSession.conf.get("spark.sql.execution.arrow.pyspark.enabled", "false")
        return {"rows": len(frame), "sizes": [len(chunk) for chunk in chunks], "null_rejected": rejected,
                "session_unchanged": before == after}

    if command == "manual_arrow_policy":
        from dataclasses import replace
        data = native.spark_dataframe
        with type(native)(data, replace(native.config, arrow_batch_size=1)) as engine:
            batches = engine._manual_arrow_conversion(1)
            sizes = [batch.num_rows for batch in batches]
            invalid = []
            for value in (0, -1, True):
                try:
                    engine._to_arrow_batches(value)
                except ValueError:
                    invalid.append(True)
                else:
                    invalid.append(False)
        with type(native)(data, replace(native.config, memory_fraction=1e-15)) as engine:
            try:
                engine._manual_arrow_conversion(1)
            except MemoryError:
                blocked = True
            else:
                blocked = False
        return {"sizes": sizes, "invalid": invalid, "blocked": blocked}
    if command == "streaming_configuration":
        from dataclasses import replace

        from truthound.execution.distributed.protocols import ExecutionMode

        config = replace(native.config, execution_mode=ExecutionMode.STREAMING, arrow_batch_size=1)
        data = native.dataset if native.engine_type == "ray" else native.spark_dataframe
        with type(native)(data, replace(config, memory_fraction=1e-15)) as limited:
            try:
                next(limited.iter_arrow_batches())
            except MemoryError:
                memory_blocked = True
            else:
                memory_blocked = False
        with type(native)(data, config) as streaming:
            iterator = streaming.iter_arrow_batches()
            first = next(iterator)
            iterator.close()
            batches = list(streaming.iter_arrow_batches())
            arrow = streaming.to_arrow_table()
            return {
                "first": first.num_rows,
                "sizes": [batch.num_rows for batch in batches],
                "rows": arrow.num_rows,
                "columns": arrow.column_names,
                "nulls": arrow.column("x").null_count,
                "after_close_rows": streaming.count_rows(),
                "memory_blocked": memory_blocked,
            }
    if command == "empty_checkpoint":
        from dataclasses import replace

        with tempfile.TemporaryDirectory(prefix="truthound-native-empty-checkpoint-") as directory:
            config = replace(native.config, checkpoint_enabled=True, checkpoint_dir=directory)
            with (
                type(native)(native.dataset.limit(0), config) as checked,
                type(native).from_checkpoint(
                    checked.checkpoint_path, replace(config, checkpoint_enabled=False)
                ) as restored,
            ):
                return {
                    "rows": restored.count_rows(),
                    "columns": restored.get_columns(),
                    "committed": (Path(checked.checkpoint_path) / "checkpoint.json").is_file(),
                }
    if command == "checkpoint_configuration":
        from dataclasses import replace

        with tempfile.TemporaryDirectory(prefix="truthound-native-checkpoint-") as directory:
            config = replace(
                native.config,
                checkpoint_enabled=True,
                checkpoint_dir=directory,
                cache_results=False,
            )
            data = native.dataset if native.engine_type == "ray" else native.spark_dataframe
            with type(native)(data, config) as checked:
                rows = checked.count_rows()
                if native.engine_type == "ray":
                    location = checked.checkpoint_path
                    with type(native).from_checkpoint(
                        location, replace(config, checkpoint_enabled=False)
                    ) as restored:
                        restored_rows = restored.count_rows()
                        restored_sum = restored.aggregate_column("x", AggregationType.SUM)
                    return {
                        "rows": rows,
                        "restored_rows": restored_rows,
                        "sum": restored_sum,
                        "committed": (Path(location) / "checkpoint.json").is_file(),
                    }
                lineage = checked.spark_dataframe.rdd.toDebugString().decode()
                return {
                    "rows": rows,
                    "checkpointed": "ReliableCheckpointRDD" in lineage,
                    "files": any(path.is_file() for path in Path(directory).rglob("*")),
                }
    if command == "profiler_checkpoint":
        from dataclasses import replace

        import polars as pl

        from truthound.profiler.distributed import DistributedProfiler, RayBackend, SparkBackend
        def create():
            backend = RayBackend() if native.engine_type == "ray" else SparkBackend()
            if native.engine_type == "spark":
                backend._spark = native.spark_dataframe.sparkSession
                backend._sc = backend._spark.sparkContext
                backend._initialized = True
            else:
                backend.initialize()
            return backend
        with tempfile.TemporaryDirectory(prefix="truthound-profiler-checkpoint-") as directory:
            backend = create()
            backend.config = replace(backend.config, checkpoint_enabled=True, checkpoint_dir=directory)
            backend.distribute_data(pl.DataFrame({"x": [1, None, 3]}), num_partitions=2)
            location = backend.checkpoint_path
            backend.shutdown()
            reopened = create()
            partitions = reopened.restore_checkpoint(location, num_partitions=2)
            results = reopened.map_partitions(DistributedProfiler._profile_partition, partitions, None)
            statistics = reopened.aggregate_results(results)["x"]
            target = next((Path(location) / "data").rglob("*.parquet"))
            target.write_bytes(target.read_bytes() + b"corruption")
            blocked = False
            try:
                reopened.restore_checkpoint(location)
            except ValueError:
                blocked = True
            reopened.shutdown()
            return {"rows": statistics["row_count"], "distinct": statistics["distinct_count"],
                    "corruption_blocked": blocked, "existing_rows": native.count_rows()}
    if command == "profiler_policies":
        import polars as pl

        from truthound.profiler.distributed import (
            PartitionStrategy,
            RayBackend,
            SparkBackend,
            WorkerResult,
        )
        backend = RayBackend() if native.engine_type == "ray" else SparkBackend()
        if native.engine_type == "spark":
            backend._spark = native.spark_dataframe.sparkSession
            backend._sc = backend._spark.sparkContext
            backend._initialized = True
        else:
            backend.initialize()
        frame = pl.DataFrame({"x": [1, 1, 2, 3], "y": [2, 2, 3, 4]})
        def callback(partition, selected):
            total = sum(value or 0 for value in selected.sum().row(0))
            return WorkerResult(partition.partition_id, {}, selected.height, 0,
                                metadata={"sum": total, "columns": selected.width, "duplicates": (1, 2) in selected.rows()})
        observed = {}
        for policy in PartitionStrategy:
            partitions = backend.distribute_data(frame, num_partitions=4, strategy=policy)
            results = backend.map_partitions(callback, partitions, frame)
            observed[policy.value] = {"sum": sum(item.metadata["sum"] for item in results),
                                     "max_columns": max(item.metadata["columns"] for item in results),
                                     "duplicate_buckets": sum(item.metadata["duplicates"] for item in results)}
        backend.shutdown()
        return observed
    if command == "profiler_global":
        import polars as pl

        from truthound.profiler.distributed import (
            DistributedProfileConfig,
            DistributedProfiler,
            RayBackend,
            SparkBackend,
        )
        backend = RayBackend() if native.engine_type == "ray" else SparkBackend()
        if native.engine_type == "spark":
            backend._spark = native.spark_dataframe.sparkSession
            backend._sc = backend._spark.sparkContext
            backend._initialized = True
        else:
            backend.initialize()
        profiler = DistributedProfiler(backend, DistributedProfileConfig(num_partitions=3))
        result = profiler.profile(pl.DataFrame({"x": [1.0, -1.0, None, 3.0], "s": ["a", "bbbb", None, "a"]}))
        options = DistributedProfiler(backend, DistributedProfileConfig(num_partitions=2, sample_size=2, include_patterns=True))
        sampled = options.profile(pl.DataFrame({"email": ["a@example.com", "b@example.com", "plain"]}))
        from truthound.profiler.distributed import WorkerResult
        def callback(partition, frame):
            return WorkerResult(partition.partition_id, {}, frame.height, 0, metadata={"observed": frame["x"].sum()})
        partitions = backend.distribute_data(pl.DataFrame({"x": [1, 2, 3]}), num_partitions=2)
        callback_results = backend.map_partitions(callback, partitions, None)
        observed_sum = sum(item.metadata["observed"] for item in callback_results)
        backend.shutdown()
        return {"profile": result.to_dict(), "sampled": sampled.to_dict(), "callback_sum": observed_sum, "existing_runtime_rows": native.count_rows()}
    if command == "basics":
        return {
            "rows": native.count_rows(),
            "nulls": native.count_nulls("x"),
            "distinct": native.count_distinct("x"),
            "sum": native.aggregate_column("x", AggregationType.SUM),
            "stats": native.get_stats("x"),
        }
    if command == "regex":
        return {
            "matching": native.count_matching_regex("g", "^a"),
            "not_matching": native.count_not_matching_regex("g", "^a"),
        }
    if command == "sample":
        with native.sample(n=2, seed=42) as child:
            return {"rows": child.count_rows(), "columns": child.get_columns()}
    if command == "range":
        return {
            "inside": native.count_in_range("x", 1, 2),
            "outside": native.count_outside_range("x", 1, 2),
        }
    if command == "polars":
        frame = native.to_polars_lazyframe().collect()
        return {"rows": frame.height, "columns": frame.columns}
    if command == "aggregations":
        return {kind.value: native.aggregate_column("x", kind) for kind in AggregationType}
    if command in {"aggregation_plan", "aggregation_spec"}:
        from truthound.execution.distributed.aggregations import (
            AggregationPlan,
            DistributedAggregator,
        )

        data = native.dataset if native.engine_type == "ray" else native.spark_dataframe
        plan = (
            AggregationPlan()
            .add_count()
            .add_count("x")
            .add_mean("x")
            .add_std("x", ddof=0)
            .add_group_by("g")
            .add_filter("x IS NULL OR x >= 1")
        )
        if command == "aggregation_spec":
            return native._aggregate_distributed(plan.to_spec())
        snapshots = list(DistributedAggregator(data).execute_streaming(plan, batch_size=1))
        return {"snapshots": snapshots, "final": DistributedAggregator(data).execute(plan)}
    if command == "values":
        context_before = (
            native.dataset.context.shuffle_strategy if native.engine_type == "ray" else None
        )
        value = {
            "nulls": native.count_nulls_all(),
            "counts": native.get_value_counts("g"),
            "nullable_counts": [
                {"value": key, "count": count}
                for key, count in native.get_value_counts("x").items()
            ],
            "duplicates": native.count_duplicates(["g"]),
            "pair_duplicates": native.count_duplicates(["x", "g"]),
            "quantiles": native.get_quantiles("x", [0, 0.5, 1]),
            "arrow": native.to_arrow_table().to_pylist(),
        }
        value["group_context_unchanged"] = context_before == (
            native.dataset.context.shuffle_strategy if native.engine_type == "ray" else None
        )
        return value
    if command == "partition":

        def spark_partition(iterator):
            count = sum(1 for _ in iterator)
            yield {"value": count, "row_count": count}

        results = native._execute_on_partitions(
            "probe",
            spark_partition
            if native.engine_type == "spark"
            else lambda batch: {"value": len(batch["x"])},
        )
        return {
            "value": sum(row.value for row in results),
            "rows": sum(row.row_count for row in results),
            "errors": sum(len(row.errors) for row in results),
        }
    if command == "invalid_regex":
        return native.count_matching_regex("g", "[")
    if command == "cache_configuration":
        from dataclasses import replace

        data = native.dataset if native.engine_type == "ray" else native.spark_dataframe
        original = data.count
        calls = []

        def observed_count():
            calls.append(True)
            return original()

        values = []
        try:
            with type(native)(data, replace(native.config, cache_results=False)) as fresh:
                data.count = observed_count
                values.extend([fresh.count_rows(), fresh.count_rows()])
                uncached = len(calls)
            calls.clear()
            with type(native)(data, replace(native.config, cache_results=True)) as cached:
                values.extend([cached.count_rows(), cached.count_rows()])
                cached_calls = len(calls)
                cached.clear_cache()
                values.append(cached.count_rows())
                cleared = len(calls)
        finally:
            data.count = original
        return {
            "uncached_calls": uncached,
            "cached_calls": cached_calls,
            "cleared_calls": cleared,
            "rows": values,
        }
    if command.startswith("partition_strategy_"):
        from dataclasses import replace

        from truthound.execution.distributed.aggregations import AggregationPlan
        from truthound.execution.distributed.protocols import AggregationScope, PartitionStrategy

        name = command.removeprefix("partition_strategy_")
        strategy = PartitionStrategy(name)
        if native.engine_type == "ray":
            import ray

            source = ray.data.from_items(
                [{"x": float(index), "g": "a" if index % 2 else "b"} for index in range(12)],
                override_num_blocks=2,
            )
        else:
            source = native.spark_session.range(12, numPartitions=2).selectExpr(
                "CAST(id AS DOUBLE) AS x", "CASE WHEN id % 2 = 0 THEN 'b' ELSE 'a' END AS g"
            )
        keys = (
            ("g",)
            if strategy in {PartitionStrategy.ROW_HASH, PartitionStrategy.COLUMN}
            else ("x",)
            if strategy == PartitionStrategy.ROW_RANGE
            else ()
        )
        previous_context = source.context.shuffle_strategy if native.engine_type == "ray" else None
        callback = (
            (lambda data, count: data.repartition(count))
            if strategy == PartitionStrategy.CUSTOM
            else None
        )
        config = replace(
            native.config,
            num_partitions=3,
            partition_strategy=strategy,
            partition_columns=keys,
            partitioner=callback,
        )
        with type(native)(source, config) as selected:
            plan = (
                AggregationPlan(scope=AggregationScope.PARTITION)
                .add_count()
                .add_minmax("x")
                .add_group_by("g")
            )
            result = selected._aggregate_distributed(plan.to_spec())
            count = selected.num_partitions
        single_rows = None
        if strategy == PartitionStrategy.ROUND_ROBIN:
            with type(native)(source, replace(config, num_partitions=1)) as single:
                single_rows = single.count_rows()
        unchanged = previous_context == (
            source.context.shuffle_strategy if native.engine_type == "ray" else None
        )
        return {
            "partitions": result["partitions"],
            "count": count,
            "context_unchanged": unchanged,
            "single_rows": single_rows,
        }
    if command == "eager_configuration":
        from dataclasses import replace

        from truthound.execution.distributed.protocols import ExecutionMode

        config = replace(native.config, execution_mode=ExecutionMode.EAGER, cache_results=False)
        if native.engine_type == "ray":
            from ray.data.dataset import MaterializedDataset

            source = native.dataset.map_batches(lambda frame: frame, batch_format="pyarrow")
            with type(native)(source, config) as eager:
                return {
                    "materialized": isinstance(eager.dataset, MaterializedDataset),
                    "rows": eager.count_rows(),
                }
        source = native.spark_dataframe.filter("x IS NOT NULL")
        with type(native)(source, config) as eager:
            owned = eager.spark_dataframe.is_cached
            rows = eager.count_rows()
        cleaned = not source.is_cached
        source.cache().count()
        try:
            with type(native)(source, config) as borrowed:
                borrowed_rows = borrowed.count_rows()
            retained = source.is_cached
        finally:
            source.unpersist()
        return {
            "materialized": owned,
            "rows": rows,
            "owned_cleaned": cleaned,
            "borrowed_retained": retained,
            "borrowed_rows": borrowed_rows,
        }
    if command == "partition_configuration":
        from dataclasses import replace

        from truthound.execution.distributed.protocols import AggregationScope, AggregationSpec

        data = native.dataset if native.engine_type == "ray" else native.spark_dataframe
        cfg = replace(native.config, num_partitions=2)
        with type(native)(data, cfg) as root:
            initial = root.num_partitions
            cfg.num_partitions = 1
            with root.repartition(1) as child:
                explicit = child.num_partitions
                rows = child.count_rows()
            captured = root.config.num_partitions
            root_count = root.num_partitions
            spec = (
                AggregationSpec(scope=AggregationScope.PARTITION).add("*", "count").add("x", "sum")
            )
            separate = root._aggregate_distributed(spec)
            spec.filter_condition = "x < 0"
            empty = root._aggregate_distributed(spec)
        return {
            "initial": initial,
            "explicit": explicit,
            "rows": rows,
            "captured": captured,
            "root": root_count,
            "separate": separate,
            "empty": empty,
        }
    if command == "spark_false_flags":
        from dataclasses import replace

        cfg = replace(native.config, arrow_enabled=False, adaptive_enabled=False)
        with type(native)(native.spark_dataframe, cfg) as configured:
            cfg.extra_spark_conf["spark.sql.shuffle.partitions"] = "99"
            captured = dict(configured.config.extra_spark_conf)
            return {
                "arrow": configured.spark_session.conf.get(
                    "spark.sql.execution.arrow.pyspark.enabled"
                ),
                "adaptive": configured.spark_session.conf.get("spark.sql.adaptive.enabled"),
                "captured": captured,
            }
    if command == "ray_configuration":
        import time

        import ray
        from ray.data import DataContext

        from truthound.execution.distributed.ray_engine import RayEngineConfig, RayExecutionEngine

        @ray.remote(num_cpus=0)
        class Counter:
            def __init__(self):
                self.current = self.peak = self.calls = 0

            def enter(self):
                self.current += 1
                self.calls += 1
                self.peak = max(self.peak, self.current)

            def leave(self):
                self.current -= 1

            def result(self):
                return self.peak, self.calls

        counter = Counter.remote()
        source = ray.data.from_items([{"x": index} for index in range(4)], override_num_blocks=4)
        previous_global = DataContext.get_current().target_max_block_size

        def task_batch(batch):
            ray.get(counter.enter.remote())
            try:
                time.sleep(0.15)
                return {"value": len(batch["x"])}
            finally:
                ray.get(counter.leave.remote())

        with RayExecutionEngine(
            source, RayEngineConfig(batch_size=1, concurrency=1, target_max_block_size=1024 * 1024)
        ) as configured:
            tasks = configured._execute_on_partitions("configured-task", task_batch)
            target = configured.dataset.context.target_max_block_size
            global_unchanged = previous_global == DataContext.get_current().target_max_block_size
        with RayExecutionEngine(
            source, RayEngineConfig(batch_size=1, use_actors=True, actor_pool_size=1)
        ) as configured:
            actors = configured._execute_on_partitions(
                "configured-actor",
                lambda batch: {
                    "value": len(batch["x"]),
                    "metadata": {"actor": bool(ray.get_runtime_context().get_actor_id())},
                },
            )
        peak, calls = ray.get(counter.result.remote())
        return {
            "task_rows": sum(row.row_count for row in tasks),
            "peak": peak,
            "calls": calls,
            "actor_rows": sum(row.row_count for row in actors),
            "all_actor": all(row.metadata["actor"] for row in actors),
            "target": target,
            "global_unchanged": global_unchanged,
        }
    raise ValueError("Unknown native probe command")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("backend", choices=("ray", "spark"))
    parser.add_argument("--log-path", type=Path, required=True)
    arguments = parser.parse_args()
    wire = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    descriptor = os.open(arguments.log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.dup2(descriptor, 1)
    os.dup2(descriptor, 2)
    os.close(descriptor)
    primary_error = None
    primary_diagnostic = None
    try:
        print("native-probe:core-import", flush=True)
        from truthound.mcp.encoding import encode

        print("native-probe:core-imported", flush=True)

        # Ray's Unix socket paths must fit the platform limit, including its
        # generated session suffix. macOS's default temp directory is too long.
        with (
            runtime_directory("th-native-") as directory,
            engine(arguments.backend, Path(directory)) as native,
        ):
            wire.write(
                json.dumps(
                    {
                        "ok": True,
                        "ready": True,
                        "version": importlib.metadata.version(
                            "ray" if arguments.backend == "ray" else "pyspark"
                        ),
                    }
                )
                + "\n"
            )
            for line in sys.stdin:
                request = json.loads(line)
                if request == {"close": True}:
                    break
                try:
                    value = execute(native, request["command"])
                except Exception as error:
                    # Capture lazy JVM-backed diagnostics before SDK teardown.
                    primary_error = error
                    primary_diagnostic = failure_diagnostic(error)
                    raise
                wire.write(json.dumps({"ok": True, "value": encode(value)}, allow_nan=False) + "\n")
    except Exception as error:
        traceback.print_exc()
        diagnostic = primary_diagnostic or failure_diagnostic(error)
        if primary_error is not None and error is not primary_error:
            diagnostic["cleanup_failure"] = failure_diagnostic(error)
        wire.write(json.dumps({"ok": False, **diagnostic}) + "\n")
        return 1
    finally:
        wire.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
