"""Profile exact synthetic workloads separately from benchmark acceptance gates."""

from __future__ import annotations

import argparse
import cProfile
import json
import os
import pstats
import tempfile
from dataclasses import asdict
from pathlib import Path


def diagnose(workload_id="sqlite-unique"):
    # Set before importing Polars, matching the parity worker's native budget.
    for name in (
        "POLARS_MAX_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = "1"
    os.environ["TRUTHOUND_BENCHMARK_WORKER_THREADS"] = "1"
    import polars as pl

    from truthound.benchmark._parity_worker import _truthound_execute
    from truthound.benchmark.workloads import load_workload

    if pl.thread_pool_size() != 1:
        raise RuntimeError("Diagnostic must start in a fresh single-thread-budget interpreter")

    workload = load_workload(
        Path(__file__).resolve().parents[1] / "benchmarks/workloads" / (workload_id + ".json")
    )
    profiler = cProfile.Profile()
    with tempfile.TemporaryDirectory(prefix="truthound-sql-diagnostic-") as directory:
        root = Path(directory)
        profiler.enable()
        try:
            observation = _truthound_execute(
                workload, project_root=root / "workspace", warm_iterations=7, artifact_dir=root
            )
        finally:
            profiler.disable()
    assert (
        observation.correctness_passed
        and observation.observed_issue_count == workload.expected.issue_count
    )
    stats = pstats.Stats(profiler).stats
    functions = [
        dict(
            file=Path(filename).name,
            line=line,
            function=function,
            calls=counts[1],
            own_seconds=counts[2],
            cumulative_seconds=counts[3],
        )
        for (filename, line, function), counts in stats.items()
    ]
    functions.sort(key=lambda row: row["cumulative_seconds"], reverse=True)
    return {
        "diagnostic_only": True,
        "observation": asdict(observation),
        "warm_wall_over_process_cpu": [
            wall / cpu if cpu > 0 else None
            for wall, cpu in zip(
                observation.metadata["warm_durations_seconds"],
                observation.metadata["warm_process_cpu_seconds"],
                strict=True,
            )
        ],
        "profile_functions": functions[:40],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--workload", default="sqlite-unique",
        choices=("sqlite-unique", "local-mixed-core-suite", "local-schema"),
    )
    args = parser.parse_args()
    report = diagnose(args.workload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps({"diagnostic_only": True, "profile_functions": report["profile_functions"][:12]})
    )


if __name__ == "__main__":
    main()
