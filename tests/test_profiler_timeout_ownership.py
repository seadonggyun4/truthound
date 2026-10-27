"""Actual native timeout work, unknown threads, startup budgets and owned descendants."""

import asyncio
import json
import os
import subprocess
import sys
import threading
import time
from contextlib import suppress
from functools import partial
from pathlib import Path

import polars as pl
import psutil
import pytest

from truthound.profiler.generators.base import GeneratedRule, RuleCategory
from truthound.profiler.generators.suite_generator import ValidationSuite
from truthound.profiler.integration.adapters import ValidatorRegistry
from truthound.profiler.integration.executor import AsyncSuiteExecutor, SuiteExecutor
from truthound.profiler.integration.protocols import ExecutionContext
from truthound.profiler.process_timeout import (
    ExecutionBackend,
    ProcessExecutionStrategy,
    ProcessTimeoutConfig,
    ProcessTimeoutExecutor,
)

pytestmark = pytest.mark.fault


class SlowValidator:
    def validate(self, frame):
        time.sleep(0.3)
        return []


def private_failure():
    raise RuntimeError("synthetic-private-native-process-query")


def unpicklable_result():
    return lambda: "synthetic-private-process-result"


def spawn_owned_descendant(marker):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    marker = Path(marker)
    owner = psutil.Process(os.getpid())
    descendant = psutil.Process(child.pid)
    marker.write_text(json.dumps({"owner": [owner.pid, owner.create_time()], "descendant": [descendant.pid, descendant.create_time()]}))
    time.sleep(30)


def alive(record):
    try:
        process = psutil.Process(record[0])
        return process.create_time() == record[1] and process.status() not in {psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD}
    except psutil.NoSuchProcess:
        return False


def cleanup_owned(record):
    try:
        process = psutil.Process(record[0])
        if process.create_time() == record[1]:
            process.kill()
            with suppress(psutil.TimeoutExpired):
                process.wait(timeout=2)
    except psutil.NoSuchProcess:
        pass


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("parallel", [False, True])
def test_native_suite_deadline_is_enforced_instead_of_ignored(asynchronous, parallel):
    registry = ValidatorRegistry()
    registry.register_factory("SlowNative", lambda rule: SlowValidator())
    rule = GeneratedRule("slow", "SlowNative", RuleCategory.SCHEMA)
    suite = ValidationSuite(name="deadline", rules=(rule,) * (2 if parallel else 1))
    executor = (AsyncSuiteExecutor if asynchronous else SuiteExecutor)(registry=registry)
    context = ExecutionContext(parallel=parallel, timeout_seconds=0.05)
    result = asyncio.run(executor.execute_async(suite, pl.DataFrame({"x": [1]}), context)) if asynchronous else executor.execute(suite, pl.DataFrame({"x": [1]}), context)
    assert not result.success and result.failed_rules == len(suite.rules)
    assert result.executed_rules == len(suite.rules)
    assert all(isinstance(error, TimeoutError) for error in result.errors)


def test_native_thread_timeout_does_not_claim_termination_or_retry_unstopped_work():
    released = threading.Event()
    attempts = []

    def blocked():
        attempts.append(1)
        released.wait(2)
        return 1

    try:
        config = ProcessTimeoutConfig(default_backend=ExecutionBackend.THREAD, max_retries=2, enable_resource_monitoring=False, enable_circuit_breaker=False)
        result = ProcessTimeoutExecutor(config).execute(blocked, timeout_seconds=0.01)
        assert result.timed_out and not result.metrics.was_terminated
        assert result.outcome_unknown and attempts == [1]
    finally:
        released.set()


def test_actual_native_process_executes_importable_callable_and_redacts_failure():
    native = ProcessExecutionStrategy()
    result = native.execute(partial(pow, 2, 3), timeout_seconds=5)
    assert result.success and result.value == 8
    failure = native.execute(private_failure, timeout_seconds=5)
    assert not failure.success and not failure.timed_out
    assert "synthetic-private-native-process-query" not in str(failure.error) + json.dumps(failure.to_dict())


def test_native_result_serialization_failure_is_not_disguised_as_timeout():
    result = ProcessExecutionStrategy().execute(unpicklable_result, timeout_seconds=2)
    assert not result.success and not result.timed_out
    assert result.error is not None
    assert "synthetic-private-process-result" not in str(result.error)


def test_native_process_timeout_reaps_actual_owned_descendant(tmp_path):
    marker = tmp_path / "owned.json"
    result = None
    try:
        result = ProcessExecutionStrategy(graceful_timeout=0.1).execute(partial(spawn_owned_descendant, str(marker)), timeout_seconds=3)
        assert marker.exists(), "The actual native callable must start; serialization failure is not coverage"
        records = json.loads(marker.read_text())
        assert result.timed_out and result.metrics.was_terminated
        assert not alive(records["owner"]) and not alive(records["descendant"])
    finally:
        if marker.exists():
            for record in json.loads(marker.read_text()).values():
                cleanup_owned(record)


def abrupt_exit():
    os._exit(73)


def test_native_abrupt_exit_is_unknown_failure_instead_of_timeout():
    result = ProcessExecutionStrategy().execute(abrupt_exit, timeout_seconds=5)
    assert not result.success and not result.timed_out and result.outcome_unknown
    assert result.error.error_type == "ChildProcessError"


def test_native_parent_death_fences_actual_child_group(tmp_path):
    marker = tmp_path / "orphan-owned.json"
    driver = tmp_path / "native-parent.py"
    driver.write_text(
        "from functools import partial\n"
        "from truthound.profiler.process_timeout import ProcessExecutionStrategy\n"
        "from tests.test_profiler_timeout_ownership import spawn_owned_descendant\n"
        "if __name__ == '__main__':\n"
        f"    ProcessExecutionStrategy().execute(partial(spawn_owned_descendant, {str(marker)!r}), timeout_seconds=30)\n"
    )
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root) + os.pathsep + environment.get("PYTHONPATH", "")
    parent = subprocess.Popen([sys.executable, str(driver)], cwd=root, env=environment,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 10
        while not marker.exists() and time.monotonic() < deadline:
            if parent.poll() is not None:
                break
            threading.Event().wait(0.02)
        assert marker.exists(), "Actual native child/descendant must start"
        records = json.loads(marker.read_text())
        assert alive(records["owner"]) and alive(records["descendant"])
        parent.kill()
        parent.wait(timeout=3)
        deadline = time.monotonic() + 3
        while any(alive(record) for record in records.values()) and time.monotonic() < deadline:
            threading.Event().wait(0.02)
        assert not any(alive(record) for record in records.values())
    finally:
        if parent.poll() is None:
            parent.kill()
        parent.wait(timeout=3)
        if marker.exists():
            for record in json.loads(marker.read_text()).values():
                cleanup_owned(record)
