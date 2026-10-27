"""Selected native suite rules must not disappear or become successful on failure."""

import asyncio
import logging

import polars as pl
import pytest

from truthound.profiler.generators.base import GeneratedRule, RuleCategory
from truthound.profiler.generators.suite_generator import ValidationSuite
from truthound.profiler.integration.adapters import ValidatorRegistry
from truthound.profiler.integration.executor import AsyncSuiteExecutor, SuiteExecutor
from truthound.profiler.integration.protocols import ExecutionContext

pytestmark = pytest.mark.fault


def suite():
    return ValidationSuite(
        name="selected",
        rules=(GeneratedRule("selected", "SyntheticNativeValidator", RuleCategory.SCHEMA),),
    )


@pytest.mark.parametrize("asynchronous", [False, True])
def test_missing_validator_is_failure_not_successful_empty_execution(asynchronous, caplog):
    executor = AsyncSuiteExecutor() if asynchronous else SuiteExecutor()
    with caplog.at_level(logging.DEBUG):
        result = (
            asyncio.run(executor.execute_async(suite(), pl.DataFrame({"x": [1]})))
            if asynchronous
            else executor.execute(suite(), pl.DataFrame({"x": [1]}))
        )
    assert not result.success and result.skipped_rules == 1 and result.errors
    assert result.executed_rules == 0


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("parallel", [False, True])
def test_explicit_native_registry_is_used_and_error_attempt_counted(asynchronous, parallel, caplog):
    registry = ValidatorRegistry()

    class FailedValidator:
        def validate(self, frame):
            raise RuntimeError("synthetic-private-native-validation")

    registry.register_factory("SyntheticNativeValidator", lambda rule: FailedValidator())
    executor_type = AsyncSuiteExecutor if asynchronous else SuiteExecutor
    executor = executor_type(registry=registry)
    native_suite = suite()
    if parallel:
        native_suite = ValidationSuite(name="two", rules=native_suite.rules * 2)
    with caplog.at_level(logging.DEBUG):
        result = (
            asyncio.run(executor.execute_async(native_suite, pl.DataFrame({"x": [1]}), ExecutionContext(parallel=parallel)))
            if asynchronous
            else executor.execute(native_suite, pl.DataFrame({"x": [1]}), ExecutionContext(parallel=parallel))
        )
    assert not result.success and result.executed_rules == len(native_suite.rules)
    assert result.failed_rules == len(native_suite.rules) and result.skipped_rules == 0
    assert all(type(error) is RuntimeError for error in result.errors)
    assert "synthetic-private-native-validation" not in caplog.text
    assert "synthetic-private-native-validation" not in str(result.report)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_native_dry_run_never_validates_and_completes_lifecycle(asynchronous):
    registry = ValidatorRegistry()
    events = []

    class Validator:
        def validate(self, frame):
            raise AssertionError("Dry run executed validator")

    class Listener:
        def on_suite_start(self, *args):
            events.append("start")

        def on_suite_complete(self, *args):
            events.append("complete")

    registry.register_factory("SyntheticNativeValidator", lambda rule: Validator())
    executor = (AsyncSuiteExecutor if asynchronous else SuiteExecutor)(registry=registry, listeners=[Listener()])
    context = ExecutionContext(dry_run=True)
    result = asyncio.run(executor.execute_async(suite(), pl.DataFrame({"x": [1]}), context)) if asynchronous else executor.execute(suite(), pl.DataFrame({"x": [1]}), context)
    assert result.success and result.metadata["dry_run"]
    assert events == ["start", "complete"]


def test_native_listener_failure_cannot_report_success_or_log_body(caplog):
    registry = ValidatorRegistry()

    class Validator:
        def validate(self, frame):
            return []

    class Listener:
        def on_suite_start(self, *args):
            raise RuntimeError("synthetic-private-listener")

        def on_rule_start(self, *args):
            return None

        def on_rule_complete(self, *args):
            return None

        def on_suite_complete(self, *args):
            return None

    registry.register_factory("SyntheticNativeValidator", lambda rule: Validator())
    result = SuiteExecutor(registry=registry, listeners=[Listener()]).execute(suite(), pl.DataFrame({"x": [1]}))
    assert not result.success and result.passed_rules == 1 and result.errors
    assert "synthetic-private-listener" not in caplog.text + str(result.report)


def test_native_execution_guard_stops_following_validator_after_revocation():
    registry = ValidatorRegistry()
    events = []
    revoked = False

    def guard():
        if revoked:
            raise PermissionError("revoked")

    class Validator:
        def validate(self, frame):
            nonlocal revoked
            events.append("validated")
            revoked = True
            return []

    registry.register_factory("SyntheticNativeValidator", lambda rule: Validator())
    executor = SuiteExecutor(registry=registry)
    executor.set_execution_guard(guard)
    selected = ValidationSuite(name="two", rules=suite().rules * 2)
    with pytest.raises(PermissionError):
        executor.execute(selected, pl.DataFrame({"x": [1]}))
    assert events == ["validated"]


def test_native_parallel_lifecycle_and_fail_fast_drain_running_work_only():
    import threading

    registry = ValidatorRegistry()
    release = threading.Event()
    validated, events = [], []

    class Selected:
        def __init__(self, name):
            self.name = name

        def validate(self, frame):
            validated.append(self.name)
            if self.name == "first":
                return ["synthetic validation issue"]
            release.wait(2)
            return []

    class Listener:
        def on_suite_start(self, *args):
            events.append("suite-start")

        def on_suite_complete(self, *args):
            events.append("suite-complete")

        def on_rule_start(self, rule):
            events.append(("start", rule.name))

        def on_rule_complete(self, rule, *args):
            events.append(("complete", rule.name))

    class Progress:
        def report_progress(self, *args):
            pass

        def report_result(self, name, *args):
            if name == "first":
                release.set()

    registry.register_factory("SyntheticNativeValidator", lambda rule: Selected(rule.name))
    rules = tuple(GeneratedRule(name, "SyntheticNativeValidator", RuleCategory.SCHEMA) for name in ("first", "second", "third", "fourth"))
    executor = SuiteExecutor(registry=registry, listeners=[Listener()], progress_reporter=Progress())
    result = executor.execute(ValidationSuite("selected", rules), pl.DataFrame({"x": [1]}), ExecutionContext(parallel=True, max_workers=1, fail_fast=True))
    assert not result.success and result.failed_rules == 1
    assert result.executed_rules == len(validated) <= 2
    assert result.skipped_rules + result.executed_rules == len(rules)
    assert events[0] == "suite-start" and events[-1] == "suite-complete"
    assert [event for event in events if isinstance(event, tuple) and event[0] == "start"] == [("start", name) for name in validated]
    assert sorted(event[1] for event in events if isinstance(event, tuple) and event[0] == "complete") == sorted(validated)
