"""Saga Testing Framework.

This module provides a comprehensive testing framework for saga patterns,
enabling validation of complex transaction scenarios.
"""

from __future__ import annotations

import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Callable, Generator
from uuid import uuid4

from truthound.checkpoint.transaction.saga.definition import (
    SagaDefinition,
    SagaStepDefinition,
)
from truthound.checkpoint.transaction.saga.state_machine import SagaState
from truthound.checkpoint.transaction.saga.runner import (
    SagaRunner,
    SagaExecutionContext,
    SagaExecutionResult,
    SagaMetrics,
)

if TYPE_CHECKING:
    from truthound.checkpoint.actions.base import ActionResult, BaseAction
    from truthound.checkpoint.checkpoint import CheckpointResult


class FailureType(str, Enum):
    """Types of failures that can be injected."""

    EXCEPTION = "exception"  # Raise exception
    TIMEOUT = "timeout"  # Simulate timeout
    SLOW = "slow"  # Slow execution
    INTERMITTENT = "intermittent"  # Random failures
    PARTIAL = "partial"  # Partial failure
    NETWORK = "network"  # Network-like failure
    RESOURCE = "resource"  # Resource exhaustion
    COMPENSATION_FAIL = "compensation_fail"  # Fail during compensation


@dataclass
class FailureSpec:
    """Specification for a failure injection.

    Attributes:
        step_id: Step to inject failure at.
        failure_type: Type of failure.
        probability: Probability of failure (0.0-1.0).
        delay_seconds: Delay before failure.
        message: Error message.
        retry_success_after: Succeed after N retries.
        affect_compensation: Whether failure affects compensation.
    """

    step_id: str
    failure_type: FailureType = FailureType.EXCEPTION
    probability: float = 1.0
    delay_seconds: float = 0.0
    message: str = "Injected failure"
    retry_success_after: int = 0
    affect_compensation: bool = False

    _attempt_count: int = field(default=0, init=False)

    def should_fail(self) -> bool:
        """Check if failure should occur."""
        self._attempt_count += 1

        if self.retry_success_after > 0:
            if self._attempt_count > self.retry_success_after:
                return False

        if self.probability < 1.0:
            return random.random() < self.probability

        return True


class FailureInjector:
    """Injects failures into saga execution for testing.

    Example:
        >>> injector = FailureInjector()
        >>> injector.fail_step("payment", FailureType.TIMEOUT)
        >>> injector.fail_step("shipping", FailureType.INTERMITTENT, probability=0.3)
        >>>
        >>> harness = SagaTestHarness(saga, injector=injector)
        >>> result = harness.run()
    """

    def __init__(self) -> None:
        """Initialize failure injector."""
        self._failure_specs: dict[str, FailureSpec] = {}
        self._execution_log: list[dict[str, Any]] = []

    def fail_step(
        self,
        step_id: str,
        failure_type: FailureType = FailureType.EXCEPTION,
        probability: float = 1.0,
        delay_seconds: float = 0.0,
        message: str = "",
        retry_success_after: int = 0,
    ) -> "FailureInjector":
        """Configure a step to fail.

        Args:
            step_id: Step to fail.
            failure_type: Type of failure.
            probability: Failure probability.
            delay_seconds: Delay before failure.
            message: Error message.
            retry_success_after: Succeed after N retries.

        Returns:
            Self for chaining.
        """
        self._failure_specs[step_id] = FailureSpec(
            step_id=step_id,
            failure_type=failure_type,
            probability=probability,
            delay_seconds=delay_seconds,
            message=message or f"Injected {failure_type.value} failure at {step_id}",
            retry_success_after=retry_success_after,
        )
        return self

    def fail_compensation(
        self,
        step_id: str,
        probability: float = 1.0,
    ) -> "FailureInjector":
        """Configure a step's compensation to fail.

        Args:
            step_id: Step whose compensation should fail.
            probability: Failure probability.

        Returns:
            Self for chaining.
        """
        self._failure_specs[f"{step_id}_compensation"] = FailureSpec(
            step_id=step_id,
            failure_type=FailureType.COMPENSATION_FAIL,
            probability=probability,
            message=f"Compensation failure at {step_id}",
            affect_compensation=True,
        )
        return self

    def check_and_inject(
        self,
        step_id: str,
        is_compensation: bool = False,
    ) -> None:
        """Check if failure should be injected and apply it.

        Args:
            step_id: Step being executed.
            is_compensation: Whether this is compensation execution.

        Raises:
            Exception: If failure should be injected.
        """
        key = f"{step_id}_compensation" if is_compensation else step_id
        spec = self._failure_specs.get(key)

        if spec is None:
            return

        if not spec.should_fail():
            self._log_execution(step_id, "passed", is_compensation)
            return

        self._log_execution(step_id, spec.failure_type.value, is_compensation)

        if spec.delay_seconds > 0:
            time.sleep(spec.delay_seconds)

        if spec.failure_type == FailureType.TIMEOUT:
            time.sleep(60)  # Simulate long operation
            raise TimeoutError(spec.message)
        elif spec.failure_type == FailureType.SLOW:
            time.sleep(spec.delay_seconds or 5)
            return  # Don't fail, just slow
        elif spec.failure_type == FailureType.NETWORK:
            raise ConnectionError(spec.message)
        elif spec.failure_type == FailureType.RESOURCE:
            raise MemoryError(spec.message)
        else:
            raise RuntimeError(spec.message)

    def _log_execution(
        self,
        step_id: str,
        result: str,
        is_compensation: bool,
    ) -> None:
        """Log execution attempt."""
        self._execution_log.append({
            "step_id": step_id,
            "result": result,
            "is_compensation": is_compensation,
            "timestamp": datetime.now().isoformat(),
        })

    def get_execution_log(self) -> list[dict[str, Any]]:
        """Get execution log."""
        return list(self._execution_log)

    def reset(self) -> None:
        """Reset failure injector state."""
        self._execution_log.clear()
        for spec in self._failure_specs.values():
            spec._attempt_count = 0


@dataclass
class SagaAssertion:
    """An assertion about saga execution.

    Attributes:
        name: Assertion name.
        predicate: Function to check assertion.
        message: Failure message.
        severity: Assertion severity.
    """

    name: str
    predicate: Callable[[SagaExecutionResult], bool]
    message: str = ""
    severity: str = "error"  # "error", "warning", "info"

    def check(self, result: SagaExecutionResult) -> tuple[bool, str]:
        """Check the assertion.

        Returns:
            Tuple of (passed, message).
        """
        try:
            passed = self.predicate(result)
            if passed:
                return True, f"[PASS] {self.name}"
            else:
                return False, f"[FAIL] {self.name}: {self.message}"
        except Exception as e:
            return False, f"[ERROR] {self.name}: {e}"


@dataclass
class SagaScenario:
    """A test scenario for saga execution.

    Attributes:
        name: Scenario name.
        description: Scenario description.
        saga: Saga definition.
        failure_injector: Optional failure injector.
        setup: Optional setup function.
        teardown: Optional teardown function.
        assertions: List of assertions to check.
        expected_state: Expected final state.
        expected_completed_steps: Expected completed steps.
        expected_compensated_steps: Expected compensated steps.
        timeout_seconds: Scenario timeout.
    """

    name: str
    description: str = ""
    saga: SagaDefinition | None = None
    failure_injector: FailureInjector | None = None
    setup: Callable[[], None] | None = None
    teardown: Callable[[], None] | None = None
    assertions: list[SagaAssertion] = field(default_factory=list)
    expected_state: SagaState | None = None
    expected_completed_steps: list[str] | None = None
    expected_compensated_steps: list[str] | None = None
    timeout_seconds: float = 60.0

    def add_assertion(
        self,
        name: str,
        predicate: Callable[[SagaExecutionResult], bool],
        message: str = "",
    ) -> "SagaScenario":
        """Add an assertion to the scenario.

        Args:
            name: Assertion name.
            predicate: Check function.
            message: Failure message.

        Returns:
            Self for chaining.
        """
        self.assertions.append(
            SagaAssertion(name=name, predicate=predicate, message=message)
        )
        return self

    def expect_success(self) -> "SagaScenario":
        """Add success expectation."""
        return self.add_assertion(
            "success",
            lambda r: r.success,
            "Expected saga to succeed",
        )

    def expect_failure(self) -> "SagaScenario":
        """Add failure expectation."""
        return self.add_assertion(
            "failure",
            lambda r: not r.success,
            "Expected saga to fail",
        )

    def expect_step_completed(self, step_id: str) -> "SagaScenario":
        """Add step completion expectation."""
        return self.add_assertion(
            f"step_{step_id}_completed",
            lambda r: step_id in r.completed_steps,
            f"Expected step {step_id} to complete",
        )

    def expect_step_compensated(self, step_id: str) -> "SagaScenario":
        """Add step compensation expectation."""
        return self.add_assertion(
            f"step_{step_id}_compensated",
            lambda r: step_id in r.compensated_steps,
            f"Expected step {step_id} to be compensated",
        )

    def expect_compensation_count(self, count: int) -> "SagaScenario":
        """Add compensation count expectation."""
        return self.add_assertion(
            "compensation_count",
            lambda r: len(r.compensated_steps) == count,
            f"Expected {count} compensations",
        )


class ScenarioBuilder:
    """Fluent builder for test scenarios.

    Example:
        >>> scenario = (
        ...     ScenarioBuilder("happy_path")
        ...     .with_saga(my_saga)
        ...     .expect_success()
        ...     .expect_step_completed("step1")
        ...     .expect_step_completed("step2")
        ...     .build()
        ... )
    """

    def __init__(self, name: str) -> None:
        """Initialize scenario builder.

        Args:
            name: Scenario name.
        """
        self._scenario = SagaScenario(name=name)

    def description(self, desc: str) -> "ScenarioBuilder":
        """Set scenario description."""
        self._scenario.description = desc
        return self

    def with_saga(self, saga: SagaDefinition) -> "ScenarioBuilder":
        """Set saga to test."""
        self._scenario.saga = saga
        return self

    def with_failure(
        self,
        step_id: str,
        failure_type: FailureType = FailureType.EXCEPTION,
        **kwargs: Any,
    ) -> "ScenarioBuilder":
        """Add failure injection."""
        if self._scenario.failure_injector is None:
            self._scenario.failure_injector = FailureInjector()

        self._scenario.failure_injector.fail_step(
            step_id, failure_type, **kwargs
        )
        return self

    def with_compensation_failure(
        self,
        step_id: str,
        probability: float = 1.0,
    ) -> "ScenarioBuilder":
        """Add compensation failure."""
        if self._scenario.failure_injector is None:
            self._scenario.failure_injector = FailureInjector()

        self._scenario.failure_injector.fail_compensation(step_id, probability)
        return self

    def with_setup(self, setup: Callable[[], None]) -> "ScenarioBuilder":
        """Set setup function."""
        self._scenario.setup = setup
        return self

    def with_teardown(self, teardown: Callable[[], None]) -> "ScenarioBuilder":
        """Set teardown function."""
        self._scenario.teardown = teardown
        return self

    def with_timeout(self, seconds: float) -> "ScenarioBuilder":
        """Set scenario timeout."""
        self._scenario.timeout_seconds = seconds
        return self

    def expect_success(self) -> "ScenarioBuilder":
        """Expect saga to succeed."""
        self._scenario.expect_success()
        self._scenario.expected_state = SagaState.COMPLETED
        return self

    def expect_failure(self) -> "ScenarioBuilder":
        """Expect saga to fail."""
        self._scenario.expect_failure()
        return self

    def expect_state(self, state: SagaState) -> "ScenarioBuilder":
        """Expect specific final state."""
        self._scenario.expected_state = state
        self._scenario.add_assertion(
            f"state_{state.value}",
            lambda r, s=state: r.state == s,
            f"Expected state {state.value}",
        )
        return self

    def expect_step_completed(self, step_id: str) -> "ScenarioBuilder":
        """Expect step to complete."""
        self._scenario.expect_step_completed(step_id)
        if self._scenario.expected_completed_steps is None:
            self._scenario.expected_completed_steps = []
        self._scenario.expected_completed_steps.append(step_id)
        return self

    def expect_step_compensated(self, step_id: str) -> "ScenarioBuilder":
        """Expect step to be compensated."""
        self._scenario.expect_step_compensated(step_id)
        if self._scenario.expected_compensated_steps is None:
            self._scenario.expected_compensated_steps = []
        self._scenario.expected_compensated_steps.append(step_id)
        return self

    def expect_steps_completed(self, *step_ids: str) -> "ScenarioBuilder":
        """Expect multiple steps to complete."""
        for step_id in step_ids:
            self.expect_step_completed(step_id)
        return self

    def expect_steps_compensated(self, *step_ids: str) -> "ScenarioBuilder":
        """Expect multiple steps to be compensated."""
        for step_id in step_ids:
            self.expect_step_compensated(step_id)
        return self

    def expect_compensation_count(self, count: int) -> "ScenarioBuilder":
        """Expect specific compensation count."""
        self._scenario.expect_compensation_count(count)
        return self

    def add_assertion(
        self,
        name: str,
        predicate: Callable[[SagaExecutionResult], bool],
        message: str = "",
    ) -> "ScenarioBuilder":
        """Add custom assertion."""
        self._scenario.add_assertion(name, predicate, message)
        return self

    def build(self) -> SagaScenario:
        """Build the scenario."""
        return self._scenario


@dataclass
class ScenarioResult:
    """Result of running a scenario.

    Attributes:
        scenario_name: Name of the scenario.
        passed: Whether all assertions passed.
        execution_result: Saga execution result.
        assertion_results: Results of each assertion.
        duration_ms: Scenario duration.
        error: Error if scenario failed to run.
    """

    scenario_name: str
    passed: bool
    execution_result: SagaExecutionResult | None = None
    assertion_results: list[tuple[bool, str]] = field(default_factory=list)
    duration_ms: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "scenario_name": self.scenario_name,
            "passed": self.passed,
            "assertion_results": [
                {"passed": p, "message": m} for p, m in self.assertion_results
            ],
            "duration_ms": self.duration_ms,
            "error": self.error,
        }


class SagaTestHarness:
    """Test harness for comprehensive saga testing.

    Example:
        >>> harness = SagaTestHarness(saga)
        >>>
        >>> # Add scenarios
        >>> harness.add_scenario(
        ...     ScenarioBuilder("happy_path")
        ...     .expect_success()
        ...     .expect_steps_completed("step1", "step2", "step3")
        ...     .build()
        ... )
        >>> harness.add_scenario(
        ...     ScenarioBuilder("step2_failure")
        ...     .with_failure("step2")
        ...     .expect_failure()
        ...     .expect_step_completed("step1")
        ...     .expect_step_compensated("step1")
        ...     .build()
        ... )
        >>>
        >>> # Run all scenarios
        >>> report = harness.run_all()
        >>> print(report)
    """

    def __init__(
        self,
        saga: SagaDefinition | None = None,
        runner: SagaRunner | None = None,
        context_factory: Callable[[], SagaExecutionContext] | None = None,
    ) -> None:
        """Initialize test harness.

        Args:
            saga: Default saga to test.
            runner: Saga runner to use.
            context_factory: Factory for creating execution contexts.
        """
        self._saga = saga
        self._runner = runner or SagaRunner()
        self._context_factory = context_factory
        self._scenarios: list[SagaScenario] = []
        self._results: list[ScenarioResult] = []

    def add_scenario(self, scenario: SagaScenario) -> "SagaTestHarness":
        """Add a test scenario.

        Args:
            scenario: Scenario to add.

        Returns:
            Self for chaining.
        """
        if scenario.saga is None:
            scenario.saga = self._saga
        self._scenarios.append(scenario)
        return self

    def create_scenario(self, name: str) -> ScenarioBuilder:
        """Create a new scenario builder.

        Args:
            name: Scenario name.

        Returns:
            Scenario builder.
        """
        builder = ScenarioBuilder(name)
        if self._saga:
            builder.with_saga(self._saga)
        return builder

    def run_scenario(self, scenario: SagaScenario) -> ScenarioResult:
        """Run a single scenario.

        Args:
            scenario: Scenario to run.

        Returns:
            Scenario result.
        """
        start_time = time.time()

        # Setup
        if scenario.setup:
            try:
                scenario.setup()
            except Exception as e:
                return ScenarioResult(
                    scenario_name=scenario.name,
                    passed=False,
                    error=f"Setup failed: {e}",
                )

        try:
            # Create context
            if self._context_factory:
                context = self._context_factory()
            else:
                from truthound.checkpoint.checkpoint import CheckpointResult

                # Create mock checkpoint result
                mock_result = type(
                    "MockCheckpointResult",
                    (),
                    {"status": "success", "errors": None, "metadata": {}},
                )()
                context = SagaExecutionContext(checkpoint_result=mock_result)

            # Get saga
            saga = scenario.saga
            if saga is None:
                return ScenarioResult(
                    scenario_name=scenario.name,
                    passed=False,
                    error="No saga defined for scenario",
                )

            # Wrap actions with failure injector if present
            if scenario.failure_injector:
                self._wrap_with_injector(saga, scenario.failure_injector)

            # Execute saga
            result = self._runner.execute(saga, context)

            # Check assertions
            assertion_results = []
            for assertion in scenario.assertions:
                passed, message = assertion.check(result)
                assertion_results.append((passed, message))

            # Check expected state
            if scenario.expected_state:
                passed = result.state == scenario.expected_state
                assertion_results.append((
                    passed,
                    f"[{'PASS' if passed else 'FAIL'}] Expected state: {scenario.expected_state.value}",
                ))

            # All assertions passed?
            all_passed = all(p for p, _ in assertion_results)

            duration_ms = (time.time() - start_time) * 1000

            return ScenarioResult(
                scenario_name=scenario.name,
                passed=all_passed,
                execution_result=result,
                assertion_results=assertion_results,
                duration_ms=duration_ms,
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return ScenarioResult(
                scenario_name=scenario.name,
                passed=False,
                duration_ms=duration_ms,
                error=str(e),
            )

        finally:
            # Teardown
            if scenario.teardown:
                try:
                    scenario.teardown()
                except Exception as e:
                    pass  # Log but don't fail

            # Reset failure injector
            if scenario.failure_injector:
                scenario.failure_injector.reset()

    def _wrap_with_injector(
        self,
        saga: SagaDefinition,
        injector: FailureInjector,
    ) -> None:
        """Wrap saga actions with failure injection."""
        for step in saga.steps:
            if step.action:
                original_execute = step.action.execute

                def wrapped_execute(
                    checkpoint_result: Any,
                    step_id: str = step.step_id,
                    original: Callable = original_execute,
                ) -> Any:
                    injector.check_and_inject(step_id)
                    return original(checkpoint_result)

                step.action.execute = wrapped_execute

    def run_all(self) -> str:
        """Run all scenarios and return report.

        Returns:
            Test report string.
        """
        self._results.clear()

        for scenario in self._scenarios:
            result = self.run_scenario(scenario)
            self._results.append(result)

        return self._generate_report()

    def _generate_report(self) -> str:
        """Generate test report."""
        lines = [
            "=" * 60,
            "SAGA TEST REPORT",
            "=" * 60,
            "",
        ]

        passed_count = sum(1 for r in self._results if r.passed)
        total_count = len(self._results)

        lines.append(f"Scenarios: {passed_count}/{total_count} passed")
        lines.append("")

        for result in self._results:
            status = "PASS" if result.passed else "FAIL"
            lines.append(f"[{status}] {result.scenario_name} ({result.duration_ms:.1f}ms)")

            if result.error:
                lines.append(f"  Error: {result.error}")

            for passed, message in result.assertion_results:
                lines.append(f"  {message}")

            lines.append("")

        lines.append("=" * 60)

        return "\n".join(lines)

    def get_results(self) -> list[ScenarioResult]:
        """Get all scenario results."""
        return list(self._results)

    # ==========================================================================
    # Predefined Scenario Generators
    # ==========================================================================

    def generate_happy_path_scenario(self) -> SagaScenario:
        """Generate a happy path scenario."""
        scenario = self.create_scenario("happy_path")
        scenario.description("All steps complete successfully")
        scenario.expect_success()

        if self._saga:
            for step in self._saga.steps:
                scenario.expect_step_completed(step.step_id)

        return scenario.build()

    def generate_failure_scenarios(self) -> Generator[SagaScenario, None, None]:
        """Generate failure scenarios for each step."""
        if not self._saga:
            return

        for step in self._saga.steps:
            scenario = (
                self.create_scenario(f"failure_at_{step.step_id}")
                .description(f"Failure at step {step.name}")
                .with_failure(step.step_id)
                .expect_failure()
            )

            # Expect prior steps to complete
            for prior in self._saga.steps:
                if prior.order < step.order:
                    scenario.expect_step_completed(prior.step_id)
                    if prior.has_compensation():
                        scenario.expect_step_compensated(prior.step_id)

            yield scenario.build()

    def generate_compensation_failure_scenarios(
        self,
    ) -> Generator[SagaScenario, None, None]:
        """Generate scenarios where compensation fails."""
        if not self._saga:
            return

        compensatable_steps = [s for s in self._saga.steps if s.has_compensation()]

        for step in compensatable_steps:
            # Find a later step to fail
            later_steps = [s for s in self._saga.steps if s.order > step.order]
            if not later_steps:
                continue

            fail_at = later_steps[0]

            scenario = (
                self.create_scenario(f"compensation_failure_at_{step.step_id}")
                .description(
                    f"Step {fail_at.step_id} fails, compensation at {step.step_id} also fails"
                )
                .with_failure(fail_at.step_id)
                .with_compensation_failure(step.step_id)
                .expect_failure()
                .expect_state(SagaState.FAILED)
            )

            yield scenario.build()

    def generate_retry_scenarios(self) -> Generator[SagaScenario, None, None]:
        """Generate scenarios testing retry behavior."""
        if not self._saga:
            return

        for step in self._saga.steps:
            if step.retry_config.max_attempts > 0:
                # Succeed after retries
                scenario = (
                    self.create_scenario(f"retry_success_{step.step_id}")
                    .description(f"Step {step.name} succeeds after retries")
                    .with_failure(
                        step.step_id,
                        FailureType.INTERMITTENT,
                        retry_success_after=step.retry_config.max_attempts - 1,
                    )
                    .expect_success()
                    .expect_step_completed(step.step_id)
                )

                yield scenario.build()

                # Exceed max retries
                scenario = (
                    self.create_scenario(f"retry_exhausted_{step.step_id}")
                    .description(f"Step {step.name} exceeds max retries")
                    .with_failure(step.step_id)
                    .expect_failure()
                )

                yield scenario.build()

    def add_standard_scenarios(self) -> "SagaTestHarness":
        """Add all standard test scenarios."""
        # Happy path
        self.add_scenario(self.generate_happy_path_scenario())

        # Failure at each step
        for scenario in self.generate_failure_scenarios():
            self.add_scenario(scenario)

        # Compensation failures
        for scenario in self.generate_compensation_failure_scenarios():
            self.add_scenario(scenario)

        # Retry scenarios
        for scenario in self.generate_retry_scenarios():
            self.add_scenario(scenario)

        return self
