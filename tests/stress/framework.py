"""Core stress testing framework.

Provides the main runner and configuration for stress tests.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)


class LoadPhase(Enum):
    """Phases of a stress test."""

    WARMUP = "warmup"
    RAMP_UP = "ramp_up"
    STEADY = "steady"
    RAMP_DOWN = "ramp_down"
    COOLDOWN = "cooldown"


@dataclass
class LoadProfile:
    """Configuration for load generation profile.

    Attributes:
        warmup_seconds: Duration of warmup phase.
        ramp_up_seconds: Duration of ramp up phase.
        steady_seconds: Duration of steady state phase.
        ramp_down_seconds: Duration of ramp down phase.
        cooldown_seconds: Duration of cooldown phase.
        initial_rate: Initial requests per second.
        target_rate: Target requests per second.
        min_rate: Minimum rate during ramp down.
    """

    warmup_seconds: float = 10.0
    ramp_up_seconds: float = 30.0
    steady_seconds: float = 60.0
    ramp_down_seconds: float = 30.0
    cooldown_seconds: float = 10.0
    initial_rate: float = 1.0
    target_rate: float = 100.0
    min_rate: float = 1.0

    @property
    def total_duration(self) -> float:
        """Get total test duration in seconds."""
        return (
            self.warmup_seconds
            + self.ramp_up_seconds
            + self.steady_seconds
            + self.ramp_down_seconds
            + self.cooldown_seconds
        )

    def get_rate_at(self, elapsed_seconds: float) -> float:
        """Get the target rate at a given elapsed time.

        Args:
            elapsed_seconds: Time elapsed since test start.

        Returns:
            Target rate in requests per second.
        """
        # Warmup phase - use initial rate
        if elapsed_seconds < self.warmup_seconds:
            return self.initial_rate

        # Ramp up phase - linear increase
        ramp_up_start = self.warmup_seconds
        ramp_up_end = ramp_up_start + self.ramp_up_seconds
        if elapsed_seconds < ramp_up_end:
            progress = (elapsed_seconds - ramp_up_start) / self.ramp_up_seconds
            return self.initial_rate + (self.target_rate - self.initial_rate) * progress

        # Steady phase - maintain target rate
        steady_end = ramp_up_end + self.steady_seconds
        if elapsed_seconds < steady_end:
            return self.target_rate

        # Ramp down phase - linear decrease
        ramp_down_end = steady_end + self.ramp_down_seconds
        if elapsed_seconds < ramp_down_end:
            progress = (elapsed_seconds - steady_end) / self.ramp_down_seconds
            return self.target_rate - (self.target_rate - self.min_rate) * progress

        # Cooldown phase - use min rate
        return self.min_rate

    def get_phase_at(self, elapsed_seconds: float) -> LoadPhase:
        """Get the current phase at a given elapsed time."""
        if elapsed_seconds < self.warmup_seconds:
            return LoadPhase.WARMUP

        if elapsed_seconds < self.warmup_seconds + self.ramp_up_seconds:
            return LoadPhase.RAMP_UP

        if elapsed_seconds < (
            self.warmup_seconds + self.ramp_up_seconds + self.steady_seconds
        ):
            return LoadPhase.STEADY

        if elapsed_seconds < (
            self.warmup_seconds
            + self.ramp_up_seconds
            + self.steady_seconds
            + self.ramp_down_seconds
        ):
            return LoadPhase.RAMP_DOWN

        return LoadPhase.COOLDOWN


@dataclass
class StressTestConfig:
    """Configuration for a stress test.

    Attributes:
        name: Test name.
        load_profile: Load generation profile.
        concurrency: Maximum concurrent operations.
        timeout_seconds: Timeout for each operation.
        enable_chaos: Whether to enable chaos engineering.
        chaos_probability: Probability of failure injection (0.0-1.0).
        collect_traces: Whether to collect detailed traces.
        target_success_rate: Target success rate (0.0-1.0).
        max_latency_p99_ms: Maximum acceptable P99 latency.
    """

    name: str = "stress_test"
    load_profile: LoadProfile = field(default_factory=LoadProfile)
    concurrency: int = 100
    timeout_seconds: float = 30.0
    enable_chaos: bool = False
    chaos_probability: float = 0.05
    collect_traces: bool = False
    target_success_rate: float = 0.99
    max_latency_p99_ms: float = 1000.0
    stop_on_failure_rate: float = 0.2


@dataclass
class OperationResult:
    """Result of a single operation."""

    operation_id: str
    success: bool
    start_time: datetime
    end_time: datetime
    latency_ms: float
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StressTestResult:
    """Result of a stress test run.

    Attributes:
        config: Test configuration.
        start_time: When test started.
        end_time: When test ended.
        total_operations: Total operations executed.
        successful_operations: Number of successful operations.
        failed_operations: Number of failed operations.
        latencies_ms: List of latencies in milliseconds.
        errors: List of error messages.
        phases: Statistics per phase.
        passed: Whether test passed acceptance criteria.
    """

    config: StressTestConfig
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    phases: dict[str, dict[str, Any]] = field(default_factory=dict)
    passed: bool = False

    @property
    def duration_seconds(self) -> float:
        """Get test duration in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return (datetime.now() - self.start_time).total_seconds()

    @property
    def success_rate(self) -> float:
        """Get success rate (0.0-1.0)."""
        if self.total_operations == 0:
            return 1.0
        return self.successful_operations / self.total_operations

    @property
    def throughput_per_second(self) -> float:
        """Get throughput in operations per second."""
        duration = self.duration_seconds
        if duration == 0:
            return 0.0
        return self.total_operations / duration

    def latency_percentile(self, percentile: float) -> float:
        """Get latency percentile in milliseconds."""
        if not self.latencies_ms:
            return 0.0
        sorted_latencies = sorted(self.latencies_ms)
        index = int(percentile / 100 * (len(sorted_latencies) - 1))
        return sorted_latencies[index]

    @property
    def latency_p50(self) -> float:
        """Get P50 latency."""
        return self.latency_percentile(50)

    @property
    def latency_p95(self) -> float:
        """Get P95 latency."""
        return self.latency_percentile(95)

    @property
    def latency_p99(self) -> float:
        """Get P99 latency."""
        return self.latency_percentile(99)

    @property
    def latency_avg(self) -> float:
        """Get average latency."""
        if not self.latencies_ms:
            return 0.0
        return sum(self.latencies_ms) / len(self.latencies_ms)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "config": {
                "name": self.config.name,
                "concurrency": self.config.concurrency,
                "target_rate": self.config.load_profile.target_rate,
            },
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "total_operations": self.total_operations,
            "successful_operations": self.successful_operations,
            "failed_operations": self.failed_operations,
            "success_rate": self.success_rate,
            "throughput_per_second": self.throughput_per_second,
            "latency": {
                "avg_ms": self.latency_avg,
                "p50_ms": self.latency_p50,
                "p95_ms": self.latency_p95,
                "p99_ms": self.latency_p99,
            },
            "passed": self.passed,
            "phases": self.phases,
        }


class StressTestRunner:
    """Runner for stress tests.

    Coordinates load generation, operation execution, and result collection.

    Example:
        >>> async def my_operation(op_id: str) -> bool:
        ...     # Perform operation
        ...     return True
        >>>
        >>> runner = StressTestRunner(
        ...     config=StressTestConfig(name="saga_stress"),
        ...     operation=my_operation,
        ... )
        >>> result = await runner.run()
        >>> print(f"Success rate: {result.success_rate:.2%}")
    """

    def __init__(
        self,
        config: StressTestConfig,
        operation: Callable[[str], Awaitable[bool]],
        setup: Callable[[], Awaitable[None]] | None = None,
        teardown: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        """Initialize stress test runner.

        Args:
            config: Test configuration.
            operation: Async function to execute for each operation.
            setup: Optional setup function.
            teardown: Optional teardown function.
        """
        self._config = config
        self._operation = operation
        self._setup = setup
        self._teardown = teardown

        self._result: StressTestResult | None = None
        self._running = False
        self._operation_counter = 0
        self._semaphore: asyncio.Semaphore | None = None
        self._stop_event: asyncio.Event | None = None

    async def run(self) -> StressTestResult:
        """Run the stress test.

        Returns:
            Test result.
        """
        self._result = StressTestResult(config=self._config)
        self._running = True
        self._operation_counter = 0
        self._semaphore = asyncio.Semaphore(self._config.concurrency)
        self._stop_event = asyncio.Event()

        try:
            # Setup
            if self._setup:
                logger.info(f"Running setup for {self._config.name}")
                await self._setup()

            # Run test
            logger.info(f"Starting stress test: {self._config.name}")
            await self._run_test()

        except Exception as e:
            logger.error(f"Stress test failed: {e}")
            self._result.errors.append(str(e))

        finally:
            self._running = False
            self._result.end_time = datetime.now()

            # Teardown
            if self._teardown:
                logger.info(f"Running teardown for {self._config.name}")
                await self._teardown()

        # Evaluate pass/fail criteria
        self._evaluate_result()

        return self._result

    async def _run_test(self) -> None:
        """Run the main test loop."""
        profile = self._config.load_profile
        start_time = time.monotonic()
        current_phase = LoadPhase.WARMUP
        phase_stats: dict[LoadPhase, dict[str, int]] = {
            phase: {"operations": 0, "successes": 0, "failures": 0}
            for phase in LoadPhase
        }

        pending_tasks: set[asyncio.Task] = set()

        while True:
            elapsed = time.monotonic() - start_time

            # Check if test is complete
            if elapsed >= profile.total_duration:
                break

            # Check stop event
            if self._stop_event and self._stop_event.is_set():
                logger.warning("Stress test stopped early")
                break

            # Check failure rate
            if self._result:
                failure_rate = 1.0 - self._result.success_rate
                if (
                    self._result.total_operations > 100
                    and failure_rate > self._config.stop_on_failure_rate
                ):
                    logger.warning(
                        f"Stopping test due to high failure rate: {failure_rate:.2%}"
                    )
                    break

            # Get current phase and rate
            phase = profile.get_phase_at(elapsed)
            rate = profile.get_rate_at(elapsed)

            # Log phase transitions
            if phase != current_phase:
                logger.info(f"Phase transition: {current_phase.value} -> {phase.value}")
                current_phase = phase

            # Calculate how many operations to start
            interval = 1.0 / rate if rate > 0 else 1.0

            # Launch operation
            task = asyncio.create_task(
                self._execute_operation(current_phase, phase_stats)
            )
            pending_tasks.add(task)
            task.add_done_callback(pending_tasks.discard)

            # Wait for next operation
            await asyncio.sleep(interval)

        # Wait for pending operations
        if pending_tasks:
            logger.info(f"Waiting for {len(pending_tasks)} pending operations")
            await asyncio.gather(*pending_tasks, return_exceptions=True)

        # Store phase statistics
        if self._result:
            self._result.phases = {
                phase.value: stats for phase, stats in phase_stats.items()
            }

    async def _execute_operation(
        self,
        phase: LoadPhase,
        phase_stats: dict[LoadPhase, dict[str, int]],
    ) -> None:
        """Execute a single operation."""
        if not self._semaphore or not self._result:
            return

        self._operation_counter += 1
        operation_id = f"op_{self._operation_counter}"

        async with self._semaphore:
            start_time = datetime.now()
            start_mono = time.monotonic()

            try:
                # Execute with timeout
                success = await asyncio.wait_for(
                    self._operation(operation_id),
                    timeout=self._config.timeout_seconds,
                )

                end_mono = time.monotonic()
                latency_ms = (end_mono - start_mono) * 1000

                self._result.total_operations += 1
                self._result.latencies_ms.append(latency_ms)
                phase_stats[phase]["operations"] += 1

                if success:
                    self._result.successful_operations += 1
                    phase_stats[phase]["successes"] += 1
                else:
                    self._result.failed_operations += 1
                    phase_stats[phase]["failures"] += 1

            except asyncio.TimeoutError:
                self._result.total_operations += 1
                self._result.failed_operations += 1
                self._result.errors.append(f"{operation_id}: Timeout")
                phase_stats[phase]["operations"] += 1
                phase_stats[phase]["failures"] += 1

            except Exception as e:
                self._result.total_operations += 1
                self._result.failed_operations += 1
                self._result.errors.append(f"{operation_id}: {str(e)}")
                phase_stats[phase]["operations"] += 1
                phase_stats[phase]["failures"] += 1

    def _evaluate_result(self) -> None:
        """Evaluate test result against acceptance criteria."""
        if not self._result:
            return

        passed = True
        reasons = []

        # Check success rate
        if self._result.success_rate < self._config.target_success_rate:
            passed = False
            reasons.append(
                f"Success rate {self._result.success_rate:.2%} < "
                f"target {self._config.target_success_rate:.2%}"
            )

        # Check P99 latency
        if self._result.latency_p99 > self._config.max_latency_p99_ms:
            passed = False
            reasons.append(
                f"P99 latency {self._result.latency_p99:.1f}ms > "
                f"target {self._config.max_latency_p99_ms:.1f}ms"
            )

        self._result.passed = passed

        if passed:
            logger.info(f"Stress test PASSED: {self._config.name}")
        else:
            logger.warning(
                f"Stress test FAILED: {self._config.name} - {', '.join(reasons)}"
            )

    def stop(self) -> None:
        """Stop the running test."""
        if self._stop_event:
            self._stop_event.set()

    @property
    def is_running(self) -> bool:
        """Check if test is running."""
        return self._running

    @property
    def result(self) -> StressTestResult | None:
        """Get current result."""
        return self._result
