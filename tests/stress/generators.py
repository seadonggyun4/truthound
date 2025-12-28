"""Load generators for stress testing.

Provides various load patterns for generating test traffic.
"""

from __future__ import annotations

import asyncio
import math
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncIterator, Callable, Awaitable

import logging

logger = logging.getLogger(__name__)


@dataclass
class LoadEvent:
    """Event representing a load operation to execute.

    Attributes:
        event_id: Unique identifier for the event.
        timestamp: When the event should be executed.
        payload: Optional payload for the event.
        priority: Event priority (higher = more important).
    """

    event_id: str
    timestamp: datetime
    payload: dict[str, Any] = field(default_factory=dict)
    priority: int = 0


class LoadGenerator(ABC):
    """Base class for load generators.

    Load generators produce a stream of load events according
    to a specific pattern.
    """

    def __init__(
        self,
        name: str = "load_generator",
        duration_seconds: float = 60.0,
    ) -> None:
        """Initialize load generator.

        Args:
            name: Generator name.
            duration_seconds: Total duration in seconds.
        """
        self._name = name
        self._duration = duration_seconds
        self._event_counter = 0
        self._running = False
        self._start_time: float | None = None

    @property
    def name(self) -> str:
        """Get generator name."""
        return self._name

    @property
    def duration(self) -> float:
        """Get duration in seconds."""
        return self._duration

    @property
    def is_running(self) -> bool:
        """Check if generator is running."""
        return self._running

    @abstractmethod
    def get_rate_at(self, elapsed: float) -> float:
        """Get the target rate at a given elapsed time.

        Args:
            elapsed: Elapsed time in seconds.

        Returns:
            Target rate in events per second.
        """
        ...

    async def generate(self) -> AsyncIterator[LoadEvent]:
        """Generate load events.

        Yields:
            Load events according to the generator pattern.
        """
        self._running = True
        self._start_time = time.monotonic()
        self._event_counter = 0

        try:
            while True:
                elapsed = time.monotonic() - self._start_time

                # Check if duration exceeded
                if elapsed >= self._duration:
                    break

                # Get current rate
                rate = self.get_rate_at(elapsed)

                if rate <= 0:
                    await asyncio.sleep(0.1)
                    continue

                # Calculate interval
                interval = 1.0 / rate

                # Generate event
                self._event_counter += 1
                event = LoadEvent(
                    event_id=f"{self._name}_{self._event_counter}",
                    timestamp=datetime.now(),
                    payload={"elapsed": elapsed, "rate": rate},
                )

                yield event

                # Wait for next event
                await asyncio.sleep(interval)

        finally:
            self._running = False

    def get_elapsed(self) -> float:
        """Get elapsed time since start."""
        if self._start_time is None:
            return 0.0
        return time.monotonic() - self._start_time

    def get_progress(self) -> float:
        """Get progress (0.0 to 1.0)."""
        elapsed = self.get_elapsed()
        return min(elapsed / self._duration, 1.0) if self._duration > 0 else 1.0


class ConstantLoadGenerator(LoadGenerator):
    """Generate load at a constant rate.

    Example:
        >>> generator = ConstantLoadGenerator(rate=10.0, duration_seconds=60)
        >>> async for event in generator.generate():
        ...     print(f"Event: {event.event_id}")
    """

    def __init__(
        self,
        rate: float = 10.0,
        duration_seconds: float = 60.0,
        name: str = "constant",
    ) -> None:
        """Initialize constant load generator.

        Args:
            rate: Events per second.
            duration_seconds: Total duration.
            name: Generator name.
        """
        super().__init__(name=name, duration_seconds=duration_seconds)
        self._rate = rate

    def get_rate_at(self, elapsed: float) -> float:
        """Get constant rate."""
        return self._rate


class RampUpLoadGenerator(LoadGenerator):
    """Generate load with linear ramp-up pattern.

    Starts at initial_rate and linearly increases to target_rate
    over the ramp_duration, then maintains target_rate.

    Example:
        >>> generator = RampUpLoadGenerator(
        ...     initial_rate=1.0,
        ...     target_rate=100.0,
        ...     ramp_duration=30.0,
        ...     duration_seconds=120,
        ... )
    """

    def __init__(
        self,
        initial_rate: float = 1.0,
        target_rate: float = 100.0,
        ramp_duration: float = 30.0,
        duration_seconds: float = 60.0,
        name: str = "ramp_up",
    ) -> None:
        """Initialize ramp-up load generator.

        Args:
            initial_rate: Starting rate.
            target_rate: Target rate.
            ramp_duration: Duration of ramp-up phase.
            duration_seconds: Total duration.
            name: Generator name.
        """
        super().__init__(name=name, duration_seconds=duration_seconds)
        self._initial_rate = initial_rate
        self._target_rate = target_rate
        self._ramp_duration = ramp_duration

    def get_rate_at(self, elapsed: float) -> float:
        """Get rate with linear ramp-up."""
        if elapsed < self._ramp_duration:
            progress = elapsed / self._ramp_duration
            return (
                self._initial_rate
                + (self._target_rate - self._initial_rate) * progress
            )
        return self._target_rate


class SpikeLoadGenerator(LoadGenerator):
    """Generate load with periodic spikes.

    Maintains a base rate with periodic spikes to a higher rate.

    Example:
        >>> generator = SpikeLoadGenerator(
        ...     base_rate=10.0,
        ...     spike_rate=100.0,
        ...     spike_duration=5.0,
        ...     spike_interval=30.0,
        ... )
    """

    def __init__(
        self,
        base_rate: float = 10.0,
        spike_rate: float = 100.0,
        spike_duration: float = 5.0,
        spike_interval: float = 30.0,
        duration_seconds: float = 120.0,
        name: str = "spike",
    ) -> None:
        """Initialize spike load generator.

        Args:
            base_rate: Normal rate between spikes.
            spike_rate: Rate during spikes.
            spike_duration: Duration of each spike.
            spike_interval: Interval between spike starts.
            duration_seconds: Total duration.
            name: Generator name.
        """
        super().__init__(name=name, duration_seconds=duration_seconds)
        self._base_rate = base_rate
        self._spike_rate = spike_rate
        self._spike_duration = spike_duration
        self._spike_interval = spike_interval

    def get_rate_at(self, elapsed: float) -> float:
        """Get rate with periodic spikes."""
        # Calculate position within current interval
        position_in_interval = elapsed % self._spike_interval

        # Check if we're in a spike
        if position_in_interval < self._spike_duration:
            return self._spike_rate

        return self._base_rate


class WaveLoadGenerator(LoadGenerator):
    """Generate load with sinusoidal wave pattern.

    Creates smooth oscillations between min and max rates.

    Example:
        >>> generator = WaveLoadGenerator(
        ...     min_rate=10.0,
        ...     max_rate=50.0,
        ...     period_seconds=60.0,
        ... )
    """

    def __init__(
        self,
        min_rate: float = 10.0,
        max_rate: float = 50.0,
        period_seconds: float = 60.0,
        duration_seconds: float = 180.0,
        name: str = "wave",
    ) -> None:
        """Initialize wave load generator.

        Args:
            min_rate: Minimum rate (trough).
            max_rate: Maximum rate (peak).
            period_seconds: Wave period.
            duration_seconds: Total duration.
            name: Generator name.
        """
        super().__init__(name=name, duration_seconds=duration_seconds)
        self._min_rate = min_rate
        self._max_rate = max_rate
        self._period = period_seconds

    def get_rate_at(self, elapsed: float) -> float:
        """Get rate with sinusoidal pattern."""
        # Calculate sine value (0 to 2*pi over period)
        phase = (elapsed / self._period) * 2 * math.pi

        # Normalize sine from [-1, 1] to [0, 1]
        normalized = (math.sin(phase) + 1) / 2

        # Scale to rate range
        return self._min_rate + (self._max_rate - self._min_rate) * normalized


class StepLoadGenerator(LoadGenerator):
    """Generate load with step increments.

    Increases rate in discrete steps at fixed intervals.

    Example:
        >>> generator = StepLoadGenerator(
        ...     initial_rate=10.0,
        ...     step_increment=10.0,
        ...     step_duration=30.0,
        ...     max_rate=100.0,
        ... )
    """

    def __init__(
        self,
        initial_rate: float = 10.0,
        step_increment: float = 10.0,
        step_duration: float = 30.0,
        max_rate: float = 100.0,
        duration_seconds: float = 180.0,
        name: str = "step",
    ) -> None:
        """Initialize step load generator.

        Args:
            initial_rate: Starting rate.
            step_increment: Rate increase per step.
            step_duration: Duration of each step.
            max_rate: Maximum rate cap.
            duration_seconds: Total duration.
            name: Generator name.
        """
        super().__init__(name=name, duration_seconds=duration_seconds)
        self._initial_rate = initial_rate
        self._step_increment = step_increment
        self._step_duration = step_duration
        self._max_rate = max_rate

    def get_rate_at(self, elapsed: float) -> float:
        """Get rate with step increments."""
        # Calculate current step
        step = int(elapsed / self._step_duration)

        # Calculate rate
        rate = self._initial_rate + step * self._step_increment

        # Cap at max rate
        return min(rate, self._max_rate)


class RandomLoadGenerator(LoadGenerator):
    """Generate load with random variations.

    Base rate with random variations within a specified range.

    Example:
        >>> generator = RandomLoadGenerator(
        ...     base_rate=50.0,
        ...     variation_percent=0.3,  # +/- 30%
        ... )
    """

    def __init__(
        self,
        base_rate: float = 50.0,
        variation_percent: float = 0.2,
        change_interval: float = 5.0,
        duration_seconds: float = 120.0,
        name: str = "random",
        seed: int | None = None,
    ) -> None:
        """Initialize random load generator.

        Args:
            base_rate: Base rate.
            variation_percent: Variation as percentage (0.0-1.0).
            change_interval: How often to change rate.
            duration_seconds: Total duration.
            name: Generator name.
            seed: Random seed for reproducibility.
        """
        super().__init__(name=name, duration_seconds=duration_seconds)
        self._base_rate = base_rate
        self._variation = variation_percent
        self._change_interval = change_interval
        self._rng = random.Random(seed)
        self._current_rate = base_rate
        self._last_change = 0.0

    def get_rate_at(self, elapsed: float) -> float:
        """Get rate with random variations."""
        # Check if time to change
        if elapsed - self._last_change >= self._change_interval:
            # Calculate new rate with random variation
            variation = self._rng.uniform(-self._variation, self._variation)
            self._current_rate = self._base_rate * (1 + variation)
            self._current_rate = max(1.0, self._current_rate)  # Minimum 1 req/sec
            self._last_change = elapsed

        return self._current_rate


class CompositeLoadGenerator(LoadGenerator):
    """Combine multiple load generators.

    Runs generators sequentially or in parallel.

    Example:
        >>> generator = CompositeLoadGenerator([
        ...     RampUpLoadGenerator(initial_rate=1, target_rate=50, duration_seconds=30),
        ...     ConstantLoadGenerator(rate=50, duration_seconds=60),
        ...     RampUpLoadGenerator(initial_rate=50, target_rate=1, duration_seconds=30),
        ... ], mode="sequential")
    """

    def __init__(
        self,
        generators: list[LoadGenerator],
        mode: str = "sequential",
        name: str = "composite",
    ) -> None:
        """Initialize composite load generator.

        Args:
            generators: List of generators to combine.
            mode: "sequential" or "parallel".
            name: Generator name.
        """
        total_duration = sum(g.duration for g in generators)
        super().__init__(name=name, duration_seconds=total_duration)
        self._generators = generators
        self._mode = mode

    def get_rate_at(self, elapsed: float) -> float:
        """Get combined rate from generators."""
        if self._mode == "sequential":
            # Find which generator is active
            cumulative = 0.0
            for generator in self._generators:
                if elapsed < cumulative + generator.duration:
                    return generator.get_rate_at(elapsed - cumulative)
                cumulative += generator.duration
            return 0.0
        else:
            # Parallel: sum all rates
            return sum(g.get_rate_at(elapsed) for g in self._generators)

    async def generate(self) -> AsyncIterator[LoadEvent]:
        """Generate events from composite generators."""
        if self._mode == "sequential":
            for generator in self._generators:
                async for event in generator.generate():
                    yield event
        else:
            # Parallel: merge all generator streams
            async def generator_wrapper(gen: LoadGenerator) -> AsyncIterator[LoadEvent]:
                async for event in gen.generate():
                    yield event

            # Create tasks for all generators
            queues: list[asyncio.Queue[LoadEvent | None]] = [
                asyncio.Queue() for _ in self._generators
            ]

            async def fill_queue(gen: LoadGenerator, queue: asyncio.Queue) -> None:
                async for event in gen.generate():
                    await queue.put(event)
                await queue.put(None)

            tasks = [
                asyncio.create_task(fill_queue(gen, queue))
                for gen, queue in zip(self._generators, queues)
            ]

            # Merge events from all queues
            active_queues = set(range(len(queues)))
            while active_queues:
                for i in list(active_queues):
                    try:
                        event = queues[i].get_nowait()
                        if event is None:
                            active_queues.remove(i)
                        else:
                            yield event
                    except asyncio.QueueEmpty:
                        pass
                await asyncio.sleep(0.01)

            # Clean up tasks
            for task in tasks:
                task.cancel()
