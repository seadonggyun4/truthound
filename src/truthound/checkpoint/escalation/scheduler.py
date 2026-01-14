"""APScheduler-based Escalation Scheduler.

This module provides scheduling capabilities for escalation policies
using APScheduler for reliable job execution.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


class SchedulerType(str, Enum):
    """Type of scheduler implementation."""

    APSCHEDULER = "apscheduler"
    ASYNCIO = "asyncio"
    MEMORY = "memory"


class JobStatus(str, Enum):
    """Status of a scheduled job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


@dataclass
class ScheduledJob:
    """Representation of a scheduled escalation job.

    Attributes:
        id: Unique job identifier.
        record_id: Associated escalation record ID.
        policy_name: Escalation policy name.
        target_level: Target escalation level.
        scheduled_at: When the job is scheduled to run.
        created_at: When the job was created.
        status: Current job status.
        attempts: Number of execution attempts.
        last_error: Last error message if failed.
        metadata: Additional job metadata.
    """

    id: str
    record_id: str
    policy_name: str
    target_level: int
    scheduled_at: datetime
    created_at: datetime = field(default_factory=datetime.now)
    status: JobStatus = JobStatus.PENDING
    attempts: int = 0
    max_attempts: int = 3
    last_error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_due(self) -> bool:
        """Check if job is due for execution."""
        return self.scheduled_at <= datetime.now() and self.status == JobStatus.PENDING

    @property
    def can_retry(self) -> bool:
        """Check if job can be retried."""
        return self.attempts < self.max_attempts

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "record_id": self.record_id,
            "policy_name": self.policy_name,
            "target_level": self.target_level,
            "scheduled_at": self.scheduled_at.isoformat(),
            "created_at": self.created_at.isoformat(),
            "status": self.status.value,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "last_error": self.last_error,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScheduledJob:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            record_id=data["record_id"],
            policy_name=data["policy_name"],
            target_level=data["target_level"],
            scheduled_at=datetime.fromisoformat(data["scheduled_at"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            status=JobStatus(data.get("status", "pending")),
            attempts=data.get("attempts", 0),
            max_attempts=data.get("max_attempts", 3),
            last_error=data.get("last_error"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class SchedulerConfig:
    """Configuration for the escalation scheduler.

    Attributes:
        scheduler_type: Type of scheduler to use.
        check_interval_seconds: Interval for checking due jobs.
        max_concurrent_jobs: Maximum concurrent job executions.
        job_timeout_seconds: Timeout for individual job execution.
        retry_delay_seconds: Delay between retries.
        cleanup_interval_seconds: Interval for cleaning up old jobs.
        cleanup_age_hours: Age threshold for cleanup.
        apscheduler_config: APScheduler-specific configuration.
    """

    scheduler_type: SchedulerType = SchedulerType.ASYNCIO
    check_interval_seconds: int = 10
    max_concurrent_jobs: int = 100
    job_timeout_seconds: int = 300
    retry_delay_seconds: int = 60
    cleanup_interval_seconds: int = 3600
    cleanup_age_hours: int = 24
    apscheduler_config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "scheduler_type": self.scheduler_type.value,
            "check_interval_seconds": self.check_interval_seconds,
            "max_concurrent_jobs": self.max_concurrent_jobs,
            "job_timeout_seconds": self.job_timeout_seconds,
            "retry_delay_seconds": self.retry_delay_seconds,
            "cleanup_interval_seconds": self.cleanup_interval_seconds,
            "cleanup_age_hours": self.cleanup_age_hours,
            "apscheduler_config": self.apscheduler_config,
        }


# Type alias for job executor callback
JobExecutor = Callable[[ScheduledJob], Any]
AsyncJobExecutor = Callable[[ScheduledJob], Any]


@runtime_checkable
class EscalationSchedulerProtocol(Protocol):
    """Protocol for escalation scheduler implementations."""

    @property
    def is_running(self) -> bool:
        """Check if scheduler is running."""
        ...

    def start(self) -> None:
        """Start the scheduler."""
        ...

    def stop(self) -> None:
        """Stop the scheduler."""
        ...

    def schedule_escalation(
        self,
        record_id: str,
        policy_name: str,
        target_level: int,
        delay: timedelta,
        metadata: dict[str, Any] | None = None,
    ) -> ScheduledJob:
        """Schedule an escalation.

        Args:
            record_id: Escalation record ID.
            policy_name: Policy name.
            target_level: Target escalation level.
            delay: Delay before execution.
            metadata: Optional job metadata.

        Returns:
            Created ScheduledJob.
        """
        ...

    def cancel_escalation(self, record_id: str) -> int:
        """Cancel all scheduled escalations for a record.

        Args:
            record_id: Record ID to cancel.

        Returns:
            Number of jobs cancelled.
        """
        ...

    def get_scheduled_jobs(self, record_id: str | None = None) -> list[ScheduledJob]:
        """Get scheduled jobs.

        Args:
            record_id: Optional filter by record ID.

        Returns:
            List of scheduled jobs.
        """
        ...

    def reschedule(
        self,
        job_id: str,
        new_time: datetime,
    ) -> ScheduledJob | None:
        """Reschedule a job.

        Args:
            job_id: Job ID to reschedule.
            new_time: New scheduled time.

        Returns:
            Updated job or None if not found.
        """
        ...


class BaseEscalationScheduler(ABC):
    """Abstract base class for escalation schedulers."""

    def __init__(
        self,
        config: SchedulerConfig | None = None,
        executor: AsyncJobExecutor | None = None,
    ) -> None:
        """Initialize scheduler.

        Args:
            config: Scheduler configuration.
            executor: Job executor callback.
        """
        self._config = config or SchedulerConfig()
        self._executor = executor
        self._is_running = False
        self._jobs: dict[str, ScheduledJob] = {}
        self._jobs_by_record: dict[str, set[str]] = {}

    @property
    def config(self) -> SchedulerConfig:
        """Get scheduler configuration."""
        return self._config

    @property
    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._is_running

    def set_executor(self, executor: AsyncJobExecutor) -> None:
        """Set the job executor callback.

        Args:
            executor: Job executor callback.
        """
        self._executor = executor

    @abstractmethod
    def start(self) -> None:
        """Start the scheduler."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop the scheduler."""
        pass

    def schedule_escalation(
        self,
        record_id: str,
        policy_name: str,
        target_level: int,
        delay: timedelta,
        metadata: dict[str, Any] | None = None,
    ) -> ScheduledJob:
        """Schedule an escalation."""
        import hashlib

        # Generate job ID
        job_id = hashlib.sha256(
            f"{record_id}:{policy_name}:{target_level}:{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]

        job = ScheduledJob(
            id=job_id,
            record_id=record_id,
            policy_name=policy_name,
            target_level=target_level,
            scheduled_at=datetime.now() + delay,
            metadata=metadata or {},
        )

        self._jobs[job_id] = job

        # Track jobs by record
        if record_id not in self._jobs_by_record:
            self._jobs_by_record[record_id] = set()
        self._jobs_by_record[record_id].add(job_id)

        self._schedule_job(job)

        logger.info(
            f"Scheduled escalation job {job_id} for record {record_id} "
            f"level {target_level} at {job.scheduled_at}"
        )

        return job

    @abstractmethod
    def _schedule_job(self, job: ScheduledJob) -> None:
        """Internal method to schedule a job with the backend."""
        pass

    def cancel_escalation(self, record_id: str) -> int:
        """Cancel all scheduled escalations for a record."""
        job_ids = self._jobs_by_record.get(record_id, set()).copy()
        cancelled = 0

        for job_id in job_ids:
            if self._cancel_job(job_id):
                cancelled += 1

        return cancelled

    def _cancel_job(self, job_id: str) -> bool:
        """Cancel a specific job."""
        job = self._jobs.get(job_id)
        if not job:
            return False

        if job.status in (JobStatus.PENDING, JobStatus.PAUSED):
            job.status = JobStatus.CANCELLED
            self._remove_job_from_backend(job_id)
            logger.info(f"Cancelled job {job_id}")
            return True

        return False

    @abstractmethod
    def _remove_job_from_backend(self, job_id: str) -> None:
        """Remove a job from the scheduler backend."""
        pass

    def get_scheduled_jobs(self, record_id: str | None = None) -> list[ScheduledJob]:
        """Get scheduled jobs."""
        if record_id:
            job_ids = self._jobs_by_record.get(record_id, set())
            return [self._jobs[jid] for jid in job_ids if jid in self._jobs]
        return list(self._jobs.values())

    def reschedule(
        self,
        job_id: str,
        new_time: datetime,
    ) -> ScheduledJob | None:
        """Reschedule a job."""
        job = self._jobs.get(job_id)
        if not job or job.status != JobStatus.PENDING:
            return None

        job.scheduled_at = new_time
        self._reschedule_in_backend(job)
        logger.info(f"Rescheduled job {job_id} to {new_time}")

        return job

    @abstractmethod
    def _reschedule_in_backend(self, job: ScheduledJob) -> None:
        """Reschedule a job in the backend."""
        pass

    async def _execute_job(self, job: ScheduledJob) -> None:
        """Execute a scheduled job."""
        if not self._executor:
            logger.error(f"No executor set for job {job.id}")
            return

        job.status = JobStatus.RUNNING
        job.attempts += 1

        try:
            result = self._executor(job)
            if asyncio.iscoroutine(result):
                await asyncio.wait_for(
                    result,
                    timeout=self._config.job_timeout_seconds,
                )
            job.status = JobStatus.COMPLETED
            logger.info(f"Job {job.id} completed successfully")

        except asyncio.TimeoutError:
            job.last_error = f"Job timed out after {self._config.job_timeout_seconds}s"
            self._handle_job_failure(job)

        except Exception as e:
            job.last_error = str(e)
            self._handle_job_failure(job)
            logger.exception(f"Job {job.id} failed: {e}")

    def _handle_job_failure(self, job: ScheduledJob) -> None:
        """Handle a failed job."""
        if job.can_retry:
            job.status = JobStatus.PENDING
            job.scheduled_at = datetime.now() + timedelta(
                seconds=self._config.retry_delay_seconds
            )
            self._schedule_job(job)
            logger.info(
                f"Job {job.id} scheduled for retry at {job.scheduled_at}"
            )
        else:
            job.status = JobStatus.FAILED
            logger.error(f"Job {job.id} failed after {job.attempts} attempts")

    def cleanup_old_jobs(self) -> int:
        """Remove completed/failed jobs older than threshold."""
        threshold = datetime.now() - timedelta(hours=self._config.cleanup_age_hours)
        to_remove = []

        for job_id, job in self._jobs.items():
            if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                if job.created_at < threshold:
                    to_remove.append(job_id)

        for job_id in to_remove:
            job = self._jobs.pop(job_id)
            if job.record_id in self._jobs_by_record:
                self._jobs_by_record[job.record_id].discard(job_id)

        return len(to_remove)


class AsyncioScheduler(BaseEscalationScheduler):
    """Asyncio-based escalation scheduler.

    Uses asyncio tasks for lightweight scheduling without
    external dependencies.
    """

    def __init__(
        self,
        config: SchedulerConfig | None = None,
        executor: AsyncJobExecutor | None = None,
    ) -> None:
        """Initialize asyncio scheduler."""
        super().__init__(config, executor)
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._check_task: asyncio.Task[None] | None = None
        self._cleanup_task: asyncio.Task[None] | None = None

    def start(self) -> None:
        """Start the scheduler."""
        if self._is_running:
            return

        self._is_running = True

        # Schedule periodic check for due jobs
        loop = asyncio.get_event_loop()
        self._check_task = loop.create_task(self._periodic_check())
        self._cleanup_task = loop.create_task(self._periodic_cleanup())

        logger.info("AsyncioScheduler started")

    def stop(self) -> None:
        """Stop the scheduler."""
        self._is_running = False

        # Cancel all pending tasks
        for task in self._tasks.values():
            task.cancel()
        self._tasks.clear()

        if self._check_task:
            self._check_task.cancel()
        if self._cleanup_task:
            self._cleanup_task.cancel()

        logger.info("AsyncioScheduler stopped")

    def _schedule_job(self, job: ScheduledJob) -> None:
        """Schedule a job with asyncio."""
        delay = (job.scheduled_at - datetime.now()).total_seconds()
        if delay < 0:
            delay = 0

        async def delayed_execute() -> None:
            await asyncio.sleep(delay)
            if job.status == JobStatus.PENDING:
                await self._execute_job(job)

        loop = asyncio.get_event_loop()
        task = loop.create_task(delayed_execute())
        self._tasks[job.id] = task

    def _remove_job_from_backend(self, job_id: str) -> None:
        """Remove a job from asyncio tasks."""
        task = self._tasks.pop(job_id, None)
        if task and not task.done():
            task.cancel()

    def _reschedule_in_backend(self, job: ScheduledJob) -> None:
        """Reschedule a job in asyncio."""
        self._remove_job_from_backend(job.id)
        self._schedule_job(job)

    async def _periodic_check(self) -> None:
        """Periodically check for due jobs."""
        while self._is_running:
            await asyncio.sleep(self._config.check_interval_seconds)

            for job in list(self._jobs.values()):
                if job.is_due and job.id not in self._tasks:
                    self._schedule_job(job)

    async def _periodic_cleanup(self) -> None:
        """Periodically clean up old jobs."""
        while self._is_running:
            await asyncio.sleep(self._config.cleanup_interval_seconds)
            removed = self.cleanup_old_jobs()
            if removed > 0:
                logger.info(f"Cleaned up {removed} old jobs")


class APSchedulerWrapper(BaseEscalationScheduler):
    """APScheduler-based escalation scheduler.

    Uses APScheduler for robust, persistent job scheduling
    with support for multiple job stores.

    Requires: pip install apscheduler>=4.0
    """

    def __init__(
        self,
        config: SchedulerConfig | None = None,
        executor: AsyncJobExecutor | None = None,
    ) -> None:
        """Initialize APScheduler wrapper."""
        super().__init__(config, executor)
        self._scheduler: Any = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Ensure APScheduler is initialized."""
        if self._initialized:
            return

        try:
            from apscheduler.schedulers.asyncio import AsyncIOScheduler
            from apscheduler.triggers.date import DateTrigger

            self._DateTrigger = DateTrigger

            scheduler_config = self._config.apscheduler_config.copy()

            # Set default job defaults
            job_defaults = scheduler_config.pop("job_defaults", {})
            job_defaults.setdefault("coalesce", True)
            job_defaults.setdefault("max_instances", self._config.max_concurrent_jobs)
            job_defaults.setdefault("misfire_grace_time", 60)

            self._scheduler = AsyncIOScheduler(
                job_defaults=job_defaults,
                **scheduler_config,
            )

            self._initialized = True
            logger.info("APScheduler initialized successfully")

        except ImportError:
            logger.warning(
                "APScheduler not installed. Install with: pip install apscheduler>=4.0"
            )
            raise RuntimeError(
                "APScheduler not available. Install with: pip install apscheduler>=4.0"
            )

    def start(self) -> None:
        """Start the scheduler."""
        if self._is_running:
            return

        self._ensure_initialized()
        self._scheduler.start()
        self._is_running = True

        # Schedule cleanup job
        self._scheduler.add_job(
            self._cleanup_wrapper,
            "interval",
            seconds=self._config.cleanup_interval_seconds,
            id="escalation_cleanup",
            replace_existing=True,
        )

        logger.info("APScheduler started")

    def stop(self) -> None:
        """Stop the scheduler."""
        if not self._is_running:
            return

        if self._scheduler:
            self._scheduler.shutdown(wait=True)

        self._is_running = False
        logger.info("APScheduler stopped")

    def _schedule_job(self, job: ScheduledJob) -> None:
        """Schedule a job with APScheduler."""
        if not self._scheduler:
            logger.error("Scheduler not initialized")
            return

        trigger = self._DateTrigger(run_date=job.scheduled_at)

        self._scheduler.add_job(
            self._job_wrapper,
            trigger,
            args=[job.id],
            id=job.id,
            replace_existing=True,
            name=f"escalation_{job.record_id}_{job.target_level}",
        )

    def _remove_job_from_backend(self, job_id: str) -> None:
        """Remove a job from APScheduler."""
        if self._scheduler:
            try:
                self._scheduler.remove_job(job_id)
            except Exception:
                pass  # Job may not exist

    def _reschedule_in_backend(self, job: ScheduledJob) -> None:
        """Reschedule a job in APScheduler."""
        if self._scheduler:
            trigger = self._DateTrigger(run_date=job.scheduled_at)
            self._scheduler.reschedule_job(job.id, trigger=trigger)

    async def _job_wrapper(self, job_id: str) -> None:
        """Wrapper to execute a job from APScheduler."""
        job = self._jobs.get(job_id)
        if job:
            await self._execute_job(job)

    async def _cleanup_wrapper(self) -> None:
        """Wrapper for cleanup job."""
        removed = self.cleanup_old_jobs()
        if removed > 0:
            logger.info(f"Cleaned up {removed} old jobs")


class InMemoryScheduler(BaseEscalationScheduler):
    """In-memory scheduler for testing.

    Provides immediate job execution without actual scheduling.
    """

    def __init__(
        self,
        config: SchedulerConfig | None = None,
        executor: AsyncJobExecutor | None = None,
    ) -> None:
        """Initialize in-memory scheduler."""
        super().__init__(config, executor)
        self._pending_jobs: list[ScheduledJob] = []

    def start(self) -> None:
        """Start the scheduler."""
        self._is_running = True
        logger.info("InMemoryScheduler started")

    def stop(self) -> None:
        """Stop the scheduler."""
        self._is_running = False
        logger.info("InMemoryScheduler stopped")

    def _schedule_job(self, job: ScheduledJob) -> None:
        """Add job to pending list."""
        self._pending_jobs.append(job)

    def _remove_job_from_backend(self, job_id: str) -> None:
        """Remove job from pending list."""
        self._pending_jobs = [j for j in self._pending_jobs if j.id != job_id]

    def _reschedule_in_backend(self, job: ScheduledJob) -> None:
        """Reschedule job in pending list."""
        pass  # Time is already updated in job object

    async def process_due_jobs(self) -> int:
        """Process all due jobs (for testing).

        Returns:
            Number of jobs processed.
        """
        processed = 0
        due_jobs = [j for j in self._pending_jobs if j.is_due]

        for job in due_jobs:
            await self._execute_job(job)
            processed += 1

        self._pending_jobs = [j for j in self._pending_jobs if j.status == JobStatus.PENDING]

        return processed

    def get_pending_count(self) -> int:
        """Get count of pending jobs."""
        return len([j for j in self._pending_jobs if j.status == JobStatus.PENDING])


def create_scheduler(
    config: SchedulerConfig | None = None,
    executor: AsyncJobExecutor | None = None,
) -> BaseEscalationScheduler:
    """Factory function to create a scheduler.

    Args:
        config: Scheduler configuration.
        executor: Job executor callback.

    Returns:
        Appropriate scheduler implementation.
    """
    cfg = config or SchedulerConfig()

    if cfg.scheduler_type == SchedulerType.APSCHEDULER:
        try:
            return APSchedulerWrapper(cfg, executor)
        except RuntimeError:
            logger.warning("Falling back to AsyncioScheduler")
            return AsyncioScheduler(cfg, executor)

    elif cfg.scheduler_type == SchedulerType.MEMORY:
        return InMemoryScheduler(cfg, executor)

    else:
        return AsyncioScheduler(cfg, executor)
