"""Tests for escalation scheduler."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

import pytest

from truthound.checkpoint.escalation.scheduler import (
    AsyncioScheduler,
    InMemoryScheduler,
    JobStatus,
    ScheduledJob,
    SchedulerConfig,
    SchedulerType,
    create_scheduler,
)


class TestScheduledJob:
    """Tests for ScheduledJob."""

    def test_create_job(self) -> None:
        """Test creating a scheduled job."""
        job = ScheduledJob(
            id="job-123",
            record_id="record-1",
            policy_name="policy-1",
            target_level=2,
            scheduled_at=datetime.now() + timedelta(minutes=15),
        )
        assert job.id == "job-123"
        assert job.target_level == 2
        assert job.status == JobStatus.PENDING

    def test_job_is_due(self) -> None:
        """Test job due checking."""
        past_job = ScheduledJob(
            id="job-1",
            record_id="record-1",
            policy_name="policy-1",
            target_level=1,
            scheduled_at=datetime.now() - timedelta(minutes=1),
        )
        assert past_job.is_due

        future_job = ScheduledJob(
            id="job-2",
            record_id="record-1",
            policy_name="policy-1",
            target_level=1,
            scheduled_at=datetime.now() + timedelta(minutes=10),
        )
        assert not future_job.is_due

    def test_job_can_retry(self) -> None:
        """Test retry capability checking."""
        job = ScheduledJob(
            id="job-1",
            record_id="record-1",
            policy_name="policy-1",
            target_level=1,
            scheduled_at=datetime.now(),
            attempts=0,
            max_attempts=3,
        )
        assert job.can_retry

        job.attempts = 3
        assert not job.can_retry

    def test_job_serialization(self) -> None:
        """Test job serialization round-trip."""
        job = ScheduledJob(
            id="job-123",
            record_id="record-1",
            policy_name="policy-1",
            target_level=2,
            scheduled_at=datetime.now(),
            metadata={"key": "value"},
        )

        data = job.to_dict()
        restored = ScheduledJob.from_dict(data)

        assert restored.id == job.id
        assert restored.record_id == job.record_id
        assert restored.target_level == job.target_level
        assert restored.metadata["key"] == "value"


class TestSchedulerConfig:
    """Tests for SchedulerConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = SchedulerConfig()
        assert config.scheduler_type == SchedulerType.ASYNCIO
        assert config.check_interval_seconds == 10
        assert config.max_concurrent_jobs == 100
        assert config.job_timeout_seconds == 300

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = SchedulerConfig(
            scheduler_type=SchedulerType.MEMORY,
            check_interval_seconds=5,
            max_concurrent_jobs=50,
        )
        assert config.scheduler_type == SchedulerType.MEMORY
        assert config.check_interval_seconds == 5
        assert config.max_concurrent_jobs == 50


class TestInMemoryScheduler:
    """Tests for InMemoryScheduler."""

    def test_start_stop(self) -> None:
        """Test starting and stopping scheduler."""
        scheduler = InMemoryScheduler()
        assert not scheduler.is_running

        scheduler.start()
        assert scheduler.is_running

        scheduler.stop()
        assert not scheduler.is_running

    def test_schedule_escalation(self) -> None:
        """Test scheduling an escalation."""
        scheduler = InMemoryScheduler()
        scheduler.start()

        job = scheduler.schedule_escalation(
            record_id="record-1",
            policy_name="policy-1",
            target_level=2,
            delay=timedelta(minutes=15),
        )

        assert job.record_id == "record-1"
        assert job.target_level == 2
        assert job.status == JobStatus.PENDING

    def test_cancel_escalation(self) -> None:
        """Test cancelling scheduled escalations."""
        scheduler = InMemoryScheduler()
        scheduler.start()

        scheduler.schedule_escalation(
            record_id="record-1",
            policy_name="policy-1",
            target_level=2,
            delay=timedelta(minutes=15),
        )
        scheduler.schedule_escalation(
            record_id="record-1",
            policy_name="policy-1",
            target_level=3,
            delay=timedelta(minutes=30),
        )

        cancelled = scheduler.cancel_escalation("record-1")
        assert cancelled == 2

        jobs = scheduler.get_scheduled_jobs("record-1")
        assert all(j.status == JobStatus.CANCELLED for j in jobs)

    def test_get_scheduled_jobs(self) -> None:
        """Test getting scheduled jobs."""
        scheduler = InMemoryScheduler()
        scheduler.start()

        scheduler.schedule_escalation(
            "record-1", "policy-1", 2, timedelta(minutes=5)
        )
        scheduler.schedule_escalation(
            "record-2", "policy-1", 2, timedelta(minutes=10)
        )

        all_jobs = scheduler.get_scheduled_jobs()
        assert len(all_jobs) == 2

        record1_jobs = scheduler.get_scheduled_jobs("record-1")
        assert len(record1_jobs) == 1

    def test_reschedule(self) -> None:
        """Test rescheduling a job."""
        scheduler = InMemoryScheduler()
        scheduler.start()

        job = scheduler.schedule_escalation(
            "record-1", "policy-1", 2, timedelta(minutes=15)
        )
        original_time = job.scheduled_at

        new_time = datetime.now() + timedelta(minutes=30)
        updated = scheduler.reschedule(job.id, new_time)

        assert updated is not None
        assert updated.scheduled_at == new_time
        assert updated.scheduled_at != original_time

    def test_get_pending_count(self) -> None:
        """Test getting pending job count."""
        scheduler = InMemoryScheduler()
        scheduler.start()

        scheduler.schedule_escalation(
            "record-1", "policy-1", 2, timedelta(minutes=5)
        )
        scheduler.schedule_escalation(
            "record-2", "policy-1", 2, timedelta(minutes=10)
        )

        assert scheduler.get_pending_count() == 2

    @pytest.mark.asyncio
    async def test_process_due_jobs(self) -> None:
        """Test processing due jobs."""
        scheduler = InMemoryScheduler()
        executed_jobs = []

        async def executor(job: ScheduledJob) -> None:
            executed_jobs.append(job.id)

        scheduler.set_executor(executor)
        scheduler.start()

        # Schedule a job that's immediately due
        job = scheduler.schedule_escalation(
            "record-1", "policy-1", 2, timedelta(seconds=0)
        )

        processed = await scheduler.process_due_jobs()
        assert processed == 1
        assert job.id in executed_jobs

    @pytest.mark.asyncio
    async def test_job_execution_failure_retry(self) -> None:
        """Test job retry on failure."""
        scheduler = InMemoryScheduler()
        call_count = 0

        async def failing_executor(job: ScheduledJob) -> None:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Temporary failure")

        scheduler.set_executor(failing_executor)
        scheduler.start()

        job = scheduler.schedule_escalation(
            "record-1", "policy-1", 2, timedelta(seconds=0)
        )

        # First attempt fails
        await scheduler.process_due_jobs()
        assert job.attempts == 1
        assert job.status == JobStatus.PENDING

    def test_cleanup_old_jobs(self) -> None:
        """Test cleaning up old jobs."""
        scheduler = InMemoryScheduler(
            SchedulerConfig(cleanup_age_hours=1)
        )
        scheduler.start()

        # Create and complete a job
        job = scheduler.schedule_escalation(
            "record-1", "policy-1", 2, timedelta(seconds=0)
        )
        job.status = JobStatus.COMPLETED
        job.created_at = datetime.now() - timedelta(hours=2)

        removed = scheduler.cleanup_old_jobs()
        assert removed == 1


class TestAsyncioScheduler:
    """Tests for AsyncioScheduler."""

    def test_start_stop(self) -> None:
        """Test starting and stopping scheduler."""
        scheduler = AsyncioScheduler()
        assert not scheduler.is_running

        # Note: Full start requires running event loop
        # This tests the basic state management
        scheduler._is_running = True
        assert scheduler.is_running

    def test_schedule_escalation_basic(self) -> None:
        """Test basic escalation scheduling."""
        scheduler = AsyncioScheduler()
        scheduler._is_running = True

        # Manual job tracking (without event loop)
        job = scheduler.schedule_escalation(
            record_id="record-1",
            policy_name="policy-1",
            target_level=2,
            delay=timedelta(minutes=15),
        )

        assert job.record_id == "record-1"
        assert job in scheduler._jobs.values()

    def test_cancel_escalation_basic(self) -> None:
        """Test basic escalation cancellation."""
        scheduler = AsyncioScheduler()
        scheduler._is_running = True

        scheduler.schedule_escalation(
            "record-1", "policy-1", 2, timedelta(minutes=15)
        )

        cancelled = scheduler.cancel_escalation("record-1")
        assert cancelled == 1


class TestCreateScheduler:
    """Tests for scheduler factory function."""

    def test_create_memory_scheduler(self) -> None:
        """Test creating memory scheduler."""
        config = SchedulerConfig(scheduler_type=SchedulerType.MEMORY)
        scheduler = create_scheduler(config)
        assert isinstance(scheduler, InMemoryScheduler)

    def test_create_asyncio_scheduler(self) -> None:
        """Test creating asyncio scheduler."""
        config = SchedulerConfig(scheduler_type=SchedulerType.ASYNCIO)
        scheduler = create_scheduler(config)
        assert isinstance(scheduler, AsyncioScheduler)

    def test_create_default_scheduler(self) -> None:
        """Test creating scheduler with defaults."""
        scheduler = create_scheduler()
        assert isinstance(scheduler, AsyncioScheduler)

    def test_create_with_executor(self) -> None:
        """Test creating scheduler with executor."""
        async def executor(job: ScheduledJob) -> None:
            pass

        scheduler = create_scheduler(executor=executor)
        assert scheduler._executor == executor
