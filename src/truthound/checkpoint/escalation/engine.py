"""Escalation Policy Engine.

This module provides the core orchestration logic for the
escalation policy system.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import Awaitable, Callable
from contextlib import nullcontext, suppress
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from truthound.checkpoint.escalation.protocols import (
    BaseEscalationStore,
    EscalationLevel,
    EscalationPolicy,
    EscalationPolicyConfig,
    EscalationRecord,
    EscalationResult,
    EscalationStats,
    EscalationTarget,
    EscalationTrigger,
)
from truthound.checkpoint.escalation.scheduler import (
    BaseEscalationScheduler,
    ScheduledJob,
    SchedulerConfig,
    create_scheduler,
)
from truthound.checkpoint.escalation.states import (
    EscalationState,
    EscalationStateManager,
)
from truthound.checkpoint.escalation.stores import create_store

logger = logging.getLogger(__name__)


# Type alias for notification handler
NotificationHandler = Callable[
    [EscalationRecord, EscalationLevel, list[EscalationTarget]],
    Awaitable[bool],
]

# Type alias for condition evaluator
ConditionEvaluator = Callable[
    [EscalationRecord, dict[str, Any]],
    Awaitable[bool],
]


@runtime_checkable
class EscalationEngineProtocol(Protocol):
    """Protocol for escalation engine implementations."""

    async def trigger(
        self,
        incident_id: str,
        context: dict[str, Any] | None = None,
        policy_name: str | None = None,
    ) -> EscalationResult:
        """Trigger a new escalation."""
        ...

    async def acknowledge(
        self,
        record_id: str,
        acknowledged_by: str,
    ) -> EscalationResult:
        """Acknowledge an escalation."""
        ...

    async def resolve(
        self,
        record_id: str,
        resolved_by: str,
    ) -> EscalationResult:
        """Resolve an escalation."""
        ...

    async def cancel(
        self,
        record_id: str,
        cancelled_by: str,
        reason: str = "",
    ) -> EscalationResult:
        """Cancel an escalation."""
        ...

    def get_record(self, record_id: str) -> EscalationRecord | None:
        """Get an escalation record."""
        ...

    def get_active_escalations(
        self,
        policy_name: str | None = None,
    ) -> list[EscalationRecord]:
        """Get active escalations."""
        ...


@dataclass
class EscalationEngineConfig:
    """Configuration for the escalation engine.

    Attributes:
        store_type: Storage backend type.
        store_config: Storage backend configuration.
        scheduler_config: Scheduler configuration.
        default_notification_handler: Default notification handler.
        condition_evaluators: Custom condition evaluators by name.
        check_business_hours: Whether to check business hours.
        metrics_enabled: Enable metrics collection.
    """

    store_type: str = "memory"
    store_config: dict[str, Any] = field(default_factory=dict)
    scheduler_config: SchedulerConfig | None = None
    default_notification_handler: NotificationHandler | None = None
    condition_evaluators: dict[str, ConditionEvaluator] = field(default_factory=dict)
    check_business_hours: bool = True
    metrics_enabled: bool = True


class EscalationEngine:
    """Core escalation policy engine.

    Orchestrates the complete escalation lifecycle including
    triggering, notifications, acknowledgment, and resolution.

    Example:
        >>> engine = EscalationEngine(config)
        >>> engine.register_policy(policy)
        >>> await engine.start()
        >>> result = await engine.trigger("incident-123", context)
        >>> await engine.acknowledge(result.record.id, "user-123")
    """

    def __init__(
        self,
        config: EscalationEngineConfig | None = None,
    ) -> None:
        """Initialize escalation engine.

        Args:
            config: Engine configuration.
        """
        self._config = config or EscalationEngineConfig()
        scheduler_config = self._config.scheduler_config
        if scheduler_config and scheduler_config.journal_path:
            database = self._config.store_config.get("database", ":memory:")
            if (
                self._config.store_type != "sqlite"
                or str(database) == ":memory:"
                or Path(database).resolve() != Path(scheduler_config.journal_path).resolve()
            ):
                raise ValueError(
                    "Durable execution requires journal and records in one SQLite file"
                )
        self._policies: dict[str, EscalationPolicy] = {}
        self._store: BaseEscalationStore = create_store(
            self._config.store_type,
            **self._config.store_config,
        )
        self._scheduler: BaseEscalationScheduler = create_scheduler(
            self._config.scheduler_config,
            self._execute_escalation_job,
        )
        self._state_manager = EscalationStateManager()
        self._notification_handler = self._config.default_notification_handler
        self._condition_evaluators = self._config.condition_evaluators.copy()
        self._is_running = False
        self._lock = asyncio.Lock()
        self._notification_journal = None
        if self._config.scheduler_config and self._config.scheduler_config.journal_path:
            from truthound.checkpoint.escalation.journal import ExecutionJournal

            self._scheduler._journal.close()
            self._notification_journal = ExecutionJournal(
                self._config.scheduler_config.journal_path,
                connection=self._store._get_connection(),
            )
            self._scheduler._journal = self._notification_journal
        self._closed = False
        self._execution_guard = None

    @property
    def is_running(self) -> bool:
        """Check if engine is running."""
        return self._is_running

    @property
    def store(self) -> BaseEscalationStore:
        """Get the underlying store."""
        return self._store

    def register_policy(self, policy: EscalationPolicy) -> None:
        """Register an escalation policy.

        Args:
            policy: Policy to register.
        """
        self._policies[policy.name] = policy
        logger.info(f"Registered escalation policy: {policy.name}")

    def unregister_policy(self, policy_name: str) -> bool:
        """Unregister an escalation policy.

        Args:
            policy_name: Policy name to unregister.

        Returns:
            True if unregistered, False if not found.
        """
        if policy_name in self._policies:
            del self._policies[policy_name]
            logger.info(f"Unregistered escalation policy: {policy_name}")
            return True
        return False

    def get_policy(self, policy_name: str) -> EscalationPolicy | None:
        """Get a registered policy.

        Args:
            policy_name: Policy name.

        Returns:
            Policy if found, None otherwise.
        """
        return self._policies.get(policy_name)

    def set_notification_handler(self, handler: NotificationHandler) -> None:
        """Set the notification handler.

        Args:
            handler: Async function to send notifications.
        """
        self._notification_handler = handler

    def register_condition_evaluator(
        self,
        name: str,
        evaluator: ConditionEvaluator,
    ) -> None:
        """Register a custom condition evaluator.

        Args:
            name: Condition name.
            evaluator: Async evaluator function.
        """
        self._condition_evaluators[name] = evaluator

    async def start(self) -> None:
        """Start the escalation engine."""
        if self._is_running:
            return

        self._ensure_owner()
        self._scheduler.start()
        self._is_running = True
        logger.info("Escalation engine started")

    def bind_recovery_scope(self, scope: str) -> None:
        """Pin durable recovery to the current operator authorization."""
        if self._notification_journal:
            self._notification_journal.bind_scope(scope)

    def set_execution_guard(self, guard: Callable[[], None]) -> None:
        """Set an operator authorization check invoked before scheduled effects."""
        self._execution_guard = guard

    def _ensure_owner(self) -> None:
        if self._execution_guard:
            self._execution_guard()
        if self._closed:
            raise RuntimeError("Escalation engine is closed")
        if self._notification_journal:
            self._notification_journal.acquire_owner()

    @staticmethod
    def _policy_digest(policy):
        return hashlib.sha256(
            json.dumps(policy.to_dict(), sort_keys=True, allow_nan=False).encode()
        ).hexdigest()

    def _transaction(self):
        return (
            self._notification_journal.transaction()
            if self._notification_journal
            else nullcontext()
        )

    @staticmethod
    def _notification_identity(record, level):
        return f"notification:{record.id}:{record.escalation_count}:{level.level}"

    def _complete_notification(self, record, level):
        if self._notification_journal and self._notification_handler:
            self._notification_journal.finish(
                self._notification_identity(record, level), "completed"
            )

    def recovery_status(self, *, limit: int = 100, offset: int = 0) -> dict[str, Any]:
        """Report durable pending and uncertain attempts without replaying effects."""
        journal = self._notification_journal
        if not 1 <= limit <= 1000 or offset < 0:
            raise ValueError("Invalid recovery page")
        rows = journal.attempts(limit=limit + 1, offset=offset) if journal else []
        return {
            "enabled": journal is not None,
            "attempts": [
                {"identity": identity, "state": state} for identity, state, _ in rows[:limit]
            ],
            "next_offset": offset + limit if len(rows) > limit else None,
        }

    async def recovery_resolve(self, identity: str, resolution: str) -> dict[str, Any]:
        """Record an operator's confirmation or discard; never send a notification."""
        async with self._lock:
            self._ensure_owner()
            if self._is_running or not self._notification_journal:
                raise RuntimeError("Recovery requires a stopped durable engine")
            journal = self._notification_journal
            rows = journal.connection.execute(
                "SELECT payload FROM execution_attempts WHERE identity=?", (identity,)
            ).fetchone()
            if not rows or not identity.startswith(("notification:", "job:")):
                raise ValueError("Unknown escalation attempt")
            with journal.transaction():
                journal.resolve_unknown(identity, resolution)
                if identity.startswith("notification:"):
                    payload = json.loads(rows[0])
                    record = self._store.get(payload["record_id"])
                    if record is None:
                        raise ValueError("Missing incident record")
                    self._scheduler.cancel_escalation(record.id)
                    record.next_escalation_at = None
                    if resolution == "completed":
                        policy = self._policies.get(record.policy_name)
                        if (
                            not policy
                            or not policy.enabled
                            or payload.get("policy_digest") != self._policy_digest(policy)
                        ):
                            raise ValueError("Notification policy changed")
                        level = policy.get_level(record.current_level)
                        if level is None or not record.is_active:
                            raise ValueError("Incident cannot continue")
                        record.notification_count += len(level.targets)
                        if record.state == "escalating":
                            self._state_manager.complete_escalation(record)
                        following = policy.get_next_level(level.level)
                        if following:
                            record.next_escalation_at = datetime.now() + following.delay
                            self._scheduler.schedule_escalation(
                                record.id,
                                policy.name,
                                following.level,
                                following.delay,
                                metadata={"policy_digest": self._policy_digest(policy)},
                            )
                    self._store.save(record)
            return {"identity": identity, "state": resolution}

    async def recovery_forget(self, identities: list[str]) -> dict[str, Any]:
        """Remove explicitly selected terminal receipts while quiescent."""
        async with self._lock:
            self._ensure_owner()
            if self._is_running or not self._notification_journal:
                raise RuntimeError("Retention requires a stopped durable engine")
            if any(not identity.startswith(("notification:", "job:")) for identity in identities):
                raise ValueError("Invalid escalation attempt identity")
            return {"removed": self._notification_journal.forget_terminal(identities)}

    async def close(self) -> None:
        """Stop, drain tasks, then release durable ownership and connections."""
        await self.stop()
        async with self._lock:
            if self._closed:
                return
            if self._notification_journal:
                self._notification_journal.close()
                self._store._get_connection().close()
            self._closed = True

    async def stop(self) -> None:
        """Stop the escalation engine."""
        tasks = set(getattr(self._scheduler, "_tasks", {}).values())
        tasks.update(self._scheduler._active_tasks)
        tasks.discard(asyncio.current_task())
        self._scheduler.stop()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        # APScheduler queues its native shutdown callback on this event loop.
        await asyncio.sleep(0)
        self._is_running = False
        logger.info("Escalation engine stopped")

    async def trigger(
        self,
        incident_id: str,
        context: dict[str, Any] | None = None,
        policy_name: str | None = None,
        trigger_type: EscalationTrigger = EscalationTrigger.UNACKNOWLEDGED,
    ) -> EscalationResult:
        """Trigger a new escalation.

        Args:
            incident_id: External incident identifier.
            context: Optional context data.
            policy_name: Policy to use (or first matching).
            trigger_type: Type of trigger.

        Returns:
            EscalationResult with created record.
        """
        async with self._lock:
            self._ensure_owner()
            # Find policy
            policy = self._find_policy(policy_name, context or {}, trigger_type)
            if not policy:
                return EscalationResult.failure_result(
                    None,
                    "trigger",
                    f"No matching policy found for {policy_name or 'auto'}",
                )

            # Check for existing active escalation
            existing = self._store.get_by_incident(incident_id)
            for record in existing:
                if record.is_active and record.policy_name == policy.name:
                    logger.info(f"Active escalation already exists for incident {incident_id}")
                    return EscalationResult.success_result(
                        record,
                        "trigger_existing",
                        message="Existing escalation found",
                    )

            # Check business hours
            if policy.business_hours_only and not self._is_business_hours(policy):
                logger.info(f"Outside business hours for policy {policy.name}")
                # Schedule for next business hours
                # For now, just proceed with escalation

            if self._notification_journal:
                for previous in existing:
                    blocked = self._notification_journal.connection.execute(
                        "SELECT 1 FROM execution_attempts WHERE identity LIKE 'notification:%' "
                        "AND json_extract(payload,'$.record_id')=? "
                        "AND state IN ('pending','attempting','unknown') LIMIT 1",
                        (previous.id,),
                    ).fetchone()
                    if blocked:
                        raise RuntimeError("Incident has an unresolved notification outcome")

            # Create record
            record = EscalationRecord.create(
                incident_id=incident_id,
                policy_name=policy.name,
                context=context or {},
            )

            # Start escalation
            transition = self._state_manager.start(record)
            if not transition:
                return EscalationResult.failure_result(
                    record,
                    "trigger",
                    "Failed to start escalation",
                )

            # Get first level
            first_level = policy.get_level(1)
            if not first_level:
                return EscalationResult.failure_result(
                    record,
                    "trigger",
                    "Policy has no level 1",
                )

            # Persist incident identity before any external effect, so restart
            # deduplication can recognize an interrupted initial notification.
            if self._notification_journal:
                self._store.save(record)
            # Send initial notification
            notified_targets = await self._send_notifications(
                record,
                first_level,
            )

            with self._transaction():
                self._complete_notification(record, first_level)
                record.notification_count += len(notified_targets)

                # Schedule next escalation
                next_level = policy.get_next_level(1)
                if next_level:
                    record.next_escalation_at = datetime.now() + next_level.delay
                    self._scheduler.schedule_escalation(
                        record_id=record.id,
                        policy_name=policy.name,
                        target_level=next_level.level,
                        delay=next_level.delay,
                        metadata={"policy_digest": self._policy_digest(policy)}
                        if self._notification_journal
                        else None,
                    )

                # Save record
                self._store.save(record)

            logger.info(
                f"Triggered escalation {record.id} for incident {incident_id} "
                f"at level {record.current_level}"
            )

            return EscalationResult.success_result(
                record,
                "trigger",
                notified_targets,
                f"Escalation triggered at level {record.current_level}",
            )

    async def acknowledge(
        self,
        record_id: str,
        acknowledged_by: str,
    ) -> EscalationResult:
        """Acknowledge an escalation.

        Args:
            record_id: Record ID to acknowledge.
            acknowledged_by: Who acknowledged.

        Returns:
            EscalationResult.
        """
        async with self._lock:
            self._ensure_owner()
            record = self._store.get(record_id)
            if not record:
                return EscalationResult.failure_result(
                    None,
                    "acknowledge",
                    f"Record not found: {record_id}",
                )

            if not EscalationState(record.state).allows_acknowledgment:
                return EscalationResult.failure_result(
                    record,
                    "acknowledge",
                    f"Cannot acknowledge in state: {record.state}",
                )

            transition = self._state_manager.acknowledge(record, acknowledged_by)
            if not transition:
                return EscalationResult.failure_result(
                    record,
                    "acknowledge",
                    "Acknowledgment transition failed",
                )

            # Cancel pending escalations
            cancelled = self._scheduler.cancel_escalation(record_id)
            record.next_escalation_at = None

            self._store.save(record)

            logger.info(
                f"Escalation {record_id} acknowledged by {acknowledged_by}, "
                f"cancelled {cancelled} pending jobs"
            )

            return EscalationResult.success_result(
                record,
                "acknowledge",
                message=f"Acknowledged by {acknowledged_by}",
            )

    async def resolve(
        self,
        record_id: str,
        resolved_by: str,
    ) -> EscalationResult:
        """Resolve an escalation.

        Args:
            record_id: Record ID to resolve.
            resolved_by: Who resolved.

        Returns:
            EscalationResult.
        """
        async with self._lock:
            self._ensure_owner()
            record = self._store.get(record_id)
            if not record:
                return EscalationResult.failure_result(
                    None,
                    "resolve",
                    f"Record not found: {record_id}",
                )

            if not EscalationState(record.state).allows_resolution:
                return EscalationResult.failure_result(
                    record,
                    "resolve",
                    f"Cannot resolve in state: {record.state}",
                )

            transition = self._state_manager.resolve(record, resolved_by)
            if not transition:
                return EscalationResult.failure_result(
                    record,
                    "resolve",
                    "Resolution transition failed",
                )

            # Cancel pending escalations
            self._scheduler.cancel_escalation(record_id)
            record.next_escalation_at = None

            self._store.save(record)

            logger.info(f"Escalation {record_id} resolved by {resolved_by}")

            return EscalationResult.success_result(
                record,
                "resolve",
                message=f"Resolved by {resolved_by}",
            )

    async def cancel(
        self,
        record_id: str,
        cancelled_by: str,
        reason: str = "",
    ) -> EscalationResult:
        """Cancel an escalation.

        Args:
            record_id: Record ID to cancel.
            cancelled_by: Who cancelled.
            reason: Cancellation reason.

        Returns:
            EscalationResult.
        """
        async with self._lock:
            self._ensure_owner()
            record = self._store.get(record_id)
            if not record:
                return EscalationResult.failure_result(
                    None,
                    "cancel",
                    f"Record not found: {record_id}",
                )

            transition = self._state_manager.cancel(record, cancelled_by, reason)
            if not transition:
                return EscalationResult.failure_result(
                    record,
                    "cancel",
                    "Cancellation transition failed",
                )

            self._scheduler.cancel_escalation(record_id)
            record.next_escalation_at = None

            self._store.save(record)

            logger.info(f"Escalation {record_id} cancelled by {cancelled_by}: {reason}")

            return EscalationResult.success_result(
                record,
                "cancel",
                message=f"Cancelled by {cancelled_by}",
            )

    async def escalate(
        self,
        record_id: str,
        force: bool = False,
    ) -> EscalationResult:
        """Manually escalate to next level.

        Args:
            record_id: Record ID to escalate.
            force: Force escalation even if acknowledged.

        Returns:
            EscalationResult.
        """
        async with self._lock:
            self._ensure_owner()
            record = self._store.get(record_id)
            if not record:
                return EscalationResult.failure_result(
                    None,
                    "escalate",
                    f"Record not found: {record_id}",
                )

            policy = self._policies.get(record.policy_name)
            if not policy:
                return EscalationResult.failure_result(
                    record,
                    "escalate",
                    f"Policy not found: {record.policy_name}",
                )

            # Check if escalation is allowed
            state = EscalationState(record.state)
            if not force and not state.is_active:
                return EscalationResult.failure_result(
                    record,
                    "escalate",
                    f"Cannot escalate in state: {record.state}",
                )

            return await self._perform_escalation(record, policy)

    def get_record(self, record_id: str) -> EscalationRecord | None:
        """Get an escalation record.

        Args:
            record_id: Record ID.

        Returns:
            Record if found.
        """
        return self._store.get(record_id)

    def get_records_for_incident(self, incident_id: str) -> list[EscalationRecord]:
        """Get all records for an incident.

        Args:
            incident_id: Incident ID.

        Returns:
            List of records.
        """
        return self._store.get_by_incident(incident_id)

    def get_active_escalations(
        self,
        policy_name: str | None = None,
    ) -> list[EscalationRecord]:
        """Get active escalations.

        Args:
            policy_name: Optional policy filter.

        Returns:
            List of active records.
        """
        return self._store.get_active(policy_name)

    def get_stats(self) -> EscalationStats:
        """Get escalation statistics.

        Returns:
            Current statistics.
        """
        return self._store.get_stats()

    async def _execute_escalation_job(self, job: ScheduledJob) -> None:
        """Execute a scheduled escalation job.

        This is called by the scheduler when an escalation is due.

        Args:
            job: The scheduled job.
        """
        async with self._lock:
            self._ensure_owner()
            record = self._store.get(job.record_id)
            if not record:
                logger.warning(f"Record not found for job {job.id}: {job.record_id}")
                return

            if not record.is_active:
                logger.info(f"Skipping escalation for inactive record {record.id}: {record.state}")
                return

            policy = self._policies.get(record.policy_name)
            if not policy:
                logger.error(f"Policy not found: {record.policy_name}")
                return

            if self._notification_journal and (
                not policy.enabled
                or job.metadata.get("policy_digest") != self._policy_digest(policy)
            ):
                raise RuntimeError("Recovered job policy changed or was disabled")

            # Check if we should escalate to this level
            if record.current_level >= job.target_level:
                logger.info(
                    f"Record {record.id} already at level {record.current_level}, "
                    f"target was {job.target_level}"
                )
                return

            await self._perform_escalation(record, policy)

    async def _perform_escalation(
        self,
        record: EscalationRecord,
        policy: EscalationPolicy,
    ) -> EscalationResult:
        """Perform the actual escalation.

        Args:
            record: Record to escalate.
            policy: Policy to use.

        Returns:
            EscalationResult.
        """
        if self._notification_journal:
            blocked = self._notification_journal.connection.execute(
                "SELECT 1 FROM execution_attempts WHERE identity LIKE 'notification:%' "
                "AND json_extract(payload,'$.record_id')=? "
                "AND state IN ('pending','attempting','unknown') LIMIT 1",
                (record.id,),
            ).fetchone()
            if blocked:
                raise RuntimeError("Incident has an unresolved notification outcome")
        if not policy.enabled:
            raise RuntimeError("Escalation policy is disabled")
        next_level = policy.get_next_level(record.current_level)

        if not next_level:
            # Max level reached
            if policy.max_escalations > 0 and record.escalation_count >= policy.max_escalations:
                self._state_manager.timeout(record, "max_escalations_reached")
                self._store.save(record)
                return EscalationResult.success_result(
                    record,
                    "max_level",
                    message="Maximum escalation level reached",
                )

            # Repeat last level
            next_level = policy.get_level(record.current_level)
            if not next_level:
                self._state_manager.timeout(record)
                self._store.save(record)
                return EscalationResult.failure_result(
                    record,
                    "escalate",
                    "No level configuration found",
                )

        # Check level conditions
        if next_level.conditions:
            conditions_met = await self._evaluate_conditions(
                record,
                next_level.conditions,
            )
            if not conditions_met:
                logger.info(f"Level {next_level.level} conditions not met for {record.id}")
                return EscalationResult.success_result(
                    record,
                    "conditions_not_met",
                    message="Level conditions not met",
                )

        # Transition to escalating state
        transition = self._state_manager.escalate(
            record,
            next_level.level,
            "scheduler",
        )

        if not transition:
            return EscalationResult.failure_result(
                record,
                "escalate",
                "State transition failed",
            )

        # Send notifications
        notified_targets = await self._send_notifications(record, next_level)
        with self._transaction():
            self._complete_notification(record, next_level)
            record.notification_count += len(notified_targets)

            # Complete escalation
            self._state_manager.complete_escalation(record)

            # Schedule next escalation
            self._scheduler.cancel_escalation(record.id)
            after_next = policy.get_next_level(next_level.level)

            if after_next:
                record.next_escalation_at = datetime.now() + after_next.delay
                self._scheduler.schedule_escalation(
                    record_id=record.id,
                    policy_name=policy.name,
                    target_level=after_next.level,
                    delay=after_next.delay,
                    metadata={"policy_digest": self._policy_digest(policy)}
                    if self._notification_journal
                    else None,
                )
            elif next_level.repeat_count > 0:
                # Schedule repeat notification
                record.next_escalation_at = datetime.now() + next_level.repeat_interval
                self._scheduler.schedule_escalation(
                    record_id=record.id,
                    policy_name=policy.name,
                    target_level=next_level.level,
                    delay=next_level.repeat_interval,
                    metadata={"policy_digest": self._policy_digest(policy)}
                    if self._notification_journal
                    else None,
                )
            else:
                record.next_escalation_at = None

            self._store.save(record)

        logger.info(
            f"Escalated {record.id} to level {record.current_level}, "
            f"notified {len(notified_targets)} targets"
        )

        return EscalationResult.success_result(
            record,
            "escalate",
            notified_targets,
            f"Escalated to level {record.current_level}",
        )

    async def _send_notifications(
        self,
        record: EscalationRecord,
        level: EscalationLevel,
    ) -> list[EscalationTarget]:
        """Send notifications to targets.

        Args:
            record: Escalation record.
            level: Level with targets.

        Returns:
            List of successfully notified targets.
        """
        if not self._notification_handler:
            logger.warning("No notification handler configured")
            return []

        notified: list[EscalationTarget] = []
        attempt = None
        if self._notification_journal:
            attempt = f"notification:{record.id}:{record.escalation_count}:{level.level}"
            with self._transaction():
                self._store.save(record)
                self._notification_journal.enqueue(
                    attempt,
                    {
                        "record_id": record.id,
                        "policy_digest": self._policy_digest(self._policies[record.policy_name]),
                    },
                )
                if not self._notification_journal.claim(attempt):
                    raise RuntimeError("Notification outcome already recorded or unknown")

        try:
            # Sort targets by priority
            sorted_targets = sorted(level.targets, key=lambda t: t.priority)

            success = await self._notification_handler(record, level, sorted_targets)
            if success:
                notified.extend(sorted_targets)
            else:
                if attempt:
                    raise RuntimeError("Notification outcome is unknown")
                logger.warning(f"Notification handler returned failure for {record.id}")

        except BaseException as e:
            if attempt:
                # Never mask the original error with a second persistence error.
                with suppress(Exception):
                    self._notification_journal.finish(attempt, "unknown")
                raise
            if not isinstance(e, Exception):
                raise
            logger.exception(f"Error sending notifications for {record.id}: {e}")

        return notified

    async def _evaluate_conditions(
        self,
        record: EscalationRecord,
        conditions: dict[str, Any],
    ) -> bool:
        """Evaluate level conditions.

        Args:
            record: Escalation record.
            conditions: Condition definitions.

        Returns:
            True if all conditions met.
        """
        for condition_name, condition_config in conditions.items():
            evaluator = self._condition_evaluators.get(condition_name)
            if evaluator:
                try:
                    result = await evaluator(record, condition_config)
                    if not result:
                        return False
                except Exception as e:
                    logger.exception(f"Error evaluating condition {condition_name}: {e}")
                    return False
            else:
                logger.warning(f"Unknown condition: {condition_name}")

        return True

    def _find_policy(
        self,
        policy_name: str | None,
        context: dict[str, Any],
        trigger_type: EscalationTrigger,
    ) -> EscalationPolicy | None:
        """Find matching policy.

        Args:
            policy_name: Explicit policy name.
            context: Trigger context.
            trigger_type: Type of trigger.

        Returns:
            Matching policy or None.
        """
        if policy_name:
            return self._policies.get(policy_name)

        # Find first matching enabled policy
        severity = context.get("severity", "info")

        for policy in self._policies.values():
            if not policy.enabled:
                continue

            if trigger_type not in policy.triggers:
                continue

            if severity not in policy.severity_filter:
                continue

            return policy

        return None

    def _is_business_hours(self, policy: EscalationPolicy) -> bool:
        """Check if currently within business hours.

        Args:
            policy: Policy with business hours config.

        Returns:
            True if within business hours.
        """
        try:
            from zoneinfo import ZoneInfo

            tz = ZoneInfo(policy.timezone)
            now = datetime.now(tz)

            # Check day of week (0 = Monday)
            if now.weekday() not in policy.business_days:
                return False

            # Check hour
            return policy.business_hours_start <= now.hour < policy.business_hours_end

        except Exception:
            # Default to always business hours if timezone fails
            return True


class EscalationPolicyManager:
    """High-level manager for multiple escalation policies.

    Provides a convenient interface for managing policies and
    coordinating multiple engines.

    Example:
        >>> manager = EscalationPolicyManager()
        >>> manager.load_policies_from_config(config)
        >>> await manager.start()
    """

    def __init__(
        self,
        config: EscalationPolicyConfig | None = None,
    ) -> None:
        """Initialize policy manager.

        Args:
            config: Policy configuration.
        """
        self._config = config or EscalationPolicyConfig()
        self._engine: EscalationEngine | None = None
        self._is_running = False

    @property
    def engine(self) -> EscalationEngine | None:
        """Get the escalation engine."""
        return self._engine

    @property
    def is_running(self) -> bool:
        """Check if manager is running."""
        return self._is_running

    def configure(
        self,
        engine_config: EscalationEngineConfig | None = None,
    ) -> None:
        """Configure the manager.

        Args:
            engine_config: Engine configuration.
        """
        self._engine = EscalationEngine(engine_config)

        # Register policies from config
        for policy in self._config.policies.values():
            self._engine.register_policy(policy)

    def add_policy(self, policy: EscalationPolicy) -> None:
        """Add a policy.

        Args:
            policy: Policy to add.
        """
        self._config.add_policy(policy)
        if self._engine:
            self._engine.register_policy(policy)

    def remove_policy(self, policy_name: str) -> bool:
        """Remove a policy.

        Args:
            policy_name: Policy name.

        Returns:
            True if removed.
        """
        if policy_name in self._config.policies:
            del self._config.policies[policy_name]
            if self._engine:
                self._engine.unregister_policy(policy_name)
            return True
        return False

    def get_policy(self, policy_name: str) -> EscalationPolicy | None:
        """Get a policy.

        Args:
            policy_name: Policy name.

        Returns:
            Policy if found.
        """
        return self._config.policies.get(policy_name)

    def list_policies(self) -> list[str]:
        """List all policy names.

        Returns:
            List of policy names.
        """
        return list(self._config.policies.keys())

    async def start(self) -> None:
        """Start the manager."""
        if not self._engine:
            self.configure()

        if self._engine:
            await self._engine.start()
            self._is_running = True

    async def stop(self) -> None:
        """Stop the manager."""
        if self._engine:
            await self._engine.stop()
        self._is_running = False

    async def trigger(
        self,
        incident_id: str,
        context: dict[str, Any] | None = None,
        policy_name: str | None = None,
    ) -> EscalationResult:
        """Trigger an escalation.

        Args:
            incident_id: Incident ID.
            context: Optional context.
            policy_name: Optional policy name.

        Returns:
            EscalationResult.
        """
        if not self._engine:
            return EscalationResult.failure_result(
                None,
                "trigger",
                "Engine not configured",
            )

        return await self._engine.trigger(
            incident_id,
            context,
            policy_name or self._config.default_policy,
        )

    def get_stats(self) -> EscalationStats:
        """Get statistics.

        Returns:
            Current statistics.
        """
        if self._engine:
            return self._engine.get_stats()
        return EscalationStats()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EscalationPolicyManager:
        """Create manager from dictionary configuration.

        Args:
            data: Configuration dictionary.

        Returns:
            Configured manager.
        """
        config = EscalationPolicyConfig(
            default_policy=data.get("default_policy", "default"),
            global_enabled=data.get("global_enabled", True),
            store_type=data.get("store_type", "memory"),
            store_config=data.get("store_config", {}),
            max_concurrent_escalations=data.get("max_concurrent_escalations", 1000),
            cleanup_interval_minutes=data.get("cleanup_interval_minutes", 60),
            metrics_enabled=data.get("metrics_enabled", True),
        )

        # Load policies
        for policy_data in data.get("policies", []):
            policy = EscalationPolicy.from_dict(policy_data)
            config.add_policy(policy)

        return cls(config)
