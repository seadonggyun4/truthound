"""Tests for escalation policy engine."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest

from truthound.checkpoint.escalation.engine import (
    EscalationEngine,
    EscalationEngineConfig,
    EscalationPolicyManager,
)
from truthound.checkpoint.escalation.protocols import (
    EscalationLevel,
    EscalationPolicy,
    EscalationRecord,
    EscalationTarget,
    EscalationTrigger,
)


class TestEscalationEngine:
    """Tests for EscalationEngine."""

    def test_create_engine(self) -> None:
        """Test creating an escalation engine."""
        engine = EscalationEngine()
        assert not engine.is_running
        assert engine.store is not None

    def test_register_policy(self) -> None:
        """Test registering a policy."""
        engine = EscalationEngine()
        policy = EscalationPolicy(
            name="test_policy",
            levels=[EscalationLevel(level=1)],
        )

        engine.register_policy(policy)
        assert engine.get_policy("test_policy") == policy

    def test_unregister_policy(self) -> None:
        """Test unregistering a policy."""
        engine = EscalationEngine()
        policy = EscalationPolicy(name="test")
        engine.register_policy(policy)

        assert engine.unregister_policy("test")
        assert engine.get_policy("test") is None
        assert not engine.unregister_policy("test")  # Already removed

    @pytest.mark.asyncio
    async def test_trigger_escalation(self) -> None:
        """Test triggering an escalation."""
        engine = EscalationEngine()

        policy = EscalationPolicy(
            name="test_policy",
            levels=[
                EscalationLevel(
                    level=1,
                    delay_minutes=0,
                    targets=[EscalationTarget.user("user-1")],
                ),
                EscalationLevel(
                    level=2,
                    delay_minutes=15,
                    targets=[EscalationTarget.user("user-2")],
                ),
            ],
            triggers=[EscalationTrigger.UNACKNOWLEDGED],
            severity_filter=["critical", "high", "medium", "low", "info"],
        )
        engine.register_policy(policy)

        # Set up a notification handler
        notifications_sent = []

        async def mock_handler(
            record: EscalationRecord,
            level: EscalationLevel,
            targets: list[EscalationTarget],
        ) -> bool:
            notifications_sent.append({
                "record_id": record.id,
                "level": level.level,
                "targets": [t.identifier for t in targets],
            })
            return True

        engine.set_notification_handler(mock_handler)

        await engine.start()

        result = await engine.trigger(
            incident_id="incident-123",
            context={"severity": "high"},
            policy_name="test_policy",
        )

        assert result.success
        assert result.record is not None
        assert result.record.incident_id == "incident-123"
        assert result.record.policy_name == "test_policy"
        assert result.record.current_level == 1
        assert len(notifications_sent) == 1

        await engine.stop()

    @pytest.mark.asyncio
    async def test_trigger_without_policy(self) -> None:
        """Test triggering with nonexistent policy."""
        engine = EscalationEngine()
        await engine.start()

        result = await engine.trigger(
            incident_id="incident-123",
            policy_name="nonexistent",
        )

        assert not result.success
        assert "No matching policy" in result.error

        await engine.stop()

    @pytest.mark.asyncio
    async def test_acknowledge_escalation(self) -> None:
        """Test acknowledging an escalation."""
        engine = EscalationEngine()
        policy = EscalationPolicy(
            name="test",
            levels=[EscalationLevel(level=1)],
        )
        engine.register_policy(policy)

        async def mock_handler(*args: Any, **kwargs: Any) -> bool:
            return True

        engine.set_notification_handler(mock_handler)
        await engine.start()

        # Trigger escalation
        trigger_result = await engine.trigger("inc-1", {}, "test")
        record_id = trigger_result.record.id

        # Acknowledge
        ack_result = await engine.acknowledge(record_id, "user-123")

        assert ack_result.success
        assert ack_result.record.state == "acknowledged"
        assert ack_result.record.acknowledged_by == "user-123"

        await engine.stop()

    @pytest.mark.asyncio
    async def test_acknowledge_nonexistent(self) -> None:
        """Test acknowledging nonexistent record."""
        engine = EscalationEngine()
        await engine.start()

        result = await engine.acknowledge("nonexistent", "user-123")

        assert not result.success
        assert "not found" in result.error

        await engine.stop()

    @pytest.mark.asyncio
    async def test_resolve_escalation(self) -> None:
        """Test resolving an escalation."""
        engine = EscalationEngine()
        policy = EscalationPolicy(
            name="test",
            levels=[EscalationLevel(level=1)],
        )
        engine.register_policy(policy)

        async def mock_handler(*args: Any, **kwargs: Any) -> bool:
            return True

        engine.set_notification_handler(mock_handler)
        await engine.start()

        trigger_result = await engine.trigger("inc-1", {}, "test")
        record_id = trigger_result.record.id

        # First acknowledge
        await engine.acknowledge(record_id, "user-123")

        # Then resolve
        resolve_result = await engine.resolve(record_id, "user-123")

        assert resolve_result.success
        assert resolve_result.record.state == "resolved"
        assert resolve_result.record.resolved_by == "user-123"

        await engine.stop()

    @pytest.mark.asyncio
    async def test_cancel_escalation(self) -> None:
        """Test cancelling an escalation."""
        engine = EscalationEngine()
        policy = EscalationPolicy(
            name="test",
            levels=[EscalationLevel(level=1)],
        )
        engine.register_policy(policy)

        async def mock_handler(*args: Any, **kwargs: Any) -> bool:
            return True

        engine.set_notification_handler(mock_handler)
        await engine.start()

        trigger_result = await engine.trigger("inc-1", {}, "test")
        record_id = trigger_result.record.id

        cancel_result = await engine.cancel(
            record_id, "admin", "False alarm"
        )

        assert cancel_result.success
        assert cancel_result.record.state == "cancelled"

        await engine.stop()

    @pytest.mark.asyncio
    async def test_get_active_escalations(self) -> None:
        """Test getting active escalations."""
        engine = EscalationEngine()
        policy1 = EscalationPolicy(name="policy-1", levels=[EscalationLevel(level=1)])
        policy2 = EscalationPolicy(name="policy-2", levels=[EscalationLevel(level=1)])
        engine.register_policy(policy1)
        engine.register_policy(policy2)

        async def mock_handler(*args: Any, **kwargs: Any) -> bool:
            return True

        engine.set_notification_handler(mock_handler)
        await engine.start()

        await engine.trigger("inc-1", {}, "policy-1")
        await engine.trigger("inc-2", {}, "policy-1")
        await engine.trigger("inc-3", {}, "policy-2")

        all_active = engine.get_active_escalations()
        assert len(all_active) == 3

        policy1_active = engine.get_active_escalations("policy-1")
        assert len(policy1_active) == 2

        await engine.stop()

    @pytest.mark.asyncio
    async def test_get_records_for_incident(self) -> None:
        """Test getting records for an incident."""
        engine = EscalationEngine()
        policy = EscalationPolicy(name="test", levels=[EscalationLevel(level=1)])
        engine.register_policy(policy)

        async def mock_handler(*args: Any, **kwargs: Any) -> bool:
            return True

        engine.set_notification_handler(mock_handler)
        await engine.start()

        await engine.trigger("inc-1", {}, "test")

        records = engine.get_records_for_incident("inc-1")
        assert len(records) == 1
        assert records[0].incident_id == "inc-1"

        await engine.stop()

    @pytest.mark.asyncio
    async def test_existing_escalation_returns_existing(self) -> None:
        """Test that triggering existing escalation returns the existing one."""
        engine = EscalationEngine()
        policy = EscalationPolicy(name="test", levels=[EscalationLevel(level=1)])
        engine.register_policy(policy)

        async def mock_handler(*args: Any, **kwargs: Any) -> bool:
            return True

        engine.set_notification_handler(mock_handler)
        await engine.start()

        result1 = await engine.trigger("inc-1", {}, "test")
        result2 = await engine.trigger("inc-1", {}, "test")

        assert result1.success
        assert result2.success
        assert result1.record.id == result2.record.id
        assert result2.action == "trigger_existing"

        await engine.stop()

    def test_get_stats(self) -> None:
        """Test getting engine statistics."""
        engine = EscalationEngine()
        stats = engine.get_stats()

        assert stats.total_escalations == 0
        assert stats.active_escalations == 0

    @pytest.mark.asyncio
    async def test_condition_evaluator(self) -> None:
        """Test custom condition evaluator."""
        engine = EscalationEngine()

        custom_conditions_evaluated = []

        async def check_severity(
            record: EscalationRecord,
            config: dict[str, Any],
        ) -> bool:
            custom_conditions_evaluated.append(config)
            min_severity = config.get("min_severity", "info")
            record_severity = record.context.get("severity", "info")
            severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
            return severity_order.get(record_severity, 4) <= severity_order.get(min_severity, 4)

        engine.register_condition_evaluator("severity_check", check_severity)

        policy = EscalationPolicy(
            name="test",
            levels=[
                EscalationLevel(level=1),
                EscalationLevel(
                    level=2,
                    delay_minutes=0,
                    conditions={"severity_check": {"min_severity": "high"}},
                ),
            ],
        )
        engine.register_policy(policy)

        async def mock_handler(*args: Any, **kwargs: Any) -> bool:
            return True

        engine.set_notification_handler(mock_handler)
        await engine.start()

        await engine.trigger("inc-1", {"severity": "critical"}, "test")

        await engine.stop()


class TestEscalationPolicyManager:
    """Tests for EscalationPolicyManager."""

    def test_create_manager(self) -> None:
        """Test creating a policy manager."""
        manager = EscalationPolicyManager()
        assert not manager.is_running
        assert manager.engine is None

    def test_add_policy(self) -> None:
        """Test adding a policy."""
        manager = EscalationPolicyManager()
        policy = EscalationPolicy(name="test")

        manager.add_policy(policy)
        assert manager.get_policy("test") == policy

    def test_remove_policy(self) -> None:
        """Test removing a policy."""
        manager = EscalationPolicyManager()
        policy = EscalationPolicy(name="test")
        manager.add_policy(policy)

        assert manager.remove_policy("test")
        assert manager.get_policy("test") is None

    def test_list_policies(self) -> None:
        """Test listing policy names."""
        manager = EscalationPolicyManager()
        manager.add_policy(EscalationPolicy(name="policy-1"))
        manager.add_policy(EscalationPolicy(name="policy-2"))

        policies = manager.list_policies()
        assert "policy-1" in policies
        assert "policy-2" in policies

    @pytest.mark.asyncio
    async def test_start_stop(self) -> None:
        """Test starting and stopping manager."""
        manager = EscalationPolicyManager()

        await manager.start()
        assert manager.is_running
        assert manager.engine is not None

        await manager.stop()
        assert not manager.is_running

    @pytest.mark.asyncio
    async def test_trigger_via_manager(self) -> None:
        """Test triggering escalation via manager."""
        manager = EscalationPolicyManager()
        manager.add_policy(
            EscalationPolicy(
                name="default",
                levels=[EscalationLevel(level=1)],
            )
        )

        await manager.start()

        # Set notification handler on engine
        async def mock_handler(*args: Any, **kwargs: Any) -> bool:
            return True

        manager.engine.set_notification_handler(mock_handler)

        result = await manager.trigger("inc-1", {})

        assert result.success
        assert result.record.policy_name == "default"

        await manager.stop()

    def test_from_dict(self) -> None:
        """Test creating manager from dictionary."""
        config = {
            "default_policy": "main",
            "global_enabled": True,
            "store_type": "memory",
            "policies": [
                {
                    "name": "main",
                    "levels": [{"level": 1, "delay_minutes": 0}],
                },
                {
                    "name": "secondary",
                    "levels": [{"level": 1, "delay_minutes": 5}],
                },
            ],
        }

        manager = EscalationPolicyManager.from_dict(config)

        assert manager.get_policy("main") is not None
        assert manager.get_policy("secondary") is not None
        assert len(manager.list_policies()) == 2

    def test_get_stats_before_start(self) -> None:
        """Test getting stats before engine started."""
        manager = EscalationPolicyManager()
        stats = manager.get_stats()

        assert stats.total_escalations == 0

    @pytest.mark.asyncio
    async def test_trigger_without_engine(self) -> None:
        """Test triggering without configured engine."""
        manager = EscalationPolicyManager()
        # Don't start - engine not configured

        result = await manager.trigger("inc-1", {})

        assert not result.success
        assert "not configured" in result.error


class TestEscalationEngineConfig:
    """Tests for EscalationEngineConfig."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = EscalationEngineConfig()

        assert config.store_type == "memory"
        assert config.check_business_hours
        assert config.metrics_enabled

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = EscalationEngineConfig(
            store_type="sqlite",
            store_config={"database": ":memory:"},
            check_business_hours=False,
        )

        assert config.store_type == "sqlite"
        assert not config.check_business_hours

    def test_create_engine_with_config(self) -> None:
        """Test creating engine with custom config."""
        config = EscalationEngineConfig(
            store_type="sqlite",
            store_config={"database": ":memory:"},
        )
        engine = EscalationEngine(config)

        # Engine should use SQLite store
        from truthound.checkpoint.escalation.stores import SQLiteEscalationStore
        assert isinstance(engine.store, SQLiteEscalationStore)
