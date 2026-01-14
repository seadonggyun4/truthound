"""Tests for escalation policy protocols and data types."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from truthound.checkpoint.escalation.protocols import (
    EscalationLevel,
    EscalationPolicy,
    EscalationPolicyConfig,
    EscalationRecord,
    EscalationResult,
    EscalationStats,
    EscalationTarget,
    EscalationTrigger,
    TargetType,
)


class TestEscalationTarget:
    """Tests for EscalationTarget."""

    def test_create_user_target(self) -> None:
        """Test creating a user target."""
        target = EscalationTarget.user("user-123", "John Doe")
        assert target.type == TargetType.USER
        assert target.identifier == "user-123"
        assert target.name == "John Doe"

    def test_create_team_target(self) -> None:
        """Test creating a team target."""
        target = EscalationTarget.team("team-123", "Engineering")
        assert target.type == TargetType.TEAM
        assert target.identifier == "team-123"

    def test_create_channel_target(self) -> None:
        """Test creating a channel target."""
        target = EscalationTarget.channel("#alerts", "Alerts Channel")
        assert target.type == TargetType.CHANNEL
        assert target.identifier == "#alerts"

    def test_create_email_target(self) -> None:
        """Test creating an email target."""
        target = EscalationTarget.email("alert@example.com")
        assert target.type == TargetType.EMAIL
        assert target.identifier == "alert@example.com"

    def test_create_webhook_target(self) -> None:
        """Test creating a webhook target."""
        target = EscalationTarget.webhook("https://example.com/webhook")
        assert target.type == TargetType.WEBHOOK
        assert target.identifier == "https://example.com/webhook"

    def test_target_to_dict(self) -> None:
        """Test target serialization."""
        target = EscalationTarget.user("user-123", "John Doe", priority=1)
        data = target.to_dict()
        assert data["type"] == "user"
        assert data["identifier"] == "user-123"
        assert data["priority"] == 1

    def test_target_from_dict(self) -> None:
        """Test target deserialization."""
        data = {
            "type": "team",
            "identifier": "team-123",
            "name": "Engineering",
            "priority": 2,
        }
        target = EscalationTarget.from_dict(data)
        assert target.type == TargetType.TEAM
        assert target.identifier == "team-123"
        assert target.priority == 2

    def test_target_with_metadata(self) -> None:
        """Test target with custom metadata."""
        target = EscalationTarget(
            type=TargetType.CUSTOM,
            identifier="custom-123",
            metadata={"action": "pagerduty", "urgency": "high"},
        )
        assert target.metadata["action"] == "pagerduty"
        assert target.metadata["urgency"] == "high"


class TestEscalationLevel:
    """Tests for EscalationLevel."""

    def test_create_level(self) -> None:
        """Test creating an escalation level."""
        targets = [EscalationTarget.user("user-1")]
        level = EscalationLevel(
            level=1,
            delay_minutes=15,
            targets=targets,
        )
        assert level.level == 1
        assert level.delay_minutes == 15
        assert len(level.targets) == 1

    def test_level_delay_property(self) -> None:
        """Test delay timedelta property."""
        level = EscalationLevel(level=1, delay_minutes=30)
        assert level.delay == timedelta(minutes=30)

    def test_level_repeat_interval(self) -> None:
        """Test repeat interval property."""
        level = EscalationLevel(
            level=1,
            repeat_count=3,
            repeat_interval_minutes=10,
        )
        assert level.repeat_interval == timedelta(minutes=10)

    def test_level_auto_resolve_timeout(self) -> None:
        """Test auto-resolve timeout property."""
        level = EscalationLevel(level=1, auto_resolve_minutes=60)
        assert level.auto_resolve_timeout == timedelta(minutes=60)

    def test_level_auto_resolve_disabled(self) -> None:
        """Test disabled auto-resolve returns None."""
        level = EscalationLevel(level=1, auto_resolve_minutes=0)
        assert level.auto_resolve_timeout is None

    def test_level_serialization(self) -> None:
        """Test level serialization round-trip."""
        level = EscalationLevel(
            level=2,
            delay_minutes=15,
            targets=[EscalationTarget.user("user-1")],
            repeat_count=2,
        )
        data = level.to_dict()
        restored = EscalationLevel.from_dict(data)
        assert restored.level == level.level
        assert restored.delay_minutes == level.delay_minutes
        assert len(restored.targets) == 1


class TestEscalationPolicy:
    """Tests for EscalationPolicy."""

    def test_create_policy(self) -> None:
        """Test creating an escalation policy."""
        policy = EscalationPolicy(
            name="critical_alerts",
            description="Handle critical alerts",
            levels=[
                EscalationLevel(level=1, delay_minutes=0),
                EscalationLevel(level=2, delay_minutes=15),
            ],
        )
        assert policy.name == "critical_alerts"
        assert len(policy.levels) == 2
        assert policy.enabled

    def test_policy_max_level(self) -> None:
        """Test max level calculation."""
        policy = EscalationPolicy(
            name="test",
            levels=[
                EscalationLevel(level=1),
                EscalationLevel(level=3),
                EscalationLevel(level=2),
            ],
        )
        assert policy.max_level == 3

    def test_policy_get_level(self) -> None:
        """Test getting level by number."""
        level1 = EscalationLevel(level=1, delay_minutes=0)
        level2 = EscalationLevel(level=2, delay_minutes=15)
        policy = EscalationPolicy(name="test", levels=[level1, level2])

        assert policy.get_level(1) == level1
        assert policy.get_level(2) == level2
        assert policy.get_level(3) is None

    def test_policy_get_next_level(self) -> None:
        """Test getting next level."""
        policy = EscalationPolicy(
            name="test",
            levels=[
                EscalationLevel(level=1),
                EscalationLevel(level=2),
                EscalationLevel(level=3),
            ],
        )

        next_level = policy.get_next_level(1)
        assert next_level is not None
        assert next_level.level == 2

        next_level = policy.get_next_level(3)
        assert next_level is None

    def test_policy_cooldown(self) -> None:
        """Test cooldown timedelta."""
        policy = EscalationPolicy(name="test", cooldown_minutes=30)
        assert policy.cooldown == timedelta(minutes=30)

    def test_policy_severity_filter(self) -> None:
        """Test severity filter configuration."""
        policy = EscalationPolicy(
            name="test",
            severity_filter=["critical", "high"],
        )
        assert "critical" in policy.severity_filter
        assert "high" in policy.severity_filter
        assert "low" not in policy.severity_filter

    def test_policy_triggers(self) -> None:
        """Test trigger configuration."""
        policy = EscalationPolicy(
            name="test",
            triggers=[
                EscalationTrigger.UNACKNOWLEDGED,
                EscalationTrigger.REPEATED_FAILURE,
            ],
        )
        assert EscalationTrigger.UNACKNOWLEDGED in policy.triggers
        assert EscalationTrigger.REPEATED_FAILURE in policy.triggers

    def test_policy_business_hours(self) -> None:
        """Test business hours configuration."""
        policy = EscalationPolicy(
            name="test",
            business_hours_only=True,
            business_hours_start=9,
            business_hours_end=18,
            business_days=[0, 1, 2, 3, 4],
            timezone="UTC",
        )
        assert policy.business_hours_only
        assert policy.business_hours_start == 9
        assert policy.business_hours_end == 18

    def test_policy_serialization(self) -> None:
        """Test policy serialization round-trip."""
        policy = EscalationPolicy(
            name="test",
            description="Test policy",
            levels=[EscalationLevel(level=1)],
            severity_filter=["critical"],
        )
        data = policy.to_dict()
        restored = EscalationPolicy.from_dict(data)
        assert restored.name == policy.name
        assert restored.description == policy.description
        assert len(restored.levels) == 1


class TestEscalationRecord:
    """Tests for EscalationRecord."""

    def test_create_record(self) -> None:
        """Test creating an escalation record."""
        record = EscalationRecord.create(
            incident_id="incident-123",
            policy_name="critical_alerts",
            context={"severity": "critical"},
        )
        assert record.incident_id == "incident-123"
        assert record.policy_name == "critical_alerts"
        assert record.current_level == 1
        assert record.state == "pending"

    def test_record_generate_id(self) -> None:
        """Test ID generation is unique."""
        id1 = EscalationRecord.generate_id("incident-1", "policy-1")
        id2 = EscalationRecord.generate_id("incident-1", "policy-1")
        # IDs should be different due to timestamp
        assert id1 != id2

    def test_record_is_active(self) -> None:
        """Test active state checking."""
        record = EscalationRecord.create("inc-1", "policy-1")
        record.state = "pending"
        assert record.is_active

        record.state = "active"
        assert record.is_active

        record.state = "resolved"
        assert not record.is_active

    def test_record_is_acknowledged(self) -> None:
        """Test acknowledgment checking."""
        record = EscalationRecord.create("inc-1", "policy-1")
        assert not record.is_acknowledged

        record.acknowledged_at = datetime.now()
        record.acknowledged_by = "user-123"
        assert record.is_acknowledged

    def test_record_is_resolved(self) -> None:
        """Test resolution checking."""
        record = EscalationRecord.create("inc-1", "policy-1")
        assert not record.is_resolved

        record.resolved_at = datetime.now()
        record.resolved_by = "user-123"
        assert record.is_resolved

    def test_record_duration(self) -> None:
        """Test duration calculation."""
        record = EscalationRecord.create("inc-1", "policy-1")
        record.created_at = datetime.now() - timedelta(hours=1)
        assert record.duration >= timedelta(minutes=59)

    def test_record_add_history_event(self) -> None:
        """Test adding history events."""
        record = EscalationRecord.create("inc-1", "policy-1")
        record.add_history_event("test_event", {"key": "value"})

        assert len(record.history) == 1
        assert record.history[0]["event_type"] == "test_event"
        assert record.history[0]["details"]["key"] == "value"

    def test_record_serialization(self) -> None:
        """Test record serialization round-trip."""
        record = EscalationRecord.create(
            incident_id="inc-1",
            policy_name="policy-1",
            context={"severity": "high"},
        )
        record.add_history_event("created", {})

        data = record.to_dict()
        restored = EscalationRecord.from_dict(data)

        assert restored.id == record.id
        assert restored.incident_id == record.incident_id
        assert restored.policy_name == record.policy_name
        assert len(restored.history) == 1


class TestEscalationResult:
    """Tests for EscalationResult."""

    def test_success_result(self) -> None:
        """Test creating success result."""
        record = EscalationRecord.create("inc-1", "policy-1")
        targets = [EscalationTarget.user("user-1")]

        result = EscalationResult.success_result(
            record=record,
            action="trigger",
            targets=targets,
            message="Escalation triggered",
        )

        assert result.success
        assert result.record == record
        assert result.action == "trigger"
        assert len(result.targets_notified) == 1

    def test_failure_result(self) -> None:
        """Test creating failure result."""
        result = EscalationResult.failure_result(
            record=None,
            action="trigger",
            error="Policy not found",
        )

        assert not result.success
        assert result.error == "Policy not found"


class TestEscalationStats:
    """Tests for EscalationStats."""

    def test_acknowledgment_rate(self) -> None:
        """Test acknowledgment rate calculation."""
        stats = EscalationStats(
            total_escalations=100,
            acknowledged_count=80,
        )
        assert stats.acknowledgment_rate == 80.0

    def test_resolution_rate(self) -> None:
        """Test resolution rate calculation."""
        stats = EscalationStats(
            total_escalations=100,
            resolved_count=70,
        )
        assert stats.resolution_rate == 70.0

    def test_zero_total_rates(self) -> None:
        """Test rates with zero total."""
        stats = EscalationStats(total_escalations=0)
        assert stats.acknowledgment_rate == 0.0
        assert stats.resolution_rate == 0.0

    def test_stats_serialization(self) -> None:
        """Test stats serialization."""
        stats = EscalationStats(
            total_escalations=100,
            active_escalations=10,
            acknowledged_count=50,
            resolved_count=40,
            escalations_by_level={1: 60, 2: 30, 3: 10},
        )
        data = stats.to_dict()
        assert data["total_escalations"] == 100
        assert data["acknowledgment_rate"] == 50.0


class TestEscalationPolicyConfig:
    """Tests for EscalationPolicyConfig."""

    def test_create_config(self) -> None:
        """Test creating configuration."""
        config = EscalationPolicyConfig(
            default_policy="default",
            global_enabled=True,
        )
        assert config.default_policy == "default"
        assert config.global_enabled

    def test_add_policy(self) -> None:
        """Test adding a policy."""
        config = EscalationPolicyConfig()
        policy = EscalationPolicy(name="test")
        config.add_policy(policy)

        assert "test" in config.policies
        assert config.get_policy("test") == policy

    def test_get_default_policy(self) -> None:
        """Test getting default policy."""
        config = EscalationPolicyConfig(default_policy="test")
        policy = EscalationPolicy(name="test")
        config.add_policy(policy)

        assert config.get_policy() == policy
        assert config.get_policy("test") == policy

    def test_config_serialization(self) -> None:
        """Test config serialization."""
        config = EscalationPolicyConfig(
            default_policy="test",
            store_type="memory",
            max_concurrent_escalations=500,
        )
        data = config.to_dict()
        assert data["default_policy"] == "test"
        assert data["store_type"] == "memory"
        assert data["max_concurrent_escalations"] == 500
