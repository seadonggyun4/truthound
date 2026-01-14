"""Tests for routing base module."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from truthound.checkpoint.routing.base import (
    ActionRouter,
    Route,
    RouteContext,
    RouteMode,
    RoutePriority,
    RoutingResult,
    RoutingRule,
)


# Test fixtures and helpers


@dataclass
class MockRule:
    """Simple mock rule for testing."""

    should_match: bool = True
    _description: str = "mock rule"

    def evaluate(self, context: RouteContext) -> bool:
        return self.should_match

    @property
    def description(self) -> str:
        return self._description


def create_mock_action(name: str = "test_action") -> MagicMock:
    """Create a mock action for testing."""
    action = MagicMock()
    action.name = name
    action.action_type = "mock"
    action.config.fail_checkpoint_on_error = False

    # Mock execute to return a successful result
    mock_result = MagicMock()
    mock_result.success = True
    mock_result.action_name = name
    action.execute.return_value = mock_result

    return action


def create_test_context(**kwargs) -> RouteContext:
    """Create a test RouteContext with defaults."""
    defaults = {
        "checkpoint_name": "test_checkpoint",
        "run_id": "run_123",
        "status": "failure",
        "data_asset": "test_data.csv",
        "run_time": datetime.now(),
        "total_issues": 5,
        "critical_issues": 1,
        "high_issues": 2,
        "medium_issues": 1,
        "low_issues": 1,
        "info_issues": 0,
        "pass_rate": 90.0,
        "tags": {"env": "prod"},
        "metadata": {"source": "test"},
    }
    defaults.update(kwargs)
    return RouteContext(**defaults)


# RouteContext tests


class TestRouteContext:
    """Tests for RouteContext dataclass."""

    def test_create_context(self):
        """Test basic context creation."""
        ctx = create_test_context()
        assert ctx.checkpoint_name == "test_checkpoint"
        assert ctx.status == "failure"
        assert ctx.total_issues == 5

    def test_context_to_dict(self):
        """Test context serialization."""
        ctx = create_test_context()
        d = ctx.to_dict()

        assert d["checkpoint_name"] == "test_checkpoint"
        assert d["status"] == "failure"
        assert d["total_issues"] == 5
        assert d["tags"] == {"env": "prod"}
        assert isinstance(d["run_time"], str)  # ISO format

    def test_from_checkpoint_result(self):
        """Test creating context from CheckpointResult."""
        # Create mock checkpoint result
        mock_result = MagicMock()
        mock_result.checkpoint_name = "test_cp"
        mock_result.run_id = "run_456"
        mock_result.status.value = "warning"
        mock_result.data_asset = "data.parquet"
        mock_result.run_time = datetime(2024, 1, 1, 12, 0)
        mock_result.duration_ms = 100.5
        mock_result.error = None
        mock_result.metadata = {"tags": {"env": "staging"}}

        # Mock validation result
        mock_result.validation_result.statistics.total_issues = 10
        mock_result.validation_result.statistics.critical_issues = 0
        mock_result.validation_result.statistics.high_issues = 3
        mock_result.validation_result.statistics.medium_issues = 5
        mock_result.validation_result.statistics.low_issues = 2
        mock_result.validation_result.statistics.pass_rate = 85.0

        ctx = RouteContext.from_checkpoint_result(mock_result)

        assert ctx.checkpoint_name == "test_cp"
        assert ctx.status == "warning"
        assert ctx.total_issues == 10
        assert ctx.high_issues == 3
        assert ctx.pass_rate == 85.0

    def test_from_checkpoint_result_no_validation(self):
        """Test context creation when validation_result is None."""
        mock_result = MagicMock()
        mock_result.checkpoint_name = "test_cp"
        mock_result.run_id = "run_789"
        mock_result.status.value = "error"
        mock_result.data_asset = "data.csv"
        mock_result.run_time = datetime.now()
        mock_result.duration_ms = 50.0
        mock_result.error = "Connection failed"
        mock_result.metadata = {}
        mock_result.validation_result = None

        ctx = RouteContext.from_checkpoint_result(mock_result)

        assert ctx.total_issues == 0
        assert ctx.pass_rate == 100.0
        assert ctx.error == "Connection failed"


# Route tests


class TestRoute:
    """Tests for Route dataclass."""

    def test_create_route(self):
        """Test basic route creation."""
        rule = MockRule(should_match=True)
        action = create_mock_action()

        route = Route(
            name="test_route",
            rule=rule,
            actions=[action],
            priority=RoutePriority.HIGH,
        )

        assert route.name == "test_route"
        assert route.priority == 80  # HIGH value
        assert route.enabled is True

    def test_route_matches(self):
        """Test route matching with rule."""
        ctx = create_test_context()
        rule = MockRule(should_match=True)
        route = Route(name="test", rule=rule, actions=[])

        assert route.matches(ctx) is True

    def test_route_not_matches(self):
        """Test route not matching."""
        ctx = create_test_context()
        rule = MockRule(should_match=False)
        route = Route(name="test", rule=rule, actions=[])

        assert route.matches(ctx) is False

    def test_route_disabled(self):
        """Test disabled route never matches."""
        ctx = create_test_context()
        rule = MockRule(should_match=True)
        route = Route(name="test", rule=rule, actions=[], enabled=False)

        assert route.matches(ctx) is False

    def test_route_exception_handling(self):
        """Test route handles rule exceptions gracefully."""

        class FailingRule:
            def evaluate(self, ctx):
                raise ValueError("Evaluation failed")

            @property
            def description(self):
                return "failing rule"

        ctx = create_test_context()
        route = Route(name="test", rule=FailingRule(), actions=[])

        # Should return False, not raise
        assert route.matches(ctx) is False


# ActionRouter tests


class TestActionRouter:
    """Tests for ActionRouter class."""

    def test_create_router(self):
        """Test basic router creation."""
        router = ActionRouter()

        assert router.mode == RouteMode.ALL_MATCHES
        assert len(router) == 0

    def test_create_router_with_mode(self):
        """Test router with specific mode."""
        router = ActionRouter(mode=RouteMode.FIRST_MATCH)

        assert router.mode == RouteMode.FIRST_MATCH

    def test_add_route(self):
        """Test adding routes."""
        router = ActionRouter()
        route1 = Route(
            name="route1",
            rule=MockRule(),
            actions=[],
            priority=RoutePriority.NORMAL,
        )
        route2 = Route(
            name="route2",
            rule=MockRule(),
            actions=[],
            priority=RoutePriority.HIGH,
        )

        router.add_route(route1)
        router.add_route(route2)

        assert len(router) == 2
        # Higher priority should be first
        assert router.routes[0].name == "route2"

    def test_remove_route(self):
        """Test removing routes."""
        router = ActionRouter()
        route = Route(name="test", rule=MockRule(), actions=[])
        router.add_route(route)

        assert router.remove_route("test") is True
        assert len(router) == 0
        assert router.remove_route("nonexistent") is False

    def test_get_route(self):
        """Test getting routes by name."""
        router = ActionRouter()
        route = Route(name="test", rule=MockRule(), actions=[])
        router.add_route(route)

        assert router.get_route("test") is route
        assert router.get_route("nonexistent") is None

    def test_evaluate_all_matches(self):
        """Test evaluation with ALL_MATCHES mode."""
        router = ActionRouter(mode=RouteMode.ALL_MATCHES)
        ctx = create_test_context()

        router.add_route(Route(name="r1", rule=MockRule(True), actions=[]))
        router.add_route(Route(name="r2", rule=MockRule(True), actions=[]))
        router.add_route(Route(name="r3", rule=MockRule(False), actions=[]))

        matched = router.evaluate(ctx)

        assert len(matched) == 2
        assert matched[0].name == "r1"
        assert matched[1].name == "r2"

    def test_evaluate_first_match(self):
        """Test evaluation with FIRST_MATCH mode."""
        router = ActionRouter(mode=RouteMode.FIRST_MATCH)
        ctx = create_test_context()

        router.add_route(
            Route(
                name="r1",
                rule=MockRule(True),
                actions=[],
                priority=RoutePriority.HIGH,
            )
        )
        router.add_route(
            Route(
                name="r2",
                rule=MockRule(True),
                actions=[],
                priority=RoutePriority.NORMAL,
            )
        )

        matched = router.evaluate(ctx)

        assert len(matched) == 1
        assert matched[0].name == "r1"

    def test_evaluate_priority_group(self):
        """Test evaluation with PRIORITY_GROUP mode."""
        router = ActionRouter(mode=RouteMode.PRIORITY_GROUP)
        ctx = create_test_context()

        router.add_route(
            Route(
                name="r1",
                rule=MockRule(True),
                actions=[],
                priority=RoutePriority.HIGH,
            )
        )
        router.add_route(
            Route(
                name="r2",
                rule=MockRule(True),
                actions=[],
                priority=RoutePriority.HIGH,
            )
        )
        router.add_route(
            Route(
                name="r3",
                rule=MockRule(True),
                actions=[],
                priority=RoutePriority.NORMAL,
            )
        )

        matched = router.evaluate(ctx)

        assert len(matched) == 2
        assert all(r.priority == RoutePriority.HIGH.value for r in matched)

    def test_evaluate_stop_on_match(self):
        """Test stop_on_match behavior."""
        router = ActionRouter(mode=RouteMode.ALL_MATCHES)
        ctx = create_test_context()

        router.add_route(
            Route(name="r1", rule=MockRule(True), actions=[], stop_on_match=True)
        )
        router.add_route(Route(name="r2", rule=MockRule(True), actions=[]))

        matched = router.evaluate(ctx)

        assert len(matched) == 1
        assert matched[0].name == "r1"

    def test_route_executes_actions(self):
        """Test that routing executes matched actions."""
        router = ActionRouter()
        ctx = create_test_context()

        action = create_mock_action("test_action")
        router.add_route(
            Route(name="test", rule=MockRule(True), actions=[action])
        )

        # Create mock checkpoint result
        mock_cp_result = MagicMock()
        mock_cp_result.checkpoint_name = ctx.checkpoint_name
        mock_cp_result.run_id = ctx.run_id
        mock_cp_result.status.value = ctx.status
        mock_cp_result.data_asset = ctx.data_asset
        mock_cp_result.run_time = ctx.run_time
        mock_cp_result.duration_ms = 100
        mock_cp_result.error = None
        mock_cp_result.metadata = ctx.metadata
        mock_cp_result.validation_result.statistics.total_issues = 5
        mock_cp_result.validation_result.statistics.critical_issues = 1
        mock_cp_result.validation_result.statistics.high_issues = 2
        mock_cp_result.validation_result.statistics.medium_issues = 1
        mock_cp_result.validation_result.statistics.low_issues = 1
        mock_cp_result.validation_result.statistics.pass_rate = 90.0

        result = router.route(mock_cp_result)

        assert result.has_matches
        assert len(result.executed_actions) == 1
        action.execute.assert_called_once()

    def test_route_default_actions(self):
        """Test default actions when no routes match."""
        default_action = create_mock_action("default")
        router = ActionRouter(default_actions=[default_action])
        ctx = create_test_context()

        router.add_route(Route(name="r1", rule=MockRule(False), actions=[]))

        mock_cp_result = MagicMock()
        mock_cp_result.checkpoint_name = ctx.checkpoint_name
        mock_cp_result.run_id = ctx.run_id
        mock_cp_result.status.value = ctx.status
        mock_cp_result.data_asset = ctx.data_asset
        mock_cp_result.run_time = ctx.run_time
        mock_cp_result.duration_ms = 100
        mock_cp_result.error = None
        mock_cp_result.metadata = ctx.metadata
        mock_cp_result.validation_result = None

        result = router.route(mock_cp_result)

        assert not result.has_matches
        assert len(result.executed_actions) == 1
        default_action.execute.assert_called_once()

    def test_route_no_execution(self):
        """Test routing without executing actions."""
        router = ActionRouter()
        action = create_mock_action()

        router.add_route(Route(name="test", rule=MockRule(True), actions=[action]))

        mock_cp_result = MagicMock()
        mock_cp_result.checkpoint_name = "test"
        mock_cp_result.run_id = "run_123"
        mock_cp_result.status.value = "failure"
        mock_cp_result.data_asset = "data.csv"
        mock_cp_result.run_time = datetime.now()
        mock_cp_result.duration_ms = 100
        mock_cp_result.error = None
        mock_cp_result.metadata = {}
        mock_cp_result.validation_result = None

        result = router.route(mock_cp_result, execute_actions=False)

        assert result.has_matches
        assert len(result.action_results) == 0
        action.execute.assert_not_called()

    def test_clear_routes(self):
        """Test clearing all routes."""
        router = ActionRouter()
        router.add_route(Route(name="r1", rule=MockRule(), actions=[]))
        router.add_route(Route(name="r2", rule=MockRule(), actions=[]))

        router.clear_routes()

        assert len(router) == 0


# RoutingResult tests


class TestRoutingResult:
    """Tests for RoutingResult dataclass."""

    def test_has_matches(self):
        """Test has_matches property."""
        result = RoutingResult()
        assert result.has_matches is False

        result.matched_routes.append(
            Route(name="test", rule=MockRule(), actions=[])
        )
        assert result.has_matches is True

    def test_all_successful(self):
        """Test all_successful property."""
        result = RoutingResult()
        assert result.all_successful is True  # Empty is successful

        mock_success = MagicMock()
        mock_success.success = True
        result.action_results.append(mock_success)
        assert result.all_successful is True

        mock_failure = MagicMock()
        mock_failure.success = False
        result.action_results.append(mock_failure)
        assert result.all_successful is False

    def test_to_dict(self):
        """Test serialization."""
        result = RoutingResult()
        result.matched_routes.append(
            Route(name="test", rule=MockRule(), actions=[])
        )
        result.evaluation_time_ms = 10.5

        d = result.to_dict()

        assert d["matched_routes"] == ["test"]
        assert d["evaluation_time_ms"] == 10.5


# Protocol compliance tests


class TestRoutingRuleProtocol:
    """Tests for RoutingRule protocol compliance."""

    def test_mock_rule_is_protocol_compliant(self):
        """Test that MockRule satisfies RoutingRule protocol."""
        rule = MockRule()
        assert isinstance(rule, RoutingRule)

    def test_custom_rule_protocol(self):
        """Test custom rule implementation."""

        class CustomRule:
            def evaluate(self, context: RouteContext) -> bool:
                return context.critical_issues > 0

            @property
            def description(self) -> str:
                return "Has critical issues"

        rule = CustomRule()
        ctx = create_test_context(critical_issues=1)

        assert rule.evaluate(ctx) is True
        assert rule.description == "Has critical issues"
