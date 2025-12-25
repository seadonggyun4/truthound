"""Rate limit policies and quota management.

This module provides policy-based rate limiting that allows different
limits based on user tiers, endpoints, or custom rules.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from truthound.ratelimit.core import (
    RateLimitConfig,
    RateLimitResult,
    QuotaConfig,
    QuotaExceeded,
    RateLimitScope,
    current_time,
)


# =============================================================================
# Policy Types
# =============================================================================


class PolicyPriority(Enum):
    """Priority for policy evaluation."""

    CRITICAL = 0  # Always applied first
    HIGH = 1
    NORMAL = 2
    LOW = 3


@dataclass
class PolicyMatch:
    """Result of policy matching."""

    matched: bool
    policy_name: str = ""
    config: RateLimitConfig | None = None
    priority: PolicyPriority = PolicyPriority.NORMAL
    metadata: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Policy Interface
# =============================================================================


class RateLimitPolicy(ABC):
    """Abstract base class for rate limiting policies.

    Policies determine which rate limit configuration to apply
    based on the request context.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Get policy name."""
        pass

    @property
    def priority(self) -> PolicyPriority:
        """Get policy priority."""
        return PolicyPriority.NORMAL

    @abstractmethod
    def matches(self, context: Any) -> PolicyMatch:
        """Check if policy matches the context.

        Args:
            context: Request context.

        Returns:
            PolicyMatch with result.
        """
        pass


# =============================================================================
# Built-in Policies
# =============================================================================


class DefaultPolicy(RateLimitPolicy):
    """Default policy that matches all requests."""

    def __init__(
        self,
        config: RateLimitConfig,
        name: str = "default",
    ) -> None:
        self._name = name
        self._config = config

    @property
    def name(self) -> str:
        return self._name

    @property
    def priority(self) -> PolicyPriority:
        return PolicyPriority.LOW

    def matches(self, context: Any) -> PolicyMatch:
        return PolicyMatch(
            matched=True,
            policy_name=self._name,
            config=self._config,
        )


class TierBasedPolicy(RateLimitPolicy):
    """Policy based on user/subscription tiers.

    Example:
        >>> policy = TierBasedPolicy(
        ...     tier_configs={
        ...         "free": RateLimitConfig(requests_per_second=1),
        ...         "pro": RateLimitConfig(requests_per_second=10),
        ...         "enterprise": RateLimitConfig(requests_per_second=100),
        ...     },
        ...     tier_extractor=lambda ctx: ctx.get("tier", "free"),
        ... )
    """

    def __init__(
        self,
        tier_configs: dict[str, RateLimitConfig],
        tier_extractor: Callable[[Any], str],
        default_tier: str = "free",
        name: str = "tier_based",
    ) -> None:
        self._name = name
        self._tier_configs = tier_configs
        self._tier_extractor = tier_extractor
        self._default_tier = default_tier

    @property
    def name(self) -> str:
        return self._name

    def matches(self, context: Any) -> PolicyMatch:
        tier = self._tier_extractor(context)
        config = self._tier_configs.get(tier)

        if config is None:
            config = self._tier_configs.get(self._default_tier)

        if config is None:
            return PolicyMatch(matched=False)

        return PolicyMatch(
            matched=True,
            policy_name=self._name,
            config=config,
            metadata={"tier": tier},
        )


class EndpointPolicy(RateLimitPolicy):
    """Policy based on endpoint/path.

    Example:
        >>> policy = EndpointPolicy(
        ...     endpoint_configs={
        ...         "/api/v1/search": RateLimitConfig(requests_per_second=5),
        ...         "/api/v1/export": RateLimitConfig(requests_per_second=1),
        ...     },
        ...     endpoint_extractor=lambda ctx: ctx.get("path"),
        ... )
    """

    def __init__(
        self,
        endpoint_configs: dict[str, RateLimitConfig],
        endpoint_extractor: Callable[[Any], str],
        default_config: RateLimitConfig | None = None,
        name: str = "endpoint",
    ) -> None:
        self._name = name
        self._endpoint_configs = endpoint_configs
        self._endpoint_extractor = endpoint_extractor
        self._default_config = default_config

    @property
    def name(self) -> str:
        return self._name

    def matches(self, context: Any) -> PolicyMatch:
        endpoint = self._endpoint_extractor(context)
        config = self._endpoint_configs.get(endpoint)

        if config is None:
            # Try prefix matching
            for pattern, cfg in self._endpoint_configs.items():
                if pattern.endswith("*") and endpoint.startswith(pattern[:-1]):
                    config = cfg
                    break

        if config is None:
            if self._default_config:
                config = self._default_config
            else:
                return PolicyMatch(matched=False)

        return PolicyMatch(
            matched=True,
            policy_name=self._name,
            config=config,
            metadata={"endpoint": endpoint},
        )


class IPBasedPolicy(RateLimitPolicy):
    """Policy based on IP address or CIDR ranges.

    Supports allowlists and blocklists.

    Example:
        >>> policy = IPBasedPolicy(
        ...     ip_extractor=lambda ctx: ctx.get("client_ip"),
        ...     allowlist=["10.0.0.0/8", "192.168.1.1"],
        ...     allowlist_config=RateLimitConfig(requests_per_second=1000),
        ...     default_config=RateLimitConfig(requests_per_second=10),
        ... )
    """

    def __init__(
        self,
        ip_extractor: Callable[[Any], str],
        default_config: RateLimitConfig,
        allowlist: list[str] | None = None,
        blocklist: list[str] | None = None,
        allowlist_config: RateLimitConfig | None = None,
        name: str = "ip_based",
    ) -> None:
        self._name = name
        self._ip_extractor = ip_extractor
        self._default_config = default_config
        self._allowlist = set(allowlist or [])
        self._blocklist = set(blocklist or [])
        self._allowlist_config = allowlist_config

    @property
    def name(self) -> str:
        return self._name

    def matches(self, context: Any) -> PolicyMatch:
        ip = self._ip_extractor(context)

        if ip in self._blocklist:
            # Return zero-limit config for blocked IPs
            return PolicyMatch(
                matched=True,
                policy_name=self._name,
                config=RateLimitConfig(requests_per_second=0.0001),  # Effectively blocked
                metadata={"ip": ip, "blocked": True},
            )

        if ip in self._allowlist and self._allowlist_config:
            return PolicyMatch(
                matched=True,
                policy_name=self._name,
                config=self._allowlist_config,
                metadata={"ip": ip, "allowlisted": True},
            )

        return PolicyMatch(
            matched=True,
            policy_name=self._name,
            config=self._default_config,
            metadata={"ip": ip},
        )


class CompositePolicy(RateLimitPolicy):
    """Combines multiple policies with priority ordering.

    Evaluates policies in priority order and returns first match.

    Example:
        >>> policy = CompositePolicy([
        ...     ip_policy,      # Check IP first
        ...     tier_policy,    # Then check tier
        ...     endpoint_policy,# Then endpoint
        ...     default_policy, # Fallback
        ... ])
    """

    def __init__(
        self,
        policies: list[RateLimitPolicy],
        name: str = "composite",
    ) -> None:
        self._name = name
        # Sort by priority
        self._policies = sorted(
            policies,
            key=lambda p: p.priority.value,
        )

    @property
    def name(self) -> str:
        return self._name

    def matches(self, context: Any) -> PolicyMatch:
        for policy in self._policies:
            match = policy.matches(context)
            if match.matched:
                return match

        return PolicyMatch(matched=False)


class ConditionalPolicy(RateLimitPolicy):
    """Policy with custom matching logic.

    Example:
        >>> policy = ConditionalPolicy(
        ...     name="high_value_users",
        ...     condition=lambda ctx: ctx.get("total_spend", 0) > 1000,
        ...     config=RateLimitConfig(requests_per_second=100),
        ...     priority=PolicyPriority.HIGH,
        ... )
    """

    def __init__(
        self,
        condition: Callable[[Any], bool],
        config: RateLimitConfig,
        name: str = "conditional",
        priority: PolicyPriority = PolicyPriority.NORMAL,
    ) -> None:
        self._name = name
        self._condition = condition
        self._config = config
        self._priority = priority

    @property
    def name(self) -> str:
        return self._name

    @property
    def priority(self) -> PolicyPriority:
        return self._priority

    def matches(self, context: Any) -> PolicyMatch:
        if self._condition(context):
            return PolicyMatch(
                matched=True,
                policy_name=self._name,
                config=self._config,
                priority=self._priority,
            )
        return PolicyMatch(matched=False)


# =============================================================================
# Quota Manager
# =============================================================================


@dataclass
class QuotaUsage:
    """Current quota usage information."""

    name: str
    used: int
    limit: int
    period_start: float
    period_end: float
    soft_limit: int
    hard_limit: int

    @property
    def remaining(self) -> int:
        """Get remaining quota."""
        return max(0, self.limit - self.used)

    @property
    def utilization(self) -> float:
        """Get utilization percentage."""
        if self.limit == 0:
            return 0.0
        return self.used / self.limit

    @property
    def is_soft_exceeded(self) -> bool:
        """Check if soft limit exceeded."""
        return self.used >= self.soft_limit

    @property
    def is_hard_exceeded(self) -> bool:
        """Check if hard limit exceeded."""
        return self.used >= self.hard_limit


class QuotaManager:
    """Manages quota tracking and enforcement.

    Quotas are longer-term limits (daily, monthly) often tied to
    billing or subscription tiers.

    Example:
        >>> manager = QuotaManager()
        >>> manager.register_quota(QuotaConfig(
        ...     name="api_calls",
        ...     limit=10000,
        ...     period_seconds=86400,
        ... ))
        >>> manager.consume("user:123", "api_calls", 1)
    """

    def __init__(self) -> None:
        self._quotas: dict[str, QuotaConfig] = {}
        self._usage: dict[str, dict[str, QuotaUsage]] = {}  # key -> quota_name -> usage
        self._lock = threading.Lock()

    def register_quota(self, config: QuotaConfig) -> None:
        """Register a quota configuration.

        Args:
            config: Quota configuration.
        """
        with self._lock:
            self._quotas[config.name] = config

    def get_quota(self, name: str) -> QuotaConfig | None:
        """Get quota configuration by name."""
        return self._quotas.get(name)

    def get_usage(self, key: str, quota_name: str) -> QuotaUsage | None:
        """Get current quota usage.

        Args:
            key: User/entity key.
            quota_name: Quota name.

        Returns:
            QuotaUsage or None if not found.
        """
        with self._lock:
            if key not in self._usage:
                return None
            return self._usage[key].get(quota_name)

    def consume(
        self,
        key: str,
        quota_name: str,
        amount: int = 1,
        *,
        raise_on_exceed: bool = True,
    ) -> QuotaUsage:
        """Consume quota.

        Args:
            key: User/entity key.
            quota_name: Quota name.
            amount: Amount to consume.
            raise_on_exceed: Raise exception if exceeded.

        Returns:
            Updated QuotaUsage.

        Raises:
            QuotaExceeded: If hard limit exceeded and raise_on_exceed=True.
        """
        with self._lock:
            config = self._quotas.get(quota_name)
            if config is None:
                raise ValueError(f"Unknown quota: {quota_name}")

            usage = self._get_or_create_usage(key, config)

            # Check if we need to reset the period
            now = current_time()
            if now >= usage.period_end:
                usage = self._reset_period(key, config)

            # Check hard limit
            if usage.used + amount > usage.hard_limit:
                if raise_on_exceed:
                    raise QuotaExceeded(
                        f"Quota '{quota_name}' exceeded for key '{key}'",
                        quota_name=quota_name,
                    )
                return usage

            # Consume
            new_usage = QuotaUsage(
                name=quota_name,
                used=usage.used + amount,
                limit=usage.limit,
                period_start=usage.period_start,
                period_end=usage.period_end,
                soft_limit=usage.soft_limit,
                hard_limit=usage.hard_limit,
            )

            if key not in self._usage:
                self._usage[key] = {}
            self._usage[key][quota_name] = new_usage

            return new_usage

    def reset(self, key: str, quota_name: str | None = None) -> None:
        """Reset quota usage.

        Args:
            key: User/entity key.
            quota_name: Specific quota to reset, or all if None.
        """
        with self._lock:
            if key not in self._usage:
                return

            if quota_name is None:
                del self._usage[key]
            elif quota_name in self._usage[key]:
                del self._usage[key][quota_name]

    def check(
        self,
        key: str,
        quota_name: str,
        amount: int = 1,
    ) -> RateLimitResult:
        """Check if quota allows the request.

        Args:
            key: User/entity key.
            quota_name: Quota name.
            amount: Amount to check.

        Returns:
            RateLimitResult.
        """
        with self._lock:
            config = self._quotas.get(quota_name)
            if config is None:
                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    limit=0,
                    reset_at=current_time(),
                    metadata={"error": f"Unknown quota: {quota_name}"},
                )

            usage = self._get_or_create_usage(key, config)

            # Check if we need to reset the period
            now = current_time()
            if now >= usage.period_end:
                usage = self._reset_period(key, config)

            allowed = usage.used + amount <= usage.hard_limit

            return RateLimitResult(
                allowed=allowed,
                remaining=usage.remaining,
                limit=usage.limit,
                reset_at=usage.period_end,
                retry_after=usage.period_end - now if not allowed else 0,
                bucket_key=key,
                metadata={
                    "quota_name": quota_name,
                    "soft_exceeded": usage.is_soft_exceeded,
                },
            )

    def _get_or_create_usage(
        self,
        key: str,
        config: QuotaConfig,
    ) -> QuotaUsage:
        """Get or create usage tracking."""
        if key in self._usage and config.name in self._usage[key]:
            return self._usage[key][config.name]

        return self._reset_period(key, config)

    def _reset_period(
        self,
        key: str,
        config: QuotaConfig,
    ) -> QuotaUsage:
        """Reset usage for a new period."""
        now = current_time()

        usage = QuotaUsage(
            name=config.name,
            used=0,
            limit=config.limit,
            period_start=now,
            period_end=now + config.period_seconds,
            soft_limit=config.soft_limit or int(config.limit * 0.8),
            hard_limit=config.hard_limit or config.limit,
        )

        if key not in self._usage:
            self._usage[key] = {}
        self._usage[key][config.name] = usage

        return usage


# =============================================================================
# Policy Registry
# =============================================================================


class PolicyRegistry:
    """Registry for rate limit policies.

    Provides centralized policy management and resolution.

    Example:
        >>> registry = PolicyRegistry()
        >>> registry.register(tier_policy)
        >>> registry.register(endpoint_policy)
        >>> policy = registry.resolve(context)
    """

    def __init__(self) -> None:
        self._policies: list[RateLimitPolicy] = []
        self._default_config = RateLimitConfig()
        self._lock = threading.Lock()

    def register(self, policy: RateLimitPolicy) -> None:
        """Register a policy.

        Args:
            policy: Policy to register.
        """
        with self._lock:
            self._policies.append(policy)
            # Keep sorted by priority
            self._policies.sort(key=lambda p: p.priority.value)

    def unregister(self, name: str) -> bool:
        """Unregister a policy by name.

        Args:
            name: Policy name.

        Returns:
            True if removed.
        """
        with self._lock:
            initial_len = len(self._policies)
            self._policies = [p for p in self._policies if p.name != name]
            return len(self._policies) < initial_len

    def set_default(self, config: RateLimitConfig) -> None:
        """Set default rate limit config.

        Args:
            config: Default configuration.
        """
        self._default_config = config

    def resolve(self, context: Any) -> PolicyMatch:
        """Resolve policy for context.

        Args:
            context: Request context.

        Returns:
            PolicyMatch with resolved configuration.
        """
        with self._lock:
            for policy in self._policies:
                match = policy.matches(context)
                if match.matched:
                    return match

        # Return default
        return PolicyMatch(
            matched=True,
            policy_name="default",
            config=self._default_config,
        )

    def get_all_policies(self) -> list[RateLimitPolicy]:
        """Get all registered policies."""
        with self._lock:
            return list(self._policies)


# =============================================================================
# Dynamic Policy Configuration
# =============================================================================


class DynamicPolicyConfig:
    """Dynamic policy configuration that can be updated at runtime.

    Useful for feature flags or A/B testing of rate limits.

    Example:
        >>> config = DynamicPolicyConfig()
        >>> config.set_override("user:123", RateLimitConfig(requests_per_second=100))
        >>> # Later, remove override
        >>> config.remove_override("user:123")
    """

    def __init__(
        self,
        default_config: RateLimitConfig,
        refresh_interval: float = 60.0,
    ) -> None:
        self._default = default_config
        self._overrides: dict[str, RateLimitConfig] = {}
        self._refresh_interval = refresh_interval
        self._last_refresh = current_time()
        self._lock = threading.Lock()
        self._refresh_callback: Callable[[], dict[str, RateLimitConfig]] | None = None

    def get_config(self, key: str) -> RateLimitConfig:
        """Get configuration for a key.

        Args:
            key: Lookup key.

        Returns:
            Rate limit configuration.
        """
        self._maybe_refresh()

        with self._lock:
            return self._overrides.get(key, self._default)

    def set_override(self, key: str, config: RateLimitConfig) -> None:
        """Set an override configuration.

        Args:
            key: Override key.
            config: Configuration to use.
        """
        with self._lock:
            self._overrides[key] = config

    def remove_override(self, key: str) -> bool:
        """Remove an override.

        Args:
            key: Override key.

        Returns:
            True if removed.
        """
        with self._lock:
            if key in self._overrides:
                del self._overrides[key]
                return True
            return False

    def set_refresh_callback(
        self,
        callback: Callable[[], dict[str, RateLimitConfig]],
    ) -> None:
        """Set callback for refreshing overrides from external source.

        Args:
            callback: Function that returns override dictionary.
        """
        self._refresh_callback = callback

    def _maybe_refresh(self) -> None:
        """Refresh overrides if needed."""
        if self._refresh_callback is None:
            return

        now = current_time()
        if now - self._last_refresh < self._refresh_interval:
            return

        try:
            new_overrides = self._refresh_callback()
            with self._lock:
                self._overrides = new_overrides
                self._last_refresh = now
        except Exception:
            pass  # Keep existing overrides on failure
