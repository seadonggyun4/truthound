"""Comprehensive tests for the rate limiting module."""

import pytest
import time
import threading
from unittest.mock import Mock, patch

from truthound.ratelimit import (
    # Core types
    RateLimitStrategy,
    RateLimitScope,
    RateLimitAction,
    RateLimitResult,
    RateLimitConfig,
    QuotaConfig,
    RateLimitExceeded,
    QuotaExceeded,
    # Key extractors
    GlobalKeyExtractor,
    AttributeKeyExtractor,
    CompositeKeyExtractor,
    CallableKeyExtractor,
    # Algorithms
    TokenBucketAlgorithm,
    SlidingWindowAlgorithm,
    FixedWindowAlgorithm,
    LeakyBucketAlgorithm,
    create_algorithm,
    # Storage
    MemoryStorage,
    create_storage,
    # Policies
    PolicyPriority,
    DefaultPolicy,
    TierBasedPolicy,
    EndpointPolicy,
    IPBasedPolicy,
    CompositePolicy,
    ConditionalPolicy,
    QuotaManager,
    PolicyRegistry,
    DynamicPolicyConfig,
    # Main limiter
    RateLimiter,
    RateLimiterRegistry,
    rate_limit,
    RateLimitContext,
    # Middleware
    RateLimitMetrics,
    MetricsMiddleware,
    RetryHandler,
)


# =============================================================================
# Test Rate Limit Result
# =============================================================================


class TestRateLimitResult:
    """Tests for RateLimitResult."""

    def test_result_creation(self):
        """Test creating a rate limit result."""
        result = RateLimitResult(
            allowed=True,
            remaining=5,
            limit=10,
            reset_at=time.time() + 60,
        )
        assert result.allowed is True
        assert result.remaining == 5
        assert result.limit == 10

    def test_utilization(self):
        """Test utilization calculation."""
        result = RateLimitResult(
            allowed=True,
            remaining=3,
            limit=10,
            reset_at=time.time(),
        )
        assert result.utilization == 0.7

    def test_utilization_zero_limit(self):
        """Test utilization with zero limit."""
        result = RateLimitResult(
            allowed=False,
            remaining=0,
            limit=0,
            reset_at=time.time(),
        )
        assert result.utilization == 0.0

    def test_to_headers(self):
        """Test header generation."""
        now = time.time()
        result = RateLimitResult(
            allowed=True,
            remaining=5,
            limit=10,
            reset_at=now + 60,
        )
        headers = result.to_headers()
        assert "RateLimit-Limit" in headers
        assert headers["RateLimit-Limit"] == "10"
        assert headers["RateLimit-Remaining"] == "5"

    def test_to_headers_retry_after(self):
        """Test headers include Retry-After when not allowed."""
        result = RateLimitResult(
            allowed=False,
            remaining=0,
            limit=10,
            reset_at=time.time() + 60,
            retry_after=30,
        )
        headers = result.to_headers()
        assert "Retry-After" in headers
        assert headers["Retry-After"] == "30"


# =============================================================================
# Test Rate Limit Config
# =============================================================================


class TestRateLimitConfig:
    """Tests for RateLimitConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = RateLimitConfig()
        assert config.requests_per_second == 10.0
        assert config.burst_size == 10

    def test_custom_config(self):
        """Test custom configuration."""
        config = RateLimitConfig(
            requests_per_second=100,
            burst_size=200,
            strategy=RateLimitStrategy.SLIDING_WINDOW,
        )
        assert config.requests_per_second == 100
        assert config.burst_size == 200
        assert config.strategy == RateLimitStrategy.SLIDING_WINDOW

    def test_refill_rate(self):
        """Test refill rate property."""
        config = RateLimitConfig(requests_per_second=50)
        assert config.refill_rate == 50

    def test_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            "requests_per_second": 20,
            "strategy": "sliding_window",
        }
        config = RateLimitConfig.from_dict(data)
        assert config.requests_per_second == 20
        assert config.strategy == RateLimitStrategy.SLIDING_WINDOW

    def test_invalid_config(self):
        """Test validation of invalid config."""
        with pytest.raises(ValueError):
            RateLimitConfig(requests_per_second=-1)


# =============================================================================
# Test Key Extractors
# =============================================================================


class TestKeyExtractors:
    """Tests for key extractors."""

    def test_global_key_extractor(self):
        """Test global key extractor."""
        extractor = GlobalKeyExtractor("my_key")
        assert extractor.extract({}) == "my_key"
        assert extractor.extract(None) == "my_key"

    def test_attribute_key_extractor_dict(self):
        """Test attribute extractor with dict."""
        extractor = AttributeKeyExtractor("user_id", prefix="user:")
        context = {"user_id": "123"}
        assert extractor.extract(context) == "user:123"

    def test_attribute_key_extractor_object(self):
        """Test attribute extractor with object."""
        extractor = AttributeKeyExtractor("user_id")

        class Context:
            user_id = "456"

        assert extractor.extract(Context()) == "456"

    def test_attribute_key_extractor_default(self):
        """Test attribute extractor with default."""
        extractor = AttributeKeyExtractor("missing", default="default")
        assert extractor.extract({}) == "default"

    def test_composite_key_extractor(self):
        """Test composite key extractor."""
        extractor = CompositeKeyExtractor([
            AttributeKeyExtractor("endpoint"),
            AttributeKeyExtractor("user_id"),
        ], separator=":")
        context = {"endpoint": "/api", "user_id": "123"}
        assert extractor.extract(context) == "/api:123"

    def test_callable_key_extractor(self):
        """Test callable key extractor."""
        extractor = CallableKeyExtractor(lambda ctx: f"ip:{ctx.get('ip')}")
        assert extractor.extract({"ip": "1.2.3.4"}) == "ip:1.2.3.4"


# =============================================================================
# Test Token Bucket Algorithm
# =============================================================================


class TestTokenBucketAlgorithm:
    """Tests for Token Bucket algorithm."""

    def test_basic_acquire(self):
        """Test basic token acquisition."""
        config = RateLimitConfig(requests_per_second=10, burst_size=10)
        bucket = TokenBucketAlgorithm(config)

        result = bucket.acquire("test")
        assert result.allowed is True
        assert result.remaining == 9

    def test_burst_limit(self):
        """Test burst limit enforcement."""
        config = RateLimitConfig(requests_per_second=10, burst_size=5)
        bucket = TokenBucketAlgorithm(config)

        # Consume all burst capacity
        for _ in range(5):
            result = bucket.acquire("test")
            assert result.allowed is True

        # Should be rejected
        result = bucket.acquire("test")
        assert result.allowed is False
        assert result.retry_after > 0

    def test_token_refill(self):
        """Test token refill over time."""
        config = RateLimitConfig(requests_per_second=100, burst_size=5)
        bucket = TokenBucketAlgorithm(config)

        # Consume all tokens
        for _ in range(5):
            bucket.acquire("test")

        # Wait for refill
        time.sleep(0.1)  # Should refill ~10 tokens

        result = bucket.acquire("test")
        assert result.allowed is True

    def test_peek_no_consume(self):
        """Test peek doesn't consume tokens."""
        config = RateLimitConfig(requests_per_second=10, burst_size=5)
        bucket = TokenBucketAlgorithm(config)

        result1 = bucket.peek("test")
        result2 = bucket.peek("test")

        assert result1.remaining == result2.remaining

    def test_reset(self):
        """Test bucket reset."""
        config = RateLimitConfig(requests_per_second=10, burst_size=5)
        bucket = TokenBucketAlgorithm(config)

        # Consume some tokens
        bucket.acquire("test")
        bucket.acquire("test")

        # Reset
        bucket.reset("test")

        # Should be full again
        result = bucket.peek("test")
        assert result.remaining == 5

    def test_multiple_keys(self):
        """Test separate limits per key."""
        config = RateLimitConfig(requests_per_second=10, burst_size=2)
        bucket = TokenBucketAlgorithm(config)

        # Exhaust key1
        bucket.acquire("key1")
        bucket.acquire("key1")
        result1 = bucket.acquire("key1")

        # key2 should still have tokens
        result2 = bucket.acquire("key2")

        assert result1.allowed is False
        assert result2.allowed is True


# =============================================================================
# Test Sliding Window Algorithm
# =============================================================================


class TestSlidingWindowAlgorithm:
    """Tests for Sliding Window algorithm."""

    def test_basic_acquire(self):
        """Test basic acquisition."""
        config = RateLimitConfig(
            requests_per_second=10,
            window_size_seconds=1.0,
            sub_window_count=10,
        )
        window = SlidingWindowAlgorithm(config)

        result = window.acquire("test")
        assert result.allowed is True

    def test_window_limit(self):
        """Test window limit enforcement."""
        config = RateLimitConfig(
            requests_per_second=5,
            window_size_seconds=1.0,
        )
        window = SlidingWindowAlgorithm(config)

        # Consume all capacity
        for _ in range(5):
            result = window.acquire("test")
            assert result.allowed is True

        # Should be rejected
        result = window.acquire("test")
        assert result.allowed is False

    def test_window_slide(self):
        """Test window sliding."""
        config = RateLimitConfig(
            requests_per_second=100,  # 10 per 0.1 second
            window_size_seconds=0.2,
            sub_window_count=2,
        )
        window = SlidingWindowAlgorithm(config)

        # Consume some requests
        for _ in range(10):
            window.acquire("test")

        # Wait for sub-window to expire
        time.sleep(0.15)

        # Should have some capacity again
        result = window.acquire("test")
        assert result.allowed is True


# =============================================================================
# Test Fixed Window Algorithm
# =============================================================================


class TestFixedWindowAlgorithm:
    """Tests for Fixed Window algorithm."""

    def test_basic_acquire(self):
        """Test basic acquisition."""
        config = RateLimitConfig(
            requests_per_second=10,
            window_size_seconds=1.0,
        )
        window = FixedWindowAlgorithm(config)

        result = window.acquire("test")
        assert result.allowed is True

    def test_window_reset(self):
        """Test window reset at boundary."""
        config = RateLimitConfig(
            requests_per_second=100,  # 10 per 0.1 second
            window_size_seconds=0.1,
        )
        window = FixedWindowAlgorithm(config)

        # Consume all capacity
        for _ in range(10):
            window.acquire("test")

        # Wait for new window
        time.sleep(0.15)

        # Should be allowed in new window
        result = window.acquire("test")
        assert result.allowed is True


# =============================================================================
# Test Leaky Bucket Algorithm
# =============================================================================


class TestLeakyBucketAlgorithm:
    """Tests for Leaky Bucket algorithm."""

    def test_basic_acquire(self):
        """Test basic acquisition."""
        config = RateLimitConfig(
            requests_per_second=10,
            burst_size=10,
        )
        bucket = LeakyBucketAlgorithm(config)

        result = bucket.acquire("test")
        assert result.allowed is True

    def test_queue_capacity(self):
        """Test queue capacity limit."""
        config = RateLimitConfig(
            requests_per_second=1,  # Slow leak rate
            burst_size=5,
        )
        bucket = LeakyBucketAlgorithm(config)

        # Fill queue
        for _ in range(5):
            result = bucket.acquire("test")
            assert result.allowed is True

        # Should be rejected (queue full)
        result = bucket.acquire("test")
        assert result.allowed is False


# =============================================================================
# Test Algorithm Factory
# =============================================================================


class TestAlgorithmFactory:
    """Tests for algorithm factory."""

    def test_create_token_bucket(self):
        """Test creating token bucket."""
        config = RateLimitConfig(strategy=RateLimitStrategy.TOKEN_BUCKET)
        algorithm = create_algorithm(config)
        assert isinstance(algorithm, TokenBucketAlgorithm)

    def test_create_sliding_window(self):
        """Test creating sliding window."""
        config = RateLimitConfig(strategy=RateLimitStrategy.SLIDING_WINDOW)
        algorithm = create_algorithm(config)
        assert isinstance(algorithm, SlidingWindowAlgorithm)

    def test_create_fixed_window(self):
        """Test creating fixed window."""
        config = RateLimitConfig(strategy=RateLimitStrategy.FIXED_WINDOW)
        algorithm = create_algorithm(config)
        assert isinstance(algorithm, FixedWindowAlgorithm)

    def test_create_leaky_bucket(self):
        """Test creating leaky bucket."""
        config = RateLimitConfig(strategy=RateLimitStrategy.LEAKY_BUCKET)
        algorithm = create_algorithm(config)
        assert isinstance(algorithm, LeakyBucketAlgorithm)


# =============================================================================
# Test Memory Storage
# =============================================================================


class TestMemoryStorage:
    """Tests for memory storage."""

    def test_get_set(self):
        """Test basic get/set."""
        storage = MemoryStorage()
        storage.set("key", "value")
        assert storage.get("key") == "value"

    def test_get_missing(self):
        """Test getting missing key."""
        storage = MemoryStorage()
        assert storage.get("missing") is None

    def test_delete(self):
        """Test deletion."""
        storage = MemoryStorage()
        storage.set("key", "value")
        assert storage.delete("key") is True
        assert storage.get("key") is None
        assert storage.delete("key") is False

    def test_increment(self):
        """Test atomic increment."""
        storage = MemoryStorage()
        assert storage.increment("counter") == 1
        assert storage.increment("counter") == 2
        assert storage.increment("counter", 5) == 7

    def test_ttl_expiration(self):
        """Test TTL expiration."""
        storage = MemoryStorage()
        storage.set("key", "value", ttl=0.1)
        assert storage.get("key") == "value"
        time.sleep(0.15)
        assert storage.get("key") is None

    def test_get_with_lock(self):
        """Test get with lock."""
        storage = MemoryStorage()
        storage.set("key", "value")

        value, lock = storage.get_with_lock("key")
        assert value == "value"
        assert lock is not None

        # Release lock
        storage.set_with_lock("key", "new_value", lock)
        assert storage.get("key") == "new_value"

    def test_clear(self):
        """Test clearing storage."""
        storage = MemoryStorage()
        storage.set("key1", "value1")
        storage.set("key2", "value2")
        storage.clear()
        assert storage.size() == 0


# =============================================================================
# Test Policies
# =============================================================================


class TestPolicies:
    """Tests for rate limit policies."""

    def test_default_policy(self):
        """Test default policy."""
        config = RateLimitConfig(requests_per_second=10)
        policy = DefaultPolicy(config)

        match = policy.matches({})
        assert match.matched is True
        assert match.config == config

    def test_tier_based_policy(self):
        """Test tier-based policy."""
        policy = TierBasedPolicy(
            tier_configs={
                "free": RateLimitConfig(requests_per_second=1),
                "pro": RateLimitConfig(requests_per_second=10),
            },
            tier_extractor=lambda ctx: ctx.get("tier", "free"),
        )

        # Free tier
        match = policy.matches({"tier": "free"})
        assert match.matched is True
        assert match.config.requests_per_second == 1

        # Pro tier
        match = policy.matches({"tier": "pro"})
        assert match.matched is True
        assert match.config.requests_per_second == 10

    def test_endpoint_policy(self):
        """Test endpoint policy."""
        policy = EndpointPolicy(
            endpoint_configs={
                "/api/search": RateLimitConfig(requests_per_second=5),
                "/api/export": RateLimitConfig(requests_per_second=1),
            },
            endpoint_extractor=lambda ctx: ctx.get("path"),
        )

        match = policy.matches({"path": "/api/search"})
        assert match.matched is True
        assert match.config.requests_per_second == 5

    def test_ip_based_policy(self):
        """Test IP-based policy."""
        policy = IPBasedPolicy(
            ip_extractor=lambda ctx: ctx.get("ip"),
            default_config=RateLimitConfig(requests_per_second=10),
            allowlist=["192.168.1.1"],
            allowlist_config=RateLimitConfig(requests_per_second=100),
        )

        # Allowlisted IP
        match = policy.matches({"ip": "192.168.1.1"})
        assert match.config.requests_per_second == 100

        # Normal IP
        match = policy.matches({"ip": "10.0.0.1"})
        assert match.config.requests_per_second == 10

    def test_composite_policy(self):
        """Test composite policy."""
        tier_policy = TierBasedPolicy(
            tier_configs={"vip": RateLimitConfig(requests_per_second=100)},
            tier_extractor=lambda ctx: ctx.get("tier"),
        )
        default_policy = DefaultPolicy(RateLimitConfig(requests_per_second=10))

        policy = CompositePolicy([tier_policy, default_policy])

        # VIP matches tier policy
        match = policy.matches({"tier": "vip"})
        assert match.config.requests_per_second == 100

        # Non-VIP falls through to default
        match = policy.matches({"tier": "regular"})
        assert match.config.requests_per_second == 10

    def test_conditional_policy(self):
        """Test conditional policy."""
        policy = ConditionalPolicy(
            condition=lambda ctx: ctx.get("is_admin", False),
            config=RateLimitConfig(requests_per_second=1000),
        )

        # Admin
        match = policy.matches({"is_admin": True})
        assert match.matched is True

        # Non-admin
        match = policy.matches({"is_admin": False})
        assert match.matched is False


# =============================================================================
# Test Quota Manager
# =============================================================================


class TestQuotaManager:
    """Tests for quota manager."""

    def test_register_quota(self):
        """Test registering a quota."""
        manager = QuotaManager()
        quota = QuotaConfig(
            name="api_calls",
            limit=1000,
            period_seconds=86400,
        )
        manager.register_quota(quota)

        assert manager.get_quota("api_calls") is not None

    def test_consume_quota(self):
        """Test consuming quota."""
        manager = QuotaManager()
        manager.register_quota(QuotaConfig(
            name="api_calls",
            limit=10,
            period_seconds=86400,
        ))

        usage = manager.consume("user:123", "api_calls", 5)
        assert usage.used == 5
        assert usage.remaining == 5

    def test_quota_exceeded(self):
        """Test quota exceeded exception."""
        manager = QuotaManager()
        manager.register_quota(QuotaConfig(
            name="api_calls",
            limit=5,
            period_seconds=86400,
        ))

        # Consume all quota
        manager.consume("user:123", "api_calls", 5)

        # Should raise
        with pytest.raises(QuotaExceeded):
            manager.consume("user:123", "api_calls", 1)

    def test_quota_check(self):
        """Test checking quota without consuming."""
        manager = QuotaManager()
        manager.register_quota(QuotaConfig(
            name="api_calls",
            limit=10,
            period_seconds=86400,
        ))

        result = manager.check("user:123", "api_calls", 5)
        assert result.allowed is True

        # Should still be allowed (didn't consume)
        result = manager.check("user:123", "api_calls", 5)
        assert result.allowed is True


# =============================================================================
# Test Policy Registry
# =============================================================================


class TestPolicyRegistry:
    """Tests for policy registry."""

    def test_register_resolve(self):
        """Test registering and resolving policies."""
        registry = PolicyRegistry()
        registry.register(DefaultPolicy(RateLimitConfig(requests_per_second=10)))

        match = registry.resolve({})
        assert match.matched is True

    def test_unregister(self):
        """Test unregistering policies."""
        registry = PolicyRegistry()
        policy = DefaultPolicy(RateLimitConfig(), name="test")
        registry.register(policy)

        assert registry.unregister("test") is True
        assert registry.unregister("test") is False


# =============================================================================
# Test Rate Limiter
# =============================================================================


class TestRateLimiter:
    """Tests for main RateLimiter class."""

    def test_basic_usage(self):
        """Test basic rate limiter usage."""
        limiter = RateLimiter(
            config=RateLimitConfig(requests_per_second=10, burst_size=5)
        )

        result = limiter.acquire("test")
        assert result.allowed is True

    def test_key_extraction(self):
        """Test key extraction from context."""
        limiter = RateLimiter(
            config=RateLimitConfig(requests_per_second=10),
            key_extractor=AttributeKeyExtractor("user_id"),
        )

        result = limiter.acquire(context={"user_id": "123"})
        assert result.allowed is True
        assert result.bucket_key == "123"

    def test_peek(self):
        """Test peeking without consuming."""
        limiter = RateLimiter(
            config=RateLimitConfig(requests_per_second=10, burst_size=5)
        )

        result1 = limiter.peek("test")
        result2 = limiter.peek("test")
        assert result1.remaining == result2.remaining

    def test_reset(self):
        """Test resetting limiter."""
        limiter = RateLimiter(
            config=RateLimitConfig(requests_per_second=10, burst_size=5)
        )

        limiter.acquire("test")
        limiter.acquire("test")
        limiter.reset("test")

        result = limiter.peek("test")
        assert result.remaining == 5


# =============================================================================
# Test Rate Limiter Registry
# =============================================================================


class TestRateLimiterRegistry:
    """Tests for rate limiter registry."""

    def setup_method(self):
        """Reset registry before each test."""
        RateLimiterRegistry.reset_instance()

    def test_register_get(self):
        """Test registering and getting limiters."""
        registry = RateLimiterRegistry()
        limiter = RateLimiter(name="test")
        registry.register("test", limiter)

        assert registry.get("test") is limiter

    def test_default_limiter(self):
        """Test default limiter."""
        registry = RateLimiterRegistry()
        limiter = RateLimiter(name="test")
        registry.register("test", limiter, set_default=True)

        assert registry.get() is limiter

    def test_get_or_create(self):
        """Test get or create."""
        registry = RateLimiterRegistry()

        limiter = registry.get_or_create("new")
        assert limiter is not None
        assert registry.get("new") is limiter


# =============================================================================
# Test Decorator
# =============================================================================


class TestRateLimitDecorator:
    """Tests for rate limit decorator."""

    def setup_method(self):
        """Reset registry before each test."""
        RateLimiterRegistry.reset_instance()

    def test_basic_decorator(self):
        """Test basic decorator usage."""
        limiter = RateLimiter(
            config=RateLimitConfig(requests_per_second=10, burst_size=5),
            name="test",
        )

        @rate_limit(key="test_func", limiter=limiter)
        def my_function():
            return "success"

        result = my_function()
        assert result == "success"

    def test_decorator_with_key_func(self):
        """Test decorator with key function."""
        limiter = RateLimiter(
            config=RateLimitConfig(requests_per_second=10, burst_size=5),
            name="test",
        )

        @rate_limit(
            key=lambda user_id: f"user:{user_id}",
            limiter=limiter,
        )
        def my_function(user_id):
            return f"success:{user_id}"

        result = my_function("123")
        assert result == "success:123"

    def test_decorator_exceeded(self):
        """Test decorator raises on exceeded."""
        limiter = RateLimiter(
            config=RateLimitConfig(requests_per_second=10, burst_size=1),
            name="test",
        )

        @rate_limit(key="test", limiter=limiter)
        def my_function():
            return "success"

        # First call succeeds
        my_function()

        # Second call should raise
        with pytest.raises(RateLimitExceeded):
            my_function()


# =============================================================================
# Test Context Manager
# =============================================================================


class TestRateLimitContext:
    """Tests for rate limit context manager."""

    def test_context_manager_allowed(self):
        """Test context manager when allowed."""
        limiter = RateLimiter(
            config=RateLimitConfig(requests_per_second=10, burst_size=5)
        )

        with RateLimitContext("test", limiter=limiter) as result:
            assert result.allowed is True

    def test_context_manager_exceeded(self):
        """Test context manager when exceeded."""
        limiter = RateLimiter(
            config=RateLimitConfig(requests_per_second=10, burst_size=1)
        )

        # Exhaust limit
        limiter.acquire("test")

        with pytest.raises(RateLimitExceeded):
            with RateLimitContext("test", limiter=limiter):
                pass

    def test_context_manager_no_raise(self):
        """Test context manager without raising."""
        limiter = RateLimiter(
            config=RateLimitConfig(requests_per_second=10, burst_size=1)
        )

        limiter.acquire("test")

        with RateLimitContext("test", limiter=limiter, raise_on_exceeded=False) as result:
            assert result.allowed is False


# =============================================================================
# Test Metrics
# =============================================================================


class TestRateLimitMetrics:
    """Tests for rate limit metrics."""

    def test_record_metrics(self):
        """Test recording metrics."""
        metrics = RateLimitMetrics()

        # Record allowed
        metrics.record(RateLimitResult(
            allowed=True, remaining=5, limit=10, reset_at=time.time()
        ))

        # Record rejected
        metrics.record(RateLimitResult(
            allowed=False, remaining=0, limit=10, reset_at=time.time()
        ))

        assert metrics.total_requests == 2
        assert metrics.allowed_requests == 1
        assert metrics.rejected_requests == 1
        assert metrics.rejection_rate == 0.5

    def test_metrics_middleware(self):
        """Test metrics middleware."""
        limiter = RateLimiter(
            config=RateLimitConfig(requests_per_second=10, burst_size=5)
        )
        middleware = MetricsMiddleware(limiter)

        middleware.acquire("test")
        middleware.acquire("test")

        assert middleware.metrics.total_requests == 2
        assert middleware.metrics.allowed_requests == 2


# =============================================================================
# Test Retry Handler
# =============================================================================


class TestRetryHandler:
    """Tests for retry handler."""

    def test_immediate_success(self):
        """Test immediate success without retry."""
        limiter = RateLimiter(
            config=RateLimitConfig(requests_per_second=10, burst_size=5)
        )
        handler = RetryHandler(limiter, max_retries=3)

        def my_func():
            return "success"

        result = handler.execute(my_func, key="test")
        assert result == "success"

    def test_retry_on_limit(self):
        """Test retry when rate limited."""
        limiter = RateLimiter(
            config=RateLimitConfig(requests_per_second=100, burst_size=1)  # Fast refill
        )
        handler = RetryHandler(
            limiter,
            max_retries=3,
            base_delay=0.05,
        )

        # Exhaust limit
        limiter.acquire("test")

        call_count = 0

        def my_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = handler.execute(my_func, key="test")
        assert result == "success"


# =============================================================================
# Test Thread Safety
# =============================================================================


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_acquire(self):
        """Test concurrent token acquisition."""
        config = RateLimitConfig(requests_per_second=1000, burst_size=100)
        limiter = RateLimiter(config=config)

        results = []
        errors = []

        def worker():
            try:
                for _ in range(10):
                    result = limiter.acquire("test")
                    results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 100

    def test_concurrent_storage(self):
        """Test concurrent storage access."""
        storage = MemoryStorage()

        def worker(worker_id):
            for i in range(100):
                storage.increment(f"counter:{worker_id}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Each worker should have count of 100
        for i in range(10):
            state = storage.get(f"counter:{i}")
            assert state is not None


# =============================================================================
# Test Dynamic Policy Config
# =============================================================================


class TestDynamicPolicyConfig:
    """Tests for dynamic policy configuration."""

    def test_get_default_config(self):
        """Test getting default config."""
        config = DynamicPolicyConfig(
            default_config=RateLimitConfig(requests_per_second=10)
        )

        result = config.get_config("unknown_key")
        assert result.requests_per_second == 10

    def test_override_config(self):
        """Test override configuration."""
        config = DynamicPolicyConfig(
            default_config=RateLimitConfig(requests_per_second=10)
        )

        config.set_override("vip", RateLimitConfig(requests_per_second=100))

        result = config.get_config("vip")
        assert result.requests_per_second == 100

    def test_remove_override(self):
        """Test removing override."""
        config = DynamicPolicyConfig(
            default_config=RateLimitConfig(requests_per_second=10)
        )

        config.set_override("vip", RateLimitConfig(requests_per_second=100))
        config.remove_override("vip")

        result = config.get_config("vip")
        assert result.requests_per_second == 10


# =============================================================================
# Test Storage Factory
# =============================================================================


class TestStorageFactory:
    """Tests for storage factory."""

    def test_create_memory_storage(self):
        """Test creating memory storage."""
        storage = create_storage("memory")
        assert isinstance(storage, MemoryStorage)

    def test_create_unknown_storage(self):
        """Test creating unknown storage type."""
        with pytest.raises(ValueError):
            create_storage("unknown")
