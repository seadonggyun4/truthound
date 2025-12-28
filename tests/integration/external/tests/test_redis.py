"""Redis integration tests.

Tests Redis functionality including:
- Basic key-value operations
- TTL and expiration
- Hash operations
- List operations
- Set operations
- Pub/Sub
- Connection pooling
- Cluster mode (if available)

These tests can run against:
- Docker containers (default)
- Local Redis instances
- Mock Redis (for fast testing)
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from tests.integration.external.backends.redis_backend import RedisBackend
    from tests.integration.external.providers.mock_provider import MockRedisService


# =============================================================================
# Mock Redis Tests (Fast, No External Dependencies)
# =============================================================================


class TestMockRedis:
    """Tests using mock Redis for fast execution."""

    def test_string_operations(self, mock_redis: "MockRedisService") -> None:
        """Test basic string operations."""
        # Set and get
        assert mock_redis.set("key1", "value1")
        assert mock_redis.get("key1") == "value1"

        # Overwrite
        mock_redis.set("key1", "value2")
        assert mock_redis.get("key1") == "value2"

        # Get non-existent key
        assert mock_redis.get("nonexistent") is None

    def test_delete_operations(self, mock_redis: "MockRedisService") -> None:
        """Test delete operations."""
        mock_redis.set("key1", "value1")
        mock_redis.set("key2", "value2")
        mock_redis.set("key3", "value3")

        # Delete single key
        assert mock_redis.delete("key1") == 1
        assert mock_redis.get("key1") is None

        # Delete multiple keys
        assert mock_redis.delete("key2", "key3") == 2
        assert mock_redis.get("key2") is None
        assert mock_redis.get("key3") is None

        # Delete non-existent key
        assert mock_redis.delete("nonexistent") == 0

    def test_exists_operation(self, mock_redis: "MockRedisService") -> None:
        """Test exists operation."""
        mock_redis.set("key1", "value1")
        mock_redis.set("key2", "value2")

        assert mock_redis.exists("key1") == 1
        assert mock_redis.exists("key1", "key2") == 2
        assert mock_redis.exists("key1", "nonexistent") == 1
        assert mock_redis.exists("nonexistent") == 0

    def test_ttl_operations(self, mock_redis: "MockRedisService") -> None:
        """Test TTL operations."""
        # Set with TTL
        mock_redis.set("expiring_key", "value", ex=2)
        assert mock_redis.ttl("expiring_key") > 0
        assert mock_redis.get("expiring_key") == "value"

        # Wait for expiration
        time.sleep(2.1)
        assert mock_redis.get("expiring_key") is None
        assert mock_redis.ttl("expiring_key") == -2  # Key doesn't exist

    def test_expire_command(self, mock_redis: "MockRedisService") -> None:
        """Test EXPIRE command."""
        mock_redis.set("key1", "value1")
        assert mock_redis.ttl("key1") == -1  # No TTL

        mock_redis.expire("key1", 10)
        assert 0 < mock_redis.ttl("key1") <= 10

    def test_keys_pattern(self, mock_redis: "MockRedisService") -> None:
        """Test KEYS pattern matching."""
        mock_redis.set("user:1", "alice")
        mock_redis.set("user:2", "bob")
        mock_redis.set("session:1", "data")

        user_keys = mock_redis.keys("user:*")
        assert len(user_keys) == 2
        assert "user:1" in user_keys
        assert "user:2" in user_keys

        all_keys = mock_redis.keys("*")
        assert len(all_keys) == 3

    def test_hash_operations(self, mock_redis: "MockRedisService") -> None:
        """Test hash operations."""
        # HSET
        assert mock_redis.hset("user:1", "name", "alice") == 1
        assert mock_redis.hset("user:1", "email", "alice@example.com") == 1
        assert mock_redis.hset("user:1", "name", "Alice") == 0  # Update returns 0

        # HGET
        assert mock_redis.hget("user:1", "name") == "Alice"
        assert mock_redis.hget("user:1", "nonexistent") is None

        # HGETALL
        data = mock_redis.hgetall("user:1")
        assert data == {"name": "Alice", "email": "alice@example.com"}

        # HDEL
        assert mock_redis.hdel("user:1", "email") == 1
        assert mock_redis.hgetall("user:1") == {"name": "Alice"}

    def test_list_operations(self, mock_redis: "MockRedisService") -> None:
        """Test list operations."""
        # LPUSH
        assert mock_redis.lpush("mylist", "a") == 1
        assert mock_redis.lpush("mylist", "b", "c") == 3

        # LRANGE
        assert mock_redis.lrange("mylist", 0, -1) == ["c", "b", "a"]
        assert mock_redis.lrange("mylist", 0, 1) == ["c", "b"]

        # RPUSH
        assert mock_redis.rpush("mylist", "d") == 4
        assert mock_redis.lrange("mylist", 0, -1) == ["c", "b", "a", "d"]

        # LLEN
        assert mock_redis.llen("mylist") == 4

    def test_set_operations(self, mock_redis: "MockRedisService") -> None:
        """Test set operations."""
        # SADD
        assert mock_redis.sadd("myset", "a", "b", "c") == 3
        assert mock_redis.sadd("myset", "a") == 0  # Already exists

        # SMEMBERS
        members = mock_redis.smembers("myset")
        assert members == {"a", "b", "c"}

        # SISMEMBER
        assert mock_redis.sismember("myset", "a") is True
        assert mock_redis.sismember("myset", "x") is False

        # SCARD
        assert mock_redis.scard("myset") == 3

    def test_ping(self, mock_redis: "MockRedisService") -> None:
        """Test PING command."""
        assert mock_redis.ping() == "PONG"

    def test_flushdb(self, mock_redis: "MockRedisService") -> None:
        """Test FLUSHDB command."""
        mock_redis.set("key1", "value1")
        mock_redis.set("key2", "value2")

        assert mock_redis.flushdb() is True
        assert mock_redis.get("key1") is None
        assert mock_redis.get("key2") is None


# =============================================================================
# Docker Redis Tests (Real Redis Instance)
# =============================================================================


@pytest.mark.redis
@pytest.mark.integration
class TestDockerRedis:
    """Tests using Docker Redis container."""

    def test_connection(self, redis_backend: "RedisBackend") -> None:
        """Test Redis connection."""
        assert redis_backend.is_running
        assert redis_backend.is_healthy

    def test_health_check(self, redis_backend: "RedisBackend") -> None:
        """Test health check."""
        result = redis_backend.health_check()
        assert result.healthy
        assert "redis_version" in result.details

    def test_basic_operations(self, redis_client) -> None:
        """Test basic Redis operations."""
        # Set
        assert redis_client.set("test_key", "test_value")

        # Get
        value = redis_client.get("test_key")
        assert value == "test_value"

        # Delete
        assert redis_client.delete("test_key") == 1
        assert redis_client.get("test_key") is None

    def test_ttl_with_expiry(self, redis_client) -> None:
        """Test TTL with expiry."""
        redis_client.set("expiring", "value", ex=5)

        # Should exist
        assert redis_client.get("expiring") == "value"

        # TTL should be positive
        ttl = redis_client.ttl("expiring")
        assert 0 < ttl <= 5

    def test_hash_operations(self, redis_client) -> None:
        """Test hash operations with real Redis."""
        # Set hash fields
        redis_client.hset("test_hash", "field1", "value1")
        redis_client.hset("test_hash", "field2", "value2")

        # Get single field
        assert redis_client.hget("test_hash", "field1") == "value1"

        # Get all fields
        all_fields = redis_client.hgetall("test_hash")
        assert all_fields == {"field1": "value1", "field2": "value2"}

    def test_list_operations(self, redis_client) -> None:
        """Test list operations with real Redis."""
        # Push to list
        redis_client.lpush("test_list", "a", "b", "c")

        # Get list
        items = redis_client.lrange("test_list", 0, -1)
        assert items == ["c", "b", "a"]

        # List length
        assert redis_client.llen("test_list") == 3

    def test_set_operations(self, redis_client) -> None:
        """Test set operations with real Redis."""
        # Add to set
        redis_client.sadd("test_set", "a", "b", "c")

        # Check membership
        assert redis_client.sismember("test_set", "a")
        assert not redis_client.sismember("test_set", "x")

        # Get all members
        members = redis_client.smembers("test_set")
        assert members == {"a", "b", "c"}

    def test_pipeline(self, redis_client) -> None:
        """Test Redis pipeline for batch operations."""
        pipe = redis_client.pipeline()
        pipe.set("pipe_key1", "value1")
        pipe.set("pipe_key2", "value2")
        pipe.get("pipe_key1")
        pipe.get("pipe_key2")

        results = pipe.execute()
        assert results[0] is True  # SET result
        assert results[1] is True  # SET result
        assert results[2] == "value1"  # GET result
        assert results[3] == "value2"  # GET result

    def test_transaction(self, redis_client) -> None:
        """Test Redis transaction with MULTI/EXEC."""
        pipe = redis_client.pipeline(transaction=True)
        pipe.set("tx_key", "0")
        pipe.incr("tx_key")
        pipe.incr("tx_key")
        pipe.incr("tx_key")

        results = pipe.execute()
        assert redis_client.get("tx_key") == "3"

    def test_info(self, redis_backend: "RedisBackend") -> None:
        """Test Redis INFO command."""
        info = redis_backend.get_info()
        assert "redis_version" in info
        assert "connected_clients" in info

    def test_dbsize(self, redis_backend: "RedisBackend") -> None:
        """Test key count."""
        # Flush first
        redis_backend.flush_all()
        assert redis_backend.get_key_count() == 0

        # Add some keys
        redis_backend.client.set("key1", "value1")
        redis_backend.client.set("key2", "value2")

        assert redis_backend.get_key_count() == 2


# =============================================================================
# Distributed Coordination Tests
# =============================================================================


@pytest.mark.redis
@pytest.mark.integration
class TestRedisDistributedCoordination:
    """Tests for distributed coordination features."""

    def test_distributed_lock(self, redis_client) -> None:
        """Test distributed locking."""
        lock_key = "test_lock"

        # Acquire lock
        acquired = redis_client.set(
            lock_key, "owner1", nx=True, ex=10
        )
        assert acquired is True

        # Try to acquire same lock (should fail)
        not_acquired = redis_client.set(
            lock_key, "owner2", nx=True, ex=10
        )
        assert not_acquired is None

        # Release lock
        redis_client.delete(lock_key)

        # Now can acquire
        acquired = redis_client.set(
            lock_key, "owner2", nx=True, ex=10
        )
        assert acquired is True

    def test_counter(self, redis_client) -> None:
        """Test atomic counter operations."""
        counter_key = "test_counter"
        redis_client.set(counter_key, 0)

        # Increment
        assert redis_client.incr(counter_key) == 1
        assert redis_client.incr(counter_key) == 2
        assert redis_client.incrby(counter_key, 10) == 12

        # Decrement
        assert redis_client.decr(counter_key) == 11
        assert redis_client.decrby(counter_key, 5) == 6

    def test_rate_limiting(self, redis_client) -> None:
        """Test rate limiting pattern."""
        rate_key = "rate_limit:user:123"
        window_seconds = 2
        max_requests = 5

        # Simulate requests
        for i in range(max_requests):
            count = redis_client.incr(rate_key)
            if i == 0:
                redis_client.expire(rate_key, window_seconds)
            assert count <= max_requests

        # Should be at limit
        assert redis_client.get(rate_key) == str(max_requests)

        # Additional request should exceed limit
        count = redis_client.incr(rate_key)
        assert int(count) > max_requests

        # Wait for window to expire
        time.sleep(window_seconds + 0.1)
        assert redis_client.get(rate_key) is None
