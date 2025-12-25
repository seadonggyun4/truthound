"""Request Fingerprinting for Idempotency.

This module provides utilities for generating unique fingerprints
from request data to create idempotency keys.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Sequence


@dataclass
class RequestFingerprint:
    """Represents a unique fingerprint for a request.

    The fingerprint is used to generate idempotency keys that uniquely
    identify a specific request, enabling deduplication.

    Attributes:
        key: The generated fingerprint key.
        components: Components used to generate the fingerprint.
        algorithm: Hash algorithm used.
        created_at: When the fingerprint was created.
        metadata: Additional fingerprint metadata.
    """

    key: str
    components: dict[str, Any] = field(default_factory=dict)
    algorithm: str = "sha256"
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        algorithm: str = "sha256",
        prefix: str = "",
    ) -> "RequestFingerprint":
        """Create a fingerprint from a dictionary.

        Args:
            data: Dictionary of key-value pairs.
            algorithm: Hash algorithm to use.
            prefix: Optional prefix for the key.

        Returns:
            RequestFingerprint instance.

        Example:
            >>> fp = RequestFingerprint.from_dict({
            ...     "action": "validate",
            ...     "dataset_id": "ds-123",
            ...     "version": 1,
            ... })
            >>> print(fp.key)
        """
        # Create canonical JSON representation
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))

        # Generate hash
        hasher = hashlib.new(algorithm)
        hasher.update(canonical.encode("utf-8"))
        hash_value = hasher.hexdigest()

        key = f"{prefix}{hash_value}" if prefix else hash_value

        return cls(
            key=key,
            components=data,
            algorithm=algorithm,
        )

    @classmethod
    def from_args(
        cls,
        *args: Any,
        prefix: str = "",
        **kwargs: Any,
    ) -> "RequestFingerprint":
        """Create a fingerprint from function arguments.

        Args:
            *args: Positional arguments.
            prefix: Optional prefix for the key.
            **kwargs: Keyword arguments.

        Returns:
            RequestFingerprint instance.

        Example:
            >>> fp = RequestFingerprint.from_args(
            ...     "validate_data",
            ...     dataset_id="ds-123",
            ...     options={"strict": True},
            ... )
        """
        data = {
            "args": [_serialize_value(a) for a in args],
            "kwargs": {k: _serialize_value(v) for k, v in sorted(kwargs.items())},
        }
        return cls.from_dict(data, prefix=prefix)

    @classmethod
    def from_string(
        cls,
        value: str,
        algorithm: str = "sha256",
        prefix: str = "",
    ) -> "RequestFingerprint":
        """Create a fingerprint from a string.

        Args:
            value: String value to hash.
            algorithm: Hash algorithm.
            prefix: Optional prefix.

        Returns:
            RequestFingerprint instance.
        """
        hasher = hashlib.new(algorithm)
        hasher.update(value.encode("utf-8"))
        hash_value = hasher.hexdigest()

        key = f"{prefix}{hash_value}" if prefix else hash_value

        return cls(
            key=key,
            components={"value": value},
            algorithm=algorithm,
        )

    def with_prefix(self, prefix: str) -> "RequestFingerprint":
        """Create a copy with a different prefix.

        Args:
            prefix: New prefix for the key.

        Returns:
            New RequestFingerprint with prefixed key.
        """
        return RequestFingerprint(
            key=f"{prefix}{self.key}",
            components=self.components,
            algorithm=self.algorithm,
            created_at=self.created_at,
            metadata=self.metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key": self.key,
            "components": self.components,
            "algorithm": self.algorithm,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


class FingerprintStrategy(ABC):
    """Abstract base class for fingerprint generation strategies.

    Different strategies can be used to generate fingerprints based on
    different requirements (content-based, structural, composite, etc.).
    """

    @abstractmethod
    def generate(self, data: Any) -> RequestFingerprint:
        """Generate a fingerprint from data.

        Args:
            data: Input data to fingerprint.

        Returns:
            Generated RequestFingerprint.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the strategy name."""
        pass


class ContentHashStrategy(FingerprintStrategy):
    """Strategy that hashes the full content.

    This strategy creates a fingerprint by hashing the complete
    serialized content of the input data.

    Example:
        >>> strategy = ContentHashStrategy()
        >>> fp = strategy.generate({"action": "validate", "data": [1, 2, 3]})
    """

    def __init__(
        self,
        algorithm: str = "sha256",
        prefix: str = "",
        include_types: bool = False,
    ) -> None:
        """Initialize the strategy.

        Args:
            algorithm: Hash algorithm to use.
            prefix: Optional prefix for keys.
            include_types: Whether to include type information in hash.
        """
        self._algorithm = algorithm
        self._prefix = prefix
        self._include_types = include_types

    @property
    def name(self) -> str:
        return "content_hash"

    def generate(self, data: Any) -> RequestFingerprint:
        """Generate fingerprint by hashing content."""
        if self._include_types:
            serialized = self._serialize_with_types(data)
        else:
            serialized = _serialize_value(data)

        return RequestFingerprint.from_dict(
            {"content": serialized},
            algorithm=self._algorithm,
            prefix=self._prefix,
        )

    def _serialize_with_types(self, value: Any) -> dict[str, Any]:
        """Serialize value with type information."""
        return {
            "type": type(value).__name__,
            "value": _serialize_value(value),
        }


class StructuralHashStrategy(FingerprintStrategy):
    """Strategy that hashes only selected fields.

    This strategy creates a fingerprint by hashing only specific
    fields from the input data, ignoring transient fields.

    Example:
        >>> strategy = StructuralHashStrategy(
        ...     fields=["action", "dataset_id"],
        ...     ignore=["timestamp", "request_id"],
        ... )
        >>> fp = strategy.generate({
        ...     "action": "validate",
        ...     "dataset_id": "ds-123",
        ...     "timestamp": "2024-01-01",  # Ignored
        ... })
    """

    def __init__(
        self,
        fields: Sequence[str] | None = None,
        ignore: Sequence[str] | None = None,
        algorithm: str = "sha256",
        prefix: str = "",
    ) -> None:
        """Initialize the strategy.

        Args:
            fields: Fields to include (if None, includes all except ignored).
            ignore: Fields to ignore.
            algorithm: Hash algorithm.
            prefix: Key prefix.
        """
        self._fields = set(fields) if fields else None
        self._ignore = set(ignore) if ignore else set()
        self._algorithm = algorithm
        self._prefix = prefix

    @property
    def name(self) -> str:
        return "structural_hash"

    def generate(self, data: Any) -> RequestFingerprint:
        """Generate fingerprint from selected fields."""
        if not isinstance(data, dict):
            raise ValueError("StructuralHashStrategy requires dict input")

        filtered = self._filter_fields(data)
        return RequestFingerprint.from_dict(
            filtered,
            algorithm=self._algorithm,
            prefix=self._prefix,
        )

    def _filter_fields(self, data: dict[str, Any]) -> dict[str, Any]:
        """Filter fields based on configuration."""
        result = {}

        for key, value in data.items():
            # Skip ignored fields
            if key in self._ignore:
                continue

            # Include if fields is None (all) or key is in fields
            if self._fields is None or key in self._fields:
                result[key] = _serialize_value(value)

        return result


class CompositeFingerprint(FingerprintStrategy):
    """Strategy that combines multiple fingerprint strategies.

    This strategy applies multiple strategies and combines their
    results into a single fingerprint.

    Example:
        >>> composite = CompositeFingerprint([
        ...     StructuralHashStrategy(fields=["action", "target"]),
        ...     ContentHashStrategy(prefix="content:"),
        ... ])
        >>> fp = composite.generate(data)
    """

    def __init__(
        self,
        strategies: Sequence[FingerprintStrategy],
        combiner: Callable[[Sequence[str]], str] | None = None,
        prefix: str = "",
    ) -> None:
        """Initialize the strategy.

        Args:
            strategies: List of strategies to apply.
            combiner: Function to combine fingerprint keys.
            prefix: Key prefix.
        """
        self._strategies = list(strategies)
        self._combiner = combiner or self._default_combiner
        self._prefix = prefix

    @property
    def name(self) -> str:
        return "composite"

    def generate(self, data: Any) -> RequestFingerprint:
        """Generate combined fingerprint."""
        fingerprints = [s.generate(data) for s in self._strategies]
        keys = [fp.key for fp in fingerprints]

        combined_key = self._combiner(keys)
        if self._prefix:
            combined_key = f"{self._prefix}{combined_key}"

        return RequestFingerprint(
            key=combined_key,
            components={
                "strategies": [s.name for s in self._strategies],
                "keys": keys,
            },
            metadata={
                "composite": True,
                "strategy_count": len(self._strategies),
            },
        )

    @staticmethod
    def _default_combiner(keys: Sequence[str]) -> str:
        """Default combiner - hash the concatenated keys."""
        combined = "|".join(keys)
        return hashlib.sha256(combined.encode()).hexdigest()


# =============================================================================
# Utility Functions
# =============================================================================


def _serialize_value(value: Any) -> Any:
    """Serialize a value for hashing.

    Converts complex types to JSON-serializable format.
    """
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, bytes):
        return value.hex()

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]

    if isinstance(value, dict):
        return {str(k): _serialize_value(v) for k, v in sorted(value.items())}

    if isinstance(value, set):
        return sorted([_serialize_value(v) for v in value])

    if hasattr(value, "to_dict"):
        return _serialize_value(value.to_dict())

    if hasattr(value, "__dict__"):
        return _serialize_value(value.__dict__)

    # Fallback to string representation
    return str(value)


def generate_fingerprint(
    data: Any,
    strategy: FingerprintStrategy | None = None,
    prefix: str = "",
) -> RequestFingerprint:
    """Generate a fingerprint using the specified strategy.

    Convenience function for generating fingerprints.

    Args:
        data: Input data.
        strategy: Fingerprint strategy (defaults to ContentHashStrategy).
        prefix: Key prefix.

    Returns:
        Generated fingerprint.
    """
    if strategy is None:
        strategy = ContentHashStrategy(prefix=prefix)

    return strategy.generate(data)


def quick_fingerprint(*args: Any, **kwargs: Any) -> str:
    """Generate a quick fingerprint key from arguments.

    Convenience function for simple use cases.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Fingerprint key string.
    """
    return RequestFingerprint.from_args(*args, **kwargs).key
