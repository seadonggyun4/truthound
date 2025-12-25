"""Baggage management for distributed tracing.

Baggage provides a mechanism to propagate user-defined key-value pairs
across service boundaries. This is useful for passing contextual data
like user IDs, tenant IDs, or feature flags.

Important: Baggage is propagated in-band with trace context, so it
adds overhead to every request. Use it sparingly.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping


# =============================================================================
# Baggage
# =============================================================================


@dataclass
class Baggage:
    """Container for baggage key-value pairs.

    Baggage is immutable by design to prevent race conditions.
    Use the factory methods to create modified versions.

    Example:
        >>> baggage = Baggage()
        >>> baggage = baggage.set("user_id", "123")
        >>> baggage = baggage.set("tenant", "acme")
        >>> print(baggage.get("user_id"))
        '123'
    """

    _entries: dict[str, str] = field(default_factory=dict)

    def get(self, key: str, default: str | None = None) -> str | None:
        """Get a baggage value.

        Args:
            key: Baggage key.
            default: Default value if not found.

        Returns:
            Baggage value or default.
        """
        return self._entries.get(key, default)

    def set(self, key: str, value: str) -> "Baggage":
        """Create new baggage with added/updated entry.

        Args:
            key: Baggage key.
            value: Baggage value.

        Returns:
            New Baggage with the entry.
        """
        new_entries = dict(self._entries)
        new_entries[key] = value
        return Baggage(_entries=new_entries)

    def remove(self, key: str) -> "Baggage":
        """Create new baggage with entry removed.

        Args:
            key: Key to remove.

        Returns:
            New Baggage without the entry.
        """
        new_entries = {k: v for k, v in self._entries.items() if k != key}
        return Baggage(_entries=new_entries)

    def clear(self) -> "Baggage":
        """Create empty baggage.

        Returns:
            Empty Baggage.
        """
        return Baggage()

    def __contains__(self, key: str) -> bool:
        """Check if key exists in baggage."""
        return key in self._entries

    def __len__(self) -> int:
        """Get number of entries."""
        return len(self._entries)

    def __iter__(self) -> Iterator[str]:
        """Iterate over keys."""
        return iter(self._entries)

    def items(self) -> Iterator[tuple[str, str]]:
        """Iterate over key-value pairs."""
        return iter(self._entries.items())

    def keys(self) -> Iterator[str]:
        """Iterate over keys."""
        return iter(self._entries.keys())

    def values(self) -> Iterator[str]:
        """Iterate over values."""
        return iter(self._entries.values())

    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary.

        Returns:
            Dictionary of baggage entries.
        """
        return dict(self._entries)

    @classmethod
    def from_dict(cls, entries: Mapping[str, str]) -> "Baggage":
        """Create baggage from dictionary.

        Args:
            entries: Dictionary of entries.

        Returns:
            New Baggage.
        """
        return cls(_entries=dict(entries))

    @classmethod
    def empty(cls) -> "Baggage":
        """Create empty baggage.

        Returns:
            Empty Baggage.
        """
        return cls()


# =============================================================================
# Context Storage
# =============================================================================


class _BaggageStorage:
    """Thread-local storage for baggage."""

    _local = threading.local()

    @classmethod
    def get_baggage(cls) -> Baggage:
        """Get current baggage."""
        if not hasattr(cls._local, "baggage"):
            cls._local.baggage = Baggage()
        return cls._local.baggage

    @classmethod
    def set_baggage(cls, baggage: Baggage) -> None:
        """Set current baggage."""
        cls._local.baggage = baggage

    @classmethod
    def clear(cls) -> None:
        """Clear baggage."""
        cls._local.baggage = Baggage()


# =============================================================================
# Context Management Functions
# =============================================================================


def get_baggage() -> Baggage:
    """Get the current baggage.

    Returns:
        Current Baggage.
    """
    return _BaggageStorage.get_baggage()


def get_baggage_item(key: str) -> str | None:
    """Get a single baggage item.

    Args:
        key: Baggage key.

    Returns:
        Value or None.
    """
    return _BaggageStorage.get_baggage().get(key)


def set_baggage(key: str, value: str) -> Baggage:
    """Set a baggage item in current context.

    Args:
        key: Baggage key.
        value: Baggage value.

    Returns:
        Updated Baggage.
    """
    baggage = _BaggageStorage.get_baggage()
    new_baggage = baggage.set(key, value)
    _BaggageStorage.set_baggage(new_baggage)
    return new_baggage


def remove_baggage(key: str) -> Baggage:
    """Remove a baggage item from current context.

    Args:
        key: Key to remove.

    Returns:
        Updated Baggage.
    """
    baggage = _BaggageStorage.get_baggage()
    new_baggage = baggage.remove(key)
    _BaggageStorage.set_baggage(new_baggage)
    return new_baggage


def clear_baggage() -> Baggage:
    """Clear all baggage from current context.

    Returns:
        Empty Baggage.
    """
    new_baggage = Baggage()
    _BaggageStorage.set_baggage(new_baggage)
    return new_baggage


def set_baggage_context(baggage: Baggage) -> None:
    """Set the entire baggage context.

    Args:
        baggage: Baggage to set.
    """
    _BaggageStorage.set_baggage(baggage)


# =============================================================================
# Context Manager
# =============================================================================


from contextlib import contextmanager
from typing import Iterator


@contextmanager
def baggage_context(**entries: str) -> Iterator[Baggage]:
    """Context manager for temporary baggage.

    Adds baggage entries for the duration of the context,
    then restores the previous state.

    Args:
        **entries: Baggage entries to add.

    Yields:
        Baggage with entries.

    Example:
        >>> with baggage_context(user_id="123", tenant="acme"):
        ...     # Baggage is available here
        ...     do_something()
        >>> # Baggage is restored here
    """
    previous = _BaggageStorage.get_baggage()

    # Add new entries
    baggage = previous
    for key, value in entries.items():
        baggage = baggage.set(key, value)

    _BaggageStorage.set_baggage(baggage)

    try:
        yield baggage
    finally:
        _BaggageStorage.set_baggage(previous)


@contextmanager
def baggage_from_dict(entries: Mapping[str, str]) -> Iterator[Baggage]:
    """Context manager for baggage from dictionary.

    Args:
        entries: Dictionary of baggage entries.

    Yields:
        Baggage with entries.
    """
    previous = _BaggageStorage.get_baggage()

    # Add new entries
    baggage = previous
    for key, value in entries.items():
        baggage = baggage.set(key, value)

    _BaggageStorage.set_baggage(baggage)

    try:
        yield baggage
    finally:
        _BaggageStorage.set_baggage(previous)
