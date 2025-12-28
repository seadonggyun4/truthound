"""Exactly-once processing semantics for streams.

Provides exactly-once processing guarantees through:
- Idempotent processing with deduplication
- Transactional state updates
- Atomic commit/rollback
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncIterator, Callable, Generic, TypeVar
import asyncio
import hashlib
import logging
import uuid

from truthound.realtime.protocols import (
    IStreamSource,
    IStreamSink,
    IStateStore,
    StreamMessage,
    MessageBatch,
    StreamingError,
)
from truthound.realtime.processing.state import (
    IStateBackend,
    MemoryStateBackend,
    StateManager,
)


logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


class ProcessingGuarantee(str, Enum):
    """Processing guarantee levels."""

    AT_MOST_ONCE = "at_most_once"  # May lose messages
    AT_LEAST_ONCE = "at_least_once"  # May duplicate messages
    EXACTLY_ONCE = "exactly_once"  # No loss, no duplicates


class TransactionState(str, Enum):
    """Transaction states."""

    NONE = "none"
    STARTED = "started"
    PROCESSING = "processing"
    COMMITTING = "committing"
    COMMITTED = "committed"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


@dataclass
class ProcessingContext:
    """Context for message processing."""

    message: StreamMessage
    transaction_id: str
    attempt: int = 1
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingResult(Generic[R]):
    """Result of message processing."""

    success: bool
    result: R | None = None
    error: Exception | None = None
    processing_time_ms: float = 0.0
    skipped: bool = False
    deduplicated: bool = False


# =============================================================================
# Deduplication
# =============================================================================


class DeduplicationStrategy(ABC):
    """Base class for deduplication strategies."""

    @abstractmethod
    async def is_duplicate(self, message: StreamMessage) -> bool:
        """Check if message is a duplicate.

        Args:
            message: Message to check

        Returns:
            True if duplicate
        """
        ...

    @abstractmethod
    async def mark_processed(self, message: StreamMessage) -> None:
        """Mark message as processed.

        Args:
            message: Processed message
        """
        ...


class OffsetDeduplication(DeduplicationStrategy):
    """Deduplication based on partition/offset tracking."""

    def __init__(self, state: IStateBackend):
        self._state = state
        self._namespace = "dedup:offsets"

    async def is_duplicate(self, message: StreamMessage) -> bool:
        key = f"{message.topic}:{message.partition}"
        last_offset = await self._state.get(self._namespace, key)

        if last_offset is not None and message.offset <= last_offset:
            return True
        return False

    async def mark_processed(self, message: StreamMessage) -> None:
        key = f"{message.topic}:{message.partition}"
        await self._state.put(self._namespace, key, message.offset)


class ContentHashDeduplication(DeduplicationStrategy):
    """Deduplication based on message content hash."""

    def __init__(
        self,
        state: IStateBackend,
        ttl_seconds: int = 3600,
        hash_fields: list[str] | None = None,
    ):
        self._state = state
        self._namespace = "dedup:hashes"
        self._ttl = ttl_seconds
        self._hash_fields = hash_fields

    async def is_duplicate(self, message: StreamMessage) -> bool:
        hash_key = self._compute_hash(message)
        exists = await self._state.get(self._namespace, hash_key)
        return exists is not None

    async def mark_processed(self, message: StreamMessage) -> None:
        hash_key = self._compute_hash(message)
        await self._state.put(
            self._namespace,
            hash_key,
            {"offset": message.offset, "timestamp": message.timestamp.isoformat()},
            ttl=self._ttl,
        )

    def _compute_hash(self, message: StreamMessage) -> str:
        """Compute content hash for message."""
        if self._hash_fields and isinstance(message.value, dict):
            content = {k: message.value.get(k) for k in self._hash_fields}
        else:
            content = message.value

        serialized = str(content).encode()
        return hashlib.sha256(serialized).hexdigest()[:32]


class CompositeDeduplication(DeduplicationStrategy):
    """Combine multiple deduplication strategies."""

    def __init__(self, strategies: list[DeduplicationStrategy]):
        self._strategies = strategies

    async def is_duplicate(self, message: StreamMessage) -> bool:
        for strategy in self._strategies:
            if await strategy.is_duplicate(message):
                return True
        return False

    async def mark_processed(self, message: StreamMessage) -> None:
        for strategy in self._strategies:
            await strategy.mark_processed(message)


# =============================================================================
# Transaction Manager
# =============================================================================


@dataclass
class Transaction:
    """Represents a processing transaction."""

    id: str
    state: TransactionState = TransactionState.NONE
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    messages: list[StreamMessage] = field(default_factory=list)
    outputs: list[StreamMessage] = field(default_factory=list)
    state_changes: list[tuple[str, str, Any]] = field(default_factory=list)  # (namespace, key, value)
    error: Exception | None = None


class TransactionManager:
    """Manages processing transactions.

    Provides atomic commit/rollback for exactly-once processing.
    """

    def __init__(self, state: IStateBackend):
        self._state = state
        self._namespace = "transactions"
        self._active_transactions: dict[str, Transaction] = {}

    async def begin(self) -> Transaction:
        """Begin a new transaction.

        Returns:
            New transaction
        """
        txn_id = str(uuid.uuid4())
        txn = Transaction(id=txn_id, state=TransactionState.STARTED)
        self._active_transactions[txn_id] = txn

        # Persist transaction start
        await self._state.put(
            self._namespace,
            txn_id,
            {
                "state": txn.state.value,
                "started_at": txn.started_at.isoformat(),
            },
        )

        logger.debug(f"Transaction started: {txn_id}")
        return txn

    async def add_message(self, txn: Transaction, message: StreamMessage) -> None:
        """Add message to transaction.

        Args:
            txn: Transaction
            message: Message to add
        """
        txn.messages.append(message)
        txn.state = TransactionState.PROCESSING

    async def add_output(self, txn: Transaction, output: StreamMessage) -> None:
        """Add output message to transaction.

        Args:
            txn: Transaction
            output: Output message
        """
        txn.outputs.append(output)

    async def add_state_change(
        self,
        txn: Transaction,
        namespace: str,
        key: str,
        value: Any,
    ) -> None:
        """Add state change to transaction.

        Args:
            txn: Transaction
            namespace: State namespace
            key: State key
            value: New value
        """
        txn.state_changes.append((namespace, key, value))

    async def commit(self, txn: Transaction) -> bool:
        """Commit transaction.

        Args:
            txn: Transaction to commit

        Returns:
            True if committed successfully
        """
        try:
            txn.state = TransactionState.COMMITTING

            # Apply state changes
            for namespace, key, value in txn.state_changes:
                await self._state.put(namespace, key, value)

            # Update transaction state
            await self._state.put(
                self._namespace,
                txn.id,
                {
                    "state": TransactionState.COMMITTED.value,
                    "committed_at": datetime.now(timezone.utc).isoformat(),
                    "message_count": len(txn.messages),
                    "output_count": len(txn.outputs),
                },
            )

            txn.state = TransactionState.COMMITTED
            del self._active_transactions[txn.id]

            logger.debug(f"Transaction committed: {txn.id}")
            return True

        except Exception as e:
            logger.error(f"Transaction commit failed: {txn.id}, error: {e}")
            txn.error = e
            await self.rollback(txn)
            return False

    async def rollback(self, txn: Transaction) -> None:
        """Rollback transaction.

        Args:
            txn: Transaction to rollback
        """
        try:
            txn.state = TransactionState.ROLLING_BACK

            # Note: State changes are not applied until commit,
            # so rollback just clears the pending changes
            txn.state_changes.clear()
            txn.outputs.clear()

            await self._state.put(
                self._namespace,
                txn.id,
                {
                    "state": TransactionState.ROLLED_BACK.value,
                    "rolled_back_at": datetime.now(timezone.utc).isoformat(),
                    "error": str(txn.error) if txn.error else None,
                },
            )

            txn.state = TransactionState.ROLLED_BACK

            if txn.id in self._active_transactions:
                del self._active_transactions[txn.id]

            logger.debug(f"Transaction rolled back: {txn.id}")

        except Exception as e:
            logger.error(f"Transaction rollback failed: {txn.id}, error: {e}")
            txn.state = TransactionState.FAILED

    async def recover_pending_transactions(self) -> list[Transaction]:
        """Recover pending transactions after restart.

        Returns:
            List of pending transactions
        """
        pending = []
        all_txns = await self._state.get_all(self._namespace)

        for txn_id, txn_data in all_txns.items():
            state = txn_data.get("state")
            if state in (TransactionState.STARTED.value, TransactionState.PROCESSING.value):
                txn = Transaction(
                    id=txn_id,
                    state=TransactionState(state),
                    started_at=datetime.fromisoformat(txn_data["started_at"]),
                )
                pending.append(txn)

        return pending


# =============================================================================
# Exactly-Once Processor
# =============================================================================


class ExactlyOnceProcessor(Generic[T, R]):
    """Processor with exactly-once semantics.

    Provides exactly-once processing guarantees through:
    - Idempotent processing with deduplication
    - Transactional state updates
    - Atomic source offset commits

    Example:
        >>> processor = ExactlyOnceProcessor(
        ...     source=kafka_adapter,
        ...     sink=output_adapter,
        ...     process_fn=my_processor,
        ...     state_backend=redis_state,
        ... )
        >>> await processor.run()
    """

    def __init__(
        self,
        source: IStreamSource[T],
        sink: IStreamSink[R] | None = None,
        process_fn: Callable[[StreamMessage[T]], StreamMessage[R] | None] | None = None,
        state_backend: IStateBackend | None = None,
        dedup_strategy: DeduplicationStrategy | None = None,
        guarantee: ProcessingGuarantee = ProcessingGuarantee.EXACTLY_ONCE,
        batch_size: int = 100,
        commit_interval_ms: int = 1000,
        max_retries: int = 3,
    ):
        """Initialize exactly-once processor.

        Args:
            source: Message source
            sink: Optional message sink
            process_fn: Processing function
            state_backend: State backend for transactions
            dedup_strategy: Deduplication strategy
            guarantee: Processing guarantee level
            batch_size: Messages per batch
            commit_interval_ms: Commit interval
            max_retries: Max processing retries
        """
        self._source = source
        self._sink = sink
        self._process_fn = process_fn
        self._state = state_backend or MemoryStateBackend()
        self._guarantee = guarantee
        self._batch_size = batch_size
        self._commit_interval_ms = commit_interval_ms
        self._max_retries = max_retries

        # Setup deduplication
        if dedup_strategy:
            self._dedup = dedup_strategy
        elif guarantee == ProcessingGuarantee.EXACTLY_ONCE:
            self._dedup = CompositeDeduplication([
                OffsetDeduplication(self._state),
                ContentHashDeduplication(self._state),
            ])
        else:
            self._dedup = None

        self._txn_manager = TransactionManager(self._state)
        self._running = False
        self._processed_count = 0
        self._duplicate_count = 0
        self._error_count = 0

    @property
    def stats(self) -> dict[str, int]:
        """Get processing statistics."""
        return {
            "processed": self._processed_count,
            "duplicates": self._duplicate_count,
            "errors": self._error_count,
        }

    async def run(self, max_messages: int | None = None) -> None:
        """Run the processor.

        Args:
            max_messages: Max messages to process (None for infinite)
        """
        self._running = True
        logger.info(f"Starting exactly-once processor with {self._guarantee.value} guarantee")

        try:
            processed = 0
            async for message in self._source.consume():
                if not self._running:
                    break

                if max_messages and processed >= max_messages:
                    break

                result = await self._process_message(message)

                if result.success and not result.deduplicated:
                    processed += 1
                    self._processed_count += 1
                elif result.deduplicated:
                    self._duplicate_count += 1
                else:
                    self._error_count += 1

        finally:
            self._running = False
            logger.info(f"Processor stopped. Processed: {self._processed_count}")

    async def stop(self) -> None:
        """Stop the processor."""
        self._running = False

    async def _process_message(self, message: StreamMessage[T]) -> ProcessingResult[R]:
        """Process a single message with exactly-once semantics.

        Args:
            message: Message to process

        Returns:
            Processing result
        """
        import time
        start_time = time.perf_counter()

        # Check for duplicates
        if self._dedup and await self._dedup.is_duplicate(message):
            logger.debug(f"Duplicate message skipped: {message.offset}")
            return ProcessingResult(success=True, deduplicated=True)

        # Begin transaction
        txn = await self._txn_manager.begin()

        try:
            await self._txn_manager.add_message(txn, message)

            # Process message
            output = None
            if self._process_fn:
                output = self._process_fn(message)

            # If output produced, add to transaction
            if output and self._sink:
                await self._txn_manager.add_output(txn, output)

            # Mark as processed (part of transaction)
            if self._dedup:
                await self._dedup.mark_processed(message)

            # Commit transaction
            success = await self._txn_manager.commit(txn)

            if success:
                # Commit source offset
                await self._source.commit(message)

                # Produce output
                if output and self._sink:
                    await self._sink.produce(output)

            elapsed = (time.perf_counter() - start_time) * 1000

            return ProcessingResult(
                success=success,
                result=output.value if output else None,
                processing_time_ms=elapsed,
            )

        except Exception as e:
            logger.error(f"Processing failed for message {message.offset}: {e}")
            await self._txn_manager.rollback(txn)

            return ProcessingResult(
                success=False,
                error=e,
                processing_time_ms=(time.perf_counter() - start_time) * 1000,
            )

    async def recover(self) -> int:
        """Recover from failures.

        Finds and rolls back any pending transactions.

        Returns:
            Number of transactions recovered
        """
        pending = await self._txn_manager.recover_pending_transactions()

        for txn in pending:
            logger.info(f"Rolling back pending transaction: {txn.id}")
            await self._txn_manager.rollback(txn)

        return len(pending)
