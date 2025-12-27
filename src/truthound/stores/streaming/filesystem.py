"""Streaming filesystem store implementation.

This module provides a streaming-capable filesystem store that uses JSONL
(JSON Lines) format for efficient streaming of large validation results.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator, Iterator
from uuid import uuid4

from truthound.stores.streaming.base import (
    ChunkInfo,
    CompressionType,
    StreamingConfig,
    StreamingFormat,
    StreamingMetrics,
    StreamingStore,
    StreamingValidationStore,
    StreamSession,
    StreamStatus,
)
from truthound.stores.streaming.reader import (
    AsyncStreamReader,
    BaseStreamReader,
    StreamingResultReader,
)
from truthound.stores.streaming.writer import (
    AsyncStreamWriter,
    BaseStreamWriter,
    StreamingResultWriter,
)

if TYPE_CHECKING:
    from truthound.stores.results import ValidationResult, ValidatorResult


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class StreamingFileSystemConfig(StreamingConfig):
    """Configuration for streaming filesystem store.

    Attributes:
        base_path: Base directory for streaming storage.
        create_dirs: Whether to create directories if they don't exist.
        preserve_chunks: Whether to keep chunks after finalization.
        manifest_filename: Name of the manifest file.
    """

    base_path: str = ".truthound/streaming"
    create_dirs: bool = True
    preserve_chunks: bool = True
    manifest_filename: str = "_manifest.json"


# =============================================================================
# Streaming Filesystem Store
# =============================================================================


class StreamingFileSystemStore(StreamingValidationStore[StreamingFileSystemConfig]):
    """Streaming filesystem store using JSONL format.

    This store is optimized for handling large validation results that
    cannot fit in memory. It uses:

    - JSONL (JSON Lines) format for efficient streaming
    - Chunked storage for manageable file sizes
    - Manifest files for session tracking
    - Atomic writes for data integrity
    - Optional compression (gzip, zstd, lz4)

    Example:
        >>> store = StreamingFileSystemStore(base_path=".truthound/streaming")
        >>>
        >>> # Create a streaming session
        >>> session = store.create_session("run_001", "large_dataset.csv")
        >>>
        >>> # Write results incrementally
        >>> with store.create_writer(session) as writer:
        ...     for result in validation_results:
        ...         writer.write_result(result)
        >>>
        >>> # Finalize and get complete result
        >>> final_result = store.finalize_result(session)
        >>>
        >>> # Read results back efficiently
        >>> for result in store.iter_results("run_001"):
        ...     process(result)
    """

    def __init__(
        self,
        base_path: str = ".truthound/streaming",
        format: StreamingFormat = StreamingFormat.JSONL,
        compression: CompressionType = CompressionType.NONE,
        chunk_size: int = 10000,
        buffer_size: int = 1000,
        **kwargs: Any,
    ):
        """Initialize the streaming filesystem store.

        Args:
            base_path: Base directory for streaming storage.
            format: Output format (jsonl, csv).
            compression: Compression algorithm.
            chunk_size: Records per chunk.
            buffer_size: In-memory buffer size.
            **kwargs: Additional configuration options.
        """
        config = StreamingFileSystemConfig(
            base_path=base_path,
            format=format,
            compression=compression,
            chunk_size=chunk_size,
            buffer_size=buffer_size,
            **{k: v for k, v in kwargs.items() if hasattr(StreamingFileSystemConfig, k)},
        )
        super().__init__(config)
        self._base_path = Path(base_path)

    @classmethod
    def _default_config(cls) -> StreamingFileSystemConfig:
        """Create default configuration."""
        return StreamingFileSystemConfig()

    def _do_initialize(self) -> None:
        """Initialize the store directory."""
        if self._config.create_dirs:
            self._base_path.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Session Management
    # -------------------------------------------------------------------------

    def create_session(
        self,
        run_id: str,
        data_asset: str,
        metadata: dict[str, Any] | None = None,
    ) -> StreamSession:
        """Create a new streaming session.

        Args:
            run_id: Validation run identifier.
            data_asset: Name of the data asset.
            metadata: Optional session metadata.

        Returns:
            A new streaming session.
        """
        self.initialize()

        session_id = f"{run_id}_{uuid4().hex[:8]}"
        session = StreamSession(
            session_id=session_id,
            run_id=run_id,
            data_asset=data_asset,
            status=StreamStatus.PENDING,
            config=self._config,
            metadata=metadata or {},
        )

        # Create session directory
        session_path = self._base_path / run_id
        session_path.mkdir(parents=True, exist_ok=True)

        # Write initial manifest
        self._write_manifest(session)

        self._active_sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> StreamSession | None:
        """Get an existing session."""
        # Check active sessions
        if session_id in self._active_sessions:
            return self._active_sessions[session_id]

        # Try to load from disk
        for run_path in self._base_path.iterdir():
            if not run_path.is_dir():
                continue
            manifest_path = run_path / self._config.manifest_filename
            if manifest_path.exists():
                try:
                    with open(manifest_path) as f:
                        data = json.load(f)
                    session = StreamSession.from_dict(data)
                    if session.session_id == session_id:
                        return session
                except (json.JSONDecodeError, KeyError):
                    continue

        return None

    def resume_session(self, session_id: str) -> StreamSession:
        """Resume an interrupted session.

        Args:
            session_id: Session identifier.

        Returns:
            The resumed session.

        Raises:
            ValueError: If session cannot be resumed.
        """
        session = self.get_session(session_id)
        if session is None:
            raise ValueError(f"Session not found: {session_id}")

        if session.status == StreamStatus.COMPLETED:
            raise ValueError(f"Session already completed: {session_id}")

        if session.status == StreamStatus.FAILED:
            # Reset to active for retry
            session.status = StreamStatus.ACTIVE

        session.status = StreamStatus.ACTIVE
        session.updated_at = datetime.now()

        self._active_sessions[session_id] = session
        return session

    def _close_session(self, session: StreamSession) -> None:
        """Close and finalize a session."""
        if session.session_id in self._active_sessions:
            del self._active_sessions[session.session_id]
        session.status = StreamStatus.COMPLETED
        session.updated_at = datetime.now()
        # Only write manifest if directory exists
        session_path = self._base_path / session.run_id
        if session_path.exists():
            self._write_manifest(session)

    def _write_manifest(self, session: StreamSession) -> None:
        """Write session manifest to disk."""
        session_path = self._base_path / session.run_id
        manifest_path = session_path / self._config.manifest_filename

        manifest_data = session.to_dict()
        manifest_data["config"] = {
            "format": self._config.format.value,
            "compression": self._config.compression.value,
            "chunk_size": self._config.chunk_size,
            "buffer_size": self._config.buffer_size,
        }

        # Atomic write
        temp_path = manifest_path.with_suffix(".tmp")
        with open(temp_path, "w") as f:
            json.dump(manifest_data, f, indent=2, default=str)
        temp_path.rename(manifest_path)

    # -------------------------------------------------------------------------
    # Writer Operations
    # -------------------------------------------------------------------------

    def create_writer(self, session: StreamSession) -> StreamingResultWriter:
        """Create a writer for the session.

        Args:
            session: The streaming session.

        Returns:
            A streaming writer instance.
        """
        self.initialize()
        return StreamingResultWriter(
            session=session,
            config=self._config,
            base_path=self._base_path,
        )

    async def create_async_writer(self, session: StreamSession) -> AsyncStreamWriter:
        """Create an async writer for the session.

        Args:
            session: The streaming session.

        Returns:
            An async streaming writer instance.
        """
        writer = self.create_writer(session)
        return AsyncStreamWriter(writer)

    # -------------------------------------------------------------------------
    # Reader Operations
    # -------------------------------------------------------------------------

    def create_reader(self, run_id: str) -> StreamingResultReader:
        """Create a reader for a run's results.

        Args:
            run_id: The run ID to read.

        Returns:
            A streaming reader instance.
        """
        self.initialize()
        run_path = self._base_path / run_id
        return StreamingResultReader(run_path, self._config)

    async def create_async_reader(self, run_id: str) -> AsyncStreamReader:
        """Create an async reader for a run's results.

        Args:
            run_id: The run ID to read.

        Returns:
            An async streaming reader instance.
        """
        reader = self.create_reader(run_id)
        return AsyncStreamReader(reader)

    def iter_results(
        self,
        run_id: str,
        batch_size: int = 1000,
    ) -> Iterator["ValidatorResult"]:
        """Iterate over results for a run.

        Args:
            run_id: The run ID to iterate.
            batch_size: Number of records per batch.

        Yields:
            Individual validator results.
        """
        reader = self.create_reader(run_id)
        with reader:
            yield from reader.iter_results()

    async def aiter_results(
        self,
        run_id: str,
        batch_size: int = 1000,
    ) -> AsyncIterator["ValidatorResult"]:
        """Async iterate over results for a run.

        Args:
            run_id: The run ID to iterate.
            batch_size: Number of records per batch.

        Yields:
            Individual validator results.
        """
        reader = await self.create_async_reader(run_id)
        async with reader:
            async for result in reader.aiter_results():
                yield result

    # -------------------------------------------------------------------------
    # Chunk Management
    # -------------------------------------------------------------------------

    def list_chunks(self, run_id: str) -> list[ChunkInfo]:
        """List all chunks for a run.

        Args:
            run_id: The run ID.

        Returns:
            List of chunk information.
        """
        self.initialize()

        run_path = self._base_path / run_id
        if not run_path.exists():
            return []

        # Try to load from manifest
        manifest_path = run_path / self._config.manifest_filename
        if manifest_path.exists():
            try:
                with open(manifest_path) as f:
                    data = json.load(f)
                session = StreamSession.from_dict(data)
                return session.chunks
            except (json.JSONDecodeError, KeyError):
                pass

        # Fall back to discovery
        return self._discover_chunks(run_path)

    def _discover_chunks(self, run_path: Path) -> list[ChunkInfo]:
        """Discover chunks by scanning the directory."""
        chunks: list[ChunkInfo] = []

        patterns = ["*.jsonl", "*.jsonl.gz", "*.jsonl.zst", "*.jsonl.lz4"]
        chunk_files: list[Path] = []
        for pattern in patterns:
            chunk_files.extend(run_path.glob(pattern))

        chunk_files = [f for f in chunk_files if not f.name.startswith("_")]
        chunk_files.sort(key=lambda p: p.name)

        for idx, path in enumerate(chunk_files):
            byte_size = path.stat().st_size
            estimated_records = max(1, byte_size // 200)

            chunk_info = ChunkInfo(
                chunk_id=path.stem.split(".")[0],
                chunk_index=idx,
                record_count=estimated_records,
                byte_size=byte_size,
                start_offset=0,
                end_offset=estimated_records,
                path=str(path),
            )
            chunks.append(chunk_info)

        return chunks

    def get_chunk(self, chunk_info: ChunkInfo) -> list["ValidatorResult"]:
        """Get records from a specific chunk.

        Args:
            chunk_info: The chunk to retrieve.

        Returns:
            Records from the chunk.
        """
        from truthound.stores.results import ValidatorResult
        from truthound.stores.streaming.reader import get_decompressor, get_deserializer

        chunk_path = Path(chunk_info.path)
        if not chunk_path.exists():
            return []

        with open(chunk_path, "rb") as f:
            compressed_data = f.read()

        decompressor = get_decompressor(self._config.compression)
        deserializer = get_deserializer(self._config.format)

        data = decompressor.decompress(compressed_data)
        records = list(deserializer.deserialize(data))

        return [ValidatorResult.from_dict(r) for r in records]

    def delete_chunks(self, run_id: str) -> int:
        """Delete all chunks for a run.

        Args:
            run_id: The run ID.

        Returns:
            Number of chunks deleted.
        """
        self.initialize()

        run_path = self._base_path / run_id
        if not run_path.exists():
            return 0

        chunks = self.list_chunks(run_id)
        count = len(chunks)

        # Delete the entire run directory
        shutil.rmtree(run_path)

        return count

    # -------------------------------------------------------------------------
    # Validation Result Operations
    # -------------------------------------------------------------------------

    def stream_write_result(
        self,
        session: StreamSession,
        result: "ValidatorResult",
    ) -> None:
        """Write a single validator result to the stream.

        Args:
            session: The streaming session.
            result: The validator result to write.
        """
        if session.session_id not in self._active_sessions:
            raise ValueError(f"Session not active: {session.session_id}")

        # Get or create writer for session
        writer = self._get_or_create_writer(session)
        writer.write_result(result)

    def stream_write_batch(
        self,
        session: StreamSession,
        results: list["ValidatorResult"],
    ) -> None:
        """Write a batch of validator results to the stream.

        Args:
            session: The streaming session.
            results: The validator results to write.
        """
        if session.session_id not in self._active_sessions:
            raise ValueError(f"Session not active: {session.session_id}")

        writer = self._get_or_create_writer(session)
        writer.write_results(results)

    def _get_or_create_writer(self, session: StreamSession) -> StreamingResultWriter:
        """Get or create a writer for a session."""
        writer_key = f"_writer_{session.session_id}"
        if not hasattr(self, writer_key):
            writer = self.create_writer(session)
            setattr(self, writer_key, writer)
        return getattr(self, writer_key)

    def finalize_result(
        self,
        session: StreamSession,
        additional_metadata: dict[str, Any] | None = None,
    ) -> "ValidationResult":
        """Finalize the streaming session and create a ValidationResult.

        Args:
            session: The streaming session.
            additional_metadata: Optional additional metadata.

        Returns:
            The complete ValidationResult.
        """
        from truthound.stores.results import (
            ResultStatistics,
            ResultStatus,
            ValidationResult,
        )

        # Close any active writer
        writer_key = f"_writer_{session.session_id}"
        if hasattr(self, writer_key):
            writer = getattr(self, writer_key)
            writer.close()
            delattr(self, writer_key)

        # Aggregate statistics from chunks
        total_validators = 0
        passed_validators = 0
        failed_validators = 0
        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}

        # Read all results to compute statistics
        all_results: list["ValidatorResult"] = []
        for result in self.iter_results(session.run_id):
            all_results.append(result)
            total_validators += 1
            if result.success:
                passed_validators += 1
            else:
                failed_validators += 1
                if result.severity and result.severity in severity_counts:
                    severity_counts[result.severity] += 1

        # Determine status
        if severity_counts["critical"] > 0:
            status = ResultStatus.FAILURE
        elif failed_validators > 0:
            status = ResultStatus.WARNING
        else:
            status = ResultStatus.SUCCESS

        statistics = ResultStatistics(
            total_validators=total_validators,
            passed_validators=passed_validators,
            failed_validators=failed_validators,
            total_issues=failed_validators,
            critical_issues=severity_counts["critical"],
            high_issues=severity_counts["high"],
            medium_issues=severity_counts["medium"],
            low_issues=severity_counts["low"],
            execution_time_ms=session.metrics.average_throughput,
        )

        # Create final result
        metadata = session.metadata.copy()
        if additional_metadata:
            metadata.update(additional_metadata)
        metadata["streaming"] = {
            "chunks": len(session.chunks),
            "total_records": session.metrics.records_written,
            "bytes_written": session.metrics.bytes_written,
            "average_throughput": session.metrics.average_throughput,
        }

        result = ValidationResult(
            run_id=session.run_id,
            run_time=session.started_at,
            data_asset=session.data_asset,
            status=status,
            results=all_results,
            statistics=statistics,
            metadata=metadata,
        )

        # Mark session complete
        self._close_session(session)

        return result

    def get_streaming_stats(self, run_id: str) -> dict[str, Any]:
        """Get statistics about a streaming run.

        Args:
            run_id: The run ID.

        Returns:
            Statistics dictionary.
        """
        self.initialize()

        run_path = self._base_path / run_id
        manifest_path = run_path / self._config.manifest_filename

        if not manifest_path.exists():
            return {}

        with open(manifest_path) as f:
            data = json.load(f)

        session = StreamSession.from_dict(data)

        return {
            "run_id": run_id,
            "data_asset": session.data_asset,
            "status": session.status.value,
            "chunks": len(session.chunks),
            "total_records": session.metrics.records_written,
            "bytes_written": session.metrics.bytes_written,
            "flush_count": session.metrics.flush_count,
            "errors": session.metrics.errors,
            "started_at": session.started_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "average_throughput": session.metrics.average_throughput,
        }

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def list_runs(self) -> list[str]:
        """List all run IDs in the store.

        Returns:
            List of run IDs.
        """
        self.initialize()

        runs: list[str] = []
        for path in self._base_path.iterdir():
            if path.is_dir() and not path.name.startswith("_"):
                runs.append(path.name)

        return sorted(runs)

    def cleanup_incomplete_sessions(self) -> int:
        """Clean up incomplete or failed sessions.

        Returns:
            Number of sessions cleaned up.
        """
        self.initialize()

        cleaned = 0
        for run_id in self.list_runs():
            run_path = self._base_path / run_id
            manifest_path = run_path / self._config.manifest_filename

            if not manifest_path.exists():
                # No manifest - remove
                shutil.rmtree(run_path)
                cleaned += 1
                continue

            try:
                with open(manifest_path) as f:
                    data = json.load(f)
                session = StreamSession.from_dict(data)

                if session.status in (StreamStatus.FAILED, StreamStatus.ABORTED):
                    shutil.rmtree(run_path)
                    cleaned += 1
            except (json.JSONDecodeError, KeyError):
                shutil.rmtree(run_path)
                cleaned += 1

        return cleaned

    def get_storage_size(self, run_id: str | None = None) -> int:
        """Get total storage size in bytes.

        Args:
            run_id: Optional run ID to get size for specific run.

        Returns:
            Total size in bytes.
        """
        self.initialize()

        if run_id:
            run_path = self._base_path / run_id
            if not run_path.exists():
                return 0
            return sum(f.stat().st_size for f in run_path.rglob("*") if f.is_file())

        return sum(f.stat().st_size for f in self._base_path.rglob("*") if f.is_file())
