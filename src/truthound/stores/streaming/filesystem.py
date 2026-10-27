"""Streaming filesystem store implementation.

This module provides a streaming-capable filesystem store that uses JSONL
(JSON Lines) format for efficient streaming of large validation results.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import AsyncIterator, Iterator  # noqa: TC003 - public annotation introspection
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from truthound.stores.streaming.base import (
    ChunkInfo,
    CompressionType,
    StreamingConfig,
    StreamingFormat,
    StreamingValidationStore,
    StreamSession,
    StreamStatus,
)
from truthound.stores.streaming.reader import (
    AsyncStreamReader,
    StreamingResultReader,
    _validator_result,
)
from truthound.stores.streaming.writer import (
    AsyncStreamWriter,
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


class StreamingRecoveryRequired(RuntimeError):
    """Stored bytes and the durable stream manifest require explicit reconciliation."""


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

    def _run_path(self, run_id):
        if not isinstance(run_id, str) or not run_id or run_id in {".", ".."} or any(value in run_id for value in ("/", "\\", "\x00")):
            raise ValueError("A native stream run identifier must be one path component")
        result = self._base_path / run_id
        if result.is_symlink() or result.resolve().parent != self._base_path.resolve():
            raise ValueError("Native stream run must remain inside the selected store")
        return result

    def _manifest_path(self, run_id):
        name = self._config.manifest_filename
        if not name or name in {".", ".."} or any(value in name for value in ("/", "\\", "\x00")):
            raise ValueError("A native stream manifest must be one path component")
        return self._run_path(run_id) / name

    def get_session_for_run(self, run_id: str) -> StreamSession | None:
        path = self._manifest_path(run_id)
        self._guard_io()
        if not path.exists():
            return None
        if path.is_symlink():
            raise StreamingRecoveryRequired("Native stream manifest requires reconciliation")
        with path.open(encoding="utf-8") as stream:
            session = StreamSession.from_dict(json.load(stream))
        if session.run_id != run_id:
            raise StreamingRecoveryRequired("Native stream identity requires reconciliation")
        return session

    def list_sessions(self, limit: int = 100, offset: int = 0) -> list[StreamSession]:
        if limit < 1 or offset < 0:
            raise ValueError("Invalid native streaming session page")
        self.initialize()
        result = []
        for run_id in sorted(self.list_runs()):
            self._guard_io()
            session = self.get_session_for_run(run_id)
            if session is None:
                raise StreamingRecoveryRequired("Native stream manifest is missing; reconciliation is required")
            result.append(session)
            if len(result) >= offset + limit:
                break
        return result[offset:]

    def _validate_session_chunks(self, session):
        from truthound.stores.streaming.reader import get_decompressor, get_deserializer

        run_path = self._run_path(session.run_id)
        expected, offset = set(), 0
        maximum = session.config.max_memory_mb * 1024 ** 2
        for index, chunk in enumerate(session.chunks):
            self._guard_io()
            path = Path(chunk.path)
            if not path.is_absolute():
                path = run_path / path.name
            if path.is_symlink() or path.resolve().parent != run_path.resolve() or not path.is_file():
                raise StreamingRecoveryRequired("Native stream chunk requires reconciliation")
            if chunk.chunk_index != index or chunk.start_offset != offset or chunk.end_offset != offset + chunk.record_count:
                raise StreamingRecoveryRequired("Native stream offsets require reconciliation")
            if path.stat().st_size != chunk.byte_size or chunk.byte_size > maximum:
                raise StreamingRecoveryRequired("Native stream size requires reconciliation")
            with path.open("rb") as stream:
                encoded = stream.read(maximum + 1)
            decoded = get_decompressor(session.config.compression).decompress(encoded)
            if chunk.checksum and hashlib.md5(decoded).hexdigest() != chunk.checksum:
                raise StreamingRecoveryRequired("Native stream checksum requires reconciliation")
            if len(list(get_deserializer(session.config.format).deserialize(decoded))) != chunk.record_count:
                raise StreamingRecoveryRequired("Native stream record count requires reconciliation")
            expected.add(path.name)
            offset = chunk.end_offset
        actual = {path.name for path in run_path.iterdir() if "_chunk_" in path.name}
        if actual != expected:
            raise StreamingRecoveryRequired("Unconfirmed native stream chunks require reconciliation")

    def recovery_status(self, run_id: str) -> dict[str, Any]:
        """Inspect exact retained evidence while no live writer owns the run."""
        from truthound.stores.streaming.recovery import status

        return status(self, run_id)

    def reconcile_stream(self, run_id: str, *, resolution: str, expected_fingerprint: str) -> dict[str, Any]:
        """Explicitly adopt or discard only unconfirmed chunks of a valid prefix."""
        from truthound.stores.streaming.recovery import reconcile

        return reconcile(self, run_id, resolution=resolution, expected_fingerprint=expected_fingerprint)

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
        session_path = self._run_path(run_id)
        self._manifest_path(run_id)
        self._guard_io()
        self.initialize()
        if session_path.exists():
            raise ValueError("A native stream run already exists")

        session_id = f"{run_id}_{uuid4().hex[:8]}"
        session = StreamSession(
            session_id=session_id,
            run_id=run_id,
            data_asset=data_asset,
            status=StreamStatus.PENDING,
            config=self._config,
            metadata=metadata or {},
        )

        from truthound.stores.streaming._filesystem_io import StreamFileLease, run_lease_path

        with StreamFileLease(run_lease_path(self._base_path, run_id)):
            self._guard_io()
            session_path.mkdir(exist_ok=False)
            self._write_manifest(session)

        self._active_sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> StreamSession | None:
        """Get an existing session."""
        # Check active sessions
        if session_id in self._active_sessions:
            return self._active_sessions[session_id]

        self.initialize()
        for run_id in sorted(self.list_runs()):
            session = self.get_session_for_run(run_id)
            if session is not None and session.session_id == session_id:
                return session

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

        from truthound.stores.streaming._filesystem_io import StreamFileLease, run_lease_path

        with StreamFileLease(run_lease_path(self._base_path, session.run_id)):
            session = self.get_session_for_run(session.run_id)
            if session is None or session.session_id != session_id:
                raise StreamingRecoveryRequired("Native stream identity requires reconciliation")
            self._validate_session_chunks(session)
            if session.status == StreamStatus.COMPLETED:
                raise ValueError("Session already completed")
            if session.status == StreamStatus.ABORTED:
                raise ValueError("An aborted native stream cannot be resumed")
            session.status = StreamStatus.ACTIVE
            session.updated_at = datetime.now()
            self._write_manifest(session)
            self._active_sessions[session_id] = session
        return session

    def _close_session(self, session: StreamSession) -> None:
        """Close and finalize a session."""
        writer = self._writers.get(session.session_id)
        if writer is not None and writer._closed:
            self._active_sessions.pop(session.session_id, None)
            return
        if session.session_id in self._active_sessions:
            del self._active_sessions[session.session_id]
        if session.status not in {StreamStatus.FAILED, StreamStatus.ABORTED}:
            session.status = StreamStatus.COMPLETED
        session.updated_at = datetime.now()
        # Only write manifest if directory exists
        session_path = self._run_path(session.run_id)
        if session_path.exists():
            self._write_manifest(session)

    def _write_manifest(self, session: StreamSession) -> None:
        """Write session manifest to disk."""
        self._guard_io()
        manifest_path = self._manifest_path(session.run_id)

        manifest_data = session.to_dict()
        manifest_data["config"] = session.config.to_dict()

        from truthound.stores.streaming._filesystem_io import publish

        publish(manifest_path, json.dumps(manifest_data, indent=2, default=str).encode("utf-8"))

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
        self._assert_writer_available(session)
        return self._track_writer(session, StreamingResultWriter(
            session=session,
            config=session.config,
            base_path=self._base_path,
            manifest_filename=self._config.manifest_filename,
            require_manifest=True,
        ))

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
        run_path = self._run_path(run_id)
        session = self.get_session_for_run(run_id)
        if session is None:
            raise StreamingRecoveryRequired("Native stream manifest is missing; reconciliation is required")
        self._validate_session_chunks(session)
        return self._track_reader(StreamingResultReader(run_path, manifest_filename=self._config.manifest_filename))

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
    ) -> Iterator[ValidatorResult]:
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
    ) -> AsyncIterator[ValidatorResult]:
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

        session = self.get_session_for_run(run_id)
        if session is None:
            if self._run_path(run_id).exists():
                raise StreamingRecoveryRequired("Native stream manifest is missing; reconciliation is required")
            return []
        return session.chunks

    def get_chunk(self, chunk_info: ChunkInfo) -> list[ValidatorResult]:
        """Get records from a specific chunk.

        Args:
            chunk_info: The chunk to retrieve.

        Returns:
            Records from the chunk.
        """
        from truthound.stores.streaming._filesystem_io import contained_path
        from truthound.stores.streaming.reader import get_decompressor, get_deserializer

        self._guard_io()
        chunk_path = Path(chunk_info.path).absolute()
        run_path = self._run_path(chunk_path.parent.name).absolute()
        chunk_path = contained_path(run_path, chunk_path)
        session = self.get_session_for_run(run_path.name)
        if session is None or chunk_info.to_dict() not in [item.to_dict() for item in session.chunks]:
            raise StreamingRecoveryRequired("Native stream chunk is not confirmed by its manifest")
        with chunk_path.open("rb") as stream:
            compressed_data = stream.read(session.config.max_memory_mb * 1024 ** 2 + 1)
        if len(compressed_data) > session.config.max_memory_mb * 1024 ** 2:
            raise StreamingRecoveryRequired("Native stream chunk exceeds its configured memory bound")
        data = get_decompressor(session.config.compression).decompress(compressed_data)
        if chunk_info.checksum and hashlib.md5(data).hexdigest() != chunk_info.checksum:
            raise StreamingRecoveryRequired("Native stream checksum requires reconciliation")
        records = list(get_deserializer(session.config.format).deserialize(data))
        if len(records) != chunk_info.record_count:
            raise StreamingRecoveryRequired("Native stream record count requires reconciliation")
        return [_validator_result(record) for record in records]

    def delete_chunks(self, run_id: str) -> int:
        """Delete all chunks for a run.

        Args:
            run_id: The run ID.

        Returns:
            Number of chunks deleted.
        """
        self.initialize()

        run_path = self._run_path(run_id)
        if not run_path.exists():
            return 0

        from truthound.stores.streaming._filesystem_io import StreamFileLease, run_lease_path

        with StreamFileLease(run_lease_path(self._base_path, run_id)):
            self._guard_io()
            chunks = self.list_chunks(run_id)
            shutil.rmtree(run_path)
            self._active_sessions = {key: value for key, value in self._active_sessions.items() if value.run_id != run_id}
            return len(chunks)

    # -------------------------------------------------------------------------
    # Validation Result Operations
    # -------------------------------------------------------------------------

    def stream_write_result(
        self,
        session: StreamSession,
        result: ValidatorResult,
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
        results: list[ValidatorResult],
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
        writer = self._writers.get(session.session_id)
        if writer is None or writer._closed:
            writer = self.create_writer(session)
        return writer

    def finalize_result(
        self,
        session: StreamSession,
        additional_metadata: dict[str, Any] | None = None,
    ) -> ValidationResult:
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

        if session.status in {StreamStatus.FAILED, StreamStatus.ABORTED}:
            raise StreamingRecoveryRequired("A failed native stream cannot produce a completed result")

        # Include writers created directly as well as stream_write_* helpers.
        writer = self._writers.get(session.session_id)
        if writer is not None:
            writer.close()
        self._validate_session_chunks(session)


        # Aggregate statistics from chunks
        total_validators = 0
        passed_validators = 0
        failed_validators = 0
        total_issues = 0
        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}

        # Read all results to compute statistics
        all_results: list[ValidatorResult] = []
        for result in self.iter_results(session.run_id):
            all_results.append(result)
            total_validators += 1
            if result.success:
                passed_validators += 1
            else:
                failed_validators += 1
                total_issues += result.count
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
            total_issues=total_issues,
            critical_issues=severity_counts["critical"],
            high_issues=severity_counts["high"],
            medium_issues=severity_counts["medium"],
            low_issues=severity_counts["low"],
            execution_time_ms=(session.metrics.end_time - session.metrics.start_time).total_seconds() * 1000
            if session.metrics.end_time is not None and session.metrics.start_time is not None else 0,
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

        run_path = self._run_path(run_id)
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
            if path.is_dir() and not path.is_symlink() and not path.name.startswith("_"):
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
            run_path = self._run_path(run_id)
            manifest_path = run_path / self._config.manifest_filename

            if not manifest_path.exists():
                raise StreamingRecoveryRequired("Native stream manifest is missing; explicit reconciliation is required")

            try:
                with open(manifest_path) as f:
                    data = json.load(f)
                session = StreamSession.from_dict(data)

                if session.status in (StreamStatus.FAILED, StreamStatus.ABORTED):
                    self.delete_chunks(run_id)
                    cleaned += 1
            except (json.JSONDecodeError, KeyError) as error:
                raise StreamingRecoveryRequired("Corrupt native stream evidence requires explicit reconciliation") from error

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
            run_path = self._run_path(run_id)
            if not run_path.exists():
                return 0
            return sum(f.stat().st_size for f in run_path.rglob("*") if f.is_file())

        return sum(f.stat().st_size for f in self._base_path.rglob("*") if f.is_file())
