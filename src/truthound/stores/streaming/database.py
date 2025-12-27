"""Streaming database store implementation with cursor-based iteration.

This module provides a streaming-capable database store that uses server-side
cursors for efficient handling of large validation results.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, AsyncIterator, Iterator
from uuid import uuid4

from truthound.stores.streaming.base import (
    ChunkInfo,
    CompressionType,
    StreamingConfig,
    StreamingFormat,
    StreamingMetrics,
    StreamingValidationStore,
    StreamSession,
    StreamStatus,
)
from truthound.stores.streaming.reader import (
    AsyncStreamReader,
    BaseStreamReader,
    get_decompressor,
    get_deserializer,
)
from truthound.stores.streaming.writer import (
    AsyncStreamWriter,
    BaseStreamWriter,
    get_compressor,
    get_serializer,
)

if TYPE_CHECKING:
    from truthound.stores.results import ValidationResult, ValidatorResult


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class StreamingDatabaseConfig(StreamingConfig):
    """Configuration for streaming database store.

    Attributes:
        connection_url: SQLAlchemy connection URL.
        table_prefix: Prefix for table names.
        pool_size: Connection pool size.
        max_overflow: Maximum pool overflow connections.
        use_server_cursor: Use server-side cursors for reads.
        cursor_fetch_size: Number of rows to fetch per cursor iteration.
        batch_insert_size: Number of rows per batch insert.
    """

    connection_url: str = ""
    table_prefix: str = "truthound_streaming_"
    pool_size: int = 5
    max_overflow: int = 10
    use_server_cursor: bool = True
    cursor_fetch_size: int = 1000
    batch_insert_size: int = 1000

    def validate(self) -> None:
        """Validate configuration."""
        super().validate()
        if not self.connection_url:
            raise ValueError("Database connection URL is required")


# =============================================================================
# Database Streaming Writer
# =============================================================================


class DatabaseStreamWriter(BaseStreamWriter):
    """Database streaming writer with batch inserts.

    Uses batch INSERT statements for efficient streaming writes.
    """

    def __init__(
        self,
        session: StreamSession,
        config: StreamingDatabaseConfig,
        engine: Any,
        session_maker: Any,
        results_table: Any,
    ):
        """Initialize the database writer.

        Args:
            session: The streaming session.
            config: Database streaming configuration.
            engine: SQLAlchemy engine.
            session_maker: SQLAlchemy session maker.
            results_table: Results table object.
        """
        super().__init__(session, config)
        self.db_config = config
        self._engine = engine
        self._session_maker = session_maker
        self._results_table = results_table
        self._db_session: Any = None
        self._pending_rows: list[dict[str, Any]] = []

    def _get_db_session(self) -> Any:
        """Get or create database session."""
        if self._db_session is None:
            self._db_session = self._session_maker()
        return self._db_session

    def _write_chunk(self, chunk_info: ChunkInfo, data: bytes) -> None:
        """Write chunk data to database."""
        # For database, we don't write chunks as blobs
        # Instead, we write individual records
        # The 'data' is compressed serialized records
        decompressor = get_decompressor(self.config.compression)
        deserializer = get_deserializer(self.config.format)

        raw_data = decompressor.decompress(data)
        records = list(deserializer.deserialize(raw_data))

        # Batch insert records
        db_session = self._get_db_session()
        try:
            for i in range(0, len(records), self.db_config.batch_insert_size):
                batch = records[i : i + self.db_config.batch_insert_size]
                rows = [
                    {
                        "run_id": self.session.run_id,
                        "chunk_id": chunk_info.chunk_id,
                        "chunk_index": chunk_info.chunk_index,
                        "record_index": i + j,
                        "data_json": json.dumps(record, default=str),
                        "created_at": datetime.utcnow(),
                    }
                    for j, record in enumerate(batch)
                ]
                db_session.execute(self._results_table.insert(), rows)
            db_session.commit()
        except Exception:
            db_session.rollback()
            raise

        chunk_info.path = f"db://{self.session.run_id}/{chunk_info.chunk_id}"

    def _write_session_state(self) -> None:
        """Write session state to database."""
        # Session state is stored in a separate sessions table
        pass

    def _finalize(self) -> None:
        """Finalize the stream."""
        if self._db_session:
            self._db_session.close()
            self._db_session = None


# =============================================================================
# Database Streaming Reader
# =============================================================================


class DatabaseStreamReader(BaseStreamReader):
    """Database streaming reader with server-side cursors.

    Uses server-side cursors for memory-efficient iteration over large results.
    """

    def __init__(
        self,
        run_id: str,
        engine: Any,
        config: StreamingDatabaseConfig,
        results_table: Any,
    ):
        """Initialize the database reader.

        Args:
            run_id: The run ID to read.
            engine: SQLAlchemy engine.
            config: Database streaming configuration.
            results_table: Results table object.
        """
        self._run_id = run_id
        self._engine = engine
        self._db_config = config
        self._results_table = results_table
        self._connection: Any = None
        self._cursor: Any = None
        self._chunks: list[ChunkInfo] = []
        self._current_rows: list[Any] = []
        self._row_index = 0
        self._exhausted = False

        # Initialize chunks from database
        self._load_chunks()

        super().__init__(config)

    def _load_chunks(self) -> None:
        """Load chunk information from database."""
        from sqlalchemy import text

        with self._engine.connect() as conn:
            result = conn.execute(
                text(
                    f"""
                    SELECT DISTINCT chunk_id, chunk_index, COUNT(*) as record_count
                    FROM {self._results_table.name}
                    WHERE run_id = :run_id
                    GROUP BY chunk_id, chunk_index
                    ORDER BY chunk_index
                    """
                ),
                {"run_id": self._run_id},
            )

            for row in result:
                chunk_info = ChunkInfo(
                    chunk_id=row.chunk_id,
                    chunk_index=row.chunk_index,
                    record_count=row.record_count,
                    byte_size=0,
                    start_offset=0,
                    end_offset=row.record_count,
                    path=f"db://{self._run_id}/{row.chunk_id}",
                )
                self._chunks.append(chunk_info)

    def _get_chunks(self) -> list[ChunkInfo]:
        """Get list of chunks."""
        return self._chunks

    def _read_chunk(self, chunk_info: ChunkInfo) -> bytes:
        """Read a chunk from database.

        Note: For database reader, we use cursor-based reading instead.
        This method is not typically called.
        """
        from sqlalchemy import text

        with self._engine.connect() as conn:
            result = conn.execute(
                text(
                    f"""
                    SELECT data_json
                    FROM {self._results_table.name}
                    WHERE run_id = :run_id AND chunk_id = :chunk_id
                    ORDER BY record_index
                    """
                ),
                {"run_id": self._run_id, "chunk_id": chunk_info.chunk_id},
            )

            records = [json.loads(row.data_json) for row in result]
            serializer = get_serializer(self.config.format)
            return serializer.serialize_batch(records)

    def read(self) -> dict[str, Any] | None:
        """Read a single record using cursor."""
        if self._exhausted:
            return None

        # Use cursor-based reading for efficiency
        if not self._current_rows or self._row_index >= len(self._current_rows):
            self._fetch_next_batch()
            if not self._current_rows:
                self._exhausted = True
                return None

        row = self._current_rows[self._row_index]
        self._row_index += 1

        return json.loads(row.data_json) if hasattr(row, "data_json") else row

    def _fetch_next_batch(self) -> None:
        """Fetch next batch of rows using server-side cursor."""
        from sqlalchemy import text

        if self._connection is None:
            self._connection = self._engine.connect()

            # Use server-side cursor if supported
            if self._db_config.use_server_cursor:
                # PostgreSQL: use stream_results
                # MySQL: use server_side_cursors
                execution_options = {"stream_results": True}
                self._cursor = self._connection.execution_options(
                    **execution_options
                ).execute(
                    text(
                        f"""
                        SELECT data_json
                        FROM {self._results_table.name}
                        WHERE run_id = :run_id
                        ORDER BY chunk_index, record_index
                        """
                    ),
                    {"run_id": self._run_id},
                )
            else:
                self._cursor = self._connection.execute(
                    text(
                        f"""
                        SELECT data_json
                        FROM {self._results_table.name}
                        WHERE run_id = :run_id
                        ORDER BY chunk_index, record_index
                        """
                    ),
                    {"run_id": self._run_id},
                )

        # Fetch next batch
        self._current_rows = list(
            self._cursor.fetchmany(self._db_config.cursor_fetch_size)
        )
        self._row_index = 0

        if not self._current_rows:
            self._exhausted = True

    def close(self) -> None:
        """Close the reader and database connection."""
        super().close()
        if self._cursor is not None:
            self._cursor.close()
            self._cursor = None
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def reset(self) -> None:
        """Reset reader to beginning."""
        super().reset()
        if self._cursor is not None:
            self._cursor.close()
            self._cursor = None
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        self._current_rows = []
        self._row_index = 0
        self._exhausted = False


# =============================================================================
# Streaming Database Store
# =============================================================================


class StreamingDatabaseStore(StreamingValidationStore[StreamingDatabaseConfig]):
    """Streaming database store with cursor-based iteration.

    This store is optimized for handling large validation results in databases:

    - Server-side cursors for memory-efficient reads
    - Batch inserts for efficient writes
    - Transaction management
    - Connection pooling

    Example:
        >>> store = StreamingDatabaseStore(
        ...     connection_url="postgresql://user:pass@localhost/db",
        ... )
        >>>
        >>> session = store.create_session("run_001", "large_dataset.csv")
        >>> with store.create_writer(session) as writer:
        ...     for result in validation_results:
        ...         writer.write_result(result)
        >>>
        >>> # Efficiently iterate over results with cursor
        >>> for result in store.iter_results("run_001"):
        ...     process(result)
    """

    def __init__(
        self,
        connection_url: str,
        table_prefix: str = "truthound_streaming_",
        pool_size: int = 5,
        max_overflow: int = 10,
        use_server_cursor: bool = True,
        cursor_fetch_size: int = 1000,
        **kwargs: Any,
    ):
        """Initialize the streaming database store.

        Args:
            connection_url: SQLAlchemy connection URL.
            table_prefix: Prefix for table names.
            pool_size: Connection pool size.
            max_overflow: Maximum pool overflow connections.
            use_server_cursor: Use server-side cursors for reads.
            cursor_fetch_size: Rows per cursor fetch.
            **kwargs: Additional configuration options.
        """
        config = StreamingDatabaseConfig(
            connection_url=connection_url,
            table_prefix=table_prefix,
            pool_size=pool_size,
            max_overflow=max_overflow,
            use_server_cursor=use_server_cursor,
            cursor_fetch_size=cursor_fetch_size,
            **{k: v for k, v in kwargs.items() if hasattr(StreamingDatabaseConfig, k)},
        )
        super().__init__(config)

        self._engine: Any = None
        self._session_maker: Any = None
        self._metadata: Any = None
        self._results_table: Any = None
        self._sessions_table: Any = None

    @classmethod
    def _default_config(cls) -> StreamingDatabaseConfig:
        """Create default configuration."""
        return StreamingDatabaseConfig()

    def _do_initialize(self) -> None:
        """Initialize database engine and create tables."""
        try:
            from sqlalchemy import (
                Column,
                DateTime,
                Integer,
                MetaData,
                String,
                Table,
                Text,
                create_engine,
            )
            from sqlalchemy.orm import sessionmaker
        except ImportError:
            raise ImportError("sqlalchemy library required for database streaming store")

        # Create engine
        self._engine = create_engine(
            self._config.connection_url,
            pool_size=self._config.pool_size,
            max_overflow=self._config.max_overflow,
            pool_pre_ping=True,
        )

        self._session_maker = sessionmaker(bind=self._engine)
        self._metadata = MetaData()

        # Define results table
        self._results_table = Table(
            f"{self._config.table_prefix}results",
            self._metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("run_id", String(255), nullable=False, index=True),
            Column("chunk_id", String(255), nullable=False, index=True),
            Column("chunk_index", Integer, nullable=False),
            Column("record_index", Integer, nullable=False),
            Column("data_json", Text, nullable=False),
            Column("created_at", DateTime, nullable=False),
        )

        # Define sessions table
        self._sessions_table = Table(
            f"{self._config.table_prefix}sessions",
            self._metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("session_id", String(255), unique=True, nullable=False, index=True),
            Column("run_id", String(255), nullable=False, index=True),
            Column("data_asset", String(500), nullable=False),
            Column("status", String(50), nullable=False),
            Column("metadata_json", Text),
            Column("metrics_json", Text),
            Column("chunks_json", Text),
            Column("started_at", DateTime, nullable=False),
            Column("updated_at", DateTime, nullable=False),
            Column("checkpoint_offset", Integer, default=0),
        )

        # Create tables
        self._metadata.create_all(self._engine)

    def close(self) -> None:
        """Close the store and database connections."""
        super().close()
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None

    # -------------------------------------------------------------------------
    # Session Management
    # -------------------------------------------------------------------------

    def create_session(
        self,
        run_id: str,
        data_asset: str,
        metadata: dict[str, Any] | None = None,
    ) -> StreamSession:
        """Create a new streaming session."""
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

        # Save to database
        self._save_session(session)
        self._active_sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> StreamSession | None:
        """Get an existing session."""
        if session_id in self._active_sessions:
            return self._active_sessions[session_id]

        # Load from database
        self.initialize()
        from sqlalchemy import text

        with self._engine.connect() as conn:
            result = conn.execute(
                text(
                    f"""
                    SELECT * FROM {self._sessions_table.name}
                    WHERE session_id = :session_id
                    """
                ),
                {"session_id": session_id},
            )
            row = result.fetchone()

            if row is None:
                return None

            return StreamSession(
                session_id=row.session_id,
                run_id=row.run_id,
                data_asset=row.data_asset,
                status=StreamStatus(row.status),
                metadata=json.loads(row.metadata_json) if row.metadata_json else {},
                metrics=StreamingMetrics(**json.loads(row.metrics_json))
                if row.metrics_json
                else StreamingMetrics(),
                chunks=[ChunkInfo.from_dict(c) for c in json.loads(row.chunks_json)]
                if row.chunks_json
                else [],
                started_at=row.started_at,
                updated_at=row.updated_at,
                checkpoint_offset=row.checkpoint_offset,
            )

    def resume_session(self, session_id: str) -> StreamSession:
        """Resume an interrupted session."""
        session = self.get_session(session_id)
        if session is None:
            raise ValueError(f"Session not found: {session_id}")

        if session.status == StreamStatus.COMPLETED:
            raise ValueError(f"Session already completed: {session_id}")

        session.status = StreamStatus.ACTIVE
        session.updated_at = datetime.now()
        self._save_session(session)

        self._active_sessions[session_id] = session
        return session

    def _close_session(self, session: StreamSession) -> None:
        """Close and finalize a session."""
        if session.session_id in self._active_sessions:
            del self._active_sessions[session.session_id]
        session.status = StreamStatus.COMPLETED
        session.updated_at = datetime.now()
        self._save_session(session)

    def _save_session(self, session: StreamSession) -> None:
        """Save session to database."""
        from sqlalchemy import text

        db_session = self._session_maker()
        try:
            # Check if exists
            result = db_session.execute(
                text(
                    f"""
                    SELECT id FROM {self._sessions_table.name}
                    WHERE session_id = :session_id
                    """
                ),
                {"session_id": session.session_id},
            )
            existing = result.fetchone()

            if existing:
                # Update
                db_session.execute(
                    text(
                        f"""
                        UPDATE {self._sessions_table.name}
                        SET status = :status,
                            metadata_json = :metadata_json,
                            metrics_json = :metrics_json,
                            chunks_json = :chunks_json,
                            updated_at = :updated_at,
                            checkpoint_offset = :checkpoint_offset
                        WHERE session_id = :session_id
                        """
                    ),
                    {
                        "session_id": session.session_id,
                        "status": session.status.value,
                        "metadata_json": json.dumps(session.metadata, default=str),
                        "metrics_json": json.dumps(session.metrics.to_dict()),
                        "chunks_json": json.dumps(
                            [c.to_dict() for c in session.chunks]
                        ),
                        "updated_at": session.updated_at,
                        "checkpoint_offset": session.checkpoint_offset,
                    },
                )
            else:
                # Insert
                db_session.execute(
                    self._sessions_table.insert().values(
                        session_id=session.session_id,
                        run_id=session.run_id,
                        data_asset=session.data_asset,
                        status=session.status.value,
                        metadata_json=json.dumps(session.metadata, default=str),
                        metrics_json=json.dumps(session.metrics.to_dict()),
                        chunks_json=json.dumps([c.to_dict() for c in session.chunks]),
                        started_at=session.started_at,
                        updated_at=session.updated_at,
                        checkpoint_offset=session.checkpoint_offset,
                    )
                )
            db_session.commit()
        except Exception:
            db_session.rollback()
            raise
        finally:
            db_session.close()

    # -------------------------------------------------------------------------
    # Writer Operations
    # -------------------------------------------------------------------------

    def create_writer(self, session: StreamSession) -> DatabaseStreamWriter:
        """Create a writer for the session."""
        self.initialize()
        return DatabaseStreamWriter(
            session=session,
            config=self._config,
            engine=self._engine,
            session_maker=self._session_maker,
            results_table=self._results_table,
        )

    async def create_async_writer(self, session: StreamSession) -> AsyncStreamWriter:
        """Create an async writer for the session."""
        writer = self.create_writer(session)
        return AsyncStreamWriter(writer)

    # -------------------------------------------------------------------------
    # Reader Operations
    # -------------------------------------------------------------------------

    def create_reader(self, run_id: str) -> DatabaseStreamReader:
        """Create a reader for a run's results."""
        self.initialize()
        return DatabaseStreamReader(
            run_id=run_id,
            engine=self._engine,
            config=self._config,
            results_table=self._results_table,
        )

    async def create_async_reader(self, run_id: str) -> AsyncStreamReader:
        """Create an async reader for a run's results."""
        reader = self.create_reader(run_id)
        return AsyncStreamReader(reader)

    def iter_results(
        self,
        run_id: str,
        batch_size: int = 1000,
    ) -> Iterator["ValidatorResult"]:
        """Iterate over results for a run using cursor."""
        reader = self.create_reader(run_id)
        with reader:
            yield from reader.iter_results()

    async def aiter_results(
        self,
        run_id: str,
        batch_size: int = 1000,
    ) -> AsyncIterator["ValidatorResult"]:
        """Async iterate over results for a run."""
        reader = await self.create_async_reader(run_id)
        async with reader:
            async for result in reader.aiter_results():
                yield result

    # -------------------------------------------------------------------------
    # Chunk Management
    # -------------------------------------------------------------------------

    def list_chunks(self, run_id: str) -> list[ChunkInfo]:
        """List all chunks for a run."""
        self.initialize()
        reader = self.create_reader(run_id)
        return reader._chunks

    def get_chunk(self, chunk_info: ChunkInfo) -> list["ValidatorResult"]:
        """Get records from a specific chunk."""
        from truthound.stores.results import ValidatorResult
        from sqlalchemy import text

        run_id = chunk_info.path.replace("db://", "").split("/")[0]

        with self._engine.connect() as conn:
            result = conn.execute(
                text(
                    f"""
                    SELECT data_json
                    FROM {self._results_table.name}
                    WHERE run_id = :run_id AND chunk_id = :chunk_id
                    ORDER BY record_index
                    """
                ),
                {"run_id": run_id, "chunk_id": chunk_info.chunk_id},
            )

            return [ValidatorResult.from_dict(json.loads(row.data_json)) for row in result]

    def delete_chunks(self, run_id: str) -> int:
        """Delete all chunks for a run."""
        self.initialize()
        from sqlalchemy import text

        with self._engine.connect() as conn:
            # Get count first
            result = conn.execute(
                text(
                    f"""
                    SELECT COUNT(DISTINCT chunk_id) as count
                    FROM {self._results_table.name}
                    WHERE run_id = :run_id
                    """
                ),
                {"run_id": run_id},
            )
            count = result.scalar() or 0

            # Delete results
            conn.execute(
                text(
                    f"""
                    DELETE FROM {self._results_table.name}
                    WHERE run_id = :run_id
                    """
                ),
                {"run_id": run_id},
            )

            # Delete session
            conn.execute(
                text(
                    f"""
                    DELETE FROM {self._sessions_table.name}
                    WHERE run_id = :run_id
                    """
                ),
                {"run_id": run_id},
            )

            conn.commit()
            return count

    # -------------------------------------------------------------------------
    # Validation Result Operations
    # -------------------------------------------------------------------------

    def stream_write_result(
        self,
        session: StreamSession,
        result: "ValidatorResult",
    ) -> None:
        """Write a single validator result to the stream."""
        if session.session_id not in self._active_sessions:
            raise ValueError(f"Session not active: {session.session_id}")

        writer = self._get_or_create_writer(session)
        writer.write_result(result)

    def stream_write_batch(
        self,
        session: StreamSession,
        results: list["ValidatorResult"],
    ) -> None:
        """Write a batch of validator results to the stream."""
        if session.session_id not in self._active_sessions:
            raise ValueError(f"Session not active: {session.session_id}")

        writer = self._get_or_create_writer(session)
        writer.write_results(results)

    def _get_or_create_writer(self, session: StreamSession) -> DatabaseStreamWriter:
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
        """Finalize the streaming session and create a ValidationResult."""
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

        # Aggregate statistics using cursor
        total_validators = 0
        passed_validators = 0
        failed_validators = 0
        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}

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
        )

        metadata = session.metadata.copy()
        if additional_metadata:
            metadata.update(additional_metadata)
        metadata["streaming"] = {
            "storage": "database",
            "chunks": len(session.chunks),
            "total_records": session.metrics.records_written,
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

        self._close_session(session)
        return result

    def get_streaming_stats(self, run_id: str) -> dict[str, Any]:
        """Get statistics about a streaming run."""
        self.initialize()
        from sqlalchemy import text

        with self._engine.connect() as conn:
            # Get record count
            result = conn.execute(
                text(
                    f"""
                    SELECT COUNT(*) as count
                    FROM {self._results_table.name}
                    WHERE run_id = :run_id
                    """
                ),
                {"run_id": run_id},
            )
            record_count = result.scalar() or 0

            # Get session info
            result = conn.execute(
                text(
                    f"""
                    SELECT * FROM {self._sessions_table.name}
                    WHERE run_id = :run_id
                    """
                ),
                {"run_id": run_id},
            )
            row = result.fetchone()

            if row is None:
                return {"run_id": run_id, "record_count": record_count}

            return {
                "run_id": run_id,
                "data_asset": row.data_asset,
                "status": row.status,
                "record_count": record_count,
                "storage": "database",
                "started_at": row.started_at.isoformat() if row.started_at else None,
                "updated_at": row.updated_at.isoformat() if row.updated_at else None,
            }

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def list_runs(self) -> list[str]:
        """List all run IDs in the store."""
        self.initialize()
        from sqlalchemy import text

        with self._engine.connect() as conn:
            result = conn.execute(
                text(
                    f"""
                    SELECT DISTINCT run_id FROM {self._sessions_table.name}
                    ORDER BY run_id
                    """
                )
            )
            return [row.run_id for row in result]

    def get_record_count(self, run_id: str) -> int:
        """Get total record count for a run."""
        self.initialize()
        from sqlalchemy import text

        with self._engine.connect() as conn:
            result = conn.execute(
                text(
                    f"""
                    SELECT COUNT(*) FROM {self._results_table.name}
                    WHERE run_id = :run_id
                    """
                ),
                {"run_id": run_id},
            )
            return result.scalar() or 0
