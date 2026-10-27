"""Streaming database store implementation with cursor-based iteration.

This module provides a streaming-capable database store that uses server-side
cursors for efficient handling of large validation results.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import AsyncIterator, Iterator  # noqa: TC003 - public annotation introspection
from dataclasses import dataclass, replace
from datetime import datetime
from functools import wraps
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from truthound.stores.streaming.base import (
    ChunkInfo,
    StreamingConfig,
    StreamingValidationStore,
    StreamSession,
    StreamStatus,
)
from truthound.stores.streaming.reader import (
    AsyncStreamReader,
    BaseStreamReader,
    _validator_result,
    get_decompressor,
    get_deserializer,
)
from truthound.stores.streaming.writer import (
    AsyncStreamWriter,
    BaseStreamWriter,
    get_serializer,
)

if TYPE_CHECKING:
    from truthound.stores.results import ValidationResult, ValidatorResult


def _serialized_connection(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._database_lock:
            return method(self, *args, **kwargs)
    return call


def _record_checksum(records):
    content = json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode("utf-8")
    return "db-json-sha256:" + hashlib.sha256(content).hexdigest()


def _settings_json(session):
    return json.dumps(["truthound-stream-settings-v1", session.config.to_dict(), session.metadata], default=str)


def _decode_session(row):
    metadata = json.loads(row.metadata_json) if row.metadata_json else {}
    settings = {}
    if isinstance(metadata, list):
        if len(metadata) != 3 or metadata[0] != "truthound-stream-settings-v1":
            raise ValueError("Invalid native streaming database settings")
        _, settings, metadata = metadata
    return StreamSession.from_dict({
        "session_id": row.session_id, "run_id": row.run_id,
        "data_asset": row.data_asset, "status": row.status,
        "config": settings, "metadata": metadata,
        "metrics": json.loads(row.metrics_json) if row.metrics_json else {},
        "chunks": json.loads(row.chunks_json) if row.chunks_json else [],
        "started_at": row.started_at.isoformat() if isinstance(row.started_at, datetime) else row.started_at,
        "updated_at": row.updated_at.isoformat() if isinstance(row.updated_at, datetime) else row.updated_at,
        "checkpoint_offset": row.checkpoint_offset,
    })


def _run_lease(engine, run_id):
    if engine.dialect.name != "sqlite" or engine.url.database in {None, "", ":memory:"}:
        return None
    import hashlib
    from pathlib import Path

    from truthound.stores.streaming._filesystem_io import StreamFileLease

    database = Path(engine.url.database).absolute()
    identity = hashlib.sha256(run_id.encode()).hexdigest()
    return StreamFileLease(database.with_name(f".{database.name}.stream-{identity}"))


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
        import re

        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.table_prefix):
            raise ValueError("Native streaming table prefix must be a SQL identifier")
        if self.cursor_fetch_size < 1 or self.batch_insert_size < 1:
            raise ValueError("Native streaming database batch sizes must be positive")


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
        state_writer: Any = None,
        database_lock: Any = None,
    ):
        """Initialize the database writer.

        Args:
            session: The streaming session.
            config: Database streaming configuration.
            engine: SQLAlchemy engine.
            session_maker: SQLAlchemy session maker.
            results_table: Results table object.
        """
        self._database_lock = database_lock or threading.RLock()
        super().__init__(session, config)
        self.db_config = config
        self._engine = engine
        self._session_maker = session_maker
        self._results_table = results_table
        self._db_session: Any = None
        self._pending_rows: list[dict[str, Any]] = []
        self._state_writer = state_writer
        self._ownership_lease = _run_lease(engine, session.run_id)

    @_serialized_connection
    def _get_db_session(self) -> Any:
        """Get or create database session."""
        if self._db_session is None:
            self._db_session = self._session_maker()
        return self._db_session

    @_serialized_connection
    def _write_chunk(self, chunk_info: ChunkInfo, data: bytes) -> None:
        """Write chunk data to database."""
        # For database, we don't write chunks as blobs
        # Instead, we write individual records
        # The 'data' is compressed serialized records
        decompressor = get_decompressor(self.config.compression)
        deserializer = get_deserializer(self.config.format)

        raw_data = decompressor.decompress(data)
        records = list(deserializer.deserialize(raw_data))
        chunk_info.checksum = _record_checksum(records)

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

    @_serialized_connection
    def _write_session_state(self) -> None:
        """Write session state to database."""
        if self._state_writer is None:
            raise RuntimeError("A database stream writer requires a session state owner")
        self._state_writer(self.session)

    def _abort(self) -> None:
        self._finalize()

    @_serialized_connection
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
        database_lock: Any = None,
    ):
        """Initialize the database reader.

        Args:
            run_id: The run ID to read.
            engine: SQLAlchemy engine.
            config: Database streaming configuration.
            results_table: Results table object.
        """
        self._database_lock = database_lock or threading.RLock()
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

    @_serialized_connection
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

    @_serialized_connection
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

    @_serialized_connection
    def read(self) -> dict[str, Any] | None:
        """Read a single record using cursor."""
        self._guard_io()
        if self._closed:
            raise ValueError("Native stream reader is closed")
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

    @_serialized_connection
    def _fetch_next_batch(self) -> None:
        """Fetch next batch of rows using server-side cursor."""
        from sqlalchemy import text

        self._guard_io()

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
        self.metrics.record_read(
            len(self._current_rows),
            sum(len(row.data_json.encode("utf-8")) for row in self._current_rows),
        )
        self._row_index = 0

        if not self._current_rows:
            self._exhausted = True

    @_serialized_connection
    def close(self) -> None:
        """Close the reader and database connection."""
        super().close()
        try:
            if self._cursor is not None:
                self._cursor.close()
                self._cursor = None
        finally:
            if self._connection is not None:
                self._connection.close()
                self._connection = None

    @_serialized_connection
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

        self._database_lock = threading.RLock()
        self._engine: Any = None
        self._session_maker: Any = None
        self._metadata: Any = None
        self._results_table: Any = None
        self._sessions_table: Any = None

    @_serialized_connection
    def initialize(self) -> None:
        super().initialize()

    @classmethod
    def _default_config(cls) -> StreamingDatabaseConfig:
        """Create default configuration."""
        return StreamingDatabaseConfig()

    @_serialized_connection
    def _do_initialize(self) -> None:
        """Initialize database engine and create tables."""
        try:
            from sqlalchemy import (
                Column,
                DateTime,
                Index,
                Integer,
                MetaData,
                String,
                Table,
                Text,
                create_engine,
            )
            from sqlalchemy.orm import sessionmaker
        except ImportError:
            raise ImportError("sqlalchemy library required for database streaming store") from None

        from sqlalchemy.engine import make_url

        selected_url = make_url(self._config.connection_url)
        options = {"pool_pre_ping": True}
        if selected_url.get_backend_name() == "sqlite" and selected_url.database in {None, "", ":memory:"}:
            from sqlalchemy.pool import StaticPool

            # Async wrappers run native calls on executor threads. A memory DB
            # belongs to this engine's single connection across those calls.
            options.update(poolclass=StaticPool, connect_args={"check_same_thread": False})
        else:
            options.update(pool_size=self._config.pool_size, max_overflow=self._config.max_overflow)
        self._engine = create_engine(self._config.connection_url, **options)

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

        run_index = Index(f"{self._config.table_prefix}unique_run", self._sessions_table.c.run_id, unique=True)

        # Create tables
        try:
            self._metadata.create_all(self._engine)
            # Existing databases are checked too; ambiguous runs fail closed.
            run_index.create(self._engine, checkfirst=True)
        except Exception as initialization_error:
            try:
                self._engine.dispose()
            except Exception as close_error:
                raise ExceptionGroup("Native streaming initialization and cleanup failed", [initialization_error, close_error]) from None
            finally:
                self._engine = None
            raise

    def close(self) -> None:
        """Close the store and database connections."""
        try:
            super().close()
        finally:
            if self._engine is not None:
                with self._database_lock:
                    self._engine.dispose()
                    self._engine = None
                    self._initialized = False

    # -------------------------------------------------------------------------
    # Session Management
    # -------------------------------------------------------------------------

    @_serialized_connection
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

    @_serialized_connection
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

            return _decode_session(row)

    def resume_session(self, session_id: str) -> StreamSession:
        """Resume an interrupted session."""
        session = self.get_session(session_id)
        if session is None:
            raise ValueError(f"Session not found: {session_id}")

        if session.status == StreamStatus.COMPLETED:
            raise ValueError(f"Session already completed: {session_id}")

        from contextlib import nullcontext

        with _run_lease(self._engine, session.run_id) or nullcontext():
            # Bypass an earlier in-memory observation while the lease is held.
            self._active_sessions.pop(session_id, None)
            session = self.get_session(session_id)
            if session is None or session.status in {StreamStatus.COMPLETED, StreamStatus.ABORTED}:
                raise ValueError("A terminal or missing native stream cannot be resumed")
            self._validate_session_chunks(session)
            session.status = StreamStatus.ACTIVE
            session.updated_at = datetime.now()
            self._save_session(session)
            self._active_sessions[session_id] = session
        return session

    @_serialized_connection
    def _validate_session_chunks(self, session: StreamSession) -> None:
        self._guard_io()
        reader = DatabaseStreamReader(session.run_id, self._engine, self._config, self._results_table, database_lock=self._database_lock)
        try:
            actual = reader._get_chunks()
            if [(item.chunk_id, item.chunk_index, item.record_count) for item in actual] != [
                (item.chunk_id, item.chunk_index, item.record_count) for item in session.chunks
            ]:
                raise RuntimeError("Unconfirmed native database chunks require reconciliation")
            from sqlalchemy import select

            with self._engine.connect() as connection:
                for chunk in session.chunks:
                    self._guard_io()
                    if chunk.checksum and chunk.checksum.startswith("db-json-sha256:"):
                        records = [json.loads(row.data_json) for row in connection.execute(
                            select(self._results_table.c.data_json).where(
                                self._results_table.c.run_id == session.run_id,
                                self._results_table.c.chunk_id == chunk.chunk_id,
                            ).order_by(self._results_table.c.record_index)
                        )]
                        if _record_checksum(records) != chunk.checksum:
                            raise RuntimeError("Confirmed native database stream checksum changed")
        finally:
            reader.close()

    @_serialized_connection
    def get_session_for_run(self, run_id: str) -> StreamSession | None:
        self.initialize()
        self._guard_io()
        from sqlalchemy import select

        with self._engine.connect() as connection:
            identity = connection.execute(select(self._sessions_table.c.session_id).where(
                self._sessions_table.c.run_id == run_id
            )).scalar_one_or_none()
        return self.get_session(identity) if identity is not None else None

    @_serialized_connection
    def list_sessions(self, limit: int = 100, offset: int = 0) -> list[StreamSession]:
        if limit < 1 or offset < 0:
            raise ValueError("Invalid native streaming session page")
        self.initialize()
        self._guard_io()
        from sqlalchemy import select

        with self._engine.connect() as connection:
            identities = connection.execute(select(self._sessions_table.c.session_id).order_by(
                self._sessions_table.c.run_id
            ).offset(offset).limit(limit)).scalars().all()
        return [self.get_session(identity) for identity in identities]

    def recovery_status(self, run_id: str) -> dict[str, Any]:
        """Inspect retained SQLite rows under native run ownership."""
        from truthound.stores.streaming.database_recovery import status

        return status(self, run_id)

    def reconcile_stream(self, run_id: str, *, resolution: str, expected_fingerprint: str) -> dict[str, Any]:
        """Explicitly reconcile unconfirmed SQLite rows in one transaction."""
        from truthound.stores.streaming.database_recovery import reconcile

        return reconcile(self, run_id, resolution=resolution, expected_fingerprint=expected_fingerprint)

    def _close_session(self, session: StreamSession) -> None:
        """Close and finalize a session."""
        if session.session_id in self._active_sessions:
            del self._active_sessions[session.session_id]
        if session.status not in {StreamStatus.FAILED, StreamStatus.ABORTED}:
            session.status = StreamStatus.COMPLETED
        session.updated_at = datetime.now()
        self._save_session(session)

    @_serialized_connection
    def _save_session(self, session: StreamSession) -> None:
        """Save session to database."""
        from sqlalchemy import text

        self._guard_io()
        saved_metadata = _settings_json(session)

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
                        "metadata_json": saved_metadata,
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
                        metadata_json=saved_metadata,
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

    @_serialized_connection
    def create_writer(self, session: StreamSession) -> DatabaseStreamWriter:
        """Create a writer for the session."""
        self.initialize()
        self._assert_writer_available(session)
        settings = replace(self._config, **{
            key: getattr(session.config, key) for key in session.config.to_dict()
        })
        writer = DatabaseStreamWriter(
            session=session,
            config=settings,
            engine=self._engine,
            session_maker=self._session_maker,
            results_table=self._results_table,
            state_writer=self._save_session,
            database_lock=self._database_lock,
        )
        try:
            from sqlalchemy import select

            with self._engine.connect() as connection:
                saved = connection.execute(select(self._sessions_table.c.status, self._sessions_table.c.chunks_json).where(
                    self._sessions_table.c.session_id == session.session_id
                )).one()
            if saved.status in {"completed", "aborted"} or json.loads(saved.chunks_json) != [chunk.to_dict() for chunk in session.chunks]:
                raise RuntimeError("Stale native database stream state requires reconciliation")
            self._validate_session_chunks(session)
        except BaseException:
            if writer._ownership_lease is not None:
                writer._ownership_lease.close()
            raise
        return self._track_writer(session, writer)

    async def create_async_writer(self, session: StreamSession) -> AsyncStreamWriter:
        """Create an async writer for the session."""
        writer = self.create_writer(session)
        return AsyncStreamWriter(writer)

    # -------------------------------------------------------------------------
    # Reader Operations
    # -------------------------------------------------------------------------

    @_serialized_connection
    def create_reader(self, run_id: str) -> DatabaseStreamReader:
        """Create a reader for a run's results."""
        self.initialize()
        self._guard_io()
        session = self.get_session_for_run(run_id)
        if session is None:
            raise RuntimeError("Native database stream session is missing; reconciliation is required")
        self._validate_session_chunks(session)
        reader = DatabaseStreamReader(
            run_id=run_id,
            engine=self._engine,
            config=self._config,
            results_table=self._results_table,
            database_lock=self._database_lock,
        )
        reader._chunks = session.chunks
        return self._track_reader(reader)

    async def create_async_reader(self, run_id: str) -> AsyncStreamReader:
        """Create an async reader for a run's results."""
        reader = self.create_reader(run_id)
        return AsyncStreamReader(reader)

    def iter_results(
        self,
        run_id: str,
        batch_size: int = 1000,
    ) -> Iterator[ValidatorResult]:
        """Iterate over results for a run using cursor."""
        reader = self.create_reader(run_id)
        with reader:
            yield from reader.iter_results()

    async def aiter_results(
        self,
        run_id: str,
        batch_size: int = 1000,
    ) -> AsyncIterator[ValidatorResult]:
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
        try:
            return reader._chunks
        finally:
            reader.close()

    @_serialized_connection
    def get_chunk(self, chunk_info: ChunkInfo) -> list[ValidatorResult]:
        """Get records from a specific chunk."""
        from sqlalchemy import text


        self.initialize()
        self._guard_io()
        from sqlalchemy import select

        with self._engine.connect() as connection:
            run_id = connection.execute(select(self._results_table.c.run_id).where(
                self._results_table.c.chunk_id == chunk_info.chunk_id
            ).distinct()).scalar_one_or_none()
        if run_id is None:
            raise RuntimeError("Native database stream chunk is missing")
        session = self.get_session_for_run(run_id)
        if session is None:
            raise RuntimeError("Native database stream session is missing")
        self._validate_session_chunks(session)
        if chunk_info.to_dict() not in [chunk.to_dict() for chunk in session.chunks]:
            raise RuntimeError("Native database stream chunk is not confirmed")

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

            return [_validator_result(json.loads(row.data_json)) for row in result]

    @_serialized_connection
    def delete_chunks(self, run_id: str) -> int:
        """Delete all chunks for a run."""
        self.initialize()
        from contextlib import nullcontext

        from sqlalchemy import text

        self._guard_io()
        if any(writer.session.run_id == run_id and not writer._closed for writer in self._writers.values()):
            raise RuntimeError("A native stream writer already owns this run")
        with _run_lease(self._engine, run_id) or nullcontext(), self._engine.connect() as conn:
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
            self._active_sessions = {key: item for key, item in self._active_sessions.items() if item.run_id != run_id}
            return count

    # -------------------------------------------------------------------------
    # Validation Result Operations
    # -------------------------------------------------------------------------

    def stream_write_result(
        self,
        session: StreamSession,
        result: ValidatorResult,
    ) -> None:
        """Write a single validator result to the stream."""
        if session.session_id not in self._active_sessions:
            raise ValueError(f"Session not active: {session.session_id}")

        writer = self._get_or_create_writer(session)
        writer.write_result(result)

    def stream_write_batch(
        self,
        session: StreamSession,
        results: list[ValidatorResult],
    ) -> None:
        """Write a batch of validator results to the stream."""
        if session.session_id not in self._active_sessions:
            raise ValueError(f"Session not active: {session.session_id}")

        writer = self._get_or_create_writer(session)
        writer.write_results(results)

    def _get_or_create_writer(self, session: StreamSession) -> DatabaseStreamWriter:
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
        """Finalize the streaming session and create a ValidationResult."""
        from truthound.stores.results import (
            ResultStatistics,
            ResultStatus,
            ValidationResult,
        )

        if session.status in {StreamStatus.FAILED, StreamStatus.ABORTED}:
            raise RuntimeError("A failed native stream cannot produce a completed result")
        writer = self._writers.get(session.session_id)
        if writer is not None:
            writer.close()
        self._validate_session_chunks(session)


        # Aggregate statistics using cursor
        total_validators = 0
        passed_validators = 0
        failed_validators = 0
        total_issues = 0
        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}

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
