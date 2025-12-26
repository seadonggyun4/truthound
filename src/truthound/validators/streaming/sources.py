"""Streaming data sources for validators.

Provides various streaming sources for processing large datasets:
- File-based streaming (CSV, Parquet, JSON, Arrow IPC)
- Arrow Flight streaming for distributed processing
- Memory-mapped file streaming for low-memory processing

Memory Optimization:
    These sources enable processing datasets larger than available memory
    by reading data in chunks without loading the entire file.

    # Stream through a 100GB Parquet file in 100K row chunks:
    with ParquetStreamingSource("huge_data.parquet", chunk_size=100_000) as source:
        for chunk_df in source:
            issues = validator.validate(chunk_df.lazy())
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, TypeVar, Generic
import tempfile
import os

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq


T = TypeVar("T")


# =============================================================================
# Base Classes
# =============================================================================


@dataclass
class StreamingSourceConfig:
    """Base configuration for streaming sources.

    Attributes:
        chunk_size: Number of rows per chunk
        columns: Specific columns to load (None = all)
        skip_rows: Number of rows to skip at the start
        max_rows: Maximum total rows to read (None = all)
    """

    chunk_size: int = 100_000
    columns: list[str] | None = None
    skip_rows: int = 0
    max_rows: int | None = None


class StreamingSource(ABC, Generic[T]):
    """Abstract base class for streaming data sources.

    Streaming sources provide an iterator interface for reading data
    in chunks, enabling memory-efficient processing of large datasets.

    Subclasses must implement:
    - __iter__(): Yield DataFrame chunks
    - __len__(): Return total row count (if known)

    Example:
        with MyStreamingSource("data.parquet") as source:
            for chunk_df in source:
                process(chunk_df)
    """

    def __init__(self, config: StreamingSourceConfig | None = None, **kwargs: Any):
        self.config = config or StreamingSourceConfig(**kwargs)
        self._is_open = False
        self._rows_read = 0

    @abstractmethod
    def __iter__(self) -> Iterator[pl.DataFrame]:
        """Iterate over DataFrame chunks."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return total row count (may be estimated)."""
        pass

    def __enter__(self) -> "StreamingSource":
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def open(self) -> None:
        """Open the source for reading."""
        self._is_open = True
        self._rows_read = 0

    def close(self) -> None:
        """Close the source."""
        self._is_open = False

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def rows_read(self) -> int:
        return self._rows_read


# =============================================================================
# File-Based Streaming Sources
# =============================================================================


@dataclass
class FileStreamingConfig(StreamingSourceConfig):
    """Configuration for file-based streaming sources.

    Attributes:
        file_path: Path to the data file
        use_mmap: Use memory mapping for reduced memory usage
    """

    file_path: str = ""
    use_mmap: bool = True


class ParquetStreamingSource(StreamingSource):
    """Streaming source for Parquet files.

    Uses PyArrow's streaming reader to read Parquet files in row groups,
    enabling processing of files larger than available memory.

    Memory Optimization:
        - Uses row group streaming (no full file load)
        - Supports column projection
        - Optional memory mapping

    Example:
        source = ParquetStreamingSource(
            "huge_data.parquet",
            chunk_size=100_000,
            columns=["id", "value"],  # Only load these columns
        )
        with source:
            for chunk in source:
                validate(chunk)
    """

    def __init__(
        self,
        file_path: str | Path,
        chunk_size: int = 100_000,
        columns: list[str] | None = None,
        use_mmap: bool = True,
        **kwargs: Any,
    ):
        config = FileStreamingConfig(
            file_path=str(file_path),
            chunk_size=chunk_size,
            columns=columns,
            use_mmap=use_mmap,
            **kwargs,
        )
        super().__init__(config)
        self._file_path = Path(file_path)
        self._parquet_file: pq.ParquetFile | None = None
        self._total_rows: int = 0

    def open(self) -> None:
        """Open the Parquet file for streaming."""
        super().open()
        self._parquet_file = pq.ParquetFile(
            self._file_path,
            memory_map=self.config.use_mmap,
        )
        self._total_rows = self._parquet_file.metadata.num_rows

    def close(self) -> None:
        """Close the Parquet file."""
        if self._parquet_file:
            self._parquet_file = None
        super().close()

    def __len__(self) -> int:
        if self._parquet_file:
            return self._total_rows
        # Open temporarily to get count
        with pq.ParquetFile(self._file_path) as pf:
            return pf.metadata.num_rows

    def __iter__(self) -> Iterator[pl.DataFrame]:
        if not self._is_open or not self._parquet_file:
            raise RuntimeError("Source not open. Use 'with' statement or call open().")

        # Stream by row groups
        num_row_groups = self._parquet_file.metadata.num_row_groups
        rows_yielded = 0
        max_rows = self.config.max_rows

        for rg_idx in range(num_row_groups):
            # Skip if we've hit max_rows
            if max_rows is not None and rows_yielded >= max_rows:
                break

            # Read row group
            table = self._parquet_file.read_row_group(
                rg_idx,
                columns=self.config.columns,
            )

            # Convert to Polars
            df = pl.from_arrow(table)

            # Apply max_rows limit within row group
            if max_rows is not None:
                remaining = max_rows - rows_yielded
                if len(df) > remaining:
                    df = df.head(remaining)

            # Skip rows if needed
            if self.config.skip_rows > 0 and rows_yielded == 0:
                if len(df) <= self.config.skip_rows:
                    rows_yielded += len(df)
                    continue
                df = df.slice(self.config.skip_rows)

            # Yield in chunk_size batches
            for offset in range(0, len(df), self.config.chunk_size):
                chunk = df.slice(offset, self.config.chunk_size)
                self._rows_read += len(chunk)
                rows_yielded += len(chunk)
                yield chunk

                if max_rows is not None and rows_yielded >= max_rows:
                    break


class CSVStreamingSource(StreamingSource):
    """Streaming source for CSV files.

    Uses Polars' lazy scanning with slicing for memory-efficient
    CSV processing.

    Example:
        source = CSVStreamingSource(
            "large_data.csv",
            chunk_size=50_000,
            separator=",",
        )
        with source:
            for chunk in source:
                validate(chunk)
    """

    def __init__(
        self,
        file_path: str | Path,
        chunk_size: int = 100_000,
        columns: list[str] | None = None,
        separator: str = ",",
        has_header: bool = True,
        skip_rows: int = 0,
        max_rows: int | None = None,
        **kwargs: Any,
    ):
        config = StreamingSourceConfig(
            chunk_size=chunk_size,
            columns=columns,
            skip_rows=skip_rows,
            max_rows=max_rows,
        )
        super().__init__(config)
        self._file_path = Path(file_path)
        self._separator = separator
        self._has_header = has_header
        self._total_rows: int | None = None
        self._lazy_frame: pl.LazyFrame | None = None

    def open(self) -> None:
        """Open the CSV file for streaming."""
        super().open()
        self._lazy_frame = pl.scan_csv(
            self._file_path,
            separator=self._separator,
            has_header=self._has_header,
            skip_rows=self.config.skip_rows,
        )

        # Select columns if specified
        if self.config.columns:
            self._lazy_frame = self._lazy_frame.select(self.config.columns)

    def close(self) -> None:
        """Close the CSV source."""
        self._lazy_frame = None
        super().close()

    def __len__(self) -> int:
        if self._total_rows is not None:
            return self._total_rows

        # Count rows (this requires scanning the file)
        if self._lazy_frame is not None:
            self._total_rows = self._lazy_frame.select(pl.len()).collect().item()
        else:
            lf = pl.scan_csv(self._file_path)
            self._total_rows = lf.select(pl.len()).collect().item()

        return self._total_rows

    def __iter__(self) -> Iterator[pl.DataFrame]:
        if not self._is_open or self._lazy_frame is None:
            raise RuntimeError("Source not open. Use 'with' statement or call open().")

        total_rows = len(self)
        max_rows = self.config.max_rows or total_rows
        rows_yielded = 0

        for offset in range(0, total_rows, self.config.chunk_size):
            if rows_yielded >= max_rows:
                break

            remaining = min(self.config.chunk_size, max_rows - rows_yielded)
            chunk = self._lazy_frame.slice(offset, remaining).collect()

            if len(chunk) == 0:
                break

            self._rows_read += len(chunk)
            rows_yielded += len(chunk)
            yield chunk


class JSONLStreamingSource(StreamingSource):
    """Streaming source for JSON Lines (JSONL/NDJSON) files.

    Reads JSON Lines files line by line in chunks.

    Example:
        source = JSONLStreamingSource(
            "events.jsonl",
            chunk_size=10_000,
        )
        with source:
            for chunk in source:
                validate(chunk)
    """

    def __init__(
        self,
        file_path: str | Path,
        chunk_size: int = 100_000,
        columns: list[str] | None = None,
        **kwargs: Any,
    ):
        config = StreamingSourceConfig(
            chunk_size=chunk_size,
            columns=columns,
            **kwargs,
        )
        super().__init__(config)
        self._file_path = Path(file_path)
        self._lazy_frame: pl.LazyFrame | None = None

    def open(self) -> None:
        """Open the JSONL file for streaming."""
        super().open()
        self._lazy_frame = pl.scan_ndjson(self._file_path)

        if self.config.columns:
            self._lazy_frame = self._lazy_frame.select(self.config.columns)

    def close(self) -> None:
        self._lazy_frame = None
        super().close()

    def __len__(self) -> int:
        if self._lazy_frame is not None:
            return self._lazy_frame.select(pl.len()).collect().item()
        return pl.scan_ndjson(self._file_path).select(pl.len()).collect().item()

    def __iter__(self) -> Iterator[pl.DataFrame]:
        if not self._is_open or self._lazy_frame is None:
            raise RuntimeError("Source not open. Use 'with' statement or call open().")

        total_rows = len(self)
        max_rows = self.config.max_rows or total_rows
        rows_yielded = 0

        for offset in range(0, total_rows, self.config.chunk_size):
            if rows_yielded >= max_rows:
                break

            remaining = min(self.config.chunk_size, max_rows - rows_yielded)
            chunk = self._lazy_frame.slice(offset, remaining).collect()

            if len(chunk) == 0:
                break

            self._rows_read += len(chunk)
            rows_yielded += len(chunk)
            yield chunk


# =============================================================================
# Arrow IPC Streaming
# =============================================================================


class ArrowIPCStreamingSource(StreamingSource):
    """Streaming source for Arrow IPC files.

    Uses Arrow's streaming reader for zero-copy reading of Arrow IPC files.
    This is the most memory-efficient format for large datasets.

    Features:
        - Zero-copy reading (minimal memory overhead)
        - Preserves Arrow schema and metadata
        - Supports memory mapping

    Example:
        source = ArrowIPCStreamingSource(
            "data.arrow",
            chunk_size=100_000,
        )
        with source:
            for chunk in source:
                validate(chunk)
    """

    def __init__(
        self,
        file_path: str | Path,
        chunk_size: int = 100_000,
        columns: list[str] | None = None,
        use_mmap: bool = True,
        **kwargs: Any,
    ):
        config = FileStreamingConfig(
            file_path=str(file_path),
            chunk_size=chunk_size,
            columns=columns,
            use_mmap=use_mmap,
            **kwargs,
        )
        super().__init__(config)
        self._file_path = Path(file_path)
        self._reader: pa.RecordBatchFileReader | None = None
        self._source_file = None

    def open(self) -> None:
        """Open the Arrow IPC file."""
        super().open()

        if self.config.use_mmap:
            self._source_file = pa.memory_map(str(self._file_path), "r")
            self._reader = pa.ipc.open_file(self._source_file)
        else:
            self._reader = pa.ipc.open_file(str(self._file_path))

    def close(self) -> None:
        """Close the Arrow IPC file."""
        self._reader = None
        if self._source_file:
            self._source_file.close()
            self._source_file = None
        super().close()

    def __len__(self) -> int:
        if self._reader:
            return sum(
                self._reader.get_batch(i).num_rows
                for i in range(self._reader.num_record_batches)
            )
        with pa.ipc.open_file(str(self._file_path)) as reader:
            return sum(
                reader.get_batch(i).num_rows
                for i in range(reader.num_record_batches)
            )

    def __iter__(self) -> Iterator[pl.DataFrame]:
        if not self._is_open or not self._reader:
            raise RuntimeError("Source not open. Use 'with' statement or call open().")

        max_rows = self.config.max_rows
        rows_yielded = 0
        buffer: list[pa.RecordBatch] = []
        buffer_rows = 0

        for batch_idx in range(self._reader.num_record_batches):
            if max_rows is not None and rows_yielded >= max_rows:
                break

            batch = self._reader.get_batch(batch_idx)

            # Select columns if specified
            if self.config.columns:
                batch = batch.select(self.config.columns)

            buffer.append(batch)
            buffer_rows += batch.num_rows

            # Yield when buffer reaches chunk_size
            while buffer_rows >= self.config.chunk_size:
                # Combine and split
                combined = pa.Table.from_batches(buffer)
                chunk_table = combined.slice(0, self.config.chunk_size)
                remaining_table = combined.slice(self.config.chunk_size)

                chunk_df = pl.from_arrow(chunk_table)

                # Apply max_rows limit
                if max_rows is not None:
                    limit = max_rows - rows_yielded
                    if len(chunk_df) > limit:
                        chunk_df = chunk_df.head(limit)

                self._rows_read += len(chunk_df)
                rows_yielded += len(chunk_df)
                yield chunk_df

                # Update buffer
                if remaining_table.num_rows > 0:
                    buffer = [remaining_table.to_batches()[0]] if remaining_table.to_batches() else []
                    buffer_rows = remaining_table.num_rows
                else:
                    buffer = []
                    buffer_rows = 0

                if max_rows is not None and rows_yielded >= max_rows:
                    break

        # Yield remaining buffer
        if buffer and (max_rows is None or rows_yielded < max_rows):
            combined = pa.Table.from_batches(buffer)
            chunk_df = pl.from_arrow(combined)

            if max_rows is not None:
                limit = max_rows - rows_yielded
                if len(chunk_df) > limit:
                    chunk_df = chunk_df.head(limit)

            if len(chunk_df) > 0:
                self._rows_read += len(chunk_df)
                yield chunk_df


# =============================================================================
# Arrow Flight Streaming (for distributed processing)
# =============================================================================


@dataclass
class ArrowFlightConfig(StreamingSourceConfig):
    """Configuration for Arrow Flight streaming.

    Attributes:
        host: Flight server host
        port: Flight server port
        ticket: Flight ticket for the data stream
        use_tls: Use TLS encryption
        token: Authentication token
    """

    host: str = "localhost"
    port: int = 8815
    ticket: bytes = b""
    use_tls: bool = False
    token: str | None = None
    timeout_seconds: float = 60.0


class ArrowFlightStreamingSource(StreamingSource):
    """Streaming source using Arrow Flight protocol.

    Arrow Flight is designed for high-performance data transfer between
    processes and machines. This source connects to a Flight server
    and streams data in record batches.

    Features:
        - Network streaming from remote servers
        - Parallel data retrieval
        - Zero-copy when possible
        - Authentication support

    Example:
        source = ArrowFlightStreamingSource(
            host="data-server.example.com",
            port=8815,
            ticket=b"dataset-1234",
        )
        with source:
            for chunk in source:
                validate(chunk)
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8815,
        ticket: bytes = b"",
        use_tls: bool = False,
        token: str | None = None,
        chunk_size: int = 100_000,
        columns: list[str] | None = None,
        **kwargs: Any,
    ):
        config = ArrowFlightConfig(
            host=host,
            port=port,
            ticket=ticket,
            use_tls=use_tls,
            token=token,
            chunk_size=chunk_size,
            columns=columns,
            **kwargs,
        )
        super().__init__(config)
        self._client: Any = None
        self._reader: Any = None
        self._total_rows: int | None = None

    def _check_flight_available(self) -> None:
        """Check if Arrow Flight is available."""
        try:
            import pyarrow.flight  # noqa: F401
        except ImportError:
            raise ImportError(
                "pyarrow.flight is required for Arrow Flight streaming. "
                "Install with: pip install pyarrow[flight]"
            )

    def open(self) -> None:
        """Connect to the Flight server."""
        super().open()
        self._check_flight_available()

        import pyarrow.flight as flight

        # Build connection string
        scheme = "grpc+tls" if self.config.use_tls else "grpc"
        location = f"{scheme}://{self.config.host}:{self.config.port}"

        # Create client
        self._client = flight.connect(location)

        # Authenticate if token provided
        if self.config.token:
            self._client.authenticate_basic_token("", self.config.token)

        # Create reader from ticket
        ticket = flight.Ticket(self.config.ticket)
        self._reader = self._client.do_get(ticket)

    def close(self) -> None:
        """Disconnect from the Flight server."""
        if self._reader:
            self._reader.close()
            self._reader = None
        if self._client:
            self._client.close()
            self._client = None
        super().close()

    def __len__(self) -> int:
        # Flight doesn't always provide row count upfront
        if self._total_rows is not None:
            return self._total_rows
        return -1  # Unknown

    def __iter__(self) -> Iterator[pl.DataFrame]:
        if not self._is_open or not self._reader:
            raise RuntimeError("Source not open. Use 'with' statement or call open().")

        max_rows = self.config.max_rows
        rows_yielded = 0
        buffer: list[pa.RecordBatch] = []
        buffer_rows = 0

        # Stream record batches from Flight
        for batch in self._reader:
            if max_rows is not None and rows_yielded >= max_rows:
                break

            # Select columns if specified
            if self.config.columns:
                batch = batch.select(self.config.columns)

            buffer.append(batch)
            buffer_rows += batch.num_rows

            # Yield when buffer reaches chunk_size
            while buffer_rows >= self.config.chunk_size:
                combined = pa.Table.from_batches(buffer)
                chunk_table = combined.slice(0, self.config.chunk_size)
                remaining_table = combined.slice(self.config.chunk_size)

                chunk_df = pl.from_arrow(chunk_table)

                if max_rows is not None:
                    limit = max_rows - rows_yielded
                    if len(chunk_df) > limit:
                        chunk_df = chunk_df.head(limit)

                self._rows_read += len(chunk_df)
                rows_yielded += len(chunk_df)
                yield chunk_df

                if remaining_table.num_rows > 0:
                    buffer = [remaining_table.to_batches()[0]] if remaining_table.to_batches() else []
                    buffer_rows = remaining_table.num_rows
                else:
                    buffer = []
                    buffer_rows = 0

                if max_rows is not None and rows_yielded >= max_rows:
                    break

        # Yield remaining buffer
        if buffer and (max_rows is None or rows_yielded < max_rows):
            combined = pa.Table.from_batches(buffer)
            chunk_df = pl.from_arrow(combined)

            if max_rows is not None:
                limit = max_rows - rows_yielded
                if len(chunk_df) > limit:
                    chunk_df = chunk_df.head(limit)

            if len(chunk_df) > 0:
                self._rows_read += len(chunk_df)
                self._total_rows = (self._total_rows or 0) + len(chunk_df)
                yield chunk_df


# =============================================================================
# Utility Functions
# =============================================================================


def create_streaming_source(
    source: str | Path | pl.LazyFrame,
    chunk_size: int = 100_000,
    columns: list[str] | None = None,
    **kwargs: Any,
) -> StreamingSource:
    """Create an appropriate streaming source based on input type.

    Automatically detects file type and creates the appropriate
    streaming source.

    Args:
        source: File path, URL, or LazyFrame
        chunk_size: Rows per chunk
        columns: Columns to select (None = all)
        **kwargs: Additional source-specific options

    Returns:
        Appropriate StreamingSource instance

    Example:
        # Automatically detects Parquet
        source = create_streaming_source("data.parquet")

        # Explicitly configure
        source = create_streaming_source(
            "data.csv",
            chunk_size=50_000,
            separator=";",
        )
    """
    if isinstance(source, pl.LazyFrame):
        return LazyFrameStreamingSource(source, chunk_size=chunk_size, columns=columns)

    path = Path(source)
    suffix = path.suffix.lower()

    if suffix in (".parquet", ".pq"):
        return ParquetStreamingSource(path, chunk_size=chunk_size, columns=columns, **kwargs)
    elif suffix in (".csv", ".tsv"):
        separator = "\t" if suffix == ".tsv" else kwargs.pop("separator", ",")
        return CSVStreamingSource(path, chunk_size=chunk_size, columns=columns, separator=separator, **kwargs)
    elif suffix in (".jsonl", ".ndjson"):
        return JSONLStreamingSource(path, chunk_size=chunk_size, columns=columns, **kwargs)
    elif suffix in (".arrow", ".ipc", ".feather"):
        return ArrowIPCStreamingSource(path, chunk_size=chunk_size, columns=columns, **kwargs)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


class LazyFrameStreamingSource(StreamingSource):
    """Streaming source wrapping a Polars LazyFrame.

    Enables using the streaming interface with existing LazyFrames.

    Example:
        lf = pl.scan_parquet("data.parquet")
        source = LazyFrameStreamingSource(lf, chunk_size=100_000)
        with source:
            for chunk in source:
                validate(chunk)
    """

    def __init__(
        self,
        lazy_frame: pl.LazyFrame,
        chunk_size: int = 100_000,
        columns: list[str] | None = None,
        **kwargs: Any,
    ):
        config = StreamingSourceConfig(chunk_size=chunk_size, columns=columns, **kwargs)
        super().__init__(config)
        self._lazy_frame = lazy_frame
        if columns:
            self._lazy_frame = self._lazy_frame.select(columns)
        self._total_rows: int | None = None

    def __len__(self) -> int:
        if self._total_rows is None:
            self._total_rows = self._lazy_frame.select(pl.len()).collect().item()
        return self._total_rows

    def __iter__(self) -> Iterator[pl.DataFrame]:
        if not self._is_open:
            raise RuntimeError("Source not open. Use 'with' statement or call open().")

        total_rows = len(self)
        max_rows = self.config.max_rows or total_rows
        rows_yielded = 0

        for offset in range(0, total_rows, self.config.chunk_size):
            if rows_yielded >= max_rows:
                break

            remaining = min(self.config.chunk_size, max_rows - rows_yielded)
            chunk = self._lazy_frame.slice(offset, remaining).collect()

            if len(chunk) == 0:
                break

            self._rows_read += len(chunk)
            rows_yielded += len(chunk)
            yield chunk
