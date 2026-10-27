"""Data generators for benchmark scenarios.

This module provides extensible data generators that create realistic
test data for various benchmarking scenarios.

Generator types:
- TabularDataGenerator: General-purpose tabular data
- TimeSeriesDataGenerator: Time-series data with patterns
- FinancialDataGenerator: Stock/crypto market data
- TextDataGenerator: Text-heavy data with PII patterns
"""

from __future__ import annotations

import json
import math
import random
import re
import string
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import polars as pl

# =============================================================================
# Configuration
# =============================================================================


class ColumnType(str, Enum):
    """Types of columns for data generation."""

    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    CATEGORY = "category"
    UUID = "uuid"
    EMAIL = "email"
    PHONE = "phone"
    IP_ADDRESS = "ip_address"
    URL = "url"
    JSON = "json"


class DataPattern(str, Enum):
    """Data patterns for generation."""

    UNIFORM = "uniform"
    NORMAL = "normal"
    EXPONENTIAL = "exponential"
    SKEWED = "skewed"
    SEQUENTIAL = "sequential"
    CYCLIC = "cyclic"
    RANDOM_WALK = "random_walk"
    SPARSE = "sparse"  # Many nulls
    CONSTANT = "constant"


@dataclass
class ColumnSpec:
    """Specification for a generated column."""

    name: str
    dtype: ColumnType
    pattern: DataPattern = DataPattern.UNIFORM
    null_ratio: float | None = None
    cardinality: int = 0  # 0 = unlimited
    min_value: float | None = None
    max_value: float | None = None
    categories: list[str] | None = None
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.dtype, self.pattern = ColumnType(self.dtype), DataPattern(self.pattern)
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Generated column name must be nonempty")
        if type(self.cardinality) is not int or self.cardinality < 0:
            raise ValueError("Cardinality must be a nonnegative integer")
        if self.null_ratio is not None and (
            type(self.null_ratio) not in (int, float)
            or not math.isfinite(self.null_ratio)
            or not 0 <= self.null_ratio <= 1
        ):
            raise ValueError("Column null ratio must be finite and between zero and one")
        for value in (self.min_value, self.max_value):
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
                raise ValueError("Numeric bounds must be finite numbers")
        if (
            self.min_value is not None
            and self.max_value is not None
            and self.min_value > self.max_value
        ):
            raise ValueError("Minimum exceeds maximum")
        self.params, self.categories = deepcopy(self.params), deepcopy(self.categories)
        allowed = (
            {"min_length", "max_length"}
            if self.dtype == ColumnType.STRING
            else {"period"}
            if self.dtype in {ColumnType.INTEGER, ColumnType.FLOAT}
            else set()
        )
        if set(self.params) - allowed:
            raise ValueError("Unsupported generated-column parameter")
        for name, value in self.params.items():
            if type(value) is not int or value < (1 if name == "period" else 0):
                raise ValueError("Invalid generated-column parameter")
        if self.params.get("min_length", 5) > self.params.get("max_length", 20):
            raise ValueError("Minimum string length exceeds maximum")
        if self.categories is not None and (
            not self.categories or any(not isinstance(value, str) for value in self.categories)
        ):
            raise ValueError("Categories must be nonempty strings")
        if self.dtype not in {ColumnType.INTEGER, ColumnType.FLOAT}:
            allowed_patterns = {
                DataPattern.UNIFORM,
                DataPattern.CONSTANT,
                DataPattern.SPARSE,
                DataPattern.CYCLIC,
            }
            if self.dtype in {ColumnType.DATE, ColumnType.DATETIME}:
                allowed_patterns.add(DataPattern.SEQUENTIAL)
            if self.pattern not in allowed_patterns:
                raise ValueError("Pattern is not supported for this column type")


@dataclass
class GeneratorConfig:
    """Configuration for data generation.

    Controls output size, randomness, and data characteristics.
    """

    row_count: int = 100_000
    seed: int | None = None
    null_ratio: float = 0.01
    duplicate_ratio: float = 0.0
    column_specs: list[ColumnSpec] = field(default_factory=list)

    # Performance
    chunk_size: int = 100_000
    parallelize: bool = True

    def __post_init__(self):
        if type(self.row_count) is not int or self.row_count < 0:
            raise ValueError("row_count must be a nonnegative integer")
        if type(self.chunk_size) is not int or self.chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        if type(self.parallelize) is not bool:
            raise ValueError("parallelize must be boolean")
        if self.seed is not None and (type(self.seed) is not int or self.seed < 0):
            raise ValueError("seed must be a nonnegative integer")
        for value in (self.null_ratio, self.duplicate_ratio):
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Generator ratios must be finite values between zero and one")
        self.column_specs = deepcopy(self.column_specs)
        if any(not isinstance(spec, ColumnSpec) for spec in self.column_specs):
            raise ValueError("column_specs must contain ColumnSpec values")
        self.column_specs = [replace(spec) for spec in self.column_specs]
        names = [spec.name for spec in self.column_specs]
        if len(set(names)) != len(names):
            raise ValueError("Duplicate generated column names")

    def with_size(self, row_count: int) -> GeneratorConfig:
        """Create a copy with different row count."""
        return GeneratorConfig(
            row_count=row_count,
            seed=self.seed,
            null_ratio=self.null_ratio,
            duplicate_ratio=self.duplicate_ratio,
            column_specs=deepcopy(self.column_specs),
            chunk_size=self.chunk_size,
            parallelize=self.parallelize,
        )


# =============================================================================
# Base Generator
# =============================================================================


class DataGenerator(ABC):
    """Abstract base class for data generators.

    Provides the framework for creating benchmark data.

    Example:
        class MyGenerator(DataGenerator):
            name = "my_generator"
            description = "Generates my custom data"

            def generate(self, config):
                return pl.DataFrame({...})
    """

    name: str = "base"
    description: str = ""

    def __init__(self, seed: int | None = None):
        if seed is not None and (type(seed) is not int or seed < 0):
            raise ValueError("seed must be a nonnegative integer")
        self._seed = seed
        self._rng = random.Random(seed)

    def _begin(self, config):
        config = replace(deepcopy(config))
        if config.column_specs and not isinstance(self, TabularDataGenerator):
            raise ValueError("Custom column specifications require the tabular generator")
        seed = config.seed if config.seed is not None else self._seed
        if seed is not None:
            self._rng = random.Random(seed)
        self._generation_seed = seed if seed is not None else self._rng.getrandbits(32)
        return config

    def _timestamps(self, count, frequency):
        parts = re.findall(r"([0-9]+)(ns|us|ms|mo|s|m|h|d|w|q|y)", frequency)
        if (
            not parts
            or "".join(number + unit for number, unit in parts) != frequency
            or not any(int(number) > 0 for number, _ in parts)
        ):
            raise ValueError("Frequency must be a positive native duration")
        if count == 0:
            return pl.Series("timestamp", [], dtype=pl.Datetime("us"))
        start = datetime(2024, 1, 1)
        offset = "".join(str(int(number) * count) + unit for number, unit in parts)
        end = pl.select(pl.lit(start).dt.offset_by(offset)).item()
        return pl.datetime_range(start, end, interval=frequency, eager=True).head(count)

    def _finish(self, frame, config, *, apply_nulls=True):
        if apply_nulls and config.null_ratio:
            frame = pl.DataFrame([self._apply_nulls(series, config.null_ratio) for series in frame])
        if config.duplicate_ratio and frame.height:
            count = int(frame.height * config.duplicate_ratio)
            if count:
                base = frame.head(max(1, frame.height - count))
                duplicates = base[
                    self._rng.choices(range(base.height), k=frame.height - base.height)
                ]
                frame = pl.concat([base, duplicates])
        return (
            pl.concat(list(frame.iter_slices(config.chunk_size)), rechunk=False)
            if frame.height
            else frame
        )

    @abstractmethod
    def generate(self, config: GeneratorConfig) -> pl.DataFrame:
        """Generate a DataFrame according to the configuration.

        Args:
            config: Generation configuration

        Returns:
            Generated DataFrame
        """
        pass

    def generate_lazy(self, config: GeneratorConfig) -> pl.LazyFrame:
        """Generate a LazyFrame.

        Default implementation wraps generate(). Override for
        streaming generation.

        Args:
            config: Generation configuration

        Returns:
            Generated LazyFrame
        """
        return self.generate(config).lazy()

    def _apply_nulls(
        self,
        series: pl.Series,
        null_ratio: float,
    ) -> pl.Series:
        """Apply null values to a series."""
        if null_ratio <= 0:
            return series

        n = len(series)
        null_count = int(n * null_ratio)

        if null_count == 0:
            return series

        # Create mask for null positions
        null_indices = set(self._rng.sample(range(n), null_count))

        # Convert to list, apply nulls, and create new series
        values = series.to_list()
        for idx in null_indices:
            values[idx] = None

        return pl.Series(series.name or "", values, dtype=series.dtype)

    def _generate_string(
        self,
        n: int,
        min_length: int = 5,
        max_length: int = 20,
    ) -> list[str]:
        """Generate random strings."""
        return [
            "".join(
                self._rng.choices(
                    string.ascii_letters + string.digits,
                    k=self._rng.randint(min_length, max_length),
                )
            )
            for _ in range(n)
        ]

    def _generate_emails(self, n: int) -> list[str]:
        """Generate random email addresses."""
        domains = ["example.com", "test.org", "sample.net", "demo.io"]
        return [f"user{i}@{self._rng.choice(domains)}" for i in range(n)]

    def _generate_phones(self, n: int) -> list[str]:
        """Generate random phone numbers."""
        return [
            f"+1-{self._rng.randint(200, 999)}-{self._rng.randint(200, 999)}-{self._rng.randint(1000, 9999)}"
            for _ in range(n)
        ]

    def _generate_uuids(self, n: int) -> list[str]:
        """Generate random UUIDs."""
        import uuid

        return [str(uuid.UUID(int=self._rng.getrandbits(128), version=4)) for _ in range(n)]

    def _generate_ips(self, n: int) -> list[str]:
        """Generate random IP addresses."""
        return [
            f"{self._rng.randint(1, 255)}.{self._rng.randint(0, 255)}."
            f"{self._rng.randint(0, 255)}.{self._rng.randint(1, 254)}"
            for _ in range(n)
        ]


# =============================================================================
# Tabular Data Generator
# =============================================================================


class TabularDataGenerator(DataGenerator):
    """Generator for general-purpose tabular data.

    Creates realistic tabular data with configurable columns,
    distributions, and data quality characteristics.

    Example:
        generator = TabularDataGenerator()
        config = GeneratorConfig(
            row_count=1_000_000,
            column_specs=[
                ColumnSpec("id", ColumnType.INTEGER, DataPattern.SEQUENTIAL),
                ColumnSpec("value", ColumnType.FLOAT, DataPattern.NORMAL),
                ColumnSpec("category", ColumnType.CATEGORY, categories=["A", "B", "C"]),
            ],
        )
        df = generator.generate(config)
    """

    name = "tabular"
    description = "General-purpose tabular data generator"

    def generate(self, config: GeneratorConfig) -> pl.DataFrame:
        """Generate tabular data."""
        config = self._begin(config)

        n = config.row_count

        # Use default schema if no specs provided
        if not config.column_specs:
            return self._finish(self._generate_default_schema(n, config.null_ratio), config)

        seeds = [self._rng.getrandbits(64) for _ in config.column_specs]

        def generate_column(pair):
            spec, seed = pair
            worker = TabularDataGenerator(seed)
            count = min(n, spec.cardinality) if spec.cardinality else n
            if spec.pattern == DataPattern.CONSTANT:
                count = min(n, 1)
            elif spec.pattern == DataPattern.CYCLIC and spec.dtype not in {
                ColumnType.INTEGER,
                ColumnType.FLOAT,
            }:
                count = min(count, 10)
            series = worker._generate_column(count, spec)
            dtype = {
                ColumnType.INTEGER: pl.Int64,
                ColumnType.FLOAT: pl.Float64,
                ColumnType.BOOLEAN: pl.Boolean,
                ColumnType.DATE: pl.Date,
                ColumnType.DATETIME: pl.Datetime("us"),
            }.get(spec.dtype, pl.String)
            series = series.cast(dtype).rename(spec.name)
            if count < n:
                indices = (
                    [i % count for i in range(n)]
                    if spec.pattern in {DataPattern.CONSTANT, DataPattern.CYCLIC}
                    else worker._rng.choices(range(count), k=n)
                )
                series = series[indices]
            null_ratio = spec.null_ratio if spec.null_ratio is not None else config.null_ratio
            if spec.pattern == DataPattern.SPARSE:
                null_ratio = max(null_ratio, 0.8)
            return worker._apply_nulls(series, null_ratio)

        pairs = list(zip(config.column_specs, seeds, strict=True))
        if config.parallelize and len(pairs) > 1:
            with ThreadPoolExecutor(max_workers=min(4, len(pairs))) as executor:
                columns = list(executor.map(generate_column, pairs))
        else:
            columns = [generate_column(pair) for pair in pairs]
        df = pl.DataFrame(columns)

        return self._finish(df, config, apply_nulls=False)

    def _generate_default_schema(
        self,
        n: int,
        null_ratio: float,
    ) -> pl.DataFrame:
        """Generate data with a default schema using Polars native operations.

        Optimized to use Polars expressions instead of Python loops for
        significantly better performance on large datasets.
        """
        seed = self._generation_seed % (1 << 64)

        # Use Polars native random generation (much faster than Python loops)
        df = pl.DataFrame(
            {
                "id": pl.arange(0, n, eager=True),
            }
        ).with_columns(
            [
                # Integer column: random integers 0-1000
                (
                    pl.lit(0).cast(pl.Int64)
                    + pl.arange(0, n, eager=False).shuffle(seed=seed) % 1001
                ).alias("int_col"),
                # Float column: approximate normal distribution using uniform
                # mean=100, std=25 approximated via uniform [25, 175]
                (
                    pl.lit(25.0)
                    + pl.arange(0, n, eager=False)
                    .shuffle(seed=(seed + 1) % (1 << 64))
                    .cast(pl.Float64)
                    % 150
                ).alias("float_col"),
                # Category column
                pl.Series("category_col", ["A", "B", "C", "D"] * (n // 4 + 1))
                .head(n)
                .shuffle(seed=(seed + 2) % (1 << 64)),
                # Boolean column
                (pl.arange(0, n, eager=False).shuffle(seed=(seed + 3) % (1 << 64)) % 2 == 0).alias(
                    "bool_col"
                ),
            ]
        )

        # String column - use fixed-length strings for speed
        string_vals = [f"str_{i:08d}" for i in range(min(n, 10000))]
        if n > 10000:
            # Repeat for larger datasets
            string_vals = (string_vals * (n // 10000 + 1))[:n]
        df = df.with_columns(
            pl.Series("string_col", string_vals).shuffle(seed=(seed + 4) % (1 << 64))
        )

        # Date column using Polars native date generation
        date_range = pl.datetime_range(
            datetime(2020, 1, 1),
            datetime(2024, 12, 31),
            interval="1d",
            eager=True,
        )
        df = df.with_columns(
            date_range.sample(n, with_replacement=True, seed=(seed + 5) % (1 << 64)).alias(
                "date_col"
            )
        )

        return df

    def _generate_column(self, n: int, spec: ColumnSpec) -> pl.Series:
        """Generate a single column according to spec."""
        if spec.dtype == ColumnType.INTEGER:
            return self._generate_integer(n, spec)
        elif spec.dtype == ColumnType.FLOAT:
            return self._generate_float(n, spec)
        elif spec.dtype == ColumnType.STRING:
            return pl.Series(self._generate_string(n, **spec.params))
        elif spec.dtype == ColumnType.BOOLEAN:
            return pl.Series(self._rng.choices([True, False], k=n))
        elif spec.dtype == ColumnType.CATEGORY:
            categories = spec.categories or ["cat_A", "cat_B", "cat_C"]
            return pl.Series(self._rng.choices(categories, k=n))
        elif spec.dtype == ColumnType.EMAIL:
            return pl.Series(self._generate_emails(n))
        elif spec.dtype == ColumnType.PHONE:
            return pl.Series(self._generate_phones(n))
        elif spec.dtype == ColumnType.UUID:
            return pl.Series(self._generate_uuids(n))
        elif spec.dtype == ColumnType.IP_ADDRESS:
            return pl.Series(self._generate_ips(n))
        elif spec.dtype == ColumnType.URL:
            return pl.Series(
                [f"https://example.com/item/{self._rng.getrandbits(64):016x}" for _ in range(n)]
            )
        elif spec.dtype == ColumnType.JSON:
            return pl.Series(
                [
                    json.dumps({"index": i, "value": self._rng.random()}, allow_nan=False)
                    for i in range(n)
                ]
            )
        elif spec.dtype == ColumnType.DATE:
            return self._generate_dates(n, spec)
        elif spec.dtype == ColumnType.DATETIME:
            return self._generate_datetimes(n, spec)
        else:
            return pl.Series(self._generate_string(n))

    def _generate_numeric(self, n, spec, *, integer):
        minimum = spec.min_value if spec.min_value is not None else 0
        maximum = (
            spec.max_value if spec.max_value is not None else (1_000_000 if integer else 1000.0)
        )
        if integer:
            minimum, maximum = int(minimum), int(maximum)
        if minimum > maximum:
            raise ValueError("Minimum exceeds maximum")
        span = maximum - minimum
        period = spec.params.get("period", 10)
        values, current = [], minimum + span / 2
        for i in range(n):
            pattern = spec.pattern
            if not span or pattern == DataPattern.CONSTANT:
                value = minimum
            elif pattern == DataPattern.SEQUENTIAL:
                value = (
                    minimum + i % (int(span) + 1) if integer else minimum + span * i / max(1, n - 1)
                )
            elif pattern == DataPattern.CYCLIC:
                value = (
                    minimum + i % min(period, int(span) + 1)
                    if integer
                    else minimum + span * (i % period) / max(1, period - 1)
                )
            elif pattern == DataPattern.NORMAL:
                value = self._rng.gauss(minimum + span / 2, span / 6)
            elif pattern == DataPattern.EXPONENTIAL:
                value = minimum + self._rng.expovariate(3 / span)
            elif pattern == DataPattern.SKEWED:
                value = minimum + span * self._rng.betavariate(0.5, 2)
            elif pattern == DataPattern.RANDOM_WALK:
                current = max(minimum, min(maximum, current + self._rng.gauss(0, span / 100)))
                value = current
            else:
                value = (
                    self._rng.randint(minimum, maximum)
                    if integer
                    else self._rng.uniform(minimum, maximum)
                )
            value = max(minimum, min(maximum, value))
            values.append(int(value) if integer else float(value))
        return pl.Series(values, dtype=pl.Int64 if integer else pl.Float64)

    def _generate_integer(self, n: int, spec: ColumnSpec) -> pl.Series:
        return self._generate_numeric(n, spec, integer=True)

    def _generate_float(self, n: int, spec: ColumnSpec) -> pl.Series:
        return self._generate_numeric(n, spec, integer=False)

    def _generate_dates(self, n: int, spec: ColumnSpec) -> pl.Series:
        """Generate date values."""
        start = datetime(2020, 1, 1)
        end = datetime(2024, 12, 31)
        delta = (end - start).days

        if spec.pattern == DataPattern.SEQUENTIAL:
            dates = [start + timedelta(days=i % delta) for i in range(n)]
        else:
            dates = [start + timedelta(days=self._rng.randint(0, delta)) for _ in range(n)]

        return pl.Series(dates).cast(pl.Date)

    def _generate_datetimes(self, n: int, spec: ColumnSpec) -> pl.Series:
        """Generate datetime values."""
        start = datetime(2020, 1, 1)
        end = datetime(2024, 12, 31)
        delta_seconds = int((end - start).total_seconds())

        if spec.pattern == DataPattern.SEQUENTIAL:
            interval = delta_seconds // n if n else 0
            datetimes = [start + timedelta(seconds=i * interval) for i in range(n)]
        else:
            datetimes = [
                start + timedelta(seconds=self._rng.randint(0, delta_seconds)) for _ in range(n)
            ]

        return pl.Series(datetimes)

    def _add_duplicates(
        self,
        df: pl.DataFrame,
        ratio: float,
    ) -> pl.DataFrame:
        """Add duplicate rows to the DataFrame."""
        n = len(df)
        dup_count = int(n * ratio)

        if dup_count == 0:
            return df

        # Sample rows to duplicate
        dup_indices = self._rng.sample(range(n), min(dup_count, n))
        duplicates = df[dup_indices]

        return pl.concat([df, duplicates])


# =============================================================================
# Time Series Data Generator
# =============================================================================


class TimeSeriesDataGenerator(DataGenerator):
    """Generator for time-series data with patterns.

    Creates time-series data with trends, seasonality, and noise.

    Example:
        generator = TimeSeriesDataGenerator()
        config = GeneratorConfig(row_count=100_000)
        df = generator.generate(config)
    """

    name = "timeseries"
    description = "Time-series data with patterns"

    def __init__(
        self,
        seed: int | None = None,
        frequency: str = "1h",
        trend: float = 0.0,
        seasonality_period: int = 24,
        seasonality_amplitude: float = 10.0,
        noise_std: float = 5.0,
    ):
        super().__init__(seed)
        self.frequency = frequency
        self.trend = trend
        self.seasonality_period = seasonality_period
        self.seasonality_amplitude = seasonality_amplitude
        self.noise_std = noise_std

    def generate(self, config: GeneratorConfig) -> pl.DataFrame:
        """Generate time-series data."""
        config = self._begin(config)

        n = config.row_count
        import math

        timestamps = self._timestamps(n, self.frequency)

        # Generate values with trend + seasonality + noise
        values = []
        for i in range(n):
            trend_component = self.trend * i
            seasonal_component = self.seasonality_amplitude * math.sin(
                2 * math.pi * i / self.seasonality_period
            )
            noise_component = self._rng.gauss(0, self.noise_std)
            value = 100 + trend_component + seasonal_component + noise_component
            values.append(value)

        return self._finish(
            pl.DataFrame(
                {
                    "timestamp": timestamps,
                    "value": values,
                    "hour": [(i % 24) for i in range(n)],
                    "day_of_week": [(i // 24) % 7 for i in range(n)],
                }
            ),
            config,
        )


# =============================================================================
# Financial Data Generator
# =============================================================================


class FinancialDataGenerator(DataGenerator):
    """Generator for financial market data.

    Creates realistic OHLCV (Open, High, Low, Close, Volume) data
    for stocks and cryptocurrencies.

    Example:
        generator = FinancialDataGenerator(symbol="AAPL")
        config = GeneratorConfig(row_count=1_000_000)
        df = generator.generate(config)
    """

    name = "financial"
    description = "Stock/crypto OHLCV data"

    def __init__(
        self,
        seed: int | None = None,
        symbol: str = "AAPL",
        base_price: float = 100.0,
        volatility: float = 0.02,
        frequency: str = "1s",
    ):
        super().__init__(seed)
        self.symbol = symbol
        self.base_price = base_price
        self.volatility = volatility
        self.frequency = frequency

    def generate(self, config: GeneratorConfig) -> pl.DataFrame:
        """Generate OHLCV data."""
        config = self._begin(config)

        n = config.row_count

        timestamps = self._timestamps(n, self.frequency)

        # Generate price using geometric Brownian motion
        prices = [self.base_price] if n else []
        for _ in range(n - 1):
            drift = 0.0001  # Small upward drift
            shock = self._rng.gauss(0, self.volatility)
            prices.append(prices[-1] * (1 + drift + shock))

        # Generate OHLCV
        opens = prices
        highs = [p * (1 + abs(self._rng.gauss(0, self.volatility / 2))) for p in prices]
        lows = [p * (1 - abs(self._rng.gauss(0, self.volatility / 2))) for p in prices]
        closes = [lows[i] + self._rng.random() * (highs[i] - lows[i]) for i in range(n)]
        volumes = [int(self._rng.expovariate(1 / 100000)) for _ in range(n)]

        return self._finish(
            pl.DataFrame(
                {
                    "timestamp": timestamps,
                    "symbol": [self.symbol] * n,
                    "open": opens,
                    "high": highs,
                    "low": lows,
                    "close": closes,
                    "volume": volumes,
                    "vwap": [
                        (high + low + close) / 3
                        for high, low, close in zip(highs, lows, closes, strict=True)
                    ],
                }
            ),
            config,
        )


# =============================================================================
# Text Data Generator
# =============================================================================


class TextDataGenerator(DataGenerator):
    """Generator for text-heavy data with PII patterns.

    Creates data with emails, names, addresses, and other
    PII-like patterns for testing scanners.

    Example:
        generator = TextDataGenerator()
        config = GeneratorConfig(row_count=100_000)
        df = generator.generate(config)
    """

    name = "text"
    description = "Text data with PII patterns"

    # Sample data pools
    FIRST_NAMES = [
        "James",
        "Mary",
        "John",
        "Patricia",
        "Robert",
        "Jennifer",
        "Michael",
        "Linda",
        "William",
        "Elizabeth",
        "David",
        "Barbara",
    ]
    LAST_NAMES = [
        "Smith",
        "Johnson",
        "Williams",
        "Brown",
        "Jones",
        "Garcia",
        "Miller",
        "Davis",
        "Rodriguez",
        "Martinez",
        "Hernandez",
        "Lopez",
    ]
    DOMAINS = ["gmail.com", "yahoo.com", "outlook.com", "company.com"]
    CITIES = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"]
    STATES = ["NY", "CA", "IL", "TX", "AZ"]

    def generate(self, config: GeneratorConfig) -> pl.DataFrame:
        """Generate text data with PII."""
        config = self._begin(config)

        n = config.row_count

        first_names = [self._rng.choice(self.FIRST_NAMES) for _ in range(n)]
        last_names = [self._rng.choice(self.LAST_NAMES) for _ in range(n)]

        return self._finish(
            pl.DataFrame(
                {
                    "id": pl.arange(0, n, eager=True),
                    "first_name": first_names,
                    "last_name": last_names,
                    "email": [
                        f"{fn.lower()}.{ln.lower()}{self._rng.randint(1, 999)}@{self._rng.choice(self.DOMAINS)}"
                        for fn, ln in zip(first_names, last_names, strict=True)
                    ],
                    "phone": self._generate_phones(n),
                    "ssn": [
                        f"{self._rng.randint(100, 999)}-{self._rng.randint(10, 99)}-{self._rng.randint(1000, 9999)}"
                        for _ in range(n)
                    ],
                    "credit_card": [
                        f"{self._rng.randint(4000, 5999)}-{self._rng.randint(1000, 9999)}-"
                        f"{self._rng.randint(1000, 9999)}-{self._rng.randint(1000, 9999)}"
                        for _ in range(n)
                    ],
                    "ip_address": self._generate_ips(n),
                    "address": [
                        f"{self._rng.randint(100, 9999)} {self._rng.choice(['Main', 'Oak', 'Pine', 'Elm'])} "
                        f"{self._rng.choice(['St', 'Ave', 'Blvd', 'Rd'])}"
                        for _ in range(n)
                    ],
                    "city": [self._rng.choice(self.CITIES) for _ in range(n)],
                    "state": [self._rng.choice(self.STATES) for _ in range(n)],
                    "zip_code": [f"{self._rng.randint(10000, 99999)}" for _ in range(n)],
                }
            ),
            config,
        )


# =============================================================================
# Registry
# =============================================================================


class GeneratorRegistry:
    """Registry for data generator discovery and management."""

    def __init__(self) -> None:
        self._generators: dict[str, type[DataGenerator]] = {}
        self._lock = threading.Lock()

    def register(
        self,
        generator_class: type[DataGenerator],
    ) -> type[DataGenerator]:
        """Register a generator class."""
        with self._lock:
            name = generator_class.name
            if name in self._generators:
                raise ValueError(f"Generator '{name}' is already registered")
            self._generators[name] = generator_class
        return generator_class

    def get(self, name: str) -> type[DataGenerator]:
        """Get a generator class by name."""
        with self._lock:
            if name not in self._generators:
                available = list(self._generators.keys())
                raise KeyError(f"Generator '{name}' not found. Available: {available}")
            return self._generators[name]

    def list_names(self) -> list[str]:
        """List all registered generator names."""
        with self._lock:
            return list(self._generators.keys())

    def create(self, name: str, **kwargs: Any) -> DataGenerator:
        """Create a generator instance by name."""
        generator_class = self.get(name)
        return generator_class(**kwargs)


# Global registry instance
generator_registry = GeneratorRegistry()

# Register built-in generators
generator_registry.register(TabularDataGenerator)
generator_registry.register(TimeSeriesDataGenerator)
generator_registry.register(FinancialDataGenerator)
generator_registry.register(TextDataGenerator)


def register_generator(cls: type[DataGenerator]) -> type[DataGenerator]:
    """Decorator to register a generator with the global registry."""
    return generator_registry.register(cls)
