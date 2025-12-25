"""E2E Test Fixtures and Data Generators.

This module provides reusable test fixtures and synthetic data generators
for comprehensive E2E testing of Truthound's core paths.

Features:
- Multiple data format generators (CSV, Parquet, JSON)
- Configurable data characteristics (nulls, patterns, anomalies)
- Scenario-based fixture organization
- Factory functions for common test setups

Example:
    from tests.e2e.fixtures import DataGenerator, create_test_data

    # Generate test data with specific characteristics
    generator = DataGenerator(
        row_count=1000,
        null_ratio=0.05,
        include_pii=True,
    )
    df = generator.generate()

    # Use factory function
    data_file = create_test_data(tmp_path, format="csv", scenario="pii")
"""

from __future__ import annotations

import json
import random
import string
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Sequence, TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from truthound.profiler import TableProfile, ValidationSuite


# =============================================================================
# Enums and Types
# =============================================================================


class DataScenario(str, Enum):
    """Pre-defined data scenarios for testing."""

    CLEAN = "clean"  # No issues
    NULLS = "nulls"  # Contains null values
    PII = "pii"  # Contains PII data
    ANOMALIES = "anomalies"  # Contains anomalies
    MIXED = "mixed"  # Mixed data quality issues
    EDGE_CASES = "edge_cases"  # Edge case values
    LARGE = "large"  # Large dataset
    KOREAN = "korean"  # Korean language data
    MULTITYPE = "multitype"  # Multiple data types


class DataFormat(str, Enum):
    """Supported data file formats."""

    CSV = "csv"
    PARQUET = "parquet"
    JSON = "json"
    NDJSON = "ndjson"


# =============================================================================
# Data Generator Configuration
# =============================================================================


@dataclass
class ColumnSpec:
    """Specification for generating a column."""

    name: str
    dtype: str  # "int", "float", "string", "bool", "date", "datetime"
    null_ratio: float = 0.0
    unique_ratio: float = 0.5
    pattern: str | None = None  # "email", "phone", "uuid", etc.
    min_value: float | None = None
    max_value: float | None = None
    categories: list[str] | None = None
    generator: Callable[[], Any] | None = None


@dataclass
class DataConfig:
    """Configuration for data generation."""

    row_count: int = 1000
    columns: list[ColumnSpec] = field(default_factory=list)
    seed: int | None = None
    scenario: DataScenario = DataScenario.CLEAN

    def __post_init__(self) -> None:
        if self.seed is not None:
            random.seed(self.seed)


# =============================================================================
# Value Generators
# =============================================================================


class ValueGenerator:
    """Generator for various data types and patterns."""

    # Korean data patterns
    KOREAN_NAMES = [
        "김민수", "이영희", "박지훈", "최수진", "정현우",
        "강서연", "조민준", "윤하늘", "장예진", "임도현",
    ]
    KOREAN_CITIES = [
        "서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종",
    ]

    # English data patterns
    FIRST_NAMES = [
        "Alice", "Bob", "Charlie", "Diana", "Edward",
        "Fiona", "George", "Hannah", "Ivan", "Julia",
    ]
    LAST_NAMES = [
        "Smith", "Johnson", "Williams", "Brown", "Jones",
        "Garcia", "Miller", "Davis", "Rodriguez", "Martinez",
    ]
    DOMAINS = ["example.com", "test.org", "sample.net", "demo.io"]

    @classmethod
    def generate_string(cls, length: int = 10) -> str:
        """Generate random string."""
        return "".join(random.choices(string.ascii_letters, k=length))

    @classmethod
    def generate_email(cls) -> str:
        """Generate fake email address."""
        name = random.choice(cls.FIRST_NAMES).lower()
        domain = random.choice(cls.DOMAINS)
        return f"{name}{random.randint(1, 999)}@{domain}"

    @classmethod
    def generate_phone(cls) -> str:
        """Generate fake phone number."""
        return f"010-{random.randint(1000, 9999)}-{random.randint(1000, 9999)}"

    @classmethod
    def generate_uuid(cls) -> str:
        """Generate fake UUID."""
        import uuid
        return str(uuid.uuid4())

    @classmethod
    def generate_ip(cls) -> str:
        """Generate fake IP address."""
        return ".".join(str(random.randint(0, 255)) for _ in range(4))

    @classmethod
    def generate_korean_rrn(cls) -> str:
        """Generate fake Korean RRN (주민등록번호)."""
        # Format: YYMMDD-GNNNNNN
        year = random.randint(70, 99)
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        gender = random.choice([1, 2])
        rest = random.randint(100000, 999999)
        return f"{year:02d}{month:02d}{day:02d}-{gender}{rest}"

    @classmethod
    def generate_korean_name(cls) -> str:
        """Generate Korean name."""
        return random.choice(cls.KOREAN_NAMES)

    @classmethod
    def generate_name(cls) -> str:
        """Generate English name."""
        return f"{random.choice(cls.FIRST_NAMES)} {random.choice(cls.LAST_NAMES)}"

    @classmethod
    def generate_date(
        cls,
        start_year: int = 2020,
        end_year: int = 2024,
    ) -> datetime:
        """Generate random date."""
        start = datetime(start_year, 1, 1)
        end = datetime(end_year, 12, 31)
        delta = end - start
        random_days = random.randint(0, delta.days)
        return start + timedelta(days=random_days)

    @classmethod
    def generate_float(
        cls,
        min_val: float = 0.0,
        max_val: float = 100.0,
    ) -> float:
        """Generate random float."""
        return random.uniform(min_val, max_val)

    @classmethod
    def generate_int(
        cls,
        min_val: int = 0,
        max_val: int = 1000,
    ) -> int:
        """Generate random integer."""
        return random.randint(min_val, max_val)


# =============================================================================
# Data Generator
# =============================================================================


class DataGenerator:
    """Main data generator for E2E tests.

    Generates synthetic data with configurable characteristics
    for testing various scenarios.

    Example:
        generator = DataGenerator(
            row_count=1000,
            null_ratio=0.05,
            include_pii=True,
        )
        df = generator.generate()
    """

    def __init__(
        self,
        row_count: int = 1000,
        null_ratio: float = 0.0,
        include_pii: bool = False,
        include_korean: bool = False,
        seed: int | None = None,
    ):
        self.row_count = row_count
        self.null_ratio = null_ratio
        self.include_pii = include_pii
        self.include_korean = include_korean

        if seed is not None:
            random.seed(seed)

    def generate(self) -> pl.DataFrame:
        """Generate a DataFrame with configured characteristics."""
        data: dict[str, list[Any]] = {}

        # Basic columns
        data["id"] = list(range(1, self.row_count + 1))
        data["name"] = self._generate_column(
            ValueGenerator.generate_name,
            self.null_ratio,
        )
        data["age"] = self._generate_column(
            lambda: ValueGenerator.generate_int(18, 80),
            self.null_ratio / 2,  # Less nulls for numeric
        )
        data["score"] = self._generate_column(
            lambda: round(ValueGenerator.generate_float(0, 100), 2),
            self.null_ratio,
        )
        data["created_at"] = self._generate_column(
            lambda: ValueGenerator.generate_date().isoformat(),
            0.0,  # No nulls for dates
        )

        # PII columns
        if self.include_pii:
            data["email"] = self._generate_column(
                ValueGenerator.generate_email,
                self.null_ratio,
            )
            data["phone"] = self._generate_column(
                ValueGenerator.generate_phone,
                self.null_ratio,
            )
            data["ip_address"] = self._generate_column(
                ValueGenerator.generate_ip,
                self.null_ratio * 2,
            )

        # Korean data columns
        if self.include_korean:
            data["korean_name"] = self._generate_column(
                ValueGenerator.generate_korean_name,
                self.null_ratio,
            )
            data["rrn"] = self._generate_column(
                ValueGenerator.generate_korean_rrn,
                self.null_ratio,
            )

        return pl.DataFrame(data)

    def _generate_column(
        self,
        generator: Callable[[], Any],
        null_ratio: float,
    ) -> list[Any]:
        """Generate a column with optional nulls."""
        values = []
        for _ in range(self.row_count):
            if random.random() < null_ratio:
                values.append(None)
            else:
                values.append(generator())
        return values


# =============================================================================
# Format-Specific Generators
# =============================================================================


class FileGenerator(ABC):
    """Abstract base class for file generators."""

    format_name: str = "base"
    file_extension: str = ".txt"

    def __init__(self, data_generator: DataGenerator | None = None):
        self.data_generator = data_generator or DataGenerator()

    @abstractmethod
    def generate(self, path: Path) -> Path:
        """Generate file at the given path."""
        pass

    def _ensure_extension(self, path: Path) -> Path:
        """Ensure correct file extension."""
        if path.suffix != self.file_extension:
            return path.with_suffix(self.file_extension)
        return path


class CSVGenerator(FileGenerator):
    """CSV file generator."""

    format_name = "csv"
    file_extension = ".csv"

    def generate(self, path: Path) -> Path:
        """Generate CSV file."""
        path = self._ensure_extension(path)
        df = self.data_generator.generate()
        df.write_csv(path)
        return path


class ParquetGenerator(FileGenerator):
    """Parquet file generator."""

    format_name = "parquet"
    file_extension = ".parquet"

    def generate(self, path: Path) -> Path:
        """Generate Parquet file."""
        path = self._ensure_extension(path)
        df = self.data_generator.generate()
        df.write_parquet(path)
        return path


class JSONGenerator(FileGenerator):
    """JSON file generator (newline-delimited)."""

    format_name = "json"
    file_extension = ".ndjson"

    def generate(self, path: Path) -> Path:
        """Generate NDJSON file."""
        path = self._ensure_extension(path)
        df = self.data_generator.generate()
        df.write_ndjson(path)
        return path


# =============================================================================
# Scenario Fixtures
# =============================================================================


@dataclass
class ScenarioFixture:
    """Base class for scenario-based test fixtures."""

    name: str
    description: str
    tmp_path: Path

    def setup(self) -> None:
        """Set up the fixture."""
        pass

    def teardown(self) -> None:
        """Tear down the fixture."""
        pass


@dataclass
class ValidationScenario(ScenarioFixture):
    """Fixture for validation testing scenarios."""

    data_file: Path | None = None
    schema_file: Path | None = None
    expected_issues: int = 0
    expected_issue_types: list[str] = field(default_factory=list)

    def setup(self) -> None:
        """Set up validation scenario."""
        if self.data_file is None:
            generator = DataGenerator(row_count=100, null_ratio=0.1)
            csv_gen = CSVGenerator(generator)
            self.data_file = csv_gen.generate(self.tmp_path / "data.csv")


@dataclass
class ProfilingScenario(ScenarioFixture):
    """Fixture for profiling testing scenarios."""

    data_file: Path | None = None
    expected_columns: int = 0
    expected_row_count: int = 0
    expected_patterns: list[str] = field(default_factory=list)

    def setup(self) -> None:
        """Set up profiling scenario."""
        if self.data_file is None:
            generator = DataGenerator(
                row_count=1000,
                null_ratio=0.05,
                include_pii=True,
            )
            parquet_gen = ParquetGenerator(generator)
            self.data_file = parquet_gen.generate(self.tmp_path / "data.parquet")
            self.expected_row_count = 1000
            self.expected_columns = 8  # Basic + PII columns


@dataclass
class SuiteGenerationScenario(ScenarioFixture):
    """Fixture for suite generation testing scenarios."""

    profile_file: Path | None = None
    expected_rule_count: int = 0
    expected_categories: list[str] = field(default_factory=list)

    def setup(self) -> None:
        """Set up suite generation scenario."""
        pass  # Profile must be created separately


# =============================================================================
# Pre-defined Scenarios
# =============================================================================


def create_clean_data_scenario(
    tmp_path: Path,
    row_count: int = 100,
) -> tuple[Path, pl.DataFrame]:
    """Create a clean data scenario with no quality issues."""
    generator = DataGenerator(
        row_count=row_count,
        null_ratio=0.0,
        seed=42,
    )
    df = generator.generate()
    data_file = tmp_path / "clean_data.csv"
    df.write_csv(data_file)
    return data_file, df


def create_null_data_scenario(
    tmp_path: Path,
    row_count: int = 100,
    null_ratio: float = 0.2,
) -> tuple[Path, pl.DataFrame]:
    """Create a data scenario with null values."""
    generator = DataGenerator(
        row_count=row_count,
        null_ratio=null_ratio,
        seed=42,
    )
    df = generator.generate()
    data_file = tmp_path / "null_data.csv"
    df.write_csv(data_file)
    return data_file, df


def create_pii_data_scenario(
    tmp_path: Path,
    row_count: int = 100,
) -> tuple[Path, pl.DataFrame]:
    """Create a data scenario with PII data."""
    generator = DataGenerator(
        row_count=row_count,
        null_ratio=0.05,
        include_pii=True,
        seed=42,
    )
    df = generator.generate()
    data_file = tmp_path / "pii_data.parquet"
    df.write_parquet(data_file)
    return data_file, df


def create_korean_data_scenario(
    tmp_path: Path,
    row_count: int = 100,
) -> tuple[Path, pl.DataFrame]:
    """Create a data scenario with Korean data."""
    generator = DataGenerator(
        row_count=row_count,
        null_ratio=0.05,
        include_korean=True,
        seed=42,
    )
    df = generator.generate()
    data_file = tmp_path / "korean_data.csv"
    df.write_csv(data_file)
    return data_file, df


def create_edge_case_scenario(
    tmp_path: Path,
) -> tuple[Path, pl.DataFrame]:
    """Create a data scenario with edge cases."""
    df = pl.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "empty_string": ["", "a", "", "b", ""],
        "whitespace": ["  ", " x ", "\t", "\n", "normal"],
        "unicode": ["🎉", "한글", "日本語", "العربية", "emoji 🚀"],
        "long_string": ["x" * 10000, "short", "y" * 5000, "a", "z" * 1000],
        "extreme_int": [0, -2147483648, 2147483647, 1, -1],
        "extreme_float": [0.0, float("inf"), float("-inf"), 1e308, 1e-308],
        "special_values": ["null", "NULL", "None", "N/A", ""],
    })
    data_file = tmp_path / "edge_case_data.parquet"
    df.write_parquet(data_file)
    return data_file, df


def create_mixed_quality_scenario(
    tmp_path: Path,
    row_count: int = 500,
) -> tuple[Path, pl.DataFrame]:
    """Create a data scenario with mixed quality issues."""
    generator = DataGenerator(
        row_count=row_count,
        null_ratio=0.15,
        include_pii=True,
        include_korean=True,
        seed=42,
    )
    df = generator.generate()

    # Add some anomalies
    anomaly_df = pl.DataFrame({
        "id": [row_count + 1, row_count + 2],
        "name": ["OUTLIER_NAME_12345" * 10, None],
        "age": [999, -5],  # Anomalous ages
        "score": [1000.0, -100.0],  # Anomalous scores
        "created_at": ["2099-12-31", "1900-01-01"],
        "email": ["invalid-email", "also@invalid@email.com"],
        "phone": ["not-a-phone", "123"],
        "ip_address": ["999.999.999.999", "invalid"],
        "korean_name": [None, ""],
        "rrn": ["invalid-rrn", "000000-0000000"],
    })

    combined = pl.concat([df, anomaly_df], how="diagonal")
    data_file = tmp_path / "mixed_quality_data.parquet"
    combined.write_parquet(data_file)
    return data_file, combined


# =============================================================================
# Factory Functions
# =============================================================================


def create_test_data(
    tmp_path: Path,
    format: str = "csv",
    scenario: str = "clean",
    row_count: int = 100,
    **kwargs: Any,
) -> Path:
    """Factory function to create test data files.

    Args:
        tmp_path: Temporary directory path
        format: File format (csv, parquet, json)
        scenario: Data scenario (clean, nulls, pii, korean, edge_cases, mixed)
        row_count: Number of rows to generate
        **kwargs: Additional arguments for generator

    Returns:
        Path to the generated data file
    """
    scenarios = {
        "clean": lambda: create_clean_data_scenario(tmp_path, row_count),
        "nulls": lambda: create_null_data_scenario(
            tmp_path, row_count, kwargs.get("null_ratio", 0.2)
        ),
        "pii": lambda: create_pii_data_scenario(tmp_path, row_count),
        "korean": lambda: create_korean_data_scenario(tmp_path, row_count),
        "edge_cases": lambda: create_edge_case_scenario(tmp_path),
        "mixed": lambda: create_mixed_quality_scenario(tmp_path, row_count),
    }

    if scenario not in scenarios:
        raise ValueError(f"Unknown scenario: {scenario}. Available: {list(scenarios.keys())}")

    data_file, df = scenarios[scenario]()

    # Convert format if needed
    target_format = DataFormat(format)
    if target_format == DataFormat.CSV and not data_file.suffix == ".csv":
        new_path = data_file.with_suffix(".csv")
        df.write_csv(new_path)
        return new_path
    elif target_format == DataFormat.PARQUET and not data_file.suffix == ".parquet":
        new_path = data_file.with_suffix(".parquet")
        df.write_parquet(new_path)
        return new_path
    elif target_format in (DataFormat.JSON, DataFormat.NDJSON):
        new_path = data_file.with_suffix(".ndjson")
        df.write_ndjson(new_path)
        return new_path

    return data_file


def create_profile(
    data_file: Path,
    **profiler_kwargs: Any,
) -> "TableProfile":
    """Create a profile from a data file.

    Args:
        data_file: Path to data file
        **profiler_kwargs: Arguments for DataProfiler

    Returns:
        Generated TableProfile
    """
    from truthound.profiler import profile_file

    return profile_file(str(data_file), **profiler_kwargs)


def create_validation_suite(
    profile: "TableProfile",
    **suite_kwargs: Any,
) -> "ValidationSuite":
    """Create a validation suite from a profile.

    Args:
        profile: Table profile
        **suite_kwargs: Arguments for generate_suite

    Returns:
        Generated ValidationSuite
    """
    from truthound.profiler import generate_suite

    return generate_suite(profile, **suite_kwargs)


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Enums
    "DataScenario",
    "DataFormat",
    # Configuration
    "ColumnSpec",
    "DataConfig",
    # Generators
    "ValueGenerator",
    "DataGenerator",
    "FileGenerator",
    "CSVGenerator",
    "ParquetGenerator",
    "JSONGenerator",
    # Scenario Fixtures
    "ScenarioFixture",
    "ValidationScenario",
    "ProfilingScenario",
    "SuiteGenerationScenario",
    # Pre-defined Scenarios
    "create_clean_data_scenario",
    "create_null_data_scenario",
    "create_pii_data_scenario",
    "create_korean_data_scenario",
    "create_edge_case_scenario",
    "create_mixed_quality_scenario",
    # Factory Functions
    "create_test_data",
    "create_profile",
    "create_validation_suite",
]
