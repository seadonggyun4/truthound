"""Native generated workloads must honor seed, size and timestamp frequency."""

import pytest

from truthound.benchmark.generators import (
    ColumnSpec,
    ColumnType,
    DataPattern,
    FinancialDataGenerator,
    GeneratorConfig,
    TabularDataGenerator,
    TimeSeriesDataGenerator,
)


def test_default_generator_uses_requested_seed_and_null_ratio():
    generator = TabularDataGenerator(seed=99)
    first = generator.generate(GeneratorConfig(row_count=50, seed=0, null_ratio=0.2))
    second = generator.generate(GeneratorConfig(row_count=50, seed=1, null_ratio=0.2))
    assert not first.equals(second)
    assert first.equals(generator.generate(GeneratorConfig(row_count=50, seed=0, null_ratio=0.2)))
    assert any(value > 0 for value in first.null_count().row(0))


@pytest.mark.parametrize(
    "generator", [TimeSeriesDataGenerator(frequency="1d"), FinancialDataGenerator(frequency="1h")]
)
def test_time_generators_preserve_requested_rows_at_nondefault_frequency(generator):
    frame = generator.generate(GeneratorConfig(row_count=5, seed=0, null_ratio=0))
    assert frame.height == 5
    assert frame["timestamp"].n_unique() == 5


def test_seeded_uuid_and_zero_numeric_bound_are_respected():
    generator = TabularDataGenerator()
    config = GeneratorConfig(
        row_count=10,
        seed=0,
        null_ratio=0,
        column_specs=[
            ColumnSpec("identity", ColumnType.UUID),
            ColumnSpec("zero", ColumnType.INTEGER, min_value=0, max_value=0),
        ],
    )
    first = generator.generate(config)
    assert first.equals(generator.generate(config))
    assert first["zero"].to_list() == [0] * 10


@pytest.mark.parametrize(
    "generator", [FinancialDataGenerator(), TimeSeriesDataGenerator(), TabularDataGenerator()]
)
def test_empty_generation_retains_empty_shape(generator):
    assert generator.generate(GeneratorConfig(row_count=0, null_ratio=0)).height == 0


@pytest.mark.parametrize(
    "generator", [FinancialDataGenerator(), TimeSeriesDataGenerator(), TabularDataGenerator()]
)
def test_native_generators_apply_nulls_and_duplicates_without_increasing_rows(generator):
    frame = generator.generate(
        GeneratorConfig(row_count=50, seed=42, null_ratio=0.2, duplicate_ratio=0.2)
    )
    assert frame.height == 50
    assert frame.unique().height <= 40
    assert any(value > 0 for value in frame.null_count().row(0))


def test_text_generator_nulls_are_configurable_without_losing_schema():
    from truthound.benchmark.generators import TextDataGenerator

    frame = TextDataGenerator().generate(GeneratorConfig(row_count=50, seed=42, null_ratio=1))
    assert frame["email"].null_count() == 50
    assert str(frame.schema["email"]) == "String"


def test_generated_cardinality_json_url_and_numeric_patterns_are_real():
    import json

    config = GeneratorConfig(
        row_count=24,
        seed=4,
        null_ratio=0,
        column_specs=[
            ColumnSpec("few", ColumnType.STRING, cardinality=3),
            ColumnSpec("json", ColumnType.JSON),
            ColumnSpec("url", ColumnType.URL),
            ColumnSpec(
                "constant", ColumnType.FLOAT, DataPattern.CONSTANT, min_value=7, max_value=9
            ),
            ColumnSpec(
                "cyclic",
                ColumnType.INTEGER,
                DataPattern.CYCLIC,
                min_value=0,
                max_value=5,
                params={"period": 3},
            ),
        ],
    )
    frame = TabularDataGenerator().generate(config)
    assert frame["few"].n_unique() <= 3
    assert all(isinstance(json.loads(value), dict) for value in frame["json"])
    assert all(value.startswith("https://example.com/") for value in frame["url"])
    assert frame["constant"].to_list() == [7.0] * 24
    assert frame["cyclic"].to_list()[:6] == [0, 1, 2, 0, 1, 2]


def test_parallel_columns_preserve_values_and_chunk_size_and_explicit_zero_nulls():
    from dataclasses import replace

    config = GeneratorConfig(
        row_count=20,
        seed=5,
        null_ratio=1,
        chunk_size=7,
        column_specs=[
            ColumnSpec("x", ColumnType.INTEGER, null_ratio=0),
            ColumnSpec("y", ColumnType.STRING, params={"min_length": 3, "max_length": 3}),
        ],
    )
    parallel = TabularDataGenerator().generate(config)
    sequential = TabularDataGenerator().generate(replace(config, parallelize=False))
    assert parallel.equals(sequential)
    assert [chunk.len() for chunk in parallel["x"].get_chunks()] == [7, 7, 6]
    assert parallel["x"].null_count() == 0
    assert parallel["y"].null_count() == 20


@pytest.mark.parametrize(
    "pattern",
    list(DataPattern),
)
def test_numeric_patterns_respect_finite_bounds_and_zero_rows(pattern):
    config = GeneratorConfig(
        row_count=24,
        seed=5,
        null_ratio=0,
        column_specs=[
            ColumnSpec("x", ColumnType.FLOAT, pattern, min_value=-3, max_value=0),
        ],
    )
    frame = TabularDataGenerator().generate(config)
    assert all(-3 <= value <= 0 for value in frame["x"].drop_nulls())
    empty = TabularDataGenerator().generate(config.with_size(0))
    assert empty.height == 0 and str(empty.schema["x"]) == "Float64"
