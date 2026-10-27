"""Native profile failures must remain visible through best-effort execution and restore."""

import json

import polars as pl
import pytest

from truthound.profiler import ColumnProfiler, DataProfiler, PhysicalTypeInferrer
from truthound.profiler.base import TypeInferrer
from truthound.profiler.column_profiler import BasicStatsAnalyzer, ColumnAnalyzer
from truthound.profiler.table_profiler import TableAnalyzer

pytestmark = pytest.mark.fault


class FailedAnalyzer(ColumnAnalyzer):
    name = "failed-analyzer"

    def analyze(self, *args):
        raise RuntimeError("synthetic-private-analyzer-query")


class FailedInferrer(TypeInferrer):
    name = "failed-inferrer"
    priority = 10000

    def infer(self, *args):
        raise RuntimeError("synthetic-private-inference")


class FailedTable(TableAnalyzer):
    name = "failed-table"

    def analyze(self, *args):
        raise RuntimeError("synthetic-private-table-query")


@pytest.mark.parametrize("failure", ["analyzer", "inferrer", "table", "parallel"])
def test_native_failures_are_incomplete_without_private_messages(failure):
    columns = ColumnProfiler(
        analyzers=[BasicStatsAnalyzer(), FailedAnalyzer()]
        if failure == "analyzer"
        else [BasicStatsAnalyzer()],
        type_inferrers=[FailedInferrer(), PhysicalTypeInferrer()]
        if failure == "inferrer"
        else [PhysicalTypeInferrer()],
    )
    if failure == "parallel":
        native = columns.profile_column

        def broken(name, *args):
            if name == "y":
                raise RuntimeError("synthetic-private-column-failure")
            return native(name, *args)

        columns.profile_column = broken
    profile = DataProfiler(
        column_profiler=columns,
        table_analyzers=[FailedTable()] if failure == "table" else [],
        parallel=failure == "parallel",
    ).profile(pl.DataFrame({"x": [1, None], "y": [2, 3]}).lazy())
    assert not profile.complete
    issues = list(profile.execution_issues) + [
        issue for column in profile.columns for issue in column.execution_issues
    ]
    assert issues and all(issue.error_type == "RuntimeError" for issue in issues)
    assert "synthetic-private-" not in json.dumps(profile.to_dict())
    if failure in {"analyzer", "inferrer"}:
        assert all(
            not column.complete and not column.suggested_validators for column in profile.columns
        )
    if failure == "parallel":
        assert profile.column_count == 2 and len(profile.columns) == 1
        assert issues[0].column == "y"


@pytest.mark.parametrize("format", ["versioned", "plain"])
def test_native_incomplete_profile_roundtrip_does_not_become_complete(tmp_path, format):
    profile = DataProfiler(
        column_profiler=ColumnProfiler(analyzers=[BasicStatsAnalyzer(), FailedAnalyzer()]),
        table_analyzers=[FailedTable()],
    ).profile(pl.DataFrame({"x": [1, None]}).lazy())
    if format == "versioned":
        from truthound.profiler.schema import ProfileSerializer

        native = ProfileSerializer()
        restored = native.deserialize(native.serialize(profile))
    else:
        from truthound.profiler.table_profiler import load_profile, save_profile

        path = tmp_path / "partial.json"
        save_profile(profile, path)
        restored = load_profile(path)
    assert not restored.complete and restored.execution_issues == profile.execution_issues
    assert restored["x"].execution_issues == profile["x"].execution_issues
    assert restored.to_dict() == profile.to_dict()


def test_native_config_controls_sampling_correlations_and_distribution_selection():
    from truthound.profiler import ProfilerConfig

    frame = pl.DataFrame({"x": list(range(10)), "y": [value * 2 for value in range(10)]}).lazy()
    config = ProfilerConfig(sample_size=3, include_correlations=True, include_distributions=False)
    profile = DataProfiler(config=config).profile(frame)
    assert profile.row_count == 3 and all(column.row_count == 3 for column in profile.columns)
    assert profile.correlations and profile.correlations[0][:2] == ("x", "y")
    assert all(column.distribution is None for column in profile.columns)
    # Explicit legacy options override the provided config without mutating it.
    overridden = DataProfiler(config=config, sample_size=2, include_correlations=False).profile(
        frame
    )
    assert overridden.row_count == 2 and not overridden.correlations
    assert config.sample_size == 3 and config.include_correlations


def test_native_correlation_pair_failure_is_not_silently_complete(monkeypatch):
    from truthound.profiler.table_profiler import CorrelationAnalyzer

    def failed(*args, **kwargs):
        raise RuntimeError("synthetic-private-correlation")

    monkeypatch.setattr(pl, "corr", failed)
    profile = DataProfiler(table_analyzers=[CorrelationAnalyzer()]).profile(
        pl.DataFrame({"x": [1, 2], "y": [2, 4]}).lazy()
    )
    assert not profile.complete and profile.execution_issues[0].component == "correlation"
    assert "synthetic-private-correlation" not in json.dumps(profile.to_dict())


def test_native_automatic_suite_rejects_incomplete_profile_before_any_generator():
    from truthound.profiler.errors import IncompleteProfileError
    from truthound.profiler.generators.suite_generator import ValidationSuiteGenerator

    profile = DataProfiler(column_profiler=ColumnProfiler(analyzers=[FailedAnalyzer()])).profile(
        pl.DataFrame({"x": [1, 2]}).lazy()
    )

    class Forbidden:
        name = "forbidden"
        categories = set()

        def generate(self, *args):
            raise AssertionError("Must not consume incomplete profile")

    with pytest.raises(IncompleteProfileError):
        ValidationSuiteGenerator(generators=[Forbidden()]).generate_from_profile(profile)


def test_native_suite_generator_failure_cannot_return_a_successful_partial_suite():
    from truthound.profiler.errors import ProfilerError
    from truthound.profiler.generators.suite_generator import ValidationSuiteGenerator

    profile = DataProfiler().profile(pl.DataFrame({"x": [1, 2]}).lazy())

    class FailedGenerator:
        name = "failed-generator"
        categories = set()

        def generate(self, *args):
            raise RuntimeError("synthetic-private-rule-generation")

    with pytest.raises(ProfilerError) as failure:
        ValidationSuiteGenerator(generators=[FailedGenerator()]).generate_from_profile(profile)
    assert "synthetic-private-rule-generation" not in str(failure.value)


@pytest.mark.parametrize("explicit_only", [False, True])
def test_native_suite_dictionary_adapter_retains_incomplete_verdict(explicit_only):
    from truthound.profiler.errors import IncompleteProfileError
    from truthound.profiler.generators.suite_generator import generate_suite

    profile = DataProfiler(column_profiler=ColumnProfiler(analyzers=[FailedAnalyzer()])).profile(
        pl.DataFrame({"x": [1, 2]}).lazy()
    )
    data = profile.to_dict()
    if explicit_only:
        data.pop("execution_issues", None)
        for column in data["columns"]:
            column.pop("execution_issues", None)
    with pytest.raises(IncompleteProfileError) as failure:
        generate_suite(data)
    assert not failure.value.profile.complete


def test_native_suite_dictionary_adapter_preserves_numeric_rules():
    from truthound.profiler.generators.suite_generator import generate_suite

    profile = DataProfiler().profile(pl.DataFrame({"x": [1, 2, 3]}).lazy())
    direct = generate_suite(profile)
    restored = generate_suite(profile.to_dict())
    assert [rule.to_dict() for rule in restored.rules] == [rule.to_dict() for rule in direct.rules]
    assert profile["x"].distribution is not None
    assert any(rule.validator_class == "RangeValidator" for rule in direct.rules)


def test_native_suite_rejects_legacy_report_with_missing_columns():
    from truthound.profiler.errors import IncompleteProfileError
    from truthound.profiler.generators.suite_generator import generate_suite

    with pytest.raises(IncompleteProfileError):
        generate_suite({"source": "synthetic", "row_count": 3, "column_count": 2, "columns": []})


def test_native_profiler_guard_stops_following_component_after_revocation():
    revoked = False
    events = []

    def guard():
        if revoked:
            raise PermissionError("revoked")

    class First(ColumnAnalyzer):
        name = "first"

        def analyze(self, *args):
            nonlocal revoked
            revoked = True
            events.append("first")
            return {}

    class Following(ColumnAnalyzer):
        name = "following"

        def analyze(self, *args):
            events.append("following")
            return {}

    native = DataProfiler(column_profiler=ColumnProfiler(analyzers=[First(), Following()]))
    native.set_execution_guard(guard)
    with pytest.raises(PermissionError):
        native.profile(pl.DataFrame({"x": [1]}).lazy())
    assert events == ["first"]


@pytest.mark.parametrize("kind", ["plain", "versioned", "suite"])
@pytest.mark.parametrize("malformed", [{"$type": "object", "entries": {}}, {}, {"rules": "not-a-list", "columns": "not-a-list"}])
def test_native_restore_rejects_corrupt_or_wrong_encoding_instead_of_empty_success(tmp_path, kind, malformed):
    from truthound.profiler.generators.suite_generator import load_suite
    from truthound.profiler.schema import ProfileSerializer
    from truthound.profiler.table_profiler import load_profile

    path = tmp_path / "corrupt.json"
    path.write_text(json.dumps(malformed))
    with pytest.raises((ValueError, TypeError)):
        if kind == "versioned":
            ProfileSerializer().deserialize(malformed)
        elif kind == "plain":
            load_profile(path)
        else:
            load_suite(path)
    assert json.loads(path.read_text()) == malformed
