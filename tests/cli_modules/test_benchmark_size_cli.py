from datetime import datetime

import pytest
from typer.testing import CliRunner

from truthound.benchmark import BenchmarkRunner, BenchmarkSize
from truthound.benchmark.base import (
    BenchmarkCategory,
    BenchmarkMetrics,
    BenchmarkResult,
    EnvironmentInfo,
)
from truthound.cli_modules.advanced.benchmark import app


@pytest.mark.contract
@pytest.mark.parametrize('override,expected', [(None, 100_000_000), ('17', 17)])
def test_stress_size_reaches_runner_without_allocating_input(monkeypatch, override, expected):
    calls = []

    def run(self, name, **kwargs):
        calls.append((name, kwargs['row_count'], self.config.size_override))
        return BenchmarkResult(
            benchmark_name=name,
            category=BenchmarkCategory.PROFILING,
            success=True,
            metrics=BenchmarkMetrics(),
            environment=EnvironmentInfo.capture(),
            completed_at=datetime.now(),
        )

    monkeypatch.setattr(BenchmarkRunner, 'run', run)
    args = ['run', 'profile', '--size', 'stress', '--format', 'json']
    if override:
        args += ['--rows', override]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert calls == [('profile', expected, None if override else BenchmarkSize.STRESS)]
