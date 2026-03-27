import polars as pl

from truthound.core import ScanPlanner, ValidationRuntime, ValidationSuite, build_validation_asset
from truthound.datasources.sql import SQLiteDataSource


def test_polars_backend_conformance_exposes_polars_asset():
    asset = build_validation_asset(pl.DataFrame({'id': [1, 2], 'value': ['a', 'b']}))

    assert asset.backend_name == 'polars'
    assert asset.capabilities.parallel is True
    assert asset.capabilities.pushdown is False
    assert asset.to_lazyframe().collect().shape == (2, 2)


def test_sql_pushdown_backend_conformance_executes(tmp_path):
    df = pl.DataFrame({'id': [1, 2], 'email': ['a@example.com', None]})
    source = SQLiteDataSource.from_dataframe(
        df,
        table='users',
        database=str(tmp_path / 'truthound-core.db'),
    )
    suite = ValidationSuite.from_legacy(validators=['null'])
    asset = build_validation_asset(source=source, pushdown=True)
    plan = ScanPlanner().plan(suite=suite, asset=asset, pushdown=True)
    run_result = ValidationRuntime().execute(asset=asset, plan=plan)

    assert asset.backend_name == 'sql'
    assert plan.execution_mode == 'pushdown'
    assert plan.planned_execution_mode == 'pushdown'
    assert run_result.planned_execution_mode == 'pushdown'
    assert run_result.execution_mode == 'pushdown'
    assert any(issue.issue_type == 'null' for issue in run_result.issues)
