import json

import pytest
from typer.testing import CliRunner

from truthound.cli import app


@pytest.fixture
def runner():
    return CliRunner()


def test_plugins_help_snapshot(runner):
    result = runner.invoke(app, ['plugins', '--help'])

    assert result.exit_code == 0
    assert 'list' in result.output
    assert 'load' in result.output
    assert 'disable' in result.output


def test_plugins_list_json_snapshot(runner):
    result = runner.invoke(app, ['plugins', 'list', '--json'])

    assert result.exit_code == 0
    assert isinstance(json.loads(result.output), list)
