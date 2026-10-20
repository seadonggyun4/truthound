"""Optional account authorization must not excuse hermetic/startup failures."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.integration.cloud_dw import conftest as cloud


@pytest.mark.parametrize("authorized", [False, True])
def test_live_guard_uses_fixture_closure_not_entire_directory(monkeypatch, authorized):
    monkeypatch.setenv("TRUTHOUND_CLOUD_DW_RUN_LIVE", "1" if authorized else "0")
    root = Path(cloud.__file__).parent
    marked = {}
    items = []
    for name, path, fixtures in (
        ("live", root / "test_connection.py", ["any_dataset", "any_backend"]),
        ("generator", root / "test_schema.py", ["bigquery_generator"]),
        ("unrelated", root.parent / "test_other.py", ["any_backend"]),
    ):
        marked[name] = []
        items.append(
            SimpleNamespace(
                path=path,
                fixturenames=fixtures,
                keywords={},
                add_marker=marked[name].append,
            )
        )
    config = SimpleNamespace(
        getoption=lambda option: False if option == "--skip-expensive" else None
    )
    cloud.pytest_collection_modifyitems(config, items)
    assert [m.name for m in marked["live"]] == (
        ["live_account"] if authorized else ["live_account", "skip"]
    )
    assert marked["generator"] == marked["unrelated"] == []
