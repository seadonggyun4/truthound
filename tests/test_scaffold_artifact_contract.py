"""Native scaffolds cannot write outside their selected root or follow linked files."""

import pytest

from truthound.cli_modules.scaffolding.base import ScaffoldResult


def test_scaffold_paths_are_preflighted_before_any_file_is_written(tmp_path):
    root = tmp_path / "output"
    result = ScaffoldResult(success=True)
    result.add_file("valid.py", "value = 1")
    result.add_file("../escaped.py", "value = 2")
    with pytest.raises(ValueError):
        result.write_files(root)
    assert not (root / "valid.py").exists()
    assert not (tmp_path / "escaped.py").exists()


def test_scaffold_overwrite_cannot_follow_an_existing_symlink(tmp_path):
    root = tmp_path / "output"
    root.mkdir()
    private = tmp_path / "private.py"
    private.write_text("original")
    (root / "linked.py").symlink_to(private)
    result = ScaffoldResult(success=True)
    result.add_file("linked.py", "replacement", overwrite=True)
    with pytest.raises(ValueError):
        result.write_files(root)
    assert private.read_text() == "original"


@pytest.mark.parametrize("minimum", ["3.10", "invalid", "3.11\nmalformed", True])
def test_plugin_rejects_unsupported_python_before_emitting_files(minimum):
    from truthound.cli_modules.scaffolding import PluginScaffold, ScaffoldConfig

    result = PluginScaffold().generate(
        ScaffoldConfig(
            name="synthetic_plugin", template_variant="validator", extra={"python_version": minimum}
        )
    )
    assert not result.success
    assert not result.files


def test_plugin_metadata_can_be_installed_with_current_core():
    import tomllib

    from packaging.specifiers import SpecifierSet

    from truthound.cli_modules.scaffolding import PluginScaffold, ScaffoldConfig

    result = PluginScaffold().generate(
        ScaffoldConfig(name="synthetic_plugin", template_variant="validator")
    )
    assert result.success, result.errors
    metadata = tomllib.loads(
        next(f.content for f in result.files if str(f.path) == "pyproject.toml")
    )
    versions = SpecifierSet(metadata["project"]["requires-python"])
    assert "3.10" not in versions
    assert "3.11" in versions


@pytest.mark.parametrize("name", ["class", "return", "import", "synthetic\n"])
def test_scaffold_rejects_names_that_cannot_be_imported(name):
    from truthound.cli_modules.scaffolding import ScaffoldConfig
    with pytest.raises(ValueError):
        ScaffoldConfig(name=name)
