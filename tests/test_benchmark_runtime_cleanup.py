"""Owned CI runtimes cannot delete another run or follow host aliases."""

import importlib.util
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.fault


@pytest.fixture
def runtime():
    path = Path(__file__).resolve().parents[1] / "verification/ci/benchmark_runtime.py"
    spec = importlib.util.spec_from_file_location("benchmark_runtime_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def environment(tmp_path):
    return {
        "RUNNER_TEMP": str(tmp_path.resolve()), "GITHUB_RUN_ID": "42",
        "GITHUB_RUN_ATTEMPT": "1", "GITHUB_SHA": "a" * 40,
        "RELEASE_VENV": str(tmp_path.resolve() / "truthound-release-42-1/venv"),
    }


def test_only_claimed_runtime_is_removed_without_following_descendant_links(runtime, tmp_path):
    env = environment(tmp_path)
    owned = Path(env["RELEASE_VENV"])
    owned.mkdir(parents=True)
    outside = tmp_path / "another-run"
    outside.mkdir()
    (outside / "evidence").write_text("synthetic")
    (owned / "alias").symlink_to(outside, target_is_directory=True)
    runtime.claim(env)
    with pytest.raises(ValueError):
        runtime.cleanup({**env, "GITHUB_RUN_ID": "43"})
    assert owned.exists()
    runtime.cleanup(env)
    assert not owned.parent.exists()
    assert (outside / "evidence").read_text() == "synthetic"


@pytest.mark.parametrize("tamper", ["identity", "root-alias", "marker-alias"])
def test_changed_owner_or_alias_retains_all_runtime_content(runtime, tmp_path, tamper):
    env = environment(tmp_path)
    owned = Path(env["RELEASE_VENV"])
    owned.mkdir(parents=True)
    (owned / "evidence").write_text("synthetic")
    runtime.claim(env)
    marker = owned.parent / ".truthound-owned-runtime"
    if tamper == "identity":
        marker.write_text(json.dumps({"run": "43", "attempt": "1", "sha": "a" * 40}))
    elif tamper == "marker-alias":
        original = owned.parent / "original-marker"
        marker.rename(original)
        marker.symlink_to(original)
    else:
        original = tmp_path / "retained"
        owned.parent.rename(original)
        owned.parent.symlink_to(original, target_is_directory=True)
    with pytest.raises(ValueError):
        runtime.cleanup(env)
    assert (owned / "evidence").read_text() == "synthetic"
